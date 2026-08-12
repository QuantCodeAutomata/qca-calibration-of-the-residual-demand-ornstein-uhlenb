"""
ou_model.py
===========
Ornstein–Uhlenbeck process with a deterministic Fourier seasonal mean.

Model
-----
    dY_t = −κ (Y_t − θ_t) dt + ν dW_t

    θ_t = β + Σ_j [ ζ_j cos(ω_j t) + η_j sin(ω_j t) ]

Exact discretisation (step Δ in days):
    Y_{t+Δ} = φ Y_t + (1−φ) θ_t + ε_t ,  φ = exp(−κΔ)
    ε_t ~ N(0, ν²(1 − φ²) / (2κ))

Estimation (Appendix B profile procedure):
    For each φ on a grid the seasonal parameters are estimated by OLS,
    the SSR is recorded, and the grid value minimising SSR is chosen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Nine seasonal periods (days): intra-day, intra-week, intra-year.
PERIODS_DAYS: list[float] = [
    1 / 4,        # 6 h intra-day
    1 / 3,        # 8 h intra-day
    1 / 2,        # 12 h intra-day
    1.0,          # daily
    7 / 2,        # 3.5-day intra-week
    7.0,          # weekly
    365.25 / 4,   # quarterly
    365.25 / 2,   # semi-annual
    365.25,       # annual
]

#: Published parameter values from Appendix B.
PUBLISHED_PARAMS: dict = {
    "kappa": 0.3500,
    "beta": 0.6118,
    "nu": 0.1114,
    "zeta": [
         0.4100,  0.1606, -2.4238, -1.5101,
         0.0841,  0.2984, -0.0113,  0.0563,  0.0912,
    ],
    "eta": [
         0.2714, -0.6401,  2.8156, -0.9522,
        -0.2479,  0.0982, -0.0162,  0.0451, -0.0527,
    ],
    "periods_days": PERIODS_DAYS,
}


# ---------------------------------------------------------------------------
# Fourier design matrix
# ---------------------------------------------------------------------------

def build_fourier_design(
    t_days: np.ndarray,
    periods_days: list[float],
) -> pd.DataFrame:
    """Build a cosine/sine Fourier design matrix.

    For each period ``P`` in *periods_days* two columns are appended:
    ``cos_j = cos(2π t / P)`` and ``sin_j = sin(2π t / P)``.

    Parameters
    ----------
    t_days:
        1-D array of time values in days.
    periods_days:
        Ordered list of seasonal periods in days.  Nine periods yield
        18 columns.

    Returns
    -------
    pandas.DataFrame
        Shape ``(len(t_days), 2 * len(periods_days))`` with columns
        ``cos_1, sin_1, cos_2, sin_2, …``.
    """
    t = np.asarray(t_days, dtype=float)
    cols: dict[str, np.ndarray] = {}
    for j, period in enumerate(periods_days, start=1):
        omega = 2.0 * np.pi / period
        cols[f"cos_{j}"] = np.cos(omega * t)
        cols[f"sin_{j}"] = np.sin(omega * t)
    # Pass explicit index so the DataFrame always has len(t) rows, even when
    # periods_days is empty (pd.DataFrame({}) would produce 0 rows otherwise).
    return pd.DataFrame(cols, index=range(len(t)))


# ---------------------------------------------------------------------------
# Seasonal mean
# ---------------------------------------------------------------------------

def compute_seasonal_mean(
    t_days: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Evaluate the Fourier seasonal mean θ_t at every point in *t_days*.

    Parameters
    ----------
    t_days:
        1-D array of time values in days.
    params:
        Dictionary with keys ``beta`` (float), ``zeta`` (array-like of length
        *n_periods*), ``eta`` (array-like of length *n_periods*), and
        optionally ``periods_days`` (defaults to :data:`PERIODS_DAYS`).

    Returns
    -------
    numpy.ndarray
        Shape ``(len(t_days),)``, dtype float.
    """
    t = np.asarray(t_days, dtype=float)
    periods = params.get("periods_days", PERIODS_DAYS)
    beta = float(params["beta"])
    zeta = np.asarray(params["zeta"], dtype=float)
    eta = np.asarray(params["eta"], dtype=float)

    D = build_fourier_design(t, periods)
    # D.values columns are interleaved: cos_1, sin_1, cos_2, sin_2, …
    cos_vals = D.values[:, 0::2]  # columns 0, 2, 4, …  → cos terms
    sin_vals = D.values[:, 1::2]  # columns 1, 3, 5, …  → sin terms

    return beta + cos_vals @ zeta + sin_vals @ eta


# ---------------------------------------------------------------------------
# Profile OU estimation (Appendix B)
# ---------------------------------------------------------------------------

def profile_ou_estimation(
    Y: np.ndarray,
    t_days: np.ndarray,
    Delta: float,
    phi_grid: np.ndarray,
) -> dict:
    """Estimate OU parameters by profiling over a grid of mean-reversion speeds.

    **Algorithm (Appendix B)**

    For fixed φ = exp(−κΔ) the exact discretisation regression

        Y_{t+Δ} = φ Y_t + (1−φ) θ_t + ε_t

    is linear in the seasonal parameters [β, ζ_1, η_1, …, ζ_J, η_J].
    Denote W = [1, cos_1, sin_1, …] (fixed across φ).  Then

        SSR(φ) = ‖(I − H_W)(Y_next − φ Y_curr)‖²
               = r₁·r₁ − 2φ r₁·r₂ + φ² r₂·r₂

    where r₁ = M_W Y_next, r₂ = M_W Y_curr and M_W = I − H_W.  This
    quadratic in φ is evaluated on *phi_grid* without any matrix inversion
    inside the loop; the seasonal parameters are recovered at the optimal φ*
    via a single solve.

    Parameters
    ----------
    Y:
        Normalised residual demand time series (1-D array, length N).
    t_days:
        Corresponding time grid in days (length N).
    Delta:
        Sampling interval in days (15 / 1440 for 15-min data).
    phi_grid:
        1-D array of candidate φ values in (0, 1).

    Returns
    -------
    dict with keys:
        ``kappa``, ``beta``, ``nu``, ``zeta``, ``eta``,
        ``phi_star``, ``phi_grid``, ``ssr_grid``.
    """
    Y = np.asarray(Y, dtype=float)
    t_days = np.asarray(t_days, dtype=float)
    phi_grid = np.asarray(phi_grid, dtype=float)

    n_periods = len(PERIODS_DAYS)
    Y_curr = Y[:-1]
    Y_next = Y[1:]
    t_curr = t_days[:-1]
    n_obs = len(Y_curr)  # N − 1

    # Build fixed design matrix W = [1, cos_1, sin_1, …] shape (n_obs, 1+2J)
    D = build_fourier_design(t_curr, PERIODS_DAYS)
    W = np.column_stack([np.ones(n_obs), D.values])  # (n_obs, 19)

    # Projection residuals r1 = M_W Y_next, r2 = M_W Y_curr
    # Solved via QR for numerical stability (avoids forming W'W explicitly).
    Q, R = np.linalg.qr(W, mode="reduced")  # Q (n_obs,19), R (19,19)
    proj = lambda v: Q @ (Q.T @ v)           # H_W v
    r1 = Y_next - proj(Y_next)
    r2 = Y_curr - proj(Y_curr)

    # SSR(phi) evaluated on grid — purely scalar operations
    r1r1 = float(r1 @ r1)
    r1r2 = float(r1 @ r2)
    r2r2 = float(r2 @ r2)
    ssr_grid = r1r1 - 2.0 * phi_grid * r1r2 + phi_grid**2 * r2r2

    idx = int(np.argmin(ssr_grid))
    phi_star = float(phi_grid[idx])
    best_ssr = float(ssr_grid[idx])

    # Recover seasonal parameters at phi_star via single OLS solve
    # theta_hat = (W'W)^{-1} W' (Y_next − phi* Y_curr) / (1 − phi*)
    Z_star = Y_next - phi_star * Y_curr
    # Use the QR already computed: (W'W)^{-1} W' v = R^{-1} Q' v
    theta_hat = np.linalg.solve(R, Q.T @ Z_star) / (1.0 - phi_star)

    beta = float(theta_hat[0])
    # W columns: [1, cos_1, sin_1, cos_2, sin_2, …] → zeta at odd indices
    zeta = theta_hat[1::2].tolist()   # indices 1, 3, …, 17  (9 values)
    eta = theta_hat[2::2].tolist()    # indices 2, 4, …, 18  (9 values)

    # Continuous-time parameters
    kappa = -np.log(phi_star) / Delta
    n_params = 1 + 2 * n_periods      # intercept + 2 Fourier terms per period
    df = n_obs - n_params
    sigma2 = best_ssr / df if df > 0 else best_ssr
    nu2 = 2.0 * kappa * sigma2 / (1.0 - np.exp(-2.0 * kappa * Delta))
    nu = float(np.sqrt(max(nu2, 0.0)))

    return {
        "kappa": float(kappa),
        "beta": beta,
        "nu": nu,
        "zeta": zeta,
        "eta": eta,
        "phi_star": phi_star,
        "phi_grid": phi_grid.tolist(),
        "ssr_grid": ssr_grid.tolist(),
    }


# ---------------------------------------------------------------------------
# OU path simulation (exact discretisation)
# ---------------------------------------------------------------------------

def simulate_ou_path(
    t_days: np.ndarray,
    kappa: float,
    beta: float,
    zeta: list[float],
    eta: list[float],
    nu: float,
    Y0: float,
    Delta: float,
    periods_days: list[float] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate an OU path using exact Euler–Maruyama discretisation.

    The one-step transition is

        Y_{t+Δ} = φ Y_t + (1−φ) θ_t + ε_t

    with φ = exp(−κΔ) and ε_t ~ N(0, ν²(1−φ²)/(2κ)).

    Parameters
    ----------
    t_days:
        Time grid in days of length N; the step size is assumed uniform = Δ.
    kappa, beta, nu:
        OU scalar parameters.
    zeta, eta:
        Fourier coefficient vectors of length *n_periods*.
    Y0:
        Initial value Y(t_days[0]).
    Delta:
        Sampling interval in days.
    periods_days:
        Seasonal periods; defaults to :data:`PERIODS_DAYS`.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Simulated path of length N.
    """
    if periods_days is None:
        periods_days = PERIODS_DAYS

    t_days = np.asarray(t_days, dtype=float)
    N = len(t_days)
    rng = np.random.default_rng(seed)

    phi = np.exp(-kappa * Delta)
    cond_var = nu**2 * (1.0 - phi**2) / (2.0 * kappa)
    sigma = np.sqrt(max(cond_var, 0.0))

    params = {"beta": beta, "zeta": zeta, "eta": eta, "periods_days": periods_days}
    theta = compute_seasonal_mean(t_days, params)

    eps = rng.normal(0.0, sigma, size=N)
    Y = np.empty(N)
    Y[0] = Y0
    for i in range(1, N):
        Y[i] = phi * Y[i - 1] + (1.0 - phi) * theta[i - 1] + eps[i]

    return Y
