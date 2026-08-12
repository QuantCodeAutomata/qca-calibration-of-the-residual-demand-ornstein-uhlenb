"""
nuclear_control.py
==================
Core implementation of the optimal-switching control problem for a nuclear
power plant.

The plant produces at level P_t ∈ [P_min, P_max] and can be in one of three
operating regimes:

    0 = hold      (dP/dt = 0)
    1 = ramp_up   (dP/dt = +r, bounded at P_max)
    2 = ramp_down (dP/dt = −r, bounded at P_min)

Switching cost matrix C[i, j]  (from regime i to regime j):
──────────────────────────────────────────
        to: hold  ramp_up  ramp_down
from hold     0   4.8e-4    1.6e-4
from ramp_up  0.4e-4  0     1.6e-4
from ramp_down 4e-4  7e-4    0
──────────────────────────────────────────

**Numerical approximation**: the full 35 064-step DP is solved on a coarsened
time grid.  The closed-economy solver uses N_DP = 365 daily steps; the open-
economy solver uses N_DP = 52 weekly steps.  Forward simulations always run at
full 15-minute (1/96 day) resolution using exact OU transitions.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from src.ou_model import compute_seasonal_mean, PERIODS_DAYS, PUBLISHED_PARAMS

# ---------------------------------------------------------------------------
# Published constant parameters
# ---------------------------------------------------------------------------

#: Switching cost matrix C[from_regime, to_regime] (indices: 0=hold,1=+r,2=−r)
DEFAULT_SWITCHING_COST: np.ndarray = np.array(
    [
        [0.0,    4.8e-4, 1.6e-4],   # from hold
        [0.4e-4, 0.0,    1.6e-4],   # from ramp_up (+r)
        [4.0e-4, 7.0e-4, 0.0   ],   # from ramp_down (−r)
    ],
    dtype=np.float64,
)

#: r_bar: production ramp rate [units/day]
R_BAR: float = 0.05 / (1.0 / 96.0)   # = 4.8 per day

REGIME_NAMES: dict[int, str] = {0: "hold", 1: "ramp_up", 2: "ramp_down"}


# ---------------------------------------------------------------------------
# Production dynamics
# ---------------------------------------------------------------------------

def production_update(
    p: float,
    regime_idx: int,
    r: float,
    Delta: float,
    P_min: float,
    P_max: float,
) -> float:
    """Return P_{t+Delta} given current production P_t = p and active regime.

    Parameters
    ----------
    p:
        Current production level, must be in [P_min, P_max].
    regime_idx:
        0 = hold, 1 = ramp_up (+r), 2 = ramp_down (−r).
    r:
        Ramp rate in units per day.
    Delta:
        Time step in days.
    P_min, P_max:
        Production bounds.

    Returns
    -------
    float : new production level, clipped to [P_min, P_max].
    """
    if regime_idx == 1:          # ramp_up
        return min(p + r * Delta, P_max)
    if regime_idx == 2:          # ramp_down
        return max(p - r * Delta, P_min)
    return float(p)              # hold


# ---------------------------------------------------------------------------
# Running cost functions
# ---------------------------------------------------------------------------

def closed_economy_running_cost(
    p: float,
    y: float,
    gamma_1: float,
    lambda_1: float,
    lambda_2: float,
) -> float:
    """Instantaneous cost for the closed-economy model.

    f(p, y) = lambda_1 * (p−y)^+ + lambda_2 * (y−p)^+ + gamma_1 * p

    Parameters
    ----------
    p: production level
    y: demand level
    gamma_1: marginal operational cost coefficient
    lambda_1: surplus penalty coefficient
    lambda_2: shortage penalty coefficient
    """
    return lambda_1 * max(p - y, 0.0) + lambda_2 * max(y - p, 0.0) + gamma_1 * p


def psi_price(
    m: float,
    s_L: float,
    s_M: float,
    s_H: float,
    a: float,
) -> float:
    """Piecewise electricity selling price as a function of net demand m.

    Three-level step function:
        m <= 0 : s_L  (oversupply / low-price regime)
        0 < m <= a : s_M  (normal regime)
        m > a  : s_H  (scarcity / high-price regime)
    """
    if m <= 0.0:
        return s_L
    if m <= a:
        return s_M
    return s_H


def price_taker_running_cost(
    p: float,
    y: float,
    m: float,
    gamma_1: float,
    s_L: float,
    s_M: float,
    s_H: float,
    a: float,
    delta: float,
) -> float:
    """Price-taker running cost.

    f_PT = (psi(m) + delta) * (y−p)^+ − psi(m) * (p−y)^+ + gamma_1 * p

    * When y > p: plant purchases (y−p) from grid at (psi+delta).
    * When p > y: plant sells (p−y) to grid at psi.
    """
    psi = psi_price(m, s_L, s_M, s_H, a)
    return (psi + delta) * max(y - p, 0.0) - psi * max(p - y, 0.0) + gamma_1 * p


def price_maker_running_cost(
    p: float,
    y: float,
    m: float,
    gamma_1: float,
    s_L: float,
    s_M: float,
    s_H: float,
    a: float,
    delta: float,
) -> float:
    """Price-maker running cost.

    The plant's net position (y−p) shifts the market price:
        Z = m + y − p  (effective net demand seen by market)
    f_PM = (psi(Z) + delta) * (y−p)^+ − psi(Z) * (p−y)^+ + gamma_1 * p
    """
    Z = m + y - p
    psi = psi_price(Z, s_L, s_M, s_H, a)
    return (psi + delta) * max(y - p, 0.0) - psi * max(p - y, 0.0) + gamma_1 * p


# ---------------------------------------------------------------------------
# Numba-accelerated inner DP kernels
#
# Strategy: use a conditional decorator _jit so that when numba is present
# each helper is JIT-compiled in declaration order.  Because Python evaluates
# decorators eagerly, _bilinear2d is compiled before _dp_step_closed, and
# numba resolves the call in nopython mode correctly (it sees a Dispatcher
# object in the global scope, not a plain Python function).
# ---------------------------------------------------------------------------

_jit = (
    numba.jit(nopython=True, cache=True) if _HAS_NUMBA else (lambda f: f)
)


@_jit
def _bilinear2d(
    V: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x: float,
    y: float,
) -> float:
    """Bilinear interpolation on 2-D array V[ix, iy] at point (x, y)."""
    nx = x_grid.shape[0]
    ny = y_grid.shape[0]
    ix = int(np.searchsorted(x_grid, x, side="right")) - 1
    if ix < 0:
        ix = 0
    if ix > nx - 2:
        ix = nx - 2
    iy = int(np.searchsorted(y_grid, y, side="right")) - 1
    if iy < 0:
        iy = 0
    if iy > ny - 2:
        iy = ny - 2
    dx = (x - x_grid[ix]) / (x_grid[ix + 1] - x_grid[ix])
    dy = (y - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy])
    return (
        V[ix,     iy    ] * (1.0 - dx) * (1.0 - dy)
        + V[ix + 1, iy    ] * dx         * (1.0 - dy)
        + V[ix,     iy + 1] * (1.0 - dx) * dy
        + V[ix + 1, iy + 1] * dx         * dy
    )


@_jit
def _trilinear3d(
    V: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    x: float,
    y: float,
    z: float,
) -> float:
    """Trilinear interpolation on 3-D array V[ix, iy, iz] at (x, y, z)."""
    nx = x_grid.shape[0]
    ny = y_grid.shape[0]
    nz = z_grid.shape[0]
    ix = int(np.searchsorted(x_grid, x, side="right")) - 1
    if ix < 0:
        ix = 0
    if ix > nx - 2:
        ix = nx - 2
    iy = int(np.searchsorted(y_grid, y, side="right")) - 1
    if iy < 0:
        iy = 0
    if iy > ny - 2:
        iy = ny - 2
    iz = int(np.searchsorted(z_grid, z, side="right")) - 1
    if iz < 0:
        iz = 0
    if iz > nz - 2:
        iz = nz - 2
    dx = (x - x_grid[ix]) / (x_grid[ix + 1] - x_grid[ix])
    dy = (y - y_grid[iy]) / (y_grid[iy + 1] - y_grid[iy])
    dz = (z - z_grid[iz]) / (z_grid[iz + 1] - z_grid[iz])
    return (
        V[ix,     iy,     iz    ] * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        + V[ix + 1, iy,     iz    ] * dx         * (1.0 - dy) * (1.0 - dz)
        + V[ix,     iy + 1, iz    ] * (1.0 - dx) * dy         * (1.0 - dz)
        + V[ix + 1, iy + 1, iz    ] * dx         * dy         * (1.0 - dz)
        + V[ix,     iy,     iz + 1] * (1.0 - dx) * (1.0 - dy) * dz
        + V[ix + 1, iy,     iz + 1] * dx         * (1.0 - dy) * dz
        + V[ix,     iy + 1, iz + 1] * (1.0 - dx) * dy         * dz
        + V[ix + 1, iy + 1, iz + 1] * dx         * dy         * dz
    )


@_jit
def _dp_step_closed(
    V_next: np.ndarray,
    V_curr: np.ndarray,
    pi_step: np.ndarray,
    P_grid: np.ndarray,
    Y_grid: np.ndarray,
    theta_k: float,
    kappa: float,
    nu: float,
    Delta_DP: float,
    gamma1: float,
    lambda1: float,
    lambda2: float,
    r_bar: float,
    P_min: float,
    P_max: float,
    C: np.ndarray,
    cost_offset: float,
) -> None:
    """One backward DP step for the closed-economy model.

    Parameters
    ----------
    cost_offset:
        State-independent constant added to the running cost per unit time.
        Used to verify Remark 1 invariance (constant shifts in running cost
        do not change the optimal policy).

    Mutates V_curr (shape 3×n_P×n_Y) and pi_step (shape 3×n_P×n_Y, int8).
    """
    n_P = P_grid.shape[0]
    n_Y = Y_grid.shape[0]
    alpha = math.exp(-kappa * Delta_DP)
    sigma = nu * math.sqrt((1.0 - alpha * alpha) / (2.0 * kappa))
    r_step = r_bar * Delta_DP

    Y_lo = Y_grid[0]
    Y_hi = Y_grid[n_Y - 1]

    for j_y in range(n_Y):
        y = Y_grid[j_y]
        mu = y * alpha + theta_k * (1.0 - alpha)
        y_p = mu + sigma
        y_m = mu - sigma
        if y_p > Y_hi:
            y_p = Y_hi
        if y_p < Y_lo:
            y_p = Y_lo
        if y_m > Y_hi:
            y_m = Y_hi
        if y_m < Y_lo:
            y_m = Y_lo

        for j_p in range(n_P):
            p = P_grid[j_p]
            surplus = p - y
            shortage = y - p
            if surplus < 0.0:
                surplus = 0.0
            if shortage < 0.0:
                shortage = 0.0
            run = Delta_DP * (
                lambda1 * surplus + lambda2 * shortage + gamma1 * p + cost_offset
            )

            pn0 = p
            pn1 = p + r_step
            if pn1 > P_max:
                pn1 = P_max
            pn2 = p - r_step
            if pn2 < P_min:
                pn2 = P_min

            for l in range(3):
                best_val = 1.0e18
                best_m = l
                for m in range(3):
                    if m == 0:
                        pn = pn0
                    elif m == 1:
                        pn = pn1
                    else:
                        pn = pn2
                    v_p = _bilinear2d(V_next[m], P_grid, Y_grid, pn, y_p)
                    v_m = _bilinear2d(V_next[m], P_grid, Y_grid, pn, y_m)
                    val = C[l, m] + run + 0.5 * (v_p + v_m)
                    if val < best_val:
                        best_val = val
                        best_m = m
                V_curr[l, j_p, j_y] = best_val
                pi_step[l, j_p, j_y] = best_m


@_jit
def _dp_step_open(
    V_next: np.ndarray,
    V_curr: np.ndarray,
    pi_step: np.ndarray,
    P_grid: np.ndarray,
    Y_grid: np.ndarray,
    M_grid: np.ndarray,
    theta_Y: float,
    theta_M: float,
    kappa_Y: float,
    nu_Y: float,
    kappa_M: float,
    nu_M: float,
    Delta_DP: float,
    gamma1: float,
    lambda1: float,
    lambda2: float,
    r_bar: float,
    P_min: float,
    P_max: float,
    C: np.ndarray,
    s_L: float,
    s_M_val: float,
    s_H: float,
    a: float,
    delta_spread: float,
    mode: int,
) -> None:
    """One backward DP step for the open-economy model.

    mode=0: price-taker,  mode=1: price-maker.
    Mutates V_curr (3×n_P×n_Y×n_M) and pi_step (3×n_P×n_Y×n_M, int8).
    """
    n_P = P_grid.shape[0]
    n_Y = Y_grid.shape[0]
    n_M = M_grid.shape[0]

    aY = math.exp(-kappa_Y * Delta_DP)
    sY = nu_Y * math.sqrt((1.0 - aY * aY) / (2.0 * kappa_Y))
    aM = math.exp(-kappa_M * Delta_DP)
    sM = nu_M * math.sqrt((1.0 - aM * aM) / (2.0 * kappa_M))

    r_step = r_bar * Delta_DP
    Y_lo = Y_grid[0]
    Y_hi = Y_grid[n_Y - 1]
    M_lo = M_grid[0]
    M_hi = M_grid[n_M - 1]

    for j_m in range(n_M):
        mv = M_grid[j_m]
        mu_M = mv * aM + theta_M * (1.0 - aM)
        mp = mu_M + sM
        mm = mu_M - sM
        if mp > M_hi:
            mp = M_hi
        if mp < M_lo:
            mp = M_lo
        if mm > M_hi:
            mm = M_hi
        if mm < M_lo:
            mm = M_lo

        for j_y in range(n_Y):
            y = Y_grid[j_y]
            mu_Y = y * aY + theta_Y * (1.0 - aY)
            yp = mu_Y + sY
            ym = mu_Y - sY
            if yp > Y_hi:
                yp = Y_hi
            if yp < Y_lo:
                yp = Y_lo
            if ym > Y_hi:
                ym = Y_hi
            if ym < Y_lo:
                ym = Y_lo

            for j_p in range(n_P):
                p = P_grid[j_p]
                shortage = y - p
                surplus = p - y
                if shortage < 0.0:
                    shortage = 0.0
                if surplus < 0.0:
                    surplus = 0.0

                if mode == 0:  # price-taker: psi depends on mv
                    if mv <= 0.0:
                        psi_v = s_L
                    elif mv <= a:
                        psi_v = s_M_val
                    else:
                        psi_v = s_H
                else:           # price-maker: psi depends on Z = mv + y - p
                    Z = mv + y - p
                    if Z <= 0.0:
                        psi_v = s_L
                    elif Z <= a:
                        psi_v = s_M_val
                    else:
                        psi_v = s_H

                run = Delta_DP * (
                    (psi_v + delta_spread) * shortage
                    - psi_v * surplus
                    + gamma1 * p
                )

                pn0 = p
                pn1 = p + r_step
                if pn1 > P_max:
                    pn1 = P_max
                pn2 = p - r_step
                if pn2 < P_min:
                    pn2 = P_min

                for l in range(3):
                    best_val = 1.0e18
                    best_m = l
                    for m in range(3):
                        if m == 0:
                            pn = pn0
                        elif m == 1:
                            pn = pn1
                        else:
                            pn = pn2
                        # 4-point Rademacher average
                        v_pp = _trilinear3d(V_next[m], P_grid, Y_grid, M_grid, pn, yp, mp)
                        v_pm = _trilinear3d(V_next[m], P_grid, Y_grid, M_grid, pn, yp, mm)
                        v_mp = _trilinear3d(V_next[m], P_grid, Y_grid, M_grid, pn, ym, mp)
                        v_mm = _trilinear3d(V_next[m], P_grid, Y_grid, M_grid, pn, ym, mm)
                        val = C[l, m] + run + 0.25 * (v_pp + v_pm + v_mp + v_mm)
                        if val < best_val:
                            best_val = val
                            best_m = m
                    V_curr[l, j_p, j_y, j_m] = best_val
                    pi_step[l, j_p, j_y, j_m] = best_m



# ---------------------------------------------------------------------------
# Closed-economy solver
# ---------------------------------------------------------------------------

class ClosedEconomySolver:
    """Semi-Lagrangian backward DP for the closed-economy nuclear-dispatch problem.

    State space: (P, Y, regime) on a coarsened time grid of N_DP daily steps.
    Rademacher (two-point) innovations approximate the OU noise distribution.
    Only two value-function slices are kept in memory at any time; the full
    policy array (int8) is stored for forward-simulation lookup.

    Parameters
    ----------
    params:
        Dictionary with keys:
        ``kappa``, ``beta``, ``nu``, ``zeta``, ``eta``, ``periods_days``,
        ``gamma_1``, ``lambda_1``, ``lambda_2``.
    grid_params:
        Dictionary with keys:
        ``P_grid``, ``Y_grid``, ``N_DP``, ``T``, ``r_bar``,
        ``P_min``, ``P_max``, ``C`` (switching cost matrix).
    """

    def __init__(self, params: dict, grid_params: dict) -> None:
        self.params = params
        self.P_grid: np.ndarray = np.asarray(grid_params["P_grid"], dtype=np.float64)
        self.Y_grid: np.ndarray = np.asarray(grid_params["Y_grid"], dtype=np.float64)
        self.N_DP: int = int(grid_params["N_DP"])
        self.T: float = float(grid_params["T"])
        self.Delta_DP: float = self.T / self.N_DP
        self.r_bar: float = float(grid_params.get("r_bar", R_BAR))
        self.P_min: float = float(grid_params.get("P_min", 0.2))
        self.P_max: float = float(grid_params.get("P_max", 0.9))
        self.C: np.ndarray = np.asarray(
            grid_params.get("C", DEFAULT_SWITCHING_COST), dtype=np.float64
        )

        n_P = len(self.P_grid)
        n_Y = len(self.Y_grid)
        self.V0: np.ndarray | None = None
        # Policy stored for all N_DP time indices, shape (N_DP, 3, n_P, n_Y)
        self.policy: np.ndarray = np.zeros(
            (self.N_DP, 3, n_P, n_Y), dtype=np.int8
        )

    # ------------------------------------------------------------------
    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """Run backward DP.

        Returns
        -------
        V0 : ndarray, shape (3, n_P, n_Y)
            Value function at t = 0 for each (regime, P, Y).
        policy : ndarray, shape (N_DP, 3, n_P, n_Y), dtype int8
            Optimal next regime at each (t_idx, current_regime, P, Y).
        """
        n_P = len(self.P_grid)
        n_Y = len(self.Y_grid)

        V_next = np.zeros((3, n_P, n_Y), dtype=np.float64)
        V_curr = np.zeros((3, n_P, n_Y), dtype=np.float64)
        pi_step = np.zeros((3, n_P, n_Y), dtype=np.int8)

        ou_params = {
            "beta": self.params["beta"],
            "zeta": self.params["zeta"],
            "eta": self.params["eta"],
            "periods_days": self.params.get("periods_days", PERIODS_DAYS),
        }

        for k in range(self.N_DP - 1, -1, -1):
            t_k = k * self.Delta_DP
            theta_k = float(
                compute_seasonal_mean(np.array([t_k]), ou_params)[0]
            )
            _dp_step_closed(
                V_next, V_curr, pi_step,
                self.P_grid, self.Y_grid,
                theta_k,
                float(self.params["kappa"]),
                float(self.params["nu"]),
                self.Delta_DP,
                float(self.params["gamma_1"]),
                float(self.params["lambda_1"]),
                float(self.params["lambda_2"]),
                self.r_bar, self.P_min, self.P_max,
                self.C,
                float(self.params.get("cost_offset", 0.0)),
            )
            self.policy[k] = pi_step
            V_next = V_curr.copy()

        self.V0 = V_next.copy()
        return self.V0, self.policy

    # ------------------------------------------------------------------
    def get_policy(
        self,
        t_idx: int,
        p: float,
        y: float,
        regime: int,
    ) -> int:
        """Look up the optimal next regime for a given state.

        Parameters
        ----------
        t_idx:
            DP time-step index (0 = beginning, N_DP-1 = last).
        p, y:
            Current production and demand levels (nearest grid point used).
        regime:
            Current regime (0, 1, or 2).

        Returns
        -------
        int : optimal next regime.
        """
        k = min(int(t_idx), self.N_DP - 1)
        j_p = int(np.argmin(np.abs(self.P_grid - p)))
        j_y = int(np.argmin(np.abs(self.Y_grid - y)))
        return int(self.policy[k, regime, j_p, j_y])


# ---------------------------------------------------------------------------
# Open-economy solver
# ---------------------------------------------------------------------------

class OpenEconomySolver:
    """Semi-Lagrangian backward DP for the open-economy (price-taker or -maker).

    State space: (P, Y, M, regime) on a coarsened time grid of N_DP steps.
    Rademacher innovations are applied independently in both Y and M directions
    (4 quadrature points per state).

    Parameters
    ----------
    params:
        Dictionary with OU parameters for Y (kappa, beta, nu, zeta, eta,
        periods_days) **plus** open-economy parameters:
        ``kappa_M``, ``beta_M``, ``nu_M``, ``zeta_M``, ``eta_M``,
        ``gamma_1``, ``lambda_1``, ``lambda_2``,
        ``s_L``, ``s_M``, ``s_H``, ``a``, ``delta``.
    grid_params:
        Dictionary with ``P_grid``, ``Y_grid``, ``M_grid``, ``N_DP``, ``T``,
        ``r_bar``, ``P_min``, ``P_max``, ``C``.
    mode:
        ``'price_taker'`` (default) or ``'price_maker'``.
    """

    def __init__(
        self,
        params: dict,
        grid_params: dict,
        mode: str = "price_taker",
    ) -> None:
        self.params = params
        self.mode = mode
        self._mode_int = 0 if mode == "price_taker" else 1

        self.P_grid: np.ndarray = np.asarray(grid_params["P_grid"], dtype=np.float64)
        self.Y_grid: np.ndarray = np.asarray(grid_params["Y_grid"], dtype=np.float64)
        self.M_grid: np.ndarray = np.asarray(grid_params["M_grid"], dtype=np.float64)
        self.N_DP: int = int(grid_params["N_DP"])
        self.T: float = float(grid_params["T"])
        self.Delta_DP: float = self.T / self.N_DP
        self.r_bar: float = float(grid_params.get("r_bar", R_BAR))
        self.P_min: float = float(grid_params.get("P_min", 0.2))
        self.P_max: float = float(grid_params.get("P_max", 0.9))
        self.C: np.ndarray = np.asarray(
            grid_params.get("C", DEFAULT_SWITCHING_COST), dtype=np.float64
        )

        n_P, n_Y, n_M = len(self.P_grid), len(self.Y_grid), len(self.M_grid)
        self.V0: np.ndarray | None = None
        self.policy: np.ndarray = np.zeros(
            (self.N_DP, 3, n_P, n_Y, n_M), dtype=np.int8
        )

    # ------------------------------------------------------------------
    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        """Run backward DP.

        Returns
        -------
        V0 : ndarray, shape (3, n_P, n_Y, n_M)
        policy : ndarray, shape (N_DP, 3, n_P, n_Y, n_M), dtype int8
        """
        n_P, n_Y, n_M = len(self.P_grid), len(self.Y_grid), len(self.M_grid)

        V_next = np.zeros((3, n_P, n_Y, n_M), dtype=np.float64)
        V_curr = np.zeros((3, n_P, n_Y, n_M), dtype=np.float64)
        pi_step = np.zeros((3, n_P, n_Y, n_M), dtype=np.int8)

        ou_Y_params = {
            "beta": self.params["beta"],
            "zeta": self.params["zeta"],
            "eta": self.params["eta"],
            "periods_days": self.params.get("periods_days", PERIODS_DAYS),
        }
        ou_M_params = {
            "beta": self.params["beta_M"],
            "zeta": self.params["zeta_M"],
            "eta": self.params["eta_M"],
            "periods_days": self.params.get("periods_days", PERIODS_DAYS),
        }

        for k in range(self.N_DP - 1, -1, -1):
            t_k = k * self.Delta_DP
            t_arr = np.array([t_k])
            theta_Y = float(compute_seasonal_mean(t_arr, ou_Y_params)[0])
            theta_M = float(compute_seasonal_mean(t_arr, ou_M_params)[0])

            _dp_step_open(
                V_next, V_curr, pi_step,
                self.P_grid, self.Y_grid, self.M_grid,
                theta_Y, theta_M,
                float(self.params["kappa"]),
                float(self.params["nu"]),
                float(self.params["kappa_M"]),
                float(self.params["nu_M"]),
                self.Delta_DP,
                float(self.params["gamma_1"]),
                float(self.params.get("lambda_1", 0.0)),
                float(self.params.get("lambda_2", 0.0)),
                self.r_bar, self.P_min, self.P_max,
                self.C,
                float(self.params["s_L"]),
                float(self.params["s_M"]),
                float(self.params["s_H"]),
                float(self.params["a"]),
                float(self.params["delta"]),
                self._mode_int,
            )
            self.policy[k] = pi_step
            V_next = V_curr.copy()

        self.V0 = V_next.copy()
        return self.V0, self.policy

    # ------------------------------------------------------------------
    def get_policy(
        self,
        t_idx: int,
        p: float,
        y: float,
        m: float,
        regime: int,
    ) -> int:
        """Look up the optimal next regime for a given 3D state."""
        k = min(int(t_idx), self.N_DP - 1)
        j_p = int(np.argmin(np.abs(self.P_grid - p)))
        j_y = int(np.argmin(np.abs(self.Y_grid - y)))
        j_m = int(np.argmin(np.abs(self.M_grid - m)))
        return int(self.policy[k, regime, j_p, j_y, j_m])
