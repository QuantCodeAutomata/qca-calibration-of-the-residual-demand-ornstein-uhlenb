"""
exp1_ou_calibration.py
======================
Experiment 1: Profile OU estimation on synthetic Italian residual demand.

Pipeline
--------
1. Generate synthetic 15-min residual demand for 2024-01-01 – 2025-12-31
   using the published Appendix B parameters (C_ref = 20 000 MW, UTC clock).
2. Normalise by C_ref to obtain the dimensionless process Y_t.
3. Run profile_ou_estimation over phi_grid = linspace(0.001, 0.999, 200).
4. Compare estimated vs published scalar parameters.
5. Produce five diagnostic figures.
6. Save parameters to results/exp1_parameters.json.
7. Save diagnostic table to results/exp1_diagnostics.csv.

Data note
---------
Real Terna 15-minute data must be downloaded manually from:
    https://www.terna.it/en/electric-system/transparency/download-center
Place the CSV at data/residual_demand.csv and update DATA_PATH below.

DST assumption
--------------
Synthetic timestamps are UTC; Terna data should be converted to UTC before use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

# Make project root importable when script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_or_generate_data
from src.ou_model import (
    PERIODS_DAYS,
    PUBLISHED_PARAMS,
    compute_seasonal_mean,
    profile_ou_estimation,
)
from src.utils import compute_acf_diagnostics, compute_standardized_residuals

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

C_REF: float = 20_000.0          # reference capacity [MW]; documented assumption
START_DATE: str = "2024-01-01"
END_DATE: str = "2025-12-31"
DELTA: float = 15.0 / 1440.0     # 15 min expressed in days
ACF_NLAGS: int = 100

# Build phi_grid from a kappa_grid (per day) rather than linspace in phi.
# For kappa=0.35 and Delta=15/1440, phi=exp(-0.35*Delta)≈0.99637 is very
# close to 1; a uniform phi grid of 200 points on [0.001, 0.999] has step
# ≈0.005 and cannot resolve the minimum.  Constructing phi=exp(-kappa*Delta)
# from a uniform kappa grid gives step ≈0.0005 in phi near kappa=0.35, which
# is sufficient.  The function signature of profile_ou_estimation is unchanged.
_KAPPA_GRID: np.ndarray = np.linspace(0.01, 10.0, 200)  # per day
PHI_GRID: np.ndarray = np.exp(-_KAPPA_GRID * DELTA)
PLOT_DAYS: int = 30               # number of days shown in fitted-vs-realized plot

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_PATH = DATA_DIR / "synthetic_residual_demand_2024_2025.csv"

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_fitted_vs_realized(
    Y: np.ndarray,
    theta: np.ndarray,
    timestamps: pd.DatetimeIndex,
    n_days: int = PLOT_DAYS,
) -> None:
    n_pts = min(n_days * 96, len(Y))  # 96 quarters per day
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timestamps[:n_pts], Y[:n_pts], lw=0.6, alpha=0.7, label="Realised $Y_t$")
    ax.plot(
        timestamps[:n_pts], theta[:n_pts],
        lw=1.5, color="tomato", label=r"Fitted $\hat{\theta}_t$",
    )
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Normalised residual demand")
    ax.set_title(f"Fitted seasonal mean vs realised Y  (first {n_days} days)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_fitted_vs_realized.png", dpi=150)
    plt.close(fig)


def _plot_residual_histogram(std_res: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(std_res, bins=80, density=True, alpha=0.65, label="Standardised residuals")
    x = np.linspace(std_res.min(), std_res.max(), 400)
    ax.plot(x, sp_stats.norm.pdf(x), lw=2, color="tomato", label="N(0, 1)")
    ax.set_xlabel("Standardised residual")
    ax.set_ylabel("Density")
    ax.set_title("Histogram of standardised residuals")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_residual_histogram.png", dpi=150)
    plt.close(fig)


def _plot_qq(std_res: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    (osm, osr), (slope, intercept, _) = sp_stats.probplot(std_res, dist="norm")
    ax.scatter(osm, osr, s=1, alpha=0.3, label="Residuals")
    ax.plot(
        [osm[0], osm[-1]],
        [slope * osm[0] + intercept, slope * osm[-1] + intercept],
        color="tomato", lw=2, label="Normal reference",
    )
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title("Normal QQ-plot of standardised residuals")
    ax.legend(markerscale=6)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_qq.png", dpi=150)
    plt.close(fig)


def _plot_acf_residuals(std_res: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    plot_acf(std_res, lags=ACF_NLAGS, ax=ax, alpha=0.05, zero=False, title="")
    ax.set_xlabel("Lag (15-min intervals)")
    ax.set_ylabel("ACF")
    ax.set_title(f"ACF of standardised residuals  (lags 1 – {ACF_NLAGS})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_acf_residuals.png", dpi=150)
    plt.close(fig)


def _plot_acf_squared_residuals(std_res: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    plot_acf(std_res**2, lags=ACF_NLAGS, ax=ax, alpha=0.05, zero=False, title="")
    ax.set_xlabel("Lag (15-min intervals)")
    ax.set_ylabel("ACF")
    ax.set_title(f"ACF of squared standardised residuals  (lags 1 – {ACF_NLAGS})")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_acf_squared_residuals.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    print(sep)
    print("Experiment 1: OU Model Calibration on Synthetic Italian Residual Demand")
    print(sep)

    # ------------------------------------------------------------------
    # Step 1 — Load / generate synthetic 15-min residual demand
    # ------------------------------------------------------------------
    print("\n[1] Loading / generating synthetic 15-min residual demand "
          f"({START_DATE} – {END_DATE}, UTC)…")

    gen_params = {
        "kappa": PUBLISHED_PARAMS["kappa"],
        "beta": PUBLISHED_PARAMS["beta"],
        "nu": PUBLISHED_PARAMS["nu"],
        "zeta": PUBLISHED_PARAMS["zeta"],
        "eta": PUBLISHED_PARAMS["eta"],
        "periods_days": PERIODS_DAYS,
        "C_ref": C_REF,
    }
    demand_mw = load_or_generate_data(
        data_path=DATA_PATH,
        start_date=START_DATE,
        end_date=END_DATE,
        params=gen_params,
        seed=42,
    )
    print(f"   Observations : {len(demand_mw):,}")
    print(f"   MW range     : [{demand_mw.min():.0f}, {demand_mw.max():.0f}]")

    # ------------------------------------------------------------------
    # Step 2 — Normalise by C_ref
    # ------------------------------------------------------------------
    print(f"\n[2] Normalising by C_ref = {C_REF:,.0f} MW…")
    Y = demand_mw.values.astype(float) / C_REF
    t_days = np.arange(len(Y), dtype=float) * DELTA
    print(f"   Y ∈ [{Y.min():.4f}, {Y.max():.4f}],  mean = {Y.mean():.4f},  "
          f"std = {Y.std():.4f}")

    # ------------------------------------------------------------------
    # Step 3 — Profile OU estimation
    # ------------------------------------------------------------------
    print(
        f"\n[3] Profile OU estimation  "
        f"(κ-grid: {_KAPPA_GRID[0]:.2f}–{_KAPPA_GRID[-1]:.2f}/day → "
        f"{len(PHI_GRID)} φ values)…"
    )
    est = profile_ou_estimation(Y, t_days, DELTA, PHI_GRID)
    print(f"   φ*  = {est['phi_star']:.6f}   →   κ̂ = {est['kappa']:.4f}")

    # ------------------------------------------------------------------
    # Step 4 — Parameter comparison
    # ------------------------------------------------------------------
    pub = PUBLISHED_PARAMS
    print("\n[4] Parameter comparison:")
    header = f"   {'Parameter':<12} {'Published':>12} {'Estimated':>12} {'Rel. Error':>12}"
    print(header)
    print("   " + "-" * (len(header) - 3))

    scalar_names = ["kappa", "beta", "nu"]
    for name in scalar_names:
        pv = pub[name]
        ev = est[name]
        rel_err = abs(ev - pv) / abs(pv) * 100
        print(f"   {name:<12} {pv:>12.4f} {ev:>12.4f} {rel_err:>11.2f}%")

    print("\n   Fourier coefficients:")
    print(f"   {'j':<4} {'ζ_pub':>10} {'ζ_est':>10} {'η_pub':>10} {'η_est':>10}")
    print("   " + "-" * 48)
    for j, (zp, ze, ep, ee) in enumerate(
        zip(pub["zeta"], est["zeta"], pub["eta"], est["eta"]), start=1
    ):
        print(f"   {j:<4} {zp:>10.4f} {ze:>10.4f} {ep:>10.4f} {ee:>10.4f}")

    # ------------------------------------------------------------------
    # Step 5 — Compute fitted mean and standardised OU innovations
    # ------------------------------------------------------------------
    est_params = {
        "beta": est["beta"],
        "zeta": est["zeta"],
        "eta": est["eta"],
        "periods_days": PERIODS_DAYS,
    }
    theta_fitted = compute_seasonal_mean(t_days, est_params)

    # One-step-ahead AR fitted values: φ Y_t + (1−φ) θ_t, length N−1
    phi_star = est["phi_star"]
    fitted_AR = phi_star * Y[:-1] + (1.0 - phi_star) * theta_fitted[:-1]

    # Conditional innovation std: σ_ε = ν √((1−φ²)/(2κ))
    sigma_eps = est["nu"] * np.sqrt(
        (1.0 - phi_star**2) / (2.0 * est["kappa"])
    )

    # Standardised OU innovations: approximately i.i.d. N(0,1) under H₀
    std_innovations = compute_standardized_residuals(Y[1:], fitted_AR, sigma_eps)

    acf_res = compute_acf_diagnostics(std_innovations, nlags=ACF_NLAGS)
    acf_sq = compute_acf_diagnostics(std_innovations**2, nlags=ACF_NLAGS)
    print(
        f"\n   Std innovations: mean = {std_innovations.mean():.4f},  "
        f"std = {std_innovations.std():.4f}"
    )
    print(f"   ACF(1) innovations = {acf_res[1]:.4f},  "
          f"ACF(1) squared = {acf_sq[1]:.4f}")

    # ------------------------------------------------------------------
    # Step 6 — Diagnostic plots
    # ------------------------------------------------------------------
    print("\n[5] Generating diagnostic plots…")
    _plot_fitted_vs_realized(Y, theta_fitted, demand_mw.index)
    _plot_residual_histogram(std_innovations)
    _plot_qq(std_innovations)
    _plot_acf_residuals(std_innovations)
    _plot_acf_squared_residuals(std_innovations)
    print("   Five plots saved to results/")

    # ------------------------------------------------------------------
    # Step 7 — Save parameters JSON
    # ------------------------------------------------------------------
    out_params = {
        "published": {
            "kappa": pub["kappa"],
            "beta": pub["beta"],
            "nu": pub["nu"],
            "zeta": pub["zeta"],
            "eta": pub["eta"],
            "periods_days": PERIODS_DAYS,
        },
        "estimated": {
            "kappa": est["kappa"],
            "beta": est["beta"],
            "nu": est["nu"],
            "zeta": [float(z) for z in est["zeta"]],
            "eta": [float(e) for e in est["eta"]],
            "phi_star": est["phi_star"],
        },
        "metadata": {
            "C_ref_mw": C_REF,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "delta_days": DELTA,
            "n_obs": int(len(Y)),
            "phi_grid_size": int(len(PHI_GRID)),
            "data_source": "synthetic (published parameters)",
            "timestamps": "UTC",
        },
    }
    params_path = RESULTS_DIR / "exp1_parameters.json"
    with open(params_path, "w") as fh:
        json.dump(out_params, fh, indent=2)
    print(f"\n[6] Parameters saved → {params_path}")

    # ------------------------------------------------------------------
    # Step 8 — Save diagnostic CSV
    # ------------------------------------------------------------------
    diag_rows = []
    for name in scalar_names:
        pv = pub[name]
        ev = est[name]
        diag_rows.append(
            {
                "parameter": name,
                "published": pv,
                "estimated": ev,
                "absolute_error": abs(ev - pv),
                "relative_error_pct": abs(ev - pv) / abs(pv) * 100,
            }
        )
    diag_df = pd.DataFrame(diag_rows)
    diag_path = RESULTS_DIR / "exp1_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False, float_format="%.6f")
    print(f"[7] Diagnostics saved → {diag_path}")

    print("\n" + sep)
    print("Experiment 1 completed successfully.")
    print(sep)
    return est


if __name__ == "__main__":
    main()
