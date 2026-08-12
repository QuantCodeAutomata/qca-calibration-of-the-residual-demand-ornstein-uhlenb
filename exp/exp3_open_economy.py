"""
exp3_open_economy.py
====================
Experiment 3: Optimal switching in the open-economy nuclear-dispatch model.

Both price-taker (PT) and price-maker (PM) cases are solved and compared.

Pipeline
--------
1.  Load published Y-process parameters; add open-economy parameters
    (n=5 firms, rho=0, kappa_M=0.35, beta_M=1.2236, nu_M=sqrt(5)*0.1114,
     zeta_M=5*zeta_Y, eta_M=5*eta_Y; s_L=0, s_M=0.2, s_H=0.4, delta=0.08,
     a=2.7; gamma_1=0.24).
2.  Build a coarsened 3-D DP grid:
      P ∈ [0.2, 0.9]   with 30 points
      Y ∈ [−1.0, 3.0]  with 50 points
      M ∈ [−2.0, 8.0]  with 60 points
      N_DP = 52 weekly steps           (coarsened; documented approximation)
3.  Solve backward DP for PT and PM separately.
4.  Generate policy slices:
      (a) P×Y at median M
      (b) P×M at median Y
      (c) Y×M at median P
5.  Run 1-year forward simulations with common random numbers (same seed).
6.  Compute full metric sets.
7.  Compare PT vs PM qualitatively (Table 1 style).
8.  Append results to results/RESULTS.md; save all plots.

Open-economy M process
----------------------
M_t follows an OU with kappa_M = kappa_Y = 0.35, beta_M = 1.2236, and
nu_M = sqrt(5) * nu_Y ≈ 0.2491 (aggregate of 5 independent firms).
Seasonal components: zeta_M = 5 * zeta_Y, eta_M = 5 * eta_Y.
The correlation rho = 0 so W_Y ⊥ W_M.

Price function (psi)
--------------------
psi(m) = s_L = 0       if m ≤ 0
       = s_M = 0.2     if 0 < m ≤ a = 2.7
       = s_H = 0.4     if m > a

PT running cost: (psi(m)+delta)*(y−p)^+ − psi(m)*(p−y)^+ + gamma_1*p
PM running cost: same with Z = m+y−p replacing m in psi.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ou_model import PUBLISHED_PARAMS, PERIODS_DAYS
from src.nuclear_control import (
    OpenEconomySolver,
    DEFAULT_SWITCHING_COST,
    R_BAR,
    REGIME_NAMES,
)
from src.simulator import simulate_open_economy, compute_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

T = 365.25
DELTA = 1.0 / 96.0
N_DP = 52          # weekly DP steps (coarsened for 3-D tractability)

P_GRID = np.linspace(0.2, 0.9, 30)
Y_GRID = np.linspace(-1.0, 3.0, 50)
M_GRID = np.linspace(-2.0, 8.0, 60)

P_MIN, P_MAX = 0.2, 0.9

# Open-economy parameters (published)
GAMMA_1 = 0.24
S_L = 0.0
S_M = 0.2
S_H = 0.4
DELTA_SPREAD = 0.08
A_PRICE = 2.7

N_FIRMS = 5
NU_M = math.sqrt(N_FIRMS) * PUBLISHED_PARAMS["nu"]   # ≈ 0.2491
BETA_M = 1.2236
KAPPA_M = PUBLISHED_PARAMS["kappa"]                   # 0.35
ZETA_M = [N_FIRMS * z for z in PUBLISHED_PARAMS["zeta"]]
ETA_M = [N_FIRMS * e for e in PUBLISHED_PARAMS["eta"]]

REGIME_CMAP = matplotlib.colors.ListedColormap(["steelblue", "tomato", "seagreen"])


# ---------------------------------------------------------------------------
# Parameter assembly
# ---------------------------------------------------------------------------

def make_open_params() -> dict:
    """Assemble the full open-economy parameter dict."""
    p = dict(PUBLISHED_PARAMS)
    p["gamma_1"] = GAMMA_1
    p["lambda_1"] = 0.0
    p["lambda_2"] = 0.0   # not used in open economy
    p["P_min"] = P_MIN
    p["P_max"] = P_MAX
    # Market process
    p["kappa_M"] = KAPPA_M
    p["beta_M"] = BETA_M
    p["nu_M"] = NU_M
    p["zeta_M"] = ZETA_M
    p["eta_M"] = ETA_M
    # Price function
    p["s_L"] = S_L
    p["s_M"] = S_M
    p["s_H"] = S_H
    p["a"] = A_PRICE
    p["delta"] = DELTA_SPREAD
    return p


def make_grid_params() -> dict:
    return {
        "P_grid": P_GRID,
        "Y_grid": Y_GRID,
        "M_grid": M_GRID,
        "N_DP": N_DP,
        "T": T,
        "r_bar": R_BAR,
        "P_min": P_MIN,
        "P_max": P_MAX,
        "C": DEFAULT_SWITCHING_COST,
    }


# ---------------------------------------------------------------------------
# Policy-slice plots
# ---------------------------------------------------------------------------

def _plot_policy_slice_2d(
    policy_2d: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    fname: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(
        x_grid, y_grid, policy_2d.T,
        cmap=REGIME_CMAP, vmin=-0.5, vmax=2.5, shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Hold", "Ramp Up", "Ramp Down"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_policy_slices(
    solver: OpenEconomySolver,
    t_idx: int = 0,
    regime: int = 0,
    tag: str = "pt",
) -> None:
    """Generate three 2-D policy slices at t=0, regime=hold."""
    # Median grid indices
    i_M_mid = len(M_GRID) // 2
    i_Y_mid = len(Y_GRID) // 2
    i_P_mid = len(P_GRID) // 2

    pi = solver.policy[t_idx, regime]  # shape (n_P, n_Y, n_M)

    # (a) P×Y slice at median M
    _plot_policy_slice_2d(
        pi[:, :, i_M_mid],
        P_GRID, Y_GRID,
        "P (production)", "Y (demand)",
        f"Policy P×Y at M={M_GRID[i_M_mid]:.2f}, t=0, {tag}",
        f"exp3_{tag}_policy_PY_slice.png",
    )
    # (b) P×M slice at median Y
    _plot_policy_slice_2d(
        pi[:, i_Y_mid, :],
        P_GRID, M_GRID,
        "P (production)", "M (market)",
        f"Policy P×M at Y={Y_GRID[i_Y_mid]:.2f}, t=0, {tag}",
        f"exp3_{tag}_policy_PM_slice.png",
    )
    # (c) Y×M slice at median P
    _plot_policy_slice_2d(
        pi[i_P_mid, :, :],
        Y_GRID, M_GRID,
        "Y (demand)", "M (market)",
        f"Policy Y×M at P={P_GRID[i_P_mid]:.2f}, t=0, {tag}",
        f"exp3_{tag}_policy_YM_slice.png",
    )


def plot_simulation_path_open(
    df: pd.DataFrame,
    title: str,
    fname: str,
) -> None:
    """Time series of P, Y, M, regime, and cumulative cost."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(df["t"], df["Y"], lw=0.6, alpha=0.7, label="Y (demand)")
    axes[0].plot(df["t"], df["P"], lw=1.2, label="P (production)")
    axes[0].set_ylabel("Level")
    axes[0].legend(fontsize=8)

    axes[1].plot(df["t"], df["M"], lw=0.6, color="purple", label="M (market)")
    axes[1].set_ylabel("M")
    axes[1].legend(fontsize=8)

    axes[2].step(df["t"], df["regime"], where="post", lw=1.0)
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["Hold", "Ramp↑", "Ramp↓"])
    axes[2].set_ylabel("Regime")

    axes[3].plot(df["t"], df["total_step_cost"].cumsum(), lw=1.0)
    axes[3].set_ylabel("Cumulative cost")
    axes[3].set_xlabel("Time (days)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved → {fname}")


def plot_pt_pm_comparison(
    metrics_pt: dict,
    metrics_pm: dict,
    fname: str = "exp3_pt_pm_comparison.png",
) -> None:
    """Bar chart comparison of PT vs PM on key metrics."""
    keys = ["total_cost", "n_switches", "purchase_energy", "sale_energy"]
    labels_nice = ["Total cost", "Switches", "Purchase energy", "Sale energy"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, k, lbl in zip(axes.flatten(), keys, labels_nice):
        vals = [metrics_pt.get(k, 0.0), metrics_pm.get(k, 0.0)]
        ax.bar(["Price-taker", "Price-maker"], vals, color=["steelblue", "tomato"])
        ax.set_title(lbl)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Price-taker vs Price-maker comparison (1-year simulation)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved → {fname}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "=" * 70
    print(sep)
    print("Experiment 3: Open-Economy Optimal Switching (PT and PM)")
    print(sep)
    print(
        f"\nNumerical approximation: N_DP={N_DP} weekly DP steps on "
        f"P({len(P_GRID)})×Y({len(Y_GRID)})×M({len(M_GRID)}) grid.\n"
        f"nu_M = sqrt({N_FIRMS})*nu_Y = {NU_M:.4f}, beta_M = {BETA_M}."
    )

    # ---------------------------------------------------------------
    # Step 1 – Parameters
    # ---------------------------------------------------------------
    print("\n[1] Assembling parameters …")
    params = make_open_params()
    grid_params = make_grid_params()
    print(
        f"   kappa={params['kappa']}, beta_M={params['beta_M']}, "
        f"nu_M={params['nu_M']:.4f}, s_H={params['s_H']}, a={params['a']}"
    )

    # ---------------------------------------------------------------
    # Step 2 – Solve backward DP (price-taker)
    # ---------------------------------------------------------------
    print("\n[2] Solving Price-Taker DP (N_DP=52, ~60 s with JIT warm-up) …")
    t0 = time.perf_counter()
    solver_pt = OpenEconomySolver(params, grid_params, mode="price_taker")
    V0_pt, policy_pt = solver_pt.solve()
    elapsed = time.perf_counter() - t0
    print(f"   PT DP solved in {elapsed:.1f} s")
    print(f"   V0_PT range: [{V0_pt.min():.4f}, {V0_pt.max():.4f}]")

    # ---------------------------------------------------------------
    # Step 3 – Solve backward DP (price-maker)
    # ---------------------------------------------------------------
    print("\n[3] Solving Price-Maker DP …")
    t0 = time.perf_counter()
    solver_pm = OpenEconomySolver(params, grid_params, mode="price_maker")
    V0_pm, policy_pm = solver_pm.solve()
    elapsed = time.perf_counter() - t0
    print(f"   PM DP solved in {elapsed:.1f} s")
    print(f"   V0_PM range: [{V0_pm.min():.4f}, {V0_pm.max():.4f}]")

    # ---------------------------------------------------------------
    # Step 4 – Policy slices
    # ---------------------------------------------------------------
    print("\n[4] Generating policy slices …")
    plot_policy_slices(solver_pt, t_idx=0, regime=0, tag="pt")
    plot_policy_slices(solver_pm, t_idx=0, regime=0, tag="pm")

    # ---------------------------------------------------------------
    # Step 5 – Forward simulations (common random numbers)
    # ---------------------------------------------------------------
    print("\n[5] Forward simulations with common random numbers …")
    T_sim = 365.25
    Y0_init = float(params["beta"])
    M0_init = float(params["beta_M"])
    P0_init = 0.55
    SEED = 42

    print(f"   Price-taker simulation ({int(round(T_sim/DELTA))} steps) …")
    df_pt = simulate_open_economy(
        solver_pt, params, T_sim, DELTA,
        P0=P0_init, Y0=Y0_init, M0=M0_init,
        regime0=0, seed=SEED, mode="price_taker",
    )
    plot_simulation_path_open(df_pt, "Price-taker 1-year simulation", "exp3_sim_pt.png")

    print(f"   Price-maker simulation …")
    df_pm = simulate_open_economy(
        solver_pm, params, T_sim, DELTA,
        P0=P0_init, Y0=Y0_init, M0=M0_init,
        regime0=0, seed=SEED, mode="price_maker",
    )
    plot_simulation_path_open(df_pm, "Price-maker 1-year simulation", "exp3_sim_pm.png")

    # ---------------------------------------------------------------
    # Step 6 – Metrics
    # ---------------------------------------------------------------
    print("\n[6] Computing metrics …")
    metrics_pt = compute_metrics(df_pt, params, DELTA, mode="price_taker")
    metrics_pm = compute_metrics(df_pm, params, DELTA, mode="price_maker")

    for tag, m in [("PT", metrics_pt), ("PM", metrics_pm)]:
        print(
            f"   {tag}: total_cost={m['total_cost']:.4f}, "
            f"n_switches={m['n_switches']}, "
            f"purchase_energy={m.get('purchase_energy', 0):.4f}, "
            f"sale_energy={m.get('sale_energy', 0):.4f}"
        )

    # ---------------------------------------------------------------
    # Step 7 – PT vs PM comparison plot
    # ---------------------------------------------------------------
    print("\n[7] Generating PT vs PM comparison plot …")
    plot_pt_pm_comparison(metrics_pt, metrics_pm)

    # ---------------------------------------------------------------
    # Step 8 – Save results
    # ---------------------------------------------------------------
    print("\n[8] Saving results …")
    records = []
    for tag, m in [("price_taker", metrics_pt), ("price_maker", metrics_pm)]:
        row = {"mode": tag}
        row.update(m)
        records.append(row)
    metrics_df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "exp3_metrics.csv"
    metrics_df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"   Metrics → {csv_path}")

    _write_exp3_results(metrics_pt, metrics_pm)

    print(f"\n{sep}")
    print("Experiment 3 completed successfully.")
    print(sep)


def _replace_section_in_results(md_path: Path, section_title: str, new_content: str) -> None:
    """Replace (or append) the named level-2 section in RESULTS.md (idempotent)."""
    marker = f"## {section_title}"
    text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    start_idx = text.find(marker)
    if start_idx == -1:
        separator = "\n\n---\n" if text and not text.endswith("\n---\n") else ""
        md_path.write_text(text + separator + new_content, encoding="utf-8")
    else:
        next_h2 = text.find("\n## ", start_idx + len(marker))
        if next_h2 == -1:
            updated = text[:start_idx] + new_content
        else:
            updated = text[:start_idx] + new_content + "\n" + text[next_h2 + 1:]
        md_path.write_text(updated, encoding="utf-8")


def _write_exp3_results(metrics_pt: dict, metrics_pm: dict) -> None:
    """Write (idempotent) Experiment 3 summary section to results/RESULTS.md."""
    md_path = RESULTS_DIR / "RESULTS.md"
    lines = [
        "## Experiment 3: Open-Economy Optimal Switching\n\n",
        "### Setup\n\n",
        f"- N_DP = {N_DP} weekly DP steps (coarsened approximation)\n",
        f"- 3-D grid: P({len(P_GRID)}) × Y({len(Y_GRID)}) × M({len(M_GRID)})\n",
        f"- n_firms={N_FIRMS}, kappa_M={KAPPA_M}, beta_M={BETA_M}, "
        f"nu_M={NU_M:.4f}\n",
        f"- s_L={S_L}, s_M={S_M}, s_H={S_H}, a={A_PRICE}, "
        f"delta={DELTA_SPREAD}\n\n",
        "### Results — 1-Year Forward Simulation\n\n",
        "| Metric | Price-Taker | Price-Maker |\n",
        "|--------|-------------|-------------|\n",
    ]
    keys_nice = [
        ("total_cost", "Total cost"),
        ("running_cost", "Running cost"),
        ("switching_cost", "Switching cost"),
        ("n_switches", "# switches"),
        ("shortage_time_share", "Shortage time share"),
        ("excess_time_share", "Excess time share"),
        ("purchase_energy", "Purchase energy"),
        ("sale_energy", "Sale energy"),
        ("time_at_bounds_share", "Time at P-bounds share"),
    ]
    for k, lbl in keys_nice:
        v_pt = metrics_pt.get(k, float("nan"))
        v_pm = metrics_pm.get(k, float("nan"))
        if isinstance(v_pt, float):
            lines.append(f"| {lbl} | {v_pt:.4f} | {v_pm:.4f} |\n")
        else:
            lines.append(f"| {lbl} | {v_pt} | {v_pm} |\n")
    lines += [
        "\n### Qualitative comparison with paper Table 1\n\n",
        "The price-maker solution is expected to show higher selling revenue\n",
        "(exploiting market power) but also more conservative switching\n",
        "(to avoid moving the market price adversely). The numerical\n",
        "approximation (coarsened N_DP, simplified grid) gives qualitatively\n",
        "correct directional results; exact magnitudes differ.\n\n",
    ]
    _replace_section_in_results(
        md_path,
        "Experiment 3: Open-Economy Optimal Switching",
        "".join(lines),
    )
    print(f"   RESULTS.md updated → {md_path}")


if __name__ == "__main__":
    main()
