"""
simulator.py
============
Forward simulation of the nuclear-plant optimal-switching model at full
15-minute (1/96-day) resolution using exact OU transitions.

The control policy is looked up from a pre-computed DP policy table (produced
by ClosedEconomySolver or OpenEconomySolver) at the nearest coarsened time
index.  All financial quantities are in the same units as the cost functions
(fraction of reference capacity per day).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ou_model import compute_seasonal_mean, PERIODS_DAYS
from src.nuclear_control import (
    ClosedEconomySolver,
    OpenEconomySolver,
    closed_economy_running_cost,
    price_taker_running_cost,
    price_maker_running_cost,
    production_update,
    DEFAULT_SWITCHING_COST,
)


# ---------------------------------------------------------------------------
# Closed-economy forward simulation
# ---------------------------------------------------------------------------

def simulate_closed_economy(
    solver: ClosedEconomySolver,
    params: dict,
    T_days: float,
    Delta: float,
    P0: float,
    Y0: float,
    regime0: int,
    seed: int,
) -> pd.DataFrame:
    """Simulate the closed-economy plant forward at 15-min resolution.

    Parameters
    ----------
    solver:
        A solved ClosedEconomySolver (``solver.solve()`` must have been called).
    params:
        Parameter dict (same as passed to solver); needs kappa, nu, beta,
        zeta, eta, periods_days, gamma_1, lambda_1, lambda_2.
    T_days:
        Simulation horizon in days.
    Delta:
        Fine time step in days (typically 1/96).
    P0, Y0:
        Initial production and demand levels.
    regime0:
        Initial regime (0=hold, 1=ramp_up, 2=ramp_down).
    seed:
        Random seed for OU innovations.

    Returns
    -------
    pandas.DataFrame with columns:
        t, P, Y, regime, running_cost, switch_cost, total_step_cost, switched
    """
    rng = np.random.default_rng(seed)
    N_steps = int(round(T_days / Delta))

    ou_params = {
        "beta": params["beta"],
        "zeta": params["zeta"],
        "eta": params["eta"],
        "periods_days": params.get("periods_days", PERIODS_DAYS),
    }

    # Pre-compute phi and conditional std for exact OU
    phi = np.exp(-params["kappa"] * Delta)
    cond_std = params["nu"] * np.sqrt((1.0 - phi * phi) / (2.0 * params["kappa"]))

    # Pre-compute DP time-index scale
    Delta_DP = solver.Delta_DP
    N_DP = solver.N_DP

    rows = []
    P, Y, regime = float(P0), float(Y0), int(regime0)

    for step in range(N_steps):
        t = step * Delta

        # DP policy lookup: map fine time to coarse DP index
        k_dp = min(int(t / Delta_DP), N_DP - 1)
        new_regime = solver.get_policy(k_dp, P, Y, regime)

        switch_cost = float(solver.C[regime, new_regime])
        run_cost = Delta * closed_economy_running_cost(
            P, Y,
            params["gamma_1"], params["lambda_1"], params["lambda_2"],
        )
        total_cost = switch_cost + run_cost

        rows.append(
            {
                "t": t,
                "P": P,
                "Y": Y,
                "regime": regime,
                "running_cost": run_cost,
                "switch_cost": switch_cost,
                "total_step_cost": total_cost,
                "switched": int(new_regime != regime),
            }
        )

        # Advance state
        regime = new_regime
        P = production_update(P, regime, solver.r_bar, Delta, solver.P_min, solver.P_max)

        theta_t = float(compute_seasonal_mean(np.array([t]), ou_params)[0])
        Y = phi * Y + (1.0 - phi) * theta_t + cond_std * rng.standard_normal()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Open-economy forward simulation
# ---------------------------------------------------------------------------

def simulate_open_economy(
    solver: OpenEconomySolver,
    params: dict,
    T_days: float,
    Delta: float,
    P0: float,
    Y0: float,
    M0: float,
    regime0: int,
    seed: int,
    mode: str = "price_taker",
) -> pd.DataFrame:
    """Simulate the open-economy plant forward at 15-min resolution.

    Parameters
    ----------
    solver:
        A solved OpenEconomySolver.
    params:
        Parameter dict; needs all closed-economy keys plus kappa_M, beta_M,
        nu_M, zeta_M, eta_M, s_L, s_M, s_H, a, delta.
    T_days, Delta:
        Horizon and fine time step.
    P0, Y0, M0:
        Initial production, demand, and market levels.
    regime0:
        Initial regime.
    seed:
        Random seed.
    mode:
        ``'price_taker'`` or ``'price_maker'``.

    Returns
    -------
    pandas.DataFrame with columns:
        t, P, Y, M, regime, running_cost, switch_cost, total_step_cost,
        switched, purchase_volume, sale_volume
    """
    rng = np.random.default_rng(seed)
    N_steps = int(round(T_days / Delta))

    ou_Y_params = {
        "beta": params["beta"],
        "zeta": params["zeta"],
        "eta": params["eta"],
        "periods_days": params.get("periods_days", PERIODS_DAYS),
    }
    ou_M_params = {
        "beta": params["beta_M"],
        "zeta": params["zeta_M"],
        "eta": params["eta_M"],
        "periods_days": params.get("periods_days", PERIODS_DAYS),
    }

    phi_Y = np.exp(-params["kappa"] * Delta)
    std_Y = params["nu"] * np.sqrt((1.0 - phi_Y ** 2) / (2.0 * params["kappa"]))
    phi_M = np.exp(-params["kappa_M"] * Delta)
    std_M = params["nu_M"] * np.sqrt(
        (1.0 - phi_M ** 2) / (2.0 * params["kappa_M"])
    )

    Delta_DP = solver.Delta_DP
    N_DP = solver.N_DP

    run_fn = (
        price_taker_running_cost if mode == "price_taker"
        else price_maker_running_cost
    )

    rows = []
    P, Y, M, regime = float(P0), float(Y0), float(M0), int(regime0)

    for step in range(N_steps):
        t = step * Delta
        k_dp = min(int(t / Delta_DP), N_DP - 1)
        new_regime = solver.get_policy(k_dp, P, Y, M, regime)

        switch_cost = float(solver.C[regime, new_regime])
        run_cost = Delta * run_fn(
            P, Y, M,
            params["gamma_1"],
            params["s_L"], params["s_M"], params["s_H"],
            params["a"], params["delta"],
        )
        total_cost = switch_cost + run_cost

        purchase_vol = Delta * max(Y - P, 0.0)
        sale_vol = Delta * max(P - Y, 0.0)

        rows.append(
            {
                "t": t,
                "P": P,
                "Y": Y,
                "M": M,
                "regime": regime,
                "running_cost": run_cost,
                "switch_cost": switch_cost,
                "total_step_cost": total_cost,
                "switched": int(new_regime != regime),
                "purchase_volume": purchase_vol,
                "sale_volume": sale_vol,
            }
        )

        regime = new_regime
        P = production_update(P, regime, solver.r_bar, Delta, solver.P_min, solver.P_max)

        t_arr = np.array([t])
        theta_Y = float(compute_seasonal_mean(t_arr, ou_Y_params)[0])
        theta_M = float(compute_seasonal_mean(t_arr, ou_M_params)[0])
        Y = phi_Y * Y + (1.0 - phi_Y) * theta_Y + std_Y * rng.standard_normal()
        M = phi_M * M + (1.0 - phi_M) * theta_M + std_M * rng.standard_normal()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    sim_df: pd.DataFrame,
    params: dict,
    Delta: float,
    mode: str = "closed",
) -> dict:
    """Compute summary performance metrics from a forward-simulation DataFrame.

    Parameters
    ----------
    sim_df:
        DataFrame returned by simulate_closed_economy or simulate_open_economy.
    params:
        Model parameters (for reference).
    Delta:
        Fine simulation time step in days.
    mode:
        ``'closed'``, ``'price_taker'``, or ``'price_maker'``.

    Returns
    -------
    dict with keys (scalars unless noted):
        total_cost, running_cost, switching_cost,
        avg_abs_tracking_error, avg_tracking_shortfall,
        shortage_energy, excess_energy,
        shortage_time_share, excess_time_share,
        n_switches, switch_frequency_per_day,
        time_at_P_min_share, time_at_P_max_share, time_at_bounds_share,
        [open-economy only] purchase_energy, sale_energy,
        [open-economy only] avg_purchase_volume_per_day, avg_sale_volume_per_day.
    """
    n = len(sim_df)
    T_sim = n * Delta  # total simulated days

    total_cost = float(sim_df["total_step_cost"].sum())
    running_cost = float(sim_df["running_cost"].sum())
    switching_cost = float(sim_df["switch_cost"].sum())

    P = sim_df["P"].values
    Y = sim_df["Y"].values

    diff = Y - P
    shortage = np.maximum(diff, 0.0)
    excess = np.maximum(-diff, 0.0)

    avg_abs_tracking = float(np.mean(np.abs(diff)))
    avg_shortfall = float(np.mean(shortage))
    shortage_energy = float(np.sum(shortage) * Delta)
    excess_energy = float(np.sum(excess) * Delta)
    shortage_time_share = float(np.mean(diff > 0.0))
    excess_time_share = float(np.mean(diff < 0.0))

    n_switches = int(sim_df["switched"].sum())
    switch_freq = n_switches / T_sim if T_sim > 0 else 0.0

    P_min = params.get("P_min", 0.2)
    P_max = params.get("P_max", 0.9)
    at_min = float(np.mean(P <= P_min + 1e-9))
    at_max = float(np.mean(P >= P_max - 1e-9))
    at_bounds = at_min + at_max

    metrics: dict = {
        "total_cost": total_cost,
        "running_cost": running_cost,
        "switching_cost": switching_cost,
        "avg_abs_tracking_error": avg_abs_tracking,
        "avg_tracking_shortfall": avg_shortfall,
        "shortage_energy": shortage_energy,
        "excess_energy": excess_energy,
        "shortage_time_share": shortage_time_share,
        "excess_time_share": excess_time_share,
        "n_switches": n_switches,
        "switch_frequency_per_day": switch_freq,
        "time_at_P_min_share": at_min,
        "time_at_P_max_share": at_max,
        "time_at_bounds_share": at_bounds,
        "T_sim_days": T_sim,
    }

    if mode in ("price_taker", "price_maker"):
        purchase_energy = float(sim_df["purchase_volume"].sum())
        sale_energy = float(sim_df["sale_volume"].sum())
        metrics["purchase_energy"] = purchase_energy
        metrics["sale_energy"] = sale_energy
        metrics["avg_purchase_per_day"] = purchase_energy / T_sim if T_sim > 0 else 0.0
        metrics["avg_sale_per_day"] = sale_energy / T_sim if T_sim > 0 else 0.0

    return metrics
