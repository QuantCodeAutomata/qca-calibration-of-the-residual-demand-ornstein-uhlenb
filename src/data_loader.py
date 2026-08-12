"""
data_loader.py
==============
Data loading and synthetic generation for Italian residual electricity demand.

Real data note
--------------
Terna (Italy's TSO) publishes 15-minute residual demand data through its
Download Center — it is **not** accessible via an open API.  To use real
data, download the CSV files from:

    https://www.terna.it/en/electric-system/transparency/download-center

and place them under ``data/``.  Pass the resulting path to
:func:`load_or_generate_data`; if the file is found it will be loaded,
otherwise synthetic data are generated from the published parameters.

DST assumption
--------------
All timestamps are generated in UTC to avoid daylight-saving discontinuities.
When using real Terna data, convert local Italian time to UTC before fitting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ou_model import PERIODS_DAYS, simulate_ou_path


def generate_synthetic_residual_demand(
    start_date: str,
    end_date: str,
    kappa: float,
    beta: float,
    nu: float,
    zeta: list[float],
    eta: list[float],
    periods_days: list[float] | None = None,
    C_ref: float = 20_000.0,
    seed: int = 42,
) -> pd.Series:
    """Generate synthetic 15-minute residual demand from published OU parameters.

    The normalised process Y_t is simulated via exact OU discretisation and
    then scaled to MW:  ``demand_MW = C_ref * Y_t``.

    Parameters
    ----------
    start_date, end_date:
        Inclusive date strings, e.g. ``"2024-01-01"`` / ``"2025-12-31"``.
        The end-of-day 23:45 UTC timestamp is included.
    kappa, beta, nu:
        OU scalar parameters (normalised, dimensionless units).
    zeta, eta:
        Fourier coefficient vectors.
    periods_days:
        Seasonal periods in days; defaults to :data:`~src.ou_model.PERIODS_DAYS`.
    C_ref:
        Reference capacity in MW used to un-normalise the process.
        Default: 20 000 MW (documented assumption for Italian grid).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pandas.Series
        15-minute residual demand in MW, indexed by UTC timestamps, named
        ``"residual_demand_mw"``.
    """
    if periods_days is None:
        periods_days = PERIODS_DAYS

    # 15-minute UTC grid; freq='15min' is unambiguous and DST-free
    timestamps = pd.date_range(
        start=start_date, end=end_date, freq="15min", tz="UTC", inclusive="both"
    )
    Delta = 15.0 / 1440.0  # days per 15-min step
    t_days = np.arange(len(timestamps), dtype=float) * Delta

    Y_norm = simulate_ou_path(
        t_days=t_days,
        kappa=kappa,
        beta=beta,
        zeta=zeta,
        eta=eta,
        nu=nu,
        Y0=beta,          # initialise at long-run mean
        Delta=Delta,
        periods_days=periods_days,
        seed=seed,
    )

    demand_mw = C_ref * Y_norm
    return pd.Series(demand_mw, index=timestamps, name="residual_demand_mw")


def load_or_generate_data(
    data_path: str | Path,
    start_date: str,
    end_date: str,
    params: dict,
    seed: int = 42,
) -> pd.Series:
    """Load residual demand from *data_path* or generate synthetic data.

    If *data_path* exists the CSV is read and returned as a Series.  Otherwise
    synthetic data are generated from *params*, saved to *data_path*, and
    returned.

    Parameters
    ----------
    data_path:
        Path to a CSV file (single column, datetime index) for real Terna data.
        The file need not exist; it will be created on first call.
    start_date, end_date:
        Date range for synthetic generation (ignored when loading real data).
    params:
        Keyword arguments forwarded to
        :func:`generate_synthetic_residual_demand` (must include ``kappa``,
        ``beta``, ``nu``, ``zeta``, ``eta``; optionally ``C_ref``,
        ``periods_days``).
    seed:
        Random seed passed to :func:`generate_synthetic_residual_demand`.

    Returns
    -------
    pandas.Series
        Residual demand in MW with a UTC DatetimeIndex.
    """
    path = Path(data_path)

    if path.exists():
        series = pd.read_csv(
            path, index_col=0, parse_dates=True
        ).squeeze("columns")
        series.name = "residual_demand_mw"
        # Ensure UTC-aware index
        if series.index.tz is None:
            series.index = series.index.tz_localize("UTC")
        return series

    # Generate and persist
    series = generate_synthetic_residual_demand(
        start_date=start_date,
        end_date=end_date,
        seed=seed,
        **params,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path)
    return series
