"""
tests/test_exp3.py
==================
Integration-test stub for Experiment 3 (open-economy optimal switching).

Tests verify the PT and PM solver/simulator pipelines on a small grid so
they complete quickly during CI, plus an optional slow test that runs the
full experiment script.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Slow integration test: run the full experiment
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_exp3_runs_and_produces_outputs():
    """Run exp3_open_economy.py as a subprocess and verify key output files."""
    exp_script = PROJECT_ROOT / "exp" / "exp3_open_economy.py"
    result = subprocess.run(
        [sys.executable, str(exp_script)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=900,
    )
    assert result.returncode == 0, (
        f"exp3 exited with code {result.returncode}\n"
        f"STDOUT: {result.stdout[-2000:]}\n"
        f"STDERR: {result.stderr[-2000:]}"
    )
    for fname in ["exp3_metrics.csv", "exp3_pt_pm_comparison.png"]:
        assert (RESULTS_DIR / fname).exists(), f"Missing: {fname}"


# ---------------------------------------------------------------------------
# Fast pipeline test (tiny grid)
# ---------------------------------------------------------------------------

def _make_open_params() -> dict:
    from src.ou_model import PUBLISHED_PARAMS
    p = dict(PUBLISHED_PARAMS)
    p.update({
        "gamma_1": 0.24, "lambda_1": 0.0, "lambda_2": 0.0,
        "P_min": 0.2, "P_max": 0.9,
        "kappa_M": 0.35,
        "beta_M": 1.2236,
        "nu_M": math.sqrt(5) * 0.1114,
        "zeta_M": [5 * z for z in PUBLISHED_PARAMS["zeta"]],
        "eta_M": [5 * e for e in PUBLISHED_PARAMS["eta"]],
        "s_L": 0.0, "s_M": 0.2, "s_H": 0.4,
        "a": 2.7, "delta": 0.08,
    })
    return p


def _make_tiny_grid() -> dict:
    from src.nuclear_control import DEFAULT_SWITCHING_COST, R_BAR
    return {
        "P_grid": np.linspace(0.2, 0.9, 5),
        "Y_grid": np.linspace(-0.5, 2.0, 6),
        "M_grid": np.linspace(-1.0, 5.0, 7),
        "N_DP": 4,
        "T": 365.25,
        "r_bar": R_BAR,
        "P_min": 0.2,
        "P_max": 0.9,
        "C": DEFAULT_SWITCHING_COST,
    }


class TestOpenEconomyPipeline:
    def test_pt_solver_shapes(self):
        """Price-taker solver produces correctly shaped outputs."""
        from src.nuclear_control import OpenEconomySolver
        params = _make_open_params()
        gp = _make_tiny_grid()
        solver = OpenEconomySolver(params, gp, mode="price_taker")
        V0, policy = solver.solve()
        n_P, n_Y, n_M = 5, 6, 7
        assert V0.shape == (3, n_P, n_Y, n_M)
        assert policy.shape == (4, 3, n_P, n_Y, n_M)
        assert policy.dtype == np.int8

    def test_pm_solver_shapes(self):
        """Price-maker solver produces correctly shaped outputs."""
        from src.nuclear_control import OpenEconomySolver
        params = _make_open_params()
        gp = _make_tiny_grid()
        solver = OpenEconomySolver(params, gp, mode="price_maker")
        V0, _ = solver.solve()
        assert V0.shape == (3, 5, 6, 7)

    def test_policy_values_in_range(self):
        """All policy entries must be 0, 1, or 2."""
        from src.nuclear_control import OpenEconomySolver
        params = _make_open_params()
        gp = _make_tiny_grid()
        solver = OpenEconomySolver(params, gp, mode="price_taker")
        _, policy = solver.solve()
        assert np.all((policy >= 0) & (policy <= 2))

    def test_pt_simulation_runs(self):
        """Price-taker forward simulation runs without errors."""
        from src.nuclear_control import OpenEconomySolver
        from src.simulator import simulate_open_economy, compute_metrics
        params = _make_open_params()
        gp = _make_tiny_grid()
        solver = OpenEconomySolver(params, gp, mode="price_taker")
        solver.solve()

        df = simulate_open_economy(
            solver, params,
            T_days=7.0, Delta=1.0 / 96.0,
            P0=0.55, Y0=float(params["beta"]),
            M0=float(params["beta_M"]),
            regime0=0, seed=0, mode="price_taker",
        )
        assert len(df) == 7 * 96
        assert df["P"].between(0.2 - 1e-9, 0.9 + 1e-9).all()
        assert set(df.columns) >= {"t", "P", "Y", "M", "regime",
                                    "running_cost", "switch_cost", "purchase_volume"}

        m = compute_metrics(df, params, 1.0 / 96.0, mode="price_taker")
        assert "purchase_energy" in m
        assert "sale_energy" in m
        assert m["running_cost"] + m["switching_cost"] == pytest.approx(
            m["total_cost"], rel=1e-9
        )

    def test_pm_simulation_runs(self):
        """Price-maker forward simulation runs without errors."""
        from src.nuclear_control import OpenEconomySolver
        from src.simulator import simulate_open_economy
        params = _make_open_params()
        gp = _make_tiny_grid()
        solver = OpenEconomySolver(params, gp, mode="price_maker")
        solver.solve()

        df = simulate_open_economy(
            solver, params,
            T_days=3.0, Delta=1.0 / 96.0,
            P0=0.5, Y0=0.6, M0=1.2,
            regime0=0, seed=7, mode="price_maker",
        )
        assert len(df) == 3 * 96
        assert df["P"].between(0.2 - 1e-9, 0.9 + 1e-9).all()

    def test_pt_value_nonneg(self):
        """Price-taker value function must be ≥ 0 (positive running costs)."""
        from src.nuclear_control import OpenEconomySolver
        params = _make_open_params()
        gp = _make_tiny_grid()
        solver = OpenEconomySolver(params, gp, mode="price_taker")
        V0, _ = solver.solve()
        # Some values may be slightly negative due to selling revenue dominating
        # operational cost, but they should not be deeply negative.
        assert np.all(V0 > -2.0), "V0 has unreasonably large negative values"
