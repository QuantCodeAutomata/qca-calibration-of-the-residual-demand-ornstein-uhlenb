"""
tests/test_exp2.py
==================
Integration-test stub for Experiment 2 (closed-economy optimal switching).

Rather than re-running the full experiment (which takes ~30 s), these tests
exercise the same building blocks that exp2_closed_economy.py uses and verify
that the output files are produced when the experiment is run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on the path so imports work from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Quick smoke test: run the experiment and check outputs
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_exp2_runs_and_produces_outputs():
    """Run exp2_closed_economy.py as a subprocess and verify key output files."""
    exp_script = PROJECT_ROOT / "exp" / "exp2_closed_economy.py"
    result = subprocess.run(
        [sys.executable, str(exp_script)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=600,    # 10-minute ceiling
    )
    assert result.returncode == 0, (
        f"exp2 exited with code {result.returncode}\n"
        f"STDOUT: {result.stdout[-2000:]}\n"
        f"STDERR: {result.stderr[-2000:]}"
    )
    expected_files = [
        "exp2_policy_t0_hold.png",
        "exp2_value_function_t0.png",
        "exp2_sim_1year.png",
        "exp2_metrics.csv",
    ]
    for fname in expected_files:
        path = RESULTS_DIR / fname
        assert path.exists(), f"Expected output file not found: {path}"


# ---------------------------------------------------------------------------
# Fast unit-level integration: verify solver + simulator pipeline
# ---------------------------------------------------------------------------

def test_exp2_solver_pipeline():
    """Smoke test: build a tiny solver, run it, and simulate 1 week."""
    from src.ou_model import PUBLISHED_PARAMS, PERIODS_DAYS
    from src.nuclear_control import ClosedEconomySolver, DEFAULT_SWITCHING_COST, R_BAR
    from src.simulator import simulate_closed_economy, compute_metrics

    params = dict(PUBLISHED_PARAMS)
    params.update({"gamma_1": 0.24, "lambda_1": 0.0, "lambda_2": 0.48,
                   "P_min": 0.2, "P_max": 0.9})
    gp = {
        "P_grid": np.linspace(0.2, 0.9, 8),
        "Y_grid": np.linspace(-0.5, 2.0, 10),
        "N_DP": 7,
        "T": 365.25,
        "r_bar": R_BAR,
        "P_min": 0.2,
        "P_max": 0.9,
        "C": DEFAULT_SWITCHING_COST,
    }

    solver = ClosedEconomySolver(params, gp)
    V0, policy = solver.solve()

    assert V0.shape == (3, 8, 10)
    assert policy.shape == (7, 3, 8, 10)
    assert np.all(V0 >= -1e-9)

    df = simulate_closed_economy(
        solver, params,
        T_days=7.0, Delta=1.0 / 96.0,
        P0=0.55, Y0=float(params["beta"]),
        regime0=0, seed=0,
    )
    assert len(df) == 7 * 96
    assert set(df.columns) >= {"t", "P", "Y", "regime", "running_cost", "switch_cost"}
    assert df["P"].between(0.2 - 1e-9, 0.9 + 1e-9).all()

    m = compute_metrics(df, params, 1.0 / 96.0, mode="closed")
    assert m["total_cost"] >= 0
    assert m["running_cost"] + m["switching_cost"] == pytest.approx(m["total_cost"], rel=1e-9)
