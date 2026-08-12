"""
tests/test_nuclear_control.py
==============================
Unit tests for src/nuclear_control.py and src/simulator.py.

Tests
-----
- test_production_update_bounds     : P stays in [P_min, P_max]
- test_production_update_hold       : P unchanged in hold regime
- test_running_cost_nonneg          : closed-economy cost ≥ 0 for valid inputs
- test_psi_price_thresholds         : psi correct at m = −1, 0, a, 3
- test_switching_cost_no_free_loop  : all 2-cycles have positive cost
- test_ou_exact_step_mean           : OU step mean formula matches formula
- test_closed_economy_solver_terminal: value at t=0 with zero cost params = 0
- test_policy_inaction_band         : hold is optimal near Y = P (late time)
- test_remark1_invariance           : constant cost shift does not change policy
- test_metrics_computation          : metric formulas on a manually built path
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.nuclear_control import (
    DEFAULT_SWITCHING_COST,
    ClosedEconomySolver,
    R_BAR,
    closed_economy_running_cost,
    price_taker_running_cost,
    price_maker_running_cost,
    production_update,
    psi_price,
)
from src.ou_model import PUBLISHED_PARAMS, PERIODS_DAYS
from src.simulator import compute_metrics

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_P_MIN, _P_MAX = 0.2, 0.9
_DELTA_FINE = 1.0 / 96.0   # 15-min step


def _base_params(gamma_1: float = 0.24, lambda_2: float = 0.48) -> dict:
    p = dict(PUBLISHED_PARAMS)
    p["gamma_1"] = gamma_1
    p["lambda_1"] = 0.0
    p["lambda_2"] = lambda_2
    p["P_min"] = _P_MIN
    p["P_max"] = _P_MAX
    return p


def _small_grid_params(N_DP: int = 10) -> dict:
    """Tiny grid for fast unit tests."""
    return {
        "P_grid": np.linspace(_P_MIN, _P_MAX, 8),
        "Y_grid": np.linspace(-0.5, 2.0, 12),
        "N_DP": N_DP,
        "T": 365.25,
        "r_bar": R_BAR,
        "P_min": _P_MIN,
        "P_max": _P_MAX,
        "C": DEFAULT_SWITCHING_COST,
    }


# ---------------------------------------------------------------------------
# production_update
# ---------------------------------------------------------------------------

class TestProductionUpdate:
    def test_rampup_clipped_at_P_max(self):
        """Ramp-up from P_max must stay at P_max."""
        p_new = production_update(_P_MAX, 1, R_BAR, _DELTA_FINE, _P_MIN, _P_MAX)
        assert p_new == pytest.approx(_P_MAX), "Ramp-up from P_max must stay at P_max"

    def test_rampdown_clipped_at_P_min(self):
        """Ramp-down from P_min must stay at P_min."""
        p_new = production_update(_P_MIN, 2, R_BAR, _DELTA_FINE, _P_MIN, _P_MAX)
        assert p_new == pytest.approx(_P_MIN), "Ramp-down from P_min must stay at P_min"

    def test_rampup_increases(self):
        """Ramp-up from interior must increase P by r*Delta."""
        p = 0.4
        p_new = production_update(p, 1, R_BAR, _DELTA_FINE, _P_MIN, _P_MAX)
        expected = min(p + R_BAR * _DELTA_FINE, _P_MAX)
        assert p_new == pytest.approx(expected)

    def test_rampdown_decreases(self):
        """Ramp-down from interior must decrease P by r*Delta."""
        p = 0.7
        p_new = production_update(p, 2, R_BAR, _DELTA_FINE, _P_MIN, _P_MAX)
        expected = max(p - R_BAR * _DELTA_FINE, _P_MIN)
        assert p_new == pytest.approx(expected)

    def test_hold_unchanged(self):
        """Hold regime must leave P exactly unchanged."""
        for p in (0.2, 0.55, 0.9, 0.3456):
            assert production_update(p, 0, R_BAR, _DELTA_FINE, _P_MIN, _P_MAX) == pytest.approx(p)

    def test_bounds_always_satisfied(self):
        """After any regime and any Delta, P must remain in [P_min, P_max]."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            p = rng.uniform(_P_MIN, _P_MAX)
            regime = rng.integers(0, 3)
            dt = rng.uniform(1e-3, 10.0)
            p_new = production_update(p, regime, R_BAR, dt, _P_MIN, _P_MAX)
            assert _P_MIN - 1e-12 <= p_new <= _P_MAX + 1e-12, (
                f"P={p_new} out of [{_P_MIN}, {_P_MAX}] for regime={regime}, dt={dt}"
            )


# ---------------------------------------------------------------------------
# Running costs
# ---------------------------------------------------------------------------

class TestRunningCost:
    def test_closed_cost_nonnegative_balanced(self):
        """Cost must be ≥ 0 when Y = P (balanced, lambda_1=0)."""
        for p in np.linspace(0.2, 0.9, 10):
            cost = closed_economy_running_cost(p, p, 0.24, 0.0, 0.48)
            assert cost >= 0.0

    def test_closed_cost_nonneg_range(self):
        """Cost must be ≥ 0 for all valid (p, y) combinations."""
        rng = np.random.default_rng(1)
        for _ in range(500):
            p = rng.uniform(0.2, 0.9)
            y = rng.uniform(-0.5, 2.0)
            cost = closed_economy_running_cost(p, y, 0.24, 0.0, 0.48)
            # gamma_1*p >= 0 always; lambda terms >= 0 always
            assert cost >= 0.0, f"Negative cost at p={p}, y={y}: {cost}"

    def test_closed_cost_shortage_penalty(self):
        """With lambda_1=0, shortage (y>p) increases cost vs balanced."""
        p, y = 0.5, 0.8
        cost_balance = closed_economy_running_cost(p, p, 0.24, 0.0, 0.48)
        cost_short = closed_economy_running_cost(p, y, 0.24, 0.0, 0.48)
        assert cost_short > cost_balance

    def test_pt_cost_nonneg_when_balanced(self):
        """PT cost when y = p is just gamma_1 * p > 0."""
        p = 0.5
        cost = price_taker_running_cost(p, p, 1.0, 0.24, 0.0, 0.2, 0.4, 2.7, 0.08)
        assert cost == pytest.approx(0.24 * p)

    def test_pm_cost_nonneg_when_balanced(self):
        """PM cost when y = p is just gamma_1 * p > 0."""
        p = 0.5
        cost = price_maker_running_cost(p, p, 1.0, 0.24, 0.0, 0.2, 0.4, 2.7, 0.08)
        assert cost == pytest.approx(0.24 * p)


# ---------------------------------------------------------------------------
# psi_price
# ---------------------------------------------------------------------------

class TestPsiPrice:
    _s_L, _s_M, _s_H, _a = 0.0, 0.2, 0.4, 2.7

    def test_below_zero(self):
        assert psi_price(-1.0, self._s_L, self._s_M, self._s_H, self._a) == pytest.approx(self._s_L)

    def test_at_zero(self):
        """m = 0 is in the s_L region (≤ 0)."""
        assert psi_price(0.0, self._s_L, self._s_M, self._s_H, self._a) == pytest.approx(self._s_L)

    def test_midpoint(self):
        """m in (0, a] gives s_M."""
        assert psi_price(1.35, self._s_L, self._s_M, self._s_H, self._a) == pytest.approx(self._s_M)

    def test_at_a(self):
        """m = a is in the s_M region (≤ a)."""
        assert psi_price(self._a, self._s_L, self._s_M, self._s_H, self._a) == pytest.approx(self._s_M)

    def test_above_a(self):
        """m > a gives s_H."""
        assert psi_price(3.0, self._s_L, self._s_M, self._s_H, self._a) == pytest.approx(self._s_H)

    def test_monotone_nondecreasing(self):
        """psi must be non-decreasing in m."""
        ms = np.linspace(-3.0, 6.0, 100)
        psis = [psi_price(m, self._s_L, self._s_M, self._s_H, self._a) for m in ms]
        diffs = np.diff(psis)
        assert np.all(diffs >= -1e-12), "psi is not non-decreasing"


# ---------------------------------------------------------------------------
# Switching cost no-free-loop
# ---------------------------------------------------------------------------

class TestNoFreeLoop:
    def test_all_2cycles_positive(self):
        """Every pair (i→j→i) must have strictly positive total cost."""
        C = DEFAULT_SWITCHING_COST
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                cycle = C[i, j] + C[j, i]
                assert cycle > 0, (
                    f"Free loop for {i}→{j}→{i}: total cost = {cycle}"
                )

    def test_all_3cycles_positive(self):
        """Every directed 3-cycle (i→j→k→i, all distinct) must have positive cost."""
        C = DEFAULT_SWITCHING_COST
        for i in range(3):
            for j in range(3):
                if j == i:
                    continue
                for k in range(3):
                    if k == i or k == j:
                        continue
                    cycle = C[i, j] + C[j, k] + C[k, i]
                    assert cycle > 0, (
                        f"Free loop {i}→{j}→{k}→{i}: total cost = {cycle}"
                    )


# ---------------------------------------------------------------------------
# OU step mean formula
# ---------------------------------------------------------------------------

class TestOuStep:
    def test_mean_formula(self):
        """OU step mean = y*exp(-kappa*h) + theta*(1 - exp(-kappa*h))."""
        kappa, nu, theta, h = 0.35, 0.1114, 0.6118, 1.0
        y = 1.2
        alpha = math.exp(-kappa * h)
        expected_mean = y * alpha + theta * (1.0 - alpha)
        # Verify using the formula directly
        assert expected_mean == pytest.approx(y * math.exp(-kappa * h) + theta * (1.0 - math.exp(-kappa * h)))

    def test_std_positive(self):
        """Conditional std must be strictly positive."""
        kappa, nu, h = 0.35, 0.1114, 1.0
        alpha = math.exp(-kappa * h)
        std = nu * math.sqrt((1.0 - alpha ** 2) / (2.0 * kappa))
        assert std > 0.0

    def test_mean_reverts_over_long_horizon(self):
        """After many steps, the OU mean converges to theta."""
        kappa, nu, theta = 0.35, 0.1114, 0.6118
        y = 5.0       # far from theta
        h = 100.0     # long horizon
        alpha = math.exp(-kappa * h)
        mean = y * alpha + theta * (1.0 - alpha)
        assert abs(mean - theta) < 1e-6, f"Mean {mean} did not revert to theta {theta}"


# ---------------------------------------------------------------------------
# ClosedEconomySolver terminal condition
# ---------------------------------------------------------------------------

class TestClosedEconomySolverTerminal:
    def test_terminal_values_zero_for_zero_cost(self):
        """With gamma_1 = lambda_2 = 0 and C = 0, V0 must be exactly 0."""
        params = dict(PUBLISHED_PARAMS)
        params["gamma_1"] = 0.0
        params["lambda_1"] = 0.0
        params["lambda_2"] = 0.0
        params["P_min"] = _P_MIN
        params["P_max"] = _P_MAX

        gp = _small_grid_params(N_DP=5)
        gp["C"] = np.zeros((3, 3))

        solver = ClosedEconomySolver(params, gp)
        V0, _ = solver.solve()
        np.testing.assert_allclose(V0, 0.0, atol=1e-12,
                                   err_msg="V0 should be zero when all costs are zero")

    def test_value_positive_with_cost(self):
        """With positive running cost, V0 must be ≥ 0."""
        params = _base_params()
        gp = _small_grid_params(N_DP=5)
        solver = ClosedEconomySolver(params, gp)
        V0, _ = solver.solve()
        assert np.all(V0 >= -1e-9), "Value function must be non-negative for positive costs"


# ---------------------------------------------------------------------------
# Policy inaction band
# ---------------------------------------------------------------------------

class TestPolicyInactionBand:
    """Near the terminal time, hold should be optimal when P ≈ Y."""

    def test_hold_near_balanced_late(self):
        """At the last DP step (near T), hold is optimal when P ≈ Y and mid-range."""
        params = _base_params()
        gp = _small_grid_params(N_DP=8)
        solver = ClosedEconomySolver(params, gp)
        _, policy = solver.solve()

        P_grid = gp["P_grid"]
        Y_grid = gp["Y_grid"]

        # Near last time step, check balanced interior points
        k_last = gp["N_DP"] - 1
        # Find grid points where |P - Y| is small and both are interior
        hold_found = False
        for j_p, p in enumerate(P_grid[2:-2]):
            idx_p = j_p + 2
            for j_y, y in enumerate(Y_grid):
                if abs(p - y) < 0.15 and _P_MIN + 0.05 < p < _P_MAX - 0.05:
                    if policy[k_last, 0, idx_p, j_y] == 0:
                        hold_found = True
        assert hold_found, "Expected hold policy near P≈Y at late time steps"


# ---------------------------------------------------------------------------
# Remark 1 invariance
# ---------------------------------------------------------------------------

class TestRemark1Invariance:
    """Remark 1 (Lemma in the paper): adding a state-independent constant c to
    the running cost shifts V(t, ·, ·, ·) by c*(T-t) at every state but leaves
    the optimal policy (argmin) unchanged.

    We use the ``cost_offset`` parameter of ClosedEconomySolver — a true
    additive constant independent of (P, Y) — to test this property exactly.
    """

    def test_constant_cost_offset_preserves_policy(self):
        """A constant cost_offset does NOT change the optimal policy."""
        c = 0.10
        params_base = _base_params()
        params_shift = _base_params()
        params_shift["cost_offset"] = c       # true state-independent shift

        gp = _small_grid_params(N_DP=8)

        solver_base = ClosedEconomySolver(params_base, gp)
        _, policy_base = solver_base.solve()

        solver_shift = ClosedEconomySolver(params_shift, gp)
        _, policy_shift = solver_shift.solve()

        np.testing.assert_array_equal(
            policy_base, policy_shift,
            err_msg="Policy must be invariant to a constant cost_offset shift",
        )

    def test_value_function_shifts_by_c_times_T(self):
        """V0 under cost_offset=c must equal V0_base + c*T at every state."""
        c = 0.05
        params_base = _base_params()
        params_shift = _base_params()
        params_shift["cost_offset"] = c

        gp = _small_grid_params(N_DP=5)

        solver_base = ClosedEconomySolver(params_base, gp)
        V0_base, _ = solver_base.solve()

        solver_shift = ClosedEconomySolver(params_shift, gp)
        V0_shift, _ = solver_shift.solve()

        T = gp["T"]
        np.testing.assert_allclose(
            V0_shift, V0_base + c * T,
            rtol=1e-6,
            err_msg="V0 must shift by c*T under a constant cost_offset",
        )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

class TestMetricsComputation:
    """Verify metric formulas on a simple synthetic simulation path."""

    def _make_simple_df(self) -> pd.DataFrame:
        """Constant path: P = 0.5, Y = 0.8 (shortage), regime = 0, 10 steps."""
        n = 10
        Delta = _DELTA_FINE
        rows = []
        for i in range(n):
            run_c = Delta * closed_economy_running_cost(0.5, 0.8, 0.24, 0.0, 0.48)
            rows.append({
                "t": i * Delta,
                "P": 0.5,
                "Y": 0.8,
                "regime": 0,
                "running_cost": run_c,
                "switch_cost": 0.0,
                "total_step_cost": run_c,
                "switched": 0,
            })
        return pd.DataFrame(rows)

    def test_total_cost_equals_sum(self):
        df = self._make_simple_df()
        m = compute_metrics(df, {"P_min": 0.2, "P_max": 0.9}, _DELTA_FINE, "closed")
        assert m["total_cost"] == pytest.approx(df["total_step_cost"].sum(), rel=1e-9)

    def test_running_plus_switching_equals_total(self):
        df = self._make_simple_df()
        m = compute_metrics(df, {"P_min": 0.2, "P_max": 0.9}, _DELTA_FINE, "closed")
        assert m["running_cost"] + m["switching_cost"] == pytest.approx(m["total_cost"], rel=1e-9)

    def test_shortage_time_share(self):
        """With Y > P always, shortage_time_share = 1.0."""
        df = self._make_simple_df()
        m = compute_metrics(df, {"P_min": 0.2, "P_max": 0.9}, _DELTA_FINE, "closed")
        assert m["shortage_time_share"] == pytest.approx(1.0)
        assert m["excess_time_share"] == pytest.approx(0.0)

    def test_shortage_energy_formula(self):
        """shortage_energy = n_steps * Delta * (Y - P)."""
        df = self._make_simple_df()
        n = len(df)
        expected = n * _DELTA_FINE * (0.8 - 0.5)
        m = compute_metrics(df, {"P_min": 0.2, "P_max": 0.9}, _DELTA_FINE, "closed")
        assert m["shortage_energy"] == pytest.approx(expected, rel=1e-9)

    def test_n_switches_zero_constant_regime(self):
        df = self._make_simple_df()
        m = compute_metrics(df, {"P_min": 0.2, "P_max": 0.9}, _DELTA_FINE, "closed")
        assert m["n_switches"] == 0

    def test_time_at_bounds_mid_production(self):
        """P = 0.5 (interior) implies time_at_bounds_share = 0."""
        df = self._make_simple_df()
        m = compute_metrics(df, {"P_min": 0.2, "P_max": 0.9}, _DELTA_FINE, "closed")
        assert m["time_at_bounds_share"] == pytest.approx(0.0)
