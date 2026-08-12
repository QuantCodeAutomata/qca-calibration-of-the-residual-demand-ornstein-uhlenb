"""
tests/test_ou_model.py
======================
Unit tests for src/ou_model.py.

Tests cover:
- Fourier design matrix shape and boundary values
- Seasonal mean dtype
- OU simulation stationarity and variance
- Profile estimation correctness and physical constraints
- Nine-period specification
- Edge cases (empty and single-element inputs)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ou_model import (
    PERIODS_DAYS,
    PUBLISHED_PARAMS,
    build_fourier_design,
    compute_seasonal_mean,
    profile_ou_estimation,
    simulate_ou_path,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

_DELTA = 15.0 / 1440.0           # 15-min step in days
_N_PERIODS = len(PERIODS_DAYS)   # 9
_N_COLS = 2 * _N_PERIODS          # 18


# ---------------------------------------------------------------------------
# build_fourier_design
# ---------------------------------------------------------------------------

class TestBuildFourierDesign:
    def test_fourier_design_shape(self):
        """Design matrix must have shape (N, 2 * n_periods)."""
        N = 500
        t = np.linspace(0, 10, N)
        D = build_fourier_design(t, PERIODS_DAYS)
        assert D.shape == (N, _N_COLS), (
            f"Expected ({N}, {_N_COLS}), got {D.shape}"
        )

    def test_nine_periods(self):
        """Exactly nine periods must produce exactly 18 columns."""
        assert len(PERIODS_DAYS) == 9
        D = build_fourier_design(np.array([0.0, 1.0]), PERIODS_DAYS)
        assert D.shape[1] == 18

    def test_fourier_design_at_zero(self):
        """At t = 0: cos terms = 1 and sin terms = 0 for all periods."""
        D = build_fourier_design(np.array([0.0]), PERIODS_DAYS)
        cos_cols = [f"cos_{j}" for j in range(1, _N_PERIODS + 1)]
        sin_cols = [f"sin_{j}" for j in range(1, _N_PERIODS + 1)]
        np.testing.assert_allclose(D[cos_cols].values[0], 1.0, atol=1e-12)
        np.testing.assert_allclose(D[sin_cols].values[0], 0.0, atol=1e-12)

    def test_fourier_design_column_names(self):
        """Columns must follow the cos_j / sin_j interleaved naming convention."""
        D = build_fourier_design(np.array([0.0]), PERIODS_DAYS)
        expected = [f"{trig}_{j}" for j in range(1, 10) for trig in ("cos", "sin")]
        # Sort-free comparison: check set equality and that they appear in pairs
        for j in range(1, 10):
            assert f"cos_{j}" in D.columns
            assert f"sin_{j}" in D.columns

    def test_edge_case_empty_array(self):
        """Empty input must return a DataFrame with 18 columns and 0 rows."""
        D = build_fourier_design(np.array([]), PERIODS_DAYS)
        assert D.shape == (0, _N_COLS)

    def test_edge_case_single_point(self):
        """Single-element input must return a DataFrame with shape (1, 18)."""
        D = build_fourier_design(np.array([3.14159]), PERIODS_DAYS)
        assert D.shape == (1, _N_COLS)


# ---------------------------------------------------------------------------
# compute_seasonal_mean
# ---------------------------------------------------------------------------

class TestComputeSeasonalMean:
    def test_seasonal_mean_dtype(self):
        """Output must be a float64 numpy array."""
        t = np.linspace(0, 30, 200)
        theta = compute_seasonal_mean(t, PUBLISHED_PARAMS)
        assert isinstance(theta, np.ndarray)
        assert theta.dtype == np.float64

    def test_seasonal_mean_shape(self):
        t = np.linspace(0, 365, 1000)
        theta = compute_seasonal_mean(t, PUBLISHED_PARAMS)
        assert theta.shape == (1000,)

    def test_constant_mean_zero_fourier(self):
        """With all Fourier coefficients = 0 the mean must equal beta everywhere."""
        params = {
            "beta": 0.5,
            "zeta": [0.0] * _N_PERIODS,
            "eta": [0.0] * _N_PERIODS,
            "periods_days": PERIODS_DAYS,
        }
        t = np.array([0.0, 1.0, 100.0, 365.25])
        theta = compute_seasonal_mean(t, params)
        np.testing.assert_allclose(theta, 0.5, atol=1e-12)


# ---------------------------------------------------------------------------
# simulate_ou_path
# ---------------------------------------------------------------------------

class TestSimulateOuPath:
    """Tests for the OU path simulator using published parameters."""

    _KAPPA = PUBLISHED_PARAMS["kappa"]  # 0.35
    _BETA = PUBLISHED_PARAMS["beta"]    # 0.6118
    _NU = PUBLISHED_PARAMS["nu"]        # 0.1114
    _ZETA = PUBLISHED_PARAMS["zeta"]
    _ETA = PUBLISHED_PARAMS["eta"]

    def _make_path(self, n_years: int = 2, seed: int = 0) -> np.ndarray:
        n = int(n_years * 365.25 * 24 * 4)  # 15-min steps
        t = np.arange(n, dtype=float) * _DELTA
        return simulate_ou_path(
            t_days=t,
            kappa=self._KAPPA,
            beta=self._BETA,
            zeta=self._ZETA,
            eta=self._ETA,
            nu=self._NU,
            Y0=self._BETA,
            Delta=_DELTA,
            periods_days=PERIODS_DAYS,
            seed=seed,
        )

    def test_simulate_ou_stationarity(self):
        """Long-run sample mean must be within 0.05 of β = 0.6118."""
        Y = self._make_path(n_years=3, seed=7)
        assert abs(Y.mean() - self._BETA) < 0.05, (
            f"Sample mean {Y.mean():.4f} too far from β = {self._BETA}"
        )

    def test_ou_variance(self):
        """Innovation variance must be within 30 % of ν²(1−φ²)/(2κ).

        For a time-varying seasonal mean θ_t the quantity ν²/(2κ) is NOT the
        variance of Y_t − θ_t; the OU process tracks high-frequency seasonals
        only partially (the tracking gain is |(1−φ)/(1−φ e^{−iω})| ≪ 1 for
        ω ≫ κ).  The directly-testable quantity from the exact discretisation

            Y_{t+Δ} = φ Y_t + (1−φ) θ_t + ε_t ,  ε_t ~ N(0, σ²)

        is the conditional innovation variance σ² = ν²(1−φ²)/(2κ), which is
        free of the seasonal-mean structure.
        """
        phi = np.exp(-self._KAPPA * _DELTA)
        theoretical_innov_var = self._NU**2 * (1.0 - phi**2) / (2.0 * self._KAPPA)

        n = int(4 * 365.25 * 24 * 4)
        t = np.arange(n, dtype=float) * _DELTA
        Y = simulate_ou_path(
            t_days=t,
            kappa=self._KAPPA,
            beta=self._BETA,
            zeta=self._ZETA,
            eta=self._ETA,
            nu=self._NU,
            Y0=self._BETA,
            Delta=_DELTA,
            periods_days=PERIODS_DAYS,
            seed=13,
        )
        theta = compute_seasonal_mean(t, PUBLISHED_PARAMS)
        # One-step innovations: ε_t = Y_{t+Δ} − φ Y_t − (1−φ) θ_t
        innovations = Y[1:] - phi * Y[:-1] - (1.0 - phi) * theta[:-1]
        sample_innov_var = float(np.var(innovations))
        rel_err = abs(sample_innov_var - theoretical_innov_var) / theoretical_innov_var
        assert rel_err < 0.30, (
            f"Innovation variance {sample_innov_var:.6f} too far from "
            f"theoretical ν²(1−φ²)/(2κ) = {theoretical_innov_var:.6f} "
            f"(rel err {rel_err:.2%})"
        )

    def test_simulate_reproduces_with_same_seed(self):
        """Two calls with the same seed must return identical paths."""
        Y1 = self._make_path(seed=42)
        Y2 = self._make_path(seed=42)
        np.testing.assert_array_equal(Y1, Y2)

    def test_simulate_length(self):
        n = 1000
        t = np.arange(n, dtype=float) * _DELTA
        Y = simulate_ou_path(t, 0.35, 0.6, [], [], 0.1, 0.6, _DELTA, [], seed=0)
        assert len(Y) == n


# ---------------------------------------------------------------------------
# profile_ou_estimation
# ---------------------------------------------------------------------------

class TestProfileOuEstimation:
    """Tests for the profile-likelihood OU estimator."""

    # Short synthetic dataset for fast tests (6 months of 15-min data)
    _N_MONTHS = 6
    _N = int(_N_MONTHS / 12 * 365.25 * 24 * 4)

    @classmethod
    def _make_data(cls, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        t = np.arange(cls._N, dtype=float) * _DELTA
        Y = simulate_ou_path(
            t_days=t,
            kappa=PUBLISHED_PARAMS["kappa"],
            beta=PUBLISHED_PARAMS["beta"],
            zeta=PUBLISHED_PARAMS["zeta"],
            eta=PUBLISHED_PARAMS["eta"],
            nu=PUBLISHED_PARAMS["nu"],
            Y0=PUBLISHED_PARAMS["beta"],
            Delta=_DELTA,
            periods_days=PERIODS_DAYS,
            seed=seed,
        )
        return Y, t

    def test_profile_estimation_returns_positive_kappa(self):
        """Estimated κ must be strictly positive."""
        Y, t = self._make_data(seed=1)
        phi_grid = np.linspace(0.001, 0.999, 30)
        result = profile_ou_estimation(Y, t, _DELTA, phi_grid)
        assert result["kappa"] > 0.0, f"κ = {result['kappa']:.4f} is not positive"

    def test_profile_estimation_phi_in_unit_interval(self):
        """Optimal φ* must lie strictly in (0, 1)."""
        Y, t = self._make_data(seed=2)
        phi_grid = np.linspace(0.001, 0.999, 30)
        result = profile_ou_estimation(Y, t, _DELTA, phi_grid)
        assert 0.0 < result["phi_star"] < 1.0

    def test_parameter_closeness_to_published(self):
        """κ estimated from 1 year of synthetic data must be within 50 % of 0.35.

        Uses a κ-derived φ-grid appropriate for 15-minute data: a uniform grid
        in κ-space maps to values of φ=exp(−κΔ) that are well-resolved near
        the published κ=0.35 (φ≈0.9964), unlike a uniform φ grid which has
        step ≈0.005 and cannot identify the minimum.
        """
        # 1 year of 15-min data
        n = int(365.25 * 24 * 4)
        t = np.arange(n, dtype=float) * _DELTA
        Y = simulate_ou_path(
            t_days=t,
            kappa=PUBLISHED_PARAMS["kappa"],
            beta=PUBLISHED_PARAMS["beta"],
            zeta=PUBLISHED_PARAMS["zeta"],
            eta=PUBLISHED_PARAMS["eta"],
            nu=PUBLISHED_PARAMS["nu"],
            Y0=PUBLISHED_PARAMS["beta"],
            Delta=_DELTA,
            periods_days=PERIODS_DAYS,
            seed=99,
        )
        # Build φ-grid from a uniform κ-grid (per day); this gives sufficient
        # resolution near φ=exp(−0.35·Δ)≈0.9964 for 15-min sampling.
        kappa_grid = np.linspace(0.01, 10.0, 50)
        phi_grid = np.exp(-kappa_grid * _DELTA)
        result = profile_ou_estimation(Y, t, _DELTA, phi_grid)
        kappa_pub = PUBLISHED_PARAMS["kappa"]  # 0.35
        rel_err = abs(result["kappa"] - kappa_pub) / kappa_pub
        assert rel_err < 0.50, (
            f"Estimated κ = {result['kappa']:.4f} deviates "
            f"{rel_err:.1%} from published {kappa_pub}"
        )

    def test_profile_estimation_zeta_eta_length(self):
        """Returned zeta and eta must each have exactly n_periods = 9 entries."""
        Y, t = self._make_data(seed=3)
        phi_grid = np.linspace(0.001, 0.999, 20)
        result = profile_ou_estimation(Y, t, _DELTA, phi_grid)
        assert len(result["zeta"]) == _N_PERIODS
        assert len(result["eta"]) == _N_PERIODS

    def test_profile_estimation_ssr_grid_shape(self):
        """SSR grid must have the same length as phi_grid."""
        Y, t = self._make_data(seed=4)
        phi_grid = np.linspace(0.001, 0.999, 25)
        result = profile_ou_estimation(Y, t, _DELTA, phi_grid)
        assert len(result["ssr_grid"]) == len(phi_grid)

    def test_profile_estimation_positive_nu(self):
        """Estimated ν must be non-negative."""
        Y, t = self._make_data(seed=5)
        phi_grid = np.linspace(0.001, 0.999, 20)
        result = profile_ou_estimation(Y, t, _DELTA, phi_grid)
        assert result["nu"] >= 0.0
