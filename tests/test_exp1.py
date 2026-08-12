"""
tests/test_exp1.py
==================
Integration tests for exp/exp1_ou_calibration.py.

These tests run the full experiment pipeline and verify:
- The script completes without raising exceptions.
- All expected output files are created.
- The JSON parameter file has the required structure and valid values.
- The diagnostics CSV has the required columns.
- Estimated parameters satisfy basic physical constraints.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers to import the experiment module
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EXP1_PATH = _PROJECT_ROOT / "exp" / "exp1_ou_calibration.py"
_RESULTS_DIR = _PROJECT_ROOT / "results"


def _import_exp1():
    """Dynamically import exp1_ou_calibration so sys.path is set correctly."""
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location("exp1_ou_calibration", _EXP1_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def exp1_result():
    """Run the experiment once and share the result across tests in this module."""
    mod = _import_exp1()
    result = mod.main()
    return result


class TestExp1Runs:
    def test_main_returns_dict(self, exp1_result):
        """main() must return a dict (the estimation result)."""
        assert isinstance(exp1_result, dict)

    def test_result_has_required_keys(self, exp1_result):
        """Returned dict must contain kappa, beta, nu, zeta, eta."""
        for key in ("kappa", "beta", "nu", "zeta", "eta"):
            assert key in exp1_result, f"Missing key '{key}' in result"

    def test_estimated_kappa_positive(self, exp1_result):
        """Estimated κ must be strictly positive."""
        assert exp1_result["kappa"] > 0.0

    def test_estimated_nu_positive(self, exp1_result):
        """Estimated ν must be non-negative."""
        assert exp1_result["nu"] >= 0.0

    def test_phi_star_in_unit_interval(self, exp1_result):
        """Optimal φ* must lie in (0, 1)."""
        assert 0.0 < exp1_result["phi_star"] < 1.0


class TestExp1OutputFiles:
    """Verify all expected artefacts are written to results/."""

    _EXPECTED_FILES = [
        "exp1_parameters.json",
        "exp1_diagnostics.csv",
        "exp1_fitted_vs_realized.png",
        "exp1_residual_histogram.png",
        "exp1_qq.png",
        "exp1_acf_residuals.png",
        "exp1_acf_squared_residuals.png",
    ]

    @pytest.mark.parametrize("filename", _EXPECTED_FILES)
    def test_output_file_exists(self, exp1_result, filename):
        """Each expected output file must exist after main() completes."""
        path = _RESULTS_DIR / filename
        assert path.exists(), f"Expected output file not found: {path}"

    @pytest.mark.parametrize("filename", _EXPECTED_FILES)
    def test_output_file_nonempty(self, exp1_result, filename):
        """Each expected output file must be non-empty."""
        path = _RESULTS_DIR / filename
        assert path.stat().st_size > 0, f"Output file is empty: {path}"


class TestExp1ParametersJson:
    """Verify the structure and content of exp1_parameters.json."""

    @pytest.fixture(scope="class")
    def params_json(self, exp1_result):
        path = _RESULTS_DIR / "exp1_parameters.json"
        with open(path) as fh:
            return json.load(fh)

    def test_top_level_keys(self, params_json):
        for key in ("published", "estimated", "metadata"):
            assert key in params_json

    def test_published_scalar_keys(self, params_json):
        for key in ("kappa", "beta", "nu", "zeta", "eta"):
            assert key in params_json["published"]

    def test_estimated_scalar_keys(self, params_json):
        for key in ("kappa", "beta", "nu", "zeta", "eta", "phi_star"):
            assert key in params_json["estimated"]

    def test_fourier_vectors_length(self, params_json):
        assert len(params_json["published"]["zeta"]) == 9
        assert len(params_json["published"]["eta"]) == 9
        assert len(params_json["estimated"]["zeta"]) == 9
        assert len(params_json["estimated"]["eta"]) == 9

    def test_metadata_n_obs_positive(self, params_json):
        assert params_json["metadata"]["n_obs"] > 0

    def test_metadata_timestamps_utc(self, params_json):
        assert params_json["metadata"]["timestamps"] == "UTC"


class TestExp1DiagnosticsCSV:
    """Verify the structure of exp1_diagnostics.csv."""

    @pytest.fixture(scope="class")
    def diag_df(self, exp1_result):
        return pd.read_csv(_RESULTS_DIR / "exp1_diagnostics.csv")

    def test_required_columns(self, diag_df):
        for col in ("parameter", "published", "estimated", "relative_error_pct"):
            assert col in diag_df.columns, f"Column '{col}' missing from diagnostics CSV"

    def test_three_scalar_parameters(self, diag_df):
        """CSV must contain exactly three rows (κ, β, ν)."""
        assert len(diag_df) == 3

    def test_parameters_are_kappa_beta_nu(self, diag_df):
        assert set(diag_df["parameter"]) == {"kappa", "beta", "nu"}

    def test_relative_errors_nonnegative(self, diag_df):
        assert (diag_df["relative_error_pct"] >= 0.0).all()
