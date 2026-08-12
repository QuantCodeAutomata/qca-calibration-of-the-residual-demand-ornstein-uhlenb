"""
utils.py
========
Shared diagnostic utilities for the OU calibration project.
"""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.stattools import acf as _sm_acf


def compute_acf_diagnostics(
    residuals: np.ndarray,
    nlags: int = 100,
) -> np.ndarray:
    """Return the sample autocorrelation function of *residuals*.

    Parameters
    ----------
    residuals:
        1-D array of (standardised) residuals.
    nlags:
        Number of lags to compute (lag 0 through *nlags*).

    Returns
    -------
    numpy.ndarray
        ACF values at lags 0, 1, …, *nlags*, shape ``(nlags + 1,)``.
    """
    residuals = np.asarray(residuals, dtype=float)
    return _sm_acf(residuals, nlags=nlags, fft=True)


def compute_standardized_residuals(
    Y: np.ndarray,
    fitted_mean: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Compute standardised residuals (Y − fitted) / σ.

    For OU diagnostics pass the one-step-ahead AR fitted values
    ``fitted = φ Y_t + (1−φ) θ_t`` and the corresponding conditional
    standard deviation ``σ = ν √((1−φ²)/(2κ))``.  The result is then
    approximately i.i.d. N(0, 1) under the correctly-specified model.

    Parameters
    ----------
    Y:
        Observed values, shape ``(N,)``.
    fitted_mean:
        Fitted / predicted values aligned with *Y*, shape ``(N,)``.
    sigma:
        Standard deviation used for normalisation (> 0).

    Returns
    -------
    numpy.ndarray
        Standardised residuals, shape ``(N,)``.
    """
    Y = np.asarray(Y, dtype=float)
    fitted_mean = np.asarray(fitted_mean, dtype=float)
    return (Y - fitted_mean) / sigma
