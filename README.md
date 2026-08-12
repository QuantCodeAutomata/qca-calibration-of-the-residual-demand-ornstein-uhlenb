# Nuclear Power Plant Optimal Switching — OU Calibration

This repository implements the Ornstein–Uhlenbeck (OU) model with a seasonal
Fourier mean for Italian residual electricity demand, as a first step toward
the optimal nuclear unit switching problem.

## Project Structure

```
.
├── data/                          # Data storage (CSV, parquet)
│   └── .gitkeep
├── src/                           # Core library
│   ├── __init__.py
│   ├── ou_model.py                # OU process + Fourier seasonal mean
│   ├── data_loader.py             # Synthetic data generation / loader
│   └── utils.py                  # Shared diagnostics utilities
├── exp/                           # Experiment entry points
│   └── exp1_ou_calibration.py    # Experiment 1: profile OU estimation
├── tests/                         # pytest test suite
│   ├── __init__.py
│   ├── test_ou_model.py
│   └── test_exp1.py
├── results/                       # Saved artefacts
│   └── RESULTS.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Model

The residual demand process `Y_t` (normalised by a reference capacity `C_ref`)
follows a mean-reverting OU process with a deterministic seasonal mean:

```
dY_t = −κ (Y_t − θ_t) dt + ν dW_t
```

where the Fourier seasonal mean is

```
θ_t = β + Σ_j [ ζ_j cos(ω_j t) + η_j sin(ω_j t) ]
```

with nine angular frequencies `ω_j = 2π / P_j` and periods (in days):

| j | Period P_j         | Interpretation         |
|---|--------------------|------------------------|
| 1 | 1/4                | 6-hour intra-day       |
| 2 | 1/3                | 8-hour intra-day       |
| 3 | 1/2                | 12-hour intra-day      |
| 4 | 1                  | Daily                  |
| 5 | 7/2                | 3.5-day intra-week     |
| 6 | 7                  | Weekly                 |
| 7 | 365.25/4           | Quarterly              |
| 8 | 365.25/2           | Semi-annual            |
| 9 | 365.25             | Annual                 |

### Published Parameters (Appendix B)

| Parameter | Value  |
|-----------|--------|
| κ         | 0.3500 |
| β         | 0.6118 |
| ν         | 0.1114 |

(See `src/ou_model.py` for the full Fourier coefficient vectors `ζ` and `η`.)

### Estimation — Profile Likelihood (Appendix B)

For each candidate `φ = exp(−κ Δ)` on a grid the exact-discretisation
regression

```
Y_{t+Δ} = φ Y_t + (1−φ) θ_t + ε_t ,   ε_t ~ N(0, σ²)
```

is solved analytically for the seasonal parameters, and the resulting SSR is
recorded.  The grid value minimising SSR defines `φ*`; scalar parameters are
recovered as:

```
κ = −ln(φ*) / Δ
ν² = 2κ σ² / (1 − exp(−2κΔ))
```

## Data

Real Italian TSO (Terna) 15-minute residual demand data must be downloaded
manually from the Terna Download Center:

> <https://www.terna.it/en/electric-system/transparency/download-center>

Place the resulting CSV under `data/` and update the `data_path` argument in
`exp/exp1_ou_calibration.py`.  **Experiment 1 uses fully synthetic data**
generated from the published parameters so it runs without network access.

> **DST assumption**: all timestamps are kept in UTC to avoid daylight-saving
> discontinuities.  Terna data should be converted to UTC before use.

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Experiment 1: profile OU calibration on synthetic data
python exp/exp1_ou_calibration.py
```

Output artefacts are saved to `results/`.

## Running Tests

```bash
pytest tests/ -v
```

## Results

See `results/RESULTS.md` for a description of all saved artefacts and a
summary of Experiment 1 outputs.
