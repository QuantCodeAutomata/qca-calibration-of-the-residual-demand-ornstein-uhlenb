# Nuclear Power Plant Optimal Switching — Repository Notes

## Project Overview
Implements the full paper pipeline: Experiment 1 (OU calibration on Italian
residual demand) plus Experiments 2 (closed economy) and 3 (open economy,
price-taker and price-maker).  The control problem determines when a plant
should hold, ramp up, or ramp down via backward DP + forward Monte Carlo.

## Experiment 1 Key Notes (`src/ou_model.py`, `exp/exp1_ou_calibration.py`)

### phi_grid Resolution for 15-min Data
For kappa=0.35/day and Delta=15/1440 days, phi=exp(-kappa*Delta)≈0.9964.
A uniform phi_grid on [0.001, 0.999] (step≈0.005) **cannot resolve the
minimum**; grid points nearest to the correct phi differ by kappa of ±0.25.
**Always construct phi_grid from a uniform kappa_grid**:
```python
kappa_grid = np.linspace(0.01, 10.0, 200)
phi_grid = np.exp(-kappa_grid * DELTA)  # DELTA = 15/1440
```

### OU Innovation vs Y-theta Variance
With time-varying seasonal mean theta_t, `Var(Y_t − theta_t) ≠ nu²/(2κ)`.
The OU process tracks high-frequency seasonals (6h period) very poorly when
kappa=0.35/day (mean-reversion time ≈3 days); Var(Y-theta) ≈ Var(theta) ≈ 8.9
for the published parameters.

The directly-testable quantity is the **innovation variance**:
```
Var(ε_t) = Var(Y_{t+Δ} − φ Y_t − (1−φ) θ_t) = ν²(1−φ²)/(2κ) ≈ 0.000129
```

### Correct Diagnostic Residuals
Use one-step innovations `ε_t = Y_{t+Δ} − φ Y_t − (1−φ) θ_t` for ACF,
histogram, and QQ-plot diagnostics (these should be i.i.d. N(0,1) under H₀).
Do NOT use Y_t − θ_t, which is autocorrelated by construction.

### build_fourier_design Edge Case
`pd.DataFrame({})` creates a 0-row DataFrame; pass `index=range(len(t))`
explicitly so the result always has N rows even when periods_days=[].

### Profile Estimation Vectorisation
Pre-compute r1 = M_W Y_next, r2 = M_W Y_curr (one QR solve).
Then SSR(phi) = r1·r1 − 2φ r1·r2 + φ² r2·r2 is evaluated on the entire grid
without any loop.  O(N·p²) one-time cost + O(grid_size) evaluation.

## Directory Layout
```
src/
  ou_model.py          – OU process utilities (seasonal mean, exact step)
  nuclear_control.py   – Core DP solvers (ClosedEconomySolver, OpenEconomySolver)
  simulator.py         – Forward simulation + metric computation
exp/
  exp2_closed_economy.py
  exp3_open_economy.py
tests/
  test_nuclear_control.py   – unit tests
  test_exp2.py / test_exp3.py – integration stubs
results/                     – all plots and RESULTS.md written here
```

## Published Parameters
| Parameter | Value |
|-----------|-------|
| kappa | 0.3500 |
| beta  | 0.6118 |
| nu    | 0.1114 |
| zeta  | [0.4100, 0.1606, -2.4238, -1.5101, 0.0841, 0.2984, -0.0113, 0.0563, 0.0912] |
| eta   | [0.2714, -0.6401, 2.8156, -0.9522, -0.2479, 0.0982, -0.0162, 0.0451, -0.0527] |
| Periods (days) | [1/4, 1/3, 1/2, 1, 7/2, 7, 365.25/4, 365.25/2, 365.25] |
| Delta | 1/96 day (15 min) |
| T     | 365.25 days |
| N_steps | 35064 |
| P_min | 0.2 |
| P_max | 0.9 |
| r_bar | 4.8 per day (= 0.05 per 15-min step) |

## Regime Indexing
- 0 = hold
- 1 = ramp_up (+r)
- 2 = ramp_down (−r)

## Switching Cost Matrix  (indices: 0=hold, 1=+r, 2=−r)
```
C = [[0,      4.8e-4, 1.6e-4],   # from hold
     [0.4e-4, 0,      1.6e-4],   # from ramp_up
     [4e-4,   7e-4,   0      ]]  # from ramp_down
```

## Numerical Approximation in DP
The full 35 064-step × 3D grid DP is intractable in Python.  We use:
- **Closed economy**: N_DP = 365 (daily steps), 2-D grid P×Y (40×80)
- **Open economy**:   N_DP = 52  (weekly steps), 3-D grid P×Y×M (30×50×60)
- Rademacher (two-point) innovations approximate the OU noise distribution.
- Forward simulations always run at full 15-min resolution (35 064 steps) using
  exact OU transitions.
- Numba @jit is used for the inner DP loops (bilinear/trilinear interpolation).

## Running the Code
```bash
cd /workspace/project
python -m pytest tests/ -v           # unit + integration tests
python exp/exp2_closed_economy.py    # writes results/exp2_*
python exp/exp3_open_economy.py      # writes results/exp3_*
```

## Key Design Decisions
1. `RegularGridInterpolator` (scipy) is used for the policy lookup during
   forward simulation; numba-JIT bilinear/trilinear is used during DP backward
   induction to avoid Python-loop overhead.
2. Only two DP value-function slices (current + next) are kept in memory;
   the full policy array is stored as `int8`.
3. The coarsened DP produces a daily (or weekly) decision boundary; the forward
   simulator looks up the nearest past DP time step for the current policy.

## Numba JIT Constraints
- JIT-compiled functions can only call other JIT-compiled functions.
  `_bilinear2d` and `_trilinear3d` must be compiled (@jit) **before** they are
  referenced inside `_dp_step_closed` / `_dp_step_open`.
- `nopython=True` (default in recent numba) forbids any Python objects; use
  plain float64 arrays and scalar arguments only.
- Cache must be cleared after restructuring the JIT call graph:
  `find . -name "__pycache__" -type d | xargs rm -rf`

## Remark 1 Invariance (cost_offset)
- `ClosedEconomySolver.solve()` accepts `cost_offset: float` (default 0).
- Adding a constant `c` to every running cost shifts `V0` by `c * T` exactly;
  the optimal policy is unchanged.
- The `verify_remark1_invariance` experiment helper uses `mismatch_tol=0.001`
  (0.1 %) to allow floating-point tie-breaking differences at grid boundaries.

## RESULTS.md Is Idempotent
- `_write_exp2_results` / `_write_exp3_results` use `_replace_section_in_results`,
  which finds the `## Experiment N:` heading and replaces its content in-place.
- Re-running experiments does **not** duplicate sections in RESULTS.md.
- Exp1 section header is `## Experiment 1: OU Model Calibration …` and is never
  touched by Exp2/Exp3 scripts.
