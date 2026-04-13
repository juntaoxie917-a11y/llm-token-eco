# Competition Sensitivity Implementation Summary

This note summarizes what was implemented for competition sensitivity analysis in the current iteration.

## Scope Delivered

- Implemented separate 1D sensitivity workflows for `u0` and `tau`.
- Reused the same interior-equilibrium classification used by market-size threshold analysis.
- Added dedicated output namespaces for sensitivity tables while preserving backward-compatible top-level mirrors.
- Added Stage 9 regression safeguards to protect existing `M`-threshold workflow behavior.

## Key Design Decisions

- Non-invasive integration:

  Existing `M` threshold logic and scripts were kept intact.
  Sensitivity logic is implemented as wrappers around existing competition evaluation entry points.

- Separate 1D workflows:

  `u0` and `tau` sensitivities run independently.
  No mandatory 2D grid workflow is used in the current shipped flow.

- Shared equilibrium criterion:

  Sensitivity uses `evaluate_market_size_once(...)` from `src/competition_threshold.py`.
  This guarantees strict/weak interior classification consistency with the threshold pipeline.

## Code Changes by Stage

### Stage 2: Config support (`u0`, `tau`)

- File: `src/competition_sensitivity_config.py`
- Added/used parser and validation for:
  - `competition.sensitivity_analysis.u0_sweep`
  - `competition.sensitivity_analysis.tau_sweep`
- Validation rules:
  - `u0` grid values must be finite.
  - `tau` grid values must be finite and strictly positive.

### Stage 3: Sensitivity runners

- File: `src/competition_sensitivity.py`
- Implemented:
  - `run_u0_sensitivity(...)`
  - `run_tau_sensitivity(...)`
- Output rows include:
  - varied parameter/value, `M`, equilibrium/interior flags,
  - `p_star`, `D_star_at_p_star`, teacher/student payoffs,
  - downstream prices and shares,
  - solver/diagnostic fields.

### Stage 4: Consistency with threshold criterion

- Sensitivity runner internals call threshold evaluator.
- Consistency note captured in:
  - `docs/competition_sensitivity_stage4_consistency.md`

### Stage 5: `u0` sensitivity experiment

- File: `experiments/exp_40_competition_sensitivity_u0.py`
- Produces:
  - row-level CSV,
  - summary JSON,
  - diagnostics JSON,
  - parameter-domain and price-domain plots (teacher + student).

### Stage 6: `tau` sensitivity experiment

- File: `experiments/exp_41_competition_sensitivity_tau.py`
- Produces same artifact structure as `u0` workflow, including:
  - small-`tau` instability diagnostics in JSON,
  - teacher/student price-domain payoff overlays.

### Stage 8: Namespaced outputs

- `exp_09` table artifacts are written to:
  - `results/tables/u0_sensitivity/`
- `exp_10` table artifacts are written to:
  - `results/tables/tau_sensitivity/`
- Backward-compatible mirror files are also written to:
  - `results/tables/u0_sensitivity_*.{csv,json}`
  - `results/tables/tau_sensitivity_*.{csv,json}`

### Stage 9: Regression safeguards

- File: `experiments/exp_42_competition_sensitivity_regression_safeguards.py`
- Checks implemented:
  1. Smoke run of `M` threshold workflow.
  2. Legacy threshold-column existence check.
  3. Shared interior-classifier consistency check.
  4. Verify `exp_09`/`exp_10` do not overwrite canonical `M` outputs.
- Report output:
  - `results/tables/competition_stage9_regression_report.json`

## How To Run

1. `u0` sensitivity:

```bash
python experiments/exp_40_competition_sensitivity_u0.py
```

1. `tau` sensitivity:

```bash
python experiments/exp_41_competition_sensitivity_tau.py
```

1. Stage 9 safeguards:

```bash
python experiments/exp_42_competition_sensitivity_regression_safeguards.py
```

## Current Completion Status

- Stage 1-6 and Stage 8-9 are implemented and runnable.
- Regression report currently indicates all checks passed (`all_passed: true`).
