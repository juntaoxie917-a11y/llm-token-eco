# Competition Sensitivity Stage 4 Consistency Note

## Goal

Ensure sensitivity workflows reuse the exact interior-equilibrium criterion used by the existing market-size (`M`) threshold workflow.

## Implementation

- `src/competition_sensitivity.py` now evaluates each parameter point via:
  - `src/competition_threshold.py`: `evaluate_market_size_once(...)`
- This guarantees shared classification logic from:
  - `classify_interior_equilibrium(...)`

No alternate interior-classification rule is implemented in the sensitivity module.

## Validation

At baseline calibration (`u0 = base`, `tau = base`, `M = base`), compared:

- `evaluate_market_size_once(...)`
- `run_u0_sensitivity(..., u0_grid=[base_u0])`
- `run_tau_sensitivity(..., tau_grid=[base_tau])`

Observed exact match on:

- `overall_interior_strict`
- `overall_interior_weak`
- `p_star`

Sample check result:

- baseline: strict=True, weak=True, p_star=18.546365914786968
- u0 runner: strict=True, weak=True, p_star=18.546365914786968
- tau runner: strict=True, weak=True, p_star=18.546365914786968

## Stage 4 Completion

- Shared interior criterion is reused by old `M` analysis and both new 1D sensitivity runners.
- Baseline classification consistency check passes.
