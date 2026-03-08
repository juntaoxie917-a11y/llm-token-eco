# Competition Stage 0 Audit

This file records Stage 0 deliverables from `docs/competition_copilot_execution_plan.md`.

## 1) Module Map (Reuse / Wrap / Add / Do Not Reuse Directly)

| Module | Classification | Why |
|---|---|---|
| `src/config_loader.py` | Reuse + wrap | Keep baseline validation unchanged; add competition-specific validation in a parallel helper/module. |
| `src/normalization.py` | Reuse as-is | Competition theory does not change anchor normalization formulas. |
| `src/scaling_laws.py` | Reuse as-is | Keep `L_student`, `L_tilde`, `gap_term` unchanged as shared technology block. |
| `src/model.py` | Reuse partially, do not reuse reduced-form objective directly | Baseline value/profit and student BR are reduced-form for non-competition setting; keep file unchanged and mirror interface in competition module. |
| `src/simulation.py` | Do not reuse directly as final competition runner | Hard outside-option baseline logic differs from downstream-logit outside option in competition model. Use as structural template only. |
| `src/simulation_soft.py` | Do not reuse directly as final competition runner | Soft participation wrapper is not the same object as downstream market outside option in competition model. |
| `src/visualization.py` | Reuse style/patterns, add parallel competition plots later | Keep plotting separated from solving; add competition-specific plotting functions in new step. |
| `experiments/exp_01_hard_outside.py` | Reuse as pattern only | Keep baseline entry script unchanged; create a separate competition experiment entry. |
| `experiments/exp_02_soft_outside.py` | Reuse as pattern only | Same as above. |

## 2) Extension Notes / TODO Markers

- TODO (Stage 1): Add competition static helpers module with:
  - competition parameter dataclass(es),
  - loss-to-quality mapping helper,
  - stable logit utility/share helpers,
  - downstream profit helpers.
- TODO (Stage 2): Add standalone downstream pricing-subgame solver at fixed `D`.
- TODO (Stage 3): Add competition-aware student BR solver that nests Stage 2.
- TODO (Stage 4): Add competition simulation runner with grid over upstream `p`.
- TODO (Stage 5): Add competition output saver + core plots.
- TODO (Stage 6): Add trustworthiness diagnostics and smoke tests.
- TODO (Stage 7): Add caching/warm-start/fallback optimizations after correctness.

## 3) Baseline Interfaces Competition Code Should Match

Competition modules should mirror baseline interface style where practical.

- Config and params pattern:
  - baseline reference: `build_params_from_config(cfg)` in `src/model.py`
  - competition target: `build_competition_params_from_config(cfg)` (parallel module)
- Structured result dataclasses:
  - baseline reference: `StudentBestResponseResult` and simulation row dataclasses
  - competition target: downstream-equilibrium result and competition BR result dataclasses
- Grid-runner separation:
  - baseline reference: `run_baseline_grid_simulation(...)`
  - competition target: `run_competition_grid_simulation(...)` (later stage)
- Dataframe export helper pattern:
  - baseline reference: `to_dataframe(...)` in simulation modules
  - competition target: same style for competition row objects (later stage)

## Explicit “Do Not Reuse Directly” List

These are valid references for style, but not final economic objects in competition:

- `student_profit(...)` in `src/model.py` (reduced-form baseline objective)
- `solve_student_best_response_direct(...)` in `src/model.py` (does not nest downstream game)
- hard/soft outside-option realization logic in `src/simulation.py` and `src/simulation_soft.py`
- `teacher_profit(...)` in `src/model.py` as full teacher payoff (competition needs upstream + downstream)

## Stage 0 Completion Check

- Baseline modules unchanged.
- Extension boundaries documented.
- Reuse / wrap / add / do-not-reuse-directly classification completed.
