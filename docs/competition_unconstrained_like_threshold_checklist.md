# Competition Interior-Equilibrium Robust Threshold Checklist

## 0. Goal and Core Claim

### Goal
Identify whether each parameter has a region where strict non-interior remains true even when `p_max` increases, i.e. non-interior is no longer caused by upstream-price cap truncation.

### Parameters
- downstream market size `M`
- outside-option utility `u0`
- price sensitivity `tau`

### Core numerical claim (simulation-level)
For parameter value `theta`, define it as `p_max`-insensitive non-interior if all conditions hold on the largest `K` tested `p_max` values:
1. `interior_equilibrium == False`
2. `teacher_reason != teacher_price_at_upper_boundary`
3. `teacher_price_at_upper_boundary == False`

Recommended `K = 2` for quick run, `K = 3` for report-quality run.

---

## 1. Freeze Baseline and Keep Existing Outputs

1. Keep all existing outputs unchanged (already generated files remain as baseline references).
2. Run new analysis in isolated output prefix only:
- `results/tables/unconstrained_like/`
- `results/figures/unconstrained_like/`
- `results/logs/`
3. Do not overwrite:
- `results/tables/competition_threshold_*`
- `results/tables/u0_sensitivity_*`
- `results/tables/tau_sensitivity_*`

Exit criterion:
- New experiment writes only under `unconstrained_like` paths.

---

## 2. Build the Two-Stage Experiment Matrix

## Stage A: coarse scan (low cost)
1. `p_max_grid = [50, 80, 120, 200]`
2. Parameter grids:
- `M`: existing threshold grid (`competition.threshold_analysis`)
- `u0`: existing `u0_sweep.grid`
- `tau`: existing `tau_sweep.grid`
3. Keep `p_points` unchanged (current default 400).

Expected output:
- one long table with columns at least:
  - `parameter`, `parameter_value`, `p_max`, `interior_equilibrium`,
  - `teacher_reason`, `teacher_price_at_upper_boundary`,
  - `student_reason`, `downstream_reason`

## Stage B: adaptive refinement (only unresolved regions)
1. Mark unresolved points from Stage A:
- still boundary-limited at the largest tested `p_max`
- or classification flips across adjacent `p_max` values
2. Expand only unresolved points with:
- `p_max` extension: add `300`, then `500` if still unresolved
- local parameter refinement: add midpoint(s) only around transition neighborhood
3. Do not globally densify all parameter values.

Exit criterion:
- unresolved share < 5% OR max `p_max` reaches 500 with stable labels.

---

## 3. Parameter-Specific Search Direction

## 3.1 `M` (market size)
1. Use current range first.
2. If all interior at higher `p_max`, extend upper range on log scale (example: 50k -> 80k -> 120k -> 200k).
3. Refine only around first robust non-interior appearance.

## 3.2 `u0`
1. Extend toward more negative values first (example: `-2.0` down to `-4.0` with coarse step).
2. Keep positive side sparse unless transition appears.
3. Refine around first robust non-interior point.

## 3.3 `tau`
1. Expand both tails, but prioritize small-`tau` side first.
2. Use coarse tail points (example: `0.1`, `0.15`, `2.5`, `3.0`) before dense scans.
3. Refine only where robust label changes.

---

## 4. Labeling and Decision Rules

For each `(parameter_value, p_max)` point, create labels:
1. `interior`
2. `non_interior_bound_limited`:
- non-interior and `teacher_price_at_upper_boundary == True`
3. `non_interior_pmax_insensitive`:
- non-interior and not upper-boundary limited

For each `parameter_value`, aggregate across `p_max` (in ascending order):
1. If largest `K` p_max all `non_interior_pmax_insensitive` -> mark `robust_non_interior = True`
2. If largest `p_max` still `non_interior_bound_limited` -> mark `still_bound_limited = True`
3. Else -> `unresolved`

Reportable threshold interval per parameter:
- first value where `robust_non_interior` becomes true,
- with previous value as lower bracket.

---

## 5. Stopping Criteria (must satisfy all for final stop)

1. Threshold bracket stability:
- endpoint change < 5% in two consecutive refinement rounds.
2. Boundary artifact control:
- upper-boundary-hit share change < 2 percentage points between last two rounds.
3. Resolution completeness:
- unresolved points <= 5% of tested points.
4. Budget cap:
- if runtime exceeds cap, stop and report unresolved region explicitly.

---

## 6. Minimum Deliverables for Final Report

1. `CSV`: long-form panel table
- `parameter`, `parameter_value`, `p_max`, classification fields, reasons.
2. `CSV`: aggregated threshold summary
- one row per parameter containing:
  - robust threshold bracket,
  - unresolved interval,
  - max tested `p_max`.
3. `JSON`: run metadata
- grids, runtime, stop reason.
4. Figures:
- per parameter: heatmap or tile plot of label over (`parameter_value`, `p_max`)
- optional: endpoint-vs-`p_max` stability line.

---

## 7. Practical Run Sequence (recommended)

1. Run Stage A once for all three parameters.
2. Inspect unresolved points and boundary-limited points.
3. Run Stage B only for unresolved neighborhoods.
4. Recompute aggregated labels and threshold brackets.
5. Apply stopping criteria.
6. Export final summary and figures.

---

## 8. Compute Budget Guidance

1. Use full `p_points=400` only for final refinement rounds.
2. For Stage A coarse scan, optionally use reduced `p_points` in a dedicated override run if needed.
3. Cache reuse should remain enabled in downstream loops.
4. Never expand both parameter grid and `p_max` grid globally at once.

---

## 9. Common Failure Modes and How to Handle

1. False stability due to boundary misclassification:
- verify effective `p_grid_override` is used for boundary checks.
2. Apparent robust non-interior caused by solver failure:
- check `teacher_solver_ok`, `student_solver_ok`, `downstream_solver_ok` before labeling robust.
3. Overfitting to one coarse grid:
- enforce at least one local refinement around each detected transition.

---

## 10. Final Interpretation Template

For each parameter, conclude with:
1. whether robust (`p_max`-insensitive) non-interior region exists,
2. estimated threshold bracket,
3. unresolved range (if any),
4. whether current evidence is sufficient for publication-level claim.

Suggested language:
- "Under tested `p_max` up to X, parameter Y shows robust non-interior beyond bracket [a, b], with no upper-boundary limitation in the final K levels."
- "For parameter Y, non-interior remains bound-limited up to `p_max = X`; robust threshold is not established yet."

---

## 11. Executable Task Table

Use this section as the direct runbook. Execute tasks in order.

| Task ID | Purpose | Command / Action | Expected Outputs | Chart Checkpoint |
|---|---|---|---|---|
| T1 | Preserve baseline outputs and create isolated output folders | `mkdir -p results/tables/unconstrained_like results/figures/unconstrained_like results/logs` | New unconstrained-like folders exist | N/A |
| T2 | Run coarse unconstrained-like stability panel (`p_max = 50,80,120,200`) | `python -u experiments/exp_12_competition_unconstrained_like_stability.py` | `results/tables/unconstrained_like/competition_unconstrained_like_stability_summary.csv`; `results/tables/unconstrained_like/competition_unconstrained_like_stability_summary.json`; `results/logs/exp_12_competition_unconstrained_like_stability_run_log.json` | N/A |
| T3 | Export long-form panel for plotting | `python -u experiments/exp_12_competition_unconstrained_like_stability.py` (already includes panel export) | `results/tables/unconstrained_like/competition_unconstrained_like_panel.csv`; `results/tables/unconstrained_like/competition_unconstrained_like_unresolved_points.csv` | N/A |
| T4 | Plot label heatmap for `M` | `python -u experiments/exp_13_competition_unconstrained_like_plots.py` | `results/figures/unconstrained_like/fig_unconstrained_M_label_heatmap.(png/pdf/svg)` | Verify high-`p_max` rows do not show artificial upper-boundary artifacts if classification is stable |
| T5 | Plot label heatmap for `u0` | `python -u experiments/exp_13_competition_unconstrained_like_plots.py` | `results/figures/unconstrained_like/fig_unconstrained_u0_label_heatmap.(png/pdf/svg)` | Verify negative tail behavior and whether high-`p_max` rows switch from bound-limited to robust non-interior |
| T6 | Plot label heatmap for `tau` | `python -u experiments/exp_13_competition_unconstrained_like_plots.py` | `results/figures/unconstrained_like/fig_unconstrained_tau_label_heatmap.(png/pdf/svg)` | Verify low-`tau` and high-`tau` sides separately |
| T7 | Plot threshold-endpoint stability vs `p_max` | `python -u experiments/exp_13_competition_unconstrained_like_plots.py` | `results/figures/unconstrained_like/fig_unconstrained_threshold_endpoint_vs_pmax.(png/pdf/svg)` | Endpoint should flatten if robust |
| T8 | Identify unresolved neighborhoods | Action: mark points that still flip labels across adjacent `p_max` or remain bound-limited at largest tested `p_max` | `results/tables/unconstrained_like/competition_unconstrained_like_unresolved_points.csv` | N/A |
| T9 | Adaptive refinement round 1 | Action: for unresolved points only, extend `p_max` with `300`; add local parameter midpoints around transitions | Updated panel and summary tables | Re-check T4-T7 figures after update |
| T10 | Adaptive refinement round 2 (only if needed) | Action: unresolved points only, extend `p_max` with `500`; local midpoint refinement again | Updated panel and summary tables | Re-check T4-T7 figures after update |
| T11 | Apply stopping criteria | Action: check endpoint change, boundary-hit share change, unresolved ratio against Section 5 thresholds | `results/tables/unconstrained_like/competition_unconstrained_like_stop_check.json` | N/A |
| T12 | Final report pack | Action: freeze final CSV/JSON and figure set; include interpretation text template from Section 10 | Final deliverables under `results/tables/unconstrained_like` and `results/figures/unconstrained_like` | Final figure bundle complete |

## 12. Direct Execution Notes

1. Recommended order: T1 -> T2 -> T3 -> T4/T5/T6 -> T7 -> T8 -> T9 -> (T10 if needed) -> T11 -> T12.
2. Do not run global dense expansions for all parameters at once.
3. For each refinement round, only update unresolved neighborhoods.
4. Keep figure style aligned with existing competition figures:
- export `pdf`, `svg`, and `png`
- keep axis labels with the same notation style (`M`, `u0`, `tau`, `p_max`)
- keep grid/legend style consistent with existing plotting helpers.

## 13. Minimal Figure Specification (for implementation)

1. Label palette (fixed across all three heatmaps):
- `interior` -> green
- `non_interior_bound_limited` -> orange
- `non_interior_pmax_insensitive` -> red
- `unresolved` -> gray
2. Heatmap axis convention:
- x-axis: parameter value
- y-axis: `p_max`
3. Include a compact legend with exact label names used in tables.
4. Save naming convention:
- `fig_unconstrained_<parameter>_label_heatmap`
- `fig_unconstrained_threshold_endpoint_vs_pmax`
