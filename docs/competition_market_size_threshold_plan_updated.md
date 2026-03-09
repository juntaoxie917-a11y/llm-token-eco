# Critical Market-Size Threshold Experiment — Direct Implementation Instructions for Copilot

This document is a **direct execution plan for Copilot**.

Its purpose is to add a new experiment that studies the **critical downstream market-size threshold** for the competition model while preserving the existing project framework.

The user has already implemented the competition model and obtained the expected figures. The next task is **not** to redesign the solver stack. The next task is to build a **new threshold-search experiment** on top of the existing competition pipeline.

The economic question is:

- hold all parameters fixed except downstream market size,
- vary market size over a chosen range,
- classify whether an **interior equilibrium** exists,
- identify the **critical threshold** such that below the threshold the interior equilibrium exists.

Important: the user may refer to this parameter as `m` in prose, but in the theory it may appear as `M`. **Follow the real project’s actual parameter name and config style. Do not force a renaming if the project already uses one convention.**

---

## Global rules

Apply these rules throughout the implementation.

1. **Do not rewrite the existing competition model.**
   Reuse the current competition simulation runner and solver stack as much as possible.

2. **Treat threshold search as a new experiment layer.**
   Add it in parallel to the existing experiment workflow.

3. **Do not hard-code repository structure.**
   Place files where the real project structure suggests, but preserve separation of responsibilities.

4. **Do not hide the equilibrium-existence criterion inside ad hoc plotting code.**
   Implement it as an explicit reusable helper.

5. **All classification results must be diagnostic-rich.**
   For each tested market size, record not only the boolean classification, but also the reasons.

6. **Do not assume monotonicity without checking it.**
   The intended economic pattern is that interior equilibrium exists for sufficiently small market size, but the program must verify this numerically instead of silently assuming it.

7. **Use the existing project workflow.**
   Keep the threshold analysis compatible with the current pattern:
   `config -> build model -> run simulation -> save outputs -> plot results`.

---

## What to build

Build a new experiment pipeline with four layers:

1. **equilibrium interiority criterion helper**
2. **single-market-size evaluation wrapper**
3. **market-size sweep / threshold search runner**
4. **outputs: tables, diagnostics, plots, summary**

The implementation should be compatible with the existing project framework.

Do not bypass the current competition runner if it already returns the needed equilibrium objects and diagnostics.

---

## Stage 1 — Add a reusable interior-equilibrium classification helper

### Your task
Create a helper that takes the result of one full competition run and decides whether it counts as an **interior equilibrium**.

### Core idea
Do **not** classify interiority using only one condition.

The classification must respect the model’s nested structure. A market-size point should be considered interior only if interiority holds at **three layers**:

1. **teacher outer optimum is interior**
2. **student best response is interior**
3. **downstream pricing subgame equilibrium is interior**

The helper should therefore compute both:

- an **overall boolean classification**, and
- a **layer-by-layer diagnostic breakdown**.

---

### Required three-layer definition

#### A. Teacher-layer interiority
Treat the equilibrium as teacher-interior only if the teacher’s optimal outer price is not on or too close to the outer price-grid boundary.

Required checks:

- the outer optimum index is not the first grid point,
- the outer optimum index is not the last grid point,
- or equivalently the optimal teacher price is farther than a tolerance from the lower and upper price bounds.

Recommended fields:

- `teacher_interior`
- `teacher_price_at_lower_boundary`
- `teacher_price_at_upper_boundary`
- `distance_to_price_lower_bound`
- `distance_to_price_upper_bound`

#### B. Student-layer interiority
Treat the equilibrium as student-interior only if the student’s best-response training choice is not on or too close to the training-choice boundary.

Required checks:

- the student solver reports success,
- `D_star` is not at the lower training bound,
- `D_star` is not at the upper training bound,
- if the project already reports `is_boundary`, reuse it directly.

Recommended fields:

- `student_interior`
- `student_solver_converged`
- `student_D_at_lower_boundary`
- `student_D_at_upper_boundary`
- `distance_to_D_lower_bound`
- `distance_to_D_upper_bound`

Optional but recommended:

- estimate a local numerical derivative of the student objective at `D_star`,
- store a first-order-condition residual such as `student_foc_residual`,
- use this only as a diagnostic confidence measure unless the project already relies on it.

#### C. Downstream-layer interiority
Treat the equilibrium as downstream-interior only if the downstream pricing subgame at the equilibrium point converges to a strictly interior equilibrium.

Required checks:

- the downstream solver reports success,
- no fallback / clipping / forced approximation was used at the equilibrium point,
- downstream prices are not on or too close to their solver bounds if such bounds exist,
- all equilibrium shares are strictly positive above tolerance.

Required share logic:

- `s_T > share_tol`
- `s_S > share_tol`
- `s_0 > share_tol`

Recommended fields:

- `downstream_interior`
- `downstream_solver_converged`
- `downstream_used_fallback`
- `downstream_price_at_boundary`
- `share_teacher_positive`
- `share_student_positive`
- `share_outside_positive`
- `min_share`
- `downstream_residual` if available

---

### Overall classification rule

The helper must return:

- `overall_interior_strict`
- optionally `overall_interior_weak`

#### Strict version
Classify a point as interior only if **all three layers** are interior:

- teacher layer interior,
- student layer interior,
- downstream layer interior,
- no solver failure at the equilibrium point,
- no fallback/clipping contamination at the equilibrium point,
- all relevant tolerances passed.

#### Weak version (optional but useful)
Provide a weaker diagnostic classification for exploratory scans.

A weak classification may ignore some soft numerical concerns, for example:

- allows larger residuals,
- focuses mainly on no boundary solution and successful solver convergence,
- still rejects obvious corner or failed solutions.

Use the weak version for exploratory plots if desired, but use the strict version for threshold reporting.

---

### Tolerance design

Do not use exact equality for boundary detection.

Add explicit tolerances, ideally in a threshold-analysis settings block or equivalent config style used by the real project.

Recommended tolerances:

- `price_boundary_tol`
- `d_boundary_tol`
- `downstream_price_boundary_tol`
- `share_tol`
- `solver_residual_tol`
- optional `student_foc_tol`

If the current project already has boundary and solver tolerances, reuse them rather than inventing a conflicting parallel system.

---

### Required output structure

Return a structured object or dict, not just a boolean.

Recommended schema:

```python
InteriorClassification(
    overall_interior_strict: bool,
    overall_interior_weak: bool | None,

    teacher_interior: bool,
    student_interior: bool,
    downstream_interior: bool,

    teacher_reason: str | None,
    student_reason: str | None,
    downstream_reason: str | None,

    p_star: float | None,
    D_star: float | None,
    P_T_star: float | None,
    P_S_star: float | None,
    s_T: float | None,
    s_S: float | None,
    s_0: float | None,

    teacher_solver_ok: bool,
    student_solver_ok: bool,
    downstream_solver_ok: bool,

    used_fallback: bool,
    min_share: float | None,
    price_distance_to_boundary: float | None,
    demand_distance_to_boundary: float | None,

    reasons: list[str],
)
```

The exact type system can follow the project’s style. Dataclass, typed dict, or plain dict are all acceptable.

---

### Required failure reasons

Do not collapse all failures into one generic string.

At minimum, distinguish these cases:

- `teacher_price_at_lower_boundary`
- `teacher_price_at_upper_boundary`
- `student_D_at_lower_boundary`
- `student_D_at_upper_boundary`
- `student_solver_failed`
- `downstream_solver_failed`
- `downstream_price_at_boundary`
- `downstream_share_too_small`
- `downstream_used_fallback`
- `residual_too_large`

These reason codes are important because the threshold experiment must later explain **why** interiority fails as market size increases.

---

### Deliverables
Implement:

- `classify_interior_equilibrium(result, threshold_settings)` or equivalent,
- a structured classification object or dict,
- clear layer-level and overall reason fields,
- strict classification and optionally weak classification.

### Do not do

- Do not start threshold search yet.
- Do not put this logic directly inside plotting code.
- Do not assume one boundary flag is enough to diagnose interiority.
- Do not silently treat fallback or clipping as a clean interior equilibrium.

### Completion criteria
This stage is complete only if:

- the classification logic is reusable,
- it explains **why** a point is classified as interior or not,
- it separates teacher/student/downstream causes,
- it works from one existing competition run result without changing the main solver logic.

### Prompt template for this stage
> Add a reusable helper that classifies whether one competition-model equilibrium is interior. Reuse existing run results and diagnostics; do not rewrite the solver. The helper must implement a three-layer criterion: teacher outer price interiority, student training-choice interiority, and downstream pricing-equilibrium interiority. It must check solver convergence, boundary proximity, fallback/clipping flags, and strictly positive shares above tolerance, and it must return structured reason codes rather than a single boolean.

---

## Stage 2 — Add a single-market-size evaluation wrapper

### Your task
Create a wrapper that evaluates the model for **one chosen market-size value** and returns a complete diagnostic record.

### Required behavior
This wrapper should:

1. start from a base config,
2. replace only the downstream market-size parameter,
3. run the existing competition experiment pipeline,
4. classify interior equilibrium using the Stage 1 helper,
5. return a structured row-like result.

### Minimum returned fields
For each tested market size, record at least:

- market size value,
- strict interior classification boolean,
- weak interior classification if implemented,
- classification reasons,
- teacher-layer / student-layer / downstream-layer flags,
- equilibrium teacher token price,
- equilibrium student demand,
- equilibrium downstream prices if available,
- equilibrium shares if available,
- teacher payoff at the optimum,
- student payoff at the optimum if available,
- boundary flags,
- solver convergence flags,
- fallback / clipping flags,
- optional summary statistics about the price-profit curve.

### Design constraint
The wrapper should **reuse the current competition runner as a black box** whenever possible.

Do not duplicate the whole simulation loop unless the current project structure truly forces it.

### Deliverables
Implement:

- `evaluate_market_size_once(...)` or equivalent,
- a result object / dict suitable for later conversion into a table,
- a minimal smoke test for two or three market-size values.

### Do not do

- Do not search for the threshold yet.
- Do not assume the first two or three points already prove monotonicity.

### Completion criteria
This stage is complete only if:

- changing only market size yields a complete experiment result,
- the wrapper returns a structured diagnostic row,
- the existing competition pipeline remains unchanged for normal experiments.

### Prompt template for this stage
> Implement a wrapper that evaluates the existing competition model at one chosen market-size value. It should copy a base config, replace only the market-size parameter, run the existing competition experiment, classify interior equilibrium using the reusable helper, and return a structured diagnostic row. Reuse the current simulation pipeline as a black box where possible.

---

## Stage 3 — Implement a coarse market-size sweep

### Your task
Build a sweep runner that evaluates the model over a grid of market-size values.

### Required behavior
The sweep runner should:

1. accept a base config and a market-size grid,
2. evaluate each point using the Stage 2 wrapper,
3. save a tidy table of per-point results,
4. compute whether the classification pattern appears monotone,
5. identify candidate transition intervals where the strict interior classification changes.

### Required output columns
The sweep table should include at least:

- market size,
- `overall_interior_strict`,
- `overall_interior_weak` if implemented,
- classification reasons,
- teacher-layer / student-layer / downstream-layer flags,
- equilibrium price,
- equilibrium demand,
- equilibrium downstream prices if available,
- equilibrium shares if available,
- teacher payoff,
- boundary flags,
- convergence flags,
- fallback flags,
- optional `distance_to_price_boundary`,
- optional `distance_to_demand_boundary`,
- optional `min_share`.

### Monotonicity check
Add a helper such as:

- `check_threshold_pattern(rows)`

This helper should determine whether the pattern is consistent with a **single threshold**. For example:

- all small market sizes classified interior,
- all large market sizes classified non-interior,
- at most one transition interval.

If the pattern is not cleanly monotone, do not pretend a unique threshold exists. Record that explicitly.

Also record whether failures near the transition are caused by:

- teacher boundary,
- student boundary,
- downstream failure,
- or mixed causes.

### Deliverables
Implement:

- `run_market_size_sweep(...)`,
- a sweep results table,
- a monotonicity / transition summary,
- a basic figure showing `overall_interior_strict` against market size or equivalent.

### Do not do

- Do not jump directly to bisection without a coarse scan.
- Do not report a threshold if the coarse scan already shows non-monotone behavior.

### Completion criteria
This stage is complete only if:

- the code can scan a user-specified market-size grid,
- it saves a tidy result table,
- it reports whether a threshold interpretation is numerically plausible.

### Prompt template for this stage
> Implement a coarse market-size sweep for the competition model. Reuse the single-point evaluation wrapper, save a tidy per-market-size table, and add a monotonicity check that reports whether the interior-equilibrium classification is consistent with a single threshold. Use the strict classification for threshold logic, and do not assume monotonicity unless the computed results support it.

---

## Stage 4 — Implement transition-interval refinement and threshold estimation

### Your task
Once a coarse scan identifies a clean transition interval, refine it numerically to estimate the critical market-size threshold.

### Required behavior
Implement a refinement procedure that:

1. starts from the transition interval found in Stage 3,
2. repeatedly evaluates intermediate market-size values,
3. narrows the interval containing the change in **strict** classification,
4. stops when the interval width is below a user-defined tolerance or when the maximum number of refinement steps is reached.

### Recommended method
Use a conservative interval-refinement approach such as:

- binary search / bisection on the strict classification predicate,
- or midpoint refinement on the detected interval.

This is appropriate only if Stage 3 indicates a clean threshold pattern.

### Threshold output
Return a threshold summary containing at least:

- lower bound for the threshold,
- upper bound for the threshold,
- midpoint estimate,
- interval width,
- number of refinement steps,
- whether the threshold estimate is trustworthy,
- explanation if refinement was skipped.

### Design constraint
Do not treat the threshold as an exact closed-form number. It is a **numerically estimated interval** unless the project already has a more precise notion.

### Deliverables
Implement:

- `refine_market_size_threshold(...)`,
- a structured threshold summary object,
- optional history table of refinement evaluations.

### Do not do

- Do not run refinement if the coarse scan showed multiple transitions or non-monotone classification.
- Do not output a fake scalar threshold without interval information.

### Completion criteria
This stage is complete only if:

- the code can refine a candidate transition interval,
- the threshold result is reported as an interval plus midpoint estimate,
- the code refuses to overclaim when the numerical pattern is not clean.

### Prompt template for this stage
> Implement a refinement step for the critical market-size threshold. Starting from a transition interval detected in the coarse sweep, repeatedly evaluate intermediate market-size values and narrow the interval where the strict interior-equilibrium classification changes. Return lower/upper bounds, midpoint estimate, interval width, diagnostics, and a history table. Only run refinement if the coarse sweep supports a single-threshold interpretation.

---

## Stage 5 — Add dedicated outputs and plots for the threshold experiment

### Your task
Make the threshold experiment produce reusable outputs in the same spirit as the existing project.

### Required outputs
Save at least the following artifacts.

1. **Coarse sweep table**
   One row per tested market size.

2. **Threshold summary**
   A compact JSON / YAML / markdown summary of the estimated threshold interval and diagnostics.

3. **Refinement history table**
   Only if refinement is actually run.

4. **Plots**
   Add dedicated plotting functions for at least:
   - strict interior classification vs market size,
   - optional weak classification vs market size,
   - equilibrium teacher price vs market size,
   - equilibrium student demand vs market size,
   - optional distance-to-boundary diagnostics,
   - optional teacher payoff at optimum vs market size,
   - optional min-share vs market size.

### Plotting rule
Plotting must remain separate from the solver and sweep logic.

Do not entangle figure generation with the numerical routines.

### Deliverables
Implement:

- save helpers or experiment output wiring,
- threshold-specific plotting helpers,
- one runnable experiment entry point.

### Do not do

- Do not only print results to stdout.
- Do not make plots depend on hidden in-memory state if a saved table already exists.

### Completion criteria
This stage is complete only if:

- the threshold experiment can be rerun and reproduced from config,
- tables and figures are saved cleanly,
- plotting functions can read saved data or structured results without re-solving the model.

### Prompt template for this stage
> Add outputs and plotting for the market-size threshold experiment. Save a coarse sweep table, threshold summary, optional refinement history, and dedicated plots such as strict interior classification vs market size, equilibrium price vs market size, equilibrium demand vs market size, and diagnostic distance-to-boundary or min-share plots. Keep plotting separate from the numerical routines.

---

## Stage 6 — Add a single runnable experiment script and config support

### Your task
Expose the threshold analysis as a standalone experiment in the project’s normal workflow.

### Required behavior
Add one dedicated experiment entry point, for example analogous to the current experiment scripts.

It should:

1. load the base config,
2. read threshold-analysis settings,
3. run the coarse sweep,
4. decide whether refinement should run,
5. save tables / summaries / plots,
6. print or log a compact final summary.

### Config support
Add a threshold-analysis config block if the project style supports it. It should allow the user to specify:

- market-size sweep range,
- number of coarse points,
- refinement tolerance,
- maximum refinement iterations,
- strict/weak classification toggle if desired,
- interiority tolerances,
- output toggles.

If the real project prefers CLI arguments over config blocks, follow the real project style instead.

### Deliverables
Implement:

- one threshold experiment entry script,
- config fields or equivalent settings wiring,
- a final compact run summary.

### Do not do

- Do not force a config redesign if the current project does not need it.
- Do not bury threshold settings inside unrelated competition config fields.

### Completion criteria
This stage is complete only if:

- the threshold experiment can be launched like the project’s other experiments,
- all necessary controls are configurable,
- normal competition experiments still run unchanged.

### Prompt template for this stage
> Expose the critical market-size analysis as a standalone experiment. Add an experiment entry point and config support for sweep range, number of points, refinement tolerance, maximum refinement steps, strict/weak classification settings if needed, and interiority tolerances. Reuse the existing project workflow instead of redesigning it.

---

## Required diagnostic logic

The threshold experiment must not collapse everything into a single boolean without explanation.

For each market-size value, record enough information to answer:

- Did the model solve cleanly?
- Was the equilibrium truly interior or just numerically near a boundary?
- Which layer failed if it was non-interior?
- Is the non-interior outcome due to economics or due to solver failure?

At minimum, preserve separate flags for:

- teacher-layer boundary issue,
- student-layer boundary issue,
- downstream solver convergence issue,
- downstream price boundary issue if applicable,
- downstream share positivity issue,
- fallback / clipping issue,
- final overall strict interior classification,
- optional overall weak interior classification.

---

## Recommended numerical logic

Use the following numerical strategy.

1. **Coarse scan first**
   Always start with a coarse market-size sweep.

2. **Check pattern before refining**
   Only refine if the classification looks like a single threshold.

3. **Use strict classification for threshold estimation**
   Weak classification can be saved for exploratory diagnosis, but the reported threshold should be based on strict interiority.

4. **Refine conservatively**
   Narrow the transition interval until tolerance is met.

5. **Report an interval, not fake precision**
   The critical market size should be reported as a bracket plus midpoint estimate.

6. **Fail honestly**
   If the pattern is non-monotone or solver failures dominate near the transition, report that no trustworthy threshold estimate was obtained.

---

## Suggested file responsibilities

Do not treat these names as mandatory. Match the real project structure.

A good responsibility split is:

- **threshold logic module**
  - interior classification helper
  - single-point evaluation wrapper
  - sweep runner
  - refinement routine
  - threshold summary logic

- **visualization module or threshold plotting module**
  - classification vs market size
  - equilibrium objects vs market size
  - diagnostic plots

- **experiment entry script**
  - load config
  - call threshold runner
  - save outputs

If the real project already has a competition analysis module, integrate there instead of creating a redundant parallel tree.

---

## Minimum final deliverables

When the implementation is complete, the project should be able to do the following:

1. run the competition model at one chosen market-size value,
2. classify whether the resulting equilibrium is interior using a three-layer criterion,
3. sweep market size over a grid,
4. determine whether a threshold interpretation is numerically supported,
5. refine the threshold interval if appropriate,
6. save tables, diagnostics, summaries, and plots,
7. leave the existing competition workflow intact.

---

## One-shot prompt for Copilot

Use or adapt the following prompt if you want Copilot to implement the whole feature in controlled steps:

> Add a new threshold-analysis experiment for the competition model that studies the critical downstream market-size parameter at which interior equilibrium ceases to exist. Reuse the current competition simulation pipeline instead of rewriting it. Implement: (1) a reusable interior-equilibrium classification helper with structured reasons, using a three-layer criterion for teacher outer price interiority, student training-choice interiority, and downstream pricing-equilibrium interiority; (2) a single-point evaluation wrapper that replaces only the market-size parameter and runs the existing competition experiment; (3) a coarse market-size sweep that saves a tidy table and checks whether the classification pattern is consistent with a single threshold; (4) an interval-refinement routine that estimates the threshold as a bracket plus midpoint when the coarse scan supports monotonicity; and (5) dedicated outputs, plots, and one runnable experiment entry point. Keep config, solver logic, sweep logic, saving, and plotting separated by responsibility. Do not hard-code repository structure, do not redesign the solver stack, and do not report a fake scalar threshold when the numerical pattern is non-monotone or solver diagnostics are unreliable.
