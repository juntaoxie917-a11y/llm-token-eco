# competition_sensitivity_u0_tau_copilot_plan_v2

## Purpose

Implement **sensitivity analysis for `u_0` and `tau`** on top of the existing **competition model**.

This document is written **directly for Copilot**. Follow it as an execution manual.

The goal is to add new experiment workflows that vary:

- `u_0`: outside-option baseline utility
- `tau`: downstream price sensitivity in logit demand

while **preserving all existing competition functionality**, especially the already completed **market-size (`M`) threshold / critical-point analysis**.

---

## Critical non-negotiable rules

### Rule 1 — Do not touch the existing `M` analysis

**Do not modify, break, rename, or silently reinterpret any code, configs, result schemas, plots, or experiment entry points that already implement the market-size (`M`) analysis, including the logic used to identify whether an interior Nash equilibrium exists below a critical market size.**

That functionality has already been implemented and validated. Treat it as frozen.

Therefore:

- do **not** rewrite the existing competition solver just to make sensitivity analysis "cleaner"
- do **not** change existing result column names used by the `M` analysis
- do **not** change existing figures produced for the `M` threshold experiment
- do **not** move files only for stylistic reasons
- do **not** merge this work into the old experiment script in a way that risks regressions

If additional hooks are needed, add them **in parallel** and keep backward compatibility.

### Rule 2 — `u_0` and `tau` must be analyzed separately

**Do not combine `u_0` sensitivity and `tau` sensitivity into one primary experiment.**

Implement them as **two separate 1D sensitivity workflows**:

- one workflow that varies `u_0` only
- one workflow that varies `tau` only

Keep their configs, tables, diagnostics, summaries, and figures separate.

A 2D `(u_0, tau)` sweep is optional and should only be added later if the 1D workflows are already stable.

### Rule 3 — Always include student total payoff plots

In every new sensitivity workflow, **explicitly generate student total payoff plots** in addition to teacher total payoff plots.

This is important because earlier plotting work did not always include the student-side total payoff figure. Do not forget it this time.

Whenever you generate:

- teacher total payoff vs token price, or
- teacher total payoff vs the varied parameter,

also generate the corresponding:

- student total payoff vs token price, or
- student total payoff vs the varied parameter.

Student total payoff should be treated as a first-class output, not an optional afterthought.

### Rule 4 — Do not create a new Python virtual environment

**Do not create `venv`, do not create a new virtual environment, and do not try to bootstrap a new Python environment.**

The user is **already inside a configured conda environment**.

Therefore:

- do **not** run `python -m venv ...`
- do **not** create `.venv`
- do **not** add environment setup scripts unless strictly necessary
- do **not** change dependency management just for this task

Assume the required environment is already active. If a package import fails, report the missing package clearly, but do not create a new environment automatically.

---

## Design principle

Use the same general pipeline style as the baseline project:

`config -> model blocks -> simulation runner -> tables/diagnostics -> plots -> experiment entry script`

The baseline codebase already separates:

- YAML config loading and validation
- technology block
- profit / optimization block
- simulation runners
- visualization
- experiment entry scripts

Keep this design principle for the new sensitivity analysis as well. Do not hard-code a fake directory structure; adapt to the real project structure if it differs. The baseline organization shows the intended responsibility split, not a mandatory folder layout. fileciteturn1file11L5-L16 fileciteturn1file2L1-L16

---

## Conceptual scope of this task

You are **not** being asked to redesign the competition model.

You are being asked to add a **comparative-statics experiment layer** over an existing competition implementation.

The competition model already uses downstream logit demand with outside option and price sensitivity:

- outside-option utility enters through `u_0`
- price sensitivity enters through `tau`
- downstream shares depend on both parameters
- teacher and student profits depend on the resulting downstream equilibrium

So this task should be implemented as a structured sweep over parameter values, not as a new model rewrite.

---

## Main implementation objective

Add **two new experiment workflows**:

1. **`u_0` sensitivity**: vary `u_0`, hold the rest fixed
2. **`tau` sensitivity**: vary `tau`, hold the rest fixed

These workflows should answer questions like:

1. How does the equilibrium outcome change when `u_0` varies, holding other parameters fixed?
2. How does the equilibrium outcome change when `tau` varies, holding other parameters fixed?
3. How do `u_0` and `tau` affect the existence of an **interior Nash equilibrium**?
4. How do they shift:
   - optimal upstream price
   - student training demand
   - downstream prices
   - downstream shares
   - teacher total profit
   - student total profit
5. Conditional on the already established `M` threshold logic, how do `u_0` and `tau` change the behavior of the interior-equilibrium region?

Primary requirement:

- implement **one-dimensional sweeps** over `u_0`
- implement **one-dimensional sweeps** over `tau`
- keep them separate

Optional requirement after the two 1D workflows are stable:

- add a **two-dimensional** grid over `(u_0, tau)`

---

## Do not assume file names

The source files in the uploaded baseline project illustrate responsibilities, but they do **not** define the real competition project structure.

Therefore:

- reuse existing competition modules where possible
- add new helpers next to existing competition experiment code if that is the least risky option
- do not force names like `src/competition/...` if the real repo already uses another layout
- do not create a large architectural refactor just to match the baseline sample

---

## Required implementation stages

# Stage 1 — Audit and freeze existing competition experiment interfaces

## Do this

Inspect the current competition implementation and identify:

- the main competition solver entry point
- the current market-size (`M`) threshold experiment entry point
- the result object / DataFrame columns produced by current competition simulations
- the current criterion used to classify whether the equilibrium is interior
- the current plotting functions used in the `M` analysis

Then document these interfaces in comments or a short internal note.

## Do not do this

- do not change existing experiment behavior
- do not rename outputs
- do not merge the new sensitivity analysis into the old `M` script yet

## Deliverable

A stable map of reusable interfaces for:

- competition solver
- competition simulation runner
- existing interior-equilibrium diagnostic
- existing result export functions

## Completion condition

You can call the current competition experiment from code and know exactly:

- what inputs it needs
- what outputs it returns
- how it decides whether an interior equilibrium exists

---

# Stage 2 — Add parameterized config support for `u_0` and `tau`

## Do this

Extend the competition config layer so that `u_0` and `tau` are explicitly configurable from experiment configs.

Requirements:

1. `u_0` must be a named config parameter for the downstream outside option.
2. `tau` must be a named config parameter for downstream price sensitivity.
3. Validation must ensure:
   - `tau > 0`
   - `u_0` is finite
4. Preserve backward compatibility:
   - if old configs already contain these fields, do not break them
   - if the competition code already stores them elsewhere, add a compatibility adapter rather than renaming everything

The baseline sample already validates YAML structure and key numeric bounds centrally; follow that style for competition configs too. fileciteturn1file2L17-L85

## Additional requirement

Define **separate sweep config sections** or equivalent runtime controls for:

- `u_0` sensitivity only
- `tau` sensitivity only

Do not design a single mixed sweep as the default path.

## Do not do this

- do not change the meaning of existing config keys used by the `M` experiment
- do not overload unrelated fields

## Deliverable

Competition configs can cleanly specify baseline values and separate sweep ranges for:

- `u_0`
- `tau`

## Completion condition

A competition run can be launched with different `u_0` / `tau` values without touching code.

---

# Stage 3 — Build non-invasive sensitivity runners

## Do this

Create **two new sensitivity runners** that wrap the existing competition simulation.

These runners must:

1. take a base competition config
2. clone or copy it safely in memory
3. override one target parameter
4. call the existing competition experiment logic
5. collect standardized outputs row by row

Implement at least these two public workflows:

- `run_u0_sensitivity(...)`
- `run_tau_sensitivity(...)`

Only after these are stable, optionally add:

- `run_u0_tau_grid_sensitivity(...)`

Use the same spirit as the baseline grid runners: loop over a grid, run a lower-level solver, collect structured rows, then summarize. The baseline project already uses this style for price-grid simulations, including row-wise diagnostics and summary outputs. fileciteturn1file6L49-L89 fileciteturn1file7L31-L78

## Required row-level outputs

Each row in the sensitivity table should contain at least:

- varied parameter name
- varied parameter value
- `M` used in this run
- equilibrium existence flag
- interior-equilibrium flag
- optimal upstream price `p_star`
- student training level at optimum
- teacher total profit at optimum
- student total profit at optimum
- downstream equilibrium prices, if available
- downstream shares, if available
- solver success / failure flag
- diagnostic message

If the current competition code already returns more fields, keep them.

## Do not do this

- do not rewrite the old solver API
- do not embed sensitivity logic inside the main equilibrium solver
- do not replace the old `M` analysis runner

## Deliverable

A new wrapper layer that performs comparative statics without altering existing competition mechanics.

## Completion condition

You can run a full sweep over either `u_0` or `tau` and get a structured table of results.

---

# Stage 4 — Reuse the existing interior-equilibrium criterion

## Do this

Sensitivity analysis must use the **same equilibrium classification rule** that was already used for the market-size threshold study.

If the current `M` analysis already has a programmatic criterion for whether an interior Nash equilibrium exists, reuse exactly that criterion.

If it exists only implicitly in old code, extract it into a small shared helper without changing its logic.

Examples of reusable diagnostics may include:

- optimum not at the price-grid boundary
- student training not at lower/upper training bound
- downstream prices interior and solver successful
- first-order residuals below tolerance
- no clipping / no failed subgame solve

Use whatever rule is **already implemented in the competition project**. Do not invent a new definition unless no reusable criterion exists.

## Why this matters

The user has already completed and trusted the `M` threshold analysis. Sensitivity results for `u_0` and `tau` must be comparable to that earlier work.

## Do not do this

- do not silently tighten or loosen the old interior-equilibrium standard
- do not redefine the threshold logic just for nicer plots

## Deliverable

A shared and stable equilibrium-classification helper reused by:

- old `M` analysis
- new `u_0` sensitivity analysis
- new `tau` sensitivity analysis

## Completion condition

For the same old `M` calibration, the new wrapper reproduces the old interior-equilibrium classification outcome.

---

# Stage 5 — Implement the dedicated `u_0` sensitivity experiment

## Do this

Implement a dedicated experiment entry point for `u_0` sensitivity.

This experiment should:

1. load the base competition config
2. read a configured grid of `u_0` values
3. hold other parameters fixed unless explicitly stated
4. optionally allow `M` to be fixed at:
   - a user-specified value, or
   - a previously identified critical / near-critical value, if the current project already stores that result
5. run the competition model for each `u_0`
6. save:
   - raw row-level results
   - summary JSON / markdown
   - figures

## Required summary outputs

At minimum compute:

- share of grid points with interior equilibrium
- first / last `u_0` where interior equilibrium appears, if applicable
- monotonicity summaries for `p_star`, profits, or shares when numerically meaningful
- number of failed runs

## Required figures

Produce at least the following:

1. `p_star` vs `u_0`
2. teacher total profit vs `u_0`
3. **student total profit vs `u_0`**
4. student training `D_star` vs `u_0`
5. interior-equilibrium indicator vs `u_0`

If downstream objects are available, also add:

6. downstream shares vs `u_0`
7. downstream prices vs `u_0`

## Additional required price-domain plots

For selected representative `u_0` values, also generate price-domain figures:

- teacher total payoff vs token price `p`
- **student total payoff vs token price `p`**

These should be parallel plots. Do not generate the teacher-side version without the student-side counterpart.

## Deliverable

A runnable experiment for `u_0` sensitivity with saved tables, diagnostics, and figures.

## Completion condition

A user can run one script and obtain a complete `u_0` sensitivity package.

---

# Stage 6 — Implement the dedicated `tau` sensitivity experiment

## Do this

Implement a dedicated experiment entry point for `tau` sensitivity.

This experiment should mirror the `u_0` workflow but vary `tau` instead.

## Required validation

- `tau > 0`
- if a very small `tau` causes numerical instability, keep the run but flag it clearly in diagnostics
- do not hide instability by clipping away failures silently

The baseline soft-outside configuration already uses `tau` as a smoothing / response parameter and validates positivity; competition should apply the same numeric discipline. `soft.yaml` includes `tau` in config, and `simulation_soft.py` enforces positivity before running. fileciteturn1file1L37-L43 fileciteturn1file6L57-L62

## Required figures

Produce at least the following:

1. `p_star` vs `tau`
2. teacher total profit vs `tau`
3. **student total profit vs `tau`**
4. student training `D_star` vs `tau`
5. interior-equilibrium indicator vs `tau`

If downstream objects are available, also add:

6. downstream prices vs `tau`
7. downstream shares vs `tau`

## Additional required price-domain plots

For selected representative `tau` values, also generate price-domain figures:

- teacher total payoff vs token price `p`
- **student total payoff vs token price `p`**

These should be parallel plots. Do not generate the teacher-side version without the student-side counterpart.

## Deliverable

A runnable experiment for `tau` sensitivity with saved tables, diagnostics, and figures.

## Completion condition

A user can run one script and obtain a complete `tau` sensitivity package.

---

# Stage 7 — Optional 2D `(u_0, tau)` grid analysis

## Do this only if the existing competition code is stable enough

Add an optional 2D grid experiment over `(u_0, tau)`.

This should produce a rectangular table with one row per `(u_0, tau)` combination and at least:

- `u_0`
- `tau`
- `M`
- interior-equilibrium flag
- `p_star`
- `D_star`
- teacher total profit
- student total profit
- solver success

## Recommended 2D figures

- heatmap of interior-equilibrium existence
- heatmap of `p_star`
- heatmap of teacher total profit
- heatmap of student total profit
- heatmap of student training

## Do not do this

- do not implement this first
- do not sacrifice 1D experiment stability for a fancy 2D workflow
- do not merge it into the 1D scripts in a way that makes `u_0` / `tau` runs harder to use

## Deliverable

An optional 2D comparative-statics extension.

## Completion condition

The 2D sweep is additive and does not affect any existing 1D experiment or old `M` analysis.

---

# Stage 8 — Save outputs in separate experiment namespaces

## Do this

Save all new sensitivity outputs under **new experiment namespaces** separate from the existing `M` threshold outputs and separate from each other.

Examples of output groups:

### `u_0` sensitivity namespace

- `u0_sensitivity_results.csv`
- `u0_sensitivity_diagnostics.json`
- `u0_sensitivity_summary.json`
- `fig_u0_*.pdf/.svg/.png`

### `tau` sensitivity namespace

- `tau_sensitivity_results.csv`
- `tau_sensitivity_diagnostics.json`
- `tau_sensitivity_summary.json`
- `fig_tau_*.pdf/.svg/.png`

The baseline scripts already separate tables, diagnostics, figures, and logs; follow that style. The experiment scripts save CSV / JSON / figure outputs into distinct result groups, and visualization is separated from simulation logic. fileciteturn1file9L34-L51 fileciteturn1file10L21-L33 fileciteturn1file8L1-L9

## Do not do this

- do not overwrite old `M` threshold tables or figures
- do not reuse the exact same output filenames as the existing market-size experiment
- do not dump `u_0` and `tau` outputs into one shared ambiguous file

## Deliverable

A clean result package for each new sensitivity experiment.

## Completion condition

The old `M` outputs remain untouched after running the new scripts.

---

# Stage 9 — Add explicit regression safeguards

## Do this

Add lightweight safeguards so the new work does not break the old `M` analysis.

Minimum safeguards:

1. a smoke test or validation run for the existing `M` experiment
2. a check that old expected output columns still exist
3. a check that the reused interior-equilibrium classification helper returns the same result as before on a known calibration
4. a check that the new `u_0` and `tau` scripts do not overwrite old output files

If the real project already has tests, integrate there. If not, add a lightweight diagnostic script.

## Deliverable

A minimal regression shield for the already completed market-size work.

## Completion condition

After implementing `u_0` / `tau` sensitivity analysis, the old `M` workflow still runs unchanged.

---

## Suggested output schema

Each 1D sensitivity CSV should have one row per parameter value and include at least:

- `parameter_name`
- `parameter_value`
- `M`
- `equilibrium_exists`
- `interior_equilibrium`
- `p_star`
- `D_star_at_p_star`
- `pi_teacher_star`
- `pi_student_star`
- `P_T_star` (if available)
- `P_S_star` (if available)
- `s_T_star` (if available)
- `s_S_star` (if available)
- `s_0_star` (if available)
- `success`
- `message`

If some fields are not available from the current competition implementation, store `NaN` rather than deleting the column.

---

## Suggested diagnostics content

Each diagnostics JSON should include:

- experiment name
- varied parameter name
- varied parameter grid
- fixed parameters summary
- `M` used
- number of runs
- number of successful runs
- number of failed runs
- number of interior-equilibrium runs
- number of boundary / degenerate runs
- note explaining that the old `M` threshold analysis was not modified
- note explaining that `u_0` and `tau` were implemented as separate sensitivity workflows

---

## Plotting reminder

When adding new visualization helpers, follow the baseline plotting principle:

- plotting functions should read simulation outputs and draw figures
- plotting functions should not run the model themselves
- export publication-friendly files (PDF/SVG and optionally PNG)

The baseline project already separates plotting from simulation and already includes student-payoff plotting helpers; preserve that division of responsibility. `visualization.py` also already contains student-payoff related plotting functions, so use that precedent and do not omit student total payoff in the new competition sensitivity figures. fileciteturn1file8L1-L9 fileciteturn1file8L139-L182 fileciteturn1file8L257-L335

---

## Environment reminder

When running experiment files:

- assume the configured conda environment is already active
- use the current Python interpreter from that environment
- do not create new environment bootstrap code

If imports fail, surface the specific missing dependency instead of creating a new environment automatically.

