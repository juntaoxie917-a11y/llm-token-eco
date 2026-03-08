# Competition Model Implementation Instructions for Copilot

This document is **not** a human-oriented roadmap. It is a **direct execution plan for Copilot**.

Its purpose is to guide code generation for the competition extension in a controlled way. Follow the stages **in order**. Do **not** jump ahead. Do **not** redesign the project architecture unless reuse absolutely requires it.

The baseline project already establishes the main design principles:
- preserve the workflow from configuration to simulation to outputs,
- reuse baseline components where possible,
- add competition-specific logic in parallel instead of rewriting baseline logic,
- avoid hard-coding file locations or repository structure,
- keep numerical solving, simulation orchestration, plotting, and saving separated by responsibility.

## Global rules for all stages

Apply these rules in every stage.

1. **Do not rewrite the baseline pipeline.**
   Reuse it where possible and extend it in parallel.

2. **Do not hard-code repository structure.**
   The provided source files are only a reference for design principles, not the literal project layout.

3. **Do not mix concerns.**
   Keep these layers separate:
   - parameter/config objects,
   - static economic helpers,
   - equilibrium solvers,
   - simulation runners,
   - output saving,
   - plotting,
   - diagnostics.

4. **Do not generate the whole competition model in one step.**
   Complete one stage, then stop.

5. **Every new solver must return structured diagnostics.**
   At minimum include convergence information, boundary/clipping flags, and a short solver message.

6. **Prefer incremental extension over replacement.**
   If a baseline function already provides a reusable pattern, mirror its interface or workflow before inventing a new one.

7. **Preserve the baseline experiment style.**
   The competition extension should still support a grid-based experiment workflow.

---

## Stage 0 — Audit the baseline interface and mark extension boundaries

### Your task
Inspect the baseline code and identify:
- which modules can be reused as-is,
- which modules should be wrapped or mirrored,
- which economic functions should not be reused directly because they are reduced-form baseline objects,
- where competition-specific code should be added in parallel.

### Deliverables
Produce:
1. a concise module map,
2. extension notes or TODO markers,
3. a list of baseline interfaces that competition code should try to match.

### Do not do
- Do not write the competition solver yet.
- Do not rewrite the project structure.
- Do not rename baseline files just to fit a new architecture.

### Completion criteria
This stage is complete only if:
- baseline code still runs unchanged,
- competition extension points are clearly marked,
- there is a clear “reuse / wrap / add new / do not reuse directly” classification.

### Prompt template for this stage
Use or adapt the following prompt:

> Audit the baseline project and identify the extension boundaries for the competition model. Reuse baseline design principles, do not rewrite the architecture, and do not implement solvers yet. Produce a concise module map showing: reusable modules, modules to wrap or mirror, modules that require competition-specific additions, and baseline functions that should not be reused directly as final economic objects.

---

## Stage 1 — Implement static competition building blocks

### Your task
Implement the competition-specific pieces that are purely functional and do not require equilibrium iteration.

### Required components
Implement the following categories:

1. **Competition parameter object(s)**
   Support at least:
   - downstream market size `M`,
   - downstream marginal costs `m_T`, `m_S`,
   - outside-option utility `u0`,
   - logit sensitivity parameter `tau`,
   - teacher quality `q_T`,
   - any student quality mapping settings.

2. **Quality mapping helpers**
   Examples:
   - `student_quality_from_loss(...)`,
   - optional teacher-quality helpers if needed.

3. **Logit utility and share helpers**
   Include:
   - utility construction,
   - stable softmax / log-sum-exp,
   - teacher share,
   - student share,
   - outside-option share,
   - optional validation helper to verify shares sum to one.

4. **Downstream profit helpers**
   Implement functions that compute teacher and student downstream profits conditional on prices, shares, and training choice.

### Deliverables
Produce:
- parameter/config dataclass(es) or equivalent objects,
- helper functions,
- a minimal test or demo showing shares and profits can be computed from manually supplied values.

### Do not do
- Do not solve equilibrium.
- Do not build the full simulation runner.
- Do not mix plotting code into these helpers.

### Completion criteria
This stage is complete only if:
- for manually chosen `D`, `P_T`, and `P_S`, the code returns stable shares and downstream profits,
- shares are nonnegative and sum to one up to numerical tolerance,
- changing student quality changes student share in an economically sensible direction.

### Prompt template for this stage
> Implement the static building blocks for the competition model without solving equilibrium yet. Add competition parameter objects, quality mapping helpers, logit utility/share helpers, and downstream profit helpers. Reuse baseline coding style where sensible. Return a minimal demo showing that manually supplied values of D, P_T, and P_S produce stable shares and profits. Do not implement the full simulation pipeline in this step.

---

## Stage 2 — Implement the downstream pricing-subgame solver as a standalone module

### Your task
Write a standalone solver for the downstream Nash pricing game, conditional on a given student training choice `D`.

### Required behavior
The solver should:
- accept `D` and competition parameters,
- derive the relevant quality objects,
- solve for equilibrium downstream prices,
- return equilibrium prices, shares, downstream profits, and diagnostics.

### Minimum output fields
Return a structured result containing at least:
- `P_T_star`,
- `P_S_star`,
- equilibrium shares,
- teacher downstream profit,
- student downstream profit,
- convergence flag,
- iteration count or function-evaluation count,
- residual norm or equivalent error measure,
- message.

### Solver strategy
Use a conservative first-pass numerical strategy such as:
- best-response iteration,
- two-dimensional root finding,
- or coarse grid plus local refinement.

### Deliverables
Produce:
- a downstream pricing-subgame solver,
- a small demo or test script that evaluates the solver at several manually chosen `D` values,
- structured diagnostics.

### Do not do
- Do not integrate with the full outer simulation yet.
- Do not optimize performance prematurely.
- Do not silently suppress convergence failures.

### Completion criteria
This stage is complete only if:
- the solver converges for several manually chosen `D` values,
- increasing student quality tends to improve the student’s equilibrium share,
- failure is explicit and diagnostic-rich rather than silent.

### Prompt template for this stage
> Implement a standalone downstream pricing-subgame solver for the competition model, conditional on a given D. Return equilibrium prices, shares, downstream profits, and structured diagnostics. Use a conservative numerical strategy and include a minimal demo for several D values. Do not integrate this solver into the full simulation pipeline yet.

---

## Stage 3 — Implement the competition-aware student best response

### Your task
Replace the reduced-form student objective with a competition-aware objective while preserving the baseline pattern of one-dimensional optimization over `D`.

### Required behavior
Implement a student best-response solver that:
- takes teacher token price `p` as input,
- optimizes over `D`,
- calls the Stage 2 downstream solver inside the objective,
- computes student equilibrium downstream profit minus training expenditure,
- returns structured diagnostics.

### Minimum output fields
Return at least:
- `D_star`,
- optimal student total payoff,
- whether the optimum is on a boundary,
- solver evaluation counts,
- message,
- optional embedded downstream equilibrium summary at the optimum.

### Design constraint
Mirror the baseline student-solver interface where practical. Add a new competition-specific solver rather than replacing the original baseline solver.

### Deliverables
Produce:
- a competition-specific student best-response solver,
- a small demo over a few teacher prices `p`,
- boundary and convergence diagnostics.

### Do not do
- Do not rewrite the baseline student solver.
- Do not add performance hacks yet.
- Do not merge this with plotting or full experiment orchestration.

### Completion criteria
This stage is complete only if:
- the solver returns sensible `D_star` values for a grid of `p`,
- `D_star` usually decreases as `p` rises,
- boundary solutions are explicitly reported.

### Prompt template for this stage
> Implement a competition-specific student best-response solver that takes teacher price p, optimizes over D, and internally calls the downstream pricing-subgame solver. Preserve the baseline style of bounded 1D optimization and structured diagnostics. Do not replace the original baseline student solver.

---

## Stage 4 — Implement the competition simulation runner

### Your task
Reconstruct the full competition simulation workflow in the same spirit as the baseline Stackelberg pipeline.

### Required behavior
Build a simulation runner that:
- constructs a grid over teacher token price `p`,
- for each `p`, solves the student best response,
- recovers the downstream equilibrium at the student optimum,
- computes teacher total profit,
- records row-level diagnostics,
- selects the optimal teacher token price.

### Minimum output fields
Return or save at least:
- `p_star`,
- `D_star` at `p_star`,
- equilibrium downstream prices at `p_star`,
- teacher total profit at the optimum,
- per-price rows containing all key equilibrium and diagnostic variables.

### Deliverables
Produce:
- a competition simulation runner,
- a per-price results table,
- an equilibrium summary object.

### Do not do
- Do not replace the grid-based workflow with a fully different architecture.
- Do not fold plotting into the runner.
- Do not discard row-level diagnostics.

### Completion criteria
This stage is complete only if:
- the code can run the competition model over a price grid,
- it identifies the teacher-optimal price,
- it retains enough row-level information for later plotting and mechanism interpretation.

### Prompt template for this stage
> Implement a competition simulation runner that follows the baseline grid-based experiment style: loop over teacher token price p, solve the student best response, recover the downstream equilibrium, compute teacher total profit, store rich per-row diagnostics, and identify the optimal price. Do not mix plotting into this step.

---

## Stage 5 — Implement output saving and core competition plots

### Your task
Turn the solver stack into a reusable experiment pipeline with saved outputs and a small initial figure set.

### Required behavior
Add output saving for at least:
- per-price CSV tables,
- JSON diagnostics,
- equilibrium summary outputs.

Add a first set of competition-specific plots. Recommended priorities:
- `D_star(p)`,
- teacher total profit vs `p`,
- downstream equilibrium prices vs `p`,
- downstream shares vs `p`.

### Design constraint
Plotting code should read saved results where practical. Do not let plotting implicitly rerun equilibrium solving.

### Deliverables
Produce:
- output-saving utilities or experiment-entry code,
- at least two or three essential plots,
- a clear separation between solver logic and plotting logic.

### Do not do
- Do not generate a full paper-figure suite in one shot.
- Do not mix figure generation with numerical solving internals.

### Completion criteria
This stage is complete only if:
- saved outputs are organized and reusable,
- the main figures can be regenerated from saved data,
- plotting remains separated from numerical solving.

### Prompt template for this stage
> Add output saving and a minimal set of competition plots. Save per-price tables and diagnostics, and generate only the most essential two or three figures first. Keep plotting separate from solving, and prefer plotting from saved results rather than rerunning the model inside the plotting functions.

---

## Stage 6 — Add trustworthiness diagnostics and stability checks

### Your task
Instrument the competition code so numerical problems can be distinguished from real model mechanisms.

### Required checks
Add checks for at least:
- shares summing to one,
- admissible price regions,
- internally consistent profit identities,
- explicit flags for failed downstream solves,
- counts of boundary solutions,
- counts of clipping or fallback use.

Add a few comparative-statics smoke tests, such as:
- higher `p` usually lowers `D_star`,
- higher student quality usually raises student share,
- a more attractive outside option reduces inside shares.

Optional:
- local numerical-derivative checks in a calibration region.

### Deliverables
Produce:
- diagnostics helpers,
- summaries of failures and boundary events,
- lightweight smoke tests or validation scripts.

### Do not do
- Do not redesign the project structure.
- Do not treat this stage as a solver rewrite unless a bug is found.

### Completion criteria
This stage is complete only if:
- diagnostics can distinguish economic behavior from numerical pathology,
- single-point failures do not silently contaminate final curves or reported optima.

### Prompt template for this stage
> Add diagnostics and trustworthiness checks to the competition model without redesigning the architecture. Include sanity checks, failure summaries, boundary-solution reporting, and a few comparative-statics smoke tests. The goal is to instrument the code, not rewrite it.

---

## Stage 7 — Optimize performance and tidy shared structure

### Your task
Only after correctness and stability are established, improve efficiency and clean up shared utilities.

### Allowed improvements
You may now add:
- caching of repeated evaluations,
- warm starts,
- coarse-grid plus local-refinement strategies,
- fallback solver hierarchy,
- shared diagnostics containers,
- shared save helpers,
- shared result dataclasses.

### Deliverables
Produce:
- a more efficient version of the validated competition pipeline,
- minor refactors that reduce duplication without changing model logic,
- evidence that outputs remain materially unchanged.

### Do not do
- Do not change the economic model.
- Do not merge unrelated modules just for elegance.
- Do not refactor away diagnostics.

### Completion criteria
This stage is complete only if:
- runtime improves without materially changing results,
- refactoring does not break previously validated stages.

### Prompt template for this stage
> Optimize the validated competition pipeline without changing the economic model or baseline workflow. Add caching, warm starts, or solver fallbacks where useful, and refactor only shared utilities that reduce duplication. Preserve diagnostics and verify that results do not materially change.

---

## Required execution order

Do not skip this order.

1. Stage 0: baseline audit and extension boundaries
2. Stage 1: static competition helpers
3. Stage 2: downstream pricing-subgame solver
4. Stage 3: competition-aware student best response
5. Stage 4: competition simulation runner
6. Stage 5: output saving and core plots
7. Stage 6: trustworthiness diagnostics
8. Stage 7: performance optimization and cleanup

---

## Master instruction to prepend to every Copilot request

Prepend the following instruction to every stage-specific prompt:

> Do not modify existing baseline modules unless required for interface reuse. Add competition-specific code in parallel, preserve the current grid-based experiment workflow, and avoid hard-coding repository structure because the provided source files are only a structural reference rather than the literal project layout.

---

## Final rule

At the end of each stage, stop and return only the code and notes required for that stage. Do not proactively implement later stages.
