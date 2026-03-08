# competition_model_ai_briefing

## Purpose

This document is a **Copilot-facing implementation brief** for the **competition extension** of the existing baseline project.

Its role is **not** to restate the full theoretical proof. Its role is to tell Copilot how to convert the theory into a simulation code pipeline that is **consistent with the baseline design principles** already used in the current project.

The baseline project already follows the pattern:

`theory -> config -> technology block -> payoff block -> solver -> simulation loop -> saved outputs -> plots`

The competition extension should preserve this logic rather than introducing an entirely different coding style.

---

## 1. What Copilot should understand first

The current codebase already contains a reusable baseline architecture:

- a **configuration layer** for parameters and grids,
- a **technology layer** for scaling-law-based student loss,
- an **economic/payoff layer**,
- a **solver layer** for the student best response,
- a **simulation layer** that loops over a grid of teacher prices,
- a **visualization/output layer**.

The competition model should be implemented as an **extension of this architecture**, not as an unrelated script.

In particular, the extension should **reuse** the existing baseline ideas wherever possible:

1. **Keep the technology block unchanged unless the theory explicitly changes it.**  
   The student loss function, anchor normalization, and scaling-law calibration logic should remain reusable.

2. **Keep parameters externalized in YAML (or an equivalent config file).**  
   Competition-specific parameters should be added as new config fields rather than hard-coded.

3. **Keep economic logic separated from experiment-running logic.**  
   Core equations should live in model/solver modules; experiment scripts should only orchestrate loading, solving, saving, and plotting.

4. **Keep the simulation research-friendly.**  
   Every stage should produce diagnostics, not just final figures.

---

## 2. What changes relative to the baseline

The baseline model is effectively a vertical pricing model with one upstream teacher and one student, where the student chooses training demand and the teacher chooses token price. In code, this is implemented as a **nested one-dimensional problem**:

- for each token price `p`, solve the student’s optimal `D*(p)`;
- then compute teacher payoff on a price grid;
- then choose the best `p`.

The competition extension keeps this high-level structure, but adds an **extra downstream game** inside the loop.

The new sequence is:

1. Teacher sets upstream token price `p`.
2. Student chooses training volume `D`.
3. Teacher and student compete downstream by choosing `P_T` and `P_S`.
4. Downstream shares and profits are computed from a logit demand system with outside option.
5. These downstream equilibrium profits feed back into the student’s Stage-2 problem and the teacher’s Stage-1 problem.

So, compared with the baseline, the competition model is a **nested equilibrium problem**:

- **inner layer**: downstream pricing equilibrium at fixed `D`;
- **middle layer**: student chooses `D` anticipating the downstream equilibrium;
- **outer layer**: teacher chooses `p` anticipating both of the above.

---

## 3. What theory needs to be preserved in code

Copilot does **not** need to reproduce the full proof structure from the theory notes. It only needs the minimal economic objects required for simulation.

### 3.1 Keep these objects

#### A. Technology block

The student loss function remains the bridge from training scale to model performance:

- `L_student(N, D)`
- optionally also expose `L_tilde(N, D)` and `gap_term(N, D)` for diagnostics/plots

This should continue to come from the existing scaling-law layer.

#### B. Quality mapping

The downstream game uses a quality index, not loss directly. Implement an explicit mapping such as:

- `q_S(D) = psi(L_student(N, D))`
- teacher quality `q_T` is fixed and exogenous

Recommended default:

- `psi(L) = -L`

But the code should make this mapping easy to replace later.

#### C. Downstream logit demand with outside option

At fixed `D`, downstream demand depends on:

- teacher downstream price `P_T`
- student downstream price `P_S`
- teacher quality `q_T`
- student quality `q_S(D)`
- outside-option utility level
- price sensitivity parameter
- market size `M`

The code should compute:

- `s_T(P_T, P_S, D)`
- `s_S(P_T, P_S, D)`
- optionally `s_0(P_T, P_S, D)`

with the invariant check:

- `s_T + s_S + s_0 = 1` up to numerical tolerance.

#### D. Downstream profits

At fixed `D`, define:

- teacher downstream profit
- student downstream profit

using constant marginal costs in the downstream market.

#### E. Total profits

The total-profit objects that matter for the upper stages are:

- teacher total profit = upstream token profit + downstream operating profit
- student total profit = downstream operating profit - training cost

These are the quantities to optimize in simulation.

---

## 4. What theory can be omitted from the implementation brief

The detailed derivations in the theory file are useful for the paper, but they do not need to be encoded as implementation instructions.

Copilot does **not** need to be told to code:

- long-form proofs of existence/uniqueness,
- symbolic derivations of comparative statics,
- proposition-style narrative,
- envelope-theorem discussion in prose,
- theorem numbering or proof formatting.

Instead, Copilot should be told to implement the **numerical counterparts**:

- a downstream equilibrium solver,
- a student best-response solver using reduced-form downstream equilibrium profit,
- a teacher outer-loop solver,
- diagnostics that numerically verify regularity conditions in the parameter region used for experiments.

---

## 5. Recommended code architecture

Do **not** assume the real project must use the exact same filenames as the baseline sample. The real project structure may differ.

Instead, preserve **responsibility boundaries**.

A good structure is:

### 5.1 Config layer

A config file should define:

- baseline technology parameters already used in the project,
- competition-specific economic parameters,
- downstream-demand parameters,
- solver options,
- plotting/output switches.

Competition-specific parameters should include at least:

- teacher downstream quality or quality anchor
- teacher downstream marginal cost `m_T`
- student downstream marginal cost `m_S`
- outside-option utility level
- price sensitivity in downstream demand
- market size `M`
- downstream price bounds or initialization values
- downstream equilibrium solver tolerances

Do not hard-code these in the solver.

### 5.2 Technology layer

Reuse the existing technology implementation if the competition extension still assumes the same student loss/scaling-law structure.

This layer should continue to expose callable functions such as:

- `L_student(N, D)`
- `L_tilde(N, D)`
- `gap_term(N, D)`

If needed, add a thin wrapper:

- `student_quality(N, D)`

but do not duplicate scaling-law logic.

### 5.3 Downstream competition layer

This is the main new module.

It should define:

1. utility/index functions for teacher, student, and outside option;
2. logit share functions;
3. downstream profit functions;
4. a solver for downstream equilibrium prices `(P_T*, P_S*)` at fixed `D`;
5. a result object/dataclass for downstream equilibrium diagnostics.

This layer should be independent from the outer price-grid simulation.

### 5.4 Student problem layer

This layer should define the student’s Stage-2 objective:

- for fixed upstream token price `p`, choose `D` to maximize student total profit,
- where student downstream profit is computed **after solving the downstream subgame**.

This is the new version of the baseline student best-response solver.

### 5.5 Teacher problem / simulation layer

This layer should implement the teacher’s Stage-1 problem:

- loop over candidate `p`,
- solve student best response,
- solve implied downstream equilibrium,
- compute teacher total profit,
- select the best `p`.

In the first implementation, a **grid search over `p`** is acceptable and is fully consistent with the baseline style.

### 5.6 Visualization/output layer

This should remain separate from the solver.

The competition extension should save tables/JSON diagnostics and generate figures from saved results rather than mixing plotting logic into equilibrium code.

---

## 6. Core implementation tasks for Copilot

Copilot should implement the competition model in the following order.

### Task 1: Reuse the baseline technology block

Do not rewrite the scaling-law machinery unless the new theory explicitly changes it.

The competition extension should continue to evaluate student performance via the same loss function used in the baseline, because the competition mechanism enters **after** training, through downstream market interaction.

### Task 2: Add a quality-mapping function

Create a small, explicit function converting student loss to downstream quality.

Recommended default:

```python
q_s = -L_student(N, D)
```

but structure it so that alternative mappings can later be plugged in.

Also include a fixed teacher quality parameter:

```python
q_t = q_teacher
```

### Task 3: Implement downstream shares

Implement a numerically stable multinomial logit demand block with three choices:

- teacher,
- student,
- outside option.

This function should accept downstream prices and qualities and return shares.

Use numerically stable exponentiation, for example by subtracting the maximum utility index before exponentiating.

### Task 4: Implement downstream profits

Given `(P_T, P_S, D)`, compute:

- shares,
- teacher downstream profit,
- student downstream profit.

This function should not yet worry about upper-stage optimization.

### Task 5: Solve the downstream pricing equilibrium at fixed `D`

This is the most important new component.

At a given training level `D`, solve the teacher–student downstream pricing Nash equilibrium.

Recommended practical approach:

- implement the two first-order conditions numerically,
- solve them jointly as a 2D root/fixed-point problem,
- return equilibrium prices, shares, and profits.

Possible numerical methods:

- `scipy.optimize.root`,
- `scipy.optimize.fsolve`,
- or an iterative best-response routine if it is sufficiently robust.

The solver should return a structured result with:

- `P_T_star`, `P_S_star`
- `s_T_star`, `s_S_star`, `s_0_star`
- `pi_T_down_star`, `pi_S_down_star`
- convergence status
- residual norms
- number of iterations/evaluations

The implementation should prefer robustness and diagnostics over elegance.

### Task 6: Define reduced-form downstream equilibrium objects as functions of `D`

Once the downstream equilibrium solver exists, expose helper functions such as:

- `solve_downstream_equilibrium(D, params)`
- `teacher_downstream_profit_equilibrium(D, ...)`
- `student_downstream_profit_equilibrium(D, ...)`

These reduced-form objects are what the upper stages will consume.

### Task 7: Replace the baseline student objective with the competition version

In the baseline, the student objective is directly computed from value minus training cost.

In the competition model, the student objective must be:

1. compute `q_S(D)` from the technology block,
2. solve the downstream pricing equilibrium at that `D`,
3. read off student downstream equilibrium profit,
4. subtract training cost `(p + k) * D`.

Then solve:

- `D*(p) = argmax student_total_profit(D; p)`

This is still a one-dimensional optimization in `D`, but each objective evaluation now contains an inner downstream-equilibrium solve.

So the student best-response solver becomes **nested**.

### Task 8: Keep the teacher outer problem close to the baseline style

The teacher’s outer-loop problem should remain structurally similar to the baseline implementation:

1. build a price grid for `p`,
2. for each `p`, solve the student best response,
3. read off the downstream equilibrium implied by that best response,
4. compute total teacher profit,
5. choose the best `p`.

This preserves continuity with the baseline code and makes debugging easier.

### Task 9: Add diagnostics for all nested layers

Because this extension is numerically more complex, Copilot must save richer diagnostics than in the baseline.

At minimum, diagnostics should report:

- downstream-equilibrium solver success rate,
- downstream residual sizes,
- how often downstream prices hit bounds,
- student best-response boundary frequency,
- whether `D*(p)` is roughly non-increasing in `p`,
- whether teacher payoff has a clear interior maximum or only a boundary maximum,
- whether shares remain in `[0,1]` numerically,
- whether `s_T + s_S + s_0` is close to `1`.

Diagnostics are important because the competition extension may fail numerically even when the equations are theoretically fine.

---

## 7. Numerical design principles Copilot should follow

### 7.1 Prioritize robustness over symbolic purity

The theory file contains analytic markup equations and comparative-statics discussion, but the first implementation should focus on a robust numerical equilibrium solver.

This means:

- safe bounds,
- stable initial guesses,
- explicit convergence checks,
- stored residuals,
- fallback logic if the preferred solver fails.

### 7.2 Keep the model modular

Do not hard-wire competition directly into the existing baseline functions in a way that destroys reusability.

Instead:

- keep baseline modules reusable,
- add competition-specific modules or subclasses/wrappers,
- expose reduced-form interfaces for upper layers.

### 7.3 Avoid overcommitting to filenames

The sample “source” files are only an implementation sketch, not the real repository layout.

So Copilot should follow **logical roles**, not exact paths.

For example, if the real project uses different names such as:

- `competition_model.py`
- `downstream_game.py`
- `run_competition_experiment.py`

that is fine, as long as the responsibilities remain clear.

### 7.4 Keep outputs reproducible

As in the baseline, each run should save:

- the effective config used,
- summary statistics,
- diagnostics,
- tabular equilibrium results,
- figures.

Avoid “plot-only” scripts that do not save the underlying numeric table.

---

## 8. Suggested data structures

Copilot should use explicit result containers/dataclasses rather than raw tuples when possible.

Recommended result types include:

### 8.1 Downstream equilibrium result

Fields may include:

- `D`
- `q_T`, `q_S`
- `P_T_star`, `P_S_star`
- `s_T_star`, `s_S_star`, `s_0_star`
- `pi_T_down_star`, `pi_S_down_star`
- `success`
- `message`
- `iterations`
- `residual_norm`
- `hit_bounds`

### 8.2 Student best-response result under competition

Fields may include:

- `p`
- `D_star`
- `pi_student_star`
- embedded downstream equilibrium summary at `D_star`
- `is_boundary`
- `boundary_side`
- optimizer success diagnostics

### 8.3 Outer simulation row

For each candidate `p`, store:

- `p`
- `D_star`
- `pi_student_total`
- `pi_teacher_total`
- downstream equilibrium prices
- downstream shares
- downstream profits
- solver diagnostics from both inner layers

This makes later plotting and debugging much easier.

---

## 9. What figures the competition extension should generate

At minimum, generate figures that correspond to the actual mechanism of the model, not just baseline-style loss plots.

Recommended figures:

1. **Student best-response training demand `D*(p)`**  
   Shows how upstream price affects training scale.

2. **Teacher total profit vs upstream price `p`**  
   Used to identify the optimal token price.

3. **Student total profit vs upstream price `p`**  
   Shows the effect of strategic upstream pricing on the student.

4. **Downstream equilibrium prices vs student training `D` or vs upstream price `p`**  
   Shows the downstream competition channel explicitly.

5. **Downstream market shares vs upstream price `p`**  
   Important because the mechanism is mediated through quality and logit demand.

6. **Student quality / loss vs upstream price `p`**  
   Connects the technology block to the competition outcome.

7. **Optional comparison figure: baseline vs competition**  
   Useful to show how adding downstream competition changes the optimal upstream price.

If the experiment section is meant for paper writing, also save figure-ready CSV tables behind every plot.

---

## 10. How Copilot should treat outside-option logic

The competition model already contains an outside option inside the downstream logit system.

Therefore, Copilot should **not** mechanically transplant the baseline hard/soft outside-option logic unless the theory explicitly says to combine them.

Important distinction:

- In the baseline, outside option was implemented as an extra participation mechanism around the student’s payoff.
- In the competition model, outside option is already built into downstream consumer choice through the logit share system.

So the default implementation for competition should treat the outside option as part of **downstream demand**, not as the old baseline entry/exit wrapper.

If later you want an additional participation layer on top of downstream competition, that should be added deliberately as a separate extension, not implicitly.

---

## 11. Recommended solver strategy for the first working version

For a first working simulation, Copilot should prefer a straightforward, debuggable pipeline:

### Stage 3: downstream equilibrium

- solve downstream prices numerically for each fixed `D`
- use a robust 2D numerical solver
- keep good initial guesses, possibly reusing nearby solutions across neighboring `D` values

### Stage 2: student best response

- use bounded 1D optimization over `D`
- each function evaluation calls the downstream-equilibrium solver
- cache repeated evaluations if needed for speed

### Stage 1: teacher pricing

- use a grid over `p`, consistent with the baseline implementation style
- after the first working version is stable, optionally refine around the best grid point

This staged approach is more appropriate than jumping immediately to a fully simultaneous high-dimensional solver.

---

## 12. Recommended pseudocode

```python
load_config()
tech = build_technology_from_config(cfg)
params = build_competition_params(cfg)

for p in p_grid:
    def student_objective(D):
        q_s = student_quality(tech, N, D)
        down_eq = solve_downstream_equilibrium(D=D, q_s=q_s, params=params)
        pi_s_down = down_eq.pi_S_down_star
        return pi_s_down - (p + k) * D

    D_star = solve_student_best_response(student_objective, D_bounds)
    down_eq_star = solve_downstream_equilibrium(D=D_star, q_s=student_quality(...), params=params)

    pi_teacher_total = (p - c) * D_star + down_eq_star.pi_T_down_star
    pi_student_total = down_eq_star.pi_S_down_star - (p + k) * D_star

    save_row(p, D_star, down_eq_star, pi_teacher_total, pi_student_total, diagnostics)

choose p_star that maximizes teacher total profit
save_tables_and_diagnostics()
make_plots()
```

This pseudocode is intentionally close to the baseline style, but inserts the downstream-equilibrium layer.

---

## 13. Baseline compatibility rules

Copilot should preserve the following compatibility principles with the existing baseline codebase.

### Rule 1
Do not rewrite the existing baseline experiment unless necessary.

### Rule 2
Do not duplicate the scaling-law implementation.

### Rule 3
Do not bury core equations in plotting scripts or experiment entry scripts.

### Rule 4
Do not hard-code project paths or assume a fake sample structure is the real repository layout.

### Rule 5
Do not silently drop diagnostics just because the nested solver becomes more complicated.

### Rule 6
When extending configs, use additive fields so the baseline configs still remain readable.

### Rule 7
Keep the competition extension runnable as its own experiment script/module.

---

## 14. What the first deliverable should be

Copilot’s first successful deliverable should be a **minimal but complete competition experiment** that can:

1. load config,
2. build the existing technology block,
3. solve downstream equilibrium numerically,
4. solve student best response for each upstream price,
5. compute teacher total profit,
6. select the teacher’s optimal upstream price,
7. save a table of results,
8. save diagnostics,
9. generate a small set of core plots.

This is enough to establish the bridge from theory to simulation.

Only after this works robustly should Copilot add:

- refined comparative-statics routines,
- faster solvers,
- multiple parameter sweeps,
- welfare calculations,
- baseline-vs-extension comparison utilities.

---

## 15. One-sentence summary for Copilot

Treat the competition model as a **baseline-style nested simulation framework** in which the existing scaling-law technology is reused, a new downstream logit-pricing equilibrium is inserted as the inner layer, and the student and teacher problems are solved around that inner equilibrium with full diagnostics and reproducible outputs.
