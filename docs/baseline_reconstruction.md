# baseline_ai_briefing

## What this project is

This codebase implements **one baseline theory** from `whole_baseline.md`, with **two baseline variants**:

- **Hard outside option**: `simulation.py` + `base.yaml` + `exp_10_baseline_base_hard.py`
- **Soft outside option**: `simulation_soft.py` + `soft.yaml` + `exp_11_baseline_base_soft.py`

Both variants share the same technology block and payoff core.

---

## Read this project in one line

The project pipeline is:

`theory -> YAML parameters -> scaling-law functions -> payoff functions -> student best response -> price-grid simulation -> plots + saved outputs`

---

## File roles

- `whole_baseline.md` — baseline theory, including outside-option repair.
- `base.yaml` — parameters for the **hard outside option** baseline run.
- `soft.yaml` — parameters for the **soft outside option** baseline run.
- `config_loader.py` — loads and validates YAML config.
- `normalization.py` — recovers scaling-law coefficients from anchor conditions.
- `scaling_laws.py` — defines supervised frontier, gap term, and distilled student loss.
- `model.py` — defines value function, profit functions, and student-side optimization.
- `simulation.py` — hard-outside baseline simulation over a price grid.
- `simulation_soft.py` — soft-outside baseline simulation over a price grid.
- `visualization.py` — plotting and figure export.
- `exp_10_baseline_base_hard.py` — runnable hard-outside experiment script.
- `exp_11_baseline_base_soft.py` — runnable soft-outside experiment script.

---

## Core variable map

| Theory | Code | Meaning |
|---|---|---|
| `N_S` | `N` | student model size used at runtime |
| reference scale for `N_S` | `N0` | normalization constant |
| `D` | `D` / `D_star` | student token demand / best response |
| `p` | `p` | teacher token price |
| `k` | `econ.k` | student per-token compute cost |
| `c` or `c_T` | `econ.c_T` | teacher marginal token cost |
| `\tilde L_S` | `tech.L_tilde(...)` | supervised capacity frontier |
| gap term | `tech.gap_term(...)` | distillation residual |
| `L_S` | `tech.L_student(...)` | realized student loss |
| `V(L_S)` | `downstream_value_linear(...)` | downstream value, currently linear |
| `\Pi_S(D)` | `student_profit(...)` | student profit |
| `\Pi_S^*(p)` | `br.pi_star` | optimized student payoff at price `p` |
| `D^*(p)` | `br.D_star` | best-response demand at price `p` |
| `\Pi_T(p)` | `teacher_profit(...)` or simulation-level teacher payoff | teacher payoff |
| hard outside option | `opted_out` logic in `simulation.py` | zero demand if optimized payoff is negative |
| soft outside option | `s_enter`, `D_soft` | smoothed participation |

---

## Equation -> code map

### 1. Supervised frontier

Theory:

```math
\tilde L_S(N_S,D)=E+\left(\frac{A}{N_S^\alpha}+\frac{B}{D^\beta}\right)^\gamma
```

Code:

- `TierATechnology.L_tilde()` in `scaling_laws.py`
- coefficients are reconstructed in `normalization.py`

### 2. Distilled student loss

Theory:

```math
L_S(N_S,D)
```

Code:

- `TierATechnology.L_student()` in `scaling_laws.py`
- implemented as:

```python
L_student = L_tilde(N, D) + gap_term(N, D)
```

### 3. Student value function

Theory:

```math
V(L_S)
```

Code currently specializes it to:

```math
V(L)=a-bL
```

implemented by:

- `downstream_value_linear()` in `model.py`

### 4. Student profit

Theory:

```math
\Pi_S(D)=V(L_S(N_S,D))-(p+k)D
```

Code:

- `student_profit()` in `model.py`

### 5. Student optimization

Theory:

```math
D^*(p)=\arg\max_{D\ge 0}\Pi_S(D)
```

Code:

- `solve_student_best_response_direct()` in `model.py`
- uses bounded 1D numerical maximization (`minimize_scalar`)

### 6. Teacher payoff and pricing

Theory:

```math
\Pi_T(p)=(p-c)D(p)
```

Code:

- hard-demand payoff helper: `teacher_profit()` in `model.py`
- actual teacher-side search is implemented in the simulation modules by **grid search over price**

---

## Where parameters live

Main parameter files:

- `base.yaml` — hard outside option run
- `soft.yaml` — soft outside option run

Important parameter groups:

- `student` — includes `N0`
- `exponents` — scaling-law exponents
- `anchors_supervised` — recovers supervised-frontier coefficients
- `anchors_gap` — recovers gap-term coefficients
- `economics` — `a`, `b`, `k`, `c_T`
- `grids` — ranges for `D` and `p`
- `solver` — numerical solver tolerances
- `soft_outside` — `tau` and related settings for the soft variant only

Load path:

- `config_loader.py` loads and validates YAML
- `scaling_laws.py` builds the technology object from config
- `model.py` builds economic parameters from config

---

## Where solving happens

### Student-side solver

Implemented in `model.py`:

- `solve_student_best_response_direct()`

This solves conditional demand for each given price `p`.

### Teacher-side solution logic

Implemented by simulation over a price grid:

- `run_baseline_grid_simulation()` in `simulation.py`
- `run_soft_grid_simulation()` in `simulation_soft.py`

Shared logic:

1. build a price grid
2. for each `p`, solve student best response
3. apply outside-option rule
4. compute teacher payoff
5. choose the price that maximizes teacher payoff

So the code solves a **grid-based Stackelberg problem**, not a symbolic equilibrium system.

---

## Hard vs soft outside option

### Hard outside option

Location:

- `simulation.py`

Logic:

- solve conditional `D_star` and `pi_star`
- if `pi_star < 0`, set realized demand and both realized payoffs to `0`

### Soft outside option

Location:

- `simulation_soft.py`

Logic:

- solve conditional `D_star` and `piS_star`
- convert `piS_star` into participation probability using `tau`
- compute effective demand `D_soft = s_enter * D_star`
- compute soft student and teacher payoffs from effective participation

---

## Which script runs experiments

- `exp_10_baseline_base_hard.py` — main runnable script for hard-outside baseline experiments
- `exp_11_baseline_base_soft.py` — main runnable script for soft-outside baseline experiments

These scripts do the full pipeline:

1. load YAML
2. build model objects
3. run simulation
4. save tables / diagnostics
5. call plotting functions

---

## Which script draws plots

- `visualization.py`

It contains plot functions for:

- scaling curves
- student profit slices
- demand curves
- teacher profit curves
- indirect student payoff curves
- soft-outside figures

---

## Which scripts save outputs

The main save logic is in the experiment entry scripts:

- `exp_10_baseline_base_hard.py`
- `exp_11_baseline_base_soft.py`

Typical outputs:

- CSV tables under `results/tables/`
- JSON diagnostics / optimum summaries
- figures under `results/figures/`
- run logs under `results/logs/`

---

## Minimal mental model for Copilot

Use this interpretation when extending the project:

1. `whole_baseline.md` defines the economics.
2. `normalization.py` + `scaling_laws.py` define the loss technology.
3. `model.py` defines profits and the student optimization problem.
4. `simulation.py` / `simulation_soft.py` implement teacher-side pricing plus outside-option mechanics.
5. `visualization.py` draws figures.
6. `exp_*.py` scripts are the runnable experiment entry points.

In short:

> This is **one baseline model with two implementation variants** (hard outside, soft outside), sharing a common technology and payoff core.
