# Distillation Tier A Baseline + Competition Extension

## Baseline Status (Completed)

The baseline pipeline is fully implemented and runnable end-to-end.

- Config and validation:
	- `config/base.yaml`, `config/soft.yaml`
	- `src/config_loader.py`
- Technology block (shared core):
	- `src/normalization.py`
	- `src/scaling_laws.py`
- Economic/payoff block:
	- `src/model.py`
- Simulation block:
	- Hard outside option: `src/simulation.py`
	- Soft outside option: `src/simulation_soft.py`
- Visualization block:
	- `src/visualization.py`
- Experiment entrypoints:
	- Hard outside: `experiments/exp_01_hard_outside.py`
	- Soft outside: `experiments/exp_02_soft_outside.py`

Baseline outputs are already generated under:

- `results/tables/` (e.g. `baseline_demand_curve.csv`, `baseline_optimum.json`, `diagnostics.json`, `soft_demand_curve.csv`, `soft_diagnostics.json`)
- `results/figures/`
- `results/logs/`

## Competition Extension Status (Stage 0-7 Completed)

The competition extension was implemented incrementally following `docs/competition_copilot_execution_plan.md`.

### Stage 0: Audit and Boundaries

- Audit note: `docs/competition_stage0_audit.md`
- Classified baseline modules into reuse/wrap/new/do-not-reuse-directly.

### Stage 1: Static Competition Building Blocks

- Module: `src/competition_static.py`
- Implemented:
	- competition parameter dataclasses,
	- quality mapping (`loss -> quality`),
	- stable downstream logit shares,
	- downstream profit helpers.
- Demo script: `experiments/exp_03_competition_stage1_demo.py`

### Stage 2: Downstream Pricing Subgame Solver

- Module: `src/competition_downstream_solver.py`
- Implemented standalone fixed-`D` downstream Nash pricing solver with:
	- structured equilibrium result,
	- convergence diagnostics,
	- fallback method.
- Demo script: `experiments/exp_04_competition_stage2_demo.py`

### Stage 3: Competition-Aware Student Best Response

- Module: `src/competition_student.py`
- Implemented nested student BR over `D` for each upstream `p`, calling Stage 2 internally.
- Demo script: `experiments/exp_05_competition_stage3_demo.py`

### Stage 4: Competition Simulation Runner

- Module: `src/competition_simulation.py`
- Implemented grid-based upstream pricing simulation with rich row-level diagnostics and optimum selection.
- Demo script: `experiments/exp_06_competition_stage4_demo.py`

### Stage 5: Output Saving + Core Plots

- Module: `src/competition_visualization.py`
- Pipeline script: `experiments/exp_07_competition_stage5_pipeline.py`
- Saved artifacts:
	- `results/tables/competition_stage5_grid_results.csv`
	- `results/tables/competition_stage5_optimum.json`
	- `results/tables/competition_stage5_diagnostics.json`
	- `results/figures/fig_comp_01_dstar_vs_p.*`
	- `results/figures/fig_comp_02_teacher_profit_vs_p.*`
	- `results/figures/fig_comp_03_downstream_prices_vs_p.*`
	- `results/figures/fig_comp_04_downstream_shares_vs_p.*`

### Stage 6: Trustworthiness Diagnostics

- Module: `src/competition_diagnostics.py`
- Script: `experiments/exp_08_competition_stage6_diagnostics.py`
- Report:
	- `results/tables/competition_stage6_report.json`

### Stage 7: Performance Optimization and Stability Evidence

- Added cache-enabled nested solve path (without changing economic model):
	- `src/competition_student.py`
	- `src/competition_simulation.py`
- Benchmark script:
	- `experiments/exp_09_competition_stage7_benchmark.py`
- Report:
	- `results/tables/competition_stage7_benchmark.json`

## Quick Run Commands

Baseline:

```bash
python experiments/exp_01_hard_outside.py
python experiments/exp_02_soft_outside.py
```

Competition pipeline and diagnostics:

```bash
python experiments/exp_07_competition_stage5_pipeline.py
python experiments/exp_08_competition_stage6_diagnostics.py
python experiments/exp_09_competition_stage7_benchmark.py
```
