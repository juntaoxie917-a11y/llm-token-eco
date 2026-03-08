# Distillation Tier A: Baseline and Competition

This repository contains two directly runnable model families for paper comparison:

- Baseline model (hard/soft outside option)
- Competition extension (downstream logit pricing game nested inside upstream pricing)

## Project Structure (Essential Files)

Core shared blocks:

- `src/config_loader.py`
- `src/normalization.py`
- `src/scaling_laws.py`
- `src/model.py`

Baseline modules:

- `src/simulation.py`
- `src/simulation_soft.py`
- `src/visualization.py`
- `experiments/exp_01_hard_outside.py`
- `experiments/exp_02_soft_outside.py`

Competition modules:

- `src/competition_static.py`
- `src/competition_downstream_solver.py`
- `src/competition_student.py`
- `src/competition_simulation.py`
- `src/competition_visualization.py`
- `experiments/exp_07_competition_stage5_pipeline.py`

Configs:

- `config/base.yaml`
- `config/soft.yaml`
- `config/competition.yaml`

`soft.yaml` and `competition.yaml` both support `run.base_config` to reuse `base.yaml` and keep only scenario-specific overrides.

Reference docs:

- `docs/baseline_reconstruction.md`
- `docs/competition_ai_briefing.md`

## How To Run

Baseline:

```bash
python experiments/exp_01_hard_outside.py
python experiments/exp_02_soft_outside.py
```

Competition:

```bash
python experiments/exp_07_competition_stage5_pipeline.py
```

## Outputs

Main outputs are written to:

- `results/tables/`
- `results/figures/`
- `results/logs/`

For competition runs, key files include:

- `results/tables/competition_stage5_grid_results.csv`
- `results/tables/competition_stage5_optimum.json`
- `results/tables/competition_stage5_diagnostics.json`
- `results/figures/fig_comp_01_dstar_vs_p.*`
- `results/figures/fig_comp_02_teacher_profit_vs_p.*`
- `results/figures/fig_comp_03_downstream_prices_vs_p.*`
- `results/figures/fig_comp_04_downstream_shares_vs_p.*`
