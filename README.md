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
- `src/competition_threshold.py`
- `src/competition_visualization.py`
- `experiments/exp_07_competition_stage5_pipeline.py`
- `experiments/exp_08_competition_market_threshold.py`

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

Competition threshold analysis (market-size critical threshold):

```bash
python experiments/exp_08_competition_market_threshold.py
```

Threshold settings are configured in `config/competition.yaml` under:

- `competition.threshold_analysis.market_size_min`
- `competition.threshold_analysis.market_size_max`
- `competition.threshold_analysis.market_size_points`
- `competition.threshold_analysis.market_size_grid` (optional explicit grid)
- `competition.threshold_analysis.run_refinement`
- `competition.threshold_analysis.refinement_tol`
- `competition.threshold_analysis.max_refinement_steps`
- `competition.threshold_analysis.tolerances.*`

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
- `results/figures/fig_comp_05_student_profit_vs_p.*`

For threshold-analysis runs (`exp_08`), key files include:

- `results/tables/competition_threshold_sweep_results.csv`
- `results/tables/competition_threshold_summary.json`
- `results/tables/competition_threshold_refinement_history.csv` (if refinement runs)
- `results/figures/threshold/fig_comp_threshold_01_strict_vs_market_size.*`
- `results/figures/threshold/fig_comp_threshold_02_weak_vs_market_size.*` (if enabled)
- `results/figures/threshold/fig_comp_threshold_03_p_star_vs_market_size.*`
- `results/figures/threshold/fig_comp_threshold_04_d_star_vs_market_size.*`
- `results/figures/threshold/fig_comp_threshold_05_teacher_payoff_vs_market_size.*`
- `results/figures/threshold/fig_comp_threshold_06_distances_vs_market_size.*`
- `results/figures/threshold/fig_comp_threshold_07_min_share_vs_market_size.*`
- `results/logs/exp_08_competition_threshold_run_log.json`
