# Distillation Tier A: Reproduction Guide

This document is a step-by-step guide to reproduce the baseline and competition experiments from this repository.

## 1) Environment

Use Python `3.10` and create the environment with one of the following methods.

### Conda

```bash
conda env create -f environment.yml
conda activate llm-econ
```

If environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate llm-econ
```

### venv + pip

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## 2) Optional Cleanup

To avoid mixing old and new outputs, clear generated artifacts before running:

```bash
python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('results/tables'), Path('results/figures'), Path('results/logs')]]"
```

## 3) Full Reproduction Order

Run from repository root in the following order:

```bash
python -m experiments.exp_10_baseline_base_hard
python -m experiments.exp_11_baseline_base_soft
python -m experiments.exp_21_baseline_sensitivity_soft_figures
python -m experiments.exp_30_competition_base_pipeline
python -m experiments.exp_45_competition_sensitivity_threshold
python -m experiments.exp_40_competition_sensitivity_u0
python -m experiments.exp_41_competition_sensitivity_tau
python -m experiments.exp_42_competition_sensitivity_regression_safeguards
python -m experiments.exp_43_competition_sensitivity_unconstrained_stability
python -m experiments.exp_44_competition_sensitivity_unconstrained_plots
```

## 4) Minimal Subset Runs

### Baseline only

```bash
python -m experiments.exp_10_baseline_base_hard
python -m experiments.exp_11_baseline_base_soft
```

### Baseline sensitivity figures

```bash
python -m experiments.exp_21_baseline_sensitivity_soft_figures
```

### Competition core + threshold

```bash
python -m experiments.exp_30_competition_base_pipeline
python -m experiments.exp_45_competition_sensitivity_threshold
```

### Competition sensitivity

```bash
python -m experiments.exp_40_competition_sensitivity_u0
python -m experiments.exp_41_competition_sensitivity_tau
```

### Regression and unconstrained-like stages

```bash
python -m experiments.exp_42_competition_sensitivity_regression_safeguards
python -m experiments.exp_43_competition_sensitivity_unconstrained_stability
python -m experiments.exp_44_competition_sensitivity_unconstrained_plots
```

## 5) Reproducibility Controls

- Default seed is `experiment.seed: 42` in `config/base.yaml`.
- `config/soft.yaml` and `config/competition.yaml` can inherit from `config/base.yaml` via `run.base_config`.
- Threshold-analysis settings are in `config/competition.yaml` under `competition.threshold_analysis.*`.

## 6) Output Artifacts

Primary output directories:

- `results/tables/`
- `results/figures/`
- `results/logs/`

Standard figure directory layout:

- `results/figures/baseline/base/`
- `results/figures/baseline/sensitivity/`
- `results/figures/baseline/sensitivity/<param>/`
- `results/figures/competition/base/`
- `results/figures/competition/sensitivity/threshold/`
- `results/figures/competition/sensitivity/u0/`
- `results/figures/competition/sensitivity/tau/`
- `results/figures/competition/sensitivity/unconstrained_like/`

Key files after a full run:

- `results/tables/baseline_demand_curve.csv`
- `results/tables/baseline_optimum.json`
- `results/tables/competition_stage5_grid_results.csv`
- `results/tables/competition_stage5_optimum.json`
- `results/tables/competition_sensitivity_threshold_sweep_results.csv`
- `results/tables/competition_sensitivity_threshold_summary.json`
- `results/tables/u0_sensitivity/u0_sensitivity_results.csv`
- `results/tables/tau_sensitivity/tau_sensitivity_results.csv`
- `results/tables/competition_stage9_regression_report.json`

## 7) Completion Checklist

- Baseline tables generated.
- Competition Stage 5 tables generated.
- Threshold sweep and summary generated.
- `u0` and `tau` sensitivity tables generated.
- Stage 9 report exists and `all_passed == true`.
