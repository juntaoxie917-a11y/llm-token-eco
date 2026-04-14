# LLM Token Economics: Reproduction Guide

This repository reproduces and extends numerical experiments for a distillation-scaling-law economy with upstream token pricing and downstream competition.

## 1. Project-to-Theory Mapping

The project has two model layers:

1. Baseline: one upstream teacher and one student.
2. Competition: teacher and student compete in downstream pricing.

Main code mapping:

1. Scaling-law technology and normalization: `src/scaling_laws.py`, `src/normalization.py`
2. Baseline economics and solvers: `src/model.py`, `src/simulation.py`, `src/simulation_soft.py`
3. Competition extension: `src/competition_*.py`
4. Threshold and sensitivity logic: `src/competition_threshold.py`, `src/competition_sensitivity.py`

Recommended theory/context documents:

1. `docs/baseline_reconstruction.md`
2. `docs/competition_modified.md`
3. `docs/competition_ai_briefing.md`

## 2. Environment Setup

Recommended Python version: `3.10`.

### Conda

```bash
conda env create -f environment.yml
conda activate llm-econ
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate llm-econ
```

For strict reproducibility from the lock file: `conda env create -f environment.lock.yml`.

### venv + pip

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## 3. Optional Cleanup Before Reproduction

To avoid mixing old and new artifacts:

```bash
python -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('results/tables'), Path('results/figures'), Path('results/logs')]]"
```

## 4. Reproduction in Experiment Order

Run experiments from repository root in the following order. Each step lists purpose, command, and key outputs.

### Step 1: Baseline Hard Outside

Purpose: reproduce the hard outside-option baseline.

```bash
python -m experiments.exp_10_baseline_base_hard
```

Key outputs:

1. `results/tables/baseline_demand_curve.csv`
2. `results/tables/baseline_optimum.json`
3. `results/logs/exp_10_baseline_base_hard_run_log.json`

### Step 2: Baseline Soft Outside

Purpose: reproduce the soft outside-option baseline.

```bash
python -m experiments.exp_11_baseline_base_soft
```

Key outputs:

1. `results/tables/soft_demand_curve.csv`
2. `results/tables/soft_diagnostics.json`

### Step 3: Baseline Sensitivity Figure Families

Purpose: generate soft-mode sensitivity figure families for `alpha`, `beta`, `gamma`, `k`, `c_T`, and `tau`.

```bash
python -m experiments.exp_21_baseline_sensitivity_soft_figures
```

Key outputs:

1. `results/tables/sens_soft_curve_families_full.csv`
2. `results/tables/sens_soft_curve_families_summary.json`
3. `results/figures/baseline/sensitivity/<param>/soft_sens_<param>_<metric>.{pdf,png,svg}`

### Step 4: Competition Base Pipeline

Purpose: run the core competition pipeline and upstream price-grid equilibrium scan.

```bash
python -m experiments.exp_30_competition_base_pipeline
```

Key outputs:

1. `results/tables/competition_stage5_grid_results.csv`
2. `results/tables/competition_stage5_optimum.json`
3. `results/logs/exp_30_competition_base_pipeline_run_log.json`

### Step 5: Competition Threshold Sensitivity

Purpose: run market-size threshold sweep and refinement.

```bash
python -m experiments.exp_45_competition_sensitivity_threshold
```

Key outputs:

1. `results/tables/competition_sensitivity_threshold_sweep_results.csv`
2. `results/tables/competition_sensitivity_threshold_summary.json`
3. `results/figures/competition/sensitivity/threshold/`
4. `results/logs/exp_45_competition_sensitivity_threshold_run_log.json`

### Step 6: Competition `u0` Sensitivity

Purpose: scan the effect of `u0` under fixed or threshold-derived market size.

```bash
python -m experiments.exp_40_competition_sensitivity_u0
```

Key outputs:

1. `results/tables/u0_sensitivity/u0_sensitivity_results.csv`
2. `results/tables/u0_sensitivity/u0_sensitivity_summary.json`
3. `results/tables/u0_sensitivity/u0_sensitivity_diagnostics.json`
4. `results/logs/exp_40_competition_sensitivity_u0_run_log.json`

### Step 7: Competition `tau` Sensitivity

Purpose: scan the effect of downstream price sensitivity `tau` on equilibrium outcomes.

```bash
python -m experiments.exp_41_competition_sensitivity_tau
```

Key outputs:

1. `results/tables/tau_sensitivity/tau_sensitivity_results.csv`
2. `results/tables/tau_sensitivity/tau_sensitivity_summary.json`
3. `results/tables/tau_sensitivity/tau_sensitivity_diagnostics.json`
4. `results/logs/exp_41_competition_sensitivity_tau_run_log.json`

### Step 8: Regression Safeguards

Purpose: validate threshold interfaces, schema compatibility, and protection against sensitivity-script overwrite of threshold artifacts.

```bash
python -m experiments.exp_42_competition_sensitivity_regression_safeguards
```

Key outputs:

1. `results/tables/competition_stage9_regression_report.json`
2. `results/logs/exp_42_competition_sensitivity_regression_safeguards_run_log.json`

### Step 9: Unconstrained-like Stability

Purpose: run additional stability analysis in unconstrained-like regions.

```bash
python -m experiments.exp_43_competition_sensitivity_unconstrained_stability
```

Key outputs:

1. `results/tables/unconstrained_like/`
2. `results/logs/exp_43_competition_sensitivity_unconstrained_stability_run_log.json`

### Step 10: Unconstrained-like Plots

Purpose: generate unconstrained-like visualization outputs.

```bash
python -m experiments.exp_44_competition_sensitivity_unconstrained_plots
```

Key outputs:

1. `results/figures/competition/sensitivity/unconstrained_like/`

## 5. Optional One-Block Run

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

## 6. Reproducibility Controls

1. Default seed: `experiment.seed: 42` in `config/base.yaml`.
2. Config inheritance: `config/soft.yaml` and `config/competition.yaml` can inherit from `config/base.yaml` using `run.base_config`.
3. Threshold controls: `competition.threshold_analysis.*` in `config/competition.yaml`.

## 7. Acceptance Checklist

At minimum, confirm these files exist and are non-empty:

1. `results/tables/baseline_optimum.json`
2. `results/tables/competition_stage5_optimum.json`
3. `results/tables/competition_sensitivity_threshold_summary.json`
4. `results/tables/u0_sensitivity/u0_sensitivity_results.csv`
5. `results/tables/tau_sensitivity/tau_sensitivity_results.csv`
6. `results/tables/competition_stage9_regression_report.json`

For safeguard completion, `all_passed` in `competition_stage9_regression_report.json` should be `true`.

## 8. License

This repository uses a split-license setup:

1. Code is licensed under the Apache License 2.0. See `LICENSE`.
2. Documentation and narrative materials are licensed under CC BY 4.0. See `LICENSE-docs`.

If a file or subdirectory declares a different license, that file-level declaration takes precedence.
