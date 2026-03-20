# Baseline Sensitivity (alpha, beta, gamma, k, c_T, tau-soft-only) - AI Execution Steps

## 0) Scope Lock
1. Analyze exactly 6 parameters: alpha, beta, gamma, k, c_T, tau.
2. Run two modes:
   - hard mode: alpha, beta, gamma, k, c_T (tau excluded)
   - soft mode: alpha, beta, gamma, k, c_T, tau
3. Reuse existing baseline equations and solvers; do not change core model logic.
4. Keep implementation minimal, deterministic, and batch runnable.

## 1) Reuse Targets
1. Config loader: src/config_loader.py
2. Technology builder: src/scaling_laws.py
3. Hard simulation: src/simulation.py
4. Soft simulation: src/simulation_soft.py
5. Reference run scripts: experiments/exp_01_hard_outside.py and experiments/exp_02_soft_outside.py

## 2) New Files
1. experiments/exp_03_sensitivity_core_and_soft.py
2. results/tables/sens_hard_oat.csv
3. results/tables/sens_hard_sobol.csv
4. results/tables/sens_soft_oat.csv
5. results/tables/sens_soft_sobol.csv
6. results/tables/sens_core_soft_summary.json

## 3) Parameter Sets and Ranges
Baseline values:
1. hard baseline from config/base.yaml
2. soft baseline from config/soft.yaml

Range rules (multiplicative around baseline):
1. alpha, beta, gamma in [0.8*x0, 1.2*x0]
2. k, c_T in [0.8*x0, 1.2*x0]
3. tau (soft only) in [0.5*tau0, 2.0*tau0]

Clamp rules:
1. all lower bounds must be > 0
2. tau lower bound >= 1e-4
3. do not change non-target parameters

## 4) Metrics To Collect Per Run
Hard mode:
1. p_star
2. D_star_at_p_star
3. pi_teacher_star
4. optout_share
5. boundary_share

Soft mode:
1. p_star
2. D_soft_at_p_star
3. pi_teacher_star
4. avg_enter_prob

## 5) Stage A - OAT (One-At-A-Time)
Hard OAT:
1. Use 9-point linear grid per parameter for alpha, beta, gamma, k, c_T.
2. For each parameter v and each grid point:
   - deep-copy baseline hard config
   - set only v to current value
   - run hard simulation
   - write one row with mode=hard, var, value, metrics
3. Save all rows to results/tables/sens_hard_oat.csv.

Soft OAT:
1. Use 9-point linear grid per parameter for alpha, beta, gamma, k, c_T, tau.
2. For each parameter v and each grid point:
   - deep-copy baseline soft config
   - set only v to current value
   - run soft simulation
   - write one row with mode=soft, var, value, metrics
3. Save all rows to results/tables/sens_soft_oat.csv.

## 6) Stage B - Sobol Global Sensitivity
Hard Sobol:
1. SALib problem:
   - num_vars=5
   - names=[alpha, beta, gamma, k, c_T]
   - bounds from Section 3
2. Generate Sobol samples with N=256 and calc_second_order=False.
3. For each sample:
   - deep-copy hard config
   - assign sampled values to 5 vars
   - run hard simulation
   - Y = pi_teacher_star
4. Compute S1 and ST; save to results/tables/sens_hard_sobol.csv with columns var,S1,ST.

Soft Sobol:
1. SALib problem:
   - num_vars=6
   - names=[alpha, beta, gamma, k, c_T, tau]
   - bounds from Section 3
2. Generate Sobol samples with N=256 and calc_second_order=False.
3. For each sample:
   - deep-copy soft config
   - assign sampled values to 6 vars
   - run soft simulation
   - Y = pi_teacher_star
4. Compute S1 and ST; save to results/tables/sens_soft_sobol.csv with columns var,S1,ST.

## 7) Stage C - Unified Summary JSON
Create results/tables/sens_core_soft_summary.json with keys:
1. baselines:
   - hard: alpha0,beta0,gamma0,k0,c_T0
   - soft: alpha0,beta0,gamma0,k0,c_T0,tau0
2. ranges:
   - hard bounds for 5 vars
   - soft bounds for 6 vars
3. oat:
   - hard top vars by absolute relative effect on pi_teacher_star
   - soft top vars by absolute relative effect on pi_teacher_star
4. sobol:
   - hard S1/ST table and top var by ST
   - soft S1/ST table and top var by ST
5. run_meta:
   - timestamp
   - oat_rows_hard
   - oat_rows_soft
   - sobol_samples_hard
   - sobol_samples_soft

## 8) Implementation Rules
1. Use deep copy of config per run; never mutate shared config object.
2. Set numpy random seed once from config experiment seed.
3. On run failure, record NaN metrics and continue.
4. Keep code in one new experiment script; do not edit model equations.
5. Ensure SALib import is optional-fail-fast with clear install message.

## 9) Validation Gates
1. Hard OAT rows = 5 * 9 = 45.
2. Soft OAT rows = 6 * 9 = 54.
3. Hard Sobol rows = 5.
4. Soft Sobol rows = 6.
5. Summary JSON exists with all keys in Section 7.
6. All output files exist and are non-empty.

## 10) Final Print Contract
After completion, print only:
1. output file paths
2. hard top-1 by OAT absolute effect
3. hard top-1 by Sobol ST
4. soft top-1 by OAT absolute effect
5. soft top-1 by Sobol ST
