# Distillation Tier A Baseline Project

Step 1 (completed):
- `config/base.yaml`: experiment configuration (anchors + exponents + grids)
- `src/config_loader.py`: YAML loading + minimal validation
- `src/normalization.py`: anchor-based closed-form normalization

Next:
- Step 2: implement scaling-law functions (`scaling_laws.py`) using normalized coefficients
- Step 3: implement payoffs (`model.py`) and direct maximization solver for D*(p)
- Step 4: run `experiments/exp_01_baseline.py` and export figures (PDF+SVG) and tables
