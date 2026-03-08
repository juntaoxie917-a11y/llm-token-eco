# Project Context

This repository contains a Python research project for economics simulations related to LLM token trading, pricing, and market mechanisms between teacher models and student models.

The codebase is used for numerical experiments based on an economic theory model. The goal is not generic software development, but reproducible computational experiments for academic research.

# Core Principles

- Preserve the economic meaning of all variables, parameters, and equations.
- Do not arbitrarily rename core symbols used in the model unless explicitly requested.
- Do not invent new equations, mechanisms, or assumptions unless they are explicitly documented in the repository.
- When model details are ambiguous, first summarize your understanding and identify missing assumptions before generating code.
- Prefer conservative modifications that preserve the existing baseline experiment.

# Research Workflow Rules

- The repository may contain both a baseline model and one or more extended models.
- The baseline model must remain runnable and reproducible.
- Extended mechanisms should be added in a modular way instead of destructively rewriting baseline logic.
- When possible, isolate new mechanisms in new functions, classes, or modules.
- Keep backward compatibility whenever extension mechanisms are turned off.

# Modeling Rules

When writing or editing code, always distinguish clearly between:

1. Exogenous parameters
2. Endogenous choice variables
3. Intermediate quantities
4. Objective/payoff/profit functions
5. Numerical outputs used for plotting or reporting

If the repository documentation defines notation, follow it strictly.

Always preserve the direction of causality implied by the model.  
Do not convert economically meaningful definitions into ad hoc programming shortcuts without explanation.

# Numerical Experiment Rules

- All experiments should be reproducible.
- Parameters should be centralized in config files, parameter dictionaries, or clearly defined input blocks.
- Random seeds must be fixed whenever randomness is involved.
- New experiment scripts should save outputs in a structured way, including figures, tables, and parameter snapshots where appropriate.
- Do not hard-code parameters in multiple places if a shared config structure already exists.
- One script should correspond to one clear experimental purpose unless explicitly requested otherwise.

# Plotting and Output Rules

- Reuse the plotting style of the existing project when possible.
- All figures should have readable axis labels, legends, and titles when appropriate.
- Output file names should be descriptive and stable.
- Do not silently overwrite important outputs unless explicitly requested.
- If comparing baseline and extended models, make the comparison explicit in file names, legends, or saved metadata.

# Code Style Rules

- Prefer readable and modular code over compact but opaque code.
- Add comments only where they improve clarity, especially around model implementation and numerical logic.
- Avoid unnecessary abstraction if it obscures the economic interpretation.
- Use clear function boundaries for:
  - parameter setup
  - equilibrium solving / optimization
  - simulation loops
  - plotting
  - result export

# Validation Rules

Before considering a coding task complete, check the following whenever relevant:

- Does the baseline still work?
- If the extension is disabled, are old results preserved?
- Are edge cases handled safely?
- Are new parameters documented?
- Are file paths and output directories created safely?
- Are plots and saved outputs interpretable?

# Communication Rules

When responding to requests about implementation:

1. First infer the relevant model context from repository documents and existing code.
2. Briefly summarize your understanding before making major structural changes.
3. If uncertainty remains, state the uncertainty explicitly instead of guessing.
4. Prefer minimal, well-justified code changes over broad rewrites.

# Priority Documents

If these files exist, consult them first before implementing model-related changes:

- docs/model_overview.md
- docs/notation.md
- docs/assumptions.md
- docs/experiment_extension_plan.md
- README.md

If those documents conflict with older code comments, prefer the documented research description unless the user explicitly says otherwise.