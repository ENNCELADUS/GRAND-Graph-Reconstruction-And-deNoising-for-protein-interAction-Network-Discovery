# HPO + NAS-lite Workflow

This document describes how to run automatic HPO and phase-1 NAS-lite on top of the existing GRAND pipeline.

## Scope

- Optuna is the only optimization backend in this repository.
- Existing `src.run` pipeline stage execution remains config-driven.
- NAS-lite phase-1 is architecture-parameter search (not full neural architecture search graph mutation).

## New Modules

- `src/optimize/run.py` - optimization entrypoint.
- `src/optimize/search_space.py` - search-space parsing and dot-path patching.
- `src/optimize/trial_runner.py` - trial execution + objective extraction from `training_step.csv`.
- `src/optimize/backends/optuna_backend.py` - Optuna backend.

## Config Contract

Use `configs/v5/v5_hpo.yaml` as the reference. New top-level sections:

- `optimization`: backend/budget/sampler/pruner/search space.
- `nas_lite`: phase-1 architecture search controls.

Key defaults:

- `optimization.backend = optuna`
- `optimization.execution.trial_stages = ["train"]`
- `optimization.execution.ddp_per_trial = false`
- `optimization.execution.catch_oom_as_pruned = true`

## Run Commands

```bash
# Optuna mainline
./scripts/hpo_optuna.sh configs/v5/v5_hpo.yaml
```

## Artifacts

For study `<study_name>`, artifacts are written under:

- `artifacts/hpo/<study_name>/trials.csv`
- `artifacts/hpo/<study_name>/best_params.yaml`

Per-trial logs/checkpoints reuse existing pipeline contracts:

- `logs/<model>/train/<run_id>/training_step.csv`
- `models/<model>/train/<run_id>/best_model.pth`

## Objective Definition

- Objective metric is read from `training_step.csv` column `Val <metric>`.
- `optimization.objective_metric` accepts `val_auprc` or `auprc` style naming.
- Direction is controlled by `optimization.direction` (`maximize`/`minimize`).

## Optional Dependencies

```bash
# In grand environment
pip install optuna
```

If a backend package is missing, the optimizer raises an actionable import error.

## Notes for HPC

- The optimizer itself is single-process by default (`ddp_per_trial=false`).
- Keep trial-level GPU usage bounded before scaling trial count.
- Start with the lightweight budget in `configs/v5/v5_hpo.yaml` and expand gradually.
