# HPO Workflow

This document describes how to run automatic HPO and phase-1 NAS-lite on top of the existing GRAND pipeline.

## Scope

- Optuna is the only optimization backend in this repository.
- `optuna` is a normal project dependency in `pyproject.toml`; use `uv sync --group dev` rather than installing it separately.
- Existing `src.run` pipeline stage execution remains config-driven.
- NAS-lite phase-1 is architecture-parameter search (not full neural architecture search graph mutation).

## New Modules

- `src/optimize/run.py` - optimization entrypoint.
- `src/optimize/search_space.py` - search-space parsing and dot-path patching.
- `src/optimize/trial_runner.py` - trial execution + objective extraction from `training_step.csv`.
- `src/optimize/backends/optuna_backend.py` - Optuna backend.

## Config Contract

Reference configs with active optimization sections include `configs/v3/v3_20260411_030200.yaml` and `configs/v3-1/v3.1.yaml`. Top-level sections:

- `optimization`: backend/budget/sampler/pruner/search space.
- `nas_lite`: phase-1 architecture search controls.

Key defaults:

- `optimization.backend = optuna`
- `optimization.execution.trial_stages = ["train"]`
- `optimization.execution.ddp_per_trial = false` for single-process trials; set it to `true` when launching optimization through `torch.distributed.run`.
- `optimization.execution.catch_oom_as_pruned = true`
- `optimization.recheck.enabled = true` optionally re-runs top-k trials across multiple seeds and writes stability-aware best params.

## Run Commands

```bash
# Direct local run
uv run python -m src.optimize.run --config configs/v3-1/v3.1.yaml

# HPC launcher scripts auto-dispatch to src.optimize.run when optimization.enabled=true
sbatch scripts/v3.sh configs/v3/v3_20260411_030200.yaml
sbatch scripts/v3_1_ablation.sh configs/v3-1/v3.1.yaml
```

## Artifacts

For study `<study_name>`, artifacts are written under:

- `artifacts/hpo/<study_name>/trials.csv`
- `artifacts/hpo/<study_name>/best_params.yaml`
- `artifacts/hpo/<study_name>/recheck_summary.csv` when `optimization.recheck.enabled=true`
- `artifacts/hpo/<study_name>/recheck_trials.csv` when `optimization.recheck.enabled=true`
- `artifacts/hpo/<study_name>/rechecked_best_params.yaml` when `optimization.recheck.enabled=true`

Per-trial logs/checkpoints reuse existing pipeline contracts:

- `logs/<model>/train/<run_id>/training_step.csv`
- `models/<model>/train/<run_id>/best_model.pth`

## Objective Definition

- Objective metric is read from `training_step.csv` column `Val <metric>`.
- `optimization.objective_metric` accepts `val_auprc` or `auprc` style naming.
- Direction is controlled by `optimization.direction` (`maximize`/`minimize`).

## Notes for HPC

- The optimizer is single-process by default. Multi-process optimization requires `optimization.execution.ddp_per_trial=true`; otherwise `src.optimize.run` raises before launching distributed trials.
- Launcher scripts use `uv run --locked --no-sync --offline` and expect `.venv/` to exist before `sbatch`.
- Keep trial-level GPU usage bounded before scaling trial count.
