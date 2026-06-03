# TCCIG PRING IO and Orchestrator

This directory contains the standalone TCCIG scaffold for the PRING-aligned
pipeline contract. It is intentionally limited to data IO, pairwise-score
orchestration, graph decision rules, metrics, and fakeable model boundaries.

It does not implement the concrete pairwise classifier, refiner model, or
training loop.

## Sandbox Requirements

This directory is a narrow implementation sandbox for the TCCIG replacement path.
Keep it small, reviewable, and tied to the GRAND repository environment.

## Hard Rules

1. Keep the editable surface tiny.
   - The agent only touches `train.py` and config files.
   - Do not spread experimental logic across extra modules unless the code is no longer reviewable in one file.
   - Do not modify `prepare.py` unless the data or metric contract is intentionally being changed.

2. Keep the path self-contained.
   - One training entrypoint.
   - One primary metric.
   - No separate package, no independent environment.

## Design Choices

### Single File to Modify

The agent only touches `train.py` and config files. This keeps the scope
manageable and diffs reviewable.

### Self-Contained

This path should not become a second project inside GRAND. It should depend on
the root `pyproject.toml`, the root `uv` workflow, PyTorch, and only the small
set of dependencies already accepted by the main repository.

## Public Entry Point

```bash
uv run python tccig/train.py --config path/to/config.yaml
```

The config owns experiment parameters. Do not hardcode tunable graph rules,
model choices, or runtime settings in `train.py`.

Required config surfaces:

```yaml
run:
  run_id: example
  log_root: logs

data:
  processed_dir: data/PRING/human/BFS

device:
  device: cpu
  backend: ddp
  mixed_precision: false

pairwise_scorer:
  target: some.module:score_pairs

refiner:
  train_target: some.module:train_refiner
  predict_target: some.module:predict_refined

graph_selection:
  rules:
    - type: threshold
      value: 0.5
    - type: top_k
      k: 10
    - type: top_m
      m: 32019
```

## PRING Contract

- `human_train_ppi_ratio5_exclusive.txt` builds `G_pairwise_train` from scorer
  outputs and train loss targets from labels.
- `human_val_ppi_ratio5_exclusive.txt` builds `G_pairwise_val` from scorer
  outputs and selects checkpoint/rule from validation labels.
- `human_test_ppi.txt` is only for ordinary pairwise metrics.
- `all_test_ppi.txt` is the topology candidate universe; its labels are ignored.
- `human_test_graph.pkl` and `test_sampled_nodes.pkl` are loaded only after
  topology predictions exist, for metrics only.

Self-pair rows are filtered from every split before scoring or graph
construction. Dropped counts are persisted in the run manifest.

## Hook Boundaries

The pairwise scorer hook receives label-free candidate pairs only. The refiner
training/prediction hooks receive pairwise-generated graph inputs and may see
train/validation targets only where the PRING contract allows them.

If refiner hooks are omitted, the scaffold raises `NotImplementedError`; this is
intentional until concrete model and training code are added.
