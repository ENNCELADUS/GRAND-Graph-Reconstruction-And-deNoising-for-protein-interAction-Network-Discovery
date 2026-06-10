# TCCIG PRING IO and Orchestrator

This directory contains the standalone TCCIG scaffold for the PRING-aligned
pipeline contract. It is intentionally limited to data IO, pairwise-score
orchestration, graph decision rules, metrics, and fakeable model boundaries,
plus the concrete S2GAE residual denoiser used by the current TCCIG path.

## Sandbox Requirements

This directory is a narrow implementation sandbox for the TCCIG replacement path.
Keep it small, reviewable, and tied to the GRAND repository environment.

## Hard Rules

1. Keep the editable surface tiny.
   - The agent only touches `train.py`, `s2gae.py`, and config files.
   - Do not spread experimental logic across extra modules unless the code is no longer reviewable in one file.
   - Do not modify `prepare.py` unless the data or metric contract is intentionally being changed.

2. Keep the path self-contained.
   - One training entrypoint.
   - One primary metric.
   - No separate package, no independent environment.

## Design Choices

### Small Concrete Refiner Module

The orchestrator stays in `train.py`. The concrete S2GAE residual denoiser lives
in `s2gae.py` so model, feature, and optimization code stays reviewable without
turning the orchestrator into the training implementation.

### Self-Contained

This path should not become a second project inside GRAND. It should depend on
the root `pyproject.toml`, the root `uv` workflow, PyTorch, and only the small
set of dependencies already accepted by the main repository.

The S2GAE refiner intentionally depends on PyG and its Torch-version-specific
extension wheels. Use `uv sync --group dev --find-links
https://data.pyg.org/whl/torch-2.10.0+cpu.html` for local CPU setup, and the
CUDA 12.8 wheel page `https://data.pyg.org/whl/torch-2.10.0+cu128.html` on HPC.

## Public Entry Point

```bash
uv run python tccig/train.py --config path/to/config.yaml
```

The config owns experiment parameters. Do not hardcode tunable graph rules,
model choices, or runtime settings in `train.py`.

## PRING Contract

- `human_train_ppi_ratio5_exclusive.txt` builds `G_pairwise_train` from scorer
  outputs, preserving scorer probabilities as graph edge weights, and train loss
  targets from labels.
- `human_val_ppi_ratio5_exclusive.txt` builds `G_pairwise_val` from scorer
  outputs, preserving scorer probabilities as graph edge weights, and provides
  validation supervision for checkpoint/calibration selection.
- Validation topology builds a true topology graph from positive rows in
  `human_val_ppi_ratio5_exclusive.txt`, samples PRING-style validation node
  buckets, scores every non-self pair inside those buckets, and selects the
  checkpoint plus one global logit-bias calibration from configured hard
  topology metrics when `refiner.topology_validation.enabled` is true. The
  hard graph rule remains `threshold=0.5`; per-node top-k and global top-M are
  not supported.
- `human_test_ppi.txt` is the binary pairwise test set. The raw frozen v3.1
  scorer baseline is a pinned historical artifact under
  `logs/tccig/pairwise_baseline`; the pipeline does not regenerate it. The
  full v3.1-plus-refiner pairwise metrics are written to
  `logs/tccig/{run_id}/pairwise_test`.
- `all_test_ppi.txt` is the topology candidate universe; its labels are ignored.
- `human_test_graph.pkl` and `test_sampled_nodes.pkl` are loaded only after
  topology predictions exist, for metrics only.

Self-pair rows are filtered from every split before scoring or graph
construction. Dropped counts are persisted in the run manifest.

## Hook Boundaries

The pairwise scorer hook receives label-free candidate pairs only. The refiner
training/prediction hooks receive pairwise-generated graph inputs and may see
train/validation targets only where the PRING contract allows them.

The concrete S2GAE refiner uses weighted PyG `GraphConv` over pairwise-generated
edges, without explicit self-loops, and trains only refiner parameters with
fixed-LR AdamW. It prepares the refiner model and optimizer through the
configured accelerator, then routes backward through `accelerator.backward`. The
pairwise scorer remains a frozen scoring boundary and is never updated by the
refiner training loop.

If refiner hooks are omitted, the scaffold raises `NotImplementedError`; this is
intentional until concrete model and training code are added.
