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
   - The agent only touches `train.py`, `prepare.py`, `s2gae.py`, `test.py`, and config files.
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

`test.py` contains only test-time pairwise/topology evaluation helpers. It does
not own training orchestration, scorer caching, or refiner inference.

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
uv run python -m tccig.train --config path/to/config.yaml
```

The config owns experiment parameters. Do not hardcode tunable graph rules,
model choices, or runtime settings in `train.py`.

## PRING Contract

- `human_train_ppi_ratio5_exclusive.txt` builds `G_pairwise_train` from scorer
  outputs, preserving scorer probabilities as graph edge weights, and train loss
  targets from labels.
- `human_val_ppi_ratio5_exclusive.txt` builds `G_pairwise_val` from scorer
  outputs, preserving scorer probabilities as graph edge weights, and provides
  validation supervision for checkpoint selection.
- Validation topology builds a true topology graph seeded from the **train**
  node universe (`load_split_node_ids(..., split_name="train")`) with edges from
  positive rows in `human_val_ppi_ratio5_exclusive.txt`, samples PRING-style
  validation node buckets, scores every non-self pair inside those buckets, and
  selects the checkpoint from configured hard topology metrics when
  `refiner.topology_validation.enabled` is true. The hard graph rule for refined
  output remains `threshold=0.5`; per-node top-k and global top-M are not
  supported. The pairwise *input* threshold that builds `G_pairwise` is resolved
  from `graph_selection.pairwise_input_threshold`: the live config
  (`configs/tccig/01.yaml`) uses `mode: target_precision` on the validation
  split, so the threshold is data-derived. The fixed `0.5` default applies only
  when no precision target is configured.
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

## Concrete Boundaries

The public Python surface is intentionally concrete: `run_tccig_pipeline(...)`,
`score_pairs_with_v3_1(...)`, and the CLI above. Dynamic
`pairwise_scorer.target`, `refiner.train_target`, and `refiner.predict_target`
config hooks are rejected at startup.

The concrete S2GAE refiner uses weighted PyG `GraphConv` over pairwise-generated
edges, without explicit self-loops, and trains only refiner parameters. The
pairwise scorer remains a frozen scoring boundary and is never updated by the
refiner training loop.

Training uses a sampled scorer-error edge objective. Frozen scorer scores and
the frozen pairwise input threshold define `TP`, `FP`, `FN`, and `TN` train
quadrants. Each epoch keeps all `FP` and `FN` hard cases, samples `TP` and `TN`
as calibration anchors, and trains edge batches with BCE plus residual anchor.
For a batch, target edges that already exist in `G_pairwise` (`FP` and `TP`) are
temporarily removed from the encoder input graph. The encoder still sees masked
full-graph context; decoder and loss run only on the sampled edge targets.

Validation and test prediction do not use training masks. Test graph truth stays
metrics-only, and topology-test labels from `all_test_ppi.txt` stay ignored.
