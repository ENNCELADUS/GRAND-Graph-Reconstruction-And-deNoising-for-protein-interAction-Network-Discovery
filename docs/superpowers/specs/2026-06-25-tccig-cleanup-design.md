# TCCIG Pipeline Cleanup & Accelerate Orchestration — Design

**Date:** 2026-06-25
**Status:** Approved (design), pending implementation plan
**Scope:** `tccig/` (train.py, s2gae.py, prepare.py, test.py), `configs/tccig/01.yaml`, `tccig/README.md`, related tests.

## Goal

Simplify the existing TCCIG PPI refiner pipeline without changing the intended
modeling method, with one intentional behavioral change: make the residual
anchor a real regularizer instead of an inert term. Concretely:

1. Remove dead code and dead/bookkeeping CSV/JSON columns.
2. Hand DDP orchestration to Hugging Face Accelerate; delete hand-rolled
   sharding and `getattr`-guarded distributed shims.
3. Fix one reporting bug (`best_validation_auprc`).
4. Make the residual anchor honest (`residual_weight` 1e-8 → 1e-3).
5. Fix two design-vs-code documentation mismatches in the README.

This is **not** a rewrite. The frozen-scorer boundary, residual identity-init,
FP/FN/TP/TN sampled-edge objective, link-prediction masking, topology
validation, and test protocol are all preserved.

## Non-Goals

- No change to the S2GAE architecture, encoder/decoder, or loss structure
  (beyond the residual weight value).
- No change to the train/val/test data contract or PRING IO.
- No new features. YAGNI applies throughout.
- No unrelated refactoring of `src/`.

## Background: Verified Findings

These were confirmed against the working tree during code review:

- `best_validation_auprc` is updated with `max(...)` every epoch
  (`s2gae.py:637`), decoupled from checkpoint selection
  (`s2gae.py:728-740`, monitor = `val_topology_loss`). The AUPRC stored in the
  checkpoint and `training_summary.json` can come from a different epoch than
  the saved weights.
- `train_topology_loss`, `train_graph_similarity_loss`,
  `train_relative_density_loss` are hardcoded `0.0` (`s2gae.py:678-680`) and
  written to CSV/JSON every epoch.
- Manual DDP surface: `_prediction_probabilities` (`s2gae.py:1015`) does manual
  strided sharding via `_rank_local_pair_indices`, then reassembles order with
  index-tagged rows; `_accelerator_reduce_sum`, `_ordered_values_from_accelerate_rows`,
  `_clip_grad_norm_with_accelerator`, `_runtime_is_distributed/_rank/_world_size`,
  and the `save_fn = getattr(...)` dance all hand-roll or guard what Accelerate
  exposes directly.
- `topology_loss` config block is forced off: `_parse_topology_loss_config`
  raises if `enabled: true` (`s2gae.py:1753`).
- Multi-rule machinery: `request.graph_rules` is a tuple but
  `_fixed_threshold_rule` (`s2gae.py:1203`) always returns `rules[0]`; README
  forbids top-k/top-M.
- `residual_weight: 1.0e-8` (`01.yaml:53`) is inert against BCE of order 1e-1.
- README implies a 0.5 input threshold default, but the live config uses
  `mode: target_precision, target_precision: 0.8` (`01.yaml:86-90`).
- The validation topology graph is seeded from the **train** node universe
  (`train.py:391-396`, `load_split_node_ids(..., split_name="train")`) plus
  val-positive edges; the README omits this.

## Workstreams

### 1. Dead code removal

- **`topology_loss`**: remove `S2GAEConfig.topology_loss`,
  `S2GAETopologyLossConfig`, `_parse_topology_loss_config`, and the
  `topology_loss` echo in `_config_to_json` (`s2gae.py:1606-1614`). Remove the
  `01.yaml:54-65` block. The genuinely-used `topology_validation` block is
  untouched.
- **Multi-rule machinery**: collapse `request.graph_rules` (tuple) to a single
  `GraphRule`. Remove `_fixed_threshold_rule`. `parse_rules` in `train.py`
  continues to validate the configured `graph_selection.rules` but resolves to
  one threshold rule.
- **Unreachable code**: remove the `ValueError`s in
  `_edge_index_and_weight_from_edges` (`s2gae.py:950-953`) — graph edges are
  always a subset of pair edges.
- **Redundant summary write**: drop the post-loop `_write_training_summary`
  call (`s2gae.py:773`); keep the per-epoch one (`s2gae.py:742`). The final
  epoch already writes final state.

### 2. CSV/JSON column trim

`TCCIG_TRAIN_CSV_COLUMNS` goes from 26 to 19 columns. Remove:

- Dead (always `0.0`): `Train Topology Loss`, `Train GS Loss`, `Train RD Loss`
- DDP bookkeeping: `Local Train Pairs`, `Global Train Pairs`,
  `Local Validation Pairs`, `Global Validation Pairs`

Keep: `Epoch`, `Epoch Time`, `Train Loss`, `Train BCE Loss`,
`Train Residual Anchor Loss`, `Train Weighted Residual Anchor Loss`,
`Train Gradient Norm`, `Val auprc`, `Val Topology Loss`,
`Internal Val graph_sim`, `Internal Val relative_density`,
`Internal Val deg_dist_mmd`, `Internal Val cc_mmd`, `Selected Rule Type`,
`Selected Rule Positive Edges`, `Monitor Metric`, `Monitor Value`,
`Peak GPU Mem MB`, `Learning Rate`.

The matching keys in the per-epoch `epoch_history` dict and the JSON history
are dropped in lockstep (`s2gae.py:670-698`, the `_append_tccig_train_csv_row`
and `_log_epoch_summary` formatters). This is a log-format change; no in-repo
downstream parser of `tccig_train_step.csv` exists.

### 3. Accelerate orchestration

Let Accelerate own sharding, gather, and reduce; delete the hand-rolled DDP
layer.

- **Prediction path** (`_prediction_probabilities`): replace manual strided
  sharding (`_rank_local_pair_indices`) + index-tagged row reassembly with an
  Accelerate-prepared `DataLoader` over pair indices plus `gather_for_metrics`.
  The encoder still runs once on the full graph per rank; only the decoder
  batching and the cross-rank gather change. Preserve the invariant that every
  pair index is covered exactly once via a defensive assertion after gather
  (Accelerate `even_batches` duplicates tail samples; `gather_for_metrics`
  drops them — assertion guards regressions).
- **Drop shims**:
  - `_accelerator_reduce_sum` → `accelerator.reduce(tensor, reduction="sum")`.
  - `_clip_grad_norm_with_accelerator` → `accelerator.clip_grad_norm_(...)`.
  - Remove `_runtime_is_distributed`, `_runtime_rank`, `_runtime_world_size`,
    `_rank_local_pair_indices`, `_rank_local_pair_count`, and the
    `save_fn = getattr(...)` fallback (use `accelerator.save` directly).
- **Single shared gather helper**: if both `train.py` and `s2gae.py` still need
  ordered gather after the above, consolidate into one helper parameterized on
  duplicate-tolerance rather than two near-duplicates.

### 4. Reporting fix

Move the `best_validation_auprc` capture into the `_is_better_monitor` block
(`s2gae.py:728`) so the reported AUPRC corresponds to the checkpointed epoch.
When monitor metric is AUPRC the value is unchanged; when it is
`val_topology_loss` it now reports the selected epoch's AUPRC instead of the
global max.

### 5. Residual anchor + documentation

- **Residual anchor**: `01.yaml` `residual_weight: 1.0e-8` → `1.0e-3`. The
  parser default (`s2gae.py:1562`) is already `0.001`; this aligns the live
  config. The documented "BCE + residual anchor" objective becomes real.
- **README**: state that the pairwise-input threshold is resolved by the live
  config via `target_precision` (the 0.5 default applies only when no precision
  target is configured), and that the validation topology graph is built over
  the **train** node universe with val-positive edges.

## Testing & Verification

- Update tests coupled to removed internals:
  `tests/unit/test_tccig_s2gae.py` (`_rank_local_pair_indices`,
  `_ordered_values_from_accelerate_rows`, `_prediction_probabilities` tests
  retargeted to the new gather path; `test_parse_config_rejects_train_topology_loss_enabled`
  updated/removed). Align `NoOpAccelerator` in `tests/runtime_helpers.py` so
  single-process behavior is identity.
- Run:
  - `uv run python -m pytest tests/unit/test_tccig_s2gae.py tests/unit/test_tccig_s2gae_validation.py tests/unit/test_tccig_prepare.py tests/unit/test_tccig_rules.py tests/unit/test_tccig_pairwise_scorer.py tests/integration/test_tccig_orchestrator.py`
  - `uv run ruff check tccig tests`
  - `uv run mypy tccig src`
- Distributed correctness is verified by single-process equivalence plus the
  post-gather coverage assertion. Multi-GPU is not exercised in CI.

## Risks

- **Highest risk: prediction gather rewrite.** Distributed gather correctness is
  subtle and only fully exercised on multi-GPU. Mitigated by the
  exactly-once coverage assertion and single-process equivalence tests.
- **Log-format change.** CSV/JSON column removal breaks any external consumer of
  `tccig_train_step.csv`; none found in-repo.
- **Residual weight change** alters training dynamics; prior `1e-8` runs are not
  comparable to new `1e-3` runs. This is intentional and approved.

## Out of Scope / Deferred

- Hoisting the per-split scorer reload onto a warm path.
- Any change to `src/` distributed or metric code.
