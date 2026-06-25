# TCCIG Refiner Stabilization + Raw/Refined Export

## Context

Run `tccig_01_pairwise_debug_20260626` produced a refined output that is
near-random (AUPRC 0.49, 85% of rows exactly 0.0) while the frozen raw scorer
feeding it is strong (AUPRC 0.79). Root cause is confirmed from the artifacts
and the code, and it is **not** primarily checkpoint selection — the refiner
training is numerically unstable end-to-end.

### Confirmed explosion chain (from `training_summary.json` + code)

1. `GraphConv` is loaded as vanilla PyG `GraphConv` (`tccig/s2gae.py:1470`), whose
   default aggregation is `aggr="add"`. On the training graph the selected rule
   produced ~1.1M positive edges in epoch 1 (`selected_rule_positive_edges`), so
   each node sums over hundreds of neighbors → hidden-state magnitude explodes.
2. Node features are raw mean-pooled ESM3 embeddings, unnormalized
   (`load_mean_pooled_node_features`, `s2gae.py:467`), so the first conv already
   starts large.
3. `CrossLayerDecoder` takes elementwise **products** of hidden states
   (`src_values * dst_hidden`, `s2gae.py:319`) → squares the magnitude.
4. `residual_refined_logits` adds that delta straight onto the clamped pairwise
   logit (`s2gae.py:337`). Epoch-1 `train_residual_anchor_loss = 7.3e14` ⟹ delta
   RMS ≈ 2.7e7, which completely swamps the raw logit → sigmoid saturates to
   0/1 → the 85%-zeros CSV.
5. `residual_weight=1e-3` is far too weak to anchor delta; `gradient_clip_norm=1.0`
   caps step size but not the loss landscape. Train loss never settles
   (`1e5`–`1e11` across all 40 epochs), so neither cls nor topo loss decreases
   monotonically.

### User decisions (this pass)

- Scope: **Rec 1 + Rec 2 + stabilize refiner.**
- Priority: **training stability** — cls loss and topo loss should decrease.
  Checkpoint-selection mechanism is explicitly deprioritized ("don't care how
  the ckp is selected now").

## Goals

1. Make refiner training numerically stable: bounded hidden states and bounded
   residual so train BCE and `val_topology_loss` trend down over epochs.
2. Export raw + refined probabilities and separate metrics so the next run is
   diagnosable without re-deriving from the `.pt` cache (Rec 1).
3. Rec 2 (checkpoint monitor) is intentionally NOT changed — keep
   `val_topology_loss` per the committed experiment design (see section D).

## Design

### A. Stabilize the encoder (degree-invariant + normalized)

All new behavior is config-gated with safe defaults so existing tests and saved
checkpoints stay valid.

- **Aggregation**: pass `aggr` to `GraphConv`. New `refiner.encoder_aggr`
  (default `"mean"`). Mean aggregation removes the degree-driven blowup at the
  source. `S2GAERefiner.__init__` and `_load_graph_conv` usage updated to thread
  `aggr` through.
- **Input feature normalization**: optional `nn.LayerNorm(input_dim)` applied to
  node features before the first conv. New `refiner.feature_norm` (default
  `true`). Standardizes raw ESM3 scale.
- **Hidden normalization**: optional `nn.LayerNorm(hidden_dim)` after each conv
  (before relu/dropout). New `refiner.hidden_norm` (default `true`). Keeps
  per-layer magnitude bounded in the 2-layer encoder.

### B. Bound the residual (prevents logit swamp regardless of encoder)

- `CrossLayerDecoder` / `residual_refined_logits` gain an optional
  `residual_scale: float | None`. When set, the applied residual is
  `residual_scale * tanh(raw_delta)`, so `|residual| <= residual_scale` and can
  never swamp the raw logit (raw logits clamp to ±13.8 today).
- New `refiner.residual_scale` (default `None` = current unbounded behavior, so
  existing unit tests calling `residual_refined_logits(p, delta)` are unchanged).
  `config_01` (next run) sets it (proposed `4.0`).
- The residual anchor loss is computed on the **applied** (post-scale) residual
  so `train_residual_anchor_loss` stays interpretable and bounded. `decode`
  returns the applied residual for the anchor.

### C. Rec 1 — export raw + refined

In `tccig/test.py`:
- `run_pairwise_test`: add `raw_probability` column to
  `human_test_ppi_pred.csv`; write `raw_metrics.json` and `refined_metrics.json`
  alongside the existing `pairwise_metrics.json` (kept for back-compat = refined).
- `_write_pairwise_predictions` extended to take both raw and refined scores.
- `_binary_metrics` reused for both.
- Raw metrics use the scorer's natural `0.5` decision threshold; refined metrics
  use the configured `refined_output_rule` threshold. (`run_topology_test` is
  intentionally left refined-only — topology metrics are graph-structure metrics,
  not a raw-vs-refined binary comparison, so a raw/refined split does not apply.)

### D. Rec 2 — checkpoint-selection change: NOT done

- Deliberately dropped. The committed `EXPERIMENT_PLAN.md` design keeps
  `monitor_metric: val_topology_loss` for the C1 topology claim and lists
  "switch to a hard topology metric" only as a fallback if the monitor decouples
  from test topology. Per the user's call, `01.yaml` keeps `val_topology_loss`
  unchanged. The selection machinery still supports `val_auprc`
  (`_resolve_monitor_value`, `s2gae.py`) if a later run wants it.

## Files

- `tccig/s2gae.py` — encoder aggregation/normalization, bounded residual, config
  parsing (`_parse_config`, `_config_to_json`), `S2GAEConfig` fields.
- `tccig/test.py` — raw/refined export.
- `configs/tccig/01.yaml` — stabilization knobs (`encoder_aggr`, `layer_norm`,
  `residual_scale`); monitor left as `val_topology_loss`.
- `tests/unit/test_tccig_s2gae.py` — new stability tests.
- `tests/unit/test_tccig_test_export.py` — raw/refined export tests.

## TDD plan (write failing tests first)

1. **Bounded residual**: with `residual_scale=4.0`, `residual_refined_logits`
   output stays within `logit_clamp ± 4.0` for extreme delta inputs (e.g. 1e7).
2. **Degree invariance / bounded encoder**: build a synthetic dense graph (one
   hub node connected to many), assert encoder hidden-state max-abs stays
   `O(1)`–`O(10)` with `aggr="mean"` + norms, vs. exploding with the old path.
3. **Loss decreases**: train `_S2GAESampledTrainStepModule` for ~30 steps on a
   small synthetic dense graph with the stabilized config; assert final BCE <
   initial BCE and no NaN/Inf.
4. **Export shape**: `run_pairwise_test` writes `raw_probability` +
   `refined_probability` columns and both metrics JSONs; raw metrics match
   `_binary_metrics` on the raw scores.
5. **Config round-trip**: new knobs parse with defaults and via `_config_to_json`.

## Verification

- `uv run python -m pytest tests/unit/test_tccig_s2gae.py tests/unit/test_tccig_s2gae_validation.py`
- `uv run ruff check . && uv run ruff format --check . && uv run mypy src tccig`
- Local env is CPU PyG; full HPC training is out of scope. Stability is verified
  via the synthetic-graph loss-decrease test, not a real run.

## Out of scope

- Any AUPRC-floor guard or composite monitor (deprioritized by user).
- Re-running on HPC / new run id orchestration (Rec 3 is just hygiene; the
  next run should use a fresh run id, noted for the operator).
- Touching the frozen pairwise scorer (confirmed correct, AUPRC 0.79).
