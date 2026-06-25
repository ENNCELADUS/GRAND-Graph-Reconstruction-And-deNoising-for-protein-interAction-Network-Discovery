# TCCIG Pipeline Cleanup & Accelerate Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the existing TCCIG PPI refiner pipeline — remove dead code and dead/bookkeeping log columns, hand DDP orchestration to Hugging Face Accelerate, fix the `best_validation_auprc` reporting bug, make the residual anchor a real `1e-3` regularizer, and correct two README mismatches — without changing the intended modeling method.

**Architecture:** The pipeline is three editable modules: `tccig/train.py` (orchestrator: scoring, threshold resolution, bundle construction, CLI), `tccig/s2gae.py` (S2GAE refiner model, training loop, validation, prediction, checkpointing, config parsing), and `tccig/prepare.py` (datasets, requests, edge sampling). Cleanup proceeds in dependency order: leaf-level dead code first (so later tasks touch smaller surfaces), then the Accelerate rewrite of the prediction/gather/reduce path, then the reporting fix, then config + docs.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric (`GraphConv`), Hugging Face Accelerate, `uv` for env/tooling, `pytest`, `ruff`, `mypy` (strict).

## Global Constraints

- Run all Python/tests/lint via `uv run` (e.g. `uv run python -m pytest`). Bootstrap with `uv sync --group dev` if `.venv/` is missing/stale.
- Absolute imports only (`from src.x import y`, `from tccig.x import y`). No `print` (use `logging`). No hardcoded tunables (use config). Strict type hints, avoid `Any`. Max nesting 4. Functions < 50 lines. Files target 200–400 lines (max 600).
- Conventional Commits: `<type>: <description>` (`feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`).
- Do NOT change the S2GAE architecture, encoder/decoder, loss structure (beyond the residual weight value), the train/val/test data contract, PRING IO, or anything in `src/`.
- `residual_weight` final value: **`1.0e-3`** (config) — the parser default is already `0.001`.
- TCCIG train CSV/JSON keeps exactly these 19 columns (drop the other 7): `Epoch`, `Epoch Time`, `Train Loss`, `Train BCE Loss`, `Train Residual Anchor Loss`, `Train Weighted Residual Anchor Loss`, `Train Gradient Norm`, `Val auprc`, `Val Topology Loss`, `Internal Val graph_sim`, `Internal Val relative_density`, `Internal Val deg_dist_mmd`, `Internal Val cc_mmd`, `Selected Rule Type`, `Selected Rule Positive Edges`, `Monitor Metric`, `Monitor Value`, `Peak GPU Mem MB`, `Learning Rate`.
- Verification commands (run after each task and at the end):
  - `uv run python -m pytest tests/unit/test_tccig_s2gae.py tests/unit/test_tccig_s2gae_validation.py tests/unit/test_tccig_prepare.py tests/unit/test_tccig_rules.py tests/unit/test_tccig_pairwise_scorer.py tests/integration/test_tccig_orchestrator.py -v`
  - `uv run ruff check tccig tests`
  - `uv run mypy tccig src`
- The integration test `tests/integration/test_tccig_orchestrator.py` runs a full single-process CPU pipeline. It is the primary single-process equivalence guard for the Accelerate rewrite — it must stay green.

---

## Spec reference

Full design: `docs/superpowers/specs/2026-06-25-tccig-cleanup-design.md`. Tasks below map to spec workstreams 1–5.

---

### Task 1: Remove the dead `topology_loss` config block

Removes `S2GAETopologyLossConfig`, `S2GAEConfig.topology_loss`, `_parse_topology_loss_config`, and the `topology_loss` echo in `_config_to_json`. The genuinely-used `topology_validation` block is untouched. Note: `S2GAETopologyLossWeights` is still used by `topology_validation.losses` — do NOT remove it; only remove the `topology_loss`-specific wrapper.

**Files:**
- Modify: `tccig/s2gae.py` (`S2GAEConfig`, `_parse_config`, `_parse_topology_loss_config`, `_config_to_json`, and `S2GAETopologyLossConfig` dataclass)
- Modify: `configs/tccig/01.yaml` (remove `refiner.topology_loss` block, lines 54-65)
- Modify: `tests/unit/test_tccig_s2gae.py` (`test_parse_config_rejects_train_topology_loss_enabled`)
- Modify: `tests/integration/test_tccig_orchestrator.py` (`_tiny_config`, remove `"topology_loss": {"enabled": False}`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `S2GAEConfig` no longer has a `topology_loss` field. `_config_to_json` output no longer has a `"topology_loss"` key.

- [ ] **Step 1: Update the parse-config test to assert the key is ignored**

Replace `test_parse_config_rejects_train_topology_loss_enabled` (test_tccig_s2gae.py:370-379) with a test that a stray `topology_loss` key is simply ignored:

```python
def test_parse_config_ignores_legacy_topology_loss_block(tmp_path: Path) -> None:
    config = _base_refiner_config(tmp_path)
    config["topology_loss"] = {
        "enabled": True,
        "weight": 0.2,
        "losses": {"alpha": 0.7, "beta": 1.5, "gamma": 0.0, "delta": 0.0},
    }

    cfg = _parse_config(config)

    assert not hasattr(cfg, "topology_loss")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_tccig_s2gae.py::test_parse_config_ignores_legacy_topology_loss_block -v`
Expected: FAIL — currently `_parse_config` raises `ValueError("refiner.topology_loss.enabled is not supported ...")`.

- [ ] **Step 3: Remove the `topology_loss` field and parser**

In `tccig/s2gae.py`:
- Delete the `topology_loss: S2GAETopologyLossConfig` field from `S2GAEConfig` (currently `s2gae.py:105`).
- Delete the `S2GAETopologyLossConfig` dataclass definition (the wrapper holding `enabled`/`weight`/`losses` used only by the removed block). Keep `S2GAETopologyLossWeights`.
- Delete `_parse_topology_loss_config` (currently `s2gae.py:1741-1775`).
- Remove the `topology_loss=_parse_topology_loss_config(...)` line from `_parse_config` (currently `s2gae.py:1565`).
- Remove the `"topology_loss": {...}` block from `_config_to_json` (currently `s2gae.py:1606-1614`).

- [ ] **Step 4: Remove the config block from the live YAML**

Delete `configs/tccig/01.yaml` lines 54-65 (the `topology_loss:` block under `refiner:`). Leave `monitor_metric` and `topology_validation` intact.

- [ ] **Step 5: Drop the key from integration fixture**

In `tests/integration/test_tccig_orchestrator.py` `_tiny_config`, remove the line `"topology_loss": {"enabled": False},` (currently :165).

- [ ] **Step 6: Run the suite**

Run: `uv run python -m pytest tests/unit/test_tccig_s2gae.py tests/integration/test_tccig_orchestrator.py -v`
Expected: PASS (including the new `test_parse_config_ignores_legacy_topology_loss_block`).

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check tccig tests
uv run mypy tccig src
git add tccig/s2gae.py configs/tccig/01.yaml tests/unit/test_tccig_s2gae.py tests/integration/test_tccig_orchestrator.py
git commit -m "refactor: remove dead tccig topology_loss config block"
```

---

### Task 2: Collapse multi-rule machinery to a single graph rule

`request.graph_rules` is a tuple but only `rules[0]` is ever used (`_fixed_threshold_rule`). Collapse the request field to one `GraphRule` and remove `_fixed_threshold_rule`. The config-level `parse_rules` validation in `train.py` stays (it rejects top-k/top-m), but the orchestrator resolves to one rule.

**Files:**
- Modify: `tccig/prepare.py` (`TrainRefinerRequest.graph_rules`)
- Modify: `tccig/s2gae.py` (`train_refiner` request usage, `_evaluate_validation_topology_rules`, remove `_fixed_threshold_rule`)
- Modify: `tccig/train.py` (`run_tccig_pipeline` construction of the request, `parse_rules` usage)
- Modify: `tests/integration/test_tccig_orchestrator.py` if it constructs `TrainRefinerRequest` directly (grep first)

**Interfaces:**
- Consumes: `parse_rules(...) -> list[GraphRule]` (unchanged in `train.py`).
- Produces: `TrainRefinerRequest.graph_rule: GraphRule` (singular, replaces `graph_rules: tuple[GraphRule, ...]`). `_evaluate_validation_topology_rules` takes `rule: GraphRule` instead of `rules: Sequence[GraphRule]`.

- [ ] **Step 1: Confirm no other consumer of `graph_rules`**

Run: `uv run python -c "import subprocess"` then grep:
`grep -rn "graph_rules" tccig tests`
Expected: occurrences only in `prepare.py` (dataclass), `s2gae.py` (`train_refiner`), `train.py` (construction). Record any test references.

- [ ] **Step 2: Update the dataclass field**

In `tccig/prepare.py` `TrainRefinerRequest` (:221-230), replace:

```python
    graph_rules: tuple[GraphRule, ...]
```

with:

```python
    graph_rule: GraphRule
```

- [ ] **Step 3: Resolve a single rule in the orchestrator**

In `tccig/train.py` `run_tccig_pipeline`, replace the `graph_rules` tuple construction (:177-179) and the request field (:208):

```python
    parsed_rules = parse_rules(
        _graph_selection(config).get("rules", [refined_output_rule.to_dict()])
    )
    graph_rule = parsed_rules[0]
```

and in the `TrainRefinerRequest(...)` call change `graph_rules=graph_rules,` to `graph_rule=graph_rule,`.

`parse_rules` still validates every configured rule (rejecting top-k/top-m); we keep only the first resolved threshold rule, matching the prior `_fixed_threshold_rule` behavior.

- [ ] **Step 4: Thread the single rule through `train_refiner`**

In `tccig/s2gae.py` `train_refiner`:
- Replace the guard `if not request.graph_rules:` (:554) with `if request.graph_rule is None:` — or remove it if `graph_rule` is now a required non-optional field (it is required; remove the empty-tuple guard and any "must be non-empty" error tied to it).
- Replace `rules=request.graph_rules,` in the `_evaluate_validation_topology_rules(...)` call (:653) with `rule=request.graph_rule,`.

- [ ] **Step 5: Simplify `_evaluate_validation_topology_rules`**

In `tccig/s2gae.py`, change the signature from `rules: Sequence[GraphRule]` to `rule: GraphRule` and delete the `fixed_rule = _fixed_threshold_rule(rules)` line (:1084); use the passed `rule` directly:

```python
def _evaluate_validation_topology_rules(
    *,
    model: S2GAERefiner,
    graph: _SplitGraph,
    pairs: Sequence[CandidatePair],
    validation_plan: InternalValidationPlan,
    rule: GraphRule,
    validation_auprc: float,
    cfg: S2GAEConfig,
    runtime: object,
) -> ValidationTopologyRuleEvaluation:
    refined_probabilities = _prediction_probabilities(
        model=model,
        graph=graph,
        batch_size=cfg.topology_validation.inference_batch_size,
        runtime=runtime,
    )
    if len(refined_probabilities) != len(pairs):
        raise ValueError("validation topology probabilities must match candidate pairs")
    metrics = _validation_topology_metrics(
        validation_plan=validation_plan,
        pairs=pairs,
        probabilities=refined_probabilities,
        rule=rule,
        validation_auprc=validation_auprc,
        cfg=cfg,
    )
    return ValidationTopologyRuleEvaluation(
        rule=rule,
        validation_metrics=metrics,
        rule_payload=rule.to_dict(),
    )
```

- [ ] **Step 6: Delete `_fixed_threshold_rule`**

Remove the `_fixed_threshold_rule` function (currently `s2gae.py:1203-1212`). The threshold-only validation it performed is already enforced by `parse_rules` in `train.py` at config time.

- [ ] **Step 7: Run the suite**

Run: `uv run python -m pytest tests/unit/test_tccig_rules.py tests/integration/test_tccig_orchestrator.py -v`
Expected: PASS. `test_graph_rules_reject_removed_top_m_and_top_k` still passes (validation lives in `parse_rules`).

- [ ] **Step 8: Lint, type-check, commit**

```bash
uv run ruff check tccig tests
uv run mypy tccig src
git add tccig/prepare.py tccig/s2gae.py tccig/train.py tests
git commit -m "refactor: collapse tccig graph rules to a single threshold rule"
```

---

### Task 3: Remove unreachable errors and the redundant summary write

Two small dead-code removals: the unreachable `ValueError`s in `_edge_index_and_weight_from_edges` (graph edges are always a subset of pair edges), and the duplicate post-loop `_write_training_summary` call.

**Files:**
- Modify: `tccig/s2gae.py` (`_edge_index_and_weight_from_edges`, post-loop summary write)

**Interfaces:**
- Consumes: nothing new.
- Produces: no signature changes.

- [ ] **Step 1: Read the function to confirm the unreachable branch**

Read `tccig/s2gae.py` around `_edge_index_and_weight_from_edges` (near :950). Confirm the `ValueError` fires only when an edge endpoint is missing from the node-id map, which cannot happen because `_collect_node_ids` includes every pair protein. If the read shows the error guards a genuinely reachable case, STOP and flag — do not remove.

- [ ] **Step 2: Remove the unreachable guard**

Delete the `raise ValueError(...)` branch(es) (currently `s2gae.py:950-953`) that handle missing node ids, keeping the normal index-lookup path.

- [ ] **Step 3: Remove the redundant post-loop summary write**

In `train_refiner`, the per-epoch loop already calls `_write_training_summary` on the main process (`s2gae.py:741-749`). Delete the second, post-loop `_write_training_summary(...)` call (currently `s2gae.py:773-780`) inside the `if request.runtime.is_main_process:` checkpoint-save block. Keep the checkpoint `save`/`torch.save` itself.

- [ ] **Step 4: Run the suite**

Run: `uv run python -m pytest tests/unit/test_tccig_s2gae.py tests/integration/test_tccig_orchestrator.py -v`
Expected: PASS. The integration test still asserts `training_summary.json` / checkpoint artifacts exist (written by the per-epoch call).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check tccig tests
uv run mypy tccig src
git add tccig/s2gae.py
git commit -m "refactor: drop unreachable tccig edge guard and duplicate summary write"
```

---

### Task 4: Trim dead and bookkeeping CSV/JSON columns

Reduce `TCCIG_TRAIN_CSV_COLUMNS` from 26 to 19. Drop the 3 always-zero topology-train columns and the 4 Local/Global pair-count bookkeeping columns, dropping their `epoch_history` keys and formatter references in lockstep.

**Files:**
- Modify: `tccig/s2gae.py` (`TCCIG_TRAIN_CSV_COLUMNS`, the `epoch_history` dict in `train_refiner`, `_append_tccig_train_csv_row`, `_log_epoch_summary`)
- Modify: `tests/integration/test_tccig_orchestrator.py` if it asserts on dropped columns (grep first)

**Interfaces:**
- Consumes: nothing new.
- Produces: `TCCIG_TRAIN_CSV_COLUMNS` is the 19-column list from Global Constraints. `epoch_history` no longer carries `train_topology_loss`, `train_graph_similarity_loss`, `train_relative_density_loss`, `local_train_pairs`, `global_train_pairs`, `local_validation_pairs`, `global_validation_pairs`.

- [ ] **Step 1: Grep for downstream references to the dropped columns/keys**

Run:
`grep -rn "Local Train Pairs\|Global Train Pairs\|Local Validation Pairs\|Global Validation Pairs\|Train Topology Loss\|Train GS Loss\|Train RD Loss\|local_train_pairs\|global_train_pairs\|local_validation_pairs\|global_validation_pairs\|train_topology_loss\|train_graph_similarity_loss\|train_relative_density_loss" tccig tests`
Record every hit; each must be removed or updated below. Note: `_rank_local_pair_count` (used to compute `local_validation_pairs`) is also removed in Task 6 — if it becomes unused here, leave its removal to Task 6 to keep this task log-only.

- [ ] **Step 2: Replace the column list**

In `tccig/s2gae.py`, replace `TCCIG_TRAIN_CSV_COLUMNS` (:59-86) with exactly:

```python
TCCIG_TRAIN_CSV_COLUMNS = [
    "Epoch",
    "Epoch Time",
    "Train Loss",
    "Train BCE Loss",
    "Train Residual Anchor Loss",
    "Train Weighted Residual Anchor Loss",
    "Train Gradient Norm",
    "Val auprc",
    "Val Topology Loss",
    "Internal Val graph_sim",
    "Internal Val relative_density",
    "Internal Val deg_dist_mmd",
    "Internal Val cc_mmd",
    "Selected Rule Type",
    "Selected Rule Positive Edges",
    "Monitor Metric",
    "Monitor Value",
    "Peak GPU Mem MB",
    "Learning Rate",
]
```

- [ ] **Step 3: Drop the keys from `epoch_history`**

In `train_refiner`, remove these keys from the `epoch_history` dict (currently `s2gae.py:678-697`): `"train_topology_loss"`, `"train_graph_similarity_loss"`, `"train_relative_density_loss"`, `"local_train_pairs"`, `"global_train_pairs"`, `"local_validation_pairs"`, `"global_validation_pairs"`. Keep `global_train_count` usage for the loss-mean denominator (`epoch_denominator`) — that local variable stays; only the history keys are removed.

- [ ] **Step 4: Update the CSV row formatter**

In `_append_tccig_train_csv_row` (around `s2gae.py:1360`), remove the row entries for the 7 dropped columns (e.g. `"Train Topology Loss": float(epoch_history["train_topology_loss"])` at :1362, and the Local/Global pair entries). The DictWriter fieldnames now come from the trimmed `TCCIG_TRAIN_CSV_COLUMNS`, so every remaining written key must still exist.

- [ ] **Step 5: Update `_log_epoch_summary`**

In `_log_epoch_summary` (around `s2gae.py:1400-1420`), remove references to `epoch_history["train_topology_loss"]` and any local/global pair-count fields from the log format string and its arguments. Keep `val_topology_loss`, rule, and the retained metrics.

- [ ] **Step 6: Remove/adjust any test assertions on dropped columns**

For each hit from Step 1 in `tests/`, remove or update the assertion. If the integration test reads `tccig_train_step.csv` headers, update the expected header set to the 19 columns.

- [ ] **Step 7: Run the suite**

Run: `uv run python -m pytest tests/unit/test_tccig_s2gae.py tests/integration/test_tccig_orchestrator.py -v`
Expected: PASS.

- [ ] **Step 8: Lint, type-check, commit**

```bash
uv run ruff check tccig tests
uv run mypy tccig src
git add tccig/s2gae.py tests
git commit -m "refactor: trim dead and bookkeeping tccig train log columns"
```

---

### Task 5: Rewrite the prediction path on Accelerate (sharding + gather)

Replace manual strided sharding (`_rank_local_pair_indices`) and index-tagged row reassembly with an Accelerate-prepared `DataLoader` over pair indices plus `gather_for_metrics`. The encoder still runs once per rank on the full graph; only decoder batching and the cross-rank gather change. A post-gather assertion enforces exactly-once coverage of every pair index. This is the highest-risk task; the single-process integration test is the equivalence guard.

**Files:**
- Modify: `tccig/s2gae.py` (`_prediction_probabilities`, gather helpers)
- Modify: `tests/unit/test_tccig_s2gae.py` (retarget the `_rank_local_pair_indices` and `_ordered_values_from_accelerate_rows` tests; the `_prediction_probabilities` single-process test stays)

**Interfaces:**
- Consumes: `runtime.accelerator` exposing `.prepare(dataloader)` and `.gather_for_metrics(tensor)`; `runtime.is_distributed: bool`.
- Produces: `_prediction_probabilities(*, model, graph, batch_size, runtime) -> list[float]` — same signature and contract (returns one probability per pair in global pair order). New private helper `_ordered_probabilities_from_indexed_rows(*, total: int, rows: torch.Tensor) -> list[float]` that maps `(index, prob)` rows to an ordered list and asserts exactly-once coverage.

- [ ] **Step 1: Write the failing test for the ordered-rows helper**

In `tests/unit/test_tccig_s2gae.py`, replace the two `_ordered_values_from_accelerate_rows` tests (:108-142) with tests for the new helper. Add to the imports (:13-28) `_ordered_probabilities_from_indexed_rows` and remove `_ordered_values_from_accelerate_rows` and `_rank_local_pair_indices`:

```python
def test_ordered_probabilities_from_indexed_rows_restores_global_order() -> None:
    rows = torch.tensor(
        [[2.0, 0.2], [0.0, 0.0], [1.0, 0.1]],
        dtype=torch.float64,
    )

    values = _ordered_probabilities_from_indexed_rows(total=3, rows=rows)

    assert values == [0.0, 0.1, 0.2]


def test_ordered_probabilities_from_indexed_rows_tolerates_duplicate_tail_rows() -> None:
    # Accelerate even_batches duplicates tail samples across ranks; the helper
    # keeps the first occurrence and still covers every index exactly once.
    rows = torch.tensor(
        [[0.0, 0.5], [1.0, 0.6], [1.0, 0.6]],
        dtype=torch.float64,
    )

    values = _ordered_probabilities_from_indexed_rows(total=2, rows=rows)

    assert values == [0.5, 0.6]


def test_ordered_probabilities_from_indexed_rows_raises_on_missing_index() -> None:
    rows = torch.tensor([[0.0, 0.5]], dtype=torch.float64)

    with pytest.raises(ValueError, match="Missing"):
        _ordered_probabilities_from_indexed_rows(total=2, rows=rows)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_tccig_s2gae.py::test_ordered_probabilities_from_indexed_rows_restores_global_order -v`
Expected: FAIL — `_ordered_probabilities_from_indexed_rows` does not exist (ImportError).

- [ ] **Step 3: Implement the ordered-rows helper**

In `tccig/s2gae.py`, add (replacing `_ordered_values_from_accelerate_rows`, `_ordered_values_from_row_tensor`, `_ordered_values_from_shards`):

```python
def _ordered_probabilities_from_indexed_rows(
    *,
    total: int,
    rows: torch.Tensor,
) -> list[float]:
    """Map ``(index, probability)`` rows to global pair order.

    Tolerates duplicate rows (Accelerate ``even_batches`` repeats tail samples)
    by keeping the first occurrence, and asserts every index is covered once.
    """
    ordered: list[float | None] = [None] * total
    for row in rows.detach().cpu():
        index = int(row[0].item())
        if index < 0 or index >= total:
            continue
        if ordered[index] is None:
            ordered[index] = float(row[1].item())
    missing = [index for index, value in enumerate(ordered) if value is None]
    if missing:
        raise ValueError(f"Missing refined probabilities for indices: {missing[:10]}")
    return [float(value) for value in ordered]
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `uv run python -m pytest tests/unit/test_tccig_s2gae.py -k ordered_probabilities_from_indexed_rows -v`
Expected: PASS (all three).

- [ ] **Step 5: Rewrite `_prediction_probabilities` to use an Accelerate DataLoader + gather**

Replace `_prediction_probabilities` (`s2gae.py:1015-1061`) with a version that encodes once, then iterates an Accelerate-prepared `DataLoader` of pair indices and gathers `(index, prob)` rows:

```python
def _prediction_probabilities(
    *,
    model: S2GAERefiner,
    graph: _SplitGraph,
    batch_size: int,
    runtime: object,
) -> list[float]:
    model.eval()
    device = graph.pairwise_probabilities.device
    total = int(graph.pairwise_probabilities.numel())
    accelerator = getattr(runtime, "accelerator", None)
    index_loader: Iterable[torch.Tensor] = DataLoader(
        TensorDataset(torch.arange(total, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=False,
    )
    prepare_fn = getattr(accelerator, "prepare", None)
    if callable(prepare_fn):
        index_loader = cast(Iterable[torch.Tensor], prepare_fn(index_loader))

    indexed_rows: list[torch.Tensor] = []
    with torch.inference_mode():
        hidden_states = model.encode(
            node_features=graph.node_features,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
        )
        for (batch_indices,) in cast(Iterable[tuple[torch.Tensor]], index_loader):
            batch_indices = batch_indices.to(device)
            refined_logits, _ = model.decode(
                hidden_states=hidden_states,
                pair_index=graph.pair_index[:, batch_indices],
                pairwise_probabilities=graph.pairwise_probabilities[batch_indices],
            )
            indexed_rows.append(
                torch.stack(
                    (
                        batch_indices.to(dtype=torch.float64),
                        torch.sigmoid(refined_logits).to(dtype=torch.float64),
                    ),
                    dim=1,
                )
            )
    local_rows = (
        torch.cat(indexed_rows, dim=0)
        if indexed_rows
        else torch.empty((0, 2), dtype=torch.float64, device=device)
    )
    gather_fn = getattr(accelerator, "gather_for_metrics", None)
    if bool(getattr(runtime, "is_distributed", False)) and callable(gather_fn):
        local_rows = cast(torch.Tensor, gather_fn(local_rows))
    return _ordered_probabilities_from_indexed_rows(total=total, rows=local_rows)
```

Add `from torch.utils.data import DataLoader, TensorDataset` to the imports if not already present (DataLoader is already imported at `s2gae.py:26`; add `TensorDataset`).

Note on correctness: with a prepared DataLoader Accelerate shards indices across ranks and `gather_for_metrics` reassembles them (dropping `even_batches` tail duplicates). In single-process mode `prepare` returns the loader unchanged and `is_distributed` is False, so we read all `total` rows directly — exactly the prior behavior. The ordered-rows helper tolerates any residual tail duplicates and asserts full coverage.

- [ ] **Step 6: Delete the now-unused sharding helper**

Remove `_rank_local_pair_indices` (`s2gae.py:1221-1233`). Keep `_batch_indices` only if still referenced elsewhere (grep `_batch_indices`; if unused after this change, remove it too).

- [ ] **Step 7: Run prediction + validation tests**

Run: `uv run python -m pytest tests/unit/test_tccig_s2gae.py::test_prediction_probabilities_encode_graph_once_across_decoder_batches tests/unit/test_tccig_s2gae.py -k "prediction or ordered" -v`
Expected: PASS. The encode-once test asserts `encode_calls == 1` and 3 probabilities — both hold because encode runs before the loop.

- [ ] **Step 8: Run the full tccig suite (single-process equivalence guard)**

Run: `uv run python -m pytest tests/integration/test_tccig_orchestrator.py -v`
Expected: PASS — the full CPU pipeline produces identical artifacts and metrics.

- [ ] **Step 9: Lint, type-check, commit**

```bash
uv run ruff check tccig tests
uv run mypy tccig src
git add tccig/s2gae.py tests/unit/test_tccig_s2gae.py
git commit -m "refactor: drive tccig prediction sharding with accelerate dataloader"
```

---

### Task 6: Replace remaining DDP shims with direct Accelerate calls

Remove the `getattr`-guarded distributed shims now that prediction no longer needs manual rank math. Use `accelerator.reduce`, `accelerator.clip_grad_norm_`, and `accelerator.save` directly. `TCCIGRuntime` always wraps a real `Accelerator`; the test double `NoOpAccelerator` already implements these.

**Files:**
- Modify: `tccig/s2gae.py` (`_accelerator_reduce_sum`, `_clip_grad_norm_with_accelerator`, `_runtime_is_distributed/_rank/_world_size`, `_rank_local_pair_count`, the `save_fn = getattr(...)` block)
- Modify: `tests/runtime_helpers.py` only if `NoOpAccelerator` lacks `clip_grad_norm_` (it must return a tensor/float)

**Interfaces:**
- Consumes: `accelerator.reduce(tensor, reduction="sum")`, `accelerator.clip_grad_norm_(params, max_norm)`, `accelerator.save(obj, path, safe_serialization=False)`, `accelerator.wait_for_everyone()`.
- Produces: no public signature changes; internal helpers removed.

- [ ] **Step 1: Confirm the test accelerator supports the direct calls**

Grep `tests/runtime_helpers.py` for `clip_grad_norm_`. `NoOpAccelerator` currently has `reduce`, `gather_for_metrics`, `pad_across_processes`, `wait_for_everyone`, `save` but NOT `clip_grad_norm_`. Add it:

```python
    def clip_grad_norm_(
        self,
        parameters: object,
        max_norm: float,
        norm_type: float = 2.0,
    ) -> torch.Tensor:
        """Clip gradients like accelerate and return the observed norm."""
        del norm_type
        params = [p for p in parameters if p.grad is not None]
        if not params:
            return torch.tensor(0.0)
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
```

- [ ] **Step 2: Replace `_clip_grad_norm_with_accelerator` call sites with direct usage**

In `train_refiner` (`s2gae.py:615-619`), replace the helper call:

```python
            gradient_norm = _clip_grad_norm_with_accelerator(
                model=train_step_model,
                runtime=request.runtime,
                gradient_clip_norm=cfg.optimization.gradient_clip_norm,
            )
```

with direct Accelerate usage, preserving the "no clip configured → just measure norm" behavior:

```python
            if cfg.optimization.gradient_clip_norm is None:
                gradient_norm = apply_gradient_clipping(
                    model=train_step_model, gradient_clip_norm=None
                )
            else:
                clipped = request.runtime.accelerator.clip_grad_norm_(
                    train_step_model.parameters(),
                    cfg.optimization.gradient_clip_norm,
                )
                gradient_norm = float(clipped.detach().cpu().item())
```

Then delete `_clip_grad_norm_with_accelerator` (`s2gae.py:413-430`). Keep `apply_gradient_clipping` (still used for the disabled-clip path and unit-tested at `test_tccig_s2gae.py:427-444`).

- [ ] **Step 3: Replace `_accelerator_reduce_sum` with a direct reduce**

Find the `_accelerator_reduce_sum(local_loss_sums, request.runtime)` call (`s2gae.py:623`) and replace with:

```python
        global_loss_sums = request.runtime.accelerator.reduce(local_loss_sums, reduction="sum")
```

Delete `_accelerator_reduce_sum` (`s2gae.py:1246-1256`). Remove the now-unused `import torch.distributed as dist` (`s2gae.py:17`) if no other reference remains (grep `dist\.`).

- [ ] **Step 4: Replace the checkpoint `save_fn = getattr(...)` dance**

In the checkpoint-save block (`s2gae.py:768-772`), replace:

```python
        save_fn = getattr(request.runtime.accelerator, "save", None)
        if callable(save_fn):
            save_fn(payload, cfg.checkpoint_path, safe_serialization=False)
        else:
            torch.save(payload, cfg.checkpoint_path)
```

with:

```python
        request.runtime.accelerator.save(payload, cfg.checkpoint_path, safe_serialization=False)
```

- [ ] **Step 5: Remove dead runtime helpers**

Delete `_runtime_is_distributed` (:1307), `_runtime_rank` (:1311), `_runtime_world_size` (:1315), and `_rank_local_pair_count` (:1236) — grep each first to confirm no remaining references (after Task 4 dropped the pair-count history keys and Task 5 dropped strided sharding, these should be unused). If `_rank_local_pair_count` still has a reference, STOP and resolve it before deleting. Keep `_runtime_barrier` (it wraps `wait_for_everyone`).

- [ ] **Step 6: Run the full tccig suite**

Run: `uv run python -m pytest tests/unit/test_tccig_s2gae.py tests/unit/test_tccig_s2gae_validation.py tests/integration/test_tccig_orchestrator.py -v`
Expected: PASS — single-process reduce/clip/save go through `NoOpAccelerator`/real `Accelerator` identically.

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check tccig tests
uv run mypy tccig src
git add tccig/s2gae.py tests/runtime_helpers.py
git commit -m "refactor: use accelerate reduce/clip/save directly in tccig refiner"
```

---

### Task 7: Consolidate the ordered-gather helper across train.py and s2gae.py

`train.py` has `_ordered_probabilities_from_rows` (scorer gather) and `s2gae.py` now has `_ordered_probabilities_from_indexed_rows`. They have the same job (map `(index, prob)` rows to ordered list, tolerate tail duplicates, assert coverage). Consolidate into one shared helper to remove the near-duplicate and the `train.py` `getattr`-guarded gather.

**Files:**
- Modify: `tccig/prepare.py` (add the shared helper — `prepare.py` is the shared dependency both `train.py` and `s2gae.py` import from)
- Modify: `tccig/train.py` (`_ordered_probabilities_from_rows` → use shared helper; simplify the gather)
- Modify: `tccig/s2gae.py` (`_ordered_probabilities_from_indexed_rows` → use shared helper)

Note: the spec marks this consolidation conditional ("if both still need ordered gather"). After Tasks 5–6 both do, so this task applies. The README's Hard Rule allows touching `prepare.py` when the contract is intentionally changed; adding a shared internal helper is acceptable and keeps the surface small.

**Interfaces:**
- Consumes: nothing new.
- Produces: `tccig.prepare.ordered_probabilities_from_indexed_rows(*, total: int, rows: torch.Tensor) -> list[float]` (public-within-package; no leading underscore since it crosses modules). Both `train.py` and `s2gae.py` import and call it.

- [ ] **Step 1: Write a failing test for the shared helper in prepare**

Add to `tests/unit/test_tccig_prepare.py`:

```python
def test_ordered_probabilities_from_indexed_rows_orders_and_dedups() -> None:
    import torch
    from tccig.prepare import ordered_probabilities_from_indexed_rows

    rows = torch.tensor([[1.0, 0.6], [0.0, 0.5], [1.0, 0.6]], dtype=torch.float64)

    assert ordered_probabilities_from_indexed_rows(total=2, rows=rows) == [0.5, 0.6]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run python -m pytest tests/unit/test_tccig_prepare.py::test_ordered_probabilities_from_indexed_rows_orders_and_dedups -v`
Expected: FAIL — `ordered_probabilities_from_indexed_rows` not defined in `tccig.prepare`.

- [ ] **Step 3: Add the shared helper to prepare.py**

In `tccig/prepare.py` add (near the other module-level helpers):

```python
def ordered_probabilities_from_indexed_rows(
    *,
    total: int,
    rows: torch.Tensor,
) -> list[float]:
    """Map ``(index, probability)`` rows to global order.

    Keeps the first occurrence of duplicate indices (Accelerate ``even_batches``
    repeats tail samples) and raises if any index is uncovered.
    """
    ordered: list[float | None] = [None] * total
    for row in rows.detach().cpu():
        index = int(row[0].item())
        if index < 0 or index >= total:
            continue
        if ordered[index] is None:
            ordered[index] = float(row[1].item())
    missing = [index for index, value in enumerate(ordered) if value is None]
    if missing:
        raise ValueError(f"Missing probabilities for indices: {missing[:10]}")
    return [float(value) for value in ordered]
```

`torch` is already imported in `prepare.py` (:13).

- [ ] **Step 4: Point s2gae.py at the shared helper**

In `tccig/s2gae.py`, remove the local `_ordered_probabilities_from_indexed_rows` (added in Task 5), import the shared one from `tccig.prepare` (add to the existing `from tccig.prepare import (...)` block), and update the call in `_prediction_probabilities` to `ordered_probabilities_from_indexed_rows(total=total, rows=local_rows)`.

- [ ] **Step 5: Point train.py at the shared helper**

In `tccig/train.py` `_ordered_probabilities_from_rows` (:582-606), replace the body's ordering loop with a call to the shared helper, keeping the distributed gather:

```python
def _ordered_probabilities_from_rows(
    *,
    total: int,
    local_rows: torch.Tensor,
    runtime: TCCIGRuntime,
) -> list[float]:
    rows = local_rows
    if runtime.is_distributed:
        rows = cast(torch.Tensor, runtime.accelerator.gather_for_metrics(local_rows))
    return ordered_probabilities_from_indexed_rows(total=total, rows=rows)
```

Add `ordered_probabilities_from_indexed_rows` to the `from tccig.prepare import (...)` block in `train.py`. Note: this drops the `pad_across_processes(..., pad_index=-1)` call. `gather_for_metrics` handles padding/truncation for metric gathering, and the shared helper ignores out-of-range/`-1` indices, so dropping the explicit pad is safe. If the scorer rows can have ragged per-rank lengths that `gather_for_metrics` cannot handle, KEEP the `pad_across_processes` line — verify against the Accelerate version in `pyproject.toml` before removing.

- [ ] **Step 6: Update the s2gae unit-test import**

In `tests/unit/test_tccig_s2gae.py`, the helper tests from Task 5 referenced the s2gae-local symbol. Repoint them to import from `tccig.prepare` (or move those two assertions to `test_tccig_prepare.py` and delete from the s2gae test). Keep coverage of order + dedup + missing-index.

- [ ] **Step 7: Run the suite**

Run: `uv run python -m pytest tests/unit/test_tccig_prepare.py tests/unit/test_tccig_s2gae.py tests/integration/test_tccig_orchestrator.py -v`
Expected: PASS.

- [ ] **Step 8: Lint, type-check, commit**

```bash
uv run ruff check tccig tests
uv run mypy tccig src
git add tccig/prepare.py tccig/train.py tccig/s2gae.py tests
git commit -m "refactor: share tccig ordered-gather helper across orchestrator and refiner"
```

---

### Task 8: Fix `best_validation_auprc` to track the selected epoch

`best_validation_auprc` is currently `max(...)` over all epochs (`s2gae.py:637`), decoupled from checkpoint selection. Move its capture into the `_is_better_monitor` block so the reported AUPRC corresponds to the checkpointed epoch.

**Files:**
- Modify: `tccig/s2gae.py` (`train_refiner`)
- Modify: `tests/integration/test_tccig_orchestrator.py` (add an assertion tying reported AUPRC to the selected epoch) OR a focused unit test if feasible

**Interfaces:**
- Consumes: nothing new.
- Produces: `S2GAERefinerState.best_validation_auprc` and the checkpoint/`training_summary.json` `best_validation_auprc` now equal the validation AUPRC at the epoch whose monitor value was selected.

- [ ] **Step 1: Write a failing assertion for the coupling**

Add to `tests/integration/test_tccig_orchestrator.py` a test that runs the pipeline with `monitor_metric="val_topology_loss"` and asserts the persisted `best_validation_auprc` equals the `Val auprc` of the selected epoch. Read the selected epoch from the checkpoint's `best_monitor_value` matched against the CSV/JSON history:

```python
def test_best_validation_auprc_matches_selected_epoch(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, "auprc_couple")
    refiner_config = config["refiner"]
    assert isinstance(refiner_config, dict)
    refiner_config["epochs"] = 3
    refiner_config["monitor_metric"] = "val_topology_loss"
    refiner_config["topology_validation"] = {
        "enabled": True,
        "node_sizes": [2],
        "samples_per_size": 1,
        "strategy": "mixed",
        "seed": 0,
        "inference_batch_size": 4,
        "compute_clustering_mmd": False,
        "losses": {"alpha": 1.0, "beta": 1.0, "gamma": 0.0, "delta": 0.0},
    }

    run_tccig_pipeline(config)

    checkpoint_path = tmp_path / "models" / "tccig" / "auprc_couple" / "best_model.pt"
    payload = torch.load(checkpoint_path, weights_only=False)
    summary_path = tmp_path / "logs" / "tccig" / "auprc_couple" / "training_summary.json"
    history = json.loads(summary_path.read_text(encoding="utf-8"))["history"]

    best_monitor = payload["best_monitor_value"]
    selected = min(history, key=lambda row: abs(row["monitor_value"] - best_monitor))
    assert payload["best_validation_auprc"] == pytest.approx(selected["val_auprc"])
```

Confirm the exact summary key names (`history`, `monitor_value`, `val_auprc`) by reading `_write_training_summary` and the `epoch_history` dict; adjust the test to match the real keys. If `torch`/`json`/`pytest` aren't imported in the test module, add them.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run python -m pytest tests/integration/test_tccig_orchestrator.py::test_best_validation_auprc_matches_selected_epoch -v`
Expected: FAIL — `best_validation_auprc` is the global max, not the selected epoch's value (unless they coincide; the 3-epoch topology-loss setup should diverge).

- [ ] **Step 3: Move the capture into the selection block**

In `train_refiner`, delete the unconditional `best_validation_auprc = max(best_validation_auprc, validation_auprc)` (`s2gae.py:637`). Initialize `best_validation_auprc = 0.0` before the loop (it already is) and set it inside the `_is_better_monitor` block (`s2gae.py:728-740`):

```python
        if best_state_dict is None or _is_better_monitor(
            value=monitor_value,
            best_value=best_monitor_value,
            monitor_metric=cfg.monitor_metric,
        ):
            best_monitor_value = monitor_value
            best_validation_auprc = validation_auprc
            best_selected_rule = selected_epoch_rule
            best_selected_rule_payload = selected_epoch_rule_payload
            checkpoint_model = _unwrap_refiner(train_step_model, request.runtime.accelerator)
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in checkpoint_model.state_dict().items()
            }
```

- [ ] **Step 4: Run the coupling test + validation test**

Run: `uv run python -m pytest tests/integration/test_tccig_orchestrator.py::test_best_validation_auprc_matches_selected_epoch tests/unit/test_tccig_s2gae_validation.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full tccig suite**

Run: `uv run python -m pytest tests/unit/test_tccig_s2gae.py tests/integration/test_tccig_orchestrator.py -v`
Expected: PASS — when `monitor_metric="val_auprc"` the selected epoch IS the max-AUPRC epoch, so existing tests are unaffected.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check tccig tests
uv run mypy tccig src
git add tccig/s2gae.py tests/integration/test_tccig_orchestrator.py
git commit -m "fix: report tccig best_validation_auprc at the selected checkpoint epoch"
```

---

### Task 9: Make the residual anchor honest + fix README mismatches

Set the live config `residual_weight` to `1e-3` (the parser default is already `0.001`), and correct the two README documentation mismatches (input-threshold default vs `target_precision`; train-node-universe topology graph).

**Files:**
- Modify: `configs/tccig/01.yaml` (`refiner.residual_weight`)
- Modify: `tccig/README.md` (PRING Contract section)

**Interfaces:**
- Consumes: nothing.
- Produces: no code interface changes.

- [ ] **Step 1: Update the live config residual weight**

In `configs/tccig/01.yaml`, change `residual_weight: 1.0e-8` (:53) to:

```yaml
  residual_weight: 1.0e-3
```

- [ ] **Step 2: Correct the README input-threshold wording**

In `tccig/README.md` PRING Contract section, update the hard-graph-rule sentence (around :69) to note that the pairwise-input threshold is resolved by the live config. Replace the relevant text with:

```markdown
The hard graph rule for refined output remains `threshold=0.5`; per-node top-k
and global top-M are not supported. The pairwise *input* threshold that builds
`G_pairwise` is resolved from `graph_selection.pairwise_input_threshold`: the
live config (`configs/tccig/01.yaml`) uses `mode: target_precision` on the
validation split, so the threshold is data-derived. The fixed `0.5` default
applies only when no precision target is configured.
```

- [ ] **Step 3: Correct the README validation-topology wording**

Update the "Validation topology builds a true topology graph from positive rows in `human_val_ppi_ratio5_exclusive.txt`" sentence (around :65-69) to add the node-universe detail:

```markdown
Validation topology builds a true topology graph seeded from the **train**
node universe (`load_split_node_ids(..., split_name="train")`) with edges from
positive rows in `human_val_ppi_ratio5_exclusive.txt`, samples PRING-style
validation node buckets, scores every non-self pair inside those buckets, and
selects the checkpoint from configured hard topology metrics when
`refiner.topology_validation.enabled` is true.
```

- [ ] **Step 4: Verify config still parses and pipeline runs**

Run: `uv run python -m pytest tests/integration/test_tccig_orchestrator.py -v`
Expected: PASS (the integration fixture sets its own `residual_weight`; this confirms no schema regression). Optionally sanity-check the YAML loads:
`uv run python -c "import yaml; yaml.safe_load(open('configs/tccig/01.yaml'))"`

- [ ] **Step 5: Lint, commit**

```bash
uv run ruff check tccig tests
git add configs/tccig/01.yaml tccig/README.md
git commit -m "docs: make tccig residual anchor honest and fix readme mismatches"
```

---

### Task 10: Full-suite verification

Final gate across the whole tccig surface.

**Files:** none (verification only).

- [ ] **Step 1: Run the full tccig test set**

Run:
```bash
uv run python -m pytest tests/unit/test_tccig_s2gae.py tests/unit/test_tccig_s2gae_validation.py tests/unit/test_tccig_prepare.py tests/unit/test_tccig_rules.py tests/unit/test_tccig_pairwise_scorer.py tests/integration/test_tccig_orchestrator.py -v
```
Expected: all PASS.

- [ ] **Step 2: Lint and type-check the full surface**

```bash
uv run ruff check tccig tests
uv run mypy tccig src
```
Expected: clean.

- [ ] **Step 3: Confirm no leftover references to removed symbols**

Run:
```bash
grep -rn "_fixed_threshold_rule\|_rank_local_pair_indices\|_rank_local_pair_count\|_accelerator_reduce_sum\|_clip_grad_norm_with_accelerator\|_runtime_is_distributed\|_ordered_values_from_accelerate_rows\|_ordered_values_from_shards\|topology_loss\|graph_rules\|Local Train Pairs\|Global Train Pairs" tccig tests
```
Expected: no matches in source (test strings referencing `parse_rules` rejection messages are fine; `graph_rule` singular is fine — confirm any hit is intentional).

- [ ] **Step 4: Final commit if any cleanup was needed**

```bash
git add -A
git commit -m "chore: finalize tccig cleanup verification"
```
(Skip if nothing changed.)

---

## Self-Review Notes

- **Spec coverage:** WS1 dead code → Tasks 1–3; WS2 column trim → Task 4; WS3 Accelerate → Tasks 5–7; WS4 reporting fix → Task 8; WS5 residual + docs → Task 9; verification → Task 10. All workstreams covered.
- **Type consistency:** `TrainRefinerRequest.graph_rule: GraphRule` (Task 2) is consumed in Task 2 Step 4; `_evaluate_validation_topology_rules(..., rule: GraphRule)` consistent across Steps 4–5. `ordered_probabilities_from_indexed_rows(*, total, rows)` defined in Task 7 matches the s2gae-local version it supersedes (Task 5). `clip_grad_norm_` added to `NoOpAccelerator` (Task 6 Step 1) returns a tensor, matching the `.detach().cpu().item()` call (Task 6 Step 2).
- **Ordering rationale:** dead-code tasks (1–4) shrink the surface before the higher-risk Accelerate rewrite (5–7). Task 7 depends on the helper introduced in Task 5 and the shim removal in Task 6. Task 8 (reporting) is independent but placed after the loop is otherwise stable. Task 9 is config/docs only.
- **Risk callouts embedded:** Task 5 Step 8 and Task 6 Step 6 use the integration test as the single-process equivalence guard; multi-GPU is not exercised in CI (documented in the spec Risks section). Task 7 Step 5 flags the `pad_across_processes` removal as conditional on the Accelerate version.
