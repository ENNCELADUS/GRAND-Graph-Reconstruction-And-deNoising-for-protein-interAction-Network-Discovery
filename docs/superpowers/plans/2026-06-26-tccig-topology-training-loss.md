# TCCIG Run 02 — Topology-Conditioned Training Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a differentiable, topology-conditioned loss to the TCCIG S2GAE refiner's *training* objective (plus an asymmetric residual anchor and warmup schedule) so the refiner learns to prune topologically-spurious edges instead of only adding them.

**Architecture:** A train-side topology plan is built once by the orchestrator (`tccig/train.py`) on the train split graph, using the existing node-bucket sampler primitives plus a new GT-positive-edge coverage-augmentation pass. Each epoch, after the per-batch BCE+anchor pass, the train step runs one full-plan differentiable topology backward over those buckets via `src.topology.finetune_losses.compute_topology_losses`. The symmetric `delta.pow(2)` residual anchor is replaced with a one-sided `relu(delta).pow(2)` penalty so deletion is free, and the topology weight is warmup-ramped via `topology_loss_scale`.

**Tech Stack:** Python 3.10+, PyTorch, PyTorch Geometric (`GraphConv`), `networkx`, Accelerate (DDP), `uv` env, `pytest`, `ruff`, `mypy`.

## Global Constraints

- Environment: `uv` + `pyproject.toml`. Run everything through `uv run` (e.g. `uv run python -m pytest`, `uv run ruff check .`, `uv run mypy src`).
- Editable surface in the TCCIG sandbox is limited to `train.py`, `prepare.py`, `s2gae.py`, `test.py`, and config files (per `tccig/README.md`). New differentiable-loss reuse comes from `src/topology/finetune_losses.py` and `src/topology/finetune_data.py` — **import, do not fork**.
- **Do NOT reuse `sample_edge_cover_subgraphs`** (`src/topology/finetune_data.py:890`). Use `sample_topology_evaluation_subgraphs` + `build_internal_validation_plan` plus the coverage-augmentation pass only.
- Code style: no `print` (use `logging`); no hardcoded tunables (use config); strict type hints (avoid `Any`); absolute imports (`from src.x import y` / `from tccig.x import y`); Google-style docstrings; functions < 50 lines; files target 200-400 lines (s2gae.py is already large — add focused helpers, do not restructure it).
- TDD: write the failing test first, run it red, implement minimally, run green, commit. Aim ≥ 80% coverage on touched code.
- `monitor_metric` stays `val_topology_loss` (unchanged). Checkpoint selection still uses the non-differentiable eval path.
- Coverage guarantee is **GT positive edges only**; assert `positive_edge_coverage == 1.0` after augmentation or raise (fail-fast).
- Concrete config defaults for `02.yaml` (decided here): asymmetric anchor `weight: 1.0e-4` (10× below 01's `1e-3`); topology `topology_weight: 1.0`; loss `weights: {alpha: 1.0, beta: 8.0, gamma: 0.5, delta: 0.1}` (mirrors 01's validation `losses` block so train and monitor agree); `schedule: {warmup_epochs: 5, ramp_epochs: 10, schedule: linear}`.

---

## File Structure

- `tccig/s2gae.py` (modify): new config fields + parsing for `residual_anchor` and `topology_training`; new `asymmetric_residual_anchor` loss helper and `S2GAELossTerms` reuse; topology-loss-term helper over an `InternalValidationPlan`; epoch-loop wiring for the per-epoch full-plan topology backward; new history/log fields.
- `tccig/train.py` (modify): new `_build_train_topology_bundle` (mirror of `_build_validation_topology_bundle`) + coverage-augmentation helper; thread the train topology plan into `TrainRefinerRequest`.
- `tccig/s2gae.py` `TrainRefinerRequest` (modify): add `train_topology` + `train_topology_plan` fields.
- `tccig/test.py` (modify): compute and write deletion diagnostics (`edges_added`, `edges_deleted`, `net_edge_delta`, `deletion_precision`) in `run_topology_test`.
- `configs/tccig/02.yaml` (create): fork of `01.yaml` with the new blocks.
- `tests/unit/test_tccig_topology_training.py` (create): coverage augmentation, asymmetric anchor, topology-term backprop/sign, warmup, config parsing.
- `tests/unit/test_tccig_test_export.py` (modify) or `tests/unit/test_tccig_deletion_diagnostics.py` (create): deletion-diagnostics unit test.
- `tests/integration/test_tccig_topology_training_stage.py` (create): short 2-epoch run produces `edges_deleted > 0` and writes new diagnostics.

---

## Task 1: Asymmetric residual anchor

**Files:**
- Modify: `tccig/s2gae.py` (near `s2gae_loss_terms`, `tccig/s2gae.py:375-396`)
- Test: `tests/unit/test_tccig_topology_training.py` (create)

**Interfaces:**
- Produces: `asymmetric_residual_anchor(delta_logits: torch.Tensor) -> torch.Tensor` returning `delta_logits.clamp(min=0.0).pow(2).mean()` (penalizes upward pushes only; negative/deletion deltas contribute 0).
- Produces: `s2gae_loss_terms(..., residual_weight: float, anchor_form: str = "symmetric")` — when `anchor_form == "asymmetric_relu"`, `residual_anchor` uses `asymmetric_residual_anchor`; otherwise the existing `delta.pow(2).mean()`. Default keeps existing behavior so 01 is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tccig_topology_training.py
"""Tests for TCCIG Run 02 topology-conditioned training loss."""

from __future__ import annotations

import torch
from tccig.s2gae import asymmetric_residual_anchor


def test_asymmetric_anchor_leaves_deletion_free() -> None:
    negative_delta = torch.tensor([-3.0, -1.0, -0.5])
    assert float(asymmetric_residual_anchor(negative_delta)) == 0.0


def test_asymmetric_anchor_penalizes_upward_push() -> None:
    positive_delta = torch.tensor([2.0, 0.0, -4.0])
    # only +2.0 contributes: (2^2 + 0 + 0) / 3
    assert float(asymmetric_residual_anchor(positive_delta)) == torch.tensor(4.0 / 3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -v`
Expected: FAIL with `ImportError: cannot import name 'asymmetric_residual_anchor'`

- [ ] **Step 3: Write minimal implementation**

```python
# tccig/s2gae.py — add near _bounded_residual / s2gae_loss_terms

def asymmetric_residual_anchor(delta_logits: torch.Tensor) -> torch.Tensor:
    """Penalize only upward (edge-adding) residual pushes; deletion is free.

    The symmetric ``delta.pow(2).mean()`` anchor pulls the refiner toward identity
    and equally discourages negative (edge-deleting) deltas. This one-sided variant
    leaves ``delta < 0`` unpenalized so the topology loss can drive pruning.
    """
    return delta_logits.clamp(min=0.0).pow(2).mean()
```

Then extend `s2gae_loss_terms` to accept `anchor_form` and branch:

```python
def s2gae_loss_terms(
    *,
    refined_logits: torch.Tensor,
    labels: torch.Tensor,
    delta_logits: torch.Tensor,
    loss_config: LossConfig,
    residual_weight: float,
    anchor_form: str = "symmetric",
) -> S2GAELossTerms:
    """Compute supervised denoising BCE plus the configured residual anchor."""
    bce = binary_classification_loss(
        logits=refined_logits,
        labels=labels,
        loss_config=loss_config,
    )
    if anchor_form == "asymmetric_relu":
        residual_anchor = asymmetric_residual_anchor(delta_logits)
    elif anchor_form == "symmetric":
        residual_anchor = delta_logits.pow(2).mean()
    else:
        raise ValueError(f"Unsupported residual anchor form: {anchor_form}")
    weighted_residual_anchor = residual_weight * residual_anchor
    return S2GAELossTerms(
        bce=bce,
        residual_anchor=residual_anchor,
        weighted_residual_anchor=weighted_residual_anchor,
        total=bce + weighted_residual_anchor,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tccig/s2gae.py tests/unit/test_tccig_topology_training.py
git commit -m "feat: add asymmetric residual anchor for tccig refiner"
```

---

## Task 2: Config surface — `residual_anchor` and `topology_training` parsing

**Files:**
- Modify: `tccig/s2gae.py` (`S2GAEConfig` `tccig/s2gae.py:84`, `_parse_config` `tccig/s2gae.py:1373`, `_config_to_json` `tccig/s2gae.py:1457`)
- Test: `tests/unit/test_tccig_topology_training.py`

**Interfaces:**
- Consumes: existing `_parse_config(config: Mapping[str, object]) -> S2GAEConfig`, helpers `_non_negative_float`, `_non_negative_int`, `_positive_int`, `_bool`, `_int_sequence`.
- Produces frozen dataclasses:
  - `S2GAEResidualAnchorConfig(form: str, weight: float)` — `form` in `{"symmetric", "asymmetric_relu"}`.
  - `S2GAETopologyTrainingConfig(enabled: bool, node_sizes: tuple[int, ...], samples_per_size: int, strategy: str, seed: int, coverage_augmentation: bool, topology_weight: float, weights: TopologyLossWeights, warmup_epochs: int, ramp_epochs: int, schedule: str)` where `weights` is `src.topology.finetune_losses.TopologyLossWeights`.
- Produces: `S2GAEConfig` gains `residual_anchor: S2GAEResidualAnchorConfig` and `topology_training: S2GAETopologyTrainingConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tccig_topology_training.py (append)
import pytest
from tccig.s2gae import _parse_config


def _base_refiner_config() -> dict:
    return {
        "encoder": "graphconv",
        "input_dim": 8,
        "embedding_cache_dir": "data/embeddings/esm3_1024",
        "monitor_metric": "val_topology_loss",
        "topology_validation": {"enabled": True, "losses": {"alpha": 1.0, "beta": 8.0, "gamma": 0.5, "delta": 0.1}},
        "optimizer": {"type": "adamw", "lr": 1e-4},
        "residual_anchor": {"form": "asymmetric_relu", "weight": 1.0e-4},
        "topology_training": {
            "enabled": True,
            "node_sizes": [20, 40],
            "samples_per_size": 5,
            "strategy": "mixed",
            "seed": 0,
            "coverage_augmentation": True,
            "topology_weight": 1.0,
            "weights": {"alpha": 1.0, "beta": 8.0, "gamma": 0.5, "delta": 0.1},
            "schedule": {"warmup_epochs": 5, "ramp_epochs": 10, "schedule": "linear"},
        },
    }


def test_parse_config_reads_residual_anchor_and_topology_training() -> None:
    cfg = _parse_config(_base_refiner_config())
    assert cfg.residual_anchor.form == "asymmetric_relu"
    assert cfg.residual_anchor.weight == pytest.approx(1.0e-4)
    assert cfg.topology_training.enabled is True
    assert cfg.topology_training.node_sizes == (20, 40)
    assert cfg.topology_training.topology_weight == pytest.approx(1.0)
    assert cfg.topology_training.weights.beta == pytest.approx(8.0)
    assert cfg.topology_training.warmup_epochs == 5
    assert cfg.topology_training.ramp_epochs == 10


def test_parse_config_defaults_topology_training_disabled() -> None:
    config = _base_refiner_config()
    del config["topology_training"]
    del config["residual_anchor"]
    cfg = _parse_config(config)
    assert cfg.topology_training.enabled is False
    assert cfg.residual_anchor.form == "symmetric"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k parse_config -v`
Expected: FAIL with `AttributeError: 'S2GAEConfig' object has no attribute 'residual_anchor'`

- [ ] **Step 3: Write minimal implementation**

Add imports at the top of `tccig/s2gae.py`:

```python
from src.topology.finetune_losses import TopologyLossWeights, topology_loss_scale
```

Add dataclasses near the other config dataclasses (after `S2GAEOptimizationConfig`, `tccig/s2gae.py:152`):

```python
@dataclass(frozen=True)
class S2GAEResidualAnchorConfig:
    """Residual anchor form and weight for the training loss."""

    form: str
    weight: float


@dataclass(frozen=True)
class S2GAETopologyTrainingConfig:
    """Differentiable topology-loss controls for the training objective."""

    enabled: bool
    node_sizes: tuple[int, ...]
    samples_per_size: int
    strategy: str
    seed: int
    coverage_augmentation: bool
    topology_weight: float
    weights: TopologyLossWeights
    warmup_epochs: int
    ramp_epochs: int
    schedule: str
```

Add fields to `S2GAEConfig` (`tccig/s2gae.py:84`):

```python
    residual_anchor: S2GAEResidualAnchorConfig
    topology_training: S2GAETopologyTrainingConfig
```

Add parser helpers and wire into `_parse_config`'s returned `S2GAEConfig(...)`:

```python
def _parse_residual_anchor_config(raw: object) -> S2GAEResidualAnchorConfig:
    if raw is None:
        return S2GAEResidualAnchorConfig(form="symmetric", weight=0.0)
    if not isinstance(raw, Mapping):
        raise ValueError("refiner.residual_anchor must be a mapping")
    form = str(raw.get("form", "symmetric"))
    if form not in {"symmetric", "asymmetric_relu"}:
        raise ValueError("refiner.residual_anchor.form must be 'symmetric' or 'asymmetric_relu'")
    return S2GAEResidualAnchorConfig(
        form=form,
        weight=_non_negative_float(raw.get("weight", 0.0), "refiner.residual_anchor.weight"),
    )


def _parse_topology_training_config(raw: object) -> S2GAETopologyTrainingConfig:
    if raw is None or not isinstance(raw, Mapping):
        return S2GAETopologyTrainingConfig(
            enabled=False,
            node_sizes=TOPOLOGY_EVAL_NODE_SIZES,
            samples_per_size=20,
            strategy="mixed",
            seed=0,
            coverage_augmentation=True,
            topology_weight=0.0,
            weights=TopologyLossWeights(),
            warmup_epochs=0,
            ramp_epochs=0,
            schedule="linear",
        )
    raw_weights = raw.get("weights", {})
    if not isinstance(raw_weights, Mapping):
        raise ValueError("refiner.topology_training.weights must be a mapping")
    raw_schedule = raw.get("schedule", {})
    if not isinstance(raw_schedule, Mapping):
        raise ValueError("refiner.topology_training.schedule must be a mapping")
    return S2GAETopologyTrainingConfig(
        enabled=_bool(raw.get("enabled", False), "refiner.topology_training.enabled"),
        node_sizes=tuple(
            _int_sequence(
                raw.get("node_sizes", TOPOLOGY_EVAL_NODE_SIZES),
                "refiner.topology_training.node_sizes",
            )
        ),
        samples_per_size=_positive_int(
            raw.get("samples_per_size", 20), "refiner.topology_training.samples_per_size"
        ),
        strategy=str(raw.get("strategy", "mixed")),
        seed=_non_negative_int(raw.get("seed", 0), "refiner.topology_training.seed"),
        coverage_augmentation=_bool(
            raw.get("coverage_augmentation", True),
            "refiner.topology_training.coverage_augmentation",
        ),
        topology_weight=_non_negative_float(
            raw.get("topology_weight", 1.0), "refiner.topology_training.topology_weight"
        ),
        weights=TopologyLossWeights(
            alpha=_non_negative_float(raw_weights.get("alpha", 1.0), "...alpha"),
            beta=_non_negative_float(raw_weights.get("beta", 8.0), "...beta"),
            gamma=_non_negative_float(raw_weights.get("gamma", 0.5), "...gamma"),
            delta=_non_negative_float(raw_weights.get("delta", 0.1), "...delta"),
        ),
        warmup_epochs=_non_negative_int(
            raw_schedule.get("warmup_epochs", 0), "...warmup_epochs"
        ),
        ramp_epochs=_non_negative_int(raw_schedule.get("ramp_epochs", 0), "...ramp_epochs"),
        schedule=str(raw_schedule.get("schedule", "linear")),
    )
```

In `_parse_config`'s `return S2GAEConfig(...)` add:

```python
        residual_anchor=_parse_residual_anchor_config(config.get("residual_anchor")),
        topology_training=_parse_topology_training_config(config.get("topology_training")),
```

Add both blocks to `_config_to_json` (mirror the `topology_validation` serialization style) so checkpoints record them. If `_non_negative_int` does not exist, add it next to `_non_negative_float` (pattern: validate `int`, raise on negative). Confirm `TOPOLOGY_EVAL_NODE_SIZES` is imported in `s2gae.py` (it is imported in `train.py`; add `from src.topology.finetune_data import InternalValidationPlan, TOPOLOGY_EVAL_NODE_SIZES`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k parse_config -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tccig/s2gae.py tests/unit/test_tccig_topology_training.py
git commit -m "feat: parse residual_anchor and topology_training config for tccig"
```

---

## Task 3: Positive-edge coverage augmentation

**Files:**
- Modify: `tccig/train.py` (new helper near `_build_validation_topology_bundle`, `tccig/train.py:382`)
- Test: `tests/unit/test_tccig_topology_training.py`

**Interfaces:**
- Consumes: `sample_topology_evaluation_subgraphs`, `build_internal_validation_plan`, `_expand_chunk_nodes` (from `src.topology.finetune_data`), `InternalValidationPlan`, `nx.Graph`.
- Produces: `augment_plan_for_positive_edge_coverage(*, graph: nx.Graph, base_sampled: dict[int, list[tuple[str, ...]]], node_sizes: Sequence[int], strategy: str, seed: int) -> tuple[dict[int, list[tuple[str, ...]]], dict[str, float | int]]` returning `(augmented_sampled, coverage_stats)` where `coverage_stats = {"base_bucket_count": int, "coverage_bucket_count": int, "positive_edge_coverage": float}` and `positive_edge_coverage == 1.0`.
- Produces: helper `_covered_positive_edges(sampled, graph) -> set[tuple[str, str]]` and `_uncovered_positive_edges(...)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tccig_topology_training.py (append)
import networkx as nx
from tccig.train import augment_plan_for_positive_edge_coverage


def test_coverage_augmentation_covers_isolated_positive_edge() -> None:
    # A dense core plus one far-apart positive edge unlikely to be sampled at size 4.
    graph = nx.Graph()
    core = [f"C{i}" for i in range(6)]
    graph.add_edges_from((core[i], core[j]) for i in range(6) for j in range(i + 1, 6))
    graph.add_edge("FARLEFT", "FARRIGHT")  # connected via no core node
    # Force a base sample that misses the far edge.
    base_sampled = {4: [tuple(sorted(core[:4]))]}

    augmented, stats = augment_plan_for_positive_edge_coverage(
        graph=graph,
        base_sampled=base_sampled,
        node_sizes=[4],
        strategy="BFS",
        seed=0,
    )

    assert stats["positive_edge_coverage"] == 1.0
    assert stats["coverage_bucket_count"] >= 1
    # the previously-missing edge endpoints now appear together in some bucket
    covered = {
        frozenset((a, b))
        for nodes in [n for buckets in augmented.values() for n in buckets]
        for a in nodes
        for b in nodes
        if graph.has_edge(a, b)
    }
    assert frozenset(("FARLEFT", "FARRIGHT")) in covered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k coverage_augmentation -v`
Expected: FAIL with `ImportError: cannot import name 'augment_plan_for_positive_edge_coverage'`

- [ ] **Step 3: Write minimal implementation**

```python
# tccig/train.py — add import
import random
from src.topology.finetune_data import _expand_chunk_nodes  # BFS/DFS/RW expansion

# tccig/train.py — new helpers

def _covered_positive_edges(
    *,
    sampled: Mapping[int, Sequence[tuple[str, ...]]],
    graph: nx.Graph,
) -> set[frozenset[str]]:
    covered: set[frozenset[str]] = set()
    for buckets in sampled.values():
        for nodes in buckets:
            node_set = set(nodes)
            for node_a, node_b in graph.subgraph(node_set).edges():
                covered.add(frozenset((node_a, node_b)))
    return covered


def augment_plan_for_positive_edge_coverage(
    *,
    graph: nx.Graph,
    base_sampled: dict[int, list[tuple[str, ...]]],
    node_sizes: Sequence[int],
    strategy: str,
    seed: int,
) -> tuple[dict[int, list[tuple[str, ...]]], dict[str, float | int]]:
    """Add coverage buckets until every GT positive edge appears in some bucket."""
    augmented = {size: list(buckets) for size, buckets in base_sampled.items()}
    base_bucket_count = sum(len(buckets) for buckets in augmented.values())
    all_positive = {frozenset((a, b)) for a, b in graph.edges()}
    covered = _covered_positive_edges(sampled=augmented, graph=graph)
    uncovered = sorted(
        (tuple(sorted(edge)) for edge in (all_positive - covered)),
    )
    target_size = max(node_sizes)
    rng = random.Random(seed)
    coverage_bucket_count = 0
    for edge in uncovered:
        if frozenset(edge) in _covered_positive_edges(sampled=augmented, graph=graph):
            continue  # already covered by a previously-added coverage bucket
        nodes = _expand_chunk_nodes(
            graph=graph,
            edge_chunk=[edge],
            target_size=target_size,
            strategy=strategy.upper() if strategy.upper() != "MIXED" else "BFS",
            rng=rng,
        )
        augmented.setdefault(target_size, []).append(tuple(sorted(nodes)))
        coverage_bucket_count += 1
    final_covered = _covered_positive_edges(sampled=augmented, graph=graph)
    coverage = 1.0 if not all_positive else len(final_covered & all_positive) / len(all_positive)
    if coverage != 1.0:
        raise ValueError(
            f"positive-edge coverage augmentation failed: coverage={coverage:.6f} < 1.0"
        )
    stats: dict[str, float | int] = {
        "base_bucket_count": base_bucket_count,
        "coverage_bucket_count": coverage_bucket_count,
        "positive_edge_coverage": coverage,
    }
    return augmented, stats
```

Note: `_expand_chunk_nodes` returns the seed edge's endpoints first, then BFS/DFS/RW neighbors, so each coverage bucket is guaranteed to contain both endpoints of its seed edge.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k coverage_augmentation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tccig/train.py tests/unit/test_tccig_topology_training.py
git commit -m "feat: add positive-edge coverage augmentation for tccig train topology plan"
```

---

## Task 4: Train-topology bundle builder (orchestrator)

**Files:**
- Modify: `tccig/train.py` (new `_build_train_topology_bundle`, mirroring `_build_validation_topology_bundle` `tccig/train.py:382`)
- Modify: `tccig/s2gae.py` (`TrainRefinerRequest` `tccig/s2gae.py:114`-area — add fields)
- Test: `tests/unit/test_tccig_topology_training.py`

**Interfaces:**
- Consumes: `load_split_node_ids`, `build_pair_supervision_graph`, `sample_topology_evaluation_subgraphs`, `build_internal_validation_plan`, `augment_plan_for_positive_edge_coverage` (Task 3), `_score_split`, `edges_from_rule`, `CandidatePair`, `SplitBundle`.
- Produces: `_build_train_topology_bundle(*, config, processed_dir, scorer_cfg, runtime, cache_dir, pairwise_input_rule) -> tuple[SplitBundle | None, InternalValidationPlan | None, dict[str, float | int]]` — returns `(bundle, plan, coverage_stats)`; `(None, None, {})` when `topology_training.enabled` is false.
- Produces: `TrainRefinerRequest` gains `train_topology: SplitBundle | None = None` and `train_topology_plan: object | None = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tccig_topology_training.py (append)
from tccig.s2gae import TrainRefinerRequest


def test_train_refiner_request_accepts_train_topology_fields() -> None:
    request = TrainRefinerRequest(
        train=None,  # type: ignore[arg-type]
        validation=None,  # type: ignore[arg-type]
        runtime=None,  # type: ignore[arg-type]
        config={},
        graph_rule=None,  # type: ignore[arg-type]
        train_topology=None,
        train_topology_plan=None,
    )
    assert request.train_topology is None
    assert request.train_topology_plan is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k train_refiner_request -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'train_topology'`

- [ ] **Step 3: Write minimal implementation**

In `tccig/s2gae.py`, extend `TrainRefinerRequest`:

```python
@dataclass(frozen=True)
class TrainRefinerRequest:
    """Concrete request for S2GAE refiner training."""

    train: SplitBundle
    validation: SplitBundle
    runtime: TCCIGRuntime
    config: Mapping[str, object]
    graph_rule: GraphRule
    validation_topology: SplitBundle | None = None
    validation_topology_plan: object | None = None
    train_topology: SplitBundle | None = None
    train_topology_plan: object | None = None
```

In `tccig/train.py`, add `_build_train_topology_bundle` mirroring `_build_validation_topology_bundle` (`tccig/train.py:382`), but: build the graph from the **train** pair file (`human_train_ppi_ratio5_exclusive.txt`) on the **train** node universe; call `sample_topology_evaluation_subgraphs`, then `augment_plan_for_positive_edge_coverage` when `coverage_augmentation` is true; build the plan from the augmented sampled dict; log the coverage stats:

```python
def _build_train_topology_bundle(
    *,
    config: Mapping[str, object],
    processed_dir: Path,
    scorer_cfg: Mapping[str, object],
    runtime: TCCIGRuntime,
    cache_dir: Path,
    pairwise_input_rule: GraphRule,
) -> tuple[SplitBundle | None, object | None, dict[str, float | int]]:
    refiner_cfg = _mapping_section(config, "refiner")
    topo_cfg = refiner_cfg.get("topology_training", {})
    if not isinstance(topo_cfg, Mapping) or not bool(topo_cfg.get("enabled", False)):
        return None, None, {}
    split_path = processed_dir / "human_BFS_split.pkl"
    node_ids = load_split_node_ids(split_path=split_path, split_name="train")
    train_graph = build_pair_supervision_graph(
        pair_path=processed_dir / "human_train_ppi_ratio5_exclusive.txt",
        node_ids=node_ids,
    )
    node_sizes = _int_sequence(
        topo_cfg.get("node_sizes", TOPOLOGY_EVAL_NODE_SIZES),
        "refiner.topology_training.node_sizes",
    )
    seed = _non_negative_int(topo_cfg.get("seed", 0), "refiner.topology_training.seed")
    strategy = str(topo_cfg.get("strategy", "mixed"))
    sampled = sample_topology_evaluation_subgraphs(
        graph=train_graph,
        seed=seed,
        strategy=strategy,
        node_sizes=node_sizes,
        samples_per_size=_positive_int(
            topo_cfg.get("samples_per_size", 20),
            "refiner.topology_training.samples_per_size",
        ),
    )
    coverage_stats: dict[str, float | int] = {}
    if bool(topo_cfg.get("coverage_augmentation", True)):
        sampled, coverage_stats = augment_plan_for_positive_edge_coverage(
            graph=train_graph,
            base_sampled={int(k): list(v) for k, v in sampled.items()},
            node_sizes=node_sizes,
            strategy=strategy,
            seed=seed,
        )
        _LOGGER.info(
            "tccig train topology coverage: base=%d coverage=%d positive_edge_coverage=%.4f",
            coverage_stats["base_bucket_count"],
            coverage_stats["coverage_bucket_count"],
            coverage_stats["positive_edge_coverage"],
        )
    plan = build_internal_validation_plan(graph=train_graph, sampled_subgraphs=sampled)
    pairs = [
        CandidatePair(record.protein_a, record.protein_b)
        for bucket in plan.buckets
        for record in bucket.pair_records
    ]
    probabilities = _score_split(
        split="train_topology",
        pairs=pairs,
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
    )
    pairwise_edges = edges_from_rule(pairs=pairs, probabilities=probabilities, rule=pairwise_input_rule)
    return (
        SplitBundle(
            split="train_topology",
            pairs=pairs,
            pairwise_probabilities=probabilities,
            pairwise_graph_edges=pairwise_edges,
        ),
        plan,
        coverage_stats,
    )
```

Then in `run_tccig_pipeline` (`tccig/train.py:198`), call it and thread into the request:

```python
    train_topology, train_topology_plan, _train_topo_stats = _build_train_topology_bundle(
        config=config,
        processed_dir=processed_dir,
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
        pairwise_input_rule=pairwise_input_rule,
    )
    refiner_state = s2gae.train_refiner(
        TrainRefinerRequest(
            train=train_bundle,
            validation=validation_bundle,
            runtime=runtime,
            config=refiner_cfg,
            graph_rule=graph_rule,
            validation_topology=validation_topology,
            validation_topology_plan=validation_topology_plan,
            train_topology=train_topology,
            train_topology_plan=train_topology_plan,
        )
    )
```

Use the module logger already defined in `train.py` (confirm its name; reuse it as `_LOGGER`). Confirm `_non_negative_int` exists in `train.py` or import/define it.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k train_refiner_request -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tccig/train.py tccig/s2gae.py tests/unit/test_tccig_topology_training.py
git commit -m "feat: build train-topology bundle with coverage-augmented plan"
```

---

## Task 5: Differentiable topology-loss term over a plan

**Files:**
- Modify: `tccig/s2gae.py` (new helper near `s2gae_loss_terms`)
- Test: `tests/unit/test_tccig_topology_training.py`

**Interfaces:**
- Consumes: `compute_topology_losses` (pairwise path), `TopologyLossWeights`, `_SplitGraph`, `S2GAERefiner.encode`/`.decode`, `InternalValidationPlan` (its `buckets[*].pair_records` and `.sampled_subgraphs`).
- Produces: `topology_plan_loss(*, refiner: S2GAERefiner, graph: _SplitGraph, plan: InternalValidationPlan, node_index: Mapping[str, int], weights: TopologyLossWeights, include_clustering_mmd: bool) -> tuple[torch.Tensor, dict[str, float]]` — returns `(loss, components)` where `components` has keys `graph_sim`, `relative_density`, `degree_mmd`, `clustering_mmd`, `total`. For each bucket it builds per-subgraph local node indexing, gathers refined `p_ij = sigmoid(refined_logit)` for that bucket's pairs, builds `target_pair_probabilities` (1.0 for GT positive edges in the bucket's `target_subgraphs`, else 0.0), calls `compute_topology_losses`, and sums across buckets (mean over buckets).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tccig_topology_training.py (append)
import torch
from src.topology.finetune_losses import TopologyLossWeights
from tccig.s2gae import topology_plan_loss


def test_topology_plan_loss_backprops_and_pressures_density_down(make_tiny_refiner_and_plan):
    # Fixture builds: a 4-node over-dense bucket (all pairs p≈0.9) whose true graph has 1 edge.
    refiner, graph, plan, node_index = make_tiny_refiner_and_plan(overdense=True)
    weights = TopologyLossWeights(alpha=1.0, beta=8.0, gamma=0.5, delta=0.0)

    loss, components = topology_plan_loss(
        refiner=refiner,
        graph=graph,
        plan=plan,
        node_index=node_index,
        weights=weights,
        include_clustering_mmd=False,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["relative_density"] > 0.0
    # gradient exists on refiner params (deletion pathway is trainable)
    grads = [p.grad for p in refiner.parameters() if p.grad is not None]
    assert grads, "topology loss did not propagate to refiner parameters"
```

Add a `make_tiny_refiner_and_plan` fixture in the test file building a `S2GAERefiner(input_dim=8, hidden_dim=4, num_layers=2, decoder_hidden_dim=8, decoder_layers=2, dropout=0.0, encoder_aggr="mean", layer_norm=True, residual_scale=4.0)`, a `_SplitGraph` over 4 nodes with all-pairs `pair_index` and `pairwise_probabilities≈0.9`, a one-bucket `InternalValidationPlan` (node_size 4) whose `target_subgraphs[0]` has a single edge, and `node_index` mapping the 4 protein IDs to rows 0-3.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k topology_plan_loss -v`
Expected: FAIL with `ImportError: cannot import name 'topology_plan_loss'`

- [ ] **Step 3: Write minimal implementation**

```python
# tccig/s2gae.py
def topology_plan_loss(
    *,
    refiner: S2GAERefiner,
    graph: _SplitGraph,
    plan: InternalValidationPlan,
    node_index: Mapping[str, int],
    weights: TopologyLossWeights,
    include_clustering_mmd: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Differentiable topology loss over all plan buckets (per-epoch full plan)."""
    device = graph.node_features.device
    hidden_states = refiner.encode(
        node_features=graph.node_features,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
    )
    totals = {"graph_sim": 0.0, "relative_density": 0.0, "degree_mmd": 0.0, "clustering_mmd": 0.0}
    total_loss = graph.node_features.new_zeros(())
    bucket_count = 0
    for bucket in plan.buckets:
        for subgraph_index, nodes in enumerate(bucket.sampled_subgraphs):
            local = {protein: idx for idx, protein in enumerate(nodes)}
            records = [r for r in bucket.pair_records if r.subgraph_index == subgraph_index]
            if len(records) == 0 or len(nodes) < 2:
                continue
            global_pairs = torch.tensor(
                [[node_index[r.protein_a], node_index[r.protein_b]] for r in records],
                dtype=torch.long,
                device=device,
            ).t().contiguous()
            refined_logits, _ = refiner.decode(
                hidden_states=hidden_states,
                pair_index=global_pairs,
                pairwise_probabilities=graph.pairwise_probabilities[
                    _pair_lookup(graph.pair_index, global_pairs)
                ],
            )
            pred = torch.sigmoid(refined_logits)
            target_graph = bucket.target_subgraphs[subgraph_index]
            target = torch.tensor(
                [1.0 if target_graph.has_edge(r.protein_a, r.protein_b) else 0.0 for r in records],
                dtype=torch.float32,
                device=device,
            )
            pair_a = torch.tensor([local[r.protein_a] for r in records], dtype=torch.long, device=device)
            pair_b = torch.tensor([local[r.protein_b] for r in records], dtype=torch.long, device=device)
            terms = compute_topology_losses(
                weights=weights,
                num_nodes=len(nodes),
                pair_index_a=pair_a,
                pair_index_b=pair_b,
                pred_pair_probabilities=pred,
                target_pair_probabilities=target,
                include_clustering_mmd=include_clustering_mmd,
            )
            total_loss = total_loss + terms["total_topology"]
            for key, term_key in (
                ("graph_sim", "graph_similarity"),
                ("relative_density", "relative_density"),
                ("degree_mmd", "degree_mmd"),
                ("clustering_mmd", "clustering_mmd"),
            ):
                totals[key] += float(terms[term_key].detach().item())
            bucket_count += 1
    if bucket_count > 0:
        total_loss = total_loss / bucket_count
        totals = {key: value / bucket_count for key, value in totals.items()}
    components = {**totals, "total": float(total_loss.detach().item())}
    return total_loss, components
```

Add a small `_pair_lookup(all_pairs: torch.Tensor, query: torch.Tensor) -> torch.Tensor` helper that maps each `query` column to its index in `graph.pair_index` via the same `min*N+max` edge-code trick used in `_masked_split_graph` (`tccig/s2gae.py:874`). If a queried pair is absent from `graph.pair_index`, fall back to deriving the raw probability directly — but for the train-topology graph the bundle's pairs are exactly the plan's pairs, so the lookup is total.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k topology_plan_loss -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tccig/s2gae.py tests/unit/test_tccig_topology_training.py
git commit -m "feat: add differentiable topology-plan loss for tccig training"
```

---

## Task 6: Warmup-scaled epoch wiring in `train_refiner`

**Files:**
- Modify: `tccig/s2gae.py` (`train_refiner` epoch loop `tccig/s2gae.py:585`-730; train step anchor form)
- Test: `tests/unit/test_tccig_topology_training.py`

**Interfaces:**
- Consumes: `topology_loss_scale`, `TopologyLossWeightSchedule` (from `src.topology.finetune_losses`), `topology_plan_loss` (Task 5), `asymmetric_residual_anchor`/`s2gae_loss_terms(anchor_form=...)` (Task 1), `S2GAETopologyTrainingConfig`/`S2GAEResidualAnchorConfig` (Task 2), `TrainRefinerRequest.train_topology`/`train_topology_plan` (Task 4).
- Produces: epoch loop computes `scale = topology_loss_scale(epoch=epoch-1, schedule=...)`, runs one full-plan topology backward per epoch scaled by `scale * topology_weight`, accumulates into the optimizer step, and appends `train_topology_loss`, `train_topo_graph_sim`, `train_topo_relative_density`, `train_topo_degree_mmd`, `train_topo_clustering_mmd`, `train_topology_scale` to `epoch_history`.
- Produces: the per-batch `_S2GAESampledTrainStepModule.forward` passes `anchor_form=self.cfg.residual_anchor.form` and `residual_weight=self.cfg.residual_anchor.weight` into `s2gae_loss_terms` (replacing `self.cfg.residual_weight`).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tccig_topology_training.py (append)
from src.topology.finetune_losses import TopologyLossWeightSchedule, topology_loss_scale


def test_topology_loss_scale_zero_during_warmup_then_ramps() -> None:
    schedule = TopologyLossWeightSchedule(warmup_epochs=5, ramp_epochs=10, schedule="linear")
    assert topology_loss_scale(epoch=0, schedule=schedule) == 0.0
    assert topology_loss_scale(epoch=4, schedule=schedule) == 0.0
    assert 0.0 < topology_loss_scale(epoch=9, schedule=schedule) < 1.0
    assert topology_loss_scale(epoch=15, schedule=schedule) == 1.0
```

This pins the warmup contract used in the loop. (The loop itself is covered by the Task 8 integration test, where `train_topology_loss` and `edges_deleted > 0` are asserted on a real short run.)

- [ ] **Step 2: Run test to verify it fails (or passes trivially, then proceed)**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k topology_loss_scale_zero -v`
Expected: PASS (this validates the imported helper contract before wiring it). If it fails, the import path is wrong — fix before continuing.

- [ ] **Step 3: Write the implementation**

In `_S2GAESampledTrainStepModule.forward` (`tccig/s2gae.py:483`), replace the `s2gae_loss_terms(...)` call's `residual_weight=self.cfg.residual_weight` with:

```python
        loss_terms = s2gae_loss_terms(
            refined_logits=refined_logits,
            labels=labels,
            delta_logits=delta,
            loss_config=self.cfg.loss_config,
            residual_weight=self.cfg.residual_anchor.weight,
            anchor_form=self.cfg.residual_anchor.form,
        )
```

In `train_refiner`, build the train-topology graph + node index once before the epoch loop (only when `cfg.topology_training.enabled` and `request.train_topology` / `request.train_topology_plan` are present), mirroring the validation-topology guard at `tccig/s2gae.py:550`:

```python
    train_topology_graph: _SplitGraph | None = None
    train_topology_node_index: dict[str, int] | None = None
    schedule = TopologyLossWeightSchedule(
        warmup_epochs=cfg.topology_training.warmup_epochs,
        ramp_epochs=cfg.topology_training.ramp_epochs,
        schedule=cfg.topology_training.schedule,
    )
    if cfg.topology_training.enabled:
        if request.train_topology is None or request.train_topology_plan is None:
            raise ValueError(
                "refiner.topology_training.enabled requires train_topology inputs"
            )
        train_topology_graph = _build_split_graph(request.train_topology, cfg=cfg, device=device)
        train_topology_node_index = _node_index_from_split_bundle(request.train_topology)
```

Add `_node_index_from_split_bundle` reusing `_collect_node_ids` ordering so the index matches `_build_split_graph`'s node ordering (sorted protein IDs):

```python
def _node_index_from_split_bundle(bundle: SplitBundle) -> dict[str, int]:
    node_ids = _collect_node_ids(pairs=bundle.pairs, graph_edges=bundle.pairwise_graph_edges)
    return {protein_id: index for index, protein_id in enumerate(node_ids)}
```

Inside the epoch loop, after the per-batch loop and `optimizer.step()` calls but within the same epoch (before validation), add one topology backward when enabled:

```python
        topology_components: dict[str, float] | None = None
        topology_scale = 0.0
        if cfg.topology_training.enabled and train_topology_graph is not None:
            assert train_topology_node_index is not None
            assert request.train_topology_plan is not None
            topology_scale = topology_loss_scale(epoch=epoch - 1, schedule=schedule)
            if topology_scale > 0.0 and cfg.topology_training.topology_weight > 0.0:
                optimizer.zero_grad(set_to_none=True)
                topo_loss, topology_components = topology_plan_loss(
                    refiner=_unwrap_refiner(train_step_model, request.runtime.accelerator),
                    graph=train_topology_graph,
                    plan=cast(InternalValidationPlan, request.train_topology_plan),
                    node_index=train_topology_node_index,
                    weights=cfg.topology_training.weights,
                    include_clustering_mmd=cfg.topology_validation.compute_clustering_mmd,
                )
                scaled = topology_scale * cfg.topology_training.topology_weight * topo_loss
                request.runtime.accelerator.backward(scaled)
                if cfg.optimization.gradient_clip_norm is not None:
                    request.runtime.accelerator.clip_grad_norm_(
                        train_step_model.parameters(), cfg.optimization.gradient_clip_norm
                    )
                optimizer.step()
```

After `epoch_history` is built, append the topology fields:

```python
        epoch_history["train_topology_scale"] = topology_scale
        if topology_components is not None:
            epoch_history.update(
                {
                    "train_topology_loss": topology_components["total"],
                    "train_topo_graph_sim": topology_components["graph_sim"],
                    "train_topo_relative_density": topology_components["relative_density"],
                    "train_topo_degree_mmd": topology_components["degree_mmd"],
                    "train_topo_clustering_mmd": topology_components["clustering_mmd"],
                }
            )
```

Notes: keep the topology backward as a **separate** optimizer step (own `zero_grad`/`step`) so the per-batch BCE+anchor accounting and the existing `train_residual_anchor_loss` reduction stay unchanged. The topology backward runs on the unwrapped refiner module (which shares parameters with the DDP-wrapped `train_step_model`); under single-process tests this is exact, and under DDP each rank computes the same deterministic plan, so gradients are consistent. If DDP gradient sync becomes an issue at scale, that is flagged in the spec's open items (memory/throughput check) and handled in the HPC dry run, not here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -v`
Expected: PASS (all topology-training unit tests)

- [ ] **Step 5: Commit**

```bash
git add tccig/s2gae.py tests/unit/test_tccig_topology_training.py
git commit -m "feat: wire warmup-scaled topology loss into tccig train_refiner"
```

---

## Task 7: Deletion diagnostics at test time

**Files:**
- Modify: `tccig/test.py` (`run_topology_test` `tccig/test.py:135`-220)
- Test: `tests/unit/test_tccig_test_export.py` (append) or `tests/unit/test_tccig_deletion_diagnostics.py` (create)

**Interfaces:**
- Consumes: `pairwise_edges` (raw `G_pairwise` from `pairwise_input_rule`), `selected_edges` (refined output from `refined_output_rule`), `pairwise_scores`, `refined_scores`, `table.pairs`, `canonical_edge`.
- Produces: `compute_deletion_diagnostics(*, raw_edges: Sequence[tuple[str, str]], refined_edges: Sequence[tuple[str, str]], pairs: Sequence[CandidatePair], raw_probabilities: Sequence[float], labels: Sequence[int] | None) -> dict[str, float]` returning `edges_added`, `edges_deleted`, `net_edge_delta`, `deletion_precision` (fraction of deleted edges whose pair is a label-0 / below-0.5 raw-probability pair; when labels absent, use raw probability < 0.5 as the "low-confidence/negative" criterion). `run_topology_test` writes these into `topology_metrics.json` under a new `"deletion_diagnostics"` key.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_tccig_deletion_diagnostics.py (create)
"""Tests for TCCIG refined-graph deletion diagnostics."""

from __future__ import annotations

from tccig.prepare import CandidatePair
from tccig.test import compute_deletion_diagnostics


def test_deletion_diagnostics_counts_added_and_deleted() -> None:
    pairs = [CandidatePair("A", "B"), CandidatePair("B", "C"), CandidatePair("A", "C")]
    raw_probabilities = [0.95, 0.40, 0.30]  # raw edges by 0.5: only (A,B)
    raw_edges = [("A", "B")]
    refined_edges = [("B", "C")]  # deleted (A,B); added (B,C)
    labels = [1, 0, 0]

    diagnostics = compute_deletion_diagnostics(
        raw_edges=raw_edges,
        refined_edges=refined_edges,
        pairs=pairs,
        raw_probabilities=raw_probabilities,
        labels=labels,
    )

    assert diagnostics["edges_deleted"] == 1.0
    assert diagnostics["edges_added"] == 1.0
    assert diagnostics["net_edge_delta"] == 0.0
    # the deleted edge (A,B) has label 1 / raw prob 0.95 -> NOT a good deletion
    assert diagnostics["deletion_precision"] == 0.0


def test_deletion_precision_high_when_deleting_negatives() -> None:
    pairs = [CandidatePair("A", "B"), CandidatePair("B", "C")]
    raw_probabilities = [0.40, 0.95]
    raw_edges = [("A", "B"), ("B", "C")]  # (A,B) is a low-confidence raw edge
    refined_edges = [("B", "C")]  # deleted (A,B), a label-0/low-prob pair
    labels = [0, 1]

    diagnostics = compute_deletion_diagnostics(
        raw_edges=raw_edges,
        refined_edges=refined_edges,
        pairs=pairs,
        raw_probabilities=raw_probabilities,
        labels=labels,
    )

    assert diagnostics["edges_deleted"] == 1.0
    assert diagnostics["deletion_precision"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_tccig_deletion_diagnostics.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_deletion_diagnostics'`

- [ ] **Step 3: Write minimal implementation**

```python
# tccig/test.py
def compute_deletion_diagnostics(
    *,
    raw_edges: Sequence[tuple[str, str]],
    refined_edges: Sequence[tuple[str, str]],
    pairs: Sequence[CandidatePair],
    raw_probabilities: Sequence[float],
    labels: Sequence[int] | None,
) -> dict[str, float]:
    """Quantify how the refiner reshapes the raw pairwise graph (adds vs deletes)."""
    raw_set = {canonical_edge(a, b) for a, b in raw_edges}
    refined_set = {canonical_edge(a, b) for a, b in refined_edges}
    deleted = raw_set - refined_set
    added = refined_set - raw_set
    raw_prob_by_edge: dict[tuple[str, str], float] = {}
    label_by_edge: dict[tuple[str, str], int] = {}
    for index, pair in enumerate(pairs):
        edge = canonical_edge(pair.protein_a, pair.protein_b)
        raw_prob_by_edge[edge] = float(raw_probabilities[index])
        if labels is not None:
            label_by_edge[edge] = int(labels[index])
    good_deletions = 0
    for edge in deleted:
        is_negative = label_by_edge.get(edge, 0) == 0 if labels is not None else True
        is_low_conf = raw_prob_by_edge.get(edge, 1.0) < RAW_SCORER_DECISION_THRESHOLD
        if is_negative or is_low_conf:
            good_deletions += 1
    deletion_precision = good_deletions / len(deleted) if deleted else 0.0
    return {
        "edges_added": float(len(added)),
        "edges_deleted": float(len(deleted)),
        "net_edge_delta": float(len(refined_set) - len(raw_set)),
        "deletion_precision": float(deletion_precision),
    }
```

In `run_topology_test`, after `selected_edges` is computed (`tccig/test.py:174`), call the helper and add it to the written `payload`:

```python
        deletion_diagnostics = compute_deletion_diagnostics(
            raw_edges=pairwise_edges,
            refined_edges=selected_edges,
            pairs=table.pairs,
            raw_probabilities=pairwise_scores,
            labels=table.labels,
        )
        payload["deletion_diagnostics"] = deletion_diagnostics
```

(Add the line inside the `if runtime.is_main_process:` block where `payload` is built.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_tccig_deletion_diagnostics.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tccig/test.py tests/unit/test_tccig_deletion_diagnostics.py
git commit -m "feat: write refined-graph deletion diagnostics in tccig topology test"
```

---

## Task 8: Config `02.yaml` + integration test (short run produces deletions)

**Files:**
- Create: `configs/tccig/02.yaml`
- Create: `tests/integration/test_tccig_topology_training_stage.py`

**Interfaces:**
- Consumes: `run_tccig_pipeline` (or `train_refiner` directly with fake fixtures), the `NoOpAccelerator` pattern from `tests/runtime_helpers.py`, and the embedding/PRING fixture setup used by existing integration tests (mirror `tests/integration/test_topology_finetune_stage.py` and any existing TCCIG stage fixtures).
- Produces: a 2-epoch run on a tiny synthetic PRING-style split that completes, writes `train_topology_loss` into `training_summary.json` history, and a topology-test artifact whose `deletion_diagnostics.edges_deleted > 0`.

- [ ] **Step 1: Create `configs/tccig/02.yaml`**

```yaml
# Fork of configs/tccig/01.yaml — adds topology-conditioned training loss (Run 02).
run:
  run_id: "02"
  log_root: logs

data:
  processed_dir: data/PRING/human/BFS

device:
  device: cuda
  backend: ddp
  mixed_precision: bf16
  find_unused_parameters: false

pairwise_scorer:
  model_config_path: configs/v3-1/0517/pair_context_gated_abba_no_cross_s47.yaml
  checkpoint_path: models/v3.1/train/pair_context_gated_abba_no_cross_s47/best_model.pth
  embedding_cache_dir: data/embeddings/esm3_1024
  batch_size: 512
  max_sequence_length: 1024
  score_cache:
    enabled: true

refiner:
  encoder: graphconv
  input_dim: 1536
  hidden_dim: 128
  num_layers: 2
  decoder_hidden_dim: 256
  decoder_layers: 2
  dropout: 0.2
  encoder_aggr: mean
  layer_norm: true
  residual_scale: 4.0
  epochs: 40
  batch_size: 4096
  edge_sampling:
    hard_fraction: 0.7
    easy_anchor_fraction: 0.3
    seed: 0
    reshuffle_easy_each_epoch: true
  loss:
    type: bce_with_logits
    pos_weight: 1.0
    label_smoothing: 0.02
  optimizer:
    type: adamw
    lr: 0.0001
    weight_decay: 0.0001
    beta1: 0.9
    beta2: 0.999
    eps: 1.0e-8
  scheduler:
    type: none
  optimization:
    gradient_clip_norm: 1.0
  residual_anchor:
    form: asymmetric_relu
    weight: 1.0e-4
  topology_training:
    enabled: true
    node_sizes: [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    samples_per_size: 20
    strategy: mixed
    seed: 0
    coverage_augmentation: true
    topology_weight: 1.0
    weights:
      alpha: 1.0
      beta: 8.0
      gamma: 0.5
      delta: 0.1
    schedule:
      warmup_epochs: 5
      ramp_epochs: 10
      schedule: linear
  monitor_metric: val_topology_loss
  topology_validation:
    enabled: true
    node_sizes: [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    samples_per_size: 20
    strategy: mixed
    seed: 0
    inference_batch_size: 4096
    compute_clustering_mmd: true
    losses:
      alpha: 1.0
      beta: 8.0
      gamma: 0.5
      delta: 0.1
  embedding_cache_dir: data/embeddings/esm3_1024
  embedding_index_path: data/embeddings/esm3_1024/index.json
  max_sequence_length: 1024
  checkpoint_path: models/tccig/s2gae/02/best_model.pt

graph_selection:
  pairwise_input_threshold:
    mode: target_precision
    target_precision: 0.8
    split: validation
  refined_output_rule:
    type: threshold
    value: 0.5
  rules:
    - type: threshold
      value: 0.5
```

- [ ] **Step 2: Write the failing integration test**

Mirror the fixture setup of the nearest existing TCCIG stage/integration test (use `rg "run_tccig_pipeline|train_refiner" tests/` to find it; reuse its synthetic embeddings cache, `human_BFS_split.pkl`, `human_train_ppi_ratio5_exclusive.txt`, and PRING test pickles). The test builds a tiny config (override `epochs: 2`, small `node_sizes: [4, 6]`, `samples_per_size: 2`, `warmup_epochs: 0`, `ramp_epochs: 1`, tiny `input_dim`), runs the pipeline with `NoOpAccelerator`, then asserts:

```python
def test_topology_training_run_deletes_edges(tmp_path, tiny_tccig_config):
    result = run_tccig_pipeline(tiny_tccig_config, build_accelerator_fn=lambda: NoOpAccelerator())
    topology_metrics = json.loads(
        (tmp_path / "logs" / "tccig" / "02" / "topology_test" / "topology_metrics.json").read_text()
    )
    diagnostics = topology_metrics["deletion_diagnostics"]
    assert diagnostics["edges_deleted"] > 0.0

    summary = json.loads(
        (tmp_path / "logs" / "tccig" / "02" / "training_summary.json").read_text()
    )
    assert any("train_topology_loss" in entry for entry in summary["history"])
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run python -m pytest tests/integration/test_tccig_topology_training_stage.py -v`
Expected: FAIL — initially on a missing fixture or because `edges_deleted == 0` if wiring is incomplete. Iterate until the assertions are meaningful (the topology loss must be active with `warmup_epochs: 0` so deletion pressure applies from epoch 1).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/integration/test_tccig_topology_training_stage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/tccig/02.yaml tests/integration/test_tccig_topology_training_stage.py
git commit -m "feat: add tccig 02 config and topology-training integration test"
```

---

## Task 9: Full verification sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the full touched-module test suite**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py tests/unit/test_tccig_deletion_diagnostics.py tests/unit/test_tccig_s2gae.py tests/unit/test_tccig_s2gae_validation.py tests/unit/test_tccig_test_export.py tests/integration/test_tccig_topology_training_stage.py -v`
Expected: all PASS. In particular confirm the existing `test_tccig_s2gae.py` still passes (the `anchor_form` default keeps 01 behavior; `s2gae_loss_terms` signature change is backward-compatible via the default argument).

- [ ] **Step 2: Lint and format**

Run: `uv run ruff check --fix . && uv run ruff format .`
Expected: no remaining errors.

- [ ] **Step 3: Type-check**

Run: `uv run mypy src tccig`
Expected: no new errors in `tccig/s2gae.py`, `tccig/train.py`, `tccig/test.py`.

- [ ] **Step 4: Commit any lint/type fixups**

```bash
git add -A
git commit -m "chore: lint and type fixups for tccig topology training"
```

---

## Self-Review

**Spec coverage:**
- Differentiable topology loss in training (Approach 1: density+degree+graph-sim+clustering) → Task 5 (`topology_plan_loss` via `compute_topology_losses` pairwise path), Task 6 (wiring).
- Asymmetric residual anchor (deletion free) → Task 1.
- Warmup ramp → Task 2 (config) + Task 6 (`topology_loss_scale`).
- Per-epoch full-plan topology backward → Task 6 (one backward/epoch over the full plan).
- GT-positive-edge coverage augmentation, log `base_bucket_count`/`coverage_bucket_count`/`positive_edge_coverage`, assert == 1.0 → Task 3 + Task 4.
- Node-bucket sampler reuse, NO `sample_edge_cover_subgraphs` → Task 3/4 use `sample_topology_evaluation_subgraphs` + `_expand_chunk_nodes` only.
- Train-topology plan built once by orchestrator under Accelerate → Task 4.
- Deletion diagnostics (`edges_added`/`edges_deleted`/`net_edge_delta`/`deletion_precision`) → Task 7.
- Logging fields (`train_topology_loss` + components) → Task 6.
- Config `02.yaml` with concrete defaults → Task 8.
- Error handling (fail-fast on coverage gap; degenerate-bucket guard `len(nodes) < 2`) → Task 3 (raise) + Task 5 (skip).
- `monitor_metric` unchanged → preserved; checkpoint selection path untouched.

**Open items deferred to HPC dry run (from spec):** per-epoch full-plan memory/throughput after coverage augmentation, and whether coverage-bucket count needs a cap on extreme-density graphs. These are runtime tuning concerns, surfaced via the logged `coverage_bucket_count` — not blocking for correctness.

**Type consistency:** `topology_plan_loss` returns `(torch.Tensor, dict[str, float])` consumed in Task 6; `compute_topology_losses` keys used are `graph_similarity`, `relative_density`, `degree_mmd`, `clustering_mmd`, `total_topology` (verified against `src/topology/finetune_losses.py:488`). `S2GAETopologyTrainingConfig.weights` is `src.topology.finetune_losses.TopologyLossWeights` used directly by `compute_topology_losses`. `TrainRefinerRequest.train_topology_plan` typed `object | None` and cast to `InternalValidationPlan` at use (matches the existing `validation_topology_plan` pattern).
