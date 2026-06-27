# TCCIG Topology Plan Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache the TCCIG topology "graph bucket" plan to disk so the refiner pipeline skips sampling + coverage augmentation + O(n²) pair materialization on every run after the first.

**Architecture:** A new `src/topology/plan_cache.py` module owns cache-key hashing, JSON load/write, and payload⇄`InternalValidationPlan` (de)serialization (storing node sets + canonical induced edges, never `pair_records`). `tccig/train.py` gets an incremental rewrite of `augment_plan_for_positive_edge_coverage` and a shared `_load_or_build_topology_plan` helper that coordinates DDP ranks (main builds+writes, others barrier then read), mirroring the existing `_score_split` / `_load_score_cache` pattern.

**Tech Stack:** Python 3.10+, networkx, pytest, `uv` for env/test/lint. JSON cache via the existing `write_json` helper.

## Global Constraints

- Environment: run all Python through `uv run` (e.g. `uv run python -m pytest`, `uv run ruff check`, `uv run mypy src`). Bootstrap with `uv sync --group dev` if `.venv/` is missing/stale.
- Python 3.10+; strict type hints, avoid `Any`; absolute imports only (`from src.x import y`); Google-style docstrings; no `print` (use `logging`); handle specific exceptions (no bare `except`); max nesting level 4; functions < 50 lines; files target 200–400 lines.
- Commits: Conventional Commits `<type>: <description>`. Attribution disabled globally; do not add co-author trailers.
- Cache-key correctness is the safety property: a changed graph or any sampling param MUST invalidate the cache. Never serve a stale plan.
- `strategy` must be normalized via `src.topology.finetune_data._normalize_sampling_strategy` before hashing (so `mixed` == `MIXED`).
- Canonical undirected edges use `src.topology.finetune_data._canonical_edge` — import it, do not duplicate.

---

## File Structure

- **Create** `src/topology/plan_cache.py` — cache-key metadata, JSON load/write, payload⇄plan serializers. Single responsibility: persistence + (de)serialization of `InternalValidationPlan`. No DDP, no sampling.
- **Create** `tests/unit/test_topology_plan_cache.py` — unit tests for the module (round-trip, key sensitivity, miss→write→reload, corrupt-JSON fallback).
- **Modify** `tccig/train.py` — (a) rewrite `augment_plan_for_positive_edge_coverage` (~line 410) for incremental coverage; (b) add `_load_or_build_topology_plan` helper; (c) wire it into `_build_train_topology_bundle` (~line 457). Validation bundle wiring is optional/low-priority (Task 6).
- **Modify** `tests/unit/test_tccig_topology_training.py` — add incremental-coverage equivalence tests + `_load_or_build_topology_plan` DDP-coordination tests with a fake runtime.

### Reference signatures (already in the codebase)

```python
# src/topology/finetune_data.py
SUPPORTED_SAMPLING_STRATEGIES = {"BFS", "DFS", "RANDOM_WALK", "MIXED"}  # line 18
def _normalize_sampling_strategy(strategy: str) -> str: ...             # line 281, raises ValueError
def _canonical_edge(node_a: str, node_b: str) -> tuple[str, str]: ...   # line 156
def _expand_chunk_nodes(*, graph, edge_chunk, target_size, strategy, rng) -> tuple[str, ...]: ...
def build_internal_validation_plan(*, graph, sampled_subgraphs) -> InternalValidationPlan: ...

@dataclass(frozen=True)
class InternalValidationPairRecord:        # line 62
    subgraph_index: int
    pair_index_a: int
    pair_index_b: int
    protein_a: str
    protein_b: str

@dataclass(frozen=True)
class InternalValidationNodeBucketPlan:     # line 72
    node_size: int
    sampled_subgraphs: tuple[tuple[str, ...], ...]
    target_subgraphs: tuple[nx.Graph, ...]
    pair_records: tuple[InternalValidationPairRecord, ...]

@dataclass(frozen=True)
class InternalValidationPlan:               # line 82
    buckets: tuple[InternalValidationNodeBucketPlan, ...]
    protein_ids: frozenset[str]
    total_subgraphs: int
    total_pairs: int

# tccig/prepare.py
def write_json(path: Path, payload: dict[str, Any]) -> None: ...  # line 190, indent=2, sort_keys=True
@dataclass
class TCCIGRuntime:                          # line 137
    accelerator: AcceleratorLike
    device: str; backend: str; mixed_precision: str
    is_distributed: bool; rank: int; local_rank: int; world_size: int
    is_main_process: bool

# tccig/train.py
def _runtime_barrier(runtime: TCCIGRuntime) -> None: ...   # calls accelerator.wait_for_everyone()
def _covered_positive_edges(*, sampled, graph) -> set[frozenset[str]]: ...  # line 395
def augment_plan_for_positive_edge_coverage(*, graph, base_sampled, node_sizes, strategy, seed)
    -> tuple[dict[int, list[tuple[str, ...]]], dict[str, float | int]]: ...  # line 410
```

---

## Task 1: Plan-cache serialization core (`plan_to_payload` / `payload_to_plan`)

**Files:**
- Create: `src/topology/plan_cache.py`
- Test: `tests/unit/test_topology_plan_cache.py`

**Interfaces:**
- Consumes: `InternalValidationPlan`, `InternalValidationNodeBucketPlan`, `InternalValidationPairRecord`, `_canonical_edge` from `src.topology.finetune_data`.
- Produces:
  - `plan_to_payload(plan: InternalValidationPlan) -> dict[str, object]`
  - `payload_to_plan(payload: Mapping[str, object], *, graph: nx.Graph) -> InternalValidationPlan`
  - Payload schema: `{"version": 1, "buckets": [{"node_size": int, "sampled_subgraphs": [[node,...],...], "target_edges": [[[a,b],...],...]}], "total_subgraphs": int, "total_pairs": int}`. `target_edges[i]` is the sorted canonical induced-edge list for `sampled_subgraphs[i]`. `pair_records` are NOT stored.

- [ ] **Step 1: Write the failing round-trip test**

```python
# tests/unit/test_topology_plan_cache.py
"""Tests for the TCCIG topology plan cache."""

from __future__ import annotations

import networkx as nx
import pytest

from src.topology.finetune_data import build_internal_validation_plan
from src.topology.plan_cache import payload_to_plan, plan_to_payload


def _toy_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(
        [("a", "b"), ("b", "c"), ("c", "d"), ("a", "c"), ("d", "e"), ("e", "a")]
    )
    return graph


def _toy_sampled() -> dict[int, list[tuple[str, ...]]]:
    return {3: [("a", "b", "c"), ("c", "d", "e")], 4: [("a", "b", "c", "d")]}


def _edge_set(graph: nx.Graph) -> set[frozenset[str]]:
    return {frozenset(edge) for edge in graph.edges()}


def test_payload_round_trip_preserves_plan() -> None:
    graph = _toy_graph()
    plan = build_internal_validation_plan(graph=graph, sampled_subgraphs=_toy_sampled())

    restored = payload_to_plan(plan_to_payload(plan), graph=graph)

    assert restored.total_subgraphs == plan.total_subgraphs
    assert restored.total_pairs == plan.total_pairs
    assert restored.protein_ids == plan.protein_ids
    assert len(restored.buckets) == len(plan.buckets)
    for restored_bucket, original_bucket in zip(restored.buckets, plan.buckets, strict=True):
        assert restored_bucket.node_size == original_bucket.node_size
        assert restored_bucket.sampled_subgraphs == original_bucket.sampled_subgraphs
        assert restored_bucket.pair_records == original_bucket.pair_records
        for restored_sub, original_sub in zip(
            restored_bucket.target_subgraphs, original_bucket.target_subgraphs, strict=True
        ):
            assert set(restored_sub.nodes()) == set(original_sub.nodes())
            assert _edge_set(restored_sub) == _edge_set(original_sub)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_topology_plan_cache.py::test_payload_round_trip_preserves_plan -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.topology.plan_cache'`.

- [ ] **Step 3: Implement the serializers**

```python
# src/topology/plan_cache.py
"""Disk cache for the TCCIG topology internal-validation plan.

Stores a JSON-safe payload (node sets + canonical induced edges) instead of the
live ``InternalValidationPlan`` (which holds ``nx.Graph`` objects). ``pair_records``
are regenerated on load by deterministic upper-triangle enumeration.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import networkx as nx

from src.topology.finetune_data import (
    InternalValidationNodeBucketPlan,
    InternalValidationPairRecord,
    InternalValidationPlan,
    _canonical_edge,
)

LOGGER = logging.getLogger(__name__)

PAYLOAD_VERSION = 1


def _bucket_target_edges(subgraph: nx.Graph) -> list[list[str]]:
    """Return sorted canonical induced edges of one target subgraph."""
    edges = {_canonical_edge(node_a, node_b) for node_a, node_b in subgraph.edges()}
    return [list(edge) for edge in sorted(edges)]


def plan_to_payload(plan: InternalValidationPlan) -> dict[str, object]:
    """Serialize an ``InternalValidationPlan`` to a JSON-safe payload."""
    buckets: list[dict[str, object]] = []
    for bucket in plan.buckets:
        target_edges = [_bucket_target_edges(sub) for sub in bucket.target_subgraphs]
        buckets.append(
            {
                "node_size": bucket.node_size,
                "sampled_subgraphs": [list(nodes) for nodes in bucket.sampled_subgraphs],
                "target_edges": target_edges,
            }
        )
    return {
        "version": PAYLOAD_VERSION,
        "buckets": buckets,
        "total_subgraphs": plan.total_subgraphs,
        "total_pairs": plan.total_pairs,
    }


def _rebuild_target_subgraph(nodes: Sequence[str], edges: Sequence[Sequence[str]]) -> nx.Graph:
    """Reconstruct a node-induced target subgraph from cached data."""
    subgraph = nx.Graph()
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from((edge[0], edge[1]) for edge in edges)
    return subgraph


def _pair_records_for(subgraph_index: int, nodes: Sequence[str]) -> list[InternalValidationPairRecord]:
    """Regenerate the upper-triangle pair records for one sampled subgraph."""
    records: list[InternalValidationPairRecord] = []
    for index_a, protein_a in enumerate(nodes):
        for index_b in range(index_a + 1, len(nodes)):
            records.append(
                InternalValidationPairRecord(
                    subgraph_index=subgraph_index,
                    pair_index_a=index_a,
                    pair_index_b=index_b,
                    protein_a=protein_a,
                    protein_b=nodes[index_b],
                )
            )
    return records


def payload_to_plan(payload: Mapping[str, object], *, graph: nx.Graph) -> InternalValidationPlan:
    """Rehydrate an ``InternalValidationPlan`` from a cached payload.

    ``graph`` is accepted for signature symmetry with the builder; target
    subgraphs are rebuilt from cached induced edges and do not read ``graph``.
    """
    raw_buckets = payload.get("buckets")
    if not isinstance(raw_buckets, list):
        raise ValueError("plan payload missing 'buckets' list")
    buckets: list[InternalValidationNodeBucketPlan] = []
    protein_ids: set[str] = set()
    total_subgraphs = 0
    total_pairs = 0
    for raw_bucket in raw_buckets:
        node_sets = tuple(tuple(nodes) for nodes in raw_bucket["sampled_subgraphs"])
        target_edges = raw_bucket["target_edges"]
        target_subgraphs = tuple(
            _rebuild_target_subgraph(nodes, edges)
            for nodes, edges in zip(node_sets, target_edges, strict=True)
        )
        pair_records: list[InternalValidationPairRecord] = []
        for subgraph_index, nodes in enumerate(node_sets):
            protein_ids.update(nodes)
            pair_records.extend(_pair_records_for(subgraph_index, nodes))
        buckets.append(
            InternalValidationNodeBucketPlan(
                node_size=int(raw_bucket["node_size"]),
                sampled_subgraphs=node_sets,
                target_subgraphs=target_subgraphs,
                pair_records=tuple(pair_records),
            )
        )
        total_subgraphs += len(node_sets)
        total_pairs += len(pair_records)
    return InternalValidationPlan(
        buckets=tuple(buckets),
        protein_ids=frozenset(protein_ids),
        total_subgraphs=total_subgraphs,
        total_pairs=total_pairs,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_topology_plan_cache.py::test_payload_round_trip_preserves_plan -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/topology/plan_cache.py tests/unit/test_topology_plan_cache.py
git commit -m "feat: add topology plan payload serializers"
```

---

## Task 2: Cache-key metadata (`plan_payload_metadata`)

**Files:**
- Modify: `src/topology/plan_cache.py`
- Test: `tests/unit/test_topology_plan_cache.py`

**Interfaces:**
- Consumes: `_normalize_sampling_strategy`, `_canonical_edge` from `src.topology.finetune_data`.
- Produces: `plan_payload_metadata(*, split: str, graph: nx.Graph, node_sizes: Sequence[int], samples_per_size: int, seed: int, strategy: str, coverage_augmentation: bool) -> dict[str, object]`. Returns `{"version": 1, "split", "node_count", "edge_count", "graph_hash", "node_sizes": [...], "samples_per_size", "seed", "strategy": <normalized>, "coverage_augmentation"}`.

- [ ] **Step 1: Write the failing key-sensitivity tests**

```python
# tests/unit/test_topology_plan_cache.py (append)
from src.topology.plan_cache import plan_payload_metadata


def _meta(graph: nx.Graph, **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "split": "train_topology",
        "graph": graph,
        "node_sizes": [3, 4],
        "samples_per_size": 2,
        "seed": 0,
        "strategy": "mixed",
        "coverage_augmentation": True,
    }
    kwargs.update(overrides)
    return plan_payload_metadata(**kwargs)  # type: ignore[arg-type]


def test_metadata_normalizes_strategy_case() -> None:
    graph = _toy_graph()
    assert _meta(graph, strategy="mixed") == _meta(graph, strategy="MIXED")


def test_metadata_stable_across_edge_insertion_order() -> None:
    graph_a = nx.Graph()
    graph_a.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
    graph_b = nx.Graph()
    graph_b.add_edges_from([("c", "d"), ("a", "b"), ("b", "c")])
    assert _meta(graph_a)["graph_hash"] == _meta(graph_b)["graph_hash"]


def test_metadata_changes_on_each_input() -> None:
    graph = _toy_graph()
    base = _meta(graph)
    assert _meta(graph, seed=1) != base
    assert _meta(graph, samples_per_size=3) != base
    assert _meta(graph, node_sizes=[3, 5]) != base
    assert _meta(graph, strategy="bfs") != base
    assert _meta(graph, coverage_augmentation=False) != base
    bigger = _toy_graph()
    bigger.add_edge("a", "z")  # new edge + new node
    assert _meta(bigger)["graph_hash"] != base["graph_hash"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/test_topology_plan_cache.py -k metadata -v`
Expected: FAIL — `ImportError: cannot import name 'plan_payload_metadata'`.

- [ ] **Step 3: Implement `plan_payload_metadata`**

Add to `src/topology/plan_cache.py` (add `import hashlib` and `Sequence` is already imported; add `_normalize_sampling_strategy` to the `finetune_data` import):

```python
def _graph_hash(graph: nx.Graph) -> str:
    """Stable digest over canonical edges plus the node-id set."""
    digest = hashlib.sha256()
    for node_a, node_b in sorted(
        _canonical_edge(node_a, node_b) for node_a, node_b in graph.edges()
    ):
        digest.update(node_a.encode("utf-8"))
        digest.update(b"\0")
        digest.update(node_b.encode("utf-8"))
        digest.update(b"\n")
    digest.update(b"||nodes||")
    for node in sorted(graph.nodes()):
        digest.update(str(node).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def plan_payload_metadata(
    *,
    split: str,
    graph: nx.Graph,
    node_sizes: Sequence[int],
    samples_per_size: int,
    seed: int,
    strategy: str,
    coverage_augmentation: bool,
) -> dict[str, object]:
    """Build the strict cache key for a topology plan payload."""
    return {
        "version": PAYLOAD_VERSION,
        "split": split,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "graph_hash": _graph_hash(graph),
        "node_sizes": [int(size) for size in node_sizes],
        "samples_per_size": int(samples_per_size),
        "seed": int(seed),
        "strategy": _normalize_sampling_strategy(strategy),
        "coverage_augmentation": bool(coverage_augmentation),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/test_topology_plan_cache.py -k metadata -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/topology/plan_cache.py tests/unit/test_topology_plan_cache.py
git commit -m "feat: add topology plan cache key metadata"
```

---

## Task 3: Cache load/write with mismatch + corrupt-JSON fallback

**Files:**
- Modify: `src/topology/plan_cache.py`
- Test: `tests/unit/test_topology_plan_cache.py`

**Interfaces:**
- Consumes: `write_json` from `tccig.prepare`; `plan_payload_metadata`, `plan_to_payload`, `payload_to_plan` from this module.
- Produces:
  - `load_plan_cache(*, cache_dir: Path, split: str, metadata: Mapping[str, object]) -> dict[str, object] | None` — returns the stored payload on a metadata match; `None` on missing file, corrupt JSON, or metadata mismatch.
  - `write_plan_cache(*, cache_dir: Path, split: str, metadata: Mapping[str, object], payload: Mapping[str, object]) -> None` — writes `cache_dir/plans/{split}.json` (payload+metadata) and `cache_dir/manifests/{split}_plan.json` (metadata).

- [ ] **Step 1: Write the failing load/write tests**

```python
# tests/unit/test_topology_plan_cache.py (append)
from pathlib import Path

from src.topology.plan_cache import load_plan_cache, write_plan_cache


def test_write_then_load_returns_payload(tmp_path: Path) -> None:
    graph = _toy_graph()
    plan = build_internal_validation_plan(graph=graph, sampled_subgraphs=_toy_sampled())
    metadata = _meta(graph)
    payload = plan_to_payload(plan)

    write_plan_cache(cache_dir=tmp_path, split="train_topology", metadata=metadata, payload=payload)

    assert (tmp_path / "plans" / "train_topology.json").exists()
    assert (tmp_path / "manifests" / "train_topology_plan.json").exists()
    loaded = load_plan_cache(cache_dir=tmp_path, split="train_topology", metadata=metadata)
    assert loaded is not None
    restored = payload_to_plan(loaded, graph=graph)
    assert restored.total_pairs == plan.total_pairs


def test_load_returns_none_on_metadata_mismatch(tmp_path: Path) -> None:
    graph = _toy_graph()
    payload = plan_to_payload(build_internal_validation_plan(graph=graph, sampled_subgraphs=_toy_sampled()))
    write_plan_cache(cache_dir=tmp_path, split="train_topology", metadata=_meta(graph), payload=payload)

    stale = load_plan_cache(
        cache_dir=tmp_path, split="train_topology", metadata=_meta(graph, seed=999)
    )
    assert stale is None


def test_load_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_plan_cache(cache_dir=tmp_path, split="train_topology", metadata=_meta(_toy_graph())) is None


def test_load_returns_none_on_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / "plans" / "train_topology.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert load_plan_cache(cache_dir=tmp_path, split="train_topology", metadata=_meta(_toy_graph())) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/test_topology_plan_cache.py -k "load or write" -v`
Expected: FAIL — `ImportError: cannot import name 'load_plan_cache'`.

- [ ] **Step 3: Implement load/write**

Add `import json` to `src/topology/plan_cache.py` and append:

```python
def _plans_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / "plans" / f"{split}.json"


def load_plan_cache(
    *, cache_dir: Path, split: str, metadata: Mapping[str, object]
) -> dict[str, object] | None:
    """Return the cached payload when present and the metadata matches."""
    path = _plans_path(cache_dir, split)
    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Discarding corrupt topology plan cache at %s", path)
        return None
    if not isinstance(stored, Mapping) or stored.get("metadata") != dict(metadata):
        return None
    payload = stored.get("payload")
    if not isinstance(payload, dict):
        return None
    return payload


def write_plan_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
    payload: Mapping[str, object],
) -> None:
    """Persist the plan payload and a manifest copy of its cache key."""
    write_json(
        _plans_path(cache_dir, split),
        {"metadata": dict(metadata), "payload": dict(payload)},
    )
    write_json(cache_dir / "manifests" / f"{split}_plan.json", dict(metadata))
```

Add the import near the top: `from tccig.prepare import write_json`.

> Note: `write_json` runs `json.dumps(..., sort_keys=True)`. Metadata values must be JSON-native (str/int/bool/list) — they are, by Task 2's construction. The `stored.get("metadata") != dict(metadata)` compare works because both sides are plain dicts of JSON-native values.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/test_topology_plan_cache.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Typecheck + lint the new module**

Run: `uv run ruff check src/topology/plan_cache.py tests/unit/test_topology_plan_cache.py && uv run mypy src/topology/plan_cache.py`
Expected: no errors. Fix any inline before committing.

- [ ] **Step 6: Commit**

```bash
git add src/topology/plan_cache.py tests/unit/test_topology_plan_cache.py
git commit -m "feat: add topology plan cache load and write helpers"
```

---

## Task 4: Incremental coverage augmentation rewrite

**Files:**
- Modify: `tccig/train.py` (`augment_plan_for_positive_edge_coverage`, ~line 410; `_covered_positive_edges` stays at ~line 395)
- Test: `tests/unit/test_tccig_topology_training.py`

**Interfaces:**
- Consumes: `_expand_chunk_nodes` (already imported), `_canonical_edge` (NEW import from `src.topology.finetune_data`), existing `_covered_positive_edges`.
- Produces: `augment_plan_for_positive_edge_coverage` keeps its exact signature and return contract: `tuple[dict[int, list[tuple[str, ...]]], dict[str, float | int]]` with stats keys `base_bucket_count`, `coverage_bucket_count`, `positive_edge_coverage`.

- [ ] **Step 1: Write the failing equivalence + coverage tests**

These tests pin behavior. Because the rewrite must preserve semantics, the first test captures the CURRENT output as the reference, so it should PASS once the rewrite is correct (and would fail if the rewrite changed which buckets get selected). Add to `tests/unit/test_tccig_topology_training.py`:

```python
# tests/unit/test_tccig_topology_training.py (append)
import networkx as nx

from tccig.train import augment_plan_for_positive_edge_coverage


def _coverage_graph() -> nx.Graph:
    graph = nx.Graph()
    # two clusters joined by a bridge so some positives need coverage buckets
    graph.add_edges_from(
        [("a", "b"), ("b", "c"), ("a", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("d", "f")]
    )
    return graph


def test_coverage_augmentation_reaches_full_coverage() -> None:
    graph = _coverage_graph()
    augmented, stats = augment_plan_for_positive_edge_coverage(
        graph=graph,
        base_sampled={4: [("a", "b", "c", "d")]},
        node_sizes=[4],
        strategy="bfs",
        seed=0,
    )
    assert stats["positive_edge_coverage"] == 1.0
    # every positive edge appears in some bucket
    covered: set[frozenset[str]] = set()
    for buckets in augmented.values():
        for nodes in buckets:
            for edge in graph.subgraph(set(nodes)).edges():
                covered.add(frozenset(edge))
    all_positive = {frozenset(edge) for edge in graph.edges()}
    assert covered >= all_positive


def test_coverage_augmentation_is_deterministic() -> None:
    graph = _coverage_graph()
    first, first_stats = augment_plan_for_positive_edge_coverage(
        graph=graph, base_sampled={4: [("a", "b", "c", "d")]}, node_sizes=[4], strategy="bfs", seed=0
    )
    second, second_stats = augment_plan_for_positive_edge_coverage(
        graph=graph, base_sampled={4: [("a", "b", "c", "d")]}, node_sizes=[4], strategy="bfs", seed=0
    )
    assert first == second
    assert first_stats == second_stats


def test_coverage_augmentation_no_extra_buckets_when_already_covered() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
    augmented, stats = augment_plan_for_positive_edge_coverage(
        graph=graph, base_sampled={3: [("a", "b", "c")]}, node_sizes=[3], strategy="bfs", seed=0
    )
    assert stats["coverage_bucket_count"] == 0
    assert stats["positive_edge_coverage"] == 1.0
```

- [ ] **Step 2: Run tests against the CURRENT implementation**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k coverage_augmentation -v`
Expected: PASS against the existing implementation (these assert externally-observable behavior the rewrite must preserve). This establishes the baseline before refactoring.

- [ ] **Step 3: Rewrite `augment_plan_for_positive_edge_coverage` for incremental coverage**

Add `_canonical_edge` to the `from src.topology.finetune_data import (...)` block in `tccig/train.py`. Replace the body of `augment_plan_for_positive_edge_coverage` (keep signature + docstring) with:

```python
    augmented: dict[int, list[tuple[str, ...]]] = {
        size: list(buckets) for size, buckets in base_sampled.items()
    }
    base_bucket_count = sum(len(buckets) for buckets in augmented.values())
    all_positive = {_canonical_edge(node_a, node_b) for node_a, node_b in graph.edges()}
    target_size = max(node_sizes)
    normalized_strategy = strategy.upper()
    if normalized_strategy not in {"BFS", "DFS", "RANDOM_WALK"}:
        normalized_strategy = "BFS"
    rng = random.Random(seed)

    # Maintain `covered` incrementally: seed it once from the base buckets, then
    # extend it with each newly added coverage bucket's induced edges. This avoids
    # re-scanning every bucket per uncovered edge (the previous quadratic hotspot).
    covered: set[tuple[str, str]] = {
        _canonical_edge(node_a, node_b)
        for edge in _covered_positive_edges(sampled=augmented, graph=graph)
        for node_a, node_b in [tuple(sorted(edge))]
    }
    coverage_bucket_count = 0
    for edge in sorted(all_positive - covered):
        if edge in covered:
            continue  # drained by a previously added coverage bucket
        nodes = _expand_chunk_nodes(
            graph=graph,
            edge_chunk=[(edge[0], edge[1])],
            target_size=target_size,
            strategy=normalized_strategy,
            rng=rng,
        )
        augmented.setdefault(target_size, []).append(tuple(sorted(nodes)))
        for node_a, node_b in graph.subgraph(set(nodes)).edges():
            covered.add(_canonical_edge(node_a, node_b))
        coverage_bucket_count += 1

    matched_positive = len(covered & all_positive)
    coverage = 1.0 if not all_positive else matched_positive / len(all_positive)
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

> Why the `covered` seed loop converts frozensets to canonical tuples: `_covered_positive_edges` returns `set[frozenset[str]]`; we normalize to `_canonical_edge` tuples so membership checks against `all_positive` (also canonical tuples) are consistent and sort-stable.

- [ ] **Step 4: Run the coverage tests to verify they still pass**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k coverage_augmentation -v`
Expected: PASS (3 tests) — behavior preserved after the rewrite.

- [ ] **Step 5: Run the full topology-training unit file to check for regressions**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -v`
Expected: PASS.

- [ ] **Step 6: Lint + typecheck**

Run: `uv run ruff check tccig/train.py tests/unit/test_tccig_topology_training.py && uv run mypy src`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add tccig/train.py tests/unit/test_tccig_topology_training.py
git commit -m "perf: maintain covered set incrementally in coverage augmentation"
```

---

## Task 5: `_load_or_build_topology_plan` DDP-coordinated helper

**Files:**
- Modify: `tccig/train.py` (add helper; add imports from `src.topology.plan_cache`)
- Test: `tests/unit/test_tccig_topology_training.py`

**Interfaces:**
- Consumes: `plan_payload_metadata`, `plan_to_payload`, `payload_to_plan`, `load_plan_cache`, `write_plan_cache` from `src.topology.plan_cache`; `_runtime_barrier`, `TCCIGRuntime`.
- Produces:
  ```python
  def _load_or_build_topology_plan(
      *,
      split: str,
      graph: nx.Graph,
      node_sizes: Sequence[int],
      samples_per_size: int,
      seed: int,
      strategy: str,
      coverage_augmentation: bool,
      runtime: TCCIGRuntime,
      cache_dir: Path,
      build_fn: Callable[[], tuple[InternalValidationPlan, dict[str, float | int]]],
  ) -> tuple[InternalValidationPlan, dict[str, float | int]]
  ```
  `build_fn` returns `(plan, coverage_stats)` and is invoked ONLY on the main rank on a cache miss. Coverage stats are persisted in the payload under key `"coverage_stats"` so a hit returns the same stats. Returns the rehydrated plan + stats.

- [ ] **Step 1: Write the failing coordination tests with a fake runtime**

```python
# tests/unit/test_tccig_topology_training.py (append)
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.topology.finetune_data import build_internal_validation_plan
from tccig.train import _load_or_build_topology_plan


def _fake_runtime(*, is_main_process: bool) -> object:
    accelerator = SimpleNamespace(wait_for_everyone=lambda: None)
    return SimpleNamespace(accelerator=accelerator, is_main_process=is_main_process)


def _plan_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("a", "c")])
    return graph


def test_load_or_build_writes_then_reuses_cache(tmp_path: Path) -> None:
    graph = _plan_graph()
    calls = {"n": 0}

    def build_fn() -> tuple[object, dict[str, float | int]]:
        calls["n"] += 1
        plan = build_internal_validation_plan(
            graph=graph, sampled_subgraphs={3: [("a", "b", "c")]}
        )
        return plan, {"base_bucket_count": 1, "coverage_bucket_count": 0, "positive_edge_coverage": 1.0}

    common = dict(
        split="train_topology",
        graph=graph,
        node_sizes=[3],
        samples_per_size=1,
        seed=0,
        strategy="bfs",
        coverage_augmentation=True,
        runtime=_fake_runtime(is_main_process=True),
        cache_dir=tmp_path,
        build_fn=build_fn,
    )
    plan_first, stats_first = _load_or_build_topology_plan(**common)
    plan_second, stats_second = _load_or_build_topology_plan(**common)

    assert calls["n"] == 1  # second call served from cache, build_fn not re-run
    assert stats_first == stats_second
    assert plan_second.total_pairs == plan_first.total_pairs


def test_load_or_build_non_main_rank_raises_when_cache_missing(tmp_path: Path) -> None:
    graph = _plan_graph()

    def build_fn() -> tuple[object, dict[str, float | int]]:
        raise AssertionError("build_fn must not run on a non-main rank")

    with pytest.raises(RuntimeError, match="topology plan cache was not written"):
        _load_or_build_topology_plan(
            split="train_topology",
            graph=graph,
            node_sizes=[3],
            samples_per_size=1,
            seed=0,
            strategy="bfs",
            coverage_augmentation=True,
            runtime=_fake_runtime(is_main_process=False),
            cache_dir=tmp_path,
            build_fn=build_fn,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k load_or_build -v`
Expected: FAIL — `ImportError: cannot import name '_load_or_build_topology_plan'`.

- [ ] **Step 3: Implement the helper**

Add imports to `tccig/train.py`:

```python
from src.topology.plan_cache import (
    load_plan_cache,
    plan_payload_metadata,
    plan_to_payload,
    payload_to_plan,
    write_plan_cache,
)
```

Add `InternalValidationPlan` to the `from src.topology.finetune_data import (...)` block. Then add the helper near `_score_split`:

```python
def _load_or_build_topology_plan(
    *,
    split: str,
    graph: nx.Graph,
    node_sizes: Sequence[int],
    samples_per_size: int,
    seed: int,
    strategy: str,
    coverage_augmentation: bool,
    runtime: TCCIGRuntime,
    cache_dir: Path,
    build_fn: Callable[[], tuple[InternalValidationPlan, dict[str, float | int]]],
) -> tuple[InternalValidationPlan, dict[str, float | int]]:
    """Load the topology plan from cache or build it once on the main rank.

    Mirrors ``_score_split``: all ranks attempt the load; on a miss only the main
    rank builds and writes, then a barrier lets every rank read the result.
    """
    metadata = plan_payload_metadata(
        split=split,
        graph=graph,
        node_sizes=node_sizes,
        samples_per_size=samples_per_size,
        seed=seed,
        strategy=strategy,
        coverage_augmentation=coverage_augmentation,
    )
    cached = load_plan_cache(cache_dir=cache_dir, split=split, metadata=metadata)
    if cached is not None:
        return payload_to_plan(cached, graph=graph), _coverage_stats_from_payload(cached)

    if runtime.is_main_process:
        plan, coverage_stats = build_fn()
        payload = plan_to_payload(plan)
        payload["coverage_stats"] = dict(coverage_stats)
        write_plan_cache(cache_dir=cache_dir, split=split, metadata=metadata, payload=payload)
    _runtime_barrier(runtime)

    reloaded = load_plan_cache(cache_dir=cache_dir, split=split, metadata=metadata)
    if reloaded is None:
        raise RuntimeError(f"topology plan cache was not written for split={split}")
    return payload_to_plan(reloaded, graph=graph), _coverage_stats_from_payload(reloaded)


def _coverage_stats_from_payload(payload: Mapping[str, object]) -> dict[str, float | int]:
    """Extract persisted coverage stats, defaulting to an empty contract."""
    raw = payload.get("coverage_stats", {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value for key, value in raw.items()}  # type: ignore[misc]
```

> `coverage_stats` is stored INSIDE the payload (not the metadata) so it does not affect the cache key but still round-trips. `plan_to_payload` ignores unknown keys on load, and `payload_to_plan` only reads `buckets`/totals, so the extra key is inert there.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k load_or_build -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint + typecheck**

Run: `uv run ruff check tccig/train.py && uv run mypy src`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add tccig/train.py tests/unit/test_tccig_topology_training.py
git commit -m "feat: add DDP-coordinated topology plan cache helper"
```

---

## Task 6: Wire the cache into `_build_train_topology_bundle`

**Files:**
- Modify: `tccig/train.py` (`_build_train_topology_bundle`, ~line 457)
- Test: `tests/unit/test_tccig_topology_training.py`

**Interfaces:**
- Consumes: `_load_or_build_topology_plan`, `sample_topology_evaluation_subgraphs`, `augment_plan_for_positive_edge_coverage`, `build_internal_validation_plan`.
- Produces: `_build_train_topology_bundle` unchanged signature/return: `tuple[SplitBundle | None, object | None, dict[str, float | int]]`. The sampling+coverage+plan-build now runs inside a `build_fn` closure passed to the cache helper.

- [ ] **Step 1: Write the failing "cache hit skips sampling" test**

```python
# tests/unit/test_tccig_topology_training.py (append)
import tccig.train as tccig_train


def test_build_train_topology_bundle_uses_plan_cache(tmp_path, monkeypatch) -> None:
    graph = _coverage_graph()

    monkeypatch.setattr(tccig_train, "load_split_node_ids", lambda **_: set(graph.nodes()))
    monkeypatch.setattr(tccig_train, "build_pair_supervision_graph", lambda **_: graph)

    sample_calls = {"n": 0}
    real_sample = tccig_train.sample_topology_evaluation_subgraphs

    def counting_sample(**kwargs):
        sample_calls["n"] += 1
        return real_sample(**kwargs)

    monkeypatch.setattr(tccig_train, "sample_topology_evaluation_subgraphs", counting_sample)

    captured: dict[str, object] = {}

    def fake_score_split(*, split, pairs, scorer_cfg, runtime, cache_dir):
        captured["pairs"] = pairs
        return [0.5] * len(pairs)

    monkeypatch.setattr(tccig_train, "_score_split", fake_score_split)

    config = {
        "refiner": {
            "topology_training": {
                "enabled": True,
                "node_sizes": [4],
                "samples_per_size": 1,
                "strategy": "bfs",
                "seed": 0,
                "coverage_augmentation": True,
            }
        }
    }
    runtime = _fake_runtime(is_main_process=True)
    common = dict(
        config=config,
        processed_dir=tmp_path,
        scorer_cfg={},
        runtime=runtime,
        cache_dir=tmp_path,
        pairwise_input_rule=tccig_train._resolve_refined_output_rule({}),
    )

    bundle_first, plan_first, stats_first = tccig_train._build_train_topology_bundle(**common)
    bundle_second, plan_second, stats_second = tccig_train._build_train_topology_bundle(**common)

    assert sample_calls["n"] == 1  # second run served from cache
    assert plan_first.total_pairs == plan_second.total_pairs
    assert stats_first == stats_second
    assert set(stats_first) == {"base_bucket_count", "coverage_bucket_count", "positive_edge_coverage"}
```

> If `_resolve_refined_output_rule({})` needs config keys, replace it with the simplest rule constructor the file already exposes; check the actual signature when implementing and adjust the test's `pairwise_input_rule` accordingly.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k build_train_topology_bundle_uses_plan_cache -v`
Expected: FAIL — second call re-samples (`sample_calls["n"] == 2`) because caching is not wired in yet.

- [ ] **Step 3: Refactor `_build_train_topology_bundle` to use the cache helper**

In `tccig/train.py`, replace the sampling→coverage→`build_internal_validation_plan` block (current lines ~482–507) with a `build_fn` closure handed to the helper. The block that computes `node_ids`, `train_graph`, `node_sizes`, `seed`, `strategy` stays. Replace from the `sampled = ...` assignment through the `plan = build_internal_validation_plan(...)` line with:

```python
    samples_per_size = _positive_int(
        topo_cfg.get("samples_per_size", 20),
        "refiner.topology_training.samples_per_size",
    )
    coverage_augmentation = bool(topo_cfg.get("coverage_augmentation", True))

    def _build() -> tuple[InternalValidationPlan, dict[str, float | int]]:
        sampled = sample_topology_evaluation_subgraphs(
            graph=train_graph,
            seed=seed,
            strategy=strategy,
            node_sizes=node_sizes,
            samples_per_size=samples_per_size,
        )
        stats: dict[str, float | int] = {}
        if coverage_augmentation:
            sampled, stats = augment_plan_for_positive_edge_coverage(
                graph=train_graph,
                base_sampled={int(k): list(v) for k, v in sampled.items()},
                node_sizes=node_sizes,
                strategy=strategy,
                seed=seed,
            )
        built_plan = build_internal_validation_plan(
            graph=train_graph, sampled_subgraphs=sampled
        )
        return built_plan, stats

    plan, coverage_stats = _load_or_build_topology_plan(
        split="train_topology",
        graph=train_graph,
        node_sizes=node_sizes,
        samples_per_size=samples_per_size,
        seed=seed,
        strategy=strategy,
        coverage_augmentation=coverage_augmentation,
        runtime=runtime,
        cache_dir=cache_dir,
        build_fn=_build,
    )
    if coverage_stats:
        LOGGER.info(
            "tccig train topology coverage: base=%s coverage=%s positive_edge_coverage=%.4f",
            coverage_stats.get("base_bucket_count"),
            coverage_stats.get("coverage_bucket_count"),
            float(coverage_stats.get("positive_edge_coverage", 0.0)),
        )
```

Keep the downstream `pairs = [...]`, `_score_split(...)`, `edges_from_rule(...)`, and the `return (SplitBundle(...), plan, coverage_stats)` block. Ensure the final return uses `coverage_stats` (the helper's returned stats), preserving the contract.

- [ ] **Step 4: Run the wiring test to verify it passes**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py -k build_train_topology_bundle_uses_plan_cache -v`
Expected: PASS — `sample_calls["n"] == 1`.

- [ ] **Step 5: Run the full topology unit + integration suite**

Run: `uv run python -m pytest tests/unit/test_tccig_topology_training.py tests/integration/test_tccig_topology_training_stage.py -v`
Expected: PASS. If the integration test exercises a real run, confirm a `plans/train_topology.json` is produced and reused.

- [ ] **Step 6: Lint + typecheck**

Run: `uv run ruff check tccig/train.py && uv run mypy src`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add tccig/train.py tests/unit/test_tccig_topology_training.py
git commit -m "feat: cache topology train plan across tccig runs"
```

---

## Task 7: Full verification sweep

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run python -m pytest`
Expected: PASS. Investigate and fix any regression before proceeding (do not mark complete on failures).

- [ ] **Step 2: Lint + format + typecheck the whole change**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src`
Expected: no errors. If `ruff format --check` reports diffs in touched files, run `uv run ruff format .` and re-commit.

- [ ] **Step 3: Sanity-check the cache artifacts layout**

Confirm the cache writes to `cache_dir/plans/{split}.json` and `cache_dir/manifests/{split}_plan.json`, parallel to the existing `cache_dir/scores/{split}.pt` + `cache_dir/manifests/{split}.json`. (Inspect a file produced by the integration test, or add a temporary assert in a scratch run.)

- [ ] **Step 4: Final commit if formatting changed**

```bash
git add -A
git commit -m "chore: format and lint fixups for topology plan cache"
```

---

## Notes for the implementer

- **Validation bundle (optional, low priority):** `_build_validation_topology_bundle` (~line 537) has no coverage augmentation and its scores are already cached, so the win is smaller. If wiring it in is cheap, reuse `_load_or_build_topology_plan` with `split="validation_topology"`, `coverage_augmentation=False`, and a `build_fn` that returns `(build_internal_validation_plan(...), {})`. Do NOT expand the change surface for it otherwise — it is explicitly low-priority in the spec.
- **Out of scope (do not do):** refactoring `s2gae.py::_topology_plan_loss` to consume cached labels; caching the `src/pipeline/stages/topology_finetune.py` path; storing `pair_records` in the cache.
- **Shared-filesystem assumption:** all ranks must see the same `cache_dir`. This matches the existing score cache; the post-barrier reload raises `RuntimeError` rather than diverging silently if the write is not visible.
