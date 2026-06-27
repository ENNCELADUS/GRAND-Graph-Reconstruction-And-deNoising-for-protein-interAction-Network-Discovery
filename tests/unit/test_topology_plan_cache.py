"""Tests for the TCCIG topology plan cache."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
from src.topology.finetune_data import build_internal_validation_plan
from src.topology.plan_cache import (
    load_plan_cache,
    payload_to_plan,
    plan_payload_metadata,
    plan_to_payload,
    write_plan_cache,
)


def _toy_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("a", "c"), ("d", "e"), ("e", "a")])
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


def test_metadata_records_graph_size() -> None:
    graph = _toy_graph()
    metadata = _meta(graph)
    assert metadata["node_count"] == graph.number_of_nodes()
    assert metadata["edge_count"] == graph.number_of_edges()


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
    plan = build_internal_validation_plan(graph=graph, sampled_subgraphs=_toy_sampled())
    payload = plan_to_payload(plan)
    write_plan_cache(
        cache_dir=tmp_path, split="train_topology", metadata=_meta(graph), payload=payload
    )

    stale = load_plan_cache(
        cache_dir=tmp_path, split="train_topology", metadata=_meta(graph, seed=999)
    )
    assert stale is None


def test_load_returns_none_when_absent(tmp_path: Path) -> None:
    loaded = load_plan_cache(
        cache_dir=tmp_path, split="train_topology", metadata=_meta(_toy_graph())
    )
    assert loaded is None


def test_load_returns_none_on_corrupt_json(tmp_path: Path) -> None:
    path = tmp_path / "plans" / "train_topology.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    loaded = load_plan_cache(
        cache_dir=tmp_path, split="train_topology", metadata=_meta(_toy_graph())
    )
    assert loaded is None


def test_load_returns_none_on_structurally_invalid_payload(tmp_path: Path) -> None:
    graph = _toy_graph()
    metadata = _meta(graph)
    path = tmp_path / "plans" / "train_topology.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Metadata matches, but the payload is missing the required "buckets" list,
    # which would crash payload_to_plan if treated as a hit.
    path.write_text(
        json.dumps({"metadata": dict(metadata), "payload": {"version": 1}}),
        encoding="utf-8",
    )
    loaded = load_plan_cache(cache_dir=tmp_path, split="train_topology", metadata=metadata)
    assert loaded is None


def test_load_returns_none_on_metadata_match_but_malformed_payload(tmp_path: Path) -> None:
    graph = _toy_graph()
    metadata = _meta(graph)
    path = tmp_path / "plans" / "train_topology.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    # A payload that matches metadata but is missing required bucket fields.
    document = {"metadata": dict(metadata), "payload": {"buckets": [{"node_size": 3}]}}
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_plan_cache(cache_dir=tmp_path, split="train_topology", metadata=metadata)
    assert loaded is None
