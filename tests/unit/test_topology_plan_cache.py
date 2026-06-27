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
