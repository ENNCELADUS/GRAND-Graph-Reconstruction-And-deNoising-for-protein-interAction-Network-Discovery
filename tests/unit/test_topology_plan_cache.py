"""Tests for the TCCIG topology plan cache."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest
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


def test_metadata_hash_changes_on_isolated_node() -> None:
    base = _meta(_toy_graph())
    with_isolated = _toy_graph()
    with_isolated.add_node("z")  # new node, no new edges
    assert _meta(with_isolated)["graph_hash"] != base["graph_hash"]
    assert _meta(with_isolated)["node_count"] != base["node_count"]


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


def _valid_payload() -> dict[str, object]:
    graph = _toy_graph()
    plan = build_internal_validation_plan(graph=graph, sampled_subgraphs=_toy_sampled())
    return plan_to_payload(plan)


def _mutate(mutator: object) -> dict[str, object]:
    payload = _valid_payload()
    mutator(payload)  # type: ignore[operator]
    return payload


def _set_wrong_version(payload: dict[str, object]) -> None:
    payload["version"] = 999


def _set_buckets_not_list(payload: dict[str, object]) -> None:
    payload["buckets"] = {"node_size": 3}


def _set_bucket_not_mapping(payload: dict[str, object]) -> None:
    payload["buckets"] = ["not a mapping"]


def _set_length_mismatch(payload: dict[str, object]) -> None:
    bucket = payload["buckets"][0]  # type: ignore[index]
    bucket["target_edges"] = bucket["target_edges"][:-1]


def _set_non_string_node(payload: dict[str, object]) -> None:
    payload["buckets"][0]["sampled_subgraphs"][0][0] = 42  # type: ignore[index]


def _set_edge_wrong_arity(payload: dict[str, object]) -> None:
    payload["buckets"][0]["target_edges"][0] = [["a"]]  # type: ignore[index]


def _set_edge_empty(payload: dict[str, object]) -> None:
    payload["buckets"][0]["target_edges"][0] = [[]]  # type: ignore[index]


def _set_edge_endpoint_outside_nodes(payload: dict[str, object]) -> None:
    payload["buckets"][0]["target_edges"][0] = [["zzz", "yyy"]]  # type: ignore[index]


def _set_edge_non_canonical(payload: dict[str, object]) -> None:
    nodes = payload["buckets"][0]["sampled_subgraphs"][0]  # type: ignore[index]
    high, low = max(nodes), min(nodes)
    payload["buckets"][0]["target_edges"][0] = [[high, low]]  # type: ignore[index]


def _set_total_pairs_wrong(payload: dict[str, object]) -> None:
    payload["total_pairs"] = 99999


@pytest.mark.parametrize(
    "mutator",
    [
        _set_wrong_version,
        _set_buckets_not_list,
        _set_bucket_not_mapping,
        _set_length_mismatch,
        _set_non_string_node,
        _set_edge_wrong_arity,
        _set_edge_empty,
        _set_edge_endpoint_outside_nodes,
        _set_edge_non_canonical,
        _set_total_pairs_wrong,
    ],
)
def test_load_rejects_malformed_payload(tmp_path: Path, mutator: object) -> None:
    metadata = _meta(_toy_graph())
    path = tmp_path / "plans" / "train_topology.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"metadata": dict(metadata), "payload": _mutate(mutator)}
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_plan_cache(cache_dir=tmp_path, split="train_topology", metadata=metadata)
    assert loaded is None


def test_load_accepts_valid_payload(tmp_path: Path) -> None:
    metadata = _meta(_toy_graph())
    write_plan_cache(
        cache_dir=tmp_path, split="train_topology", metadata=metadata, payload=_valid_payload()
    )
    loaded = load_plan_cache(cache_dir=tmp_path, split="train_topology", metadata=metadata)
    assert loaded is not None


def test_subset_plan_metadata_changes_with_sampler_parameters() -> None:
    import networkx as nx

    from src.topology.plan_cache import subset_plan_payload_metadata

    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c")])
    first = subset_plan_payload_metadata(
        split="train_topology",
        graph=graph,
        node_sizes=[20, 40],
        samples_per_size=2,
        seed=0,
        strategy="mixed",
        coverage_augmentation=True,
        candidate_ratio=20,
        pool_ratio=10,
        epoch_ratio=5,
        hard_fraction=0.5,
        uniform_fraction=0.5,
        hard_stratum_fraction=0.2,
        max_subgraphs_per_size=0,
        max_labeled_pairs_per_size=0,
        pair_scope="subset",
        scorer_config={},
    )
    second = subset_plan_payload_metadata(
        split="train_topology",
        graph=graph,
        node_sizes=[20, 40],
        samples_per_size=2,
        seed=0,
        strategy="mixed",
        coverage_augmentation=True,
        candidate_ratio=10,
        pool_ratio=10,
        epoch_ratio=5,
        hard_fraction=0.5,
        uniform_fraction=0.5,
        hard_stratum_fraction=0.2,
        max_subgraphs_per_size=0,
        max_labeled_pairs_per_size=0,
        pair_scope="subset",
        scorer_config={},
    )
    assert first["candidate_ratio"] == 20
    assert first != second


def test_subset_plan_metadata_embeds_scorer_identity() -> None:
    # Review finding: without scorer/checkpoint hashes the cached scored pairs can be
    # silently reused after the frozen scorer changes. The cache key MUST carry the
    # same scorer-identity block that score_cache_metadata uses.
    import networkx as nx

    from src.topology.plan_cache import subset_plan_payload_metadata

    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c")])
    kwargs = dict(
        split="train_topology",
        graph=graph,
        node_sizes=[20],
        samples_per_size=1,
        seed=0,
        strategy="mixed",
        coverage_augmentation=False,
        candidate_ratio=20,
        pool_ratio=10,
        epoch_ratio=5,
        hard_fraction=0.5,
        uniform_fraction=0.5,
        hard_stratum_fraction=0.2,
        max_subgraphs_per_size=0,
        max_labeled_pairs_per_size=0,
        pair_scope="subset",
    )
    meta = subset_plan_payload_metadata(scorer_config={"max_sequence_length": 1000}, **kwargs)
    assert "scorer" in meta
    assert meta["scorer"]["max_sequence_length"] == 1000
    other = subset_plan_payload_metadata(scorer_config={"max_sequence_length": 2000}, **kwargs)
    assert meta != other


def test_subset_cache_round_trips_and_rejects_full_plan_payload(tmp_path) -> None:
    import networkx as nx

    from src.topology.plan_cache import (
        load_plan_cache,
        load_subset_plan_cache,
        write_subset_plan_cache,
    )
    from tccig.topology_subset import (
        TopologySubsetSamplerConfig,
        build_topology_subset_plan,
        subset_plan_to_payload,
    )

    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d"])
    graph.add_edges_from([("a", "b"), ("b", "c")])
    cfg = TopologySubsetSamplerConfig(candidate_ratio=2, pool_ratio=1, epoch_ratio=1, seed=1)
    plan = build_topology_subset_plan(
        graph=graph, sampled_subgraphs={4: [("a", "b", "c", "d")]}, config=cfg,
        scorer_probabilities={},
    )
    metadata = {"pair_scope": "subset", "candidate_ratio": 2}
    write_subset_plan_cache(
        cache_dir=tmp_path, split="train_topology_subset", metadata=metadata,
        payload=subset_plan_to_payload(plan),
    )
    # Subset loader hits; the full-plan loader rejects the subset payload shape.
    assert load_subset_plan_cache(
        cache_dir=tmp_path, split="train_topology_subset", metadata=metadata
    ) is not None
    assert load_plan_cache(
        cache_dir=tmp_path, split="train_topology_subset", metadata=metadata
    ) is None
