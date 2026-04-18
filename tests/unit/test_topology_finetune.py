"""Unit tests for graph-topology fine-tuning helpers."""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import networkx as nx
import pytest
import src.topology.finetune_losses as finetune_losses_module
import torch
import torch.nn.functional as functional
from src.topology.finetune_data import (
    EdgeCoverEpochPlan,
    EmbeddingRepository,
    ExplicitNegativePairLookup,
    build_explicit_negative_lookup,
    build_internal_validation_plan,
    build_pair_supervision_graph,
    filter_graph_to_embedding_index,
    iter_supervised_pair_chunks,
    iter_subgraph_pair_chunks,
    load_split_node_ids,
    sample_edge_cover_subgraphs,
    sample_training_subgraphs,
)
from src.topology.finetune_losses import (
    TopologyLossWeights,
    build_symmetric_adjacency,
    compute_topology_losses,
    soft_graph_similarity_loss,
    soft_relative_density_loss,
)


def _write_embedding_cache(cache_dir: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for protein_id, value in {
        "P1": 1.0,
        "P2": 2.0,
        "P3": 3.0,
        "P4": 4.0,
    }.items():
        relative_path = f"embeddings/{protein_id}.pt"
        output_path = cache_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.full((2, 4), value, dtype=torch.float32), output_path)
        index[protein_id] = relative_path
    return index


def test_sample_training_subgraphs_supports_all_strategies() -> None:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("P1", "P2"),
            ("P2", "P3"),
            ("P3", "P4"),
            ("P4", "P5"),
            ("P5", "P6"),
        ]
    )

    for strategy in ["BFS", "DFS", "RANDOM_WALK", "mixed"]:
        sampled = sample_training_subgraphs(
            graph=graph,
            num_subgraphs=8,
            min_nodes=3,
            max_nodes=5,
            strategy=strategy,
            seed=13,
        )

        assert len(sampled) == 8
        assert all(3 <= len(node_ids) <= 5 for node_ids in sampled)
        assert all(set(node_ids).issubset(set(graph.nodes)) for node_ids in sampled)


def _mean_positive_edge_reuse(
    *,
    graph: nx.Graph,
    subgraphs: tuple[tuple[str, ...], ...],
) -> float:
    edge_counts: dict[tuple[str, str], int] = {
        tuple(sorted((node_a, node_b))): 0 for node_a, node_b in graph.edges()
    }
    for nodes in subgraphs:
        node_list = tuple(nodes)
        for index, node_a in enumerate(node_list):
            for node_b in node_list[index + 1 :]:
                edge = tuple(sorted((node_a, node_b)))
                if edge in edge_counts:
                    edge_counts[edge] += 1
    covered = [count for count in edge_counts.values() if count > 0]
    return sum(covered) / float(len(covered)) if covered else 0.0


def test_sample_edge_cover_subgraphs_covers_all_positive_edges() -> None:
    graph = nx.path_graph(["P1", "P2", "P3", "P4", "P5", "P6"])

    plan = sample_edge_cover_subgraphs(
        graph=graph,
        num_subgraphs=3,
        min_nodes=2,
        max_nodes=2,
        strategy="BFS",
        seed=13,
    )

    assert isinstance(plan, EdgeCoverEpochPlan)
    assert plan.total_positive_edges == 5
    assert plan.covered_positive_edges == 5
    assert plan.positive_edge_coverage_ratio == pytest.approx(1.0)
    assert len(plan.subgraphs) >= 5
    assert all(len(node_ids) == 2 for node_ids in plan.subgraphs)
    assert all(len(set(node_ids)) == len(node_ids) for node_ids in plan.subgraphs)


def test_sample_edge_cover_subgraphs_uses_bounded_shuffle_chunk_plan() -> None:
    graph = nx.path_graph(["P1", "P2", "P3", "P4", "P5", "P6", "P7"])

    plan = sample_edge_cover_subgraphs(
        graph=graph,
        num_subgraphs=0,
        min_nodes=4,
        max_nodes=4,
        strategy="BFS",
        seed=13,
        edge_chunk_size=2,
    )

    assert isinstance(plan, EdgeCoverEpochPlan)
    assert plan.total_positive_edges == 6
    assert plan.covered_positive_edges == 6
    assert plan.positive_edge_coverage_ratio == pytest.approx(1.0)
    assert len(plan.subgraphs) == 3
    assert all(2 <= len(node_ids) <= 4 for node_ids in plan.subgraphs)
    assert all(len(set(node_ids)) == len(node_ids) for node_ids in plan.subgraphs)


def test_sample_edge_cover_subgraphs_splits_sparse_chunks_by_max_nodes() -> None:
    graph = nx.Graph()
    graph.add_edges_from((f"A{i}", f"B{i}") for i in range(100))

    plan = sample_edge_cover_subgraphs(
        graph=graph,
        num_subgraphs=0,
        min_nodes=30,
        max_nodes=60,
        strategy="BFS",
        seed=13,
    )

    assert plan.covered_positive_edges == plan.total_positive_edges == 100
    assert plan.positive_edge_coverage_ratio == pytest.approx(1.0)
    assert all(len(node_ids) <= 60 for node_ids in plan.subgraphs)
    assert all(len(assigned_edges) <= 885 for assigned_edges in plan.assigned_positive_edges)


def test_sample_edge_cover_subgraphs_assigns_each_positive_edge_exactly_once() -> None:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("P1", "P2"),
            ("P2", "P3"),
            ("P1", "P3"),
            ("P3", "P4"),
        ]
    )

    plan = sample_edge_cover_subgraphs(
        graph=graph,
        num_subgraphs=0,
        min_nodes=3,
        max_nodes=3,
        strategy="BFS",
        seed=13,
        edge_chunk_size=1,
    )

    assigned_edges = [
        edge for assigned_chunk in plan.assigned_positive_edges for edge in assigned_chunk
    ]
    assert sorted(assigned_edges) == sorted(tuple(sorted(edge)) for edge in graph.edges())
    assert len(assigned_edges) == len(set(assigned_edges))


def test_sample_edge_cover_subgraphs_preassigns_each_negative_edge_at_most_once() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["P1", "P2", "P3", "P4"])
    graph.add_edges_from([("P1", "P2"), ("P3", "P4")])
    negative_lookup = ExplicitNegativePairLookup(
        negative_pairs=frozenset(
            {
                ("P1", "P3"),
                ("P1", "P4"),
                ("P2", "P3"),
                ("P2", "P4"),
            }
        ),
        partners_by_node={},
    )

    plan = sample_edge_cover_subgraphs(
        graph=graph,
        num_subgraphs=0,
        min_nodes=4,
        max_nodes=4,
        strategy="BFS",
        seed=13,
        edge_chunk_size=1,
        negative_lookup=negative_lookup,
        negative_ratio=2,
    )

    assigned_edges = [
        edge for assigned_chunk in plan.assigned_negative_edges for edge in assigned_chunk
    ]
    assert sorted(assigned_edges) == sorted(negative_lookup.negative_pairs)
    assert len(assigned_edges) == len(set(assigned_edges))


def test_sample_edge_cover_subgraphs_assigns_global_negatives_outside_subgraph_nodes() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["P1", "P2", "P3", "P4"])
    graph.add_edge("P1", "P2")
    negative_lookup = ExplicitNegativePairLookup(
        negative_pairs=frozenset({("P3", "P4")}),
        partners_by_node={},
    )

    plan = sample_edge_cover_subgraphs(
        graph=graph,
        num_subgraphs=0,
        min_nodes=2,
        max_nodes=2,
        strategy="BFS",
        seed=13,
        edge_chunk_size=1,
        negative_lookup=negative_lookup,
        negative_ratio=1,
    )

    assert plan.subgraphs == (("P1", "P2"),)
    assert plan.assigned_negative_edges == (frozenset({("P3", "P4")}),)


def test_sample_edge_cover_subgraphs_respects_epoch_floor_with_positive_edges() -> None:
    graph = nx.path_graph([f"P{i}" for i in range(1, 12)])

    plan = sample_edge_cover_subgraphs(
        graph=graph,
        num_subgraphs=5,
        min_nodes=4,
        max_nodes=4,
        strategy="DFS",
        seed=17,
    )

    assert plan.covered_positive_edges == plan.total_positive_edges == 10
    assert plan.positive_edge_coverage_ratio == pytest.approx(1.0)
    assert len(plan.subgraphs) >= 5


def test_sample_edge_cover_subgraphs_supports_zero_floor_and_still_covers_edges() -> None:
    graph = nx.path_graph(["P1", "P2", "P3", "P4"])

    plan = sample_edge_cover_subgraphs(
        graph=graph,
        num_subgraphs=0,
        min_nodes=2,
        max_nodes=2,
        strategy="BFS",
        seed=31,
    )

    assert plan.covered_positive_edges == plan.total_positive_edges == 3
    assert plan.positive_edge_coverage_ratio == pytest.approx(1.0)
    assert len(plan.subgraphs) == 3


def test_sample_edge_cover_subgraphs_supports_all_strategies() -> None:
    graph = nx.cycle_graph(["P1", "P2", "P3", "P4", "P5"])

    for strategy in ["BFS", "DFS", "RANDOM_WALK", "mixed"]:
        plan = sample_edge_cover_subgraphs(
            graph=graph,
            num_subgraphs=4,
            min_nodes=2,
            max_nodes=4,
            strategy=strategy,
            seed=19,
        )

        assert len(plan.subgraphs) >= 4
        assert plan.positive_edge_coverage_ratio == pytest.approx(1.0)
        assert all(2 <= len(node_ids) <= graph.number_of_nodes() for node_ids in plan.subgraphs)
        assert all(set(node_ids).issubset(set(graph.nodes)) for node_ids in plan.subgraphs)
        assert all(len(set(node_ids)) == len(node_ids) for node_ids in plan.subgraphs)


def test_sample_edge_cover_subgraphs_handles_disconnected_graph() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("P1", "P2"), ("P3", "P4")])
    graph.add_node("P5")

    plan = sample_edge_cover_subgraphs(
        graph=graph,
        num_subgraphs=1,
        min_nodes=2,
        max_nodes=3,
        strategy="BFS",
        seed=23,
    )

    assert plan.total_positive_edges == 2
    assert plan.covered_positive_edges == 2
    assert plan.positive_edge_coverage_ratio == pytest.approx(1.0)
    assert len(plan.subgraphs) >= 2


def test_sample_edge_cover_subgraphs_reduces_overlap_vs_random_baseline() -> None:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("P0", "P1"),
            ("P0", "P2"),
            ("P0", "P3"),
            ("P0", "P4"),
            ("P0", "P5"),
            ("P0", "P6"),
        ]
    )

    edge_cover_plan = sample_edge_cover_subgraphs(
        graph=graph,
        num_subgraphs=6,
        min_nodes=2,
        max_nodes=2,
        strategy="BFS",
        seed=29,
    )
    random_subgraphs = sample_training_subgraphs(
        graph=graph,
        num_subgraphs=6,
        min_nodes=2,
        max_nodes=2,
        strategy="BFS",
        seed=29,
    )

    assert edge_cover_plan.mean_positive_edge_reuse == pytest.approx(
        _mean_positive_edge_reuse(graph=graph, subgraphs=edge_cover_plan.subgraphs)
    )
    assert edge_cover_plan.mean_positive_edge_reuse < _mean_positive_edge_reuse(
        graph=graph,
        subgraphs=tuple(random_subgraphs),
    )


def test_filter_graph_to_embedding_index_drops_unavailable_nodes() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("P1", "P2"), ("P2", "P3"), ("P3", "P4")])

    filtered = filter_graph_to_embedding_index(
        graph=graph,
        embedding_index={"P1": "p1.pt", "P2": "p2.pt"},
    )

    assert set(filtered.nodes) == {"P1", "P2"}
    assert set(filtered.edges) == {("P1", "P2")}


def test_build_pair_supervision_graph_uses_only_positive_pairs_and_keeps_isolated_nodes(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "human_BFS_split.pkl"
    with split_path.open("wb") as handle:
        pickle.dump({"train": {"P1", "P2", "P3", "P4"}, "test": {"PX"}}, handle)
    pair_path = tmp_path / "human_train_ppi.txt"
    pair_path.write_text(
        "P1\tP2\t1\nP1\tP3\t0\nP2\tP3\t1\nPX\tP2\t1\nP4\tPY\t1\n",
        encoding="utf-8",
    )

    train_nodes = load_split_node_ids(split_path=split_path, split_name="train")
    graph = build_pair_supervision_graph(pair_path=pair_path, node_ids=train_nodes)

    assert set(graph.nodes) == {"P1", "P2", "P3", "P4"}
    assert {tuple(sorted(edge)) for edge in graph.edges} == {("P1", "P2"), ("P2", "P3")}


def test_build_explicit_negative_lookup_uses_only_in_split_negative_pairs(tmp_path: Path) -> None:
    pair_path = tmp_path / "human_train_ppi.txt"
    pair_path.write_text(
        "\n".join(
            [
                "P1\tP2\t1",
                "P1\tP3\t0",
                "P2\tP4\t0",
                "P9\tP1\t0",
                "P3\tP4\t1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    lookup = build_explicit_negative_lookup(
        pair_path=pair_path,
        node_ids={"P1", "P2", "P3", "P4"},
    )

    assert isinstance(lookup, ExplicitNegativePairLookup)
    assert lookup.negative_pairs == frozenset({("P1", "P3"), ("P2", "P4")})


def test_iter_subgraph_pair_chunks_materializes_all_labels(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_edges_from([("P1", "P2"), ("P2", "P3")])
    embedding_index = _write_embedding_cache(tmp_path / "cache")

    chunks = list(
        iter_subgraph_pair_chunks(
            graph=graph,
            nodes=("P1", "P2", "P3"),
            cache_dir=tmp_path / "cache",
            embedding_index=embedding_index,
            input_dim=4,
            max_sequence_length=8,
            pair_batch_size=2,
        )
    )

    assert [tuple(chunk.pair_index_a.tolist()) for chunk in chunks] == [(0, 0), (1,)]
    assert [tuple(chunk.pair_index_b.tolist()) for chunk in chunks] == [(1, 2), (2,)]
    assert torch.cat([chunk.label for chunk in chunks], dim=0).tolist() == [1.0, 0.0, 1.0]
    assert chunks[0].emb_a.shape == torch.Size([2, 2, 4])
    assert chunks[0].emb_b.shape == torch.Size([2, 2, 4])


def test_iter_subgraph_pair_chunks_keeps_topology_labels_independent_of_bce_assignment(
    tmp_path: Path,
) -> None:
    graph = nx.Graph()
    graph.add_edges_from([("P1", "P2"), ("P1", "P3"), ("P2", "P3")])
    embedding_index = _write_embedding_cache(tmp_path / "cache")

    chunks = list(
        iter_subgraph_pair_chunks(
            graph=graph,
            nodes=("P1", "P2", "P3"),
            assigned_positive_edges=frozenset({("P1", "P2")}),
            cache_dir=tmp_path / "cache",
            embedding_index=embedding_index,
            input_dim=4,
            max_sequence_length=8,
            pair_batch_size=8,
        )
    )

    assert torch.cat([chunk.label for chunk in chunks], dim=0).tolist() == [1.0, 1.0, 1.0]
    assert torch.cat([chunk.bce_label for chunk in chunks], dim=0).tolist() == [1.0, 0.0, 0.0]
    assert torch.cat([chunk.bce_mask for chunk in chunks], dim=0).tolist() == [1.0, 0.0, 0.0]


def test_iter_supervised_pair_chunks_includes_out_of_subgraph_negative_pairs(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    embedding_index = _write_embedding_cache(cache_dir)

    chunks = list(
        iter_supervised_pair_chunks(
            positive_edges=frozenset({("P1", "P2")}),
            negative_edges=frozenset({("P3", "P4")}),
            cache_dir=cache_dir,
            embedding_index=embedding_index,
            input_dim=4,
            max_sequence_length=8,
            pair_batch_size=8,
        )
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.nodes == ("P1", "P2", "P3", "P4")
    assert chunk.label.tolist() == [1.0, 0.0]
    assert chunk.bce_label is not None
    assert chunk.bce_mask is not None
    assert chunk.bce_label.tolist() == [1.0, 0.0]
    assert chunk.bce_mask.tolist() == [1.0, 1.0]


def test_iter_subgraph_pair_chunks_uses_assigned_negative_edges_for_bce(
    tmp_path: Path,
) -> None:
    graph = nx.Graph()
    graph.add_edge("P1", "P2")
    embedding_index = _write_embedding_cache(tmp_path / "cache")

    chunks = list(
        iter_subgraph_pair_chunks(
            graph=graph,
            nodes=("P1", "P2", "P3", "P4"),
            assigned_positive_edges=frozenset({("P1", "P2")}),
            assigned_negative_edges=frozenset({("P1", "P3")}),
            cache_dir=tmp_path / "cache",
            embedding_index=embedding_index,
            input_dim=4,
            max_sequence_length=8,
            pair_batch_size=8,
        )
    )

    pair_lookup = {
        (int(index_a), int(index_b)): (float(bce_label), float(bce_mask))
        for index_a, index_b, bce_label, bce_mask in zip(
            torch.cat([chunk.pair_index_a for chunk in chunks], dim=0).tolist(),
            torch.cat([chunk.pair_index_b for chunk in chunks], dim=0).tolist(),
            torch.cat([chunk.bce_label for chunk in chunks], dim=0).tolist(),
            torch.cat([chunk.bce_mask for chunk in chunks], dim=0).tolist(),
            strict=True,
        )
    }

    assert pair_lookup[(0, 1)] == pytest.approx((1.0, 1.0))
    assert pair_lookup[(0, 2)] == pytest.approx((0.0, 1.0))
    assert pair_lookup[(1, 2)] == pytest.approx((0.0, 0.0))


def test_embedding_repository_matches_cached_loader_and_evicts_by_byte_budget(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    embedding_index = _write_embedding_cache(cache_dir)
    repository = EmbeddingRepository(
        cache_dir=cache_dir,
        embedding_index=embedding_index,
        input_dim=4,
        max_sequence_length=8,
        max_cache_bytes=64,
    )

    first = repository.get("P1")
    second = repository.get("P2")
    reloaded = repository.get("P1")

    assert torch.allclose(first, torch.full((2, 4), 1.0, dtype=torch.float32))
    assert torch.allclose(second, torch.full((2, 4), 2.0, dtype=torch.float32))
    assert torch.allclose(reloaded, first)
    assert repository.preload(["P3", "P4"]) >= 1


def test_build_internal_validation_plan_flattens_pairs_and_targets() -> None:
    graph = nx.path_graph(["P1", "P2", "P3", "P4"])

    plan = build_internal_validation_plan(
        graph=graph,
        sampled_subgraphs={
            3: [("P1", "P2", "P3"), ("P2", "P3", "P4")],
        },
    )

    assert plan.total_subgraphs == 2
    assert plan.total_pairs == 6
    assert plan.protein_ids == frozenset({"P1", "P2", "P3", "P4"})
    assert len(plan.buckets) == 1
    assert len(plan.buckets[0].pair_records) == 6
    assert sorted(plan.buckets[0].target_subgraphs[0].edges()) == [("P1", "P2"), ("P2", "P3")]


def test_iter_subgraph_pair_chunks_uses_explicit_negative_masks_for_bce(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_edges_from([("P1", "P2"), ("P2", "P3")])
    cache_dir = tmp_path / "cache"
    embedding_index = _write_embedding_cache(cache_dir)
    extra_embeddings = {
        "P5": torch.full((2, 4), 5.0, dtype=torch.float32),
    }
    for protein_id, tensor in extra_embeddings.items():
        relative_path = f"embeddings/{protein_id}.pt"
        output_path = cache_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, output_path)
        embedding_index[protein_id] = relative_path

    pair_path = tmp_path / "human_train_ppi.txt"
    pair_path.write_text(
        "\n".join(
            [
                "P1\tP2\t1",
                "P2\tP3\t1",
                "P1\tP3\t0",
                "P1\tP4\t0",
                "P2\tP4\t0",
                "P3\tP4\t0",
                "P1\tP5\t0",
                "P2\tP5\t0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    negative_lookup = build_explicit_negative_lookup(
        pair_path=pair_path,
        node_ids={"P1", "P2", "P3", "P4", "P5"},
    )

    chunks = list(
        iter_subgraph_pair_chunks(
            graph=graph,
            nodes=("P1", "P2", "P3", "P4", "P5"),
            assigned_positive_edges=frozenset({("P1", "P2"), ("P2", "P3")}),
            assigned_negative_edges=frozenset(
                {("P1", "P3"), ("P1", "P4"), ("P2", "P4"), ("P3", "P4")}
            ),
            cache_dir=cache_dir,
            embedding_index=embedding_index,
            input_dim=4,
            max_sequence_length=8,
            pair_batch_size=3,
        )
    )

    all_topology_labels = torch.cat([chunk.label for chunk in chunks], dim=0)
    all_bce_labels = torch.cat([chunk.bce_label for chunk in chunks], dim=0)
    all_bce_masks = torch.cat([chunk.bce_mask for chunk in chunks], dim=0)
    all_pair_index_a = torch.cat([chunk.pair_index_a for chunk in chunks], dim=0).tolist()
    all_pair_index_b = torch.cat([chunk.pair_index_b for chunk in chunks], dim=0).tolist()
    pair_lookup = {
        (int(index_a), int(index_b)): (float(topology_label), float(bce_label), float(bce_mask))
        for index_a, index_b, topology_label, bce_label, bce_mask in zip(
            all_pair_index_a,
            all_pair_index_b,
            all_topology_labels.tolist(),
            all_bce_labels.tolist(),
            all_bce_masks.tolist(),
            strict=True,
        )
    }

    assert all_topology_labels.sum().item() == pytest.approx(2.0)
    assert all_bce_masks.sum().item() == pytest.approx(6.0)
    assert all_bce_labels.sum().item() == pytest.approx(2.0)
    assert set(negative_lookup.negative_pairs).issuperset(
        {("P1", "P3"), ("P1", "P4"), ("P2", "P4"), ("P3", "P4")}
    )
    assert pair_lookup[(2, 4)] == pytest.approx((0.0, 0.0, 0.0))
    assert pair_lookup[(3, 4)] == pytest.approx((0.0, 0.0, 0.0))


def test_iter_subgraph_pair_chunks_clips_negative_count_when_ratio_is_infeasible(
    tmp_path: Path,
) -> None:
    graph = nx.Graph()
    graph.add_edges_from([("P1", "P2"), ("P2", "P3"), ("P3", "P4")])
    cache_dir = tmp_path / "cache"
    embedding_index = _write_embedding_cache(cache_dir)
    pair_path = tmp_path / "human_train_ppi.txt"
    pair_path.write_text(
        "\n".join(
            [
                "P1\tP2\t1",
                "P2\tP3\t1",
                "P3\tP4\t1",
                "P1\tP3\t0",
                "P1\tP4\t0",
                "P2\tP4\t0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    negative_lookup = build_explicit_negative_lookup(
        pair_path=pair_path,
        node_ids={"P1", "P2", "P3", "P4"},
    )

    chunks = list(
        iter_subgraph_pair_chunks(
            graph=graph,
            nodes=("P1", "P2", "P3", "P4"),
            assigned_positive_edges=frozenset({("P1", "P2"), ("P2", "P3"), ("P3", "P4")}),
            assigned_negative_edges=negative_lookup.negative_pairs,
            cache_dir=cache_dir,
            embedding_index=embedding_index,
            input_dim=4,
            max_sequence_length=8,
            pair_batch_size=8,
        )
    )

    all_topology_labels = torch.cat([chunk.label for chunk in chunks], dim=0)
    all_bce_labels = torch.cat([chunk.bce_label for chunk in chunks], dim=0)
    all_bce_masks = torch.cat([chunk.bce_mask for chunk in chunks], dim=0)

    assert all_topology_labels.sum().item() == pytest.approx(3.0)
    assert all_bce_labels.sum().item() == pytest.approx(3.0)
    assert all_bce_masks.sum().item() == pytest.approx(6.0)


def test_soft_topology_losses_match_hard_metric_limits() -> None:
    target_adjacency = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    pred_adjacency = torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    gs_loss = soft_graph_similarity_loss(
        pred_adjacency=pred_adjacency,
        target_adjacency=target_adjacency,
    )
    rd_loss = soft_relative_density_loss(
        pred_adjacency=pred_adjacency,
        target_adjacency=target_adjacency,
    )

    assert gs_loss.item() == pytest.approx(0.5)
    assert rd_loss.item() == pytest.approx(0.0)


def test_soft_relative_density_loss_log_ratio_huber_stays_bounded_for_large_overdensity() -> None:
    num_nodes = 10
    target_adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    target_adjacency[0, 1] = 1.0
    target_adjacency[1, 0] = 1.0

    pred_adjacency = torch.ones((num_nodes, num_nodes), dtype=torch.float32)
    pred_adjacency.fill_diagonal_(0.0)

    rd_loss = soft_relative_density_loss(
        pred_adjacency=pred_adjacency,
        target_adjacency=target_adjacency,
        loss_form="log_ratio_huber",
    )

    expected = functional.smooth_l1_loss(
        torch.tensor(math.log(45.0), dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float32),
    )
    assert rd_loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert rd_loss.item() < 10.0


def test_compute_topology_losses_returns_weighted_total() -> None:
    pair_probabilities = torch.tensor([0.9, 0.1, 0.8], dtype=torch.float32)
    adjacency = build_symmetric_adjacency(
        num_nodes=3,
        pair_index_a=torch.tensor([0, 0, 1], dtype=torch.long),
        pair_index_b=torch.tensor([1, 2, 2], dtype=torch.long),
        pair_probabilities=pair_probabilities,
    )
    target = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    losses = compute_topology_losses(
        pred_adjacency=adjacency,
        target_adjacency=target,
        weights=TopologyLossWeights(alpha=0.5, beta=1.0, gamma=0.3, delta=0.2),
    )

    expected_total = (
        0.5 * losses["graph_similarity"]
        + losses["relative_density"]
        + 0.3 * losses["degree_mmd"]
        + 0.2 * losses["clustering_mmd"]
    )
    assert losses["total_topology"].item() == pytest.approx(expected_total.item())


def test_compute_topology_losses_pairwise_path_matches_dense_path() -> None:
    pair_index_a = torch.tensor([0, 0, 1], dtype=torch.long)
    pair_index_b = torch.tensor([1, 2, 2], dtype=torch.long)
    pred_pair_probabilities = torch.tensor([0.9, 0.1, 0.8], dtype=torch.float32)
    target_pair_probabilities = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    adjacency = build_symmetric_adjacency(
        num_nodes=3,
        pair_index_a=pair_index_a,
        pair_index_b=pair_index_b,
        pair_probabilities=pred_pair_probabilities,
    )
    target = build_symmetric_adjacency(
        num_nodes=3,
        pair_index_a=pair_index_a,
        pair_index_b=pair_index_b,
        pair_probabilities=target_pair_probabilities,
    )
    weights = TopologyLossWeights(alpha=0.5, beta=1.0, gamma=0.3, delta=0.2)

    dense_losses = compute_topology_losses(
        pred_adjacency=adjacency,
        target_adjacency=target,
        weights=weights,
    )
    pairwise_losses = compute_topology_losses(
        weights=weights,
        num_nodes=3,
        pair_index_a=pair_index_a,
        pair_index_b=pair_index_b,
        pred_pair_probabilities=pred_pair_probabilities,
        target_pair_probabilities=target_pair_probabilities,
    )

    for key in (
        "graph_similarity",
        "relative_density",
        "degree_mmd",
        "clustering_mmd",
        "total_topology",
    ):
        assert pairwise_losses[key].item() == pytest.approx(dense_losses[key].item(), abs=1e-6)


def test_topology_loss_scale_respects_warmup_and_linear_ramp() -> None:
    schedule = finetune_losses_module.TopologyLossWeightSchedule(
        warmup_epochs=2,
        ramp_epochs=3,
        schedule="linear",
    )

    assert finetune_losses_module.topology_loss_scale(epoch=0, schedule=schedule) == pytest.approx(
        0.0
    )
    assert finetune_losses_module.topology_loss_scale(epoch=1, schedule=schedule) == pytest.approx(
        0.0
    )
    assert finetune_losses_module.topology_loss_scale(epoch=2, schedule=schedule) == pytest.approx(
        0.0
    )
    assert finetune_losses_module.topology_loss_scale(epoch=3, schedule=schedule) == pytest.approx(
        1.0 / 3.0
    )
    assert finetune_losses_module.topology_loss_scale(epoch=4, schedule=schedule) == pytest.approx(
        2.0 / 3.0
    )
    assert finetune_losses_module.topology_loss_scale(epoch=5, schedule=schedule) == pytest.approx(
        1.0
    )


def test_topology_loss_scale_supports_cosine_ramp() -> None:
    schedule = finetune_losses_module.TopologyLossWeightSchedule(
        warmup_epochs=1,
        ramp_epochs=4,
        schedule="cosine",
    )

    assert finetune_losses_module.topology_loss_scale(epoch=0, schedule=schedule) == pytest.approx(
        0.0
    )
    assert finetune_losses_module.topology_loss_scale(epoch=1, schedule=schedule) == pytest.approx(
        0.0
    )
    assert finetune_losses_module.topology_loss_scale(epoch=3, schedule=schedule) == pytest.approx(
        0.5,
        abs=1e-6,
    )
    assert finetune_losses_module.topology_loss_scale(epoch=5, schedule=schedule) == pytest.approx(
        1.0
    )
