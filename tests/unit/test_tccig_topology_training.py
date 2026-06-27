"""Tests for TCCIG Run 02 topology-conditioned training loss."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
import torch
from tccig.s2gae import asymmetric_residual_anchor
from tccig.train import augment_plan_for_positive_edge_coverage


def test_asymmetric_anchor_leaves_deletion_free() -> None:
    negative_delta = torch.tensor([-3.0, -1.0, -0.5])
    assert float(asymmetric_residual_anchor(negative_delta)) == 0.0


def test_asymmetric_anchor_penalizes_upward_push() -> None:
    positive_delta = torch.tensor([2.0, 0.0, -4.0])
    # only +2.0 contributes: (2^2 + 0 + 0) / 3
    assert float(asymmetric_residual_anchor(positive_delta)) == pytest.approx(4.0 / 3.0)


def _base_refiner_config() -> dict:
    return {
        "encoder": "graphconv",
        "input_dim": 8,
        "embedding_cache_dir": "data/embeddings/esm3_1024",
        "monitor_metric": "val_topology_loss",
        "topology_validation": {
            "enabled": True,
            "losses": {"alpha": 1.0, "beta": 8.0, "gamma": 0.5, "delta": 0.1},
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
        "scheduler": {"type": "none"},
        "optimization": {"gradient_clip_norm": 1.0},
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
    from tccig.s2gae import _parse_config

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
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    del config["topology_training"]
    del config["residual_anchor"]
    cfg = _parse_config(config)
    assert cfg.topology_training.enabled is False
    assert cfg.residual_anchor.form == "symmetric"


def test_coverage_augmentation_covers_isolated_positive_edge() -> None:
    import networkx as nx
    from tccig.train import augment_plan_for_positive_edge_coverage

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


def test_coverage_augmentation_handles_positive_self_pair_in_pair_file(tmp_path: Path) -> None:
    import networkx as nx
    from src.topology.finetune_data import build_pair_supervision_graph
    from tccig.train import augment_plan_for_positive_edge_coverage

    # A positive self-pair (HPC job 928974 root cause) must be filtered by the
    # graph builder so coverage augmentation never sees a self-loop frozenset.
    pair_path = tmp_path / "human_train_ppi.txt"
    pair_path.write_text(
        "P1\tP2\t1\nP3\tP3\t1\nP3\tP4\t1\n",
        encoding="utf-8",
    )

    graph = build_pair_supervision_graph(
        pair_path=pair_path,
        node_ids={"P1", "P2", "P3", "P4"},
    )
    assert nx.number_of_selfloops(graph) == 0

    augmented, stats = augment_plan_for_positive_edge_coverage(
        graph=graph,
        base_sampled={2: [("P1", "P2")]},
        node_sizes=[2],
        strategy="BFS",
        seed=0,
    )

    assert stats["positive_edge_coverage"] == 1.0
    assert isinstance(augmented, dict)


def test_train_refiner_request_accepts_train_topology_fields() -> None:
    from tccig.s2gae import TrainRefinerRequest

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


@pytest.fixture
def make_tiny_refiner_and_plan():
    import networkx as nx
    from src.topology.finetune_data import build_internal_validation_plan
    from tccig.s2gae import S2GAERefiner, _SplitGraph

    def _build(*, overdense: bool) -> tuple[object, object, object, dict[str, int]]:
        torch.manual_seed(0)
        node_ids = ["P0", "P1", "P2", "P3"]
        node_index = {protein: idx for idx, protein in enumerate(node_ids)}
        num_nodes = len(node_ids)

        refiner = S2GAERefiner(
            encoder="graphconv",
            input_dim=8,
            hidden_dim=4,
            num_layers=2,
            decoder_hidden_dim=8,
            decoder_layers=2,
            dropout=0.0,
            encoder_aggr="mean",
            layer_norm=True,
            residual_scale=4.0,
        )
        # Force a non-zero residual decoder so deletion is trainable from the start.
        with torch.no_grad():
            output_layer = refiner.decoder.layers[-1]
            output_layer.weight.normal_(0.0, 0.1)
            output_layer.bias.fill_(0.5)

        node_features = torch.randn(num_nodes, 8)
        # All undirected pairs as candidate pairs.
        pair_columns = [
            (node_index[a], node_index[b])
            for ia, a in enumerate(node_ids)
            for b in node_ids[ia + 1 :]
        ]
        pair_index = torch.tensor(pair_columns, dtype=torch.long).t().contiguous()
        prob_value = 0.9 if overdense else 0.5
        pairwise_probabilities = torch.full((pair_index.size(1),), prob_value)
        # Encoder edges mirror the over-dense candidate set.
        edge_index = torch.cat([pair_index, pair_index.flip(0)], dim=1)
        edge_weight = torch.ones(edge_index.size(1))

        graph = _SplitGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            pair_index=pair_index,
            pairwise_probabilities=pairwise_probabilities,
        )

        # True graph for this bucket has a single edge P0-P1.
        true_graph = nx.Graph()
        true_graph.add_nodes_from(node_ids)
        true_graph.add_edge("P0", "P1")
        plan = build_internal_validation_plan(
            graph=true_graph,
            sampled_subgraphs={4: [tuple(node_ids)]},
        )
        return refiner, graph, plan, node_index

    return _build


def test_topology_plan_loss_backprops_and_pressures_density_down(
    make_tiny_refiner_and_plan: object,
) -> None:
    from src.topology.finetune_losses import TopologyLossWeights
    from tccig.s2gae import topology_plan_loss

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
    grads = [p.grad for p in refiner.parameters() if p.grad is not None]
    assert grads, "topology loss did not propagate to refiner parameters"


def test_topology_loss_scale_zero_during_warmup_then_ramps() -> None:
    from src.topology.finetune_losses import TopologyLossWeightSchedule, topology_loss_scale

    schedule = TopologyLossWeightSchedule(warmup_epochs=5, ramp_epochs=10, schedule="linear")
    assert topology_loss_scale(epoch=0, schedule=schedule) == 0.0
    assert topology_loss_scale(epoch=4, schedule=schedule) == 0.0
    assert 0.0 < topology_loss_scale(epoch=9, schedule=schedule) < 1.0
    assert topology_loss_scale(epoch=15, schedule=schedule) == 1.0


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
    base = {4: [("a", "b", "c", "d")]}
    first, first_stats = augment_plan_for_positive_edge_coverage(
        graph=graph, base_sampled=base, node_sizes=[4], strategy="bfs", seed=0
    )
    second, second_stats = augment_plan_for_positive_edge_coverage(
        graph=graph, base_sampled=base, node_sizes=[4], strategy="bfs", seed=0
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
