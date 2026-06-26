"""Tests for TCCIG Run 02 topology-conditioned training loss."""

from __future__ import annotations

import pytest
import torch
from tccig.s2gae import asymmetric_residual_anchor


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
