"""Tests for TCCIG Run 02 topology-conditioned training loss."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
import tccig.train as tccig_train
import torch
from src.topology.finetune_data import build_internal_validation_plan
from tccig.s2gae import asymmetric_residual_anchor
from tccig.train import (
    _coverage_stats_from_payload,
    _load_or_build_topology_plan,
    augment_plan_for_positive_edge_coverage,
)


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


def test_parse_config_reads_topology_only_epoch_boundary() -> None:
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    topology_training = config["topology_training"]
    assert isinstance(topology_training, dict)
    topology_training["topo_only_after_epoch"] = 7

    cfg = _parse_config(config)

    assert cfg.topology_training.topo_only_after_epoch == 7


def test_parse_config_defaults_topology_training_disabled() -> None:
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    del config["topology_training"]
    del config["residual_anchor"]
    cfg = _parse_config(config)
    assert cfg.topology_training.enabled is False
    assert cfg.residual_anchor.form == "symmetric"


def test_parse_config_defaults_topology_only_epoch_boundary_off() -> None:
    from tccig.s2gae import _parse_config

    cfg = _parse_config(_base_refiner_config())

    assert cfg.topology_training.topo_only_after_epoch is None


def test_parse_config_ignores_topology_only_epoch_boundary_when_disabled() -> None:
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    topology_training = config["topology_training"]
    assert isinstance(topology_training, dict)
    topology_training["enabled"] = False
    topology_training["topo_only_after_epoch"] = 1

    cfg = _parse_config(config)

    assert cfg.topology_training.enabled is False
    assert cfg.topology_training.topo_only_after_epoch is None


def test_parse_config_rejects_topology_only_epoch_boundary_with_zero_scale() -> None:
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    topology_training = config["topology_training"]
    assert isinstance(topology_training, dict)
    topology_training["topo_only_after_epoch"] = 1
    topology_training["schedule"] = {"warmup_epochs": 5, "ramp_epochs": 10, "schedule": "linear"}

    with pytest.raises(ValueError, match="topo_only_after_epoch.*topology loss scale"):
        _parse_config(config)


def test_parse_config_rejects_topology_only_epoch_boundary_with_zero_weight() -> None:
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    topology_training = config["topology_training"]
    assert isinstance(topology_training, dict)
    topology_training["topo_only_after_epoch"] = 1
    topology_training["topology_weight"] = 0.0
    topology_training["schedule"] = {"warmup_epochs": 0, "ramp_epochs": 0, "schedule": "linear"}

    with pytest.raises(ValueError, match="topo_only_after_epoch.*topology_weight"):
        _parse_config(config)


def test_parse_config_rejects_topology_only_epoch_boundary_with_zero_loss_weights() -> None:
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    topology_training = config["topology_training"]
    assert isinstance(topology_training, dict)
    topology_training["topo_only_after_epoch"] = 1
    topology_training["topology_weight"] = 1.0
    topology_training["weights"] = {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}
    topology_training["schedule"] = {"warmup_epochs": 0, "ramp_epochs": 0, "schedule": "linear"}

    with pytest.raises(
        ValueError,
        match="topo_only_after_epoch.*active topology loss component weights",
    ):
        _parse_config(config)


def test_parse_config_rejects_topology_only_epoch_boundary_with_delta_only_weight() -> None:
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    topology_training = config["topology_training"]
    assert isinstance(topology_training, dict)
    topology_training["topo_only_after_epoch"] = 1
    topology_training["topology_weight"] = 1.0
    topology_training["weights"] = {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 1.0}
    topology_training["schedule"] = {"warmup_epochs": 0, "ramp_epochs": 0, "schedule": "linear"}

    with pytest.raises(
        ValueError,
        match="topo_only_after_epoch.*active topology loss component weights",
    ):
        _parse_config(config)


def test_parse_config_reads_topology_subset_sampler() -> None:
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    config["topology_training"]["subset"] = {
        "enabled": True,
        "candidate_ratio": 20,
        "pool_ratio": 10,
        "epoch_ratio": 5,
        "hard_fraction": 0.5,
        "uniform_fraction": 0.5,
        "hard_stratum_fraction": 0.2,
        "seed": 11,
    }
    cfg = _parse_config(config)
    assert cfg.topology_training.subset.enabled is True
    assert cfg.topology_training.subset.candidate_ratio == 20
    assert cfg.topology_training.subset.seed == 11
    assert cfg.topology_validation.compute_clustering_mmd is True


def _calibrated_pipeline_config() -> dict[str, object]:
    return {
        "refiner": {
            "monitor_metric": "val_topology_loss",
            "topology_validation": {"enabled": True},
        },
        "graph_selection": {
            "refined_output_rule": {
                "type": "calibrated",
                "objective": "val_topology_loss",
                "grid": [0.5, 0.9, 0.97],
            }
        },
    }


def test_resolve_refined_output_rule_config_accepts_calibrated_grid() -> None:
    parsed = tccig_train._resolve_refined_output_rule_config(_calibrated_pipeline_config())

    assert parsed.calibrated is True
    assert parsed.objective == "val_topology_loss"
    assert parsed.selected_rule_source == "validation_calibration"
    assert [rule.type for rule in parsed.validation_rules] == [
        "threshold",
        "threshold",
        "threshold",
    ]
    assert [float(rule.value) for rule in parsed.validation_rules] == [0.5, 0.9, 0.97]
    assert parsed.fixed_rule.value == pytest.approx(0.5)


def test_resolve_refined_output_rule_config_accepts_calibrated_mode_alias() -> None:
    config = _calibrated_pipeline_config()
    graph_selection = config["graph_selection"]
    assert isinstance(graph_selection, dict)
    refined_rule = graph_selection["refined_output_rule"]
    assert isinstance(refined_rule, dict)
    del refined_rule["type"]
    refined_rule["mode"] = "calibrated"

    parsed = tccig_train._resolve_refined_output_rule_config(config)

    assert parsed.calibrated is True
    assert [float(rule.value) for rule in parsed.validation_rules] == [0.5, 0.9, 0.97]


def test_resolve_refined_output_rule_config_preserves_threshold_default() -> None:
    parsed = tccig_train._resolve_refined_output_rule_config({})

    assert parsed.calibrated is False
    assert parsed.objective is None
    assert parsed.selected_rule_source is None
    assert len(parsed.validation_rules) == 1
    assert parsed.validation_rules[0].to_dict() == {"type": "threshold", "value": 0.5}
    assert parsed.fixed_rule.to_dict() == {"type": "threshold", "value": 0.5}


def test_resolve_refined_output_rule_config_rejects_invalid_calibrated_objective() -> None:
    config = _calibrated_pipeline_config()
    graph_selection = config["graph_selection"]
    assert isinstance(graph_selection, dict)
    refined_rule = graph_selection["refined_output_rule"]
    assert isinstance(refined_rule, dict)
    refined_rule["objective"] = "val_auprc"

    with pytest.raises(ValueError, match="objective must be val_topology_loss"):
        tccig_train._resolve_refined_output_rule_config(config)


def test_resolve_refined_output_rule_config_rejects_empty_calibrated_grid() -> None:
    config = _calibrated_pipeline_config()
    graph_selection = config["graph_selection"]
    assert isinstance(graph_selection, dict)
    refined_rule = graph_selection["refined_output_rule"]
    assert isinstance(refined_rule, dict)
    refined_rule["grid"] = []

    with pytest.raises(ValueError, match="grid must not be empty"):
        tccig_train._resolve_refined_output_rule_config(config)


def test_resolve_refined_output_rule_config_rejects_out_of_range_calibrated_grid() -> None:
    config = _calibrated_pipeline_config()
    graph_selection = config["graph_selection"]
    assert isinstance(graph_selection, dict)
    refined_rule = graph_selection["refined_output_rule"]
    assert isinstance(refined_rule, dict)
    refined_rule["grid"] = [0.5, 1.1]

    with pytest.raises(ValueError, match="grid must be in \\[0, 1\\]"):
        tccig_train._resolve_refined_output_rule_config(config)


def test_resolve_refined_output_rule_config_requires_topology_monitor_setup() -> None:
    no_topology = _calibrated_pipeline_config()
    refiner = no_topology["refiner"]
    assert isinstance(refiner, dict)
    refiner["topology_validation"] = {"enabled": False}

    with pytest.raises(ValueError, match="topology_validation.enabled"):
        tccig_train._resolve_refined_output_rule_config(no_topology)

    wrong_monitor = _calibrated_pipeline_config()
    refiner = wrong_monitor["refiner"]
    assert isinstance(refiner, dict)
    refiner["monitor_metric"] = "val_auprc"

    with pytest.raises(ValueError, match="monitor_metric"):
        tccig_train._resolve_refined_output_rule_config(wrong_monitor)


def test_resolve_refined_output_rule_config_requires_literal_topology_enabled_true() -> None:
    config = _calibrated_pipeline_config()
    refiner = config["refiner"]
    assert isinstance(refiner, dict)
    refiner["topology_validation"] = {"enabled": "false"}

    with pytest.raises(ValueError, match="topology_validation.enabled"):
        tccig_train._resolve_refined_output_rule_config(config)


def test_effective_refined_output_rule_uses_checkpoint_rule_for_calibrated() -> None:
    from types import SimpleNamespace

    from tccig.prepare import GraphRule

    config = tccig_train._resolve_refined_output_rule_config(_calibrated_pipeline_config())
    state = SimpleNamespace(selected_rule=GraphRule(type="threshold", value=0.97))

    effective = tccig_train._effective_refined_output_rule(
        refined_rule_config=config,
        refiner_state=state,
    )

    assert effective.to_dict() == {"type": "threshold", "value": 0.97}


def test_effective_refined_output_rule_rejects_missing_calibrated_selected_rule() -> None:
    from types import SimpleNamespace

    config = tccig_train._resolve_refined_output_rule_config(_calibrated_pipeline_config())
    state = SimpleNamespace(selected_rule=None)

    with pytest.raises(RuntimeError, match="selected_rule"):
        tccig_train._effective_refined_output_rule(
            refined_rule_config=config,
            refiner_state=state,
        )


def test_effective_refined_output_rule_keeps_fixed_threshold_config() -> None:
    from types import SimpleNamespace

    from tccig.prepare import GraphRule

    config = tccig_train._resolve_refined_output_rule_config(
        {"graph_selection": {"refined_output_rule": {"type": "threshold", "value": 0.75}}}
    )
    state = SimpleNamespace(selected_rule=GraphRule(type="threshold", value=0.97))

    effective = tccig_train._effective_refined_output_rule(
        refined_rule_config=config,
        refiner_state=state,
    )

    assert effective.to_dict() == {"type": "threshold", "value": 0.75}


def test_validation_topology_evaluation_selects_grid_argmin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tccig import s2gae
    from tccig.prepare import CandidatePair, GraphRule

    prediction_calls = 0

    def fake_prediction_probabilities(**_kwargs: object) -> list[float]:
        nonlocal prediction_calls
        prediction_calls += 1
        return [0.2, 0.8]

    def fake_validation_topology_metrics(**kwargs: object) -> dict[str, float | int]:
        rule = kwargs["rule"]
        assert isinstance(rule, GraphRule)
        losses = {0.5: 10.0, 0.9: 2.0, 0.97: 5.0}
        return {
            "val_topology_loss": losses[float(rule.value)],
            "graph_sim": 0.1 + float(rule.value),
            "relative_density": 1.0,
            "deg_dist_mmd": 0.0,
            "cc_mmd": 0.0,
            "positive_edges": int(float(rule.value) * 100),
            "val_auprc": 0.42,
        }

    monkeypatch.setattr(s2gae, "_prediction_probabilities", fake_prediction_probabilities)
    monkeypatch.setattr(s2gae, "_validation_topology_metrics", fake_validation_topology_metrics)

    cfg = SimpleNamespace(
        topology_validation=SimpleNamespace(inference_batch_size=8),
    )

    result = s2gae._evaluate_validation_topology_rules(
        model=object(),  # type: ignore[arg-type]
        graph=object(),  # type: ignore[arg-type]
        pairs=[
            CandidatePair(protein_a="A", protein_b="B"),
            CandidatePair(protein_a="C", protein_b="D"),
        ],
        validation_plan=object(),  # type: ignore[arg-type]
        rules=(
            GraphRule(type="threshold", value=0.5),
            GraphRule(type="threshold", value=0.9),
            GraphRule(type="threshold", value=0.97),
        ),
        validation_auprc=0.42,
        cfg=cfg,  # type: ignore[arg-type]
        runtime=object(),
        rule_payload_source="validation_calibration",
    )

    assert prediction_calls == 1
    assert result.rule.to_dict() == {"type": "threshold", "value": 0.9}
    assert result.validation_metrics["val_topology_loss"] == pytest.approx(2.0)
    assert result.rule_payload == {
        "type": "threshold",
        "value": 0.9,
        "source": "validation_calibration",
    }
    assert result.rule_grid == (
        {
            "rule": {"type": "threshold", "value": 0.5, "source": "validation_calibration"},
            "val_topology_loss": 10.0,
            "graph_sim": 0.6,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.0,
            "cc_mmd": 0.0,
            "positive_edges": 50,
            "val_auprc": 0.42,
        },
        {
            "rule": {"type": "threshold", "value": 0.9, "source": "validation_calibration"},
            "val_topology_loss": 2.0,
            "graph_sim": 1.0,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.0,
            "cc_mmd": 0.0,
            "positive_edges": 90,
            "val_auprc": 0.42,
        },
        {
            "rule": {"type": "threshold", "value": 0.97, "source": "validation_calibration"},
            "val_topology_loss": 5.0,
            "graph_sim": 1.07,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.0,
            "cc_mmd": 0.0,
            "positive_edges": 97,
            "val_auprc": 0.42,
        },
    )


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


def test_train_refiner_fixed_rule_fallback_persists_selected_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tccig import s2gae
    from tccig.prepare import CandidatePair, GraphRule, SplitBundle, TCCIGRuntime

    class TinyAccelerator:
        def __init__(self) -> None:
            self.saved_payload: dict[str, object] | None = None

        def prepare(self, *components: object) -> object:
            if len(components) == 1:
                return components[0]
            return components

        def backward(self, loss: torch.Tensor) -> None:
            loss.backward()

        def reduce(self, value: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
            assert reduction == "sum"
            return value

        def wait_for_everyone(self) -> None:
            return None

        def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
            return model

        def save(
            self,
            obj: object,
            f: str | Path,
            safe_serialization: bool = False,
        ) -> None:
            assert safe_serialization is False
            assert isinstance(obj, dict)
            self.saved_payload = obj
            torch.save(obj, f)

    def fake_node_features(
        *,
        protein_ids: Sequence[str],
        cache_dir: Path,
        index_path: Path,
        input_dim: int,
        max_sequence_length: int | None,
        device: torch.device,
    ) -> torch.Tensor:
        del cache_dir, index_path, max_sequence_length
        rows = []
        for index, _protein_id in enumerate(protein_ids):
            rows.append([float(index + 1), float((index + 1) * 2)])
        return torch.tensor(rows, dtype=torch.float32, device=device)[:, :input_dim]

    fixed_rule = GraphRule(type="threshold", value=0.5)
    observed_rules: list[tuple[GraphRule, ...]] = []

    def fake_evaluate_validation_topology_rules(**kwargs: object) -> object:
        rules = kwargs["rules"]
        assert isinstance(rules, tuple)
        observed_rules.append(rules)
        payload = dict(rules[0].to_dict())
        return s2gae.ValidationTopologyRuleEvaluation(
            rule=rules[0],
            validation_metrics={
                "val_topology_loss": 3.0,
                "graph_sim": 0.7,
                "relative_density": 1.0,
                "deg_dist_mmd": 0.0,
                "cc_mmd": 0.0,
                "positive_edges": 2,
                "val_auprc": 0.42,
            },
            rule_payload=payload,
            rule_grid=(),
        )

    monkeypatch.setattr(s2gae, "load_mean_pooled_node_features", fake_node_features)
    monkeypatch.setattr(s2gae, "_validation_auprc", lambda **_kwargs: 0.42)
    monkeypatch.setattr(
        s2gae,
        "_evaluate_validation_topology_rules",
        fake_evaluate_validation_topology_rules,
    )

    pairs = [
        CandidatePair("A", "B"),
        CandidatePair("A", "C"),
        CandidatePair("B", "C"),
        CandidatePair("C", "D"),
    ]
    split = SplitBundle(
        split="train",
        pairs=pairs,
        pairwise_probabilities=[0.9, 0.8, 0.2, 0.1],
        pairwise_graph_edges=[("A", "B"), ("A", "C")],
        candidate_labels=[1, 0, 1, 0],
        loss_targets=[1, 0, 1, 0],
    )
    validation_topology = SplitBundle(
        split="validation_topology",
        pairs=pairs[:2],
        pairwise_probabilities=[0.9, 0.8],
        pairwise_graph_edges=[("A", "B")],
        candidate_labels=[1, 0],
    )
    accelerator = TinyAccelerator()
    runtime = TCCIGRuntime(
        accelerator=accelerator,  # type: ignore[arg-type]
        device="cpu",
        backend="single",
        mixed_precision="no",
        is_distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        is_main_process=True,
    )

    state = s2gae.train_refiner(
        s2gae.TrainRefinerRequest(
            train=split,
            validation=split,
            runtime=runtime,
            config={
                "input_dim": 2,
                "hidden_dim": 4,
                "num_layers": 1,
                "decoder_hidden_dim": 4,
                "decoder_layers": 1,
                "dropout": 0.0,
                "epochs": 1,
                "batch_size": 4,
                "embedding_cache_dir": str(tmp_path / "embeddings"),
                "monitor_metric": "val_topology_loss",
                "topology_validation": {
                    "enabled": True,
                    "inference_batch_size": 4,
                    "compute_clustering_mmd": False,
                },
                "topology_training": {"enabled": False},
                "optimizer": {"type": "adamw", "lr": 0.001},
                "scheduler": {"type": "none"},
                "optimization": {"gradient_clip_norm": None},
                "log_dir": str(tmp_path / "logs"),
                "checkpoint_path": str(tmp_path / "model.pt"),
            },
            graph_rule=fixed_rule,
            validation_topology=validation_topology,
            validation_topology_plan=object(),
        )
    )

    expected_payload = {"type": "threshold", "value": 0.5}
    assert observed_rules == [(fixed_rule,)]
    assert state.selected_rule_payload == expected_payload
    assert accelerator.saved_payload is not None
    assert accelerator.saved_payload["selected_rule"] == expected_payload

    summary = json.loads(
        (tmp_path / "logs" / "training_summary.json").read_text(encoding="utf-8")
    )
    assert summary["history"][0]["selected_rule"] == expected_payload
    assert summary["selected_rule"] == expected_payload

    csv_header = (
        (tmp_path / "logs" / "tccig_train_step.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert csv_header.split(",") == s2gae.TCCIG_TRAIN_CSV_COLUMNS
    assert "selected_rule" not in csv_header


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
        plan = build_internal_validation_plan(graph=graph, sampled_subgraphs={3: [("a", "b", "c")]})
        return plan, {
            "base_bucket_count": 1,
            "coverage_bucket_count": 0,
            "positive_edge_coverage": 1.0,
        }

    common = {
        "split": "train_topology",
        "graph": graph,
        "node_sizes": [3],
        "samples_per_size": 1,
        "seed": 0,
        "strategy": "bfs",
        "coverage_augmentation": True,
        "runtime": _fake_runtime(is_main_process=True),
        "cache_dir": tmp_path,
        "build_fn": build_fn,
    }
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


def test_build_train_topology_bundle_uses_plan_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _coverage_graph()

    monkeypatch.setattr(tccig_train, "load_split_node_ids", lambda **_: set(graph.nodes()))
    monkeypatch.setattr(tccig_train, "build_pair_supervision_graph", lambda **_: graph)

    sample_calls = {"n": 0}
    real_sample = tccig_train.sample_topology_evaluation_subgraphs

    def counting_sample(**kwargs: object) -> object:
        sample_calls["n"] += 1
        return real_sample(**kwargs)

    monkeypatch.setattr(tccig_train, "sample_topology_evaluation_subgraphs", counting_sample)

    captured: dict[str, object] = {}

    def fake_score_split(
        *,
        split: str,
        pairs: Sequence[object],
        scorer_cfg: object,
        runtime: object,
        cache_dir: object,
    ) -> list[float]:
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
    common = {
        "config": config,
        "processed_dir": tmp_path,
        "scorer_cfg": {},
        "runtime": runtime,
        "cache_dir": tmp_path,
        "pairwise_input_rule": tccig_train._resolve_refined_output_rule({}),
    }

    bundle_first, plan_first, _diag_first, stats_first = (
        tccig_train._build_train_topology_bundle(**common)
    )
    bundle_second, plan_second, _diag_second, stats_second = (
        tccig_train._build_train_topology_bundle(**common)
    )

    assert sample_calls["n"] == 1  # second run served from cache
    assert plan_first.total_pairs == plan_second.total_pairs
    assert stats_first == stats_second
    assert set(stats_first) == {
        "base_bucket_count",
        "coverage_bucket_count",
        "positive_edge_coverage",
    }


def test_build_train_topology_subset_scores_diagnostic_full_space_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _coverage_graph()

    monkeypatch.setattr(tccig_train, "load_split_node_ids", lambda **_: set(graph.nodes()))
    monkeypatch.setattr(tccig_train, "build_pair_supervision_graph", lambda **_: graph)

    calls: list[tuple[str, int]] = []

    def fake_score_split(
        *,
        split: str,
        pairs: Sequence[object],
        scorer_cfg: object,
        runtime: object,
        cache_dir: object,
    ) -> list[float]:
        calls.append((split, len(pairs)))
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
                "coverage_augmentation": False,
                "subset": {
                    "enabled": True,
                    "candidate_ratio": 2,
                    "pool_ratio": 1,
                    "epoch_ratio": 1,
                    "bias_diagnostic_max_node_size": 4,
                    "bias_diagnostic_max_subgraphs": 1,
                },
            }
        }
    }

    bundle, _plan, diagnostic_full_space, _stats = tccig_train._build_train_topology_bundle(
        config=config,
        processed_dir=tmp_path,
        scorer_cfg={},
        runtime=_fake_runtime(is_main_process=True),
        cache_dir=tmp_path,
        pairwise_input_rule=tccig_train._resolve_refined_output_rule({}),
    )

    diagnostic_calls = [
        count for split, count in calls if split == "train_topology_subset_diagnostic"
    ]
    assert diagnostic_calls == [6]
    assert diagnostic_full_space is not None
    assert {len(rows) for rows in diagnostic_full_space.values()} == {6}
    assert bundle is not None
    assert bundle.extra_node_ids is not None


def test_build_train_topology_subset_rescores_incomplete_diagnostic_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _coverage_graph()

    monkeypatch.setattr(tccig_train, "load_split_node_ids", lambda **_: set(graph.nodes()))
    monkeypatch.setattr(tccig_train, "build_pair_supervision_graph", lambda **_: graph)

    calls: list[tuple[str, int]] = []

    def fake_score_split(
        *,
        split: str,
        pairs: Sequence[object],
        scorer_cfg: object,
        runtime: object,
        cache_dir: object,
    ) -> list[float]:
        calls.append((split, len(pairs)))
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
                "coverage_augmentation": False,
                "subset": {
                    "enabled": True,
                    "candidate_ratio": 2,
                    "pool_ratio": 1,
                    "epoch_ratio": 1,
                    "bias_diagnostic_max_node_size": 4,
                    "bias_diagnostic_max_subgraphs": 1,
                },
            }
        }
    }
    common = {
        "config": config,
        "processed_dir": tmp_path,
        "scorer_cfg": {},
        "runtime": _fake_runtime(is_main_process=True),
        "cache_dir": tmp_path,
        "pairwise_input_rule": tccig_train._resolve_refined_output_rule({}),
    }

    _bundle, _plan, diagnostic_full_space, _stats = tccig_train._build_train_topology_bundle(
        **common
    )
    assert diagnostic_full_space is not None

    cache_path = tmp_path / "plans" / "train_topology_subset_diagnostic.json"
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    payload = document["payload"]
    subgraph_id = next(iter(payload))
    payload[subgraph_id].pop(next(iter(payload[subgraph_id])))
    cache_path.write_text(json.dumps(document), encoding="utf-8")

    _bundle, _plan, repaired_full_space, _stats = tccig_train._build_train_topology_bundle(
        **common
    )

    diagnostic_calls = [
        count for split, count in calls if split == "train_topology_subset_diagnostic"
    ]
    assert diagnostic_calls == [6, 6]
    assert repaired_full_space is not None
    assert {len(rows) for rows in repaired_full_space.values()} == {6}


def test_coverage_stats_from_payload_extracts_numeric_keys() -> None:
    payload = {
        "coverage_stats": {
            "base_bucket_count": 3,
            "coverage_bucket_count": 2,
            "positive_edge_coverage": 1.0,
        }
    }
    stats = _coverage_stats_from_payload(payload)
    assert stats == {
        "base_bucket_count": 3,
        "coverage_bucket_count": 2,
        "positive_edge_coverage": 1.0,
    }


def test_coverage_stats_from_payload_missing_key_returns_empty() -> None:
    assert _coverage_stats_from_payload({}) == {}


def test_coverage_stats_from_payload_non_mapping_returns_empty() -> None:
    assert _coverage_stats_from_payload({"coverage_stats": "bad"}) == {}


def test_coverage_stats_from_payload_drops_non_numeric_values() -> None:
    payload = {
        "coverage_stats": {
            "base_bucket_count": 3,
            "positive_edge_coverage": "bad",
            "coverage_bucket_count": None,
        }
    }
    # Non-numeric values must not survive: the log path calls float() on them.
    assert _coverage_stats_from_payload(payload) == {"base_bucket_count": 3}


def test_split_graph_extra_node_ids_get_features_but_not_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tccig.prepare import CandidatePair, SplitBundle
    from tccig.s2gae import _build_split_graph, _node_index_from_split_bundle, _parse_config

    def _fake_features(
        *,
        protein_ids: Sequence[str],
        cache_dir: Path,
        index_path: Path,
        input_dim: int,
        max_sequence_length: int | None,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.arange(
            len(protein_ids) * input_dim,
            dtype=torch.float32,
            device=device,
        ).reshape(len(protein_ids), input_dim)

    monkeypatch.setattr("tccig.s2gae.load_mean_pooled_node_features", _fake_features)
    cfg = _parse_config(_base_refiner_config())
    bundle = SplitBundle(
        split="train_topology",
        pairs=[CandidatePair("a", "b")],
        pairwise_probabilities=[0.8],
        pairwise_graph_edges=[],
        extra_node_ids=["c", "d"],
    )

    graph = _build_split_graph(bundle, cfg=cfg, device=torch.device("cpu"))
    node_index = _node_index_from_split_bundle(bundle)

    assert set(node_index) == {"a", "b", "c", "d"}
    assert graph.node_features.shape[0] == 4
    assert graph.pair_index.shape[1] == 1
    assert graph.edge_index.numel() == 0


def test_subgraph_bias_diagnostic_uses_supplied_full_space_probabilities() -> None:
    from tccig.s2gae import _SplitGraph, _subgraph_bias_diagnostic
    from tccig.topology_subset import (
        SamplingStratum,
        TopologyPairSample,
        TopologySubgraphEpochChunk,
        TopologySubgraphPlan,
    )

    class _IdentityRefiner:
        def encode(
            self,
            *,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: torch.Tensor,
        ) -> list[torch.Tensor]:
            return [node_features]

        def decode(
            self,
            *,
            hidden_states: Sequence[torch.Tensor],
            pair_index: torch.Tensor,
            pairwise_probabilities: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.logit(pairwise_probabilities.clamp(1.0e-6, 1.0 - 1.0e-6))
            return logits, torch.zeros_like(logits)

    subgraph = TopologySubgraphPlan(
        subgraph_id="size=4:index=0",
        node_size=4,
        nodes=("a", "b", "c", "d"),
        positives=(),
        candidate_negatives=(),
        hard_pool=(),
        uniform_pool=(),
    )
    chunk = TopologySubgraphEpochChunk(
        subgraph_id=subgraph.subgraph_id,
        node_size=subgraph.node_size,
        samples=(
            TopologyPairSample(
                pair_id="a||b",
                subgraph_id=subgraph.subgraph_id,
                node_size=4,
                protein_a="a",
                protein_b="b",
                local_index_a=0,
                local_index_b=1,
                stratum=SamplingStratum.POSITIVE,
                pi_cand=1.0,
                pi_pool_given_cand=1.0,
                pi_epoch_given_pool=1.0,
                pi_total=1.0,
                target=1.0,
                scorer_probability=0.8,
            ),
        ),
    )
    graph = _SplitGraph(
        node_features=torch.ones((4, 2), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weight=torch.empty((0,), dtype=torch.float32),
        pair_index=torch.tensor([[0], [1]], dtype=torch.long),
        pairwise_probabilities=torch.tensor([0.8], dtype=torch.float32),
    )
    diagnostic_full_space = {
        subgraph.subgraph_id: {
            "a||b": 0.8,
            "a||c": 0.1,
            "a||d": 0.2,
            "b||c": 0.3,
            "b||d": 0.4,
            "c||d": 0.5,
        }
    }

    diagnostic = _subgraph_bias_diagnostic(
        refiner=_IdentityRefiner(),  # type: ignore[arg-type]
        graph=graph,
        subgraph=subgraph,
        chunk=chunk,
        node_index={"a": 0, "b": 1, "c": 2, "d": 3},
        diagnostic_full_space=diagnostic_full_space,
    )

    assert diagnostic is not None
    assert diagnostic["full_space_pairs"] == 6
    assert diagnostic["subset_pairs"] == 1


def test_subset_diagnostic_payload_metadata_changes_with_diagnostic_knobs() -> None:
    from src.topology.plan_cache import subset_diagnostic_payload_metadata

    graph = nx.Graph()
    graph.add_edge("a", "b")
    common = {
        "split": "train_topology_subset_diagnostic",
        "graph": graph,
        "node_sizes": [4],
        "samples_per_size": 1,
        "seed": 7,
        "strategy": "bfs",
        "coverage_augmentation": True,
        "candidate_ratio": 20,
        "pool_ratio": 10,
        "epoch_ratio": 5,
        "hard_fraction": 0.5,
        "uniform_fraction": 0.5,
        "hard_stratum_fraction": 0.2,
        "max_subgraphs_per_size": 0,
        "max_labeled_pairs_per_size": 0,
        "bias_diagnostic_max_node_size": 4,
        "bias_diagnostic_max_subgraphs": 2,
        "scorer_config": {},
    }

    baseline = subset_diagnostic_payload_metadata(**common)
    changed_size = subset_diagnostic_payload_metadata(
        **{**common, "bias_diagnostic_max_node_size": 8}
    )
    changed_count = subset_diagnostic_payload_metadata(
        **{**common, "bias_diagnostic_max_subgraphs": 3}
    )

    assert baseline["pair_scope"] == "subset_diagnostic"
    assert baseline != changed_size
    assert baseline != changed_count


def test_topology_subset_chunk_loss_uses_inclusion_weights() -> None:
    from src.topology.finetune_losses import TopologyLossWeights
    from tccig.s2gae import topology_subset_chunk_loss
    from tccig.topology_subset import (
        SamplingStratum,
        TopologyPairSample,
        TopologySubgraphEpochChunk,
    )

    chunk = TopologySubgraphEpochChunk(
        subgraph_id="size=3:index=0",
        node_size=3,
        samples=(
            TopologyPairSample(
                pair_id="a||b",
                subgraph_id="size=3:index=0",
                node_size=3,
                protein_a="a",
                protein_b="b",
                local_index_a=0,
                local_index_b=1,
                stratum=SamplingStratum.POSITIVE,
                pi_cand=1.0,
                pi_pool_given_cand=1.0,
                pi_epoch_given_pool=1.0,
                pi_total=1.0,
                target=1.0,
                scorer_probability=0.9,
            ),
            TopologyPairSample(
                pair_id="b||c",
                subgraph_id="size=3:index=0",
                node_size=3,
                protein_a="b",
                protein_b="c",
                local_index_a=1,
                local_index_b=2,
                stratum=SamplingStratum.UNIFORM_NEGATIVE,
                pi_cand=0.5,
                pi_pool_given_cand=1.0,
                pi_epoch_given_pool=1.0,
                pi_total=0.5,
                target=0.0,
                scorer_probability=0.2,
            ),
        ),
    )
    refined_logits = torch.tensor([2.0, -1.0], requires_grad=True)
    loss, components = topology_subset_chunk_loss(
        refined_logits=refined_logits,
        chunk=chunk,
        weights=TopologyLossWeights(alpha=1.0, beta=1.0, gamma=1.0, delta=0.0),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert refined_logits.grad is not None
    assert components["sample_count"] == 2.0


def test_size_balanced_topology_normalizers_ignore_subgraph_imbalance() -> None:
    from tccig.s2gae import _size_balanced_chunk_scales

    scales = _size_balanced_chunk_scales([20, 20, 20, 200])
    assert scales == pytest.approx([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0])


def test_shard_chunks_partition_is_disjoint_and_complete() -> None:
    from tccig.s2gae import _shard_chunks_for_rank

    node_sizes = [20, 20, 20, 200, 200]
    world_size = 2
    seen: list[int] = []
    for rank in range(world_size):
        shard = _shard_chunks_for_rank(
            node_sizes=node_sizes, rank=rank, world_size=world_size
        )
        # Global scale must use GLOBAL counts (S=2 sizes, N_20=3, N_200=2),
        # not the per-rank counts, regardless of which chunks land on this rank.
        for global_index, scale in shard:
            expected_n = 3 if node_sizes[global_index] == 20 else 2
            assert scale == pytest.approx(1.0 / (2 * expected_n))
            seen.append(global_index)
    # Disjoint and complete cover of every chunk index exactly once.
    assert sorted(seen) == list(range(len(node_sizes)))


def test_train_refiner_accepts_subset_plan_object() -> None:
    from tccig.s2gae import TrainRefinerRequest
    from tccig.topology_subset import TopologySubsetPlan

    request = TrainRefinerRequest(
        train=None,  # type: ignore[arg-type]
        validation=None,  # type: ignore[arg-type]
        runtime=None,  # type: ignore[arg-type]
        config={},
        graph_rule=None,  # type: ignore[arg-type]
        train_topology_plan=TopologySubsetPlan(
            subgraphs=(),
            active_sizes=(),
            skipped_sizes={},
            total_positive_pairs=0,
            total_candidate_negatives=0,
            total_pool_negatives=0,
        ),
    )
    assert isinstance(request.train_topology_plan, TopologySubsetPlan)


def test_score_progress_interval_emits_periodic_events() -> None:
    from tccig.train import _score_progress_events

    events = list(_score_progress_events(total_pairs=1000, interval_pairs=250))
    assert events == [250, 500, 750, 1000]


def test_score_progress_pointer_fires_when_batch_overshoots_milestone() -> None:
    # Finding 6: `processed` advances by batch size and lands BETWEEN milestones, so an
    # exact `processed in milestones` test would never fire. Replicate the closure's
    # advancing-pointer logic and assert every crossed milestone is drained exactly once.
    from tccig.train import _score_progress_events

    milestones = _score_progress_events(total_pairs=1000, interval_pairs=250)
    fired: list[int] = []
    pointer = 0
    # Batches overshoot 250 (->300), 500/750 in one jump (->760), then finish (->1000).
    for processed in (300, 760, 1000):
        while pointer < len(milestones) and processed >= milestones[pointer]:
            fired.append(milestones[pointer])
            pointer += 1
    # Every milestone fires once, in order, despite none equalling a `processed` value.
    assert fired == [250, 500, 750, 1000]


def test_balanced_subset_configs_parse() -> None:
    from pathlib import Path

    import yaml
    from tccig.s2gae import _parse_config

    for path in (
        Path("configs/tccig/02_balanced_subset.yaml"),
        Path("configs/tccig/02_balanced_subset_smoke.yaml"),
    ):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        cfg = _parse_config(config["refiner"])
        assert cfg.topology_training.subset.enabled is True
        # Review Finding 11: clustering stays ON in validation/test metrics (spec §2
        # non-goal). The subset TRAINING path keeps clustering off independently
        # (Task 8 hardcodes include_clustering_mmd=False in the chunk loss).
        assert cfg.topology_validation.compute_clustering_mmd is True


def test_smoke_config_engages_topology_in_epoch_one() -> None:
    # Review Finding 3: the smoke config must reach a POSITIVE topology scale in
    # epoch 1, otherwise the smoke run silently exercises none of the new path.
    # train_refiner calls topology_loss_scale(epoch=epoch-1, ...), so epoch 1 uses
    # index 0. With warmup_epochs=0 and ramp_epochs=0 the scale must be > 0 there.
    from pathlib import Path

    import yaml
    from src.topology.finetune_losses import (
        TopologyLossWeightSchedule,
        topology_loss_scale,
    )
    from tccig.s2gae import _parse_config

    config = yaml.safe_load(
        Path("configs/tccig/02_balanced_subset_smoke.yaml").read_text(encoding="utf-8")
    )
    cfg = _parse_config(config["refiner"])
    # train_refiner builds the schedule from the three flat training-config fields
    # (see tccig/s2gae.py: TopologyLossWeightSchedule(...)). Mirror that here.
    schedule = TopologyLossWeightSchedule(
        warmup_epochs=cfg.topology_training.warmup_epochs,
        ramp_epochs=cfg.topology_training.ramp_epochs,
        schedule=cfg.topology_training.schedule,
    )
    scale_epoch_one = topology_loss_scale(epoch=0, schedule=schedule)
    assert scale_epoch_one > 0.0


def test_config_to_json_serializes_topology_subset_fields() -> None:
    import yaml
    from tccig.s2gae import _config_to_json, _parse_config

    raw = yaml.safe_load(
        Path("configs/tccig/02_balanced_subset_smoke.yaml").read_text(encoding="utf-8")
    )
    cfg = _parse_config(raw["refiner"])

    subset = _config_to_json(cfg)["topology_training"]["subset"]

    assert subset == {
        "enabled": True,
        "candidate_ratio": 4,
        "pool_ratio": 2,
        "epoch_ratio": 2,
        "hard_fraction": 0.5,
        "uniform_fraction": 0.5,
        "hard_stratum_fraction": 0.5,
        "seed": 0,
        "max_subgraphs_per_size": 0,
        "max_labeled_pairs_per_size": 0,
        "bias_diagnostic_every_n_epochs": 1,
        "bias_diagnostic_max_node_size": 40,
        "bias_diagnostic_max_subgraphs": 4,
    }


def test_training_summary_artifact_persists_topology_subset_block(tmp_path: Path) -> None:
    import dataclasses

    import yaml
    from tccig.s2gae import _config_to_json, _parse_config, _write_training_summary

    raw = yaml.safe_load(
        Path("configs/tccig/02_balanced_subset_smoke.yaml").read_text(encoding="utf-8")
    )
    cfg = _parse_config(raw["refiner"])
    cfg = dataclasses.replace(cfg, log_dir=tmp_path)

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    _write_training_summary(
        cfg=cfg,
        best_monitor_value=0.0,
        best_validation_auprc=0.0,
        best_selected_rule_payload=None,
        optimizer=optimizer,
        history=[],
    )

    summary = json.loads((tmp_path / "training_summary.json").read_text(encoding="utf-8"))
    # The subset block must survive into the persisted artifact, not only the in-memory
    # _config_to_json result — _write_training_summary nests it under "config".
    assert summary["config"]["topology_training"]["subset"] == _config_to_json(cfg)[
        "topology_training"
    ]["subset"]


def test_exp02_balanced_subset_config_uses_calibrated_rule_and_topology_only_probe() -> None:
    import yaml

    raw = yaml.safe_load(
        Path("configs/tccig/02_balanced_subset.yaml").read_text(encoding="utf-8")
    )

    graph_selection = raw["graph_selection"]
    refined_rule = graph_selection["refined_output_rule"]
    assert refined_rule == {
        "type": "calibrated",
        "objective": "val_topology_loss",
        "grid": [0.5, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99],
    }
    assert graph_selection["rules"] == [{"type": "threshold", "value": 0.5}]

    topology_training = raw["refiner"]["topology_training"]
    assert topology_training["topo_only_after_epoch"] == 7
    assert raw["refiner"]["monitor_metric"] == "val_topology_loss"
    assert raw["refiner"]["topology_validation"]["enabled"] is True
