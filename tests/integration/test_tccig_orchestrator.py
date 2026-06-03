"""Integration tests for the standalone TCCIG PRING orchestrator."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import networkx as nx
import torch
import yaml
from src.pipeline.stages.train import build_model
from tccig.train import (
    PairwiseScoreRequest,
    RefineRequest,
    TrainRefinerRequest,
    run_tccig_pipeline,
)

HOOK_EVENTS: list[tuple[str, object]] = []
SCORE_PAIR_COUNTS: list[tuple[str, int]] = []


def _write_pairs(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a}\t{protein_b}\t{label}\n")


def _write_tiny_pring_fixture(processed_dir: Path) -> None:
    _write_pairs(
        processed_dir / "human_train_ppi_ratio5_exclusive.txt",
        [("A", "B", 1), ("A", "A", 1), ("A", "C", 0)],
    )
    _write_pairs(
        processed_dir / "human_val_ppi_ratio5_exclusive.txt",
        [("A", "B", 1), ("B", "B", 1), ("A", "C", 0)],
    )
    _write_pairs(
        processed_dir / "human_test_ppi.txt",
        [("T1", "T2", 1), ("T1", "T1", 1), ("T2", "T3", 0)],
    )
    _write_pairs(
        processed_dir / "all_test_ppi.txt",
        [("T1", "T2", 1), ("T1", "T3", 0), ("T3", "T3", 1)],
    )

    graph = nx.Graph()
    graph.add_nodes_from(["T1", "T2", "T3"])
    graph.add_edge("T1", "T2")
    with (processed_dir / "human_test_graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)
    with (processed_dir / "test_sampled_nodes.pkl").open("wb") as handle:
        pickle.dump({2: [["T1", "T2"], ["T1", "T3"]]}, handle)
    with (processed_dir / "human_BFS_split.pkl").open("wb") as handle:
        pickle.dump({"train": {"A", "B", "C"}, "val": {"A", "B", "C"}}, handle)


def _tiny_v3_1_config() -> dict[str, object]:
    return {
        "model_config": {
            "model": "v3.1",
            "input_dim": 8,
            "d_model": 8,
            "encoder_layers": 1,
            "cross_attn_layers": 1,
            "n_heads": 2,
            "mlp_head": {
                "hidden_dims": [8, 4],
                "dropout": 0.0,
                "activation": "gelu",
                "norm": "layernorm",
            },
            "regularization": {
                "dropout": 0.0,
                "token_dropout": 0.0,
                "cross_attention_dropout": 0.0,
                "stochastic_depth": 0.0,
            },
            "rich_pooling": {"components": ["mean", "attn", "max", "gated"]},
            "pair_readout": {
                "mode": "pair_context_gated",
                "order_aggregation": "abba_max",
            },
            "interaction": {"mode": "none"},
        }
    }


def _write_tiny_v3_1_pairwise_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_config_path = tmp_path / "v3_1_abba_no_cross.yaml"
    model_config_path.write_text(yaml.safe_dump(_tiny_v3_1_config()), encoding="utf-8")

    checkpoint_path = tmp_path / "models" / "best_model.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_model(_tiny_v3_1_config()).state_dict(), checkpoint_path)

    cache_dir = tmp_path / "cache"
    index: dict[str, str] = {}
    for protein_id in ["A", "B", "C", "T1", "T2", "T3"]:
        relative_path = f"embeddings/{protein_id}.pt"
        output_path = cache_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones((3, 8), dtype=torch.float32), output_path)
        index[protein_id] = relative_path
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return model_config_path, checkpoint_path, cache_dir


def fake_score_pairs(request: PairwiseScoreRequest) -> list[float]:
    """Fake external pairwise scorer that enforces label-free inputs."""
    assert all(not hasattr(pair, "label") for pair in request.pairs)
    assert all(pair.protein_a != pair.protein_b for pair in request.pairs)
    assert not hasattr(request, "human_test_graph")
    assert not hasattr(request, "test_sampled_nodes")
    HOOK_EVENTS.append(("score", request.split))
    SCORE_PAIR_COUNTS.append((request.split, len(request.pairs)))
    return [0.9 if pair.protein_b in {"B", "T2"} else 0.1 for pair in request.pairs]


def fake_train_refiner(request: TrainRefinerRequest) -> dict[str, str]:
    """Fake trainer that sees train targets but no test graph artifacts."""
    assert request.train.loss_targets == [1, 0]
    assert request.train.candidate_labels == [1, 0]
    assert request.train.graph_edges == [("A", "B")]
    assert not hasattr(request, "human_test_graph")
    assert not hasattr(request, "test_sampled_nodes")
    HOOK_EVENTS.append(("train_refiner", len(request.train.pairs)))
    return {"state": "fake"}


def fake_predict_refined(request: RefineRequest) -> list[float]:
    """Fake refiner that receives pairwise graph inputs but not ground truth graphs."""
    assert all(not hasattr(pair, "label") for pair in request.pairs)
    assert not hasattr(request, "human_test_graph")
    assert not hasattr(request, "test_sampled_nodes")
    HOOK_EVENTS.append(("predict_refined", request.split))
    return list(request.pairwise_probabilities)


def test_tccig_orchestrator_keeps_pring_truth_out_of_model_inputs(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "PRING" / "human" / "BFS"
    _write_tiny_pring_fixture(processed_dir)
    HOOK_EVENTS.clear()

    result = run_tccig_pipeline(
        {
            "run": {"run_id": "tiny", "log_root": str(tmp_path / "logs")},
            "data": {"processed_dir": str(processed_dir)},
            "device": {
                "device": "cpu",
                "backend": "ddp",
                "mixed_precision": False,
            },
            "pairwise_scorer": {
                "target": f"{__name__}:fake_score_pairs",
            },
            "refiner": {
                "train_target": f"{__name__}:fake_train_refiner",
                "predict_target": f"{__name__}:fake_predict_refined",
            },
            "graph_selection": {
                "rules": [
                    {"type": "threshold", "value": 0.5},
                    {"type": "top_k", "k": 1},
                    {"type": "top_m", "m": 1},
                ]
            },
        }
    )

    assert ("score", "train") in HOOK_EVENTS
    assert ("score", "topology_test") in HOOK_EVENTS
    assert ("predict_refined", "topology_test") in HOOK_EVENTS
    assert result.manifest["self_pair_rows_dropped"] == {
        "train": 1,
        "validation": 1,
        "pairwise_test": 1,
        "topology_test": 1,
    }
    assert result.selected_rule["type"] in {"threshold", "top_k", "top_m"}

    topology_log_dir = tmp_path / "logs" / "tccig" / "topology_test" / "tiny"
    assert (topology_log_dir / "all_test_ppi_pred.txt").exists()
    metrics_path = topology_log_dir / "topology_metrics.json"
    metrics_csv_path = topology_log_dir / "topology_metrics.csv"
    assert metrics_path.exists()
    assert metrics_csv_path.exists()
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(metrics_payload["summary"]) == {
        "graph_sim",
        "relative_density",
        "deg_dist_mmd",
        "cc_mmd",
        "laplacian_eigen_mmd",
    }
    assert "2" in metrics_payload["per_node_size"]
    assert set(metrics_payload["details"]) == set(metrics_payload["summary"])
    assert metrics_payload["selected_rule"]["type"] in {"threshold", "top_k", "top_m"}
    assert metrics_payload["pairwise_graph_rule"] == {"type": "threshold", "value": 0.5}
    assert metrics_payload["protocol"] == {
        "candidate_universe": "all_test_ppi.txt",
        "ground_truth_graph": "human_test_graph.pkl",
        "sampled_nodes": "test_sampled_nodes.pkl",
        "test_labels_visible_to_model": False,
    }
    with metrics_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["scope"] for row in rows] == ["node_size", "summary"]
    assert rows[-1]["node_size"] == "all"
    assert rows[-1]["graph_count"] == "2"

    scoring_manifest = tmp_path / "logs" / "tccig" / "score" / "tiny" / "train.json"
    assert scoring_manifest.exists()
    assert '"pair_count": 2' in scoring_manifest.read_text(encoding="utf-8")


def test_validation_selected_rule_is_reused_for_topology_test(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "PRING" / "human" / "BFS"
    _write_tiny_pring_fixture(processed_dir)
    HOOK_EVENTS.clear()

    result = run_tccig_pipeline(
        {
            "run": {"run_id": "top_m_case", "log_root": str(tmp_path / "logs")},
            "data": {"processed_dir": str(processed_dir)},
            "device": {"device": "cpu", "backend": "ddp", "mixed_precision": False},
            "pairwise_scorer": {"target": f"{__name__}:fake_score_pairs"},
            "refiner": {
                "train_target": f"{__name__}:fake_train_refiner",
                "predict_target": f"{__name__}:fake_predict_refined",
            },
            "graph_selection": {
                "rules": [
                    {"type": "threshold", "value": 0.95},
                    {"type": "top_m", "m": 1},
                ]
            },
        }
    )

    assert result.selected_rule["type"] == "top_m"
    assert result.pairwise_metrics["f1"] == 1.0
    assert result.pairwise_metrics["mcc"] == 1.0

    selected_rule_path = (
        tmp_path / "logs" / "tccig" / "validation" / "top_m_case" / "selected_rule.json"
    )
    selected_rule = json.loads(selected_rule_path.read_text(encoding="utf-8"))
    assert selected_rule["type"] == "top_m"
    assert selected_rule["m"] == 1

    prediction_path = (
        tmp_path / "logs" / "tccig" / "topology_test" / "top_m_case" / "all_test_ppi_pred.txt"
    )
    predicted_rows = prediction_path.read_text(encoding="utf-8").strip().splitlines()
    assert predicted_rows == ["T1\tT2\t1", "T1\tT3\t0"]

    metrics_payload = json.loads(
        (
            tmp_path / "logs" / "tccig" / "topology_test" / "top_m_case" / "topology_metrics.json"
        ).read_text(encoding="utf-8")
    )
    assert metrics_payload["selected_rule"] == {"type": "top_m", "m": 1}
    assert metrics_payload["pair_counts"] == {
        "candidate_pairs": 2,
        "pairwise_graph_edges": 1,
        "refined_positive_edges": 1,
    }


def test_tccig_builds_validation_topology_bucket_all_pairs(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "PRING" / "human" / "BFS"
    _write_tiny_pring_fixture(processed_dir)
    HOOK_EVENTS.clear()
    SCORE_PAIR_COUNTS.clear()

    run_tccig_pipeline(
        {
            "run": {"run_id": "validation_topology", "log_root": str(tmp_path / "logs")},
            "data": {"processed_dir": str(processed_dir)},
            "device": {"device": "cpu", "backend": "ddp", "mixed_precision": False},
            "pairwise_scorer": {"target": f"{__name__}:fake_score_pairs"},
            "refiner": {
                "train_target": f"{__name__}:fake_train_refiner",
                "predict_target": f"{__name__}:fake_predict_refined",
                "monitor_metric": "val_auprc",
                "topology_validation": {
                    "enabled": True,
                    "node_sizes": [3],
                    "samples_per_size": 1,
                    "strategy": "mixed",
                    "seed": 7,
                    "inference_batch_size": 4,
                    "compute_clustering_mmd": False,
                },
            },
            "graph_selection": {"rules": [{"type": "threshold", "value": 0.5}]},
        }
    )

    assert ("score", "validation_topology") in HOOK_EVENTS
    assert ("validation_topology", 3) in SCORE_PAIR_COUNTS
    validation_topology_manifest = (
        tmp_path / "logs" / "tccig" / "score" / "validation_topology" / "validation_topology.json"
    )
    manifest = json.loads(validation_topology_manifest.read_text(encoding="utf-8"))
    assert manifest["pair_count"] == 3
    assert manifest["validation_topology_pairs"] == 3
    assert manifest["validation_topology_node_sizes"] == [3]


def test_tccig_orchestrator_runs_v3_1_pairwise_scorer_with_fake_refiner(
    tmp_path: Path,
) -> None:
    processed_dir = tmp_path / "data" / "PRING" / "human" / "BFS"
    _write_tiny_pring_fixture(processed_dir)
    model_config_path, checkpoint_path, cache_dir = _write_tiny_v3_1_pairwise_assets(tmp_path)
    HOOK_EVENTS.clear()

    result = run_tccig_pipeline(
        {
            "run": {"run_id": "v3_1_pairwise", "log_root": str(tmp_path / "logs")},
            "data": {"processed_dir": str(processed_dir)},
            "device": {"device": "cpu", "backend": "ddp", "mixed_precision": False},
            "pairwise_scorer": {
                "target": "tccig.train:score_pairs_with_v3_1",
                "model_config_path": str(model_config_path),
                "checkpoint_path": str(checkpoint_path),
                "embedding_cache_dir": str(cache_dir),
                "batch_size": 2,
                "max_sequence_length": 8,
            },
            "refiner": {
                "train_target": f"{__name__}:fake_train_refiner",
                "predict_target": f"{__name__}:fake_predict_refined",
            },
            "graph_selection": {
                "rules": [
                    {"type": "threshold", "value": 0.5},
                    {"type": "top_m", "m": 1},
                ]
            },
        }
    )

    assert ("train_refiner", 2) in HOOK_EVENTS
    assert ("predict_refined", "topology_test") in HOOK_EVENTS
    assert result.manifest["pair_counts"]["topology_test"] == 2
    assert result.selected_rule["type"] in {"threshold", "top_m"}

    scoring_manifest = tmp_path / "logs" / "tccig" / "score" / "v3_1_pairwise" / "train.json"
    assert scoring_manifest.exists()
    manifest = json.loads(scoring_manifest.read_text(encoding="utf-8"))
    assert manifest["pair_count"] == 2


def test_tccig_runtime_wires_deepspeed_backend_without_launching_it(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "PRING" / "human" / "BFS"
    _write_tiny_pring_fixture(processed_dir)
    accelerator_calls: list[dict[str, object]] = []

    class FakeAccelerator:
        device = "cpu"

    def fake_build_accelerator(**kwargs: object) -> FakeAccelerator:
        accelerator_calls.append(dict(kwargs))
        return FakeAccelerator()

    run_tccig_pipeline(
        {
            "run": {"run_id": "runtime_case", "log_root": str(tmp_path / "logs")},
            "data": {"processed_dir": str(processed_dir)},
            "device": {
                "device": "cuda",
                "backend": "deepspeed",
                "mixed_precision": True,
                "find_unused_parameters": True,
            },
            "pairwise_scorer": {"target": f"{__name__}:fake_score_pairs"},
            "refiner": {
                "train_target": f"{__name__}:fake_train_refiner",
                "predict_target": f"{__name__}:fake_predict_refined",
            },
            "graph_selection": {"rules": [{"type": "threshold", "value": 0.5}]},
        },
        build_accelerator_fn=fake_build_accelerator,
    )

    assert accelerator_calls == [
        {
            "requested_device": "cuda",
            "backend": "deepspeed",
            "ddp_enabled": True,
            "use_mixed_precision": True,
            "find_unused_parameters": True,
        }
    ]


def test_tccig_orchestrator_runs_s2gae_refiner_on_tiny_fixture(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "PRING" / "human" / "BFS"
    _write_tiny_pring_fixture(processed_dir)
    _, _, cache_dir = _write_tiny_v3_1_pairwise_assets(tmp_path)
    HOOK_EVENTS.clear()
    accelerator_events: list[tuple[str, int]] = []

    class FakeAccelerator:
        device = "cpu"

        def prepare(self, *args: object) -> tuple[object, ...]:
            accelerator_events.append(("prepare", len(args)))
            return args

        def backward(self, loss: torch.Tensor) -> None:
            accelerator_events.append(("backward", 1))
            loss.backward()

        def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
            accelerator_events.append(("unwrap_model", 1))
            return model

    def fake_build_accelerator(**kwargs: object) -> FakeAccelerator:
        del kwargs
        return FakeAccelerator()

    result = run_tccig_pipeline(
        {
            "run": {"run_id": "s2gae_tiny", "log_root": str(tmp_path / "logs")},
            "data": {"processed_dir": str(processed_dir)},
            "device": {"device": "cpu", "backend": "ddp", "mixed_precision": False},
            "pairwise_scorer": {"target": f"{__name__}:fake_score_pairs"},
            "refiner": {
                "train_target": "tccig.s2gae:train_refiner",
                "predict_target": "tccig.s2gae:predict_refined",
                "encoder": "graphconv",
                "input_dim": 8,
                "hidden_dim": 8,
                "num_layers": 1,
                "decoder_hidden_dim": 8,
                "decoder_layers": 1,
                "dropout": 0.0,
                "epochs": 2,
                "batch_size": 2,
                "loss": {
                    "type": "bce_with_logits",
                    "pos_weight": 1.5,
                    "label_smoothing": 0.1,
                },
                "optimizer": {
                    "type": "adamw",
                    "lr": 0.01,
                    "weight_decay": 0.0,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "eps": 1.0e-8,
                },
                "scheduler": {"type": "none"},
                "optimization": {"gradient_clip_norm": 1.0},
                "residual_weight": 0.001,
                "monitor_metric": "val_topology_loss",
                "topology_validation": {
                    "enabled": True,
                    "node_sizes": [3],
                    "samples_per_size": 1,
                    "strategy": "mixed",
                    "seed": 7,
                    "inference_batch_size": 2,
                    "compute_clustering_mmd": False,
                    "losses": {
                        "alpha": 0.5,
                        "beta": 1.0,
                        "gamma": 0.3,
                        "delta": 0.3,
                    },
                },
                "embedding_cache_dir": str(cache_dir),
                "embedding_index_path": str(cache_dir / "index.json"),
                "max_sequence_length": 8,
                "checkpoint_path": str(tmp_path / "models" / "s2gae" / "best_model.pt"),
            },
            "graph_selection": {
                "rules": [
                    {"type": "threshold", "value": 0.5},
                    {"type": "top_m", "m": 1},
                ]
            },
        },
        build_accelerator_fn=fake_build_accelerator,
    )

    assert ("prepare", 2) in accelerator_events
    assert ("backward", 1) in accelerator_events
    assert result.selected_rule["type"] in {"threshold", "top_m"}
    assert result.manifest["pair_counts"]["topology_test"] == 2
    assert (tmp_path / "models" / "s2gae" / "best_model.pt").exists()
    training_summary_path = (
        tmp_path / "logs" / "tccig" / "refiner" / "s2gae_tiny" / "training_summary.json"
    )
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    assert training_summary["monitor_metric"] == "val_topology_loss"
    assert training_summary["epochs_trained"] == 2
    assert "best_monitor_value" in training_summary
    assert training_summary["selected_rule"]["validation_metrics"]["epoch"] in {1, 2}
    assert "val_topology_loss" in training_summary["selected_rule"]["validation_metrics"]
    first_epoch = training_summary["history"][0]
    assert "train_loss" in first_epoch
    assert "train_bce_loss" in first_epoch
    assert "train_residual_anchor_loss" in first_epoch
    assert "train_weighted_residual_anchor_loss" in first_epoch
    assert "train_gradient_norm" in first_epoch
    assert first_epoch["learning_rate"] == 0.01
    assert "val_auprc" in first_epoch
    assert "val_topology_loss" in first_epoch
    assert "internal_val_graph_sim" in first_epoch
    assert training_summary["optimizer"] == {
        "type": "adamw",
        "lr": 0.01,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
    }
    assert training_summary["scheduler"] == {"type": "none"}
    assert training_summary["optimization"] == {"gradient_clip_norm": 1.0}
    assert training_summary["current_learning_rate"] == 0.01

    checkpoint = torch.load(tmp_path / "models" / "s2gae" / "best_model.pt")
    assert checkpoint["config"]["loss"] == {
        "type": "bce_with_logits",
        "pos_weight": 1.5,
        "label_smoothing": 0.1,
    }
    assert checkpoint["config"]["optimizer"] == {
        "type": "adamw",
        "lr": 0.01,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1.0e-8,
    }
    assert checkpoint["config"]["scheduler"] == {"type": "none"}
    assert checkpoint["config"]["optimization"] == {"gradient_clip_norm": 1.0}
    assert checkpoint["monitor_metric"] == "val_topology_loss"
    assert checkpoint["selected_rule"]["validation_metrics"]["epoch"] in {1, 2}
    assert "learning_rate" not in checkpoint["config"]
