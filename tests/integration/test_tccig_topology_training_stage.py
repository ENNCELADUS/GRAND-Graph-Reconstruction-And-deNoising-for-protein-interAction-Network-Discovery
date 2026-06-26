"""Integration test: TCCIG Run 02 topology-conditioned training produces deletions."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
import torch
import yaml
from src.pipeline.stages.train import build_model
from tccig.train import run_tccig_pipeline

_TRAIN_NODES = ["A", "B", "C", "D", "E", "F"]
_TEST_NODES = ["T1", "T2", "T3", "T4", "T5"]
_ALL_PROTEINS = [*_TRAIN_NODES, *_TEST_NODES]


def _write_pairs(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a}\t{protein_b}\t{label}\n")


def _write_tiny_pring_fixture(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    with (processed_dir / "human_BFS_split.pkl").open("wb") as handle:
        pickle.dump({"train": set(_TRAIN_NODES), "test": set(_TEST_NODES)}, handle)

    # Sparse train graph: a few positive edges among many candidate pairs so the
    # density loss has room to push the dense candidate set down.
    _write_pairs(
        processed_dir / "human_train_ppi_ratio5_exclusive.txt",
        [
            ("A", "B", 1),
            ("B", "C", 1),
            ("A", "C", 0),
            ("C", "D", 0),
            ("D", "E", 1),
            ("E", "F", 0),
            ("A", "D", 0),
            ("B", "E", 0),
        ],
    )
    _write_pairs(
        processed_dir / "human_val_ppi_ratio5_exclusive.txt",
        [
            ("A", "B", 1),
            ("A", "C", 0),
            ("B", "D", 0),
            ("C", "D", 1),
            ("D", "E", 1),
            ("E", "F", 0),
        ],
    )
    _write_pairs(
        processed_dir / "human_test_ppi.txt",
        [("T1", "T2", 1), ("T1", "T3", 0), ("T2", "T3", 0)],
    )
    # all_test_ppi candidate universe: dense set of pairs so several become raw
    # edges; only T1-T2 is a true positive.
    _write_pairs(
        processed_dir / "all_test_ppi.txt",
        [
            ("T1", "T2", 1),
            ("T1", "T3", 0),
            ("T2", "T3", 0),
            ("T1", "T4", 0),
            ("T2", "T4", 0),
            ("T3", "T4", 0),
            ("T1", "T5", 0),
            ("T4", "T5", 0),
        ],
    )

    graph = nx.Graph()
    graph.add_nodes_from(_TEST_NODES)
    graph.add_edge("T1", "T2")
    with (processed_dir / "human_test_graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)
    with (processed_dir / "test_sampled_nodes.pkl").open("wb") as handle:
        pickle.dump({3: [["T1", "T2", "T3"], ["T1", "T4", "T5"]]}, handle)


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
            "pair_readout": {"mode": "pair_context_gated", "order_aggregation": "abba_max"},
            "interaction": {"mode": "none"},
        }
    }


def _write_tiny_v3_1_pairwise_assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_config_path = tmp_path / "v3_1_abba_no_cross.yaml"
    model_config_path.write_text(yaml.safe_dump(_tiny_v3_1_config()), encoding="utf-8")

    checkpoint_path = tmp_path / "models" / "best_model.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.save(build_model(_tiny_v3_1_config()).state_dict(), checkpoint_path)

    cache_dir = tmp_path / "cache"
    index: dict[str, str] = {}
    for offset, protein_id in enumerate(_ALL_PROTEINS, start=1):
        relative_path = f"embeddings/{protein_id}.pt"
        output_path = cache_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.full((3, 8), float(offset), dtype=torch.float32), output_path)
        index[protein_id] = relative_path
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return model_config_path, checkpoint_path, cache_dir


def _topology_training_config(tmp_path: Path, run_id: str) -> dict[str, object]:
    processed_dir = tmp_path / "data" / "PRING" / "human" / "BFS"
    _write_tiny_pring_fixture(processed_dir)
    model_config_path, checkpoint_path, embedding_cache_dir = _write_tiny_v3_1_pairwise_assets(
        tmp_path
    )
    return {
        "run": {
            "run_id": run_id,
            "log_root": str(tmp_path / "logs"),
            "cache_root": str(tmp_path / "tccig_cache"),
        },
        "data": {"processed_dir": str(processed_dir)},
        "device": {"device": "cpu", "backend": "ddp", "mixed_precision": False},
        "pairwise_scorer": {
            "model_config_path": str(model_config_path),
            "checkpoint_path": str(checkpoint_path),
            "embedding_cache_dir": str(embedding_cache_dir),
            "batch_size": 4,
            "max_sequence_length": 8,
            "score_cache": {"enabled": True},
        },
        "refiner": {
            "encoder": "graphconv",
            "input_dim": 8,
            "hidden_dim": 4,
            "num_layers": 1,
            "decoder_hidden_dim": 4,
            "decoder_layers": 1,
            "dropout": 0.0,
            "residual_scale": 4.0,
            "epochs": 8,
            "batch_size": 8,
            "edge_sampling": {
                "hard_fraction": 0.7,
                "easy_anchor_fraction": 0.3,
                "seed": 0,
                "reshuffle_easy_each_epoch": True,
            },
            "loss": {"type": "bce_with_logits", "pos_weight": 1.0, "label_smoothing": 0.0},
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
            "residual_anchor": {"form": "asymmetric_relu", "weight": 1.0e-6},
            "topology_training": {
                "enabled": True,
                "node_sizes": [3, 4],
                "samples_per_size": 2,
                "strategy": "mixed",
                "seed": 0,
                "coverage_augmentation": True,
                "topology_weight": 5.0,
                "weights": {"alpha": 0.0, "beta": 10.0, "gamma": 0.0, "delta": 0.0},
                "schedule": {"warmup_epochs": 0, "ramp_epochs": 1, "schedule": "linear"},
            },
            "monitor_metric": "val_topology_loss",
            "topology_validation": {
                "enabled": True,
                "node_sizes": [3, 4],
                "samples_per_size": 2,
                "strategy": "mixed",
                "seed": 0,
                "inference_batch_size": 8,
                "compute_clustering_mmd": False,
                "losses": {"alpha": 1.0, "beta": 8.0, "gamma": 0.5, "delta": 0.1},
            },
            "embedding_cache_dir": str(embedding_cache_dir),
            "embedding_index_path": str(embedding_cache_dir / "index.json"),
            "max_sequence_length": 8,
            "checkpoint_path": str(tmp_path / "models" / "tccig" / run_id / "best_model.pt"),
        },
        "graph_selection": {
            # Input graph admits low-confidence pairs; the refined graph uses the
            # natural 0.5 boundary, so downward residual pressure prunes edges.
            "pairwise_input_threshold": {"mode": "fixed", "value": 0.3},
            "refined_output_rule": {"type": "threshold", "value": 0.5},
            "rules": [{"type": "threshold", "value": 0.5}],
        },
    }


def test_topology_training_run_deletes_edges_and_logs_topology_loss(tmp_path: Path) -> None:
    config = _topology_training_config(tmp_path, "02_topology_training")

    run_tccig_pipeline(config)

    run_log_dir = tmp_path / "logs" / "tccig" / "02_topology_training"
    topology_metrics = json.loads(
        (run_log_dir / "topology_test" / "topology_metrics.json").read_text(encoding="utf-8")
    )
    diagnostics = topology_metrics["deletion_diagnostics"]
    assert diagnostics["edges_deleted"] > 0.0
    assert diagnostics["net_edge_delta"] <= 0.0

    summary = json.loads((run_log_dir / "training_summary.json").read_text(encoding="utf-8"))
    history = summary["history"]
    assert any("train_topology_loss" in entry for entry in history)
    # warmup_epochs=0, ramp_epochs=1 -> epoch 1 scale 0.0, final epoch ramps to 1.0.
    assert history[-1]["train_topology_scale"] == 1.0
