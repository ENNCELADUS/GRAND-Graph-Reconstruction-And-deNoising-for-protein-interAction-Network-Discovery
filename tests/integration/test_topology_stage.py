"""Integration tests for the topology evaluation stage."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import cast

import pytest
import src.run.stage_topology_evaluate as topology_stage
import torch
from src.run.stage_topology_evaluate import run_topology_evaluation_stage
from src.run.stage_train import build_model
from src.utils.config import ConfigDict
from src.utils.data_io import build_dataloaders
from src.utils.distributed import DistributedContext
from torch.utils.data import DataLoader


def _write_split(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a}\t{protein_b}\t{label}\n")


def _write_embedding_cache(
    cache_dir: Path,
    embeddings: dict[str, torch.Tensor],
    *,
    input_dim: int,
    max_sequence_length: int,
) -> None:
    index: dict[str, str] = {}
    for protein_id, tensor in embeddings.items():
        relative_path = f"embeddings/{protein_id}.pt"
        output_path = cache_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, output_path)
        index[protein_id] = relative_path
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "esm3",
                "model_name": "esm3_sm_open_v1",
                "input_dim": input_dim,
                "max_sequence_length": max_sequence_length,
                "format": "torch_pt_per_protein",
            }
        ),
        encoding="utf-8",
    )


def _build_topology_config(tmp_path: Path) -> ConfigDict:
    benchmark_root = tmp_path / "benchmark"
    processed_dir = benchmark_root / "human" / "BFS"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "human_train_ppi.txt"
    valid_path = processed_dir / "human_val_ppi.txt"
    test_path = processed_dir / "human_test_ppi.txt"
    all_test_path = processed_dir / "all_test_ppi.txt"

    _write_split(train_path, [("P1", "P2", 1), ("P1", "P3", 0), ("P2", "P3", 1)])
    _write_split(valid_path, [("P1", "P2", 1), ("P1", "P3", 0)])
    _write_split(test_path, [("P1", "P2", 1), ("P2", "P3", 1)])
    _write_split(all_test_path, [("P1", "P2", 1), ("P1", "P3", 0), ("P2", "P3", 1)])

    gt_graph = cast(object, __import__("networkx").Graph())
    gt_graph.add_edges_from([("P1", "P2"), ("P2", "P3")])
    with (processed_dir / "human_test_graph.pkl").open("wb") as handle:
        pickle.dump(gt_graph, handle)
    with (processed_dir / "test_sampled_nodes.pkl").open("wb") as handle:
        pickle.dump({3: [["P1", "P2", "P3"]]}, handle)

    cache_dir = tmp_path / "cache"
    _write_embedding_cache(
        cache_dir=cache_dir,
        embeddings={
            "P1": torch.ones((2, 4), dtype=torch.float32),
            "P2": torch.full((2, 4), 2.0, dtype=torch.float32),
            "P3": torch.full((2, 4), 3.0, dtype=torch.float32),
        },
        input_dim=4,
        max_sequence_length=8,
    )

    baselines_path = tmp_path / "baselines.json"
    baselines_path.write_text(
        json.dumps(
            {
                "source": "integration-test",
                "rows": [
                    {
                        "category": "Seq. Sim.",
                        "model": "SPRINT",
                        "metrics": {
                            "BFS": {
                                "graph_sim": 0.2,
                                "relative_density": 1.1,
                                "deg_dist_mmd": 5.0,
                                "cc_mmd": 4.0,
                                "laplacian_eigen_mmd": 3.0,
                            },
                            "DFS": {
                                "graph_sim": 0.1,
                                "relative_density": 1.2,
                                "deg_dist_mmd": 6.0,
                                "cc_mmd": 5.0,
                                "laplacian_eigen_mmd": 4.0,
                            },
                            "RANDOM_WALK": {
                                "graph_sim": 0.3,
                                "relative_density": 1.0,
                                "deg_dist_mmd": 4.0,
                                "cc_mmd": 3.0,
                                "laplacian_eigen_mmd": 2.0,
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    return {
        "run_config": {
            "stages": ["topology_evaluate"],
            "seed": 7,
            "train_run_id": "unused_train",
            "adapt_run_id": None,
            "eval_run_id": None,
            "topology_eval_run_id": "topology_case",
            "load_checkpoint_path": str(tmp_path / "input_checkpoint.pth"),
            "save_best_only": True,
        },
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {
            "benchmark": {
                "name": "PRING",
                "root_dir": str(benchmark_root),
                "processed_dir": str(processed_dir),
                "species": "human",
                "split_strategy": "BFS",
            },
            "embeddings": {
                "source": "esm3",
                "cache_dir": str(cache_dir),
                "model_name": "esm3_sm_open_v1",
                "device": "cpu",
            },
            "max_sequence_length": 8,
            "dataloader": {
                "train_dataset": str(train_path),
                "valid_dataset": str(valid_path),
                "test_dataset": str(test_path),
                "num_workers": 0,
                "pin_memory": False,
                "drop_last": False,
                "sampling": {"strategy": "none"},
            },
        },
        "model_config": {
            "model": "v3",
            "input_dim": 4,
            "d_model": 4,
            "encoder_layers": 1,
            "cross_attn_layers": 1,
            "n_heads": 2,
            "mlp_head": {
                "hidden_dims": [4],
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
        },
        "training_config": {
            "batch_size": 2,
            "epochs": 1,
            "monitor_metric": "auprc",
            "logging": {"validation_metrics": ["auprc", "auroc"]},
            "optimizer": {"type": "adamw", "lr": 1e-3},
            "scheduler": {"type": "none"},
            "loss": {"type": "bce_with_logits", "pos_weight": 1.0, "label_smoothing": 0.0},
            "strategy": {"type": "none"},
            "domain_adaptation": {"enabled": False, "method": "none", "target_split": "test"},
        },
        "evaluate": {
            "metrics": ["auprc", "auroc"],
            "decision_threshold": {"mode": "best_f1_on_valid"},
        },
        "topology_evaluate": {
            "decision_threshold": {"mode": "best_f1_on_valid"},
            "save_pair_predictions": True,
            "report_baselines": str(baselines_path),
            "inference_batch_size": 2,
        },
    }


def test_run_topology_evaluation_stage_writes_expected_artifacts(tmp_path: Path) -> None:
    config = _build_topology_config(tmp_path)
    previous_cwd = Path.cwd()
    try:
        Path(config["run_config"]["load_checkpoint_path"]).parent.mkdir(parents=True, exist_ok=True)  # type: ignore[index]
        model = build_model(config)
        torch.save(model.state_dict(), Path(str(config["run_config"]["load_checkpoint_path"])))  # type: ignore[index]
        dataloaders = build_dataloaders(config=config)
        distributed_context = DistributedContext(ddp_enabled=False, is_distributed=False)
        __import__("os").chdir(tmp_path)
        run_topology_evaluation_stage(
            config=config,
            model=model,
            device=torch.device("cpu"),
            dataloaders=cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            run_id="topology_case",
            checkpoint_path=Path(str(config["run_config"]["load_checkpoint_path"])),  # type: ignore[index]
            distributed_context=distributed_context,
        )
    finally:
        __import__("os").chdir(previous_cwd)

    log_dir = tmp_path / "logs" / "v3" / "topology_evaluate" / "topology_case"
    assert (log_dir / "all_test_ppi_pred.txt").exists()
    assert (log_dir / "topology_metrics.json").exists()
    assert (log_dir / "topology_metrics.csv").exists()
    assert (log_dir / "graph_eval_results.pkl").exists()

    log_text = (log_dir / "log.log").read_text(encoding="utf-8")
    assert "Decision Threshold" in log_text
    assert "best_f1_on_valid" in log_text


def test_run_topology_evaluation_stage_non_main_rank_computes_topology_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_topology_config(tmp_path)
    checkpoint_path = Path(str(config["run_config"]["load_checkpoint_path"]))  # type: ignore[index]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(config)
    torch.save(model.state_dict(), checkpoint_path)
    dataloaders = build_dataloaders(config=config)

    monkeypatch.setattr(topology_stage.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(topology_stage.dist, "broadcast", lambda tensor, src: None)
    monkeypatch.setattr(
        topology_stage,
        "_build_topology_loader",
        lambda **_: (
            cast(DataLoader[dict[str, object]], dataloaders["test"]),
            [("P1", "P2"), ("P1", "P3"), ("P2", "P3")],
            [1],
            3,
        ),
    )
    monkeypatch.setattr(topology_stage, "_predict_topology_labels", lambda **_: [0])
    monkeypatch.setattr(
        topology_stage,
        "_gather_ordered_predictions",
        lambda **_: [1, 0, 1],
    )

    previous_cwd = Path.cwd()
    try:
        __import__("os").chdir(tmp_path)
        summary = run_topology_evaluation_stage(
            config=config,
            model=model,
            device=torch.device("cpu"),
            dataloaders=cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            run_id="topology_non_main",
            checkpoint_path=checkpoint_path,
            distributed_context=DistributedContext(
                ddp_enabled=True,
                is_distributed=True,
                rank=1,
                local_rank=1,
                world_size=2,
            ),
        )
    finally:
        __import__("os").chdir(previous_cwd)

    assert "graph_sim" in summary
