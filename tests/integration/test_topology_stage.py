"""Integration tests for the topology evaluation stage."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import src.pipeline.stages.topology_evaluate as topology_stage
import torch
from src.pipeline.runtime import DistributedContext
from src.pipeline.stages.topology_evaluate import run_topology_evaluation_stage
from src.pipeline.stages.train import build_model
from src.utils.config import ConfigDict
from src.utils.data_io import build_dataloaders
from tests.runtime_helpers import build_stage_runtime
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
        checkpoint_path = Path(str(config["run_config"]["load_checkpoint_path"]))  # type: ignore[index]
        __import__("os").chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_evaluate": "topology_case"},
        )
        run_topology_evaluation_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
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


def _fake_sharded_topology_result(node_sizes: tuple[int, ...]) -> dict[str, object]:
    details: dict[str, dict[int, list[float] | float]] = {
        "graph_sim": {},
        "relative_density": {},
        "deg_dist_mmd": {},
        "cc_mmd": {},
        "laplacian_eigen_mmd": {},
    }
    per_node_size: dict[int, dict[str, float | int]] = {}
    for node_size in node_sizes:
        graph_sim_values = [node_size / 100.0]
        relative_density_values = [1.0 + (node_size / 1000.0)]
        deg_dist_mmd = node_size / 10.0
        cc_mmd = node_size / 20.0
        laplacian_eigen_mmd = node_size / 40.0
        details["graph_sim"][node_size] = graph_sim_values
        details["relative_density"][node_size] = relative_density_values
        details["deg_dist_mmd"][node_size] = deg_dist_mmd
        details["cc_mmd"][node_size] = cc_mmd
        details["laplacian_eigen_mmd"][node_size] = laplacian_eigen_mmd
        per_node_size[node_size] = {
            "graph_count": 1,
            "graph_sim": graph_sim_values[0],
            "relative_density": relative_density_values[0],
            "deg_dist_mmd": deg_dist_mmd,
            "cc_mmd": cc_mmd,
            "laplacian_eigen_mmd": laplacian_eigen_mmd,
        }
    summary = {
        "graph_sim": float(np.mean([values[0] for values in details["graph_sim"].values()])),
        "relative_density": float(
            np.mean([values[0] for values in details["relative_density"].values()])
        ),
        "deg_dist_mmd": float(np.mean(list(details["deg_dist_mmd"].values()))),
        "cc_mmd": float(np.mean(list(details["cc_mmd"].values()))),
        "laplacian_eigen_mmd": float(np.mean(list(details["laplacian_eigen_mmd"].values()))),
    }
    return {
        "details": details,
        "summary": summary,
        "per_node_size": per_node_size,
    }


def test_run_topology_evaluation_stage_shards_graph_metrics_under_ddp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_topology_config(tmp_path)
    processed_dir = (
        Path(str(config["data_config"]["benchmark"]["processed_dir"]))  # type: ignore[index]
    )
    with (processed_dir / "test_sampled_nodes.pkl").open("wb") as handle:
        pickle.dump(
            {
                20: [["P1"]],
                40: [["P2"]],
                60: [["P3"]],
                80: [["P4"]],
            },
            handle,
        )

    checkpoint_path = Path(str(config["run_config"]["load_checkpoint_path"]))  # type: ignore[index]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(config)
    torch.save(model.state_dict(), checkpoint_path)
    dataloaders = build_dataloaders(config=config)

    monkeypatch.setattr(
        topology_stage,
        "_build_topology_loader",
        lambda **_: (
            cast(DataLoader[dict[str, object]], dataloaders["test"]),
            [("P1", "P2"), ("P1", "P3"), ("P2", "P3")],
            3,
        ),
    )
    monkeypatch.setattr(topology_stage, "_predict_topology_labels", lambda **_: [1, 0, 1])
    monkeypatch.setattr(topology_stage, "_resolve_decision_threshold", lambda **_: (0.5, "fixed"))

    observed_local_node_sizes: list[tuple[int, ...]] = []

    def _record_local_graph_eval(
        *,
        pred_graph: object,
        gt_graph: object,
        test_graph_nodes: object,
    ) -> dict[str, object]:
        del pred_graph, gt_graph
        assert isinstance(test_graph_nodes, dict)
        node_sizes = tuple(sorted(int(node_size) for node_size in test_graph_nodes))
        observed_local_node_sizes.append(node_sizes)
        return _fake_sharded_topology_result(node_sizes)

    def _fake_all_gather_object(
        object_list: list[object | None],
        local_result: object,
    ) -> None:
        object_list[0] = _fake_sharded_topology_result((40, 80))
        object_list[1] = local_result

    monkeypatch.setattr(topology_stage, "evaluate_predicted_graph", _record_local_graph_eval)
    monkeypatch.setattr(topology_stage.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(topology_stage.dist, "all_gather_object", _fake_all_gather_object)

    previous_cwd = Path.cwd()
    try:
        __import__("os").chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_evaluate": "topology_sharded"},
            distributed=DistributedContext(
                ddp_enabled=True,
                is_distributed=True,
                rank=1,
                local_rank=1,
                world_size=2,
            ),
        )
        summary = run_topology_evaluation_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        __import__("os").chdir(previous_cwd)

    expected_summary = cast(
        dict[str, float],
        _fake_sharded_topology_result((20, 40, 60, 80))["summary"],
    )
    assert observed_local_node_sizes == [(20, 60)]
    assert summary == pytest.approx(expected_summary)


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

    monkeypatch.setattr(
        topology_stage,
        "_build_topology_loader",
        lambda **_: (
            cast(DataLoader[dict[str, object]], dataloaders["test"]),
            [("P1", "P2"), ("P1", "P3"), ("P2", "P3")],
            3,
        ),
    )
    monkeypatch.setattr(topology_stage, "_predict_topology_labels", lambda **_: [1, 0, 1])

    previous_cwd = Path.cwd()
    try:
        __import__("os").chdir(tmp_path)
        distributed_context = DistributedContext(
            ddp_enabled=True,
            is_distributed=True,
            rank=1,
            local_rank=1,
            world_size=2,
        )
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_evaluate": "topology_non_main"},
            distributed=distributed_context,
        )
        summary = run_topology_evaluation_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        __import__("os").chdir(previous_cwd)

    assert "graph_sim" in summary
