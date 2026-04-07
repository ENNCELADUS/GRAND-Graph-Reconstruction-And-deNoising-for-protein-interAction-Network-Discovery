"""Optional local CPU E2E smoke tests."""

from __future__ import annotations

import json
import os
import pickle
from csv import DictReader
from pathlib import Path

import pytest
import src.run as run_module
import torch
from src.utils.config import ConfigDict, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "tests" / "e2e" / "artifacts" / "v3_local_cpu.yaml"


def _to_absolute_path(path_value: str) -> str:
    """Resolve config paths relative to repo root."""
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(REPO_ROOT / path)


def _resolve_data_paths(config: ConfigDict) -> ConfigDict:
    """Return config with dataset/cache paths rewritten as absolute paths."""
    data_cfg = config["data_config"]
    assert isinstance(data_cfg, dict)
    benchmark_cfg = data_cfg["benchmark"]
    dataloader_cfg = data_cfg["dataloader"]
    embeddings_cfg = data_cfg["embeddings"]
    assert isinstance(benchmark_cfg, dict)
    assert isinstance(dataloader_cfg, dict)
    assert isinstance(embeddings_cfg, dict)

    benchmark_cfg["root_dir"] = _to_absolute_path(str(benchmark_cfg["root_dir"]))
    benchmark_cfg["processed_dir"] = _to_absolute_path(str(benchmark_cfg["processed_dir"]))
    embeddings_cfg["cache_dir"] = _to_absolute_path(str(embeddings_cfg["cache_dir"]))
    dataloader_cfg["train_dataset"] = _to_absolute_path(str(dataloader_cfg["train_dataset"]))
    dataloader_cfg["valid_dataset"] = _to_absolute_path(str(dataloader_cfg["valid_dataset"]))
    dataloader_cfg["test_dataset"] = _to_absolute_path(str(dataloader_cfg["test_dataset"]))
    topology_cfg = config.get("topology_evaluate", {})
    if isinstance(topology_cfg, dict) and "report_baselines" in topology_cfg:
        topology_cfg["report_baselines"] = _to_absolute_path(str(topology_cfg["report_baselines"]))
    return config


def _write_split(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a}\t{protein_b}\t{label}\n")


def _write_embedding_cache(
    cache_dir: Path,
    *,
    input_dim: int,
    max_sequence_length: int,
) -> None:
    embeddings = {
        "P1": torch.ones((2, input_dim), dtype=torch.float32),
        "P2": torch.full((2, input_dim), 2.0, dtype=torch.float32),
        "P3": torch.full((2, input_dim), 3.0, dtype=torch.float32),
    }
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


def _tiny_topology_config(tmp_path: Path) -> ConfigDict:
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

    import networkx as nx

    gt_graph = nx.Graph()
    gt_graph.add_edges_from([("P1", "P2"), ("P2", "P3")])
    with (processed_dir / "human_test_graph.pkl").open("wb") as handle:
        pickle.dump(gt_graph, handle)
    with (processed_dir / "test_sampled_nodes.pkl").open("wb") as handle:
        pickle.dump({3: [["P1", "P2", "P3"]]}, handle)

    cache_dir = tmp_path / "cache"
    _write_embedding_cache(cache_dir=cache_dir, input_dim=4, max_sequence_length=8)

    return {
        "run_config": {
            "stages": ["train", "evaluate", "topology_evaluate"],
            "seed": 7,
            "train_run_id": "tiny_e2e_train",
            "adapt_run_id": None,
            "eval_run_id": "tiny_e2e_eval",
            "topology_eval_run_id": "tiny_e2e_topology",
            "load_checkpoint_path": None,
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
                "species": "human",
                "split_strategy": "BFS",
                "processed_dir": str(processed_dir),
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
            "epochs": 1,
            "batch_size": 2,
            "early_stopping_patience": 1,
            "monitor_metric": "auprc",
            "logging": {"validation_metrics": ["auprc", "auroc"], "heartbeat_every_n_steps": 2},
            "optimizer": {"type": "adamw", "lr": 0.0001},
            "scheduler": {"type": "none"},
            "loss": {"type": "bce_with_logits", "pos_weight": 1.0, "label_smoothing": 0.0},
            "strategy": {"type": "none"},
            "domain_adaptation": {"enabled": False, "method": "none", "target_split": "test"},
        },
        "evaluate": {
            "decision_threshold": {"mode": "best_f1_on_valid"},
            "metrics": ["auprc", "auroc", "accuracy", "f1", "precision", "recall"],
        },
        "topology_evaluate": {
            "inference_batch_size": 2,
            "decision_threshold": {"mode": "best_f1_on_valid"},
            "save_pair_predictions": True,
            "report_baselines": str(
                REPO_ROOT / "src" / "topology" / "baselines" / "pring_human_table2.json"
            ),
        },
    }


@pytest.mark.e2e
def test_local_cpu_config_artifact_is_valid() -> None:
    """Validate local CPU E2E config contract."""
    config = load_config(CONFIG_PATH)
    run_cfg = config["run_config"]
    device_cfg = config["device_config"]
    training_cfg = config["training_config"]
    assert isinstance(run_cfg, dict)
    assert isinstance(device_cfg, dict)
    assert isinstance(training_cfg, dict)
    assert run_cfg["stages"] == ["train", "evaluate"]
    assert device_cfg["device"] == "cpu"
    assert device_cfg["ddp_enabled"] is False
    assert device_cfg["use_mixed_precision"] is False

    logging_cfg = training_cfg["logging"]
    assert isinstance(logging_cfg, dict)
    validation_metrics = logging_cfg["validation_metrics"]
    assert isinstance(validation_metrics, list)
    expected_train_header = [
        "Epoch",
        "Epoch Time",
        "Train Loss",
        "Val Loss",
        *[f"Val {metric}" for metric in validation_metrics],
        "Learning Rate",
    ]
    assert expected_train_header == [
        "Epoch",
        "Epoch Time",
        "Train Loss",
        "Val Loss",
        "Val auprc",
        "Val auroc",
        "Learning Rate",
    ]
    assert run_module.EVAL_CSV_COLUMNS == [
        "split",
        "auroc",
        "auprc",
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
        "mcc",
    ]


@pytest.mark.e2e
@pytest.mark.slow
def test_local_cpu_train_evaluate_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run optional local full-pipeline smoke test on CPU."""
    if os.environ.get("GRAND_RUN_LOCAL_E2E", "0") != "1":
        pytest.skip("Set GRAND_RUN_LOCAL_E2E=1 to run local CPU E2E smoke test.")

    monkeypatch.chdir(tmp_path)
    config = _resolve_data_paths(load_config(CONFIG_PATH))
    run_module.execute_pipeline(config=config)

    train_log_dir = tmp_path / "logs" / "v3" / "train" / "local_cpu_e2e_train"
    eval_log_dir = tmp_path / "logs" / "v3" / "evaluate" / "local_cpu_e2e_eval"
    train_model = tmp_path / "models" / "v3" / "train" / "local_cpu_e2e_train" / "best_model.pth"
    assert train_model.exists()
    assert (train_log_dir / "log.log").exists()
    assert (eval_log_dir / "log.log").exists()

    train_csv = train_log_dir / "training_step.csv"
    evaluate_csv = eval_log_dir / "evaluate.csv"
    assert train_csv.exists()
    assert evaluate_csv.exists()

    with train_csv.open("r", encoding="utf-8", newline="") as handle:
        train_header = DictReader(handle).fieldnames
    with evaluate_csv.open("r", encoding="utf-8", newline="") as handle:
        eval_header = DictReader(handle).fieldnames
    assert train_header == [
        "Epoch",
        "Epoch Time",
        "Train Loss",
        "Val Loss",
        "Val auprc",
        "Val auroc",
        "Learning Rate",
    ]
    assert eval_header == run_module.EVAL_CSV_COLUMNS


@pytest.mark.e2e
def test_local_cpu_train_evaluate_topology_tiny_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a tiny end-to-end topology pipeline without the full Human all-test surface."""
    monkeypatch.chdir(tmp_path)
    run_module.execute_pipeline(config=_tiny_topology_config(tmp_path))

    train_log_dir = tmp_path / "logs" / "v3" / "train" / "tiny_e2e_train"
    eval_log_dir = tmp_path / "logs" / "v3" / "evaluate" / "tiny_e2e_eval"
    topology_log_dir = tmp_path / "logs" / "v3" / "topology_evaluate" / "tiny_e2e_topology"
    assert (train_log_dir / "training_step.csv").exists()
    assert (eval_log_dir / "evaluate.csv").exists()
    assert (topology_log_dir / "all_test_ppi_pred.txt").exists()
    assert (topology_log_dir / "topology_metrics.json").exists()
    assert (topology_log_dir / "topology_metrics.csv").exists()
