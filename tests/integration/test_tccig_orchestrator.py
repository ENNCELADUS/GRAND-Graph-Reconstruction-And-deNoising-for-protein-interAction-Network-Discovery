"""Integration tests for the concrete TCCIG Accelerate orchestrator."""

from __future__ import annotations

import csv
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pytest
import torch
import yaml
from src.pipeline.stages.train import build_model
from tccig import s2gae
from tccig.train import _build_runtime, run_tccig_pipeline


def _write_pairs(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a}\t{protein_b}\t{label}\n")


def _write_tiny_pring_fixture(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    with (processed_dir / "human_BFS_split.pkl").open("wb") as handle:
        pickle.dump({"train": {"A", "B", "C", "D"}, "test": {"T1", "T2", "T3"}}, handle)

    _write_pairs(
        processed_dir / "human_train_ppi_ratio5_exclusive.txt",
        [("A", "B", 1), ("A", "C", 0), ("B", "C", 1), ("C", "D", 0)],
    )
    _write_pairs(
        processed_dir / "human_val_ppi_ratio5_exclusive.txt",
        [("A", "B", 1), ("A", "C", 0), ("B", "D", 0), ("C", "D", 1)],
    )
    _write_pairs(
        processed_dir / "human_test_ppi.txt",
        [("T1", "T2", 1), ("T1", "T3", 0), ("T2", "T3", 0)],
    )
    _write_pairs(
        processed_dir / "all_test_ppi.txt",
        [("T1", "T2", 1), ("T1", "T3", 0), ("T2", "T3", 0)],
    )

    graph = nx.Graph()
    graph.add_nodes_from(["T1", "T2", "T3"])
    graph.add_edge("T1", "T2")
    with (processed_dir / "human_test_graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)
    with (processed_dir / "test_sampled_nodes.pkl").open("wb") as handle:
        pickle.dump({2: [["T1", "T2"], ["T1", "T3"]]}, handle)


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
    torch.manual_seed(0)
    torch.save(build_model(_tiny_v3_1_config()).state_dict(), checkpoint_path)

    cache_dir = tmp_path / "cache"
    index: dict[str, str] = {}
    for offset, protein_id in enumerate(["A", "B", "C", "D", "T1", "T2", "T3"], start=1):
        relative_path = f"embeddings/{protein_id}.pt"
        output_path = cache_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.full((3, 8), float(offset), dtype=torch.float32), output_path)
        index[protein_id] = relative_path
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return model_config_path, checkpoint_path, cache_dir


def _tiny_config(tmp_path: Path, run_id: str) -> dict[str, object]:
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
            "batch_size": 2,
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
            "epochs": 1,
            "batch_size": 2,
            "edge_sampling": {
                "hard_fraction": 0.7,
                "easy_anchor_fraction": 0.3,
                "seed": 0,
                "reshuffle_easy_each_epoch": True,
            },
            "loss": {
                "type": "bce_with_logits",
                "pos_weight": 1.0,
                "label_smoothing": 0.0,
            },
            "optimizer": {
                "type": "adamw",
                "lr": 0.001,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1.0e-8,
            },
            "scheduler": {"type": "none"},
            "optimization": {"gradient_clip_norm": 1.0},
            "residual_weight": 0.0,
            "monitor_metric": "val_auprc",
            "topology_validation": {"enabled": False},
            "embedding_cache_dir": str(embedding_cache_dir),
            "embedding_index_path": str(embedding_cache_dir / "index.json"),
            "max_sequence_length": 8,
            "checkpoint_path": str(tmp_path / "models" / "tccig" / run_id / "best_model.pt"),
        },
        "graph_selection": {
            "pairwise_input_threshold": {"mode": "fixed", "value": 0.5},
            "refined_output_rule": {"type": "threshold", "value": 0.5},
            "rules": [{"type": "threshold", "value": 0.5}],
        },
    }


def _read_topology_prediction_rows(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]


def _read_pairwise_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_tccig_runtime_uses_bf16_mixed_precision_from_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path, "bf16_runtime")
    config["device"] = {"device": "cpu", "backend": "ddp", "mixed_precision": "bf16"}
    accelerator_calls: list[dict[str, object]] = []

    class FakeAccelerator:
        device = torch.device("cpu")
        num_processes = 1
        use_distributed = False
        process_index = 0
        local_process_index = 0
        is_main_process = True

    def fake_accelerator(**kwargs: object) -> FakeAccelerator:
        accelerator_calls.append(kwargs)
        return FakeAccelerator()

    monkeypatch.setattr("tccig.train.Accelerator", fake_accelerator)

    runtime = _build_runtime(config=config, build_accelerator_fn=None)

    assert accelerator_calls[0]["mixed_precision"] == "bf16"
    assert runtime.mixed_precision == "bf16"


def test_tccig_orchestrator_runs_concrete_pipeline_and_writes_artifacts(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, "tiny")

    result = run_tccig_pipeline(config)

    assert result.pairwise_input_threshold["value"] == 0.5
    assert result.refined_output_rule == {"type": "threshold", "value": 0.5}
    assert set(result.pairwise_metrics) >= {"auprc", "auroc", "f1", "threshold"}
    assert set(result.topology_metrics) == {
        "graph_sim",
        "relative_density",
        "deg_dist_mmd",
        "cc_mmd",
        "laplacian_eigen_mmd",
    }

    run_log_dir = tmp_path / "logs" / "tccig" / "tiny"
    assert (run_log_dir / "manifest.json").exists()
    assert (run_log_dir / "pairwise_test" / "pairwise_metrics.json").exists()
    assert (run_log_dir / "topology_test" / "all_test_ppi_pred.txt").exists()
    assert (run_log_dir / "topology_test" / "topology_metrics.json").exists()
    assert (run_log_dir / "topology_test" / "topology_metrics.csv").exists()
    assert (tmp_path / "tccig_cache" / "score_cache" / "tiny" / "manifests" / "train.json").exists()


def test_tccig_orchestrator_runs_validation_topology_with_pring_train_test_split(
    tmp_path: Path,
) -> None:
    config = _tiny_config(tmp_path, "validation_topology")
    refiner_config = config["refiner"]
    assert isinstance(refiner_config, dict)
    refiner_config["monitor_metric"] = "val_topology_loss"
    refiner_config["topology_validation"] = {
        "enabled": True,
        "node_sizes": [2],
        "samples_per_size": 1,
        "strategy": "mixed",
        "seed": 0,
        "inference_batch_size": 4,
        "compute_clustering_mmd": False,
        "losses": {"alpha": 1.0, "beta": 1.0, "gamma": 0.0, "delta": 0.0},
    }

    result = run_tccig_pipeline(config)

    assert set(result.topology_metrics) == {
        "graph_sim",
        "relative_density",
        "deg_dist_mmd",
        "cc_mmd",
        "laplacian_eigen_mmd",
    }
    assert (
        tmp_path
        / "tccig_cache"
        / "score_cache"
        / "validation_topology"
        / "manifests"
        / "validation_topology.json"
    ).exists()


def test_tccig_orchestrator_rejects_removed_hook_config(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, "legacy")
    config["pairwise_scorer"] = {"target": "legacy.module:score"}

    with pytest.raises(ValueError, match="pairwise_scorer.target"):
        run_tccig_pipeline(config)


def test_tccig_script_uses_cpu_fallback_when_nvidia_smi_is_absent(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv_args_path = tmp_path / "uv_args.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" > "$GRAND_UV_ARGS_OUT"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    (tmp_path / ".venv").mkdir()

    subprocess.run(
        ["/bin/bash", str(repo_root / "scripts" / "tccig.sh")],
        cwd=repo_root,
        env={
            "GRAND_REPO_ROOT": str(tmp_path),
            "GRAND_UV_ARGS_OUT": str(uv_args_path),
            "HOME": str(tmp_path),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        },
        check=True,
        timeout=30,
    )

    assert uv_args_path.read_text(encoding="utf-8").splitlines() == [
        "run",
        "--locked",
        "--no-sync",
        "--offline",
        "python",
        "-m",
        "tccig.train",
        "--config",
        "configs/tccig/01.yaml",
    ]


def test_tccig_script_launches_accelerate_with_detected_gpu_count(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun_args_path = tmp_path / "srun_args.txt"
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        "#!/bin/bash\n"
        "printf 'GPU 0\\nGPU 1\\n'\n",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" > "$GRAND_SRUN_ARGS_OUT"\n',
        encoding="utf-8",
    )
    fake_srun.chmod(0o755)
    (tmp_path / ".venv").mkdir()

    subprocess.run(
        ["/bin/bash", str(repo_root / "scripts" / "tccig.sh")],
        cwd=repo_root,
        env={
            "GRAND_REPO_ROOT": str(tmp_path),
            "GRAND_SRUN_ARGS_OUT": str(srun_args_path),
            "HOME": str(tmp_path),
            "PATH": f"{fake_bin}:/usr/bin:/bin",
        },
        check=True,
        timeout=30,
    )

    assert srun_args_path.read_text(encoding="utf-8").splitlines() == [
        "uv",
        "run",
        "--locked",
        "--no-sync",
        "--offline",
        "accelerate",
        "launch",
        "--num_processes",
        "2",
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16",
        "--dynamo_backend",
        "no",
        "tccig/train.py",
        "--config",
        "configs/tccig/01.yaml",
    ]


def test_tccig_accelerate_cpu_smoke_preserves_topology_artifact_order(tmp_path: Path) -> None:
    single_config = _tiny_config(tmp_path, "single")
    run_tccig_pipeline(single_config)

    config = _tiny_config(tmp_path, "ddp")
    config_path = tmp_path / "tccig.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = f"{repo_root}:{os.environ.get('PYTHONPATH', '')}"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--cpu",
            "--num_processes",
            "2",
            "tccig/train.py",
            "--config",
            str(config_path),
        ],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=True,
        timeout=180,
    )

    single_rows = _read_topology_prediction_rows(
        tmp_path / "logs" / "tccig" / "single" / "topology_test" / "all_test_ppi_pred.txt"
    )
    ddp_rows = _read_topology_prediction_rows(
        tmp_path / "logs" / "tccig" / "ddp" / "topology_test" / "all_test_ppi_pred.txt"
    )
    single_pairwise_rows = _read_pairwise_prediction_rows(
        tmp_path / "logs" / "tccig" / "single" / "pairwise_test" / "human_test_ppi_pred.csv"
    )
    ddp_pairwise_rows = _read_pairwise_prediction_rows(
        tmp_path / "logs" / "tccig" / "ddp" / "pairwise_test" / "human_test_ppi_pred.csv"
    )

    assert len(ddp_pairwise_rows) == len(single_pairwise_rows)
    # Compare ordered identity, label, and the numeric refined probability so a
    # distributed gather that shards or reorders values is caught, not just a
    # schema/column-name match.
    assert [(row["protein_a"], row["protein_b"], row["label"]) for row in ddp_pairwise_rows] == [
        (row["protein_a"], row["protein_b"], row["label"]) for row in single_pairwise_rows
    ]
    assert [float(row["refined_probability"]) for row in ddp_pairwise_rows] == pytest.approx(
        [float(row["refined_probability"]) for row in single_pairwise_rows]
    )
    assert len(ddp_rows) == len(single_rows)
    # Topology artifact carries (protein_a, protein_b, hard-graph prediction);
    # compare the full row including the 0/1 prediction value.
    assert ddp_rows == single_rows
    assert all(row[2] in {"0", "1"} for row in ddp_rows)


def test_best_validation_auprc_matches_selected_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _tiny_config(tmp_path, "auprc_couple")
    refiner_config = config["refiner"]
    assert isinstance(refiner_config, dict)
    refiner_config["epochs"] = 3
    refiner_config["monitor_metric"] = "val_topology_loss"
    refiner_config["topology_validation"] = {
        "enabled": True,
        "node_sizes": [2],
        "samples_per_size": 1,
        "strategy": "mixed",
        "seed": 0,
        "inference_batch_size": 4,
        "compute_clustering_mmd": False,
        "losses": {"alpha": 1.0, "beta": 1.0, "gamma": 0.0, "delta": 0.0},
    }

    # The tiny fixture yields identical metrics every epoch, so the global-max
    # bug is indistinguishable from the correct behaviour. Force the per-epoch
    # validation AUPRC to strictly increase while the monitor (val_topology_loss)
    # stays flat: epoch 1 is selected (first strict-min wins), yet the global
    # max AUPRC belongs to epoch 3. A correct fix reports the epoch-1 value.
    increasing_auprc = iter([0.1, 0.5, 0.9])

    def _fake_validation_auprc(**_kwargs: object) -> float:
        return next(increasing_auprc)

    monkeypatch.setattr(s2gae, "_validation_auprc", _fake_validation_auprc)

    run_tccig_pipeline(config)

    checkpoint_path = tmp_path / "models" / "tccig" / "auprc_couple" / "best_model.pt"
    payload = torch.load(checkpoint_path, weights_only=False)
    summary_path = tmp_path / "logs" / "tccig" / "auprc_couple" / "training_summary.json"
    history = json.loads(summary_path.read_text(encoding="utf-8"))["history"]

    best_monitor = payload["best_monitor_value"]
    selected = min(history, key=lambda row: abs(row["monitor_value"] - best_monitor))
    assert selected["val_auprc"] == pytest.approx(0.1)
    assert payload["best_validation_auprc"] == pytest.approx(selected["val_auprc"])
