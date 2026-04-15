"""Optional HPC-style DDP smoke tests."""

from __future__ import annotations

import os
import subprocess
from csv import DictReader
from pathlib import Path

import pytest
import torch
from src.pipeline.stages.evaluate import EVAL_CSV_COLUMNS
from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "tests" / "e2e" / "artifacts" / "v3_hpc.yaml"


@pytest.mark.e2e
def test_hpc_config_artifact_is_valid() -> None:
    """Validate the HPC test config can be parsed and has required DDP flags."""
    config = load_config(CONFIG_PATH)
    run_cfg = config["run_config"]
    training_cfg = config["training_config"]
    evaluate_cfg = config["evaluate"]
    device_cfg = config["device_config"]
    assert isinstance(run_cfg, dict)
    assert isinstance(training_cfg, dict)
    assert isinstance(evaluate_cfg, dict)
    assert isinstance(device_cfg, dict)
    assert run_cfg["stages"] == ["train", "evaluate"]
    assert device_cfg["ddp_enabled"] is True

    logging_cfg = training_cfg["logging"]
    assert isinstance(logging_cfg, dict)
    validation_metrics = logging_cfg["validation_metrics"]
    assert isinstance(validation_metrics, list)
    expected_training_headers = [
        "Epoch",
        "Epoch Time",
        "Train Loss",
        "Val Loss",
        *[f"Val {metric}" for metric in validation_metrics],
        "Learning Rate",
    ]
    assert expected_training_headers == [
        "Epoch",
        "Epoch Time",
        "Train Loss",
        "Val Loss",
        "Val auprc",
        "Val auroc",
        "Learning Rate",
    ]

    eval_metrics = evaluate_cfg["metrics"]
    assert isinstance(eval_metrics, list)
    assert EVAL_CSV_COLUMNS == [
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
    assert set(EVAL_CSV_COLUMNS[1:]).issubset(set(eval_metrics))


@pytest.mark.e2e
@pytest.mark.gpu
@pytest.mark.slow
def test_hpc_ddp_train_evaluate_smoke() -> None:
    """Run optional train+evaluate torchrun smoke test for cluster environments."""
    if os.environ.get("GRAND_RUN_HPC_E2E", "0") != "1":
        pytest.skip("Set GRAND_RUN_HPC_E2E=1 to run HPC smoke test.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("HPC smoke test requires at least 2 CUDA devices.")

    world_size = 2
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={world_size}",
        "-m",
        "src.pipeline",
        "--config",
        str(CONFIG_PATH),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)

    expected_checkpoint = REPO_ROOT / "models" / "v3" / "train" / "hpc_e2e_train" / "best_model.pth"
    assert expected_checkpoint.exists()

    train_csv = REPO_ROOT / "logs" / "v3" / "train" / "hpc_e2e_train" / "training_step.csv"
    evaluate_csv = REPO_ROOT / "logs" / "v3" / "evaluate" / "hpc_e2e_eval" / "evaluate.csv"
    assert train_csv.exists()
    assert evaluate_csv.exists()

    config = load_config(CONFIG_PATH)
    training_cfg = config["training_config"]
    assert isinstance(training_cfg, dict)
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

    with train_csv.open("r", encoding="utf-8", newline="") as handle:
        train_header = DictReader(handle).fieldnames
    with evaluate_csv.open("r", encoding="utf-8", newline="") as handle:
        eval_header = DictReader(handle).fieldnames

    assert train_header == expected_train_header
    assert eval_header == EVAL_CSV_COLUMNS
