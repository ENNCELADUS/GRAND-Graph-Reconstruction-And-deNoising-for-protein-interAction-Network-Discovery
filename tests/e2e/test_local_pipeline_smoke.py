"""Optional local CPU E2E smoke tests."""

from __future__ import annotations

import os
from csv import DictReader
from pathlib import Path

import pytest
import src.run as run_module
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
    return config


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
    if os.environ.get("RELIC_RUN_LOCAL_E2E", "0") != "1":
        pytest.skip("Set RELIC_RUN_LOCAL_E2E=1 to run local CPU E2E smoke test.")

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
