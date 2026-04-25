"""Unit tests for optimization trial runner helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from src.optimize.trial_runner import (
    build_trial_run_id,
    objective_metric_to_csv_header,
    pick_objective_value,
    read_objective_history,
    run_best_full_pipeline,
)
from src.utils.config import ConfigDict


def _write_training_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = ["Epoch", "Epoch Time", "Train Loss", "Val Loss", "Val auprc", "Learning Rate"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_objective_metric_to_csv_header_accepts_val_prefix() -> None:
    assert objective_metric_to_csv_header("val_auprc") == "Val auprc"
    assert objective_metric_to_csv_header("auprc") == "Val auprc"


def test_read_objective_history_reads_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "training_step.csv"
    _write_training_csv(
        csv_path,
        [
            {
                "Epoch": 1,
                "Epoch Time": 1.1,
                "Train Loss": 0.8,
                "Val Loss": 0.7,
                "Val auprc": 0.45,
                "Learning Rate": 1.0e-4,
            },
            {
                "Epoch": 2,
                "Epoch Time": 1.2,
                "Train Loss": 0.6,
                "Val Loss": 0.5,
                "Val auprc": 0.61,
                "Learning Rate": 8.0e-5,
            },
        ],
    )

    history, column = read_objective_history(csv_path=csv_path, objective_metric="val_auprc")

    assert column == "Val auprc"
    assert history == pytest.approx([0.45, 0.61])


def test_pick_objective_value_handles_direction() -> None:
    history = [0.3, 0.5, 0.4]
    assert pick_objective_value(history=history, direction="maximize") == pytest.approx(0.5)
    assert pick_objective_value(history=history, direction="minimize") == pytest.approx(0.3)


@pytest.mark.parametrize("direction", ["up", "", "MAX"])
def test_pick_objective_value_rejects_invalid_direction(direction: str) -> None:
    with pytest.raises(ValueError, match="optimization.direction"):
        pick_objective_value(history=[0.1], direction=direction)


def test_build_trial_run_id_uses_timestamp_prefix_and_trial_suffix() -> None:
    assert (
        build_trial_run_id(run_id_prefix="20260320_110811", trial_number=7)
        == "20260320_110811_t0007"
    )


def test_run_best_full_pipeline_preserves_ddp_when_requested() -> None:
    observed: dict[str, object] = {}
    base_config: ConfigDict = {
        "run_config": {"stages": ["train"], "seed": 7, "save_best_only": True},
        "device_config": {"device": "cuda", "ddp_enabled": True, "use_mixed_precision": True},
        "model_config": {
            "model": "v3",
            "input_dim": 1536,
            "d_model": 512,
            "encoder_layers": 3,
            "cross_attn_layers": 3,
            "n_heads": 8,
        },
        "training_config": {"scheduler": {"max_lr": 1.0e-4}},
    }

    def fake_run_pipeline(config: ConfigDict) -> None:
        device_cfg = config["device_config"]
        assert isinstance(device_cfg, dict)
        observed["ddp_enabled"] = device_cfg["ddp_enabled"]
        run_cfg = config["run_config"]
        assert isinstance(run_cfg, dict)
        observed["stages"] = list(run_cfg["stages"])

    run_best_full_pipeline(
        base_config=base_config,
        search_space=[],
        best_values={},
        run_id_prefix="20260320_110811",
        run_pipeline_fn=fake_run_pipeline,
        ddp_per_trial=True,
    )

    assert observed["ddp_enabled"] is True
    assert observed["stages"] == ["train"]


def test_run_best_full_pipeline_preserves_topology_stage_when_requested() -> None:
    observed: dict[str, object] = {}
    base_config: ConfigDict = {
        "run_config": {
            "stages": ["train", "evaluate", "topology_evaluate"],
            "seed": 7,
            "save_best_only": True,
        },
        "device_config": {"device": "cuda", "ddp_enabled": True, "use_mixed_precision": True},
        "model_config": {
            "model": "v3",
            "input_dim": 1536,
            "d_model": 512,
            "encoder_layers": 3,
            "cross_attn_layers": 3,
            "n_heads": 8,
        },
        "training_config": {"scheduler": {"max_lr": 1.0e-4}},
        "topology_evaluate": {
            "decision_threshold": {"mode": "fixed", "value": 0.5},
            "report_baselines": "src/topology/baselines/pring_human_table2.json",
        },
    }

    def fake_run_pipeline(config: ConfigDict) -> None:
        run_cfg = config["run_config"]
        assert isinstance(run_cfg, dict)
        observed["stages"] = list(run_cfg["stages"])
        observed["topology_eval_run_id"] = run_cfg["topology_eval_run_id"]

    run_best_full_pipeline(
        base_config=base_config,
        search_space=[],
        best_values={},
        run_id_prefix="20260320_110811",
        run_pipeline_fn=fake_run_pipeline,
        ddp_per_trial=False,
    )

    assert observed["stages"] == ["train", "evaluate", "topology_evaluate"]
    assert observed["topology_eval_run_id"] == "20260320_110811_best_topology"
