"""Unit tests for process-level logging and DDP config flags."""

from __future__ import annotations

import logging

import pytest
import src.pipeline.bootstrap as pipeline_bootstrap
from src.pipeline.runtime import ddp_find_unused_parameters
from src.pipeline.stages.evaluate import _metrics_from_config
from src.pipeline.stages.train import _training_validation_metrics
from src.utils.config import ConfigDict


def _base_config() -> ConfigDict:
    return {
        "run_config": {"stages": ["train", "evaluate"]},
        "device_config": {"device": "cpu", "ddp_enabled": False},
        "training_config": {},
    }


def test_configure_root_logging_main_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "0")
    observed: dict[str, object] = {}
    captured_warning_flags: list[bool] = []

    def fake_capture_warnings(enabled: bool) -> None:
        captured_warning_flags.append(enabled)

    def fake_basic_config(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(logging, "captureWarnings", fake_capture_warnings)
    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    pipeline_bootstrap.configure_root_logging(logging, pipeline_bootstrap.rank_from_env())

    assert captured_warning_flags == [True]
    assert observed["level"] == logging.INFO
    assert observed["force"] is True


def test_configure_root_logging_non_main_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "3")
    observed: dict[str, object] = {}

    def fake_basic_config(**kwargs: object) -> None:
        observed.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)
    monkeypatch.setattr(logging, "captureWarnings", lambda _: None)

    pipeline_bootstrap.configure_root_logging(logging, pipeline_bootstrap.rank_from_env())

    assert observed["level"] == logging.CRITICAL
    assert observed["force"] is True


def test_ddp_find_unused_parameters_uses_strategy_default() -> None:
    config = _base_config()
    assert ddp_find_unused_parameters(config) is False

    training_cfg = config["training_config"]
    assert isinstance(training_cfg, dict)
    training_cfg["strategy"] = {"type": "staged_unfreeze"}

    assert ddp_find_unused_parameters(config) is True


def test_ddp_find_unused_parameters_honors_device_override() -> None:
    config = _base_config()
    training_cfg = config["training_config"]
    assert isinstance(training_cfg, dict)
    training_cfg["strategy"] = {"type": "staged_unfreeze"}

    device_cfg = config["device_config"]
    assert isinstance(device_cfg, dict)
    device_cfg["find_unused_parameters"] = False

    assert ddp_find_unused_parameters(config) is False


def test_ddp_find_unused_parameters_enables_shot_adaptation_by_default() -> None:
    config = _base_config()
    training_cfg = config["training_config"]
    assert isinstance(training_cfg, dict)
    training_cfg["domain_adaptation"] = {
        "enabled": True,
        "method": "shot",
        "target_split": "test",
    }

    assert ddp_find_unused_parameters(config) is True


def test_metrics_from_config_preserves_case_and_order() -> None:
    eval_cfg: ConfigDict = {"metrics": ["AUROC", "AUPRC", "accuracy"]}

    metrics = _metrics_from_config(eval_cfg)

    assert metrics == ["AUROC", "AUPRC", "accuracy"]


def test_training_validation_metrics_rejects_empty_list() -> None:
    with pytest.raises(
        ValueError,
        match="training_config.logging.validation_metrics must not be empty",
    ):
        _training_validation_metrics({"logging": {"validation_metrics": []}})


def test_metrics_from_config_rejects_non_sequence() -> None:
    with pytest.raises(ValueError, match="evaluate.metrics must be a sequence"):
        _metrics_from_config({"metrics": 123})


def test_training_validation_metrics_lowercases_entries() -> None:
    training_cfg: ConfigDict = {"logging": {"validation_metrics": ["AUPRC", "AuRoC", "accuracy"]}}

    metrics = _training_validation_metrics(training_cfg)

    assert metrics == ["auprc", "auroc", "accuracy"]
