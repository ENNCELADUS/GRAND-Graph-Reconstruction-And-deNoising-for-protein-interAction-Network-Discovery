"""Integration tests for optimization workflow with fake Optuna backend."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
import src.optimize.run as optimize_run
from src.optimize.backends.optuna_backend import run_optuna_optimization
from src.optimize.search_space import extend_with_nas_lite, parse_search_space
from src.utils.config import ConfigDict


@dataclass
class _FakeState:
    name: str


@dataclass
class _FakeFrozenTrial:
    number: int
    state: _FakeState
    value: float | None
    params: dict[str, object]
    user_attrs: dict[str, object]


class _FakeTrialPruned(Exception):
    """Fake Optuna trial-pruned exception."""


class _FakeTrial:
    def __init__(self, *, number: int, prune_step: int | None) -> None:
        self.number = number
        self.params: dict[str, object] = {}
        self.user_attrs: dict[str, object] = {}
        self._prune_step = prune_step
        self._latest_step = 0

    def suggest_float(self, name: str, low: float, high: float, **kwargs: object) -> float:
        del kwargs
        value = low + (high - low) * 0.5
        self.params[name] = value
        return value

    def suggest_int(self, name: str, low: int, high: int, **kwargs: object) -> int:
        del kwargs
        span = max(high - low + 1, 1)
        value = low + (self.number % span)
        self.params[name] = value
        return value

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        value = choices[self.number % len(choices)]
        self.params[name] = value
        return value

    def report(self, value: float, step: int) -> None:
        del value
        self._latest_step = step

    def should_prune(self) -> bool:
        return self._prune_step is not None and self._latest_step >= self._prune_step

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value


class _FakeStudy:
    def __init__(self, *, direction: str, prune_steps: dict[int, int]) -> None:
        self._direction = direction
        self._prune_steps = prune_steps
        self.trials: list[_FakeFrozenTrial] = []
        self.best_value: float = float("-inf") if direction == "maximize" else float("inf")
        self.best_params: dict[str, object] = {}

    def optimize(self, objective: Callable[[object], float], n_trials: int, timeout: int) -> None:
        del timeout
        for number in range(n_trials):
            prune_step = self._prune_steps.get(number)
            trial = _FakeTrial(number=number, prune_step=prune_step)
            try:
                value = objective(trial)
                state = _FakeState("COMPLETE")
            except _FakeTrialPruned:
                value = None
                state = _FakeState("PRUNED")

            frozen_trial = _FakeFrozenTrial(
                number=number,
                state=state,
                value=value,
                params=dict(trial.params),
                user_attrs=dict(trial.user_attrs),
            )
            self.trials.append(frozen_trial)
            if value is None:
                continue
            if self._is_better(value):
                self.best_value = value
                self.best_params = dict(trial.params)

    def _is_better(self, value: float) -> bool:
        if self._direction == "maximize":
            return value > self.best_value
        return value < self.best_value


class _FakeOptunaModule:
    class samplers:
        class TPESampler:
            def __init__(self, seed: int) -> None:
                del seed

        class RandomSampler:
            def __init__(self, seed: int) -> None:
                del seed

    class pruners:
        class MedianPruner:
            def __init__(self, n_startup_trials: int, n_warmup_steps: int) -> None:
                del n_startup_trials, n_warmup_steps

        class NopPruner:
            def __init__(self) -> None:
                pass

    TrialPruned = _FakeTrialPruned

    def __init__(self, prune_steps: dict[int, int] | None = None) -> None:
        self._prune_steps = {} if prune_steps is None else dict(prune_steps)

    def create_study(
        self,
        *,
        study_name: str,
        direction: str,
        sampler: object,
        pruner: object,
        storage: str | None,
        load_if_exists: bool,
    ) -> _FakeStudy:
        del study_name, sampler, pruner, storage, load_if_exists
        return _FakeStudy(direction=direction, prune_steps=self._prune_steps)


def _base_config() -> ConfigDict:
    return {
        "run_config": {
            "stages": ["train", "evaluate"],
            "seed": 7,
            "save_best_only": True,
        },
        "device_config": {"device": "cpu", "ddp_enabled": False, "use_mixed_precision": False},
        "data_config": {},
        "model_config": {
            "model": "v5",
            "input_dim": 1536,
            "d_model": 999,
            "encoder_layers": 2,
            "cross_attn_layers": 2,
            "n_heads": 8,
        },
        "training_config": {
            "batch_size": 16,
            "optimizer": {"lr": 1.0e-4, "weight_decay": 0.01},
            "scheduler": {"type": "none"},
            "logging": {"validation_metrics": ["auprc"]},
        },
        "evaluate": {"metrics": ["auprc"]},
        "optimization": {
            "enabled": True,
            "backend": "optuna",
            "study_name": "unit_opt",
            "objective_metric": "val_auprc",
            "direction": "maximize",
            "budget": {"n_trials": 3, "timeout_minutes": 1},
            "execution": {
                "trial_stages": ["train"],
                "ddp_per_trial": False,
                "catch_oom_as_pruned": True,
            },
            "storage": {"type": "none"},
            "sampler": {"name": "TPESampler", "seed": 7},
            "pruner": {"name": "MedianPruner", "n_startup_trials": 0, "n_warmup_steps": 0},
            "search_space": [
                {
                    "name": "optimizer_lr",
                    "path": "training_config.optimizer.lr",
                    "type": "float",
                    "low": 1.0e-5,
                    "high": 1.0e-3,
                },
                {
                    "name": "d_model",
                    "path": "model_config.d_model",
                    "type": "categorical",
                    "choices": [128, 192, 256],
                },
            ],
        },
    }


def _fake_execute_pipeline(config: ConfigDict) -> None:
    run_cfg = config["run_config"]
    assert isinstance(run_cfg, dict)
    run_id = str(run_cfg.get("train_run_id", "missing"))

    model_cfg = config["model_config"]
    assert isinstance(model_cfg, dict)
    model_name = str(model_cfg.get("model", "v5"))
    d_model = float(model_cfg.get("d_model", 192))

    training_cfg = config["training_config"]
    assert isinstance(training_cfg, dict)
    optimizer_cfg = training_cfg["optimizer"]
    assert isinstance(optimizer_cfg, dict)
    lr = float(optimizer_cfg.get("lr", 1.0e-4))

    score = d_model / 1000.0 + (1.0e-3 - lr)
    history = [max(score - 0.05, 0.0), max(score, 0.0)]

    log_dir = Path("logs") / model_name / "train" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / "training_step.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Epoch",
                "Epoch Time",
                "Train Loss",
                "Val Loss",
                "Val auprc",
                "Learning Rate",
            ],
        )
        writer.writeheader()
        for index, value in enumerate(history, start=1):
            writer.writerow(
                {
                    "Epoch": index,
                    "Epoch Time": 1.0,
                    "Train Loss": 0.5,
                    "Val Loss": 0.4,
                    "Val auprc": value,
                    "Learning Rate": lr,
                }
            )

    model_dir = Path("models") / model_name / "train" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best_model.pth").write_text("fake", encoding="utf-8")


def test_run_optimization_writes_trials_and_best_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _base_config()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.optimize.backends.optuna_backend._import_optuna",
        lambda: _FakeOptunaModule(),
    )
    monkeypatch.setattr(optimize_run, "PIPELINE_EXECUTE_FN", _fake_execute_pipeline)

    optimize_run.run_optimization(
        config=config,
        backend_override=None,
        skip_final_full_pipeline=True,
    )

    output_dir = tmp_path / "artifacts" / "hpo" / "unit_opt"
    assert (output_dir / "trials.csv").exists()
    assert (output_dir / "best_params.yaml").exists()

    with (output_dir / "trials.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3


def test_optuna_backend_marks_pruned_trials(tmp_path: Path) -> None:
    config = _base_config()
    optimization_cfg = cast(ConfigDict, config["optimization"])
    search_space = parse_search_space(optimization_cfg["search_space"])

    result = run_optuna_optimization(
        base_config=config,
        optimization_cfg=optimization_cfg,
        search_space=search_space,
        run_pipeline_fn=_fake_execute_pipeline,
        optuna_module=cast(ModuleType, _FakeOptunaModule(prune_steps={1: 1})),
    )

    states = [record.state for record in result.trial_records]
    assert "PRUNED" in states
    assert "COMPLETE" in states


def test_nas_lite_parameters_are_applied_in_trials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _base_config()
    optimization_cfg = cast(ConfigDict, config["optimization"])
    optimization_cfg["search_space"] = [
        {
            "name": "optimizer_lr",
            "path": "training_config.optimizer.lr",
            "type": "float",
            "low": 1.0e-5,
            "high": 1.0e-4,
        }
    ]
    budget_cfg = cast(ConfigDict, optimization_cfg["budget"])
    budget_cfg["n_trials"] = 2
    config["nas_lite"] = {"enabled": True, "method": "arch_params_hpo", "max_candidates": 2}

    observed_d_models: list[int] = []

    def _recording_pipeline(cfg: ConfigDict) -> None:
        model_cfg = cfg["model_config"]
        assert isinstance(model_cfg, dict)
        observed_d_models.append(int(model_cfg["d_model"]))
        _fake_execute_pipeline(cfg)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.optimize.backends.optuna_backend._import_optuna",
        lambda: _FakeOptunaModule(),
    )
    monkeypatch.setattr(optimize_run, "PIPELINE_EXECUTE_FN", _recording_pipeline)

    optimize_run.run_optimization(
        config=config,
        backend_override=None,
        skip_final_full_pipeline=True,
    )

    assert observed_d_models
    assert all(value in {128, 192, 256} for value in observed_d_models)

    parsed_space = parse_search_space(optimization_cfg["search_space"])
    extended_space = extend_with_nas_lite(root_config=config, base_search_space=parsed_space)
    assert any(parameter.name == "nas_d_model" for parameter in extended_space)
