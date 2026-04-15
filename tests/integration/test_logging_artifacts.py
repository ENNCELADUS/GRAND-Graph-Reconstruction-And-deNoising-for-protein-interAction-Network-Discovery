"""Integration tests for runtime logging and artifact schema contracts."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import cast

import pytest
import src.run as run_module
import src.run.stage_evaluate as stage_evaluate_module
import torch
from src.utils.config import ConfigDict
from src.utils.distributed import DistributedContext
from torch import nn
from torch.utils.data import DataLoader, Dataset


class _TinyPairDataset(Dataset[dict[str, torch.Tensor]]):
    """Small deterministic dataset for train/valid/test loops."""

    def __init__(self) -> None:
        self._features = torch.tensor(
            [
                [0.2, 0.4, 0.8, 0.1],
                [0.6, 0.1, 0.3, 0.9],
                [0.7, 0.9, 0.2, 0.5],
                [0.1, 0.3, 0.6, 0.7],
            ],
            dtype=torch.float32,
        )
        self._labels = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self._features.size(0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self._features[index], "label": self._labels[index]}


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
    }


class _TinyModel(nn.Module):
    """Minimal model compatible with trainer/evaluator output contract."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
        del label
        return {"logits": self.linear(x)}


def _base_config(stages: list[str] | None = None) -> ConfigDict:
    """Create a minimal valid config for pipeline stage artifact tests."""
    stage_list = ["train", "evaluate"] if stages is None else stages
    return {
        "run_config": {
            "stages": stage_list,
            "seed": 7,
            "train_run_id": "train_case",
            "adapt_run_id": "adapt_case",
            "eval_run_id": "eval_case",
            "load_checkpoint_path": None,
            "save_best_only": True,
        },
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {
            "dataloader": {
                "sampling": {"strategy": "none"},
            },
        },
        "model_config": {"model": "v3"},
        "training_config": {
            "epochs": 1,
            "early_stopping_patience": 2,
            "monitor_metric": "auprc",
            "logging": {
                "validation_metrics": ["auprc", "auroc"],
                "heartbeat_every_n_steps": 2,
            },
            "optimizer": {"type": "adamw", "lr": 1e-2},
            "scheduler": {"type": "none"},
            "loss": {"type": "bce_with_logits", "pos_weight": 1.0, "label_smoothing": 0.0},
            "strategy": {"type": "none"},
            "domain_adaptation": {"enabled": False, "method": "none", "target_split": "test"},
        },
        "evaluate": {"metrics": ["auprc", "auroc"]},
    }


def _fake_dataloaders() -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
    dataset = _TinyPairDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=_collate)
    return {"train": loader, "valid": loader, "test": loader}


def test_execute_pipeline_writes_stage_logs_and_strict_csv_headers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_build_dataloaders(
        config: ConfigDict,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
        del config, distributed, rank, world_size
        return _fake_dataloaders()

    def fake_build_model(config: ConfigDict) -> nn.Module:
        del config
        return _TinyModel()

    class _FakeAccelerator:
        device = torch.device("cpu")
        is_main_process = True
        use_distributed = False
        process_index = 0
        local_process_index = 0
        num_processes = 1
        mixed_precision = "no"

        def prepare(self, *components: object) -> object:
            if len(components) == 1:
                return components[0]
            return components

        def autocast(self) -> AbstractContextManager[None]:
            return nullcontext()

        def backward(self, loss: torch.Tensor) -> None:
            loss.backward()

        def wait_for_everyone(self) -> None:
            return None

        def unwrap_model(self, model: nn.Module) -> nn.Module:
            return model

        def save(self, obj: object, f: object, safe_serialization: bool = False) -> None:
            del safe_serialization
            torch.save(obj, f)

    def fake_build_accelerator(**kwargs: object) -> _FakeAccelerator:
        del kwargs
        return _FakeAccelerator()

    monkeypatch.setattr(run_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(run_module, "build_model", fake_build_model)
    monkeypatch.setattr(run_module, "build_accelerator", fake_build_accelerator, raising=False)

    run_module.execute_pipeline(_base_config(stages=["train", "evaluate"]))

    train_log = tmp_path / "logs" / "v3" / "train" / "train_case" / "log.log"
    eval_log = tmp_path / "logs" / "v3" / "evaluate" / "eval_case" / "log.log"
    assert train_log.exists()
    assert eval_log.exists()

    train_text = train_log.read_text(encoding="utf-8")
    eval_text = eval_log.read_text(encoding="utf-8")
    assert "Stage Start" in train_text
    assert "Epoch Start" in train_text
    assert "Epoch Done" in train_text
    assert "Stage Done" in train_text
    assert "Epoch 1 | Step 1/2" in train_text
    assert "Evaluation Metrics" in eval_text
    assert "CSV Written" in eval_text

    training_csv = tmp_path / "logs" / "v3" / "train" / "train_case" / "training_step.csv"
    evaluate_csv = tmp_path / "logs" / "v3" / "evaluate" / "eval_case" / "evaluate.csv"
    assert training_csv.exists()
    assert evaluate_csv.exists()

    train_header = training_csv.read_text(encoding="utf-8").splitlines()[0]
    eval_header = evaluate_csv.read_text(encoding="utf-8").splitlines()[0]
    assert train_header == "Epoch,Epoch Time,Train Loss,Val Loss,Val auprc,Val auroc,Learning Rate"
    assert eval_header == (
        "split,auroc,auprc,accuracy,sensitivity,specificity,precision,recall,f1,mcc"
    )


def test_non_main_process_does_not_write_stage_artifacts(tmp_path: Path) -> None:
    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        config = _base_config(stages=["train"])
        dataloaders = _fake_dataloaders()
        model = _TinyModel()
        distributed_context = DistributedContext(
            ddp_enabled=True,
            is_distributed=False,
            rank=1,
            local_rank=1,
            world_size=2,
        )
        best_checkpoint = run_module.run_training_stage(
            stage="train",
            config=config,
            model=model,
            device=torch.device("cpu"),
            dataloaders=cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            run_id="rank1_case",
            distributed_context=distributed_context,
        )
    finally:
        os.chdir(previous_cwd)

    log_dir = tmp_path / "logs" / "v3" / "train" / "rank1_case"
    assert log_dir.exists()
    assert not (log_dir / "log.log").exists()
    assert not (log_dir / "training_step.csv").exists()
    assert not best_checkpoint.exists()


def test_evaluation_uses_best_f1_threshold_from_validation_and_logs_it(
    tmp_path: Path,
) -> None:
    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        config = _base_config(stages=["evaluate"])
        run_cfg = config["run_config"]
        assert isinstance(run_cfg, dict)
        run_cfg["load_checkpoint_path"] = "artifacts/input_checkpoint.pth"
        evaluate_cfg = config["evaluate"]
        assert isinstance(evaluate_cfg, dict)
        evaluate_cfg["decision_threshold"] = {"mode": "best_f1_on_valid"}

        dataloaders = _fake_dataloaders()
        model = _TinyModel()
        checkpoint_path = Path("artifacts/input_checkpoint.pth")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

        distributed_context = DistributedContext(ddp_enabled=False, is_distributed=False)
        metrics = run_module.run_evaluation_stage(
            config=config,
            model=model,
            device=torch.device("cpu"),
            dataloaders=cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            run_id="threshold_eval_case",
            checkpoint_path=checkpoint_path,
            distributed_context=distributed_context,
        )
    finally:
        os.chdir(previous_cwd)

    eval_log = tmp_path / "logs" / "v3" / "evaluate" / "threshold_eval_case" / "log.log"
    eval_csv = tmp_path / "logs" / "v3" / "evaluate" / "threshold_eval_case" / "evaluate.csv"
    assert eval_log.exists()
    assert eval_csv.exists()

    log_text = eval_log.read_text(encoding="utf-8")
    assert "Decision Threshold" in log_text
    assert "best_f1_on_valid" in log_text

    csv_lines = eval_csv.read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 2
    assert metrics["recall"] >= 0.5


def test_evaluation_worker_rank_skips_threshold_search_and_test_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _GuardEvaluator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def select_best_f1_threshold(self, **kwargs: object) -> float:
            del kwargs
            raise AssertionError("worker ranks must skip validation threshold search")

        def evaluate(self, **kwargs: object) -> dict[str, float]:
            del kwargs
            raise AssertionError("worker ranks must skip test-set inference")

    barrier_calls: list[int] = []
    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        config = _base_config(stages=["evaluate"])
        evaluate_cfg = config["evaluate"]
        assert isinstance(evaluate_cfg, dict)
        evaluate_cfg["decision_threshold"] = {"mode": "best_f1_on_valid"}

        model = _TinyModel()
        checkpoint_path = tmp_path / "artifacts" / "worker_eval_checkpoint.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

        monkeypatch.setattr(stage_evaluate_module, "Evaluator", _GuardEvaluator)
        monkeypatch.setattr(
            stage_evaluate_module,
            "distributed_barrier",
            lambda context: barrier_calls.append(context.rank),
        )

        metrics = stage_evaluate_module.run_evaluation_stage(
            config=config,
            model=model,
            device=torch.device("cpu"),
            dataloaders=cast(dict[str, DataLoader[dict[str, object]]], _fake_dataloaders()),
            run_id="worker_eval_case",
            checkpoint_path=checkpoint_path,
            distributed_context=DistributedContext(
                ddp_enabled=True,
                is_distributed=True,
                rank=1,
                local_rank=1,
                world_size=4,
                owns_process_group=False,
            ),
        )
    finally:
        os.chdir(previous_cwd)

    eval_dir = tmp_path / "logs" / "v3" / "evaluate" / "worker_eval_case"
    assert metrics == {}
    assert barrier_calls == [1]
    assert eval_dir.exists()
    assert not (eval_dir / "evaluate.csv").exists()
    assert not (eval_dir / "log.log").exists()


def test_run_evaluation_stage_propagates_mixed_precision_setting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_use_amp: list[bool] = []

    class _RecordingEvaluator:
        def __init__(self, *args: object, use_amp: bool = False, **kwargs: object) -> None:
            del args, kwargs
            observed_use_amp.append(use_amp)

        def select_best_f1_threshold(self, **kwargs: object) -> float:
            del kwargs
            return 0.5

        def evaluate(self, **kwargs: object) -> dict[str, float]:
            del kwargs
            return {
                "auroc": 1.0,
                "auprc": 1.0,
                "accuracy": 1.0,
                "sensitivity": 1.0,
                "specificity": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "mcc": 1.0,
                "loss": 0.0,
            }

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        config = _base_config(stages=["evaluate"])
        device_cfg = cast(ConfigDict, config["device_config"])
        device_cfg["use_mixed_precision"] = True
        evaluate_cfg = cast(ConfigDict, config["evaluate"])
        evaluate_cfg["decision_threshold"] = {"mode": "best_f1_on_valid"}

        model = _TinyModel()
        checkpoint_path = tmp_path / "artifacts" / "amp_eval_checkpoint.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

        monkeypatch.setattr(stage_evaluate_module, "Evaluator", _RecordingEvaluator)
        monkeypatch.setattr(
            stage_evaluate_module,
            "_load_checkpoint",
            lambda model, checkpoint_path, device: None,
        )

        stage_evaluate_module.run_evaluation_stage(
            config=config,
            model=model,
            device=torch.device("cuda"),
            dataloaders=cast(dict[str, DataLoader[dict[str, object]]], _fake_dataloaders()),
            run_id="amp_eval_case",
            checkpoint_path=checkpoint_path,
            distributed_context=DistributedContext(ddp_enabled=False, is_distributed=False),
        )
    finally:
        os.chdir(previous_cwd)

    assert observed_use_amp == [True, True]
