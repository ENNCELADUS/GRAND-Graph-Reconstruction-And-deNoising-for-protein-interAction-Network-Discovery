"""Integration tests for stage-based pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import src.run as run_module
import torch
from src.utils.config import ConfigDict
from src.utils.distributed import DistributedContext
from torch import nn
from torch.utils.data import DataLoader, Dataset


class _EmptyDataset(Dataset[dict[str, torch.Tensor]]):
    """Empty dataset used for mocked dataloader wiring."""

    def __len__(self) -> int:
        """Return dataset length."""
        return 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Raise because no sample retrieval is expected."""
        raise IndexError(index)


class _DummyModel(nn.Module):
    """Simple model that satisfies ``nn.Module`` contract for orchestration tests."""

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return fixed output dictionary."""
        del kwargs
        return {"logits": torch.zeros((1, 1), dtype=torch.float32)}


@dataclass
class PipelineCalls:
    """Recorded mocked pipeline calls."""

    training: list[tuple[str, str]] = field(default_factory=list)
    adaptation: list[tuple[Path, str]] = field(default_factory=list)
    evaluation: list[tuple[Path, str]] = field(default_factory=list)
    topology_evaluation: list[tuple[Path, str]] = field(default_factory=list)


@pytest.fixture
def base_config() -> ConfigDict:
    """Build minimal valid config for execute_pipeline orchestration."""
    return {
        "run_config": {
            "stages": ["train", "evaluate"],
            "seed": 7,
            "train_run_id": "train_run",
            "adapt_run_id": "adapt_run",
            "eval_run_id": "eval_run",
            "topology_eval_run_id": "topology_eval_run",
            "load_checkpoint_path": "artifacts/input_checkpoint.pth",
            "save_best_only": True,
        },
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {},
        "model_config": {"model": "v3"},
        "training_config": {
            "domain_adaptation": {
                "enabled": False,
                "method": "none",
                "target_split": "test",
            }
        },
        "evaluate": {"metrics": ["accuracy"]},
        "topology_evaluate": {"save_pair_predictions": True},
    }


@pytest.fixture
def patched_pipeline(monkeypatch: pytest.MonkeyPatch) -> PipelineCalls:
    """Patch side-effectful pipeline dependencies and capture call sequence."""
    calls = PipelineCalls()

    def fake_build_dataloaders(
        config: ConfigDict,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
        del config, distributed, rank, world_size
        loader = DataLoader(_EmptyDataset(), batch_size=1)
        return {"train": loader, "valid": loader, "test": loader}

    def fake_build_model(config: ConfigDict) -> nn.Module:
        del config
        return _DummyModel()

    def fake_run_training_stage(
        stage: str,
        config: ConfigDict,
        model: nn.Module,
        device: torch.device,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
        run_id: str,
        distributed_context: DistributedContext,
    ) -> Path:
        del config, model, device, dataloaders, distributed_context
        calls.training.append((stage, run_id))
        return Path(f"artifacts/{stage}_best_model.pth")

    def fake_run_evaluation_stage(
        config: ConfigDict,
        model: nn.Module,
        device: torch.device,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
        run_id: str,
        checkpoint_path: Path,
        distributed_context: DistributedContext,
    ) -> dict[str, float]:
        del config, model, device, dataloaders, distributed_context
        calls.evaluation.append((checkpoint_path, run_id))
        return {"accuracy": 1.0}

    def fake_run_adaptation_stage(
        config: ConfigDict,
        model: nn.Module,
        device: torch.device,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
        run_id: str,
        checkpoint_path: Path,
        distributed_context: DistributedContext,
    ) -> Path:
        del config, model, device, dataloaders, distributed_context
        calls.adaptation.append((checkpoint_path, run_id))
        return Path("artifacts/adapt_best_model.pth")

    def fake_run_topology_evaluation_stage(
        config: ConfigDict,
        model: nn.Module,
        device: torch.device,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
        run_id: str,
        checkpoint_path: Path,
        distributed_context: DistributedContext,
    ) -> dict[str, float]:
        del config, model, device, dataloaders, distributed_context
        calls.topology_evaluation.append((checkpoint_path, run_id))
        return {"graph_sim": 1.0}

    def fake_initialize_distributed(ddp_enabled: bool) -> DistributedContext:
        del ddp_enabled
        return DistributedContext(ddp_enabled=False, is_distributed=False)

    def fake_cleanup_distributed(context: DistributedContext) -> None:
        del context

    def fake_resolve_device(device_name: str) -> torch.device:
        del device_name
        return torch.device("cpu")

    monkeypatch.setattr(run_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(run_module, "build_model", fake_build_model)
    monkeypatch.setattr(run_module, "run_training_stage", fake_run_training_stage)
    monkeypatch.setattr(run_module, "run_shot_adaptation_stage", fake_run_adaptation_stage)
    monkeypatch.setattr(run_module, "run_evaluation_stage", fake_run_evaluation_stage)
    monkeypatch.setattr(
        run_module,
        "run_topology_evaluation_stage",
        fake_run_topology_evaluation_stage,
    )
    monkeypatch.setattr(run_module, "initialize_distributed", fake_initialize_distributed)
    monkeypatch.setattr(run_module, "cleanup_distributed", fake_cleanup_distributed)
    monkeypatch.setattr(run_module, "resolve_device", fake_resolve_device)
    return calls


def test_execute_pipeline_all_stages(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == [("train", "train_run")]
    assert patched_pipeline.adaptation == []
    assert patched_pipeline.evaluation == [(Path("artifacts/train_best_model.pth"), "eval_run")]
    assert patched_pipeline.topology_evaluation == []


def test_execute_pipeline_train_only(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["train"]

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == [("train", "train_run")]
    assert patched_pipeline.adaptation == []
    assert patched_pipeline.evaluation == []
    assert patched_pipeline.topology_evaluation == []


def test_execute_pipeline_evaluate_only(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["evaluate"]
    run_cfg["load_checkpoint_path"] = "artifacts/eval_input_model.pth"

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == []
    assert patched_pipeline.adaptation == []
    assert patched_pipeline.evaluation == [(Path("artifacts/eval_input_model.pth"), "eval_run")]
    assert patched_pipeline.topology_evaluation == []


def test_execute_pipeline_all_stages_with_shot(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    training_cfg = base_config["training_config"]
    assert isinstance(training_cfg, dict)
    training_cfg["domain_adaptation"] = {
        "enabled": True,
        "method": "shot",
        "target_split": "test",
    }

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == [("train", "train_run")]
    assert patched_pipeline.adaptation == [(Path("artifacts/train_best_model.pth"), "adapt_run")]
    assert patched_pipeline.evaluation == [(Path("artifacts/adapt_best_model.pth"), "eval_run")]
    assert patched_pipeline.topology_evaluation == []


def test_execute_pipeline_evaluate_only_with_shot(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["evaluate"]
    run_cfg["load_checkpoint_path"] = "artifacts/eval_input_model.pth"
    training_cfg = base_config["training_config"]
    assert isinstance(training_cfg, dict)
    training_cfg["domain_adaptation"] = {
        "enabled": True,
        "method": "shot",
        "target_split": "test",
    }

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == []
    assert patched_pipeline.adaptation == [(Path("artifacts/eval_input_model.pth"), "adapt_run")]
    assert patched_pipeline.evaluation == [(Path("artifacts/adapt_best_model.pth"), "eval_run")]
    assert patched_pipeline.topology_evaluation == []


def test_execute_pipeline_train_only_with_shot_skips_adaptation(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["train"]
    training_cfg = base_config["training_config"]
    assert isinstance(training_cfg, dict)
    training_cfg["domain_adaptation"] = {
        "enabled": True,
        "method": "shot",
        "target_split": "test",
    }

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == [("train", "train_run")]
    assert patched_pipeline.adaptation == []
    assert patched_pipeline.evaluation == []
    assert patched_pipeline.topology_evaluation == []


def test_execute_pipeline_topology_evaluate_only(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["topology_evaluate"]
    run_cfg["load_checkpoint_path"] = "artifacts/topology_input_model.pth"

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == []
    assert patched_pipeline.adaptation == []
    assert patched_pipeline.evaluation == []
    assert patched_pipeline.topology_evaluation == [
        (Path("artifacts/topology_input_model.pth"), "topology_eval_run")
    ]


def test_execute_pipeline_train_evaluate_and_topology(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["train", "evaluate", "topology_evaluate"]

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == [("train", "train_run")]
    assert patched_pipeline.evaluation == [(Path("artifacts/train_best_model.pth"), "eval_run")]
    assert patched_pipeline.topology_evaluation == [
        (Path("artifacts/train_best_model.pth"), "topology_eval_run")
    ]


@pytest.mark.parametrize(
    "invalid_stages,error",
    [
        (["evaluate", "train"], "must follow"),
        (["train", "train"], "duplicates"),
        (["pretrain"], "unsupported"),
    ],
)
def test_execute_pipeline_invalid_stages_raise(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
    invalid_stages: list[str],
    error: str,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = invalid_stages
    with pytest.raises(ValueError, match=error):
        run_module.execute_pipeline(base_config)


def test_execute_pipeline_staged_unfreeze_enables_ddp_find_unused(
    monkeypatch: pytest.MonkeyPatch,
    base_config: ConfigDict,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["train"]

    device_cfg = base_config["device_config"]
    assert isinstance(device_cfg, dict)
    device_cfg["ddp_enabled"] = True
    device_cfg["device"] = "cpu"

    training_cfg = base_config["training_config"]
    assert isinstance(training_cfg, dict)
    training_cfg["strategy"] = {
        "type": "staged_unfreeze",
        "unfreeze_epoch": 1,
        "initial_trainable_prefixes": ["output_head"],
    }

    ddp_call: dict[str, object] = {}

    class _FakeDDP(nn.Module):
        def __init__(
            self,
            module: nn.Module,
            device_ids: list[int] | None = None,
            find_unused_parameters: bool = False,
        ) -> None:
            super().__init__()
            self.module = module
            ddp_call["device_ids"] = device_ids
            ddp_call["find_unused_parameters"] = find_unused_parameters

        def forward(self, *args: object, **kwargs: object) -> object:
            return self.module(*args, **kwargs)

    def fake_build_dataloaders(
        config: ConfigDict,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
        del config, distributed, rank, world_size
        loader = DataLoader(_EmptyDataset(), batch_size=1)
        return {"train": loader, "valid": loader, "test": loader}

    def fake_build_model(config: ConfigDict) -> nn.Module:
        del config
        return _DummyModel()

    def fake_initialize_distributed(ddp_enabled: bool) -> DistributedContext:
        del ddp_enabled
        return DistributedContext(
            ddp_enabled=True,
            is_distributed=True,
            rank=0,
            local_rank=0,
            world_size=2,
        )

    def fake_cleanup_distributed(context: DistributedContext) -> None:
        del context

    def fake_resolve_device(device_name: str) -> torch.device:
        del device_name
        return torch.device("cpu")

    def fake_run_training_stage(
        stage: str,
        config: ConfigDict,
        model: nn.Module,
        device: torch.device,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
        run_id: str,
        distributed_context: DistributedContext,
    ) -> Path:
        del stage, config, model, device, dataloaders, run_id, distributed_context
        return Path("artifacts/train_best_model.pth")

    monkeypatch.setattr(run_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(run_module, "build_model", fake_build_model)
    monkeypatch.setattr(run_module, "initialize_distributed", fake_initialize_distributed)
    monkeypatch.setattr(run_module, "cleanup_distributed", fake_cleanup_distributed)
    monkeypatch.setattr(run_module, "resolve_device", fake_resolve_device)
    monkeypatch.setattr(run_module, "run_training_stage", fake_run_training_stage)
    monkeypatch.setattr(run_module, "DistributedDataParallel", _FakeDDP)

    run_module.execute_pipeline(base_config)

    assert ddp_call["device_ids"] is None
    assert ddp_call["find_unused_parameters"] is True


def test_execute_pipeline_shot_adaptation_enables_ddp_find_unused(
    monkeypatch: pytest.MonkeyPatch,
    base_config: ConfigDict,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["evaluate"]
    run_cfg["load_checkpoint_path"] = "artifacts/eval_input_model.pth"

    device_cfg = base_config["device_config"]
    assert isinstance(device_cfg, dict)
    device_cfg["ddp_enabled"] = True
    device_cfg["device"] = "cpu"

    training_cfg = base_config["training_config"]
    assert isinstance(training_cfg, dict)
    training_cfg["domain_adaptation"] = {
        "enabled": True,
        "method": "shot",
        "target_split": "test",
    }

    ddp_call: dict[str, object] = {}

    class _FakeDDP(nn.Module):
        def __init__(
            self,
            module: nn.Module,
            device_ids: list[int] | None = None,
            find_unused_parameters: bool = False,
        ) -> None:
            super().__init__()
            self.module = module
            ddp_call["device_ids"] = device_ids
            ddp_call["find_unused_parameters"] = find_unused_parameters

        def forward(self, *args: object, **kwargs: object) -> object:
            return self.module(*args, **kwargs)

    def fake_build_dataloaders(
        config: ConfigDict,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
        del config, distributed, rank, world_size
        loader = DataLoader(_EmptyDataset(), batch_size=1)
        return {"train": loader, "valid": loader, "test": loader}

    def fake_build_model(config: ConfigDict) -> nn.Module:
        del config
        return _DummyModel()

    def fake_initialize_distributed(ddp_enabled: bool) -> DistributedContext:
        del ddp_enabled
        return DistributedContext(
            ddp_enabled=True,
            is_distributed=True,
            rank=0,
            local_rank=0,
            world_size=2,
        )

    def fake_cleanup_distributed(context: DistributedContext) -> None:
        del context

    def fake_resolve_device(device_name: str) -> torch.device:
        del device_name
        return torch.device("cpu")

    def fake_run_adaptation_stage(
        config: ConfigDict,
        model: nn.Module,
        device: torch.device,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
        run_id: str,
        checkpoint_path: Path,
        distributed_context: DistributedContext,
    ) -> Path:
        del config, model, device, dataloaders, run_id, checkpoint_path, distributed_context
        return Path("artifacts/adapt_best_model.pth")

    def fake_run_evaluation_stage(
        config: ConfigDict,
        model: nn.Module,
        device: torch.device,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
        run_id: str,
        checkpoint_path: Path,
        distributed_context: DistributedContext,
    ) -> dict[str, float]:
        del config, model, device, dataloaders, run_id, checkpoint_path, distributed_context
        return {"accuracy": 1.0}

    monkeypatch.setattr(run_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(run_module, "build_model", fake_build_model)
    monkeypatch.setattr(run_module, "initialize_distributed", fake_initialize_distributed)
    monkeypatch.setattr(run_module, "cleanup_distributed", fake_cleanup_distributed)
    monkeypatch.setattr(run_module, "resolve_device", fake_resolve_device)
    monkeypatch.setattr(run_module, "run_shot_adaptation_stage", fake_run_adaptation_stage)
    monkeypatch.setattr(run_module, "run_evaluation_stage", fake_run_evaluation_stage)
    monkeypatch.setattr(run_module, "DistributedDataParallel", _FakeDDP)

    run_module.execute_pipeline(base_config)

    assert ddp_call["device_ids"] is None
    assert ddp_call["find_unused_parameters"] is True
