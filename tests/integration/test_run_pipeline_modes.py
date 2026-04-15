"""Integration tests for stage-based pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
import src.pipeline.engine as run_module
import src.pipeline.runtime as pipeline_runtime
import torch
from src.pipeline.runtime import PipelineRuntime
from src.utils.config import ConfigDict
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
    topology_finetuning: list[tuple[Path | None, str]] = field(default_factory=list)
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
        runtime: PipelineRuntime,
        model: nn.Module,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    ) -> Path:
        del model, dataloaders
        stage = "train"
        calls.training.append((stage, runtime.stage_run_id(stage)))
        return Path(f"artifacts/{stage}_best_model.pth")

    def fake_run_evaluation_stage(
        runtime: PipelineRuntime,
        model: nn.Module,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    ) -> dict[str, float]:
        del model, dataloaders
        checkpoint_path = runtime.checkpoint_paths["evaluate"]
        assert checkpoint_path is not None
        calls.evaluation.append((checkpoint_path, runtime.stage_run_id("evaluate")))
        return {"accuracy": 1.0}

    def fake_run_topology_finetuning_stage(
        runtime: PipelineRuntime,
        model: nn.Module,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    ) -> Path:
        del model, dataloaders
        calls.topology_finetuning.append(
            (
                runtime.checkpoint_paths.get("topology_finetune"),
                runtime.stage_run_id("topology_finetune"),
            )
        )
        return Path("artifacts/topology_finetune_best_model.pth")

    def fake_run_adaptation_stage(
        runtime: PipelineRuntime,
        model: nn.Module,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    ) -> Path:
        del model, dataloaders
        checkpoint_path = runtime.checkpoint_paths["adapt"]
        assert checkpoint_path is not None
        calls.adaptation.append((checkpoint_path, runtime.stage_run_id("adapt")))
        return Path("artifacts/adapt_best_model.pth")

    def fake_run_topology_evaluation_stage(
        runtime: PipelineRuntime,
        model: nn.Module,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    ) -> dict[str, float]:
        del model, dataloaders
        checkpoint_path = runtime.checkpoint_paths["topology_evaluate"]
        assert checkpoint_path is not None
        calls.topology_evaluation.append(
            (checkpoint_path, runtime.stage_run_id("topology_evaluate"))
        )
        return {"graph_sim": 1.0}

    class _FakeAccelerator:
        device = torch.device("cpu")
        is_main_process = True
        use_distributed = False
        process_index = 0
        local_process_index = 0
        num_processes = 1
        mixed_precision = "no"

    def fake_build_accelerator(**kwargs: object) -> _FakeAccelerator:
        del kwargs
        return _FakeAccelerator()

    monkeypatch.setattr(run_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(run_module, "build_model", fake_build_model)
    monkeypatch.setattr(run_module, "run_training_stage", fake_run_training_stage)
    monkeypatch.setattr(
        run_module,
        "run_topology_finetuning_stage",
        fake_run_topology_finetuning_stage,
    )
    monkeypatch.setattr(run_module, "run_shot_adaptation_stage", fake_run_adaptation_stage)
    monkeypatch.setattr(run_module, "run_evaluation_stage", fake_run_evaluation_stage)
    monkeypatch.setattr(
        run_module,
        "run_topology_evaluation_stage",
        fake_run_topology_evaluation_stage,
    )
    monkeypatch.setattr(run_module, "build_accelerator", fake_build_accelerator, raising=False)
    return calls


def test_execute_pipeline_all_stages(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == [("train", "train_run")]
    assert patched_pipeline.topology_finetuning == []
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
    assert patched_pipeline.topology_finetuning == []
    assert patched_pipeline.adaptation == []
    assert patched_pipeline.evaluation == []
    assert patched_pipeline.topology_evaluation == []


def test_execute_pipeline_builds_accelerator_runtime_instead_of_manual_ddp(
    base_config: ConfigDict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator_calls: list[dict[str, object]] = []
    stage_calls: list[str] = []

    class _FakeAccelerator:
        device = torch.device("cpu")
        is_main_process = True
        use_distributed = False
        process_index = 0
        local_process_index = 0
        num_processes = 1
        mixed_precision = "no"

        def wait_for_everyone(self) -> None:
            return None

    def fake_build_accelerator(
        *,
        requested_device: str,
        ddp_enabled: bool,
        use_mixed_precision: bool,
        find_unused_parameters: bool,
    ) -> _FakeAccelerator:
        accelerator_calls.append(
            {
                "requested_device": requested_device,
                "ddp_enabled": ddp_enabled,
                "use_mixed_precision": use_mixed_precision,
                "find_unused_parameters": find_unused_parameters,
            }
        )
        return _FakeAccelerator()

    def fake_build_dataloaders(
        *args: object,
        **kwargs: object,
    ) -> dict[str, DataLoader[dict[str, torch.Tensor]]]:
        del args, kwargs
        loader = DataLoader(_EmptyDataset(), batch_size=1)
        return {"train": loader, "valid": loader, "test": loader}

    def fake_build_model(*args: object, **kwargs: object) -> nn.Module:
        del args, kwargs
        return _DummyModel()

    def fake_run_training_stage(*args: object, **kwargs: object) -> Path:
        del args, kwargs
        stage_calls.append("train")
        return Path("artifacts/train_best_model.pth")

    monkeypatch.setattr(run_module, "build_accelerator", fake_build_accelerator, raising=False)
    monkeypatch.setattr(run_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(run_module, "build_model", fake_build_model)
    monkeypatch.setattr(run_module, "run_training_stage", fake_run_training_stage)

    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["train"]

    run_module.execute_pipeline(base_config)

    assert accelerator_calls == [
        {
            "requested_device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
            "find_unused_parameters": False,
        }
    ]
    assert stage_calls == ["train"]


def test_execute_pipeline_uses_accelerator_device_without_resolve_device(
    base_config: ConfigDict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_devices: list[torch.device] = []

    class _FakeAccelerator:
        device = torch.device("cpu")
        is_main_process = True
        use_distributed = False
        process_index = 0
        local_process_index = 0
        num_processes = 1
        mixed_precision = "no"

        def wait_for_everyone(self) -> None:
            return None

    def fake_build_accelerator(**kwargs: object) -> _FakeAccelerator:
        del kwargs
        return _FakeAccelerator()

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
        runtime: PipelineRuntime,
        model: nn.Module,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    ) -> Path:
        del model, dataloaders
        observed_devices.append(runtime.device)
        return Path("artifacts/train_best_model.pth")

    def fake_run_evaluation_stage(
        runtime: PipelineRuntime,
        model: nn.Module,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    ) -> dict[str, float]:
        del model, dataloaders
        observed_devices.append(runtime.device)
        return {"accuracy": 1.0}

    monkeypatch.setattr(run_module, "build_accelerator", fake_build_accelerator, raising=False)
    monkeypatch.setattr(run_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(run_module, "build_model", fake_build_model)
    monkeypatch.setattr(run_module, "run_training_stage", fake_run_training_stage)
    monkeypatch.setattr(run_module, "run_evaluation_stage", fake_run_evaluation_stage)

    run_module.execute_pipeline(base_config)

    assert observed_devices == [torch.device("cpu"), torch.device("cpu")]


def test_execute_pipeline_distributed_worker_reuses_main_run_ids(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["train"]
    run_cfg["train_run_id"] = None

    class _WorkerAccelerator:
        device = torch.device("cpu")
        is_main_process = False
        use_distributed = True
        process_index = 1
        local_process_index = 1
        num_processes = 2
        mixed_precision = "no"

    monkeypatch.setattr(
        run_module,
        "build_accelerator",
        lambda **kwargs: _WorkerAccelerator(),
        raising=False,
    )
    monkeypatch.setattr(pipeline_runtime.torch.distributed, "is_initialized", lambda: True)

    def fake_broadcast_object_list(payload: list[object], src: int) -> None:
        del src
        payload[0] = {
            "train": "shared_train_run",
            "topology_finetune": "shared_topology_run",
            "adapt": "shared_adapt_run",
            "evaluate": "shared_eval_run",
            "topology_evaluate": "shared_topology_eval_run",
        }

    monkeypatch.setattr(
        pipeline_runtime.torch.distributed,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )
    generated_run_ids = iter(
        [
            "rank1_train_run",
            "rank1_topology_run",
            "rank1_adapt_run",
            "rank1_eval_run",
            "rank1_topology_eval_run",
        ]
    )
    monkeypatch.setattr(
        pipeline_runtime,
        "_default_generate_run_id",
        lambda existing_value: next(generated_run_ids),
    )

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == [("train", "shared_train_run")]


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
    assert patched_pipeline.topology_finetuning == []
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
    assert patched_pipeline.topology_finetuning == []
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
    assert patched_pipeline.topology_finetuning == []
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
    assert patched_pipeline.topology_finetuning == []
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
    assert patched_pipeline.topology_finetuning == []
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
    assert patched_pipeline.topology_finetuning == []
    assert patched_pipeline.evaluation == [(Path("artifacts/train_best_model.pth"), "eval_run")]
    assert patched_pipeline.topology_evaluation == [
        (Path("artifacts/train_best_model.pth"), "topology_eval_run")
    ]


def test_execute_pipeline_topology_finetune_then_evaluate(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["topology_finetune", "evaluate", "topology_evaluate"]
    run_cfg["topology_finetune_run_id"] = "topology_ft_run"
    run_cfg["load_checkpoint_path"] = "artifacts/pairwise_checkpoint.pth"

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == []
    assert patched_pipeline.topology_finetuning == [
        (Path("artifacts/pairwise_checkpoint.pth"), "topology_ft_run")
    ]
    assert patched_pipeline.evaluation == [
        (Path("artifacts/topology_finetune_best_model.pth"), "eval_run")
    ]
    assert patched_pipeline.topology_evaluation == [
        (Path("artifacts/topology_finetune_best_model.pth"), "topology_eval_run")
    ]


def test_execute_pipeline_train_then_topology_finetune_uses_train_checkpoint(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["train", "topology_finetune", "evaluate"]
    run_cfg["topology_finetune_run_id"] = "topology_ft_run"

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == [("train", "train_run")]
    assert patched_pipeline.topology_finetuning == [
        (Path("artifacts/train_best_model.pth"), "topology_ft_run")
    ]
    assert patched_pipeline.evaluation == [
        (Path("artifacts/topology_finetune_best_model.pth"), "eval_run")
    ]
    assert patched_pipeline.topology_evaluation == []


def test_execute_pipeline_topology_finetune_scratch_does_not_require_checkpoint(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["topology_finetune", "evaluate"]
    run_cfg["topology_finetune_run_id"] = "topology_ft_run"
    run_cfg["load_checkpoint_path"] = None

    base_config["topology_finetune"] = {"init_mode": "scratch"}

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == []
    assert patched_pipeline.topology_finetuning == [(None, "topology_ft_run")]
    assert patched_pipeline.evaluation == [
        (Path("artifacts/topology_finetune_best_model.pth"), "eval_run")
    ]


def test_execute_pipeline_topology_finetune_launches_without_runtime_supervision_preparation(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del patched_pipeline
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["topology_finetune"]
    run_cfg["load_checkpoint_path"] = None

    base_config["topology_finetune"] = {"init_mode": "scratch"}
    call_order: list[str] = []

    def _fake_run_topology_finetuning_stage(
        runtime: PipelineRuntime,
        model: nn.Module,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    ) -> Path:
        del runtime, model, dataloaders
        call_order.append("stage")
        return Path("artifacts/topology_finetune_best_model.pth")

    monkeypatch.setattr(
        run_module,
        "run_topology_finetuning_stage",
        _fake_run_topology_finetuning_stage,
    )

    run_module.execute_pipeline(base_config)

    assert call_order == ["stage"]


def test_execute_pipeline_train_then_topology_finetune_scratch_ignores_train_checkpoint(
    base_config: ConfigDict,
    patched_pipeline: PipelineCalls,
) -> None:
    run_cfg = base_config["run_config"]
    assert isinstance(run_cfg, dict)
    run_cfg["stages"] = ["train", "topology_finetune", "evaluate"]
    run_cfg["topology_finetune_run_id"] = "topology_ft_run"

    base_config["topology_finetune"] = {"init_mode": "scratch"}

    run_module.execute_pipeline(base_config)

    assert patched_pipeline.training == [("train", "train_run")]
    assert patched_pipeline.topology_finetuning == [(None, "topology_ft_run")]
    assert patched_pipeline.evaluation == [
        (Path("artifacts/topology_finetune_best_model.pth"), "eval_run")
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

    accelerator_call: dict[str, object] = {}

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

    class _FakeAccelerator:
        device = torch.device("cpu")
        is_main_process = True
        use_distributed = True
        process_index = 0
        local_process_index = 0
        num_processes = 2
        mixed_precision = "no"

    def fake_build_accelerator(**kwargs: object) -> _FakeAccelerator:
        accelerator_call.update(kwargs)
        return _FakeAccelerator()

    def fake_run_training_stage(
        runtime: PipelineRuntime,
        model: nn.Module,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    ) -> Path:
        del runtime, model, dataloaders
        return Path("artifacts/train_best_model.pth")

    monkeypatch.setattr(run_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(run_module, "build_model", fake_build_model)
    monkeypatch.setattr(run_module, "build_accelerator", fake_build_accelerator, raising=False)
    monkeypatch.setattr(run_module, "run_training_stage", fake_run_training_stage)

    run_module.execute_pipeline(base_config)

    assert accelerator_call["find_unused_parameters"] is True


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

    accelerator_call: dict[str, object] = {}

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

    class _FakeAccelerator:
        device = torch.device("cpu")
        is_main_process = True
        use_distributed = True
        process_index = 0
        local_process_index = 0
        num_processes = 2
        mixed_precision = "no"

    def fake_build_accelerator(**kwargs: object) -> _FakeAccelerator:
        accelerator_call.update(kwargs)
        return _FakeAccelerator()

    def fake_run_adaptation_stage(
        runtime: PipelineRuntime,
        model: nn.Module,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    ) -> Path:
        del runtime, model, dataloaders
        return Path("artifacts/adapt_best_model.pth")

    def fake_run_evaluation_stage(
        runtime: PipelineRuntime,
        model: nn.Module,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
    ) -> dict[str, float]:
        del runtime, model, dataloaders
        return {"accuracy": 1.0}

    monkeypatch.setattr(run_module, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(run_module, "build_model", fake_build_model)
    monkeypatch.setattr(run_module, "build_accelerator", fake_build_accelerator, raising=False)
    monkeypatch.setattr(run_module, "run_shot_adaptation_stage", fake_run_adaptation_stage)
    monkeypatch.setattr(run_module, "run_evaluation_stage", fake_run_evaluation_stage)

    run_module.execute_pipeline(base_config)

    assert accelerator_call["find_unused_parameters"] is True
