"""Accelerator-owned runtime helpers for pipeline execution."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

import torch

from src.adapt import should_run_shot_adaptation
from src.pipeline.bootstrap import configure_root_logging as _configure_root_logging_impl
from src.pipeline.bootstrap import set_global_seed
from src.pipeline.config import PipelineConfig
from src.utils.accelerator import (
    AcceleratorLike,
    build_accelerator,
    distributed_context_from_accelerator,
)
from src.utils.config import as_bool, as_str, get_section
from src.utils.distributed import DistributedContext
from src.utils.logging import generate_run_id as _default_generate_run_id
from src.utils.logging import prepare_stage_directories, setup_stage_logger

_STAGE_RUN_ID_KEYS: tuple[tuple[str, str], ...] = (
    ("train", "train_run_id"),
    ("topology_finetune", "topology_finetune_run_id"),
    ("adapt", "adapt_run_id"),
    ("evaluate", "eval_run_id"),
    ("topology_evaluate", "topology_eval_run_id"),
)


@dataclass(frozen=True)
class StagePaths:
    """Resolved output paths for one stage."""

    log_dir: Path
    model_dir: Path


@dataclass
class PipelineRuntime:
    """Shared runtime objects derived from one real Accelerator instance."""

    config: PipelineConfig
    accelerator: AcceleratorLike
    device: torch.device
    distributed: DistributedContext
    stage_run_ids: dict[str, str]

    @property
    def is_main_process(self) -> bool:
        """Return whether the current process is the main rank."""
        return self.distributed.is_main_process

    @property
    def is_distributed(self) -> bool:
        """Return whether distributed execution is active."""
        return self.distributed.is_distributed

    @property
    def rank(self) -> int:
        """Return global process rank."""
        return self.distributed.rank

    @property
    def world_size(self) -> int:
        """Return world size."""
        return self.distributed.world_size


def configure_root_logging(logging_module: ModuleType, rank: int) -> None:
    """Configure process-level logging."""
    _configure_root_logging_impl(logging_module=logging_module, rank=rank)


def ddp_find_unused_parameters(raw_config: dict[str, object]) -> bool:
    """Return DDP ``find_unused_parameters`` setting from config."""
    config = cast(dict[str, object], raw_config)
    device_cfg = get_section(config, "device_config")
    explicit_find_unused = device_cfg.get("find_unused_parameters")
    if explicit_find_unused is not None:
        return as_bool(explicit_find_unused, "device_config.find_unused_parameters")

    run_cfg = get_section(config, "run_config")
    from src.pipeline.engine import selected_stages

    configured_stages = selected_stages(run_cfg)
    training_cfg = get_section(config, "training_config")
    adaptation_cfg = training_cfg.get("domain_adaptation")
    has_explicit_adaptation_cfg = isinstance(adaptation_cfg, dict)
    if (
        has_explicit_adaptation_cfg
        and should_run_shot_adaptation(config)
        and "evaluate" in configured_stages
    ):
        return True

    strategy_cfg = training_cfg.get("strategy")
    if not isinstance(strategy_cfg, dict):
        return False
    strategy_type = as_str(
        strategy_cfg.get("type", "none"),
        "training_config.strategy.type",
    ).lower()
    return strategy_type == "staged_unfreeze"


def build_runtime(
    config: PipelineConfig,
    *,
    build_accelerator_fn: Callable[..., AcceleratorLike] = build_accelerator,
) -> PipelineRuntime:
    """Create the accelerator-owned runtime for one pipeline execution."""
    set_global_seed(config.run.seed)
    accelerator = build_accelerator_fn(
        requested_device=config.device.requested_device,
        ddp_enabled=config.device.ddp_enabled,
        use_mixed_precision=config.device.use_mixed_precision,
        find_unused_parameters=ddp_find_unused_parameters(config.raw),
    )
    distributed = distributed_context_from_accelerator(
        accelerator=accelerator,
        ddp_enabled=config.device.ddp_enabled,
    )
    stage_run_ids = resolve_stage_run_ids(config=config, distributed=distributed)
    return PipelineRuntime(
        config=config,
        accelerator=accelerator,
        device=accelerator.device,
        distributed=distributed,
        stage_run_ids=stage_run_ids,
    )


def resolve_stage_run_ids(
    *,
    config: PipelineConfig,
    distributed: DistributedContext,
) -> dict[str, str]:
    """Resolve one shared run-id mapping for all stages and ranks."""
    run_cfg = config.raw.get("run_config", {})
    if not isinstance(run_cfg, dict):
        raise ValueError("run_config must be a mapping")
    if not distributed.is_distributed or not torch.distributed.is_initialized():
        return {
            stage: _generate_run_id(run_cfg.get(config_key))
            for stage, config_key in _STAGE_RUN_ID_KEYS
        }

    payload: list[object] = [{}]
    if distributed.is_main_process:
        payload[0] = {
            stage: _generate_run_id(run_cfg.get(config_key))
            for stage, config_key in _STAGE_RUN_ID_KEYS
        }
    torch.distributed.broadcast_object_list(payload, src=0)
    raw_stage_run_map = payload[0]
    if not isinstance(raw_stage_run_map, dict):
        raise ValueError("Broadcast stage run IDs must be a mapping")
    stage_run_ids: dict[str, str] = {}
    for stage, _ in _STAGE_RUN_ID_KEYS:
        value = raw_stage_run_map.get(stage)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Broadcast stage run ID for '{stage}' must be a non-empty string")
        stage_run_ids[stage] = value
    return stage_run_ids


def stage_paths(runtime: PipelineRuntime, stage: str, run_id: str) -> StagePaths:
    """Resolve output directories for one stage."""
    log_dir, model_dir = prepare_stage_directories(
        model_name=runtime.config.model_name,
        stage=stage,
        run_id=run_id,
    )
    return StagePaths(log_dir=log_dir, model_dir=model_dir)


def stage_logger(
    runtime: PipelineRuntime,
    stage: str,
    run_id: str,
    log_file: Path,
) -> logging.Logger:
    """Build one stage logger scoped to the runtime rank."""
    name = f"grand.{runtime.config.model_name}.{stage}.{run_id}.rank{runtime.rank}"
    if runtime.is_main_process:
        return setup_stage_logger(name=name, log_file=log_file)
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def save_checkpoint(
    runtime: PipelineRuntime,
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> None:
    """Persist model weights to disk once on the main process."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.accelerator.wait_for_everyone()
    if runtime.is_main_process:
        state_dict = _accelerator_state_dict(runtime.accelerator, model)
        _accelerator_save(runtime.accelerator, state_dict, checkpoint_path)
    runtime.accelerator.wait_for_everyone()


def load_checkpoint(
    runtime: PipelineRuntime,
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> None:
    """Load model weights from disk into the unwrapped model."""
    state_dict = torch.load(checkpoint_path, map_location=runtime.device)
    _unwrap_model(runtime.accelerator, model).load_state_dict(state_dict)


def barrier(runtime: PipelineRuntime) -> None:
    """Synchronize all ranks."""
    runtime.accelerator.wait_for_everyone()


@contextmanager
def main_process_first(runtime: PipelineRuntime) -> Iterator[None]:
    """Run a block with main-process priority when supported."""
    manager = getattr(runtime.accelerator, "main_process_first", None)
    if callable(manager):
        with manager():
            yield
        return
    runtime.accelerator.wait_for_everyone()
    yield
    runtime.accelerator.wait_for_everyone()


def _accelerator_state_dict(
    accelerator: AcceleratorLike,
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Return the best-available model state dict from the accelerator."""
    get_state_dict = getattr(accelerator, "get_state_dict", None)
    if callable(get_state_dict):
        return cast(dict[str, torch.Tensor], get_state_dict(model))
    return _unwrap_model(accelerator, model).state_dict()


def _accelerator_save(
    accelerator: AcceleratorLike,
    obj: object,
    checkpoint_path: Path,
) -> None:
    """Persist a checkpoint using the best-available accelerator save primitive."""
    save = getattr(accelerator, "save", None)
    if callable(save):
        try:
            save(obj, checkpoint_path, safe_serialization=False)
            return
        except TypeError:
            save(obj, checkpoint_path)
            return
    torch.save(obj, checkpoint_path)


def _unwrap_model(
    accelerator: AcceleratorLike,
    model: torch.nn.Module,
) -> torch.nn.Module:
    """Return the unwrapped model instance."""
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    if callable(unwrap_model):
        return cast(torch.nn.Module, unwrap_model(model))
    return model


def _generate_run_id(existing_value: object) -> str:
    """Resolve run-id generator via the compatibility wrapper when available."""
    try:
        from src.run import pipeline_orchestrator as compatibility_orchestrator

        return compatibility_orchestrator.generate_run_id(existing_value)
    except Exception:
        return _default_generate_run_id(existing_value)


__all__ = [
    "PipelineRuntime",
    "StagePaths",
    "barrier",
    "build_runtime",
    "configure_root_logging",
    "ddp_find_unused_parameters",
    "load_checkpoint",
    "main_process_first",
    "save_checkpoint",
    "stage_logger",
    "stage_paths",
]
