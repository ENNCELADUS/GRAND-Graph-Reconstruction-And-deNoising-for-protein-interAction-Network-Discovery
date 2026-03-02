"""Top-level pipeline orchestration logic."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.adapt import should_run_shot_adaptation
from src.run.bootstrap import set_global_seed
from src.run.stage_adapt import run_shot_adaptation_stage
from src.run.stage_evaluate import run_evaluation_stage
from src.run.stage_train import _build_stage_runtime, build_model, run_training_stage
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_int,
    as_str,
    as_str_list,
    extract_model_kwargs,
    get_section,
)
from src.utils.data_io import build_dataloaders
from src.utils.device import resolve_device
from src.utils.distributed import (
    DistributedContext,
    cleanup_distributed,
    initialize_distributed,
)
from src.utils.logging import generate_run_id, log_stage_event

DataLoaderMap = dict[str, torch.utils.data.DataLoader[dict[str, object]]]

DEFAULT_STAGES: tuple[str, ...] = ("train", "evaluate")
ALLOWED_STAGES: tuple[str, ...] = DEFAULT_STAGES
STAGE_ORDER: dict[str, int] = {stage: index for index, stage in enumerate(ALLOWED_STAGES)}


def _ddp_find_unused_parameters(config: ConfigDict) -> bool:
    """Return DDP ``find_unused_parameters`` setting from config."""
    device_cfg = get_section(config, "device_config")
    explicit_find_unused = device_cfg.get("find_unused_parameters")
    if explicit_find_unused is not None:
        return as_bool(explicit_find_unused, "device_config.find_unused_parameters")

    training_cfg = get_section(config, "training_config")
    strategy_cfg = training_cfg.get("strategy")
    if not isinstance(strategy_cfg, dict):
        return False
    strategy_type = as_str(
        strategy_cfg.get("type", "none"),
        "training_config.strategy.type",
    ).lower()
    return strategy_type == "staged_unfreeze"


def _selected_stages(run_cfg: ConfigDict) -> tuple[str, ...]:
    """Return validated ordered stages from ``run_config.stages``."""
    raw_stages = run_cfg.get("stages", list(DEFAULT_STAGES))
    configured_stages = tuple(
        stage.lower() for stage in as_str_list(raw_stages, "run_config.stages")
    )
    if not configured_stages:
        raise ValueError("run_config.stages must not be empty")
    if len(set(configured_stages)) != len(configured_stages):
        raise ValueError("run_config.stages must not contain duplicates")
    unsupported = [stage for stage in configured_stages if stage not in STAGE_ORDER]
    if unsupported:
        raise ValueError(
            "run_config.stages contains unsupported stage(s): "
            f"{', '.join(sorted(set(unsupported)))}"
        )
    previous_order = -1
    for stage in configured_stages:
        stage_order = STAGE_ORDER[stage]
        if stage_order < previous_order:
            raise ValueError("run_config.stages must follow: train -> evaluate")
        previous_order = stage_order
    return configured_stages


def _selected_stages_with_adaptation(
    selected_stages: tuple[str, ...],
    *,
    shot_enabled: bool,
) -> tuple[str, ...]:
    """Return selected stages with optional SHOT adaptation before evaluation."""
    if not shot_enabled or "evaluate" not in selected_stages:
        return selected_stages
    stages_with_adaptation: list[str] = []
    for stage in selected_stages:
        if stage == "evaluate":
            stages_with_adaptation.append("adapt")
        stages_with_adaptation.append(stage)
    return tuple(stages_with_adaptation)


def _log_event_for_stages(
    *,
    stage_names: Sequence[str],
    stage_loggers: dict[str, logging.Logger],
    event: str,
    **details: object,
) -> None:
    """Emit one stage event for each selected stage logger."""
    for stage in stage_names:
        log_stage_event(stage_loggers[stage], event, **details)


def _evaluation_checkpoint_path(
    *,
    train_checkpoint_path: Path | None,
    load_checkpoint_path: Path | None,
) -> Path:
    """Resolve checkpoint path for evaluation stage."""
    if train_checkpoint_path is not None:
        return train_checkpoint_path
    if load_checkpoint_path is not None:
        return load_checkpoint_path
    raise ValueError("load_checkpoint_path is required when evaluate runs without train stage")


def _len_or_unknown(value: object) -> int | str:
    """Return ``len(value)`` when available, otherwise ``'unknown'``."""
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return "unknown"


def execute_pipeline(
    config: ConfigDict,
    *,
    build_dataloaders_fn: Callable[..., DataLoaderMap] = build_dataloaders,
    build_model_fn: Callable[[ConfigDict], nn.Module] = build_model,
    run_training_stage_fn: Callable[..., Path] = run_training_stage,
    run_adaptation_stage_fn: Callable[..., Path] = run_shot_adaptation_stage,
    run_evaluation_stage_fn: Callable[..., dict[str, float]] = run_evaluation_stage,
    initialize_distributed_fn: Callable[[bool], DistributedContext] = initialize_distributed,
    cleanup_distributed_fn: Callable[[DistributedContext], None] = cleanup_distributed,
    resolve_device_fn: Callable[[str], torch.device] = resolve_device,
    distributed_data_parallel_cls: type[DistributedDataParallel] = DistributedDataParallel,
) -> None:
    """Execute pipeline according to configured stages."""
    run_cfg = get_section(config, "run_config")
    device_cfg = get_section(config, "device_config")
    seed = as_int(run_cfg.get("seed", 0), "run_config.seed")
    set_global_seed(seed=seed)
    distributed_context = initialize_distributed_fn(
        as_bool(device_cfg.get("ddp_enabled", False), "device_config.ddp_enabled")
    )
    ddp_find_unused_parameters = _ddp_find_unused_parameters(config)
    try:
        selected_stages = _selected_stages(run_cfg)
        train_run_id = generate_run_id(run_cfg.get("train_run_id"))
        adapt_run_id = generate_run_id(run_cfg.get("adapt_run_id"))
        eval_run_id = generate_run_id(run_cfg.get("eval_run_id"))
        load_checkpoint_value = run_cfg.get("load_checkpoint_path")
        load_checkpoint_path = (
            Path(str(load_checkpoint_value))
            if isinstance(load_checkpoint_value, str) and load_checkpoint_value
            else None
        )
        model_name, _ = extract_model_kwargs(config)
        stage_run_map: dict[str, str] = {
            "train": train_run_id,
            "adapt": adapt_run_id,
            "evaluate": eval_run_id,
        }
        selected_stages_with_adaptation = _selected_stages_with_adaptation(
            selected_stages,
            shot_enabled=should_run_shot_adaptation(config),
        )
        stage_loggers: dict[str, logging.Logger] = {}
        for stage in selected_stages_with_adaptation:
            _, _, stage_logger = _build_stage_runtime(
                model_name=model_name,
                stage=stage,
                run_id=stage_run_map[stage],
                distributed_context=distributed_context,
            )
            stage_loggers[stage] = stage_logger
            if distributed_context.is_main_process:
                log_stage_event(
                    stage_logger,
                    "startup",
                    stages=list(selected_stages),
                    run_id=stage_run_map[stage],
                    seed=seed,
                    rank=distributed_context.rank,
                    world_size=distributed_context.world_size,
                )
        requested_device = as_str(device_cfg.get("device", "cpu"), "device_config.device")
        device = resolve_device_fn(requested_device)
        if distributed_context.is_distributed and device.type == "cuda":
            device = torch.device("cuda", distributed_context.local_rank)
        if distributed_context.is_main_process:
            _log_event_for_stages(
                stage_names=selected_stages_with_adaptation,
                stage_loggers=stage_loggers,
                event="device",
                resolved_device=device,
            )

        dataloaders = build_dataloaders_fn(
            config=config,
            distributed=distributed_context.is_distributed,
            rank=distributed_context.rank,
            world_size=distributed_context.world_size,
        )
        if distributed_context.is_main_process:
            _log_event_for_stages(
                stage_names=selected_stages_with_adaptation,
                stage_loggers=stage_loggers,
                event="data_ready",
                train=_len_or_unknown(dataloaders["train"].dataset),
                valid=_len_or_unknown(dataloaders["valid"].dataset),
                test=_len_or_unknown(dataloaders["test"].dataset),
            )
        model = build_model_fn(config).to(device)
        if distributed_context.is_main_process:
            log_stage_event(
                stage_loggers[selected_stages_with_adaptation[0]],
                "model_init",
                model=model_name,
                params=sum(parameter.numel() for parameter in model.parameters()),
            )
        if distributed_context.is_distributed:
            model = distributed_data_parallel_cls(
                model,
                device_ids=[distributed_context.local_rank] if device.type == "cuda" else None,
                find_unused_parameters=ddp_find_unused_parameters,
            )
        if distributed_context.is_main_process:
            _log_event_for_stages(
                stage_names=selected_stages_with_adaptation,
                stage_loggers=stage_loggers,
                event="ddp_ready",
                wrapped=distributed_context.is_distributed,
            )

        train_checkpoint_path: Path | None = None
        adapt_checkpoint_path: Path | None = None
        if "train" in selected_stages:
            if distributed_context.is_main_process:
                log_stage_event(stage_loggers["train"], "begin_training")
            train_checkpoint_path = run_training_stage_fn(
                stage="train",
                config=config,
                model=model,
                device=device,
                dataloaders=dataloaders,
                run_id=train_run_id,
                distributed_context=distributed_context,
            )
            if distributed_context.is_main_process:
                log_stage_event(stage_loggers["train"], "end_training")

        if "adapt" in selected_stages_with_adaptation:
            base_checkpoint = train_checkpoint_path
            if base_checkpoint is None:
                base_checkpoint = load_checkpoint_path
            if base_checkpoint is None:
                raise ValueError(
                    "load_checkpoint_path is required when evaluate runs with domain adaptation "
                    "and no train stage"
                )
            if distributed_context.is_main_process:
                log_stage_event(stage_loggers["adapt"], "begin_adaptation")
            adapt_checkpoint_path = run_adaptation_stage_fn(
                config=config,
                model=model,
                device=device,
                dataloaders=dataloaders,
                run_id=adapt_run_id,
                checkpoint_path=base_checkpoint,
                distributed_context=distributed_context,
            )
            if distributed_context.is_main_process:
                log_stage_event(stage_loggers["adapt"], "end_adaptation")

        if "evaluate" in selected_stages:
            checkpoint_path = (
                adapt_checkpoint_path
                if adapt_checkpoint_path is not None
                else _evaluation_checkpoint_path(
                    train_checkpoint_path=train_checkpoint_path,
                    load_checkpoint_path=load_checkpoint_path,
                )
            )
            if distributed_context.is_main_process:
                log_stage_event(stage_loggers["evaluate"], "begin_evaluation")
            run_evaluation_stage_fn(
                config=config,
                model=model,
                device=device,
                dataloaders=dataloaders,
                run_id=eval_run_id,
                checkpoint_path=checkpoint_path,
                distributed_context=distributed_context,
            )
            if distributed_context.is_main_process:
                log_stage_event(stage_loggers["evaluate"], "end_evaluation")
    finally:
        cleanup_distributed_fn(distributed_context)
