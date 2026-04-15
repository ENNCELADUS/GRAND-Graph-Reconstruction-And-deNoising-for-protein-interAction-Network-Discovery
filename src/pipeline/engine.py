"""Top-level accelerator-owned pipeline orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch import nn

from src.adapt import should_run_shot_adaptation
from src.pipeline.config import PipelineConfig
from src.pipeline.runtime import PipelineRuntime, build_runtime
from src.pipeline.topology import run_topology_evaluation_stage, run_topology_finetuning_stage
from src.run.stage_adapt import run_shot_adaptation_stage
from src.run.stage_evaluate import run_evaluation_stage
from src.run.stage_train import _build_stage_runtime, build_model, run_training_stage
from src.utils.accelerator import build_accelerator
from src.utils.config import ConfigDict, as_str, get_section
from src.utils.data_io import build_dataloaders
from src.utils.logging import log_stage_event

DataLoaderMap = dict[str, torch.utils.data.DataLoader[dict[str, object]]]

DEFAULT_STAGES: tuple[str, ...] = ("train", "evaluate")
ALLOWED_STAGES: tuple[str, ...] = (
    "train",
    "topology_finetune",
    "evaluate",
    "topology_evaluate",
)
STAGE_ORDER: dict[str, int] = {stage: index for index, stage in enumerate(ALLOWED_STAGES)}


def selected_stages(run_cfg: ConfigDict) -> tuple[str, ...]:
    """Return validated ordered stages from ``run_config.stages``."""
    raw_stages = run_cfg.get("stages", list(DEFAULT_STAGES))
    from src.utils.config import as_str_list

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
            raise ValueError(
                "run_config.stages must follow: train -> topology_finetune -> "
                "evaluate -> topology_evaluate"
            )
        previous_order = stage_order
    return configured_stages


def selected_stages_with_adaptation(
    stage_names: tuple[str, ...],
    *,
    shot_enabled: bool,
) -> tuple[str, ...]:
    """Return selected stages with optional SHOT adaptation before evaluation."""
    if not shot_enabled or "evaluate" not in stage_names:
        return stage_names
    stages_with_adaptation: list[str] = []
    for stage in stage_names:
        if stage == "evaluate":
            stages_with_adaptation.append("adapt")
        stages_with_adaptation.append(stage)
    return tuple(stages_with_adaptation)


def evaluation_checkpoint_path(
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


def topology_finetune_checkpoint_path(
    *,
    config: ConfigDict,
    train_checkpoint_path: Path | None,
    load_checkpoint_path: Path | None,
) -> Path | None:
    """Resolve checkpoint path for topology fine-tuning."""
    finetune_cfg = config.get("topology_finetune", {})
    if finetune_cfg is None:
        finetune_cfg = {}
    if not isinstance(finetune_cfg, dict):
        raise ValueError("topology_finetune must be a mapping")
    init_mode = as_str(
        finetune_cfg.get("init_mode", "warm_start"),
        "topology_finetune.init_mode",
    ).lower()
    if init_mode == "scratch":
        return None
    if init_mode != "warm_start":
        raise ValueError("topology_finetune.init_mode must be 'warm_start' or 'scratch'")
    return evaluation_checkpoint_path(
        train_checkpoint_path=train_checkpoint_path,
        load_checkpoint_path=load_checkpoint_path,
    )


def execute_pipeline(
    config: ConfigDict,
    *,
    build_dataloaders_fn: Callable[..., DataLoaderMap] = build_dataloaders,
    build_model_fn: Callable[[ConfigDict], nn.Module] = build_model,
    build_accelerator_fn: object = build_accelerator,
    run_training_stage_fn: Callable[..., Path] = run_training_stage,
    run_topology_finetuning_stage_fn: Callable[..., Path] = run_topology_finetuning_stage,
    run_adaptation_stage_fn: Callable[..., Path] = run_shot_adaptation_stage,
    run_evaluation_stage_fn: Callable[..., dict[str, float]] = run_evaluation_stage,
    run_topology_evaluation_stage_fn: Callable[
        ...,
        dict[str, float],
    ] = run_topology_evaluation_stage,
) -> None:
    """Execute pipeline according to configured stages."""
    typed_config = PipelineConfig.from_dict(config)
    runtime = build_runtime(typed_config, build_accelerator_fn=build_accelerator_fn)
    execute_pipeline_with_runtime(
        runtime,
        build_dataloaders_fn=build_dataloaders_fn,
        build_model_fn=build_model_fn,
        run_training_stage_fn=run_training_stage_fn,
        run_topology_finetuning_stage_fn=run_topology_finetuning_stage_fn,
        run_adaptation_stage_fn=run_adaptation_stage_fn,
        run_evaluation_stage_fn=run_evaluation_stage_fn,
        run_topology_evaluation_stage_fn=run_topology_evaluation_stage_fn,
    )


def execute_pipeline_with_runtime(
    runtime: PipelineRuntime,
    *,
    build_dataloaders_fn: Callable[..., DataLoaderMap],
    build_model_fn: Callable[[ConfigDict], nn.Module],
    run_training_stage_fn: Callable[..., Path],
    run_topology_finetuning_stage_fn: Callable[..., Path],
    run_adaptation_stage_fn: Callable[..., Path],
    run_evaluation_stage_fn: Callable[..., dict[str, float]],
    run_topology_evaluation_stage_fn: Callable[..., dict[str, float]],
) -> None:
    """Execute the pipeline using an already-built runtime."""
    config = runtime.config.raw
    run_cfg = get_section(config, "run_config")
    selected = selected_stages(run_cfg)
    load_checkpoint_path = (
        Path(runtime.config.run.load_checkpoint_path)
        if runtime.config.run.load_checkpoint_path is not None
        else None
    )
    stage_run_map = runtime.stage_run_ids
    train_run_id = stage_run_map["train"]
    topology_finetune_run_id = stage_run_map["topology_finetune"]
    adapt_run_id = stage_run_map["adapt"]
    eval_run_id = stage_run_map["evaluate"]
    topology_eval_run_id = stage_run_map["topology_evaluate"]
    stage_names = selected_stages_with_adaptation(
        selected,
        shot_enabled=should_run_shot_adaptation(config),
    )

    stage_loggers: dict[str, logging.Logger] = {}
    for stage in stage_names:
        _, _, logger = _build_stage_runtime(
            runtime.config.model_name,
            stage,
            stage_run_map[stage],
            runtime.distributed,
        )
        stage_loggers[stage] = logger

    dataloaders = build_dataloaders_fn(
        config=config,
        distributed=runtime.is_distributed,
        rank=runtime.rank,
        world_size=runtime.world_size,
    )
    model = build_model_fn(config).to(runtime.device)
    _log_event_for_stages(
        stage_names=stage_names,
        stage_loggers=stage_loggers,
        event="pipeline_runtime",
        device=str(runtime.device),
        distributed=runtime.is_distributed,
        world_size=runtime.world_size,
        train_batches=_len_or_unknown(dataloaders.get("train", [])),
        valid_batches=_len_or_unknown(dataloaders.get("valid", [])),
        test_batches=_len_or_unknown(dataloaders.get("test", [])),
    )

    train_checkpoint_path: Path | None = None
    finetuned_checkpoint_path: Path | None = None
    adapted_checkpoint_path: Path | None = None

    if "train" in selected:
        train_checkpoint_path = run_training_stage_fn(
            "train",
            config,
            model,
            runtime.device,
            dataloaders,
            train_run_id,
            runtime.distributed,
            runtime.accelerator,
        )

    if "topology_finetune" in selected:
        finetuned_checkpoint_path = run_topology_finetuning_stage_fn(
            config,
            model,
            runtime.device,
            dataloaders,
            topology_finetune_run_id,
            topology_finetune_checkpoint_path(
                config=config,
                train_checkpoint_path=train_checkpoint_path,
                load_checkpoint_path=load_checkpoint_path,
            ),
            runtime.distributed,
            runtime.accelerator,
        )

    if "evaluate" in selected:
        eval_input_checkpoint = finetuned_checkpoint_path or evaluation_checkpoint_path(
            train_checkpoint_path=train_checkpoint_path,
            load_checkpoint_path=load_checkpoint_path,
        )
        if should_run_shot_adaptation(config):
            adapted_checkpoint_path = run_adaptation_stage_fn(
                config,
                model,
                runtime.device,
                dataloaders,
                adapt_run_id,
                eval_input_checkpoint,
                runtime.distributed,
                runtime.accelerator,
            )
        run_evaluation_stage_fn(
            config,
            model,
            runtime.device,
            dataloaders,
            eval_run_id,
            adapted_checkpoint_path or eval_input_checkpoint,
            runtime.distributed,
            runtime.accelerator,
        )

    if "topology_evaluate" in selected:
        topology_eval_checkpoint = (
            finetuned_checkpoint_path
            or adapted_checkpoint_path
            or evaluation_checkpoint_path(
                train_checkpoint_path=train_checkpoint_path,
                load_checkpoint_path=load_checkpoint_path,
            )
        )
        run_topology_evaluation_stage_fn(
            config,
            model,
            runtime.device,
            dataloaders,
            topology_eval_run_id,
            topology_eval_checkpoint,
            runtime.distributed,
            runtime.accelerator,
        )


def _len_or_unknown(value: object) -> int | str:
    """Return ``len(value)`` when available, otherwise ``'unknown'``."""
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return "unknown"


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


__all__ = [
    "ALLOWED_STAGES",
    "DEFAULT_STAGES",
    "STAGE_ORDER",
    "evaluation_checkpoint_path",
    "execute_pipeline",
    "execute_pipeline_with_runtime",
    "selected_stages",
    "selected_stages_with_adaptation",
    "topology_finetune_checkpoint_path",
]
