"""Training-stage builders and execution helpers."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast

import torch
from torch import nn

from src.evaluate import DEFAULT_DECISION_THRESHOLD, Evaluator
from src.model import V3, V3_1, V4, V5
from src.pipeline.loops import reduce_scalar_mapping
from src.pipeline.runtime import AcceleratorLike, PipelineRuntime
from src.train.base import Trainer
from src.train.config import LossConfig, OptimizerConfig, SchedulerConfig
from src.train.strategies.lifecycle import NoOpStrategy, StagedUnfreezeStrategy, TrainingStrategy
from src.train.strategies.ohem import OHEMSampleStrategy
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_float,
    as_int,
    as_str,
    as_str_list,
    extract_model_kwargs,
    get_section,
)
from src.utils.early_stop import EarlyStopping
from src.utils.logging import append_csv_row, log_stage_event

AnnealStrategy = Literal["cos", "linear"]
ModelFactory = Callable[[ConfigDict], nn.Module]
DEFAULT_TRAINING_VAL_METRICS = ["auprc", "auroc"]
DEFAULT_HEARTBEAT_EVERY_N_STEPS = 20


def _training_logging_config(training_cfg: ConfigDict) -> ConfigDict:
    """Return ``training_config.logging`` mapping with validation."""
    logging_cfg = training_cfg.get("logging", {})
    if not isinstance(logging_cfg, dict):
        raise ValueError("training_config.logging must be a mapping")
    return cast(ConfigDict, logging_cfg)


def _build_v3_model(model_kwargs: ConfigDict) -> nn.Module:
    """Build V3 model instance."""
    return V3(**model_kwargs)


def _build_v3_1_model(model_kwargs: ConfigDict) -> nn.Module:
    """Build V3.1 model instance."""
    return V3_1(**model_kwargs)


def _build_v4_model(model_kwargs: ConfigDict) -> nn.Module:
    """Build V4 model instance."""
    return V4(**model_kwargs)


def _build_v5_model(model_kwargs: ConfigDict) -> nn.Module:
    """Build V5 model instance."""
    return V5(**model_kwargs)


MODEL_FACTORIES: dict[str, ModelFactory] = {
    "v3": _build_v3_model,
    "v3.1": _build_v3_1_model,
    "v4": _build_v4_model,
    "v5": _build_v5_model,
}


def _parse_anneal_strategy(value: object) -> AnnealStrategy:
    """Parse OneCycle anneal strategy."""
    anneal_strategy = as_str(value, "training_config.scheduler.anneal_strategy").lower()
    if anneal_strategy not in {"cos", "linear"}:
        raise ValueError("training_config.scheduler.anneal_strategy must be 'cos' or 'linear'")
    return cast(AnnealStrategy, anneal_strategy)


def _build_loss_config(training_cfg: ConfigDict) -> LossConfig:
    """Build loss configuration from ``training_config.loss``."""
    loss_cfg = training_cfg.get("loss", {})
    if not isinstance(loss_cfg, dict):
        raise ValueError("training_config.loss must be a mapping")
    return LossConfig(
        loss_type=as_str(loss_cfg.get("type", "bce_with_logits"), "training_config.loss.type"),
        pos_weight=as_float(loss_cfg.get("pos_weight", 1.0), "training_config.loss.pos_weight"),
        label_smoothing=as_float(
            loss_cfg.get("label_smoothing", 0.0), "training_config.loss.label_smoothing"
        ),
    )


def _training_validation_metrics(training_cfg: ConfigDict) -> list[str]:
    """Parse metrics to persist in ``training_step.csv``."""
    logging_cfg = _training_logging_config(training_cfg)
    return _parse_metric_names(
        raw_metrics=logging_cfg.get("validation_metrics", DEFAULT_TRAINING_VAL_METRICS),
        field_name="training_config.logging.validation_metrics",
        lowercase=True,
        allow_empty=False,
    )


def _parse_metric_names(
    raw_metrics: object,
    field_name: str,
    *,
    lowercase: bool,
    allow_empty: bool,
) -> list[str]:
    """Parse and validate configured metric names."""
    if not isinstance(raw_metrics, Sequence) or isinstance(raw_metrics, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence")
    metric_names = as_str_list(raw_metrics, field_name)
    if lowercase:
        metric_names = [metric_name.lower() for metric_name in metric_names]
    if not allow_empty and not metric_names:
        raise ValueError(f"{field_name} must not be empty")
    return metric_names


def _training_heartbeat_every_n_steps(training_cfg: ConfigDict) -> int:
    """Parse heartbeat interval for trainer progress logs."""
    logging_cfg = _training_logging_config(training_cfg)
    heartbeat_every_n_steps = as_int(
        logging_cfg.get("heartbeat_every_n_steps", DEFAULT_HEARTBEAT_EVERY_N_STEPS),
        "training_config.logging.heartbeat_every_n_steps",
    )
    if heartbeat_every_n_steps < 0:
        raise ValueError("training_config.logging.heartbeat_every_n_steps must be >= 0")
    return heartbeat_every_n_steps


def build_model(config: ConfigDict) -> nn.Module:
    """Build model from ``model_config``."""
    model_name, model_kwargs = extract_model_kwargs(config)
    factory = MODEL_FACTORIES.get(model_name)
    if factory is not None:
        return factory(model_kwargs)
    raise ValueError(f"Unknown model: {model_name}")


def build_trainer(
    config: ConfigDict,
    model: nn.Module,
    device: torch.device,
    accelerator: AcceleratorLike,
    steps_per_epoch: int = 1,
    logger: logging.Logger | None = None,
) -> tuple[Trainer, LossConfig]:
    """Instantiate trainer with optimizer/scheduler configs."""
    training_cfg = get_section(config, "training_config")
    data_cfg = get_section(config, "data_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    optimizer_cfg = get_section(training_cfg, "optimizer")
    scheduler_cfg = get_section(training_cfg, "scheduler")
    sampling_raw = dataloader_cfg.get("sampling", {})
    if not isinstance(sampling_raw, dict):
        raise ValueError("data_config.dataloader.sampling must be a mapping")
    sampling_cfg = sampling_raw

    optimizer_config = OptimizerConfig(
        optimizer_type=as_str(optimizer_cfg.get("type", "adamw"), "training_config.optimizer.type"),
        lr=as_float(optimizer_cfg.get("lr", 1e-4), "training_config.optimizer.lr"),
        beta1=as_float(optimizer_cfg.get("beta1", 0.9), "training_config.optimizer.beta1"),
        beta2=as_float(optimizer_cfg.get("beta2", 0.999), "training_config.optimizer.beta2"),
        eps=as_float(optimizer_cfg.get("eps", 1e-8), "training_config.optimizer.eps"),
        weight_decay=as_float(
            optimizer_cfg.get("weight_decay", 0.0),
            "training_config.optimizer.weight_decay",
        ),
    )
    scheduler_config = SchedulerConfig(
        scheduler_type=as_str(scheduler_cfg.get("type", "none"), "training_config.scheduler.type"),
        max_lr=as_float(
            scheduler_cfg.get("max_lr", optimizer_config.lr), "training_config.scheduler.max_lr"
        ),
        pct_start=as_float(
            scheduler_cfg.get("pct_start", 0.2),
            "training_config.scheduler.pct_start",
        ),
        div_factor=as_float(
            scheduler_cfg.get("div_factor", 25.0),
            "training_config.scheduler.div_factor",
        ),
        final_div_factor=as_float(
            scheduler_cfg.get("final_div_factor", 10000.0),
            "training_config.scheduler.final_div_factor",
        ),
        anneal_strategy=_parse_anneal_strategy(scheduler_cfg.get("anneal_strategy", "cos")),
    )

    sampling_strategy = as_str(
        sampling_cfg.get("strategy", "none"), "data_config.dataloader.sampling.strategy"
    ).lower()
    ohem_strategy = None
    if sampling_strategy == "ohem":
        batch_size = as_int(training_cfg.get("batch_size", 8), "training_config.batch_size")
        ohem_strategy = OHEMSampleStrategy(
            target_batch_size=batch_size,
            cap_protein=as_int(
                sampling_cfg.get("cap_protein", 4),
                "data_config.dataloader.sampling.cap_protein",
            ),
            warmup_epochs=as_int(
                sampling_cfg.get("warmup_epochs", 0),
                "data_config.dataloader.sampling.warmup_epochs",
            ),
        )

    device_cfg = get_section(config, "device_config")
    total_epochs = as_int(training_cfg.get("epochs", 1), "training_config.epochs")
    loss_config = _build_loss_config(training_cfg)
    trainer = Trainer(
        model=model,
        device=device,
        accelerator=accelerator,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        loss_config=loss_config,
        use_amp=as_bool(
            device_cfg.get("use_mixed_precision", False), "device_config.use_mixed_precision"
        ),
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        ohem_strategy=ohem_strategy,
        logger=logger,
        heartbeat_every_n_steps=_training_heartbeat_every_n_steps(training_cfg),
    )
    return trainer, loss_config


def build_strategy(config: ConfigDict) -> TrainingStrategy:
    """Build optional training strategy from config."""
    training_cfg = get_section(config, "training_config")
    strategy_cfg = training_cfg.get("strategy")
    if not isinstance(strategy_cfg, dict):
        return NoOpStrategy()
    strategy_type = str(strategy_cfg.get("type", "none")).lower()
    if strategy_type == "staged_unfreeze":
        prefixes_value = strategy_cfg.get("initial_trainable_prefixes", ["output_head"])
        if not isinstance(prefixes_value, list):
            raise ValueError("strategy.initial_trainable_prefixes must be a list")
        prefixes = tuple(str(prefix) for prefix in prefixes_value)
        return StagedUnfreezeStrategy(
            unfreeze_epoch=as_int(
                strategy_cfg.get("unfreeze_epoch", 1), "training_config.strategy.unfreeze_epoch"
            ),
            initial_trainable_prefixes=prefixes,
        )
    return NoOpStrategy()


def run_training_stage(
    runtime: PipelineRuntime,
    model: nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader[dict[str, object]]],
) -> Path:
    """Run stage training loop."""
    stage = "train"
    config = runtime.config.raw
    device = runtime.device
    run_id = runtime.stage_run_id(stage)
    paths = runtime.stage_paths(stage)
    log_dir = paths.log_dir
    model_dir = paths.model_dir
    stage_logger = runtime.stage_logger(stage, log_dir / "log.log")
    if runtime.is_main_process:
        log_stage_event(
            stage_logger,
            "stage_start",
            run_id=run_id,
        )
    training_cfg = get_section(config, "training_config")
    validation_metrics = _training_validation_metrics(training_cfg)

    trainer, loss_config = build_trainer(
        config=config,
        model=model,
        device=device,
        accelerator=runtime.accelerator,
        steps_per_epoch=len(dataloaders["train"]),
        logger=stage_logger,
    )
    train_loader = trainer.prepare_training_components(dataloaders["train"])
    stage_model = trainer.model
    strategy = build_strategy(config)

    monitor_metric = as_str(
        training_cfg.get("monitor_metric", "auprc"), "training_config.monitor_metric"
    ).lower()
    evaluator_metrics = sorted(set(validation_metrics + [monitor_metric]))
    device_cfg = get_section(config, "device_config")
    use_amp = device.type == "cuda" and as_bool(
        device_cfg.get("use_mixed_precision", False),
        "device_config.use_mixed_precision",
    )
    evaluator = Evaluator(
        metrics=evaluator_metrics,
        loss_config=loss_config,
        decision_threshold=DEFAULT_DECISION_THRESHOLD,
        use_amp=use_amp,
        accelerator=runtime.accelerator,
        gather_for_metrics=runtime.accelerator.use_distributed,
    )
    monitor_key = f"val_{monitor_metric}"
    patience = as_int(
        training_cfg.get("early_stopping_patience", 5),
        "training_config.early_stopping_patience",
    )
    early_stopping = EarlyStopping(patience=patience, mode="max")
    epochs = as_int(training_cfg.get("epochs", 1), "training_config.epochs")
    save_best_only = as_bool(
        get_section(config, "run_config").get("save_best_only", True),
        "run_config.save_best_only",
    )

    best_checkpoint_path = model_dir / "best_model.pth"
    csv_path = log_dir / "training_step.csv"
    csv_headers = [
        "Epoch",
        "Epoch Time",
        "Train Loss",
        "Val Loss",
        *[f"Val {metric}" for metric in validation_metrics],
        "Learning Rate",
    ]
    if runtime.is_main_process:
        log_stage_event(
            stage_logger,
            "train_config",
            epochs=epochs,
            monitor=monitor_metric,
            patience=patience,
        )
    strategy.on_train_begin(trainer)

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        if runtime.is_main_process:
            log_stage_event(stage_logger, "epoch_start", epoch=epoch + 1)
        train_sampler = train_loader.sampler
        train_batch_sampler = getattr(train_loader, "batch_sampler", None)
        set_epoch_fn = getattr(train_batch_sampler, "set_epoch", None)
        if callable(set_epoch_fn):
            set_epoch_fn(epoch)
        elif runtime.is_distributed and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        strategy.on_epoch_begin(trainer, epoch)
        train_stats = trainer.train_one_epoch(train_loader, epoch_index=epoch)
        reduced_train_stats = _reduce_train_stats(
            runtime=runtime,
            train_stats=train_stats,
        )
        stage_model.eval()
        with torch.no_grad():
            val_stats = evaluator.evaluate(
                model=stage_model,
                data_loader=dataloaders["valid"],
                device=device,
                prefix="val",
            )
        stage_model.train()
        strategy.on_epoch_end(trainer, epoch)
        epoch_seconds = time.perf_counter() - epoch_start

        row: dict[str, float | int | str] = {
            "Epoch": epoch + 1,
            "Epoch Time": epoch_seconds,
            "Train Loss": reduced_train_stats["loss"],
            "Val Loss": float(val_stats.get("val_loss", 0.0)),
            "Learning Rate": reduced_train_stats["lr"],
        }
        for metric in validation_metrics:
            row[f"Val {metric}"] = float(val_stats.get(f"val_{metric}", 0.0))
        if runtime.is_main_process:
            append_csv_row(csv_path=csv_path, row=row, fieldnames=csv_headers)
            log_stage_event(stage_logger, "csv_written", epoch=epoch + 1)

        monitor_value = float(val_stats.get(monitor_key, 0.0))
        should_stop = False
        save_best_checkpoint = False
        if runtime.is_main_process:
            improved, should_stop = early_stopping.update(monitor_value)
            save_best_checkpoint = improved
        save_best_checkpoint = _sync_flag(runtime, save_best_checkpoint)
        if save_best_checkpoint:
            runtime.save_checkpoint(stage_model, best_checkpoint_path)
            if runtime.is_main_process:
                log_stage_event(
                    stage_logger,
                    "best_saved",
                    epoch=epoch + 1,
                    monitor=monitor_key,
                    value=monitor_value,
                )
        if not save_best_only:
            epoch_checkpoint_path = model_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth"
            runtime.save_checkpoint(stage_model, epoch_checkpoint_path)
            if runtime.is_main_process:
                log_stage_event(
                    stage_logger,
                    "checkpoint_saved",
                    epoch=epoch + 1,
                )

        if runtime.is_main_process:
            val_metric_fields = {
                f"val_{m}": float(val_stats.get(f"val_{m}", 0.0)) for m in validation_metrics
            }
            log_stage_event(
                stage_logger,
                "epoch_done",
                epoch=epoch + 1,
                time=epoch_seconds,
                train_loss=reduced_train_stats["loss"],
                val_loss=float(val_stats.get("val_loss", 0.0)),
                **val_metric_fields,
            )
        if runtime.is_distributed:
            stop_flag = torch.tensor(
                [1 if should_stop and runtime.is_main_process else 0],
                device=device,
                dtype=torch.int64,
            )
            reduced_stop_flag = runtime.accelerator.reduce(stop_flag, reduction="sum")
            should_stop = bool(int(reduced_stop_flag.item()) > 0)
        if should_stop:
            if runtime.is_main_process:
                log_stage_event(stage_logger, "early_stop", epoch=epoch + 1)
            break

    fallback_save = runtime.is_main_process and not best_checkpoint_path.exists()
    if _sync_flag(runtime, fallback_save):
        runtime.save_checkpoint(stage_model, best_checkpoint_path)
    if runtime.is_main_process and fallback_save:
        log_stage_event(stage_logger, "fallback_saved")
    if runtime.is_main_process:
        log_stage_event(
            stage_logger,
            "stage_done",
            run_id=run_id,
        )
    runtime.barrier()
    return best_checkpoint_path


def _sync_flag(runtime: PipelineRuntime, flag: bool) -> bool:
    """Return a flag that is true on all ranks when any rank reports true."""
    if not runtime.is_distributed:
        return flag
    flag_tensor = torch.tensor([1 if flag else 0], device=runtime.device, dtype=torch.int64)
    reduced = runtime.accelerator.reduce(flag_tensor, reduction="sum")
    return bool(int(reduced.item()) > 0)


def _reduce_train_stats(
    *,
    runtime: PipelineRuntime,
    train_stats: dict[str, float],
) -> dict[str, float]:
    """Reduce train loss sufficient statistics across distributed ranks."""
    if not runtime.is_distributed:
        return train_stats
    reduced = reduce_scalar_mapping(
        runtime.accelerator,
        {
            "loss_sum": train_stats["loss_sum"],
            "batch_count": train_stats["batch_count"],
        },
        device=runtime.device,
        reduction="sum",
    )
    batch_count = reduced["batch_count"]
    if batch_count <= 0.0:
        raise RuntimeError("Distributed training produced zero batches across all ranks")
    return {
        **train_stats,
        "loss": reduced["loss_sum"] / batch_count,
        "loss_sum": reduced["loss_sum"],
        "batch_count": batch_count,
    }
