"""Evaluation-stage execution helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from src.evaluate import Evaluator
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.stages.train import _build_loss_config
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_float,
    as_str,
    as_str_list,
    get_section,
)
from src.utils.logging import append_csv_row, log_stage_event

EVAL_CSV_COLUMNS = [
    "split",
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1",
    "mcc",
]


def _metrics_from_config(eval_cfg: ConfigDict) -> list[str]:
    """Extract configured metric names."""
    metrics = eval_cfg.get("metrics", [])
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)):
        raise ValueError("evaluate.metrics must be a sequence")
    return as_str_list(metrics, "evaluate.metrics")


def _resolve_decision_threshold(
    *,
    eval_cfg: ConfigDict,
    evaluator: Evaluator,
    model: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader[dict[str, object]]],
    device: torch.device,
) -> tuple[float, str]:
    """Resolve fixed or validation-selected decision threshold."""
    raw_threshold = eval_cfg.get("decision_threshold", 0.5)
    field_name = "evaluate.decision_threshold"
    if isinstance(raw_threshold, bool):
        raise ValueError(f"{field_name} must be a number or mapping")
    if isinstance(raw_threshold, (int, float, str)):
        return as_float(raw_threshold, field_name), "fixed"
    if not isinstance(raw_threshold, dict):
        raise ValueError(f"{field_name} must be a number or mapping")

    threshold_cfg = raw_threshold
    mode = as_str(threshold_cfg.get("mode", "fixed"), f"{field_name}.mode").lower()
    if mode == "fixed":
        return as_float(threshold_cfg.get("value", 0.5), f"{field_name}.value"), mode
    if mode == "best_f1_on_valid":
        return (
            evaluator.select_best_f1_threshold(
                model=model,
                data_loader=dataloaders["valid"],
                device=device,
            ),
            mode,
        )
    raise ValueError("evaluate.decision_threshold.mode must be 'fixed' or 'best_f1_on_valid'")


def run_evaluation_stage(
    runtime: PipelineRuntime,
    model: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader[dict[str, object]]],
) -> dict[str, float]:
    """Run test evaluation and persist ``evaluate.csv``."""
    config = runtime.config.raw
    device = runtime.device
    device_cfg = get_section(config, "device_config")
    use_amp = device.type == "cuda" and as_bool(
        device_cfg.get("use_mixed_precision", False),
        "device_config.use_mixed_precision",
    )
    checkpoint_path = runtime.checkpoint_paths.get("evaluate")
    if checkpoint_path is None:
        raise ValueError("runtime.checkpoint_paths['evaluate'] is required")
    checkpoint_path_resolved = Path(checkpoint_path)
    run_id = runtime.stage_run_id("evaluate")
    paths = runtime.stage_paths("evaluate")
    log_dir = paths.log_dir
    logger = runtime.stage_logger("evaluate", log_dir / "log.log")
    if runtime.is_main_process:
        log_stage_event(
            logger,
            "stage_start",
            run_id=run_id,
            checkpoint=checkpoint_path_resolved,
        )
    runtime.load_checkpoint(model, checkpoint_path_resolved)
    if runtime.is_main_process:
        log_stage_event(logger, "checkpoint_loaded", path=checkpoint_path_resolved)
    model.eval()
    eval_cfg = get_section(config, "evaluate")
    training_cfg = get_section(config, "training_config")
    configured_metrics = _metrics_from_config(eval_cfg)
    metrics_to_compute = sorted(set(configured_metrics + EVAL_CSV_COLUMNS[1:]))
    loss_config = _build_loss_config(training_cfg)
    threshold_probe = Evaluator(
        metrics=metrics_to_compute,
        loss_config=loss_config,
        use_amp=use_amp,
        accelerator=runtime.accelerator,
        gather_for_metrics=runtime.accelerator.use_distributed,
    )
    decision_threshold, threshold_mode = _resolve_decision_threshold(
        eval_cfg=eval_cfg,
        evaluator=threshold_probe,
        model=model,
        dataloaders=dataloaders,
        device=device,
    )
    evaluator = Evaluator(
        metrics=metrics_to_compute,
        loss_config=loss_config,
        decision_threshold=decision_threshold,
        use_amp=use_amp,
        accelerator=runtime.accelerator,
        gather_for_metrics=runtime.accelerator.use_distributed,
    )
    if runtime.is_main_process:
        log_stage_event(
            logger,
            "decision_threshold",
            mode=threshold_mode,
            value=decision_threshold,
        )
    with torch.no_grad():
        metrics = evaluator.evaluate(
            model=model,
            data_loader=dataloaders["test"],
            device=device,
            prefix=None,
        )
    if runtime.is_main_process:
        csv_row: dict[str, float | int | str] = {"split": "test"}
        for metric_name in EVAL_CSV_COLUMNS[1:]:
            csv_row[metric_name] = float(metrics.get(metric_name, 0.0))
        append_csv_row(
            csv_path=log_dir / "evaluate.csv",
            row=csv_row,
            fieldnames=EVAL_CSV_COLUMNS,
        )
        log_stage_event(logger, "evaluation_metrics", **csv_row)
        log_stage_event(logger, "csv_written", path=log_dir / "evaluate.csv")
        log_stage_event(
            logger,
            "stage_done",
            run_id=run_id,
        )
    runtime.barrier()
    return metrics
