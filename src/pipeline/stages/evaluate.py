"""Evaluation-stage execution helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from src.evaluate import DEFAULT_DECISION_THRESHOLD, Evaluator
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
) -> tuple[float, str]:
    """Resolve the PRING-aligned fixed decision threshold."""
    raw_threshold = eval_cfg.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)
    field_name = "evaluate.decision_threshold"
    if isinstance(raw_threshold, bool):
        raise ValueError(f"{field_name} must be a number or mapping")
    if isinstance(raw_threshold, (int, float, str)):
        threshold = as_float(raw_threshold, field_name)
        if threshold != DEFAULT_DECISION_THRESHOLD:
            raise ValueError(f"{field_name} must be fixed at 0.5")
        return threshold, "fixed"
    if not isinstance(raw_threshold, dict):
        raise ValueError(f"{field_name} must be a number or mapping")

    threshold_cfg = raw_threshold
    mode = as_str(threshold_cfg.get("mode", "fixed"), f"{field_name}.mode").lower()
    if mode != "fixed":
        raise ValueError("evaluate.decision_threshold.mode must be 'fixed'")
    threshold = as_float(
        threshold_cfg.get("value", DEFAULT_DECISION_THRESHOLD),
        f"{field_name}.value",
    )
    if threshold != DEFAULT_DECISION_THRESHOLD:
        raise ValueError(f"{field_name}.value must be 0.5")
    return threshold, mode


def _resolve_evaluate_mode(eval_cfg: ConfigDict) -> str:
    """Resolve the evaluation inference mode."""
    mode = as_str(eval_cfg.get("mode", "pairwise"), "evaluate.mode").lower()
    if mode not in {"pairwise", "graph_assembly"}:
        raise ValueError("evaluate.mode must be 'pairwise' or 'graph_assembly'")
    return mode


def run_evaluation_stage(
    runtime: PipelineRuntime,
    model: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader[dict[str, object]]],
    *,
    checkpoint_path: Path,
) -> dict[str, float]:
    """Run test evaluation and persist ``evaluate.csv``."""
    config = runtime.config.raw
    device = runtime.device
    device_cfg = get_section(config, "device_config")
    use_amp = device.type == "cuda" and as_bool(
        device_cfg.get("use_mixed_precision", False),
        "device_config.use_mixed_precision",
    )
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
    evaluate_mode = _resolve_evaluate_mode(eval_cfg)
    decision_threshold, threshold_mode = _resolve_decision_threshold(
        eval_cfg=eval_cfg,
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
    if evaluate_mode == "graph_assembly":
        from src.pipeline.stages.topology_evaluate import (
            _build_topology_loader,
            _model_supports_graph_forward,
            _predict_tccig_graph_assembly_result,
            _topology_paths,
            write_graph_assembly_diagnostics,
        )

        if not _model_supports_graph_forward(model, runtime.accelerator):
            raise ValueError("evaluate.mode='graph_assembly' requires a TCCIG graph model")
        all_test_path, _, _ = _topology_paths(config)
        topology_bundle = _build_topology_loader(config=config, split_path=all_test_path)
        graph_assembly_result = _predict_tccig_graph_assembly_result(
            config=config,
            model=model,
            dataset=topology_bundle.dataset,
            records=topology_bundle.records,
            device=device,
            accelerator=runtime.accelerator,
        )
        labels = torch.tensor(topology_bundle.dataset.labels(), dtype=torch.long)
        probabilities = torch.tensor(graph_assembly_result.probabilities, dtype=torch.float32)
        predictions = torch.tensor(graph_assembly_result.predictions, dtype=torch.long)
        metrics = evaluator.metrics_from_outputs(
            labels=labels,
            probabilities=probabilities,
            predictions=predictions,
            average_loss=0.0,
            prefix=None,
        )
        if runtime.is_main_process:
            diagnostics_path = log_dir / "graph_assembly_diagnostics.json"
            write_graph_assembly_diagnostics(
                output_path=diagnostics_path,
                result=graph_assembly_result,
            )
            log_stage_event(
                logger,
                "tccig_graph_assembly_evaluate",
                assembly_rule=graph_assembly_result.assembly_rule,
                m_hat=graph_assembly_result.m_hat,
                selected_edges=graph_assembly_result.selected_edges,
                candidate_count=graph_assembly_result.candidate_count,
                pair_count=len(topology_bundle.records),
            )
            log_stage_event(logger, "graph_assembly_diagnostics_written", path=diagnostics_path)
    else:
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
