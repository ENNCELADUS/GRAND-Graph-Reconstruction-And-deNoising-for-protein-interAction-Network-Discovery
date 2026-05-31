"""Evaluation-stage execution helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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


@dataclass(frozen=True)
class TCCIGPairwiseThresholdResult:
    """Resolved TCCIG pairwise threshold and validation diagnostics."""

    threshold: float
    mode: str
    target_metric: float
    mcc: float
    f1: float
    youden: float
    predicted_positive_rate: float
    validation_positive_rate: float
    fallback_reason: str | None = None


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


def _is_tccig_config(config: ConfigDict) -> bool:
    """Return whether the config selects the TCCIG model family."""
    model_cfg = get_section(config, "model_config")
    return as_str(model_cfg.get("model", ""), "model_config.model").lower() == "tccig"


def _tccig_pairwise_threshold_config(eval_cfg: ConfigDict) -> ConfigDict:
    """Return TCCIG pairwise threshold config with the P1 default."""
    raw_threshold_cfg = eval_cfg.get("tccig_pairwise_threshold", {"mode": "validation_mcc"})
    if raw_threshold_cfg is None:
        return {"mode": "validation_mcc"}
    if not isinstance(raw_threshold_cfg, dict):
        raise ValueError("evaluate.tccig_pairwise_threshold must be a mapping")
    return cast(ConfigDict, raw_threshold_cfg)


def _binary_confusion_counts(
    *,
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Return binary confusion counts as ``tn, fp, fn, tp``."""
    label_values = labels.long()
    prediction_values = predictions.long()
    true_positive = int(((label_values == 1) & (prediction_values == 1)).sum().item())
    true_negative = int(((label_values == 0) & (prediction_values == 0)).sum().item())
    false_positive = int(((label_values == 0) & (prediction_values == 1)).sum().item())
    false_negative = int(((label_values == 1) & (prediction_values == 0)).sum().item())
    return true_negative, false_positive, false_negative, true_positive


def _safe_mcc(*, tn: int, fp: int, fn: int, tp: int) -> float:
    """Return MCC with sklearn-compatible zero-denominator behavior."""
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)),
    )
    if denominator == 0.0:
        return 0.0
    return float(((tp * tn) - (fp * fn)) / denominator)


def _threshold_metrics(
    *,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    """Compute pairwise threshold metrics for one candidate threshold."""
    predictions = (probabilities >= threshold).long()
    tn, fp, fn, tp = _binary_confusion_counts(labels=labels, predictions=predictions)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "mcc": _safe_mcc(tn=tn, fp=fp, fn=fn, tp=tp),
        "f1": f1,
        "youden": float(recall + specificity - 1.0),
        "predicted_positive_rate": (
            float(predictions.float().mean().item()) if predictions.numel() else 0.0
        ),
    }


def _tccig_threshold_result_payload(
    result: TCCIGPairwiseThresholdResult,
) -> dict[str, float | str]:
    """Return a JSON/log-safe threshold diagnostics payload."""
    payload: dict[str, float | str] = {
        "threshold_mode": result.mode,
        "threshold_value": float(result.threshold),
        "threshold_target_metric": float(result.target_metric),
        "threshold_validation_mcc": float(result.mcc),
        "threshold_validation_f1": float(result.f1),
        "threshold_validation_youden": float(result.youden),
        "threshold_predicted_positive_rate": float(result.predicted_positive_rate),
        "threshold_validation_positive_rate": float(result.validation_positive_rate),
    }
    if result.fallback_reason is not None:
        payload["threshold_fallback_reason"] = result.fallback_reason
    return payload


def _resolve_tccig_pairwise_threshold_from_validation(
    *,
    evaluator: Evaluator,
    model: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader[dict[str, object]]],
    device: torch.device,
    eval_cfg: ConfigDict,
) -> TCCIGPairwiseThresholdResult:
    """Resolve the validation-calibrated TCCIG pairwise threshold."""
    labels, probabilities, _ = evaluator.collect_probabilities_and_labels(
        model=model,
        data_loader=dataloaders["valid"],
        device=device,
    )
    return select_tccig_pairwise_threshold(
        labels=labels,
        probabilities=probabilities,
        threshold_cfg=_tccig_pairwise_threshold_config(eval_cfg),
    )


def select_tccig_pairwise_threshold(
    *,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
    threshold_cfg: ConfigDict | None,
) -> TCCIGPairwiseThresholdResult:
    """Select a TCCIG-only pairwise threshold from validation outputs."""
    cfg = {} if threshold_cfg is None else threshold_cfg
    mode = as_str(cfg.get("mode", "validation_mcc"), "evaluate.tccig_pairwise_threshold.mode")
    mode = mode.lower()
    if mode not in {"fixed", "validation_mcc", "validation_f1", "validation_youden"}:
        raise ValueError(
            "evaluate.tccig_pairwise_threshold.mode must be 'fixed', "
            "'validation_mcc', 'validation_f1', or 'validation_youden'"
        )
    flattened_labels = labels.detach().cpu().long().flatten()
    flattened_probabilities = probabilities.detach().cpu().float().flatten()
    if flattened_labels.numel() != flattened_probabilities.numel():
        raise ValueError("TCCIG threshold labels and probabilities must have matching lengths")

    validation_positive_rate = (
        float(flattened_labels.float().mean().item()) if flattened_labels.numel() else 0.0
    )
    if mode == "fixed":
        threshold = as_float(
            cfg.get("value", DEFAULT_DECISION_THRESHOLD),
            "evaluate.tccig_pairwise_threshold.value",
        )
        metrics = _threshold_metrics(
            labels=flattened_labels,
            probabilities=flattened_probabilities,
            threshold=threshold,
        )
        return TCCIGPairwiseThresholdResult(
            threshold=threshold,
            mode=mode,
            target_metric=0.0,
            mcc=metrics["mcc"],
            f1=metrics["f1"],
            youden=metrics["youden"],
            predicted_positive_rate=metrics["predicted_positive_rate"],
            validation_positive_rate=validation_positive_rate,
        )

    fallback_reason: str | None = None
    if flattened_probabilities.numel() == 0:
        fallback_reason = "validation_probabilities_empty"
    elif torch.unique(flattened_labels).numel() < 2:
        fallback_reason = "validation_labels_single_class"
    if fallback_reason is not None:
        metrics = _threshold_metrics(
            labels=flattened_labels,
            probabilities=flattened_probabilities,
            threshold=DEFAULT_DECISION_THRESHOLD,
        )
        return TCCIGPairwiseThresholdResult(
            threshold=DEFAULT_DECISION_THRESHOLD,
            mode=mode,
            target_metric=0.0,
            mcc=metrics["mcc"],
            f1=metrics["f1"],
            youden=metrics["youden"],
            predicted_positive_rate=metrics["predicted_positive_rate"],
            validation_positive_rate=validation_positive_rate,
            fallback_reason=fallback_reason,
        )

    target_metric_name = mode.removeprefix("validation_")
    candidate_thresholds = sorted(
        {float(value) for value in flattened_probabilities.tolist()} | {DEFAULT_DECISION_THRESHOLD},
    )
    best_result: TCCIGPairwiseThresholdResult | None = None
    best_score: tuple[float, float, float, float] | None = None
    for threshold in candidate_thresholds:
        metrics = _threshold_metrics(
            labels=flattened_labels,
            probabilities=flattened_probabilities,
            threshold=threshold,
        )
        target_metric = metrics[target_metric_name]
        rate_error = abs(metrics["predicted_positive_rate"] - validation_positive_rate)
        score = (target_metric, metrics["mcc"], -rate_error, -threshold)
        if best_score is None or score > best_score:
            best_score = score
            best_result = TCCIGPairwiseThresholdResult(
                threshold=threshold,
                mode=mode,
                target_metric=target_metric,
                mcc=metrics["mcc"],
                f1=metrics["f1"],
                youden=metrics["youden"],
                predicted_positive_rate=metrics["predicted_positive_rate"],
                validation_positive_rate=validation_positive_rate,
            )
    if best_result is None:
        raise ValueError("TCCIG threshold selection failed to evaluate candidates")
    return best_result


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
    is_tccig = _is_tccig_config(config)
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
    if runtime.is_main_process and is_tccig and evaluate_mode == "graph_assembly":
        log_stage_event(
            logger,
            "fixed_threshold_diagnostic",
            mode=threshold_mode,
            value=decision_threshold,
        )
    elif runtime.is_main_process:
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
        threshold_result = _resolve_tccig_pairwise_threshold_from_validation(
            evaluator=evaluator,
            model=model,
            dataloaders=dataloaders,
            device=device,
            eval_cfg=eval_cfg,
        )
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
                extra_payload=_tccig_threshold_result_payload(threshold_result),
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
            log_stage_event(
                logger,
                "tccig_pairwise_threshold",
                **_tccig_threshold_result_payload(threshold_result),
            )
            log_stage_event(logger, "graph_assembly_diagnostics_written", path=diagnostics_path)
    elif is_tccig:
        threshold_result = _resolve_tccig_pairwise_threshold_from_validation(
            evaluator=evaluator,
            model=model,
            dataloaders=dataloaders,
            device=device,
            eval_cfg=eval_cfg,
        )
        labels, probabilities, average_loss = evaluator.collect_probabilities_and_labels(
            model=model,
            data_loader=dataloaders["test"],
            device=device,
        )
        predictions = (probabilities >= threshold_result.threshold).long()
        metrics = evaluator.metrics_from_outputs(
            labels=labels,
            probabilities=probabilities,
            predictions=predictions,
            average_loss=average_loss,
            prefix=None,
        )
        if runtime.is_main_process:
            log_stage_event(
                logger,
                "tccig_pairwise_threshold",
                **_tccig_threshold_result_payload(threshold_result),
            )
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
