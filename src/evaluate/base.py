"""Generic evaluator for validation and test stages."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from src.pipeline.loops import forward_model, move_batch_to_device
from src.pipeline.runtime import AcceleratorLike
from src.train.config import LossConfig
from src.utils.losses import binary_classification_loss

BatchValue = object
BatchInput = Mapping[str, BatchValue]
BatchDict = dict[str, BatchValue]


def _safe_float(value: float) -> float:
    """Convert metric value to a finite float.

    Args:
        value: Candidate numeric metric value.

    Returns:
        Original value if finite, otherwise ``0.0``.
    """
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


class Evaluator:
    """Metric computation for a single data loader pass.

    Args:
        metrics: Metric names to compute.
        loss_config: Loss hyperparameters for consistent loss reporting.
    """

    def __init__(
        self,
        metrics: list[str],
        loss_config: LossConfig,
        *,
        accelerator: AcceleratorLike,
        decision_threshold: float = 0.5,
        use_amp: bool = False,
        gather_for_metrics: bool = False,
    ) -> None:
        self.metrics = [metric.lower() for metric in metrics]
        self.loss_config = loss_config
        self.decision_threshold = self._validate_decision_threshold(decision_threshold)
        self.use_amp = use_amp
        self.accelerator = accelerator
        self.gather_metrics = gather_for_metrics

    @staticmethod
    def _validate_decision_threshold(decision_threshold: float) -> float:
        """Validate and normalize decision threshold."""
        threshold = float(decision_threshold)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("decision_threshold must be in [0, 1]")
        return threshold

    @staticmethod
    def _batch_tensor(batch: BatchInput, key: str) -> torch.Tensor:
        """Return required tensor field from a batch dictionary."""
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Batch field '{key}' must be a torch.Tensor")
        return value

    @staticmethod
    def _move_batch_to_device(batch: BatchInput, device: torch.device) -> BatchDict:
        """Move tensor fields to target device while preserving non-tensor fields."""
        return move_batch_to_device(batch, device)

    @staticmethod
    def _forward_model(model: nn.Module, batch: BatchInput) -> dict[str, torch.Tensor]:
        """Execute model forward and validate output contract.

        Args:
            model: Model to evaluate.
            batch: Model input batch on target device.

        Returns:
            Model output dictionary.

        Raises:
            ValueError: If model output is not a dictionary.
        """
        return forward_model(model, batch)

    @staticmethod
    def _binary_stats(labels: torch.Tensor, predictions: torch.Tensor) -> tuple[float, float]:
        """Compute sensitivity and specificity.

        Args:
            labels: Ground-truth binary labels.
            predictions: Predicted binary labels.

        Returns:
            Tuple of ``(sensitivity, specificity)``.
        """
        matrix = confusion_matrix(
            labels.cpu().numpy(),
            predictions.cpu().numpy(),
            labels=[0, 1],
        )
        tn, fp, fn, tp = matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return sensitivity, specificity

    @staticmethod
    def _score_auc_metric(
        has_both_classes: bool,
        score_fn: Callable[[], float],
    ) -> float:
        """Return AUC-like score when both classes are present, else ``0.0``."""
        if not has_both_classes:
            return 0.0
        return _safe_float(score_fn())

    def _metric_scorers(
        self,
        labels: torch.Tensor,
        probabilities: torch.Tensor,
        predictions: torch.Tensor,
        has_both_classes: bool,
    ) -> dict[str, Callable[[], float]]:
        """Build score functions keyed by metric name."""
        label_array = labels.cpu().numpy()
        prob_array = probabilities.cpu().numpy()
        pred_array = predictions.cpu().numpy()
        needs_binary_stats = "sensitivity" in self.metrics or "specificity" in self.metrics
        sensitivity = 0.0
        specificity = 0.0
        if needs_binary_stats:
            sensitivity, specificity = self._binary_stats(labels=labels, predictions=predictions)

        def score_auroc() -> float:
            return self._score_auc_metric(
                has_both_classes=has_both_classes,
                score_fn=lambda: roc_auc_score(label_array, prob_array),
            )

        def score_auprc() -> float:
            return self._score_auc_metric(
                has_both_classes=has_both_classes,
                score_fn=lambda: average_precision_score(label_array, prob_array),
            )

        return {
            "auroc": score_auroc,
            "auprc": score_auprc,
            "accuracy": lambda: _safe_float(accuracy_score(label_array, pred_array)),
            "sensitivity": lambda: _safe_float(sensitivity),
            "specificity": lambda: _safe_float(specificity),
            "precision": lambda: _safe_float(
                precision_score(label_array, pred_array, zero_division=0)
            ),
            "recall": lambda: _safe_float(recall_score(label_array, pred_array, zero_division=0)),
            "f1": lambda: _safe_float(f1_score(label_array, pred_array, zero_division=0)),
            "mcc": lambda: _safe_float(matthews_corrcoef(label_array, pred_array)),
        }

    def _compute_metrics(
        self,
        labels: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> dict[str, float]:
        """Compute configured metrics for binary classification.

        Args:
            labels: Ground-truth binary labels.
            probabilities: Predicted probabilities in ``[0, 1]``.

        Returns:
            Metric dictionary without split prefix.
        """
        results: dict[str, float] = {}
        has_both_classes = torch.unique(labels).numel() > 1
        predictions = (probabilities >= self.decision_threshold).long()
        scorers = self._metric_scorers(
            labels=labels,
            probabilities=probabilities,
            predictions=predictions,
            has_both_classes=has_both_classes,
        )
        for metric in self.metrics:
            score_fn = scorers.get(metric)
            if score_fn is None:
                continue
            results[metric] = score_fn()
        return results

    def _collect_probabilities_and_labels(
        self,
        model: nn.Module,
        data_loader: DataLoader[Mapping[str, object]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Collect probabilities, labels, and average loss for one loader."""
        all_probs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_losses: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch in data_loader:
                prepared_batch = self._move_batch_to_device(batch=batch, device=device)
                with self.accelerator.autocast():
                    output = self._forward_model(model=model, batch=prepared_batch)
                logits = output["logits"]
                labels = self._batch_tensor(prepared_batch, "label").float()
                per_sample_loss = binary_classification_loss(
                    logits=logits,
                    labels=labels,
                    loss_config=self.loss_config,
                    reduction="none",
                )
                reduced_logits = (
                    logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
                )
                probs = torch.sigmoid(reduced_logits).detach()
                gathered_labels = labels.detach()
                gathered_losses = per_sample_loss.detach()
                if self.gather_metrics and self.accelerator.use_distributed:
                    probs = self.accelerator.gather_for_metrics(probs)
                    gathered_labels = self.accelerator.gather_for_metrics(gathered_labels)
                    gathered_losses = self.accelerator.gather_for_metrics(gathered_losses)
                all_probs.append(probs.cpu())
                all_labels.append(gathered_labels.cpu())
                all_losses.append(gathered_losses.cpu())

        return (
            torch.cat(all_labels, dim=0).long(),
            torch.cat(all_probs, dim=0),
            float(torch.cat(all_losses, dim=0).mean().item()),
        )

    @staticmethod
    def best_f1_threshold(labels: torch.Tensor, probabilities: torch.Tensor) -> float:
        """Return the validation threshold that maximizes F1."""
        if torch.unique(labels).numel() < 2:
            return 0.5

        label_array = labels.cpu().numpy()
        prob_array = probabilities.cpu().numpy()
        precision, recall, thresholds = precision_recall_curve(label_array, prob_array)
        if len(thresholds) == 0:
            return 0.5

        best_threshold = 0.5
        best_score = -1.0
        best_distance_to_midpoint = float("inf")
        for precision_value, recall_value, threshold in zip(
            precision[:-1],
            recall[:-1],
            thresholds,
            strict=False,
        ):
            denominator = precision_value + recall_value
            f1_value = (
                0.0 if denominator <= 0.0 else (2.0 * precision_value * recall_value) / denominator
            )
            threshold_value = float(threshold)
            distance_to_midpoint = abs(threshold_value - 0.5)
            if f1_value > best_score or (
                math.isclose(f1_value, best_score)
                and distance_to_midpoint < best_distance_to_midpoint
            ):
                best_score = float(f1_value)
                best_threshold = threshold_value
                best_distance_to_midpoint = distance_to_midpoint
        return best_threshold

    def select_best_f1_threshold(
        self,
        model: nn.Module,
        data_loader: DataLoader[Mapping[str, object]],
        device: torch.device,
    ) -> float:
        """Select decision threshold on a validation loader by F1."""
        labels, probabilities, _ = self._collect_probabilities_and_labels(
            model=model,
            data_loader=data_loader,
            device=device,
        )
        return self.best_f1_threshold(labels=labels, probabilities=probabilities)

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader[Mapping[str, object]],
        device: torch.device,
        prefix: str | None = "val",
    ) -> dict[str, float]:
        """Evaluate metrics on a loader.

        The caller controls ``model.eval()`` and ``torch.no_grad()`` context.

        Args:
            model: Model to evaluate.
            data_loader: Data loader for the split.
            device: Device where evaluation runs.
            prefix: Optional metric name prefix for output keys.

        Returns:
            Metric dictionary with prefixed names.
        """
        labels_tensor, probs_tensor, average_loss = self._collect_probabilities_and_labels(
            model=model,
            data_loader=data_loader,
            device=device,
        )
        metric_values = self._compute_metrics(labels=labels_tensor, probabilities=probs_tensor)
        metric_values["loss"] = average_loss
        if prefix is None:
            return metric_values
        return {f"{prefix}_{key}": value for key, value in metric_values.items()}
