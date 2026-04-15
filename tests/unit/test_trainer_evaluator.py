"""Unit tests for generic trainer and evaluator."""

from __future__ import annotations

import logging
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as functional
from src.evaluate import Evaluator
from src.pipeline.stages.adapt import ADAPT_CSV_COLUMNS
from src.pipeline.stages.evaluate import EVAL_CSV_COLUMNS
from src.pipeline.stages.train import _training_validation_metrics
from src.train.base import OptimizerConfig, SchedulerConfig, Trainer
from src.train.strategies.ohem import OHEMSampleStrategy
from src.utils.config import ConfigDict
from src.utils.losses import LossConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TinyDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic toy dataset for binary classification."""

    def __init__(self) -> None:
        self._features = torch.tensor(
            [
                [0.0, 1.0, 0.5, 0.1],
                [1.0, 0.0, 0.2, 0.3],
                [0.9, 0.8, 0.1, 0.2],
                [0.1, 0.2, 0.9, 0.7],
            ],
            dtype=torch.float32,
        )
        self._labels = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)

    def __len__(self) -> int:
        """Return dataset size."""
        return int(self._features.size(0))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one dataset example."""
        feature = self._features[index]
        label = self._labels[index]
        return {"x": feature, "label": label}


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
    }


class TinyModel(nn.Module):
    """Minimal model implementing required forward output contract."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.linear(x).squeeze(-1)
        per_sample_loss = functional.binary_cross_entropy_with_logits(
            logits,
            label,
            reduction="none",
        )
        loss = per_sample_loss.mean()
        return {
            "logits": logits.unsqueeze(-1),
            "loss": loss,
            "per_sample_loss": per_sample_loss,
        }


class OHEMProbeModel(nn.Module):
    """Probe model that records forward batch sizes for OHEM flow tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)
        self.forward_batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
        del label
        self.forward_batch_sizes.append(int(x.size(0)))
        logits = self.linear(x).squeeze(-1)
        return {"logits": logits.unsqueeze(-1)}


class SequenceFieldDataset(Dataset[dict[str, object]]):
    """Toy dataset including non-tensor sequence fields."""

    def __init__(self) -> None:
        self._records = [("AAAA", 1.0), ("BB", 0.0), ("CCC", 1.0), ("D", 0.0)]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one dataset example."""
        sequence, label = self._records[index]
        return {"seq_a": sequence, "label": torch.tensor(label, dtype=torch.float32)}


def _collate_sequence(batch: list[dict[str, object]]) -> dict[str, object]:
    labels: list[torch.Tensor] = []
    seq_a: list[str] = []
    for sample in batch:
        label_value = sample.get("label")
        sequence = sample.get("seq_a")
        if not isinstance(label_value, torch.Tensor):
            raise TypeError("label must be tensor")
        if not isinstance(sequence, str):
            raise TypeError("seq_a must be string")
        labels.append(label_value)
        seq_a.append(sequence)
    return {"seq_a": seq_a, "label": torch.stack(labels, dim=0)}


class SequenceAwareModel(nn.Module):
    """Toy model that consumes sequence lists plus tensor labels."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, seq_a: list[str], label: torch.Tensor) -> dict[str, torch.Tensor]:
        del label
        lengths = torch.tensor(
            [len(sequence) for sequence in seq_a],
            dtype=torch.float32,
            device=self.linear.weight.device,
        ).unsqueeze(-1)
        logits = self.linear(lengths).squeeze(-1)
        return {"logits": logits.unsqueeze(-1)}


class GradModeProbeModel(nn.Module):
    """Probe model that records gradient mode during evaluator forwards."""

    def __init__(self) -> None:
        super().__init__()
        self.grad_enabled_states: list[bool] = []

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
        del label
        self.grad_enabled_states.append(torch.is_grad_enabled())
        logits = x.sum(dim=1, keepdim=True)
        return {"logits": logits}


class FakeAccelerator:
    """Small accelerator stub used to verify trainer integration."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.use_distributed = True
        self.autocast_calls = 0
        self.backward_calls = 0
        self.prepare_calls = 0
        self.gather_for_metrics_calls = 0

    def prepare(self, *components: object) -> tuple[object, ...]:
        self.prepare_calls += 1
        return components

    def autocast(self) -> object:
        self.autocast_calls += 1
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_calls += 1
        loss.backward()

    def gather_for_metrics(self, value: torch.Tensor) -> torch.Tensor:
        self.gather_for_metrics_calls += 1
        return value


def test_trainer_runs_single_epoch() -> None:
    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        optimizer_config=OptimizerConfig(optimizer_type="adamw", lr=1e-2),
        scheduler_config=SchedulerConfig(scheduler_type="none"),
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        use_amp=False,
        total_epochs=1,
        steps_per_epoch=len(loader),
        accelerator=FakeAccelerator(),
        ohem_strategy=OHEMSampleStrategy(target_batch_size=1, cap_protein=4),
    )
    metrics = trainer.train_one_epoch(loader)
    assert "loss" in metrics
    assert "lr" in metrics
    assert metrics["loss"] >= 0.0


def test_trainer_uses_accelerator_runtime_for_backward_and_autocast() -> None:
    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
    accelerator = FakeAccelerator()
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        accelerator=accelerator,
        optimizer_config=OptimizerConfig(optimizer_type="adamw", lr=1e-2),
        scheduler_config=SchedulerConfig(scheduler_type="none"),
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        use_amp=False,
        total_epochs=1,
        steps_per_epoch=len(loader),
    )

    trainer.train_one_epoch(loader)

    assert accelerator.prepare_calls >= 1
    assert accelerator.autocast_calls == len(loader)
    assert accelerator.backward_calls == len(loader)


def test_evaluator_uses_accelerator_runtime_for_autocast_and_metric_gather() -> None:
    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
    accelerator = FakeAccelerator()
    evaluator = Evaluator(
        metrics=["accuracy", "auprc"],
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        decision_threshold=0.5,
        use_amp=False,
        accelerator=accelerator,
        gather_for_metrics=True,
    )

    metrics = evaluator.evaluate(
        model=model.eval(),
        data_loader=loader,
        device=torch.device("cpu"),
        prefix=None,
    )

    assert "accuracy" in metrics
    assert accelerator.autocast_calls == len(loader)
    assert accelerator.gather_for_metrics_calls >= len(loader) * 2


def test_trainer_handles_non_tensor_batch_fields() -> None:
    model = SequenceAwareModel()
    loader = DataLoader(
        SequenceFieldDataset(),
        batch_size=2,
        shuffle=False,
        collate_fn=_collate_sequence,
    )
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        optimizer_config=OptimizerConfig(optimizer_type="adamw", lr=1e-2),
        scheduler_config=SchedulerConfig(scheduler_type="none"),
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        use_amp=False,
        total_epochs=1,
        steps_per_epoch=len(loader),
        accelerator=FakeAccelerator(),
    )
    metrics = trainer.train_one_epoch(loader)
    assert "loss" in metrics
    assert metrics["loss"] >= 0.0


def test_trainer_heartbeat_logging(caplog: pytest.LogCaptureFixture) -> None:
    logger_name = "tests.trainer.heartbeat"
    trainer_logger = logging.getLogger(logger_name)
    trainer_logger.handlers.clear()
    trainer_logger.propagate = True
    trainer_logger.setLevel(logging.INFO)

    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=1, shuffle=False, collate_fn=_collate)
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        optimizer_config=OptimizerConfig(optimizer_type="adamw", lr=1e-2),
        scheduler_config=SchedulerConfig(scheduler_type="none"),
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        use_amp=False,
        total_epochs=1,
        steps_per_epoch=len(loader),
        accelerator=FakeAccelerator(),
        logger=trainer_logger,
        heartbeat_every_n_steps=2,
    )

    with caplog.at_level(logging.INFO, logger=logger_name):
        trainer.train_one_epoch(loader, epoch_index=0)

    messages = [record.getMessage() for record in caplog.records if record.name == logger_name]
    assert any("Epoch 1 | Step 1/4" in message for message in messages)
    assert any("Epoch 1 | Step 2/4" in message for message in messages)
    assert any("Epoch 1 | Step 4/4" in message for message in messages)
    assert not any("Epoch 1 | Step 3/4" in message for message in messages)


def test_ohem_disables_pos_weight_for_selected_batch_loss() -> None:
    model = TinyModel()
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        optimizer_config=OptimizerConfig(optimizer_type="adamw", lr=1e-2),
        scheduler_config=SchedulerConfig(scheduler_type="none"),
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=5.0, label_smoothing=0.0),
        use_amp=False,
        total_epochs=1,
        steps_per_epoch=1,
        accelerator=FakeAccelerator(),
        ohem_strategy=OHEMSampleStrategy(target_batch_size=2, cap_protein=4, warmup_epochs=0),
    )
    logits = torch.tensor([[0.0], [1.0], [-0.5]], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    batch = {
        "label": labels,
        "protein_a_id": torch.tensor([0, 1, 2], dtype=torch.long),
        "protein_b_id": torch.tensor([3, 4, 5], dtype=torch.long),
    }
    loss = trainer._select_loss(output={"logits": logits}, batch=batch, epoch_index=0)
    per_sample_unweighted = functional.binary_cross_entropy_with_logits(
        logits.squeeze(-1),
        labels,
        reduction="none",
    )
    expected = per_sample_unweighted[torch.tensor([1, 2])].mean()
    assert torch.isclose(loss, expected, atol=1e-6)


def test_ohem_training_mines_in_chunks_and_reduces_backward_batch() -> None:
    model = OHEMProbeModel()
    loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False, collate_fn=_collate)
    trainer = Trainer(
        model=model,
        device=torch.device("cpu"),
        optimizer_config=OptimizerConfig(optimizer_type="adamw", lr=1e-2),
        scheduler_config=SchedulerConfig(scheduler_type="none"),
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        use_amp=False,
        total_epochs=1,
        steps_per_epoch=1,
        accelerator=FakeAccelerator(),
        ohem_strategy=OHEMSampleStrategy(target_batch_size=2, cap_protein=4, warmup_epochs=0),
    )

    trainer.train_one_epoch(loader, epoch_index=0)

    assert model.forward_batch_sizes == [2, 2, 2]


def test_training_csv_schema_header_order_regression() -> None:
    training_cfg: ConfigDict = {
        "logging": {
            "validation_metrics": ["auprc", "auroc"],
        }
    }
    validation_metrics = _training_validation_metrics(training_cfg)
    header = [
        "Epoch",
        "Epoch Time",
        "Train Loss",
        "Val Loss",
        *[f"Val {metric}" for metric in validation_metrics],
        "Learning Rate",
    ]
    assert header == [
        "Epoch",
        "Epoch Time",
        "Train Loss",
        "Val Loss",
        "Val auprc",
        "Val auroc",
        "Learning Rate",
    ]


def test_evaluate_csv_schema_header_order_and_required_columns() -> None:
    assert EVAL_CSV_COLUMNS == [
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
    required_columns = {
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
    }
    assert required_columns.issubset(set(EVAL_CSV_COLUMNS))
    assert len(EVAL_CSV_COLUMNS) == len(required_columns)


def test_adapt_csv_schema_header_order_and_required_columns() -> None:
    assert ADAPT_CSV_COLUMNS == [
        "Epoch",
        "Epoch Time",
        "Entropy Loss",
        "Diversity Loss",
        "Pseudo CE Loss",
        "Total Loss",
        "Learning Rate",
    ]


def test_evaluator_returns_metric_dictionary() -> None:
    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
    evaluator = Evaluator(
        metrics=["accuracy", "f1", "auroc"],
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        accelerator=FakeAccelerator(),
    )
    model.eval()
    with torch.no_grad():
        metrics = evaluator.evaluate(
            model=model,
            data_loader=loader,
            device=torch.device("cpu"),
            prefix="val",
        )
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics
    assert "val_f1" in metrics
    assert "val_auroc" in metrics


def test_evaluator_handles_non_tensor_batch_fields() -> None:
    model = SequenceAwareModel()
    loader = DataLoader(
        SequenceFieldDataset(),
        batch_size=2,
        shuffle=False,
        collate_fn=_collate_sequence,
    )
    evaluator = Evaluator(
        metrics=["accuracy", "f1", "auroc"],
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        accelerator=FakeAccelerator(),
    )
    model.eval()
    with torch.no_grad():
        metrics = evaluator.evaluate(
            model=model,
            data_loader=loader,
            device=torch.device("cpu"),
            prefix="val",
        )
    assert "val_loss" in metrics
    assert "val_accuracy" in metrics


def test_evaluator_without_prefix_returns_raw_metric_names() -> None:
    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
    evaluator = Evaluator(
        metrics=["accuracy", "f1", "auroc"],
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        accelerator=FakeAccelerator(),
    )
    model.eval()
    with torch.no_grad():
        metrics = evaluator.evaluate(
            model=model,
            data_loader=loader,
            device=torch.device("cpu"),
            prefix=None,
        )
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "auroc" in metrics
    assert "loss" in metrics


def test_compute_metrics_characterization_single_class_auc_and_unknown_metric() -> None:
    evaluator = Evaluator(
        metrics=["auroc", "auprc", "accuracy", "unknown_metric"],
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        accelerator=FakeAccelerator(),
    )
    labels = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    probabilities = torch.tensor([0.9, 0.8, 0.4, 0.2], dtype=torch.float32)

    metrics = evaluator._compute_metrics(labels=labels, probabilities=probabilities)

    assert metrics["auroc"] == 0.0
    assert metrics["auprc"] == 0.0
    assert metrics["accuracy"] == 0.5
    assert "unknown_metric" not in metrics


def test_compute_metrics_characterization_binary_metric_values() -> None:
    evaluator = Evaluator(
        metrics=["accuracy", "sensitivity", "specificity", "precision", "recall", "f1", "mcc"],
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        accelerator=FakeAccelerator(),
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    probabilities = torch.tensor([0.1, 0.6, 0.9, 0.2], dtype=torch.float32)

    metrics = evaluator._compute_metrics(labels=labels, probabilities=probabilities)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["mcc"] == pytest.approx(0.0)


def test_compute_metrics_respects_custom_decision_threshold() -> None:
    evaluator = Evaluator(
        metrics=["accuracy", "precision", "recall", "f1"],
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        accelerator=FakeAccelerator(),
        decision_threshold=0.3,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    probabilities = torch.tensor([0.1, 0.4, 0.45, 0.8], dtype=torch.float32)

    metrics = evaluator._compute_metrics(labels=labels, probabilities=probabilities)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)


def test_best_f1_threshold_prefers_validation_optimum() -> None:
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    probabilities = torch.tensor([0.1, 0.4, 0.45, 0.8], dtype=torch.float32)

    threshold = Evaluator.best_f1_threshold(labels=labels, probabilities=probabilities)

    assert threshold == pytest.approx(0.45)


def test_select_best_f1_threshold_disables_grad_tracking() -> None:
    model = GradModeProbeModel()
    loader = DataLoader(TinyDataset(), batch_size=2, shuffle=False, collate_fn=_collate)
    evaluator = Evaluator(
        metrics=["f1"],
        loss_config=LossConfig(loss_type="bce_with_logits", pos_weight=1.0, label_smoothing=0.0),
        accelerator=FakeAccelerator(),
    )

    threshold = evaluator.select_best_f1_threshold(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
    )

    assert 0.0 <= threshold <= 1.0
    assert model.grad_enabled_states
    assert model.grad_enabled_states == [False, False]
