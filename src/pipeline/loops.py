"""Shared accelerator-aware loop helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import nn

from src.pipeline.runtime import AcceleratorLike

BatchValue = object
BatchInput = Mapping[str, BatchValue]
BatchDict = dict[str, BatchValue]


def move_batch_to_device(
    batch: BatchInput,
    device: torch.device,
) -> BatchDict:
    """Move tensor fields to the target device while preserving non-tensors."""
    moved_batch: BatchDict = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device, non_blocking=non_blocking)
        else:
            moved_batch[key] = value
    return moved_batch


def forward_model(
    model: nn.Module,
    batch: BatchInput,
) -> dict[str, torch.Tensor]:
    """Execute one forward pass using the repository model contract."""
    try:
        output = model(**batch)
    except TypeError:
        output = model(batch=batch)
    if not isinstance(output, dict):
        raise ValueError("Model forward output must be a dictionary")
    return cast(dict[str, torch.Tensor], output)


def reduce_scalar_mapping(
    accelerator: AcceleratorLike,
    values: Mapping[str, float],
    *,
    device: torch.device,
    reduction: str = "mean",
) -> dict[str, float]:
    """Reduce a scalar mapping across processes and return local floats."""
    if not values:
        return {}
    keys = list(values)
    tensor = torch.tensor(
        [float(values[key]) for key in keys],
        dtype=torch.float64,
        device=device,
    )
    reduced = accelerator.reduce(tensor, reduction=reduction)
    return {key: float(reduced[index].item()) for index, key in enumerate(keys)}


def gather_indexed_predictions(
    accelerator: AcceleratorLike,
    *,
    indices: list[int],
    predictions: list[int],
    total_records: int,
) -> list[int]:
    """Gather indexed predictions across processes and restore original order."""
    if len(indices) != len(predictions):
        raise ValueError("indices and predictions must have matching lengths")
    if not getattr(accelerator, "use_distributed", False):
        return [int(prediction) for prediction in predictions]
    gathered_indices = (
        accelerator.gather_for_metrics(
            torch.tensor(indices, dtype=torch.long, device=accelerator.device)
        )
        .detach()
        .cpu()
    )
    gathered_predictions = (
        accelerator.gather_for_metrics(
            torch.tensor(predictions, dtype=torch.long, device=accelerator.device)
        )
        .detach()
        .cpu()
    )
    ordered: list[int | None] = [None] * total_records
    for index, prediction in zip(
        gathered_indices.tolist(),
        gathered_predictions.tolist(),
        strict=True,
    ):
        ordered[int(index)] = int(prediction)
    missing = [index for index, prediction in enumerate(ordered) if prediction is None]
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        raise ValueError(f"Missing topology predictions for indices: {preview}")
    return [int(prediction) for prediction in ordered if prediction is not None]


__all__ = [
    "BatchDict",
    "BatchInput",
    "BatchValue",
    "forward_model",
    "gather_indexed_predictions",
    "move_batch_to_device",
    "reduce_scalar_mapping",
]
