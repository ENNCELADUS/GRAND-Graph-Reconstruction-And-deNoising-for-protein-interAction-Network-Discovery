"""Helpers for Hugging Face Accelerate runtime integration."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from src.utils.distributed import DistributedContext


@runtime_checkable
class AcceleratorLike(Protocol):
    """Minimal accelerator protocol used across the training pipeline."""

    device: torch.device
    is_main_process: bool
    use_distributed: bool
    process_index: int
    local_process_index: int
    num_processes: int
    mixed_precision: str

    def prepare(self, *components: object) -> object:
        """Prepare models, optimizers, schedulers, or loaders."""

    def autocast(self) -> AbstractContextManager[object]:
        """Return autocast context manager."""

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate one loss tensor."""

    def wait_for_everyone(self) -> None:
        """Synchronize all processes."""

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return the unwrapped model."""

    def save(self, obj: object, f: Any, safe_serialization: bool = False) -> None:
        """Persist one object once per machine."""


class LocalAccelerator:
    """Small local fallback used when no shared accelerator is injected."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.is_main_process = True
        self.use_distributed = False
        self.process_index = 0
        self.local_process_index = 0
        self.num_processes = 1
        self.mixed_precision = "no"

    def prepare(self, *components: object) -> object:
        if len(components) == 1:
            return components[0]
        return components

    def autocast(self) -> AbstractContextManager[object]:
        return torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda")

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def wait_for_everyone(self) -> None:
        return None

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def save(self, obj: object, f: Any, safe_serialization: bool = False) -> None:
        del safe_serialization
        torch.save(obj, f)


def build_accelerator(
    *,
    requested_device: str,
    ddp_enabled: bool,
    use_mixed_precision: bool,
    find_unused_parameters: bool,
) -> Accelerator:
    """Create the shared Accelerator runtime for the pipeline."""
    del ddp_enabled
    mixed_precision = "fp16" if use_mixed_precision and torch.cuda.is_available() else "no"
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=find_unused_parameters,
    )
    return Accelerator(
        cpu=requested_device.lower() == "cpu",
        mixed_precision=mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )


def distributed_context_from_accelerator(
    *,
    accelerator: AcceleratorLike,
    ddp_enabled: bool,
) -> DistributedContext:
    """Project accelerator process metadata onto the repository context type."""
    return DistributedContext(
        ddp_enabled=ddp_enabled,
        is_distributed=accelerator.use_distributed,
        rank=accelerator.process_index,
        local_rank=accelerator.local_process_index,
        world_size=accelerator.num_processes,
        owns_process_group=False,
    )


__all__ = [
    "AcceleratorLike",
    "LocalAccelerator",
    "build_accelerator",
    "distributed_context_from_accelerator",
]
