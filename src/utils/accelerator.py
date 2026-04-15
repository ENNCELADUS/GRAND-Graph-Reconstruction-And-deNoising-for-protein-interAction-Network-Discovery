"""Helpers for Hugging Face Accelerate runtime integration."""

from __future__ import annotations

from contextlib import AbstractContextManager
from os import PathLike
from typing import BinaryIO, Protocol, runtime_checkable

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs

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

    def gather_for_metrics(self, value: torch.Tensor) -> torch.Tensor:
        """Gather tensors for metric computation."""

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        """Gather tensors across processes."""

    def pad_across_processes(
        self,
        value: torch.Tensor,
        dim: int = 0,
        pad_index: int = 0,
        pad_first: bool = False,
    ) -> torch.Tensor:
        """Pad one tensor to a common size across processes."""

    def reduce(self, value: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        """Reduce one tensor across processes."""

    def wait_for_everyone(self) -> None:
        """Synchronize all processes."""

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return the unwrapped model."""

    def save(
        self,
        obj: object,
        f: str | PathLike[str] | BinaryIO,
        safe_serialization: bool = False,
    ) -> None:
        """Persist one object once per machine."""

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
    dataloader_config = DataLoaderConfiguration(non_blocking=True)
    return Accelerator(
        cpu=requested_device.lower() == "cpu",
        mixed_precision=mixed_precision,
        dataloader_config=dataloader_config,
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
    "build_accelerator",
    "distributed_context_from_accelerator",
]
