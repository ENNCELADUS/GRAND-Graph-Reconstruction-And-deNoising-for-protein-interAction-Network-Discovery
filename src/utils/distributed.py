"""Distributed helpers for DDP-capable orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Process metadata for distributed execution.

    Attributes:
        ddp_enabled: Whether DDP was requested in config.
        is_distributed: Whether process-group is initialized.
        rank: Global process rank.
        local_rank: Local process rank on node.
        world_size: Number of processes in the job.
    """

    ddp_enabled: bool
    is_distributed: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    owns_process_group: bool = False

    @property
    def is_main_process(self) -> bool:
        """Return whether this process is the main rank."""
        return self.rank == 0


def distributed_barrier(context: DistributedContext) -> None:
    """Synchronize processes when distributed mode is active.

    Args:
        context: Distributed process metadata.
    """
    if context.is_distributed and dist.is_initialized():
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[context.local_rank])
            return
        dist.barrier()


__all__ = [
    "DistributedContext",
    "distributed_barrier",
    "dist",
    "torch",
]
