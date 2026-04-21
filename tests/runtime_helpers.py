"""Shared test helpers for runtime-owned pipeline stages."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from src.pipeline.config import PipelineConfig
from src.pipeline.runtime import AcceleratorLike, DistributedContext, PipelineRuntime
from src.utils.config import ConfigDict


class NoOpAccelerator:
    """Small single-process accelerator stub for stage and loop tests."""

    def __init__(
        self,
        *,
        distributed: DistributedContext | None = None,
        device: torch.device | None = None,
    ) -> None:
        context = distributed or DistributedContext(ddp_enabled=False, is_distributed=False)
        self.device = device or torch.device("cpu")
        self.is_main_process = context.is_main_process
        self.use_distributed = context.is_distributed
        self.process_index = context.rank
        self.local_process_index = context.local_rank
        self.num_processes = context.world_size
        self.mixed_precision = "no"
        self._gradient_accumulation_steps = 1
        self.gradient_accumulation_steps_history = [1]
        self.sync_gradients = True
        self.prepare_calls = 0
        self.autocast_calls = 0
        self.backward_calls = 0
        self.gather_for_metrics_calls = 0
        self.reduce_calls = 0
        self.accumulate_calls = 0
        self.accumulate_steps_seen: list[int] = []
        self.no_sync_calls = 0
        self.step = 0

    @property
    def gradient_accumulation_steps(self) -> int:
        """Return the current accumulation window size."""
        return self._gradient_accumulation_steps

    @gradient_accumulation_steps.setter
    def gradient_accumulation_steps(self, value: int) -> None:
        """Track accumulation window changes like Accelerate does."""
        self._gradient_accumulation_steps = int(value)
        self.gradient_accumulation_steps_history.append(self._gradient_accumulation_steps)

    def prepare(self, *components: object) -> object:
        """Return components unchanged while matching accelerate's single-item behavior."""
        self.prepare_calls += 1
        if len(components) == 1:
            return components[0]
        return components

    def autocast(self) -> object:
        """Return a no-op autocast context."""
        self.autocast_calls += 1
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate a loss tensor."""
        self.backward_calls += 1
        loss.backward()

    def accumulate(self, *models: torch.nn.Module) -> object:
        """Return a no-op accumulation context with sync boundary tracking."""
        del models
        self.accumulate_calls += 1
        self.step += 1
        steps = max(1, int(self.gradient_accumulation_steps))
        self.accumulate_steps_seen.append(steps)
        self.sync_gradients = self.step % steps == 0
        return nullcontext()

    def no_sync(self, model: torch.nn.Module) -> object:
        """Return a no-op no_sync context."""
        del model
        self.no_sync_calls += 1
        return nullcontext()

    def gather_for_metrics(self, value: torch.Tensor) -> torch.Tensor:
        """Return metric tensors unchanged."""
        self.gather_for_metrics_calls += 1
        return value

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        """Return gathered tensors unchanged."""
        return value

    def pad_across_processes(
        self,
        value: torch.Tensor,
        dim: int = 0,
        pad_index: int = 0,
        pad_first: bool = False,
    ) -> torch.Tensor:
        """Return tensors unchanged."""
        del dim, pad_index, pad_first
        return value

    def reduce(self, value: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        """Return reduced tensors unchanged."""
        del reduction
        self.reduce_calls += 1
        return value

    def wait_for_everyone(self) -> None:
        """No-op synchronization."""
        return None

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return the model unchanged."""
        return model

    def save(
        self,
        obj: object,
        f: str | Path,
        safe_serialization: bool = False,
    ) -> None:
        """Save an object with torch serialization."""
        del safe_serialization
        torch.save(obj, f)


def build_stage_runtime(
    config: ConfigDict,
    *,
    stage_run_ids: dict[str, str] | None = None,
    distributed: DistributedContext | None = None,
    accelerator: AcceleratorLike | None = None,
    device: torch.device | None = None,
) -> PipelineRuntime:
    """Build a minimal ``PipelineRuntime`` for stage tests."""
    context = distributed or DistributedContext(ddp_enabled=False, is_distributed=False)
    stage_ids = {
        "train": "train_run",
        "topology_finetune": "topology_ft_run",
        "adapt": "adapt_run",
        "evaluate": "eval_run",
        "topology_evaluate": "topology_eval_run",
    }
    if stage_run_ids is not None:
        stage_ids.update(stage_run_ids)
    runtime_accelerator = accelerator or NoOpAccelerator(distributed=context, device=device)
    return PipelineRuntime(
        config=PipelineConfig.from_dict(config),
        accelerator=runtime_accelerator,
        device=runtime_accelerator.device,
        distributed=context,
        stage_run_ids=stage_ids,
    )
