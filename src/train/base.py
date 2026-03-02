"""Generic Trainer implementation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from torch.utils.data import DataLoader

from src.train.config import LossConfig, OptimizerConfig, SchedulerConfig
from src.train.strategies.ohem import OHEMSampleStrategy
from src.utils.losses import binary_classification_loss

BatchValue = object
BatchInput = Mapping[str, BatchValue]
BatchDict = dict[str, BatchValue]


class Trainer:
    """Mechanics-only training class for one epoch execution.

    This class handles forward/backward passes, optimizer/scheduler stepping,
    and optional AMP/OHEM behavior. It deliberately excludes orchestration
    concerns like checkpointing and stage transitions.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig,
        loss_config: LossConfig,
        use_amp: bool,
        total_epochs: int,
        steps_per_epoch: int,
        ohem_strategy: OHEMSampleStrategy | None = None,
        logger: logging.Logger | None = None,
        heartbeat_every_n_steps: int = 0,
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.loss_config = loss_config
        self.use_amp = use_amp and device.type == "cuda"
        self.total_epochs = total_epochs
        self.steps_per_epoch = max(1, steps_per_epoch)
        self.ohem_strategy = ohem_strategy
        self.logger = logger
        self.heartbeat_every_n_steps = max(0, int(heartbeat_every_n_steps))
        self.scaler = torch.amp.GradScaler(  # type: ignore[attr-defined]
            "cuda",
            enabled=self.use_amp,
        )
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

    def _trainable_parameters(self) -> list[nn.Parameter]:
        return [param for param in self.model.parameters() if param.requires_grad]

    def _build_optimizer(self) -> Optimizer:
        optimizer_type = self.optimizer_config.optimizer_type.lower()
        params = self._trainable_parameters()
        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                params=params,
                lr=self.optimizer_config.lr,
                betas=(self.optimizer_config.beta1, self.optimizer_config.beta2),
                eps=self.optimizer_config.eps,
                weight_decay=self.optimizer_config.weight_decay,
            )
        if optimizer_type == "sgd":
            return torch.optim.SGD(
                params=params,
                lr=self.optimizer_config.lr,
                weight_decay=self.optimizer_config.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer type: {self.optimizer_config.optimizer_type}")

    def _build_scheduler(self) -> LRScheduler | None:
        scheduler_type = self.scheduler_config.scheduler_type.lower()
        if scheduler_type in {"none", "disabled"}:
            return None
        if scheduler_type == "onecycle":
            return OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.scheduler_config.max_lr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.total_epochs,
                pct_start=self.scheduler_config.pct_start,
                div_factor=self.scheduler_config.div_factor,
                final_div_factor=self.scheduler_config.final_div_factor,
                anneal_strategy=self.scheduler_config.anneal_strategy,
            )
        raise ValueError(f"Unsupported scheduler type: {self.scheduler_config.scheduler_type}")

    def rebuild_optimizer_and_scheduler(self) -> None:
        """Rebuild optimizer and scheduler after trainable params change."""
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

    @staticmethod
    def _required_batch_tensor(
        batch: BatchInput,
        key: str,
    ) -> torch.Tensor:
        """Return required tensor batch value."""
        value = batch.get(key)
        if value is None:
            raise KeyError(f"Batch is missing required tensor field '{key}'")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Batch field '{key}' must be a torch.Tensor")
        return value

    @staticmethod
    def _optional_tensor(value: BatchValue) -> torch.Tensor | None:
        """Return tensor value or ``None`` when value is not tensor-like."""
        return value if isinstance(value, torch.Tensor) else None

    def _move_batch_to_device(self, batch: BatchInput) -> BatchDict:
        """Move tensor fields to target device while preserving non-tensor fields."""
        moved_batch: BatchDict = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value
        return moved_batch

    def _forward_model(self, batch: BatchInput) -> dict[str, torch.Tensor]:
        try:
            output = self.model(**batch)
        except TypeError:
            output = self.model(batch=batch)
        if not isinstance(output, dict):
            raise ValueError("Model forward output must be a dictionary")
        return output

    def _select_loss(
        self,
        output: dict[str, torch.Tensor],
        batch: BatchInput,
        epoch_index: int,
    ) -> torch.Tensor:
        logits = output["logits"]
        labels = self._required_batch_tensor(batch, "label").float()
        loss = binary_classification_loss(
            logits=logits,
            labels=labels,
            loss_config=self.loss_config,
            reduction="mean",
        )

        if self.ohem_strategy is None:
            return loss
        if epoch_index < self.ohem_strategy.warmup_epochs:
            return loss

        mining_loss_config = LossConfig(
            loss_type=self.loss_config.loss_type,
            pos_weight=1.0,
            label_smoothing=0.0,
        )
        with torch.no_grad():
            mining_loss = binary_classification_loss(
                logits=logits,
                labels=labels,
                loss_config=mining_loss_config,
                reduction="none",
            )
            selected_indices = self.ohem_strategy.select(
                losses=mining_loss,
                epoch_index=epoch_index,
                protein_a_ids=self._optional_tensor(batch.get("protein_a_id")),
                protein_b_ids=self._optional_tensor(batch.get("protein_b_id")),
            )

        ohem_loss_config = LossConfig(
            loss_type=self.loss_config.loss_type,
            pos_weight=1.0,
            label_smoothing=self.loss_config.label_smoothing,
        )
        per_sample = binary_classification_loss(
            logits=logits,
            labels=labels,
            loss_config=ohem_loss_config,
            reduction="none",
        )
        return per_sample[selected_indices].mean()

    def _is_ohem_mining_active(self, epoch_index: int) -> bool:
        """Return whether staged OHEM mining is active for this epoch."""
        if self.ohem_strategy is None:
            return False
        return epoch_index >= self.ohem_strategy.warmup_epochs

    def _compute_ohem_selected_indices(
        self,
        batch: BatchInput,
        epoch_index: int,
    ) -> torch.Tensor:
        """Run read-only scoring on the candidate pool and select hard examples."""
        if self.ohem_strategy is None:
            raise ValueError("OHEM strategy is not configured")
        batch_size = int(self._required_batch_tensor(batch, "label").size(0))
        chunk_size = max(1, min(batch_size, self.ohem_strategy.target_batch_size))
        mining_loss_config = LossConfig(
            loss_type=self.loss_config.loss_type,
            pos_weight=1.0,
            label_smoothing=0.0,
        )
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ),
        ):
            chunk_losses: list[torch.Tensor] = []
            for chunk_start in range(0, batch_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, batch_size)
                chunk_batch = self._slice_batch_rows(
                    batch=batch,
                    start=chunk_start,
                    end=chunk_end,
                )
                chunk_output = self._forward_model(chunk_batch)
                chunk_logits = chunk_output["logits"]
                chunk_loss = binary_classification_loss(
                    logits=chunk_logits,
                    labels=self._required_batch_tensor(chunk_batch, "label").float(),
                    loss_config=mining_loss_config,
                    reduction="none",
                )
                chunk_losses.append(chunk_loss)
            mining_loss = torch.cat(chunk_losses, dim=0)
            selected_indices = self.ohem_strategy.select(
                losses=mining_loss,
                epoch_index=epoch_index,
                protein_a_ids=self._optional_tensor(batch.get("protein_a_id")),
                protein_b_ids=self._optional_tensor(batch.get("protein_b_id")),
            )
        del mining_loss
        return selected_indices

    def _select_batch_rows(
        self,
        batch: BatchInput,
        selected_indices: torch.Tensor,
    ) -> BatchDict:
        """Select rows aligned to the sample axis from a batch dictionary."""
        selected_batch: BatchDict = {}
        label = self._required_batch_tensor(batch, "label")
        batch_size = int(label.size(0))
        selected_indices_list = selected_indices.detach().cpu().tolist()
        for key, value in batch.items():
            is_row_tensor = (
                isinstance(value, torch.Tensor)
                and value.dim() > 0
                and int(value.size(0)) == batch_size
            )
            if is_row_tensor:
                tensor_value = cast(torch.Tensor, value)
                selected_batch[key] = tensor_value.index_select(0, selected_indices)
            elif isinstance(value, list) and len(value) == batch_size:
                selected_batch[key] = [value[index] for index in selected_indices_list]
            elif isinstance(value, tuple) and len(value) == batch_size:
                selected_batch[key] = tuple(value[index] for index in selected_indices_list)
            else:
                selected_batch[key] = value
        return selected_batch

    def _slice_batch_rows(
        self,
        batch: BatchInput,
        start: int,
        end: int,
    ) -> BatchDict:
        """Slice row-aligned batch fields across a half-open sample window."""
        sliced_batch: BatchDict = {}
        label = self._required_batch_tensor(batch, "label")
        batch_size = int(label.size(0))
        for key, value in batch.items():
            is_row_tensor = (
                isinstance(value, torch.Tensor)
                and value.dim() > 0
                and int(value.size(0)) == batch_size
            )
            if is_row_tensor:
                tensor_value = cast(torch.Tensor, value)
                sliced_batch[key] = tensor_value[start:end]
            elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                sliced_batch[key] = value[start:end]
            else:
                sliced_batch[key] = value
        return sliced_batch

    def _ohem_selected_batch_loss(
        self,
        output: dict[str, torch.Tensor],
        batch: BatchInput,
    ) -> torch.Tensor:
        """Compute training loss for already-selected OHEM hard examples."""
        ohem_loss_config = LossConfig(
            loss_type=self.loss_config.loss_type,
            pos_weight=1.0,
            label_smoothing=self.loss_config.label_smoothing,
        )
        return binary_classification_loss(
            logits=output["logits"],
            labels=self._required_batch_tensor(batch, "label").float(),
            loss_config=ohem_loss_config,
            reduction="mean",
        )

    def train_one_epoch(
        self,
        train_loader: DataLoader[Mapping[str, object]],
        epoch_index: int = 0,
    ) -> dict[str, float]:
        """Run one full training epoch.

        Args:
            train_loader: Training data loader.
            epoch_index: Zero-based epoch index used for heartbeat log messages.

        Returns:
            Aggregate epoch metrics, including average loss and learning rate.
        """
        self.model.train()
        running_loss = 0.0
        batch_count = 0
        total_steps = max(1, len(train_loader))

        for batch in train_loader:
            batch_count += 1
            prepared_batch = self._move_batch_to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            if self._is_ohem_mining_active(epoch_index):
                selected_indices = self._compute_ohem_selected_indices(
                    batch=prepared_batch,
                    epoch_index=epoch_index,
                )
                selected_batch = self._select_batch_rows(
                    batch=prepared_batch,
                    selected_indices=selected_indices,
                )
                del prepared_batch
                del selected_indices
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    output = self._forward_model(selected_batch)
                    loss = self._ohem_selected_batch_loss(output=output, batch=selected_batch)
            else:
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    output = self._forward_model(prepared_batch)
                    loss = self._select_loss(
                        output=output,
                        batch=prepared_batch,
                        epoch_index=epoch_index,
                    )

            if self.use_amp:
                scaled_loss = self.scaler.scale(loss)
                torch.autograd.backward(scaled_loss)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.autograd.backward(loss)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += float(loss.detach().item())
            if self._should_log_heartbeat(step=batch_count, total_steps=total_steps):
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                if self.logger is not None:
                    self.logger.info(
                        "Epoch %d | Step %d/%d | Loss %.4f | LR %.4e",
                        epoch_index + 1,
                        batch_count,
                        total_steps,
                        running_loss / batch_count,
                        current_lr,
                    )

        average_loss = running_loss / max(1, batch_count)
        current_lr = float(self.optimizer.param_groups[0]["lr"])
        return {"loss": average_loss, "lr": current_lr}

    def _should_log_heartbeat(self, step: int, total_steps: int) -> bool:
        """Return whether heartbeat logs should be emitted for this step."""
        if self.logger is None:
            return False
        if step == 1 or step == total_steps:
            return True
        return self.heartbeat_every_n_steps > 0 and step % self.heartbeat_every_n_steps == 0


__all__ = [
    "BatchDict",
    "BatchInput",
    "BatchValue",
    "LossConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "Trainer",
]
