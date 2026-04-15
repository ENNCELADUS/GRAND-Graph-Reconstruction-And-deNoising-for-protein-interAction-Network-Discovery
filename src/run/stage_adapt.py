"""SHOT domain-adaptation stage execution helpers."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader

from src.adapt import (
    DomainAdaptationConfig,
    OutputHeadFeatureHook,
    assign_pseudo_labels,
    compute_centroids,
    diversity_loss,
    entropy_loss,
    logits_to_probabilities,
    parse_domain_adaptation_config,
    pseudo_label_loss,
)
from src.pipeline.loops import ensure_accelerator, forward_model, move_batch_to_device
from src.run.stage_train import _build_stage_runtime, _load_checkpoint, _save_checkpoint
from src.utils.accelerator import AcceleratorLike
from src.utils.config import ConfigDict, as_bool, as_int, extract_model_kwargs, get_section
from src.utils.distributed import DistributedContext
from src.utils.logging import append_csv_row, log_stage_event

BatchValue = object
BatchInput = Mapping[str, BatchValue]

ADAPT_CSV_COLUMNS = [
    "Epoch",
    "Epoch Time",
    "Entropy Loss",
    "Diversity Loss",
    "Pseudo CE Loss",
    "Total Loss",
    "Learning Rate",
]


def _centroid_accumulation_dtype(probs: torch.Tensor, features: torch.Tensor) -> torch.dtype:
    """Choose a stable dtype for centroid accumulation under mixed precision."""
    dtype = torch.promote_types(probs.dtype, features.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _move_batch_to_device(batch: BatchInput, device: torch.device) -> dict[str, BatchValue]:
    """Move tensor fields to target device while preserving non-tensor fields."""
    return move_batch_to_device(batch, device)


def _forward_batch_without_labels(batch: BatchInput) -> dict[str, BatchValue]:
    """Return model-input batch with labels removed for unlabeled SHOT adaptation."""
    return {key: value for key, value in batch.items() if key != "label"}


def _forward_model(model: nn.Module, batch: BatchInput) -> dict[str, torch.Tensor]:
    """Execute model forward and validate output contract."""
    return forward_model(model, batch)


def _infer_batch_size(batch: BatchInput) -> int:
    """Infer sample count from one batch payload."""
    label = batch.get("label")
    if isinstance(label, torch.Tensor) and label.dim() > 0:
        return int(label.size(0))
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            return int(value.size(0))
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("Unable to infer batch size from target batch")


def _set_loader_epoch(loader: DataLoader[dict[str, object]], epoch: int) -> None:
    """Set deterministic epoch seed on distributed samplers when available."""
    set_epoch = getattr(loader.sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)


def _build_target_loaders(
    config: ConfigDict,
    dataloaders: dict[str, DataLoader[dict[str, object]]],
    adaptation_config: DomainAdaptationConfig,
    distributed_context: DistributedContext,
) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]]]:
    """Build deterministic target loaders for SHOT pseudo labeling and optimization."""
    del distributed_context
    target_loader = dataloaders[adaptation_config.target_split]
    dataset = target_loader.dataset
    collate_fn = target_loader.collate_fn

    training_cfg = get_section(config, "training_config")
    data_cfg = get_section(config, "data_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")

    batch_size = as_int(training_cfg.get("batch_size", 8), "training_config.batch_size")
    num_workers = as_int(
        dataloader_cfg.get("num_workers", 0),
        "data_config.dataloader.num_workers",
    )
    pin_memory = as_bool(
        dataloader_cfg.get("pin_memory", False),
        "data_config.dataloader.pin_memory",
    )

    eval_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return eval_loader, train_loader


def _prepare_shot_optimizer_params(
    model: nn.Module,
    prefixes: Sequence[str],
    *,
    preserve_frozen_requires_grad: bool,
) -> tuple[list[nn.Parameter], int]:
    """Return optimizer parameters for SHOT while honoring frozen prefixes.

    Under DDP, changing ``requires_grad`` after wrapping can break reducer bucket
    bookkeeping. In that case we exclude frozen-prefix parameters from the
    optimizer but preserve their original ``requires_grad`` setting.
    """
    normalized_prefixes = tuple(prefixes)
    if not normalized_prefixes:
        raise ValueError("freeze prefixes must not be empty")

    def _matches(name: str) -> bool:
        return any(
            name.startswith(prefix) or name.startswith(f"module.{prefix}")
            for prefix in normalized_prefixes
        )

    optimizer_params: list[nn.Parameter] = []
    trainable_count = 0
    for name, parameter in model.named_parameters():
        if _matches(name):
            if not preserve_frozen_requires_grad:
                parameter.requires_grad = False
            continue
        parameter.requires_grad = True
        optimizer_params.append(parameter)
        trainable_count += int(parameter.numel())

    if trainable_count <= 0:
        raise ValueError("SHOT produced zero trainable parameters after applying freeze_prefixes")
    return optimizer_params, trainable_count


def _build_optimizer(
    params: Sequence[nn.Parameter],
    adaptation_config: DomainAdaptationConfig,
) -> Optimizer:
    """Build SHOT SGD optimizer for the selected trainable parameters."""
    return torch.optim.SGD(
        params=list(params),
        lr=adaptation_config.optimizer.lr,
        momentum=adaptation_config.optimizer.momentum,
        weight_decay=adaptation_config.optimizer.weight_decay,
    )


def _build_scheduler(
    optimizer: Optimizer,
    adaptation_config: DomainAdaptationConfig,
    total_steps: int,
) -> LRScheduler | None:
    """Build SHOT learning-rate scheduler."""
    scheduler_type = adaptation_config.scheduler.scheduler_type
    if scheduler_type == "none":
        return None

    gamma = adaptation_config.scheduler.gamma
    power = adaptation_config.scheduler.power

    def _lambda(step: int) -> float:
        progress = float(step) / float(max(1, total_steps))
        return (1.0 + gamma * progress) ** (-power)

    return LambdaLR(optimizer=optimizer, lr_lambda=_lambda)


def _compute_global_centroids(
    model: nn.Module,
    loader: DataLoader[dict[str, object]],
    feature_hook: OutputHeadFeatureHook,
    device: torch.device,
    use_amp: bool,
    adaptation_config: DomainAdaptationConfig,
    accelerator: AcceleratorLike,
) -> torch.Tensor:
    """Compute global SHOT centroids from target features and class probabilities."""
    del use_amp
    class_masses: torch.Tensor | None = None
    feature_sums: torch.Tensor | None = None

    model.eval()
    with torch.no_grad():
        for batch in loader:
            prepared_batch = _move_batch_to_device(batch=batch, device=device)
            forward_batch = _forward_batch_without_labels(prepared_batch)
            with accelerator.autocast():
                output = _forward_model(model=model, batch=forward_batch)
            logits = output["logits"]
            probs = logits_to_probabilities(logits=logits, epsilon=adaptation_config.epsilon)
            features = feature_hook.pop()
            if features.size(0) != probs.size(0):
                raise RuntimeError("SHOT feature/probability batch size mismatch")

            accumulation_dtype = _centroid_accumulation_dtype(probs=probs, features=features)
            probs = probs.to(dtype=accumulation_dtype)
            features = features.to(dtype=accumulation_dtype)
            batch_feature_sums = probs.transpose(0, 1) @ features
            batch_class_masses = probs.sum(dim=0)

            if feature_sums is None:
                feature_sums = batch_feature_sums
                class_masses = batch_class_masses
            else:
                feature_sums = feature_sums + batch_feature_sums
                class_masses = class_masses + batch_class_masses

    if feature_sums is None or class_masses is None:
        raise ValueError("SHOT target loader produced no batches while computing centroids")

    feature_sums = accelerator.reduce(feature_sums, reduction="sum")
    class_masses = accelerator.reduce(class_masses, reduction="sum")
    return compute_centroids(
        feature_sums=feature_sums,
        class_masses=class_masses,
        epsilon=adaptation_config.epsilon,
    )


def _compute_pseudo_labels(
    model: nn.Module,
    loader: DataLoader[dict[str, object]],
    feature_hook: OutputHeadFeatureHook,
    centroids: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    adaptation_config: DomainAdaptationConfig,
    accelerator: AcceleratorLike,
) -> torch.Tensor:
    """Compute fixed pseudo labels for one epoch in deterministic loader order."""
    del use_amp
    pseudo_labels: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            prepared_batch = _move_batch_to_device(batch=batch, device=device)
            forward_batch = _forward_batch_without_labels(prepared_batch)
            with accelerator.autocast():
                _forward_model(model=model, batch=forward_batch)
            features = feature_hook.pop()
            labels = assign_pseudo_labels(
                features=features,
                centroids=centroids,
                epsilon=adaptation_config.epsilon,
            )
            pseudo_labels.append(labels.detach().cpu())

    if not pseudo_labels:
        raise ValueError("SHOT target loader produced no batches while computing pseudo labels")
    return torch.cat(pseudo_labels, dim=0)


def _train_one_shot_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, object]],
    pseudo_labels: torch.Tensor,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    accelerator: AcceleratorLike,
    device: torch.device,
    use_amp: bool,
    adaptation_config: DomainAdaptationConfig,
    distributed_context: DistributedContext,
) -> dict[str, float]:
    """Run one SHOT optimization epoch on target data."""
    del use_amp, distributed_context
    model.train()
    pseudo_offset = 0
    step_count = 0

    entropy_sum = 0.0
    diversity_sum = 0.0
    pseudo_ce_sum = 0.0
    total_loss_sum = 0.0

    for batch in loader:
        batch_size = _infer_batch_size(batch)
        next_offset = pseudo_offset + batch_size
        if next_offset > pseudo_labels.numel():
            raise RuntimeError("Pseudo label index overflow during SHOT optimization")
        batch_pseudo = pseudo_labels[pseudo_offset:next_offset].to(device=device)
        pseudo_offset = next_offset

        prepared_batch = _move_batch_to_device(batch=batch, device=device)
        forward_batch = _forward_batch_without_labels(prepared_batch)

        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            output = _forward_model(model=model, batch=forward_batch)
            logits = output["logits"]
            probs = logits_to_probabilities(logits=logits, epsilon=adaptation_config.epsilon)

            entropy_value = entropy_loss(probabilities=probs, epsilon=adaptation_config.epsilon)
            diversity_value = diversity_loss(
                probabilities=probs,
                epsilon=adaptation_config.epsilon,
            )
            pseudo_ce_value = pseudo_label_loss(logits=logits, pseudo_labels=batch_pseudo)
            total_loss = (
                adaptation_config.entropy_weight * entropy_value
                + adaptation_config.diversity_weight * diversity_value
                + adaptation_config.beta * pseudo_ce_value
            )

        accelerator.backward(total_loss)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        step_count += 1
        entropy_sum += float(entropy_value.detach().item())
        diversity_sum += float(diversity_value.detach().item())
        pseudo_ce_sum += float(pseudo_ce_value.detach().item())
        total_loss_sum += float(total_loss.detach().item())

    if step_count <= 0:
        raise ValueError("SHOT optimization loader produced zero steps")

    metrics = torch.tensor(
        [
            entropy_sum / step_count,
            diversity_sum / step_count,
            pseudo_ce_sum / step_count,
            total_loss_sum / step_count,
            float(optimizer.param_groups[0]["lr"]),
        ],
        device=device,
        dtype=torch.float64,
    )
    reduced = accelerator.reduce(metrics, reduction="mean")
    return {
        "entropy_loss": float(reduced[0].item()),
        "diversity_loss": float(reduced[1].item()),
        "pseudo_ce_loss": float(reduced[2].item()),
        "total_loss": float(reduced[3].item()),
        "lr": float(reduced[4].item()),
    }


def run_shot_adaptation_stage(
    config: ConfigDict,
    model: nn.Module,
    device: torch.device,
    dataloaders: dict[str, DataLoader[dict[str, object]]],
    run_id: str,
    checkpoint_path: Path,
    distributed_context: DistributedContext,
    accelerator: AcceleratorLike | None = None,
) -> Path:
    """Run SHOT adaptation from one source checkpoint and return adapted checkpoint path."""
    stage_accelerator = ensure_accelerator(
        accelerator,
        device=device,
        use_mixed_precision=as_bool(
            get_section(config, "device_config").get("use_mixed_precision", False),
            "device_config.use_mixed_precision",
        ),
    )
    adaptation_config = parse_domain_adaptation_config(config)
    if not adaptation_config.enabled or adaptation_config.method != "shot":
        raise ValueError("run_shot_adaptation_stage requires enabled SHOT configuration")

    model_name, _ = extract_model_kwargs(config)
    log_dir, model_dir, logger = _build_stage_runtime(
        model_name=model_name,
        stage="adapt",
        run_id=run_id,
        distributed_context=distributed_context,
    )

    if distributed_context.is_main_process:
        log_stage_event(
            logger,
            "stage_start",
            run_id=run_id,
            checkpoint=checkpoint_path,
            method=adaptation_config.method,
        )

    _load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        accelerator=stage_accelerator,
    )

    if distributed_context.is_main_process:
        log_stage_event(logger, "checkpoint_loaded", path=checkpoint_path)

    optimizer_params, trainable_params = _prepare_shot_optimizer_params(
        model=model,
        prefixes=adaptation_config.freeze_prefixes,
        preserve_frozen_requires_grad=distributed_context.is_distributed,
    )

    eval_loader, train_loader = _build_target_loaders(
        config=config,
        dataloaders=dataloaders,
        adaptation_config=adaptation_config,
        distributed_context=distributed_context,
    )
    steps_per_epoch = len(train_loader)
    if steps_per_epoch <= 0:
        raise ValueError(
            "SHOT train loader has zero steps; increase target data or reduce batch size"
        )

    optimizer = _build_optimizer(params=optimizer_params, adaptation_config=adaptation_config)
    scheduler = _build_scheduler(
        optimizer=optimizer,
        adaptation_config=adaptation_config,
        total_steps=adaptation_config.epochs * steps_per_epoch,
    )
    if scheduler is None:
        (
            stage_model,
            optimizer,
            eval_loader,
            train_loader,
        ) = cast(
            tuple[
                nn.Module,
                Optimizer,
                DataLoader[dict[str, object]],
                DataLoader[dict[str, object]],
            ],
            stage_accelerator.prepare(model, optimizer, eval_loader, train_loader),
        )
    else:
        (
            stage_model,
            optimizer,
            eval_loader,
            train_loader,
            scheduler,
        ) = cast(
            tuple[
                nn.Module,
                Optimizer,
                DataLoader[dict[str, object]],
                DataLoader[dict[str, object]],
                LRScheduler,
            ],
            stage_accelerator.prepare(model, optimizer, eval_loader, train_loader, scheduler),
        )

    csv_path = log_dir / "adapt_step.csv"
    if distributed_context.is_main_process:
        log_stage_event(
            logger,
            "adapt_config",
            epochs=adaptation_config.epochs,
            beta=adaptation_config.beta,
            target_split=adaptation_config.target_split,
            trainable_params=trainable_params,
        )

    device_cfg = get_section(config, "device_config")
    use_amp = as_bool(
        device_cfg.get("use_mixed_precision", False),
        "device_config.use_mixed_precision",
    )
    feature_hook = OutputHeadFeatureHook(model=stage_model)
    try:
        for epoch in range(adaptation_config.epochs):
            _set_loader_epoch(eval_loader, epoch)
            _set_loader_epoch(train_loader, epoch)
            epoch_start = time.perf_counter()

            centroids = _compute_global_centroids(
                model=stage_model,
                loader=eval_loader,
                feature_hook=feature_hook,
                device=device,
                use_amp=use_amp,
                adaptation_config=adaptation_config,
                accelerator=stage_accelerator,
            )
            pseudo_labels = _compute_pseudo_labels(
                model=stage_model,
                loader=eval_loader,
                feature_hook=feature_hook,
                centroids=centroids,
                device=device,
                use_amp=use_amp,
                adaptation_config=adaptation_config,
                accelerator=stage_accelerator,
            )
            epoch_stats = _train_one_shot_epoch(
                model=stage_model,
                loader=train_loader,
                pseudo_labels=pseudo_labels,
                optimizer=optimizer,
                scheduler=scheduler,
                accelerator=stage_accelerator,
                device=device,
                use_amp=use_amp,
                adaptation_config=adaptation_config,
                distributed_context=distributed_context,
            )
            epoch_seconds = time.perf_counter() - epoch_start

            if distributed_context.is_main_process:
                row: dict[str, float | int] = {
                    "Epoch": epoch + 1,
                    "Epoch Time": epoch_seconds,
                    "Entropy Loss": epoch_stats["entropy_loss"],
                    "Diversity Loss": epoch_stats["diversity_loss"],
                    "Pseudo CE Loss": epoch_stats["pseudo_ce_loss"],
                    "Total Loss": epoch_stats["total_loss"],
                    "Learning Rate": epoch_stats["lr"],
                }
                append_csv_row(csv_path=csv_path, row=row, fieldnames=ADAPT_CSV_COLUMNS)
                log_stage_event(
                    logger,
                    "epoch_done",
                    epoch=epoch + 1,
                    time=epoch_seconds,
                    entropy_loss=epoch_stats["entropy_loss"],
                    diversity_loss=epoch_stats["diversity_loss"],
                    pseudo_ce_loss=epoch_stats["pseudo_ce_loss"],
                    total_loss=epoch_stats["total_loss"],
                    lr=epoch_stats["lr"],
                )
            stage_accelerator.wait_for_everyone()
    finally:
        feature_hook.close()

    adapted_checkpoint_path = model_dir / "best_model.pth"
    if distributed_context.is_main_process:
        _save_checkpoint(
            model=stage_model,
            checkpoint_path=adapted_checkpoint_path,
            accelerator=stage_accelerator,
        )
        log_stage_event(logger, "checkpoint_saved", path=adapted_checkpoint_path)
        log_stage_event(logger, "stage_done", run_id=run_id)
    stage_accelerator.wait_for_everyone()
    return adapted_checkpoint_path
