"""Configuration parsing for the TCCIG training stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.optim import Optimizer

from src.topology.losses import TCCIGLossWeights, TopologyLossWeightSchedule
from src.train.config import OptimizerConfig
from src.train.topology import shared as topology_train
from src.utils.config import ConfigDict, as_float, as_int, as_str, get_section


@dataclass(frozen=True)
class TCCIGTrainConfig:
    """Resolved TCCIG training settings."""

    epochs: int
    run_seed: int
    pair_batch_size: int
    gradient_accumulation_steps: int
    subgraphs_per_epoch: int
    node_sizes: tuple[int, ...]
    strategy: str
    negative_ratio: int
    edge_chunk_size: int | None
    compute_clustering_mmd: bool
    internal_validation_inference_batch_size: int
    internal_validation_compute_spectral_stats: bool
    internal_validation_compute_clustering_mmd: bool
    loss_weight_schedule: TopologyLossWeightSchedule
    loss_weights: TCCIGLossWeights
    monitor_metric: str
    early_stopping_patience: int
    optimizer: OptimizerConfig
    teacher_mask_ratio: float
    teacher_negative_ratio: int


def tccig_train_config(config: ConfigDict) -> ConfigDict:
    """Return ``tccig_train`` config with schema validation."""
    train_cfg = config.get("tccig_train", {})
    if not isinstance(train_cfg, dict):
        raise ValueError("tccig_train must be a mapping")
    return cast(ConfigDict, train_cfg)


def teacher_config(train_cfg: ConfigDict) -> ConfigDict:
    """Return optional ``tccig_train.teacher`` config."""
    teacher_cfg = train_cfg.get("teacher", {})
    if teacher_cfg is None:
        return {}
    if not isinstance(teacher_cfg, dict):
        raise ValueError("tccig_train.teacher must be a mapping")
    return cast(ConfigDict, teacher_cfg)


def parse_tccig_loss_weights(train_cfg: ConfigDict) -> TCCIGLossWeights:
    """Parse TCCIG loss weights from ``tccig_train.losses``."""
    loss_cfg = train_cfg.get("losses", {})
    if loss_cfg is None:
        loss_cfg = {}
    if not isinstance(loss_cfg, dict):
        raise ValueError("tccig_train.losses must be a mapping")
    defaults = TCCIGLossWeights()
    return TCCIGLossWeights(
        edge=as_float(loss_cfg.get("edge", defaults.edge), "tccig_train.losses.edge"),
        teacher=as_float(loss_cfg.get("teacher", defaults.teacher), "tccig_train.losses.teacher"),
        budget=as_float(loss_cfg.get("budget", defaults.budget), "tccig_train.losses.budget"),
        density=as_float(loss_cfg.get("density", defaults.density), "tccig_train.losses.density"),
        degree=as_float(loss_cfg.get("degree", defaults.degree), "tccig_train.losses.degree"),
        clustering=as_float(
            loss_cfg.get("clustering", defaults.clustering),
            "tccig_train.losses.clustering",
        ),
        rank=as_float(loss_cfg.get("rank", defaults.rank), "tccig_train.losses.rank"),
        module=as_float(loss_cfg.get("module", defaults.module), "tccig_train.losses.module"),
        spectral=as_float(
            loss_cfg.get("spectral", defaults.spectral),
            "tccig_train.losses.spectral",
        ),
        calibration=as_float(
            loss_cfg.get("calibration", defaults.calibration),
            "tccig_train.losses.calibration",
        ),
        sparse=as_float(loss_cfg.get("sparse", defaults.sparse), "tccig_train.losses.sparse"),
    )


def parse_teacher_training_config(train_cfg: ConfigDict) -> tuple[float, int]:
    """Parse online teacher corruption settings."""
    teacher_cfg = teacher_config(train_cfg)
    mask_ratio = as_float(teacher_cfg.get("mask_ratio", 0.7), "tccig_train.teacher.mask_ratio")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("tccig_train.teacher.mask_ratio must be in (0, 1)")
    negative_ratio = as_int(
        teacher_cfg.get("negative_ratio", 1),
        "tccig_train.teacher.negative_ratio",
    )
    if negative_ratio <= 0:
        raise ValueError("tccig_train.teacher.negative_ratio must be > 0")
    return mask_ratio, negative_ratio


def parse_tccig_train_config(config: ConfigDict) -> TCCIGTrainConfig:
    """Parse the full TCCIG stage config."""
    run_cfg = get_section(config, "run_config")
    training_cfg = get_section(config, "training_config")
    train_cfg = tccig_train_config(config)
    init_mode = as_str(train_cfg.get("init_mode", "scratch"), "tccig_train.init_mode").lower()
    if init_mode != "scratch":
        raise ValueError("tccig_train only supports scratch initialization")
    teacher_mask_ratio, teacher_negative_ratio = parse_teacher_training_config(train_cfg)
    node_sizes = topology_train._resolve_sampling_node_sizes(train_cfg, stage_name="tccig_train")
    optimizer_cfg = optimizer_config(train_cfg)
    return TCCIGTrainConfig(
        epochs=as_int(train_cfg.get("epochs", training_cfg.get("epochs", 1)), "tccig_train.epochs"),
        run_seed=as_int(run_cfg.get("seed", 0), "run_config.seed"),
        pair_batch_size=as_int(
            train_cfg.get("pair_batch_size", training_cfg.get("batch_size", 8)),
            "tccig_train.pair_batch_size",
        ),
        gradient_accumulation_steps=topology_train._resolve_gradient_accumulation_steps(
            train_cfg,
            stage_name="tccig_train",
        ),
        subgraphs_per_epoch=as_int(
            train_cfg.get("subgraphs_per_epoch", 0),
            "tccig_train.subgraphs_per_epoch",
        ),
        node_sizes=node_sizes,
        strategy=as_str(train_cfg.get("strategy", "mixed"), "tccig_train.strategy"),
        negative_ratio=topology_train._resolve_bce_negative_ratio(
            train_cfg,
            stage_name="tccig_train",
        ),
        edge_chunk_size=topology_train._resolve_edge_chunk_size(
            finetune_cfg=train_cfg,
            stage_name="tccig_train",
        ),
        compute_clustering_mmd=topology_train._resolve_compute_clustering_mmd(
            train_cfg,
            stage_name="tccig_train",
        ),
        internal_validation_inference_batch_size=(
            topology_train._resolve_internal_validation_inference_batch_size(
                train_cfg,
                stage_name="tccig_train",
            )
        ),
        internal_validation_compute_spectral_stats=(
            topology_train._resolve_internal_validation_compute_spectral_stats(
                train_cfg,
                stage_name="tccig_train",
            )
        ),
        internal_validation_compute_clustering_mmd=(
            topology_train._resolve_internal_validation_compute_clustering_mmd(
                train_cfg,
                stage_name="tccig_train",
            )
        ),
        loss_weight_schedule=topology_train._parse_loss_weight_schedule(
            train_cfg,
            stage_name="tccig_train",
        ),
        loss_weights=parse_tccig_loss_weights(train_cfg),
        monitor_metric=as_str(
            train_cfg.get("monitor_metric", "val_topology_loss"),
            "tccig_train.monitor_metric",
        ),
        early_stopping_patience=as_int(
            train_cfg.get(
                "early_stopping_patience",
                training_cfg.get("early_stopping_patience", 5),
            ),
            "tccig_train.early_stopping_patience",
        ),
        optimizer=parse_optimizer_config(
            optimizer_cfg,
            field_name="tccig_train.optimizer",
            default_lr=1e-5,
        ),
        teacher_mask_ratio=teacher_mask_ratio,
        teacher_negative_ratio=teacher_negative_ratio,
    )


def optimizer_config(train_cfg: ConfigDict) -> ConfigDict:
    """Return ``tccig_train.optimizer`` config."""
    optimizer_cfg = train_cfg.get("optimizer", {})
    if not isinstance(optimizer_cfg, dict):
        raise ValueError("tccig_train.optimizer must be a mapping")
    return cast(ConfigDict, optimizer_cfg)


def parse_optimizer_config(
    optimizer_cfg: ConfigDict,
    *,
    field_name: str,
    default_lr: float,
) -> OptimizerConfig:
    """Parse an optimizer config into the shared training optimizer dataclass."""
    return OptimizerConfig(
        optimizer_type=as_str(optimizer_cfg.get("type", "adamw"), f"{field_name}.type"),
        lr=as_float(optimizer_cfg.get("lr", default_lr), f"{field_name}.lr"),
        beta1=as_float(optimizer_cfg.get("beta1", 0.9), f"{field_name}.beta1"),
        beta2=as_float(optimizer_cfg.get("beta2", 0.999), f"{field_name}.beta2"),
        eps=as_float(optimizer_cfg.get("eps", 1e-8), f"{field_name}.eps"),
        weight_decay=as_float(
            optimizer_cfg.get("weight_decay", 0.0),
            f"{field_name}.weight_decay",
        ),
    )


def build_tccig_optimizer(train_cfg: TCCIGTrainConfig, model: nn.Module) -> Optimizer:
    """Build the TCCIG student optimizer."""
    optimizer_type = train_cfg.optimizer.optimizer_type.lower()
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.optimizer.lr,
            betas=(train_cfg.optimizer.beta1, train_cfg.optimizer.beta2),
            eps=train_cfg.optimizer.eps,
            weight_decay=train_cfg.optimizer.weight_decay,
        )
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=train_cfg.optimizer.lr,
            weight_decay=train_cfg.optimizer.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer type: {train_cfg.optimizer.optimizer_type}")
