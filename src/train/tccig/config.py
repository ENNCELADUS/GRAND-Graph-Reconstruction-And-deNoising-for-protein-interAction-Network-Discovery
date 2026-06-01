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
from src.utils.config import ConfigDict, as_bool, as_float, as_int, as_str, get_section


@dataclass(frozen=True)
class TCCIGRetrievalConfig:
    """Resolved retrieval objective settings."""

    backend: str
    top_k: int
    train_top_k: int
    temperature: float
    retrieval_weight: float


@dataclass(frozen=True)
class TCCIGGraphPriorTeacherConfig:
    """Resolved offline graph-prior teacher settings."""

    enabled: bool
    epochs: int
    hidden_dim: int
    num_layers: int
    decoder_hidden_dim: int
    dropout: float
    mask_ratio: float
    negative_ratio: int
    artifact_dir: str
    reuse_artifacts: bool
    struct_weight: float
    degree_weight: float


@dataclass(frozen=True)
class TCCIGHardNegativeMiningConfig:
    """Resolved online hard-negative mining settings."""

    enabled: bool
    top_k: int
    max_pairs_per_epoch: int
    refresh_every_epochs: int
    weight: float


@dataclass(frozen=True)
class TCCIGRerankerConfig:
    """Resolved local reranker settings."""

    enabled: bool
    weight: float
    adaptive_negative_temperature: float
    external_teacher_enabled: bool


@dataclass(frozen=True)
class TCCIGGraphAssemblyConfig:
    """Resolved graph assembly settings."""

    rule: str
    validation_density_budget: bool
    degree_cap_enabled: bool
    degree_cap_slack: float


@dataclass(frozen=True)
class TCCIGValidationReconstructionConfig:
    """Resolved PRING-like validation reconstruction settings."""

    enabled: bool
    full_universe: bool
    node_batch_size: int
    max_pairs: int | None
    recall_k_percent: tuple[float, ...]


@dataclass(frozen=True)
class TCCIGMonitorConfig:
    """Resolved composite monitor settings."""

    metric: str
    recall_weight: float
    auprc_weight: float
    graph_sim_weight: float
    relative_density_penalty: float
    degree_mmd_penalty: float
    clustering_mmd_penalty: float


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
    retrieval: TCCIGRetrievalConfig
    graph_prior_teacher: TCCIGGraphPriorTeacherConfig
    hard_negative_mining: TCCIGHardNegativeMiningConfig
    reranker: TCCIGRerankerConfig
    graph_assembly: TCCIGGraphAssemblyConfig
    validation_reconstruction: TCCIGValidationReconstructionConfig
    monitor: TCCIGMonitorConfig


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


def _get_optional_section(train_cfg: ConfigDict, name: str) -> ConfigDict:
    """Return an optional nested ``tccig_train`` section."""
    section = train_cfg.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"tccig_train.{name} must be a mapping")
    return cast(ConfigDict, section)


def parse_retrieval_config(train_cfg: ConfigDict) -> TCCIGRetrievalConfig:
    """Parse ``tccig_train.retrieval``."""
    cfg = _get_optional_section(train_cfg, "retrieval")
    backend = as_str(cfg.get("backend", "exact"), "tccig_train.retrieval.backend").lower()
    if backend != "exact":
        raise ValueError("tccig_train.retrieval.backend currently supports only 'exact'")
    top_k = as_int(cfg.get("top_k", 128), "tccig_train.retrieval.top_k")
    train_top_k = as_int(cfg.get("train_top_k", top_k), "tccig_train.retrieval.train_top_k")
    if top_k <= 0 or train_top_k <= 0:
        raise ValueError("tccig_train.retrieval top-k values must be > 0")
    temperature = as_float(
        cfg.get("temperature", 0.07),
        "tccig_train.retrieval.temperature",
    )
    if temperature <= 0.0:
        raise ValueError("tccig_train.retrieval.temperature must be > 0")
    return TCCIGRetrievalConfig(
        backend=backend,
        top_k=top_k,
        train_top_k=train_top_k,
        temperature=temperature,
        retrieval_weight=as_float(
            cfg.get("weight", 1.0),
            "tccig_train.retrieval.weight",
        ),
    )


def parse_graph_prior_teacher_config(train_cfg: ConfigDict) -> TCCIGGraphPriorTeacherConfig:
    """Parse ``tccig_train.graph_prior_teacher``."""
    cfg = _get_optional_section(train_cfg, "graph_prior_teacher")
    return TCCIGGraphPriorTeacherConfig(
        enabled=as_bool(cfg.get("enabled", False), "tccig_train.graph_prior_teacher.enabled"),
        epochs=as_int(cfg.get("epochs", 0), "tccig_train.graph_prior_teacher.epochs"),
        hidden_dim=as_int(cfg.get("hidden_dim", 128), "tccig_train.graph_prior_teacher.hidden_dim"),
        num_layers=as_int(cfg.get("num_layers", 2), "tccig_train.graph_prior_teacher.num_layers"),
        decoder_hidden_dim=as_int(
            cfg.get("decoder_hidden_dim", cfg.get("hidden_dim", 128)),
            "tccig_train.graph_prior_teacher.decoder_hidden_dim",
        ),
        dropout=as_float(cfg.get("dropout", 0.1), "tccig_train.graph_prior_teacher.dropout"),
        mask_ratio=as_float(
            cfg.get("mask_ratio", 0.7),
            "tccig_train.graph_prior_teacher.mask_ratio",
        ),
        negative_ratio=as_int(
            cfg.get("negative_ratio", 1),
            "tccig_train.graph_prior_teacher.negative_ratio",
        ),
        artifact_dir=as_str(
            cfg.get("artifact_dir", "graph_prior_teacher"),
            "tccig_train.graph_prior_teacher.artifact_dir",
        ),
        reuse_artifacts=as_bool(
            cfg.get("reuse_artifacts", True),
            "tccig_train.graph_prior_teacher.reuse_artifacts",
        ),
        struct_weight=as_float(
            cfg.get("struct_weight", 0.1),
            "tccig_train.graph_prior_teacher.struct_weight",
        ),
        degree_weight=as_float(
            cfg.get("degree_weight", 0.05),
            "tccig_train.graph_prior_teacher.degree_weight",
        ),
    )


def parse_hard_negative_mining_config(train_cfg: ConfigDict) -> TCCIGHardNegativeMiningConfig:
    """Parse ``tccig_train.hard_negative_mining``."""
    cfg = _get_optional_section(train_cfg, "hard_negative_mining")
    return TCCIGHardNegativeMiningConfig(
        enabled=as_bool(cfg.get("enabled", False), "tccig_train.hard_negative_mining.enabled"),
        top_k=as_int(cfg.get("top_k", 64), "tccig_train.hard_negative_mining.top_k"),
        max_pairs_per_epoch=as_int(
            cfg.get("max_pairs_per_epoch", 100_000),
            "tccig_train.hard_negative_mining.max_pairs_per_epoch",
        ),
        refresh_every_epochs=as_int(
            cfg.get("refresh_every_epochs", 1),
            "tccig_train.hard_negative_mining.refresh_every_epochs",
        ),
        weight=as_float(cfg.get("weight", 0.1), "tccig_train.hard_negative_mining.weight"),
    )


def parse_reranker_config(train_cfg: ConfigDict) -> TCCIGRerankerConfig:
    """Parse ``tccig_train.reranker``."""
    cfg = _get_optional_section(train_cfg, "reranker")
    return TCCIGRerankerConfig(
        enabled=as_bool(cfg.get("enabled", True), "tccig_train.reranker.enabled"),
        weight=as_float(cfg.get("weight", 1.0), "tccig_train.reranker.weight"),
        adaptive_negative_temperature=as_float(
            cfg.get("adaptive_negative_temperature", 4.0),
            "tccig_train.reranker.adaptive_negative_temperature",
        ),
        external_teacher_enabled=as_bool(
            cfg.get("external_teacher_enabled", False),
            "tccig_train.reranker.external_teacher_enabled",
        ),
    )


def parse_graph_assembly_config(train_cfg: ConfigDict) -> TCCIGGraphAssemblyConfig:
    """Parse ``tccig_train.graph_assembly``."""
    cfg = _get_optional_section(train_cfg, "graph_assembly")
    rule = as_str(
        cfg.get("rule", "hybrid_validation_density_degree_cap"),
        "tccig_train.graph_assembly.rule",
    )
    return TCCIGGraphAssemblyConfig(
        rule=rule,
        validation_density_budget=as_bool(
            cfg.get("validation_density_budget", True),
            "tccig_train.graph_assembly.validation_density_budget",
        ),
        degree_cap_enabled=as_bool(
            cfg.get("degree_cap_enabled", True),
            "tccig_train.graph_assembly.degree_cap_enabled",
        ),
        degree_cap_slack=as_float(
            cfg.get("degree_cap_slack", 1.0),
            "tccig_train.graph_assembly.degree_cap_slack",
        ),
    )


def parse_validation_reconstruction_config(
    train_cfg: ConfigDict,
) -> TCCIGValidationReconstructionConfig:
    """Parse ``tccig_train.validation_reconstruction``."""
    cfg = _get_optional_section(train_cfg, "validation_reconstruction")
    raw_recall = cfg.get("recall_k_percent", [1.0, 3.0, 5.0, 10.0, 20.0])
    if not isinstance(raw_recall, (list, tuple)):
        raise ValueError("tccig_train.validation_reconstruction.recall_k_percent must be a list")
    recall_k_percent = tuple(
        as_float(value, "tccig_train.validation_reconstruction.recall_k_percent")
        for value in raw_recall
    )
    raw_max_pairs = cfg.get("max_pairs")
    max_pairs = (
        None
        if raw_max_pairs is None
        else as_int(raw_max_pairs, "tccig_train.validation_reconstruction.max_pairs")
    )
    node_batch_size = as_int(
        cfg.get("node_batch_size", 64),
        "tccig_train.validation_reconstruction.node_batch_size",
    )
    if node_batch_size <= 0:
        raise ValueError("tccig_train.validation_reconstruction.node_batch_size must be > 0")
    return TCCIGValidationReconstructionConfig(
        enabled=as_bool(
            cfg.get("enabled", True),
            "tccig_train.validation_reconstruction.enabled",
        ),
        full_universe=as_bool(
            cfg.get("full_universe", True),
            "tccig_train.validation_reconstruction.full_universe",
        ),
        node_batch_size=node_batch_size,
        max_pairs=max_pairs,
        recall_k_percent=recall_k_percent,
    )


def parse_monitor_config(train_cfg: ConfigDict) -> TCCIGMonitorConfig:
    """Parse ``tccig_train.monitor``."""
    cfg = _get_optional_section(train_cfg, "monitor")
    return TCCIGMonitorConfig(
        metric=as_str(cfg.get("metric", "val_composite_score"), "tccig_train.monitor.metric"),
        recall_weight=as_float(cfg.get("recall_weight", 0.45), "tccig_train.monitor.recall_weight"),
        auprc_weight=as_float(cfg.get("auprc_weight", 0.25), "tccig_train.monitor.auprc_weight"),
        graph_sim_weight=as_float(
            cfg.get("graph_sim_weight", 0.20),
            "tccig_train.monitor.graph_sim_weight",
        ),
        relative_density_penalty=as_float(
            cfg.get("relative_density_penalty", 0.05),
            "tccig_train.monitor.relative_density_penalty",
        ),
        degree_mmd_penalty=as_float(
            cfg.get("degree_mmd_penalty", 0.03),
            "tccig_train.monitor.degree_mmd_penalty",
        ),
        clustering_mmd_penalty=as_float(
            cfg.get("clustering_mmd_penalty", 0.02),
            "tccig_train.monitor.clustering_mmd_penalty",
        ),
    )


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
        retrieval=parse_retrieval_config(train_cfg),
        graph_prior_teacher=parse_graph_prior_teacher_config(train_cfg),
        hard_negative_mining=parse_hard_negative_mining_config(train_cfg),
        reranker=parse_reranker_config(train_cfg),
        graph_assembly=parse_graph_assembly_config(train_cfg),
        validation_reconstruction=parse_validation_reconstruction_config(train_cfg),
        monitor=parse_monitor_config(train_cfg),
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
