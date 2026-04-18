"""Graph-topology fine-tuning stage for PRING training subgraphs."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import networkx as nx
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from src.embed import ensure_embeddings_ready
from src.evaluate import Evaluator
from src.pipeline.loops import move_batch_to_device, reduce_scalar_mapping
from src.pipeline.runtime import AcceleratorLike, DistributedContext, PipelineRuntime
from src.pipeline.stages.train import _build_loss_config
from src.topology.finetune_data import (
    TOPOLOGY_EVAL_NODE_SIZES,
    EdgeCoverEpochPlan,
    EmbeddingRepository,
    ExplicitNegativePairLookup,
    InternalValidationNodeBucketPlan,
    InternalValidationPairRecord,
    InternalValidationPlan,
    SubgraphPairChunk,
    build_explicit_negative_lookup,
    build_internal_validation_plan,
    build_pair_supervision_graph,
    iter_subgraph_pair_chunks,
    load_split_node_ids,
    sample_edge_cover_subgraphs,
    sample_topology_evaluation_subgraphs,
)
from src.topology.finetune_losses import (
    TopologyLossWeights,
    TopologyLossWeightSchedule,
    build_symmetric_adjacency,
    compute_topology_losses,
    topology_loss_scale,
)
from src.topology.loss_balancing import (
    TopologyAdaptiveLossState,
    TopologyGradNormConfig,
    TopologyLossNormalizationConfig,
    initialize_output_head_bias_with_prior,
    normalize_topology_loss_terms,
    update_gradnorm_task_weights,
)
from src.topology.metrics import evaluate_graph_samples
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_float,
    as_int,
    as_str,
    extract_model_kwargs,
    get_section,
)
from src.utils.early_stop import EarlyStopping
from src.utils.logging import append_csv_row, log_epoch_progress, log_stage_event
from src.utils.losses import binary_classification_loss

TOPOLOGY_FINETUNE_CSV_COLUMNS = [
    "Epoch",
    "Epoch Time",
    "Train BCE Loss",
    "Train GS Loss",
    "Train RD Loss",
    "Train Deg MMD",
    "Train Clus MMD",
    "Train Total Loss",
    "Val Pair Loss",
    "Val Topology Loss",
    "Val Loss",
    "Val auprc",
    "Internal Val graph_sim",
    "Internal Val relative_density",
    "Internal Val deg_dist_mmd",
    "Internal Val cc_mmd",
    "Planned Subgraphs",
    "Covered Positive Edges",
    "Total Positive Edges",
    "Positive Edge Coverage Ratio",
    "Mean Positive Edge Reuse",
    "edge_cover_sampling_s",
    "train_forward_backward_s",
    "val_pair_pass_s",
    "val_threshold_s",
    "internal_val_topology_s",
    "peak_gpu_mem_mb",
    "Topology Loss Scale",
    "Learning Rate",
]

TRAIN_LOSS_KEYS = (
    "bce",
    "graph_similarity",
    "relative_density",
    "degree_mmd",
    "clustering_mmd",
    "total",
)
INTERNAL_VALIDATION_SUMMARY_KEYS = (
    "graph_sim",
    "relative_density",
    "deg_dist_mmd",
    "cc_mmd",
)
DEFAULT_INTERNAL_VALIDATION_SAMPLES_PER_SIZE = 20


@dataclass(frozen=True)
class TopologyFinetuneStageContext:
    """Runtime artifacts reused across topology fine-tuning epochs."""

    train_graph: nx.Graph
    internal_val_graph: nx.Graph
    train_negative_lookup: ExplicitNegativePairLookup
    cache_dir: Path
    embedding_index: Mapping[str, str]
    embedding_repository: EmbeddingRepository
    input_dim: int
    max_sequence_length: int
    pair_batch_size: int
    internal_validation_inference_batch_size: int
    internal_validation_compute_spectral_stats: bool
    epochs: int
    run_seed: int
    use_amp: bool
    optimizer: Optimizer
    evaluator: Evaluator
    loss_weights: TopologyLossWeights
    loss_weight_schedule: TopologyLossWeightSchedule
    loss_normalization: TopologyLossNormalizationConfig
    gradnorm: TopologyGradNormConfig
    adaptive_loss_state: TopologyAdaptiveLossState
    monitor_metric: str
    early_stopping: EarlyStopping
    best_checkpoint_path: Path
    metrics_path: Path
    csv_path: Path
    internal_validation_node_sets: Mapping[int, Sequence[tuple[str, ...]]]
    internal_validation_plan: InternalValidationPlan


@dataclass(frozen=True)
class ValidationEpochResult:
    """Validation outputs used for monitoring and artifact writing."""

    decision_threshold: float
    val_pair_stats: Mapping[str, float]
    internal_val_topology_stats: Mapping[str, float]
    val_pair_pass_seconds: float
    threshold_resolution_seconds: float
    internal_validation_seconds: float
    val_topology_loss: float = 0.0
    val_total_loss: float = 0.0


@dataclass(frozen=True)
class SubgraphForwardResult:
    """Forward outputs and pair labels for one sampled training subgraph."""

    nodes: tuple[str, ...]
    logits: torch.Tensor
    labels: torch.Tensor
    bce_labels: torch.Tensor
    bce_mask: torch.Tensor
    pair_index_a: torch.Tensor
    pair_index_b: torch.Tensor


@dataclass(frozen=True)
class LocalSubgraphTask:
    """Rank-local train task, including DDP padding participation."""

    nodes: tuple[str, ...]
    assigned_positive_edges: frozenset[tuple[str, str]]
    assigned_negative_edges: frozenset[tuple[str, str]]
    is_padding: bool = False


def _empty_internal_validation_summary() -> dict[str, float]:
    """Return zeroed internal-validation summary metrics."""
    return dict.fromkeys(INTERNAL_VALIDATION_SUMMARY_KEYS, 0.0)


def _empty_validation_epoch_result() -> ValidationEpochResult:
    """Return a zeroed validation payload for non-main distributed ranks."""
    return ValidationEpochResult(
        decision_threshold=0.5,
        val_pair_stats={"val_loss": 0.0, "val_auprc": 0.0},
        internal_val_topology_stats=_empty_internal_validation_summary(),
        val_pair_pass_seconds=0.0,
        threshold_resolution_seconds=0.0,
        internal_validation_seconds=0.0,
        val_topology_loss=0.0,
        val_total_loss=0.0,
    )


def _topology_finetune_config(config: ConfigDict) -> ConfigDict:
    """Return ``topology_finetune`` config with schema validation."""
    finetune_cfg = config.get("topology_finetune", {})
    if not isinstance(finetune_cfg, dict):
        raise ValueError("topology_finetune must be a mapping")
    return cast(ConfigDict, finetune_cfg)


def _resolve_monitor_mode(monitor_metric: str) -> str:
    """Return the optimization direction for a topology finetune monitor."""
    if monitor_metric == "val_loss":
        return "min"
    return "max"


def _resolve_monitor_value(
    *,
    monitor_metric: str,
    val_pair_stats: Mapping[str, float],
    internal_val_topology_stats: Mapping[str, float],
    val_total_loss: float | None = None,
) -> float:
    """Resolve the scalar value used for checkpoint selection and early stopping."""
    return float(
        {
            "val_loss": (
                float(val_total_loss)
                if val_total_loss is not None
                else float(val_pair_stats.get("val_loss", 0.0))
            ),
            "internal_val_graph_sim": internal_val_topology_stats["graph_sim"],
            "val_graph_sim": internal_val_topology_stats["graph_sim"],
            "internal_val_relative_density": -abs(
                internal_val_topology_stats["relative_density"] - 1.0
            ),
            "val_relative_density": -abs(internal_val_topology_stats["relative_density"] - 1.0),
            "val_auprc": float(val_pair_stats.get("val_auprc", 0.0)),
        }.get(monitor_metric, internal_val_topology_stats["graph_sim"])
    )


def _resolve_epoch_seed(
    *,
    run_seed: int,
    epoch_index: int,
    distributed_context: DistributedContext,
) -> int:
    """Return the RNG seed for one topology fine-tune epoch.

    This stage samples custom subgraphs instead of using a sharded dataloader. Under DDP,
    all ranks therefore need the same epoch plan so they execute the same number of forward
    and backward steps before the next collective.
    """
    if distributed_context.is_distributed:
        return run_seed + epoch_index
    return run_seed + (1000 * distributed_context.rank) + epoch_index


def _resolve_supervision_dataset_path(
    *,
    finetune_cfg: ConfigDict,
    dataloader_cfg: ConfigDict,
    config_key: str,
    fallback_key: str,
) -> Path:
    """Resolve and validate one topology supervision dataset path."""
    raw_path = finetune_cfg.get(config_key, dataloader_cfg.get(fallback_key, ""))
    path = Path(str(raw_path))
    if not str(raw_path):
        raise ValueError(
            f"Missing topology supervision dataset path for topology_finetune.{config_key}"
        )
    if path.exists():
        return path
    raise FileNotFoundError(
        "Topology fine-tuning supervision dataset not found: "
        f"{path}. Runtime generation is disabled; prepare them offline and update "
        f"topology_finetune.{config_key} before launching the pipeline."
    )


def _load_supervision_graphs(*, config: ConfigDict) -> tuple[nx.Graph, nx.Graph]:
    """Build train/validation supervision graphs without leaking validation edges."""
    data_cfg = get_section(config, "data_config")
    benchmark_cfg = get_section(data_cfg, "benchmark")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    finetune_cfg = _topology_finetune_config(config)
    processed_dir = Path(str(benchmark_cfg.get("processed_dir", "")))
    species = as_str(benchmark_cfg.get("species", "human"), "data_config.benchmark.species")
    split_strategy = as_str(
        benchmark_cfg.get("split_strategy", "BFS"),
        "data_config.benchmark.split_strategy",
    ).upper()
    split_path = processed_dir / f"{species}_{split_strategy}_split.pkl"
    train_pair_path = _resolve_supervision_dataset_path(
        finetune_cfg=finetune_cfg,
        dataloader_cfg=dataloader_cfg,
        config_key="supervision_train_dataset",
        fallback_key="train_dataset",
    )
    valid_pair_path = _resolve_supervision_dataset_path(
        finetune_cfg=finetune_cfg,
        dataloader_cfg=dataloader_cfg,
        config_key="supervision_valid_dataset",
        fallback_key="valid_dataset",
    )
    train_nodes = load_split_node_ids(split_path=split_path, split_name="train")
    train_graph = build_pair_supervision_graph(
        pair_path=train_pair_path,
        node_ids=train_nodes,
    )
    internal_val_graph = build_pair_supervision_graph(
        pair_path=valid_pair_path,
        node_ids=train_nodes,
    )
    return train_graph, internal_val_graph


def _load_train_negative_lookup(*, config: ConfigDict) -> ExplicitNegativePairLookup:
    """Load explicit train negatives used for masked BCE supervision."""
    data_cfg = get_section(config, "data_config")
    benchmark_cfg = get_section(data_cfg, "benchmark")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    finetune_cfg = _topology_finetune_config(config)
    processed_dir = Path(str(benchmark_cfg.get("processed_dir", "")))
    species = as_str(benchmark_cfg.get("species", "human"), "data_config.benchmark.species")
    split_strategy = as_str(
        benchmark_cfg.get("split_strategy", "BFS"),
        "data_config.benchmark.split_strategy",
    ).upper()
    split_path = processed_dir / f"{species}_{split_strategy}_split.pkl"
    train_nodes = load_split_node_ids(split_path=split_path, split_name="train")
    train_pair_path = _resolve_supervision_dataset_path(
        finetune_cfg=finetune_cfg,
        dataloader_cfg=dataloader_cfg,
        config_key="supervision_train_dataset",
        fallback_key="train_dataset",
    )
    return build_explicit_negative_lookup(pair_path=train_pair_path, node_ids=train_nodes)


def _parse_loss_weights(config: ConfigDict) -> TopologyLossWeights:
    """Parse graph-loss weights from ``topology_finetune.losses``."""
    finetune_cfg = _topology_finetune_config(config)
    loss_cfg = finetune_cfg.get("losses", {})
    if not isinstance(loss_cfg, dict):
        raise ValueError("topology_finetune.losses must be a mapping")
    rd_loss_form = as_str(
        loss_cfg.get("rd_loss_form", "log_ratio_huber"),
        "topology_finetune.losses.rd_loss_form",
    ).lower()
    if rd_loss_form not in {"squared_ratio", "log_ratio", "log_ratio_huber"}:
        raise ValueError(
            "topology_finetune.losses.rd_loss_form must be one of "
            "{'squared_ratio', 'log_ratio', 'log_ratio_huber'}"
        )
    return TopologyLossWeights(
        alpha=as_float(loss_cfg.get("alpha", 0.5), "topology_finetune.losses.alpha"),
        beta=as_float(loss_cfg.get("beta", 1.0), "topology_finetune.losses.beta"),
        gamma=as_float(loss_cfg.get("gamma", 0.3), "topology_finetune.losses.gamma"),
        delta=as_float(loss_cfg.get("delta", 0.3), "topology_finetune.losses.delta"),
        histogram_sigma=as_float(
            loss_cfg.get("histogram_sigma", 1.0),
            "topology_finetune.losses.histogram_sigma",
        ),
        degree_bins=as_int(
            loss_cfg.get("degree_bins", 64),
            "topology_finetune.losses.degree_bins",
        ),
        clustering_bins=as_int(
            loss_cfg.get("clustering_bins", 100),
            "topology_finetune.losses.clustering_bins",
        ),
        rd_loss_form=rd_loss_form,
    )


def _parse_loss_weight_schedule(finetune_cfg: ConfigDict) -> TopologyLossWeightSchedule:
    """Parse topology-loss weight scheduling from ``topology_finetune`` config."""
    raw_schedule = finetune_cfg.get("loss_weight_schedule", {})
    if not isinstance(raw_schedule, dict):
        raise ValueError("topology_finetune.loss_weight_schedule must be a mapping")
    warmup_epochs = as_int(
        raw_schedule.get("warmup_epochs", 0),
        "topology_finetune.loss_weight_schedule.warmup_epochs",
    )
    ramp_epochs = as_int(
        raw_schedule.get("ramp_epochs", 0),
        "topology_finetune.loss_weight_schedule.ramp_epochs",
    )
    if warmup_epochs < 0:
        raise ValueError("topology_finetune.loss_weight_schedule.warmup_epochs must be >= 0")
    if ramp_epochs < 0:
        raise ValueError("topology_finetune.loss_weight_schedule.ramp_epochs must be >= 0")
    schedule = as_str(
        raw_schedule.get("schedule", "linear"),
        "topology_finetune.loss_weight_schedule.schedule",
    ).lower()
    if schedule not in {"linear", "cosine"}:
        raise ValueError("topology_finetune.loss_weight_schedule.schedule must be linear or cosine")
    return TopologyLossWeightSchedule(
        warmup_epochs=warmup_epochs,
        ramp_epochs=ramp_epochs,
        schedule=schedule,
    )


def _parse_loss_normalization_config(
    finetune_cfg: ConfigDict,
) -> TopologyLossNormalizationConfig:
    """Parse topology-loss normalization config."""
    raw_config = finetune_cfg.get("loss_normalization", {})
    if not isinstance(raw_config, dict):
        raise ValueError("topology_finetune.loss_normalization must be a mapping")
    ema_decay = as_float(
        raw_config.get("ema_decay", 0.95),
        "topology_finetune.loss_normalization.ema_decay",
    )
    clip_value = as_float(
        raw_config.get("clip_value", 5.0),
        "topology_finetune.loss_normalization.clip_value",
    )
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("topology_finetune.loss_normalization.ema_decay must be in [0, 1)")
    if clip_value <= 0.0:
        raise ValueError("topology_finetune.loss_normalization.clip_value must be > 0")
    return TopologyLossNormalizationConfig(
        enabled=as_bool(
            raw_config.get("enabled", False),
            "topology_finetune.loss_normalization.enabled",
        ),
        ema_decay=ema_decay,
        clip_value=clip_value,
    )


def _parse_gradnorm_config(finetune_cfg: ConfigDict) -> TopologyGradNormConfig:
    """Parse grouped GradNorm config."""
    raw_config = finetune_cfg.get("gradnorm", {})
    if not isinstance(raw_config, dict):
        raise ValueError("topology_finetune.gradnorm must be a mapping")
    alpha = as_float(raw_config.get("alpha", 0.5), "topology_finetune.gradnorm.alpha")
    learning_rate = as_float(
        raw_config.get("learning_rate", 0.02),
        "topology_finetune.gradnorm.learning_rate",
    )
    min_weight = as_float(
        raw_config.get("min_weight", 0.2),
        "topology_finetune.gradnorm.min_weight",
    )
    max_weight = as_float(
        raw_config.get("max_weight", 5.0),
        "topology_finetune.gradnorm.max_weight",
    )
    if alpha < 0.0:
        raise ValueError("topology_finetune.gradnorm.alpha must be >= 0")
    if learning_rate <= 0.0:
        raise ValueError("topology_finetune.gradnorm.learning_rate must be > 0")
    if min_weight <= 0.0:
        raise ValueError("topology_finetune.gradnorm.min_weight must be > 0")
    if max_weight < min_weight:
        raise ValueError("topology_finetune.gradnorm.max_weight must be >= min_weight")
    return TopologyGradNormConfig(
        enabled=as_bool(raw_config.get("enabled", False), "topology_finetune.gradnorm.enabled"),
        alpha=alpha,
        learning_rate=learning_rate,
        min_weight=min_weight,
        max_weight=max_weight,
    )


def _resolve_sampling_node_bounds(finetune_cfg: ConfigDict) -> tuple[int, int]:
    """Resolve subgraph node limits for topology fine-tuning."""
    raw_range = finetune_cfg.get("subgraph_node_range")
    if (
        not isinstance(raw_range, Sequence)
        or isinstance(raw_range, str | bytes)
        or len(raw_range) != 2
    ):
        raise ValueError(
            "topology_finetune.subgraph_node_range must be a two-item sequence "
            "[min_nodes, max_nodes]"
        )
    min_nodes = as_int(raw_range[0], "topology_finetune.subgraph_node_range[0]")
    max_nodes = as_int(raw_range[1], "topology_finetune.subgraph_node_range[1]")
    if min_nodes <= 0:
        raise ValueError("topology_finetune.subgraph_node_range[0] must be positive")
    if max_nodes <= 0:
        raise ValueError("topology_finetune.subgraph_node_range[1] must be positive")
    if min_nodes > max_nodes:
        raise ValueError(
            "topology_finetune.subgraph_node_range[0] must be <= "
            "topology_finetune.subgraph_node_range[1]"
        )
    return min_nodes, max_nodes


def _resolve_init_mode(finetune_cfg: ConfigDict) -> str:
    """Return topology fine-tuning initialization mode."""
    init_mode = as_str(
        finetune_cfg.get("init_mode", "warm_start"),
        "topology_finetune.init_mode",
    ).lower()
    if init_mode not in {"warm_start", "scratch"}:
        raise ValueError("topology_finetune.init_mode must be 'warm_start' or 'scratch'")
    return init_mode


def _resolve_bce_negative_ratio(finetune_cfg: ConfigDict) -> int:
    """Return the epoch-plan negative-to-positive BCE ratio."""
    negative_ratio = as_int(
        finetune_cfg.get("bce_negative_ratio", 5),
        "topology_finetune.bce_negative_ratio",
    )
    if negative_ratio < 0:
        raise ValueError("topology_finetune.bce_negative_ratio must be >= 0")
    return negative_ratio


def _resolve_edge_chunk_size(
    *,
    finetune_cfg: ConfigDict,
    max_nodes: int,
) -> int | None:
    """Return the positive-edge chunk size for one topology epoch plan."""
    raw_edge_chunk_size = finetune_cfg.get("edge_chunk_size")
    if raw_edge_chunk_size is None:
        return max(1, (max_nodes * (max_nodes - 1)) // 4)
    edge_chunk_size = as_int(
        raw_edge_chunk_size,
        "topology_finetune.edge_chunk_size",
    )
    if edge_chunk_size <= 0:
        raise ValueError("topology_finetune.edge_chunk_size must be > 0")
    return edge_chunk_size


def _resolve_gradient_accumulation_steps(finetune_cfg: ConfigDict) -> int:
    """Return the number of subgraphs to accumulate before optimizer step."""
    steps = as_int(
        finetune_cfg.get("gradient_accumulation_steps", 1),
        "topology_finetune.gradient_accumulation_steps",
    )
    if steps <= 0:
        raise ValueError("topology_finetune.gradient_accumulation_steps must be > 0")
    return steps


def _resolve_subgraphs_per_forward(finetune_cfg: ConfigDict) -> int:
    """Return the number of subgraphs concatenated into one training forward pass."""
    subgraphs_per_forward = as_int(
        finetune_cfg.get("subgraphs_per_forward", 1),
        "topology_finetune.subgraphs_per_forward",
    )
    if subgraphs_per_forward <= 0:
        raise ValueError("topology_finetune.subgraphs_per_forward must be > 0")
    return subgraphs_per_forward


def _build_internal_validation_node_sets(
    *,
    finetune_cfg: ConfigDict,
    graph: nx.Graph,
    seed: int,
) -> dict[int, list[tuple[str, ...]]]:
    """Build topology-evaluate-style node buckets for internal validation."""
    return sample_topology_evaluation_subgraphs(
        graph=graph,
        seed=seed,
        strategy="mixed",
        node_sizes=_resolve_internal_validation_node_sizes(finetune_cfg),
        samples_per_size=_resolve_internal_validation_samples_per_size(finetune_cfg),
    )


def _resolve_internal_validation_node_sizes(finetune_cfg: ConfigDict) -> tuple[int, ...]:
    """Return the node-size buckets used for internal topology validation."""
    raw_node_sizes = finetune_cfg.get("internal_validation_node_sizes", TOPOLOGY_EVAL_NODE_SIZES)
    if not isinstance(raw_node_sizes, Sequence) or isinstance(raw_node_sizes, (str, bytes)):
        raise ValueError("topology_finetune.internal_validation_node_sizes must be a sequence")
    node_sizes = [
        as_int(node_size, "topology_finetune.internal_validation_node_sizes")
        for node_size in raw_node_sizes
    ]
    if not node_sizes:
        raise ValueError("topology_finetune.internal_validation_node_sizes must not be empty")
    if any(node_size <= 0 for node_size in node_sizes):
        raise ValueError("topology_finetune.internal_validation_node_sizes must be > 0")
    return tuple(dict.fromkeys(node_sizes))


def _resolve_internal_validation_samples_per_size(finetune_cfg: ConfigDict) -> int:
    """Return the number of internal-validation subgraphs sampled per node size."""
    samples_per_size = as_int(
        finetune_cfg.get(
            "internal_validation_samples_per_size",
            DEFAULT_INTERNAL_VALIDATION_SAMPLES_PER_SIZE,
        ),
        "topology_finetune.internal_validation_samples_per_size",
    )
    if samples_per_size <= 0:
        raise ValueError("topology_finetune.internal_validation_samples_per_size must be > 0")
    return samples_per_size


def _resolve_internal_validation_compute_spectral_stats(finetune_cfg: ConfigDict) -> bool:
    """Return whether internal validation should compute Laplacian spectral metrics."""
    return as_bool(
        finetune_cfg.get("internal_validation_compute_spectral_stats", False),
        "topology_finetune.internal_validation_compute_spectral_stats",
    )


def _resolve_internal_validation_inference_batch_size(finetune_cfg: ConfigDict) -> int:
    """Return the batch size used for internal validation inference only."""
    batch_size = as_int(
        finetune_cfg.get("internal_validation_inference_batch_size", 128),
        "topology_finetune.internal_validation_inference_batch_size",
    )
    if batch_size <= 0:
        raise ValueError("topology_finetune.internal_validation_inference_batch_size must be > 0")
    return batch_size


def _resolve_embedding_cache_max_bytes(finetune_cfg: ConfigDict) -> int:
    """Return the byte ceiling for the stage-local embedding cache."""
    max_bytes = as_int(
        finetune_cfg.get("embedding_cache_max_bytes", 1_073_741_824),
        "topology_finetune.embedding_cache_max_bytes",
    )
    if max_bytes <= 0:
        raise ValueError("topology_finetune.embedding_cache_max_bytes must be > 0")
    return max_bytes


def _resolve_internal_validation_threshold(
    *,
    config: ConfigDict,
    labels: torch.Tensor,
    probabilities: torch.Tensor,
) -> tuple[float, str]:
    """Resolve the hard threshold used for internal topology validation."""
    finetune_cfg = _topology_finetune_config(config)
    evaluate_cfg = config.get("evaluate", {})
    if not isinstance(evaluate_cfg, dict):
        raise ValueError("evaluate must be a mapping")

    raw_threshold = finetune_cfg.get(
        "decision_threshold",
        cast(ConfigDict, evaluate_cfg).get("decision_threshold", 0.5),
    )
    if isinstance(raw_threshold, dict):
        mode = as_str(
            raw_threshold.get("mode", "fixed"),
            "topology_finetune.decision_threshold.mode",
        ).lower()
        if mode == "fixed":
            return (
                as_float(raw_threshold.get("value", 0.5), "topology_finetune.decision_threshold"),
                "fixed",
            )
        if mode == "best_f1_on_valid":
            return (Evaluator.best_f1_threshold(labels=labels, probabilities=probabilities), mode)
        raise ValueError(
            "topology_finetune.decision_threshold.mode must be 'fixed' or 'best_f1_on_valid'"
        )
    return (as_float(raw_threshold, "topology_finetune.decision_threshold"), "fixed")


def _move_chunk_to_device(
    *,
    chunk: SubgraphPairChunk,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert a chunk object into model-ready tensors on device."""
    bce_label = chunk.bce_label if chunk.bce_label is not None else chunk.label
    bce_mask = chunk.bce_mask if chunk.bce_mask is not None else torch.ones_like(chunk.label)
    return {
        "emb_a": chunk.emb_a.to(device),
        "emb_b": chunk.emb_b.to(device),
        "len_a": chunk.len_a.to(device),
        "len_b": chunk.len_b.to(device),
        "label": chunk.label.to(device),
        "bce_label": bce_label.to(device),
        "bce_mask": bce_mask.to(device),
        "pair_index_a": chunk.pair_index_a.to(device),
        "pair_index_b": chunk.pair_index_b.to(device),
    }


def _squeeze_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalize binary logits to a 1-D tensor."""
    if logits.dim() > 1 and logits.size(-1) == 1:
        return logits.squeeze(-1)
    return logits


def _forward_model(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Execute one forward pass under the standard model output contract."""
    if model.training and torch.is_grad_enabled():
        emb_a = batch.get("emb_a")
        emb_b = batch.get("emb_b")
        len_a = batch.get("len_a")
        len_b = batch.get("len_b")
        if emb_a is not None and emb_b is not None and len_a is not None and len_b is not None:
            logits = checkpoint(
                lambda emb_a, emb_b, len_a, len_b: model(
                    emb_a=emb_a,
                    emb_b=emb_b,
                    len_a=len_a,
                    len_b=len_b,
                )["logits"],
                emb_a,
                emb_b,
                len_a,
                len_b,
                use_reentrant=False,
            )
            return {"logits": cast(torch.Tensor, logits)}

    output = model(**batch)
    if not isinstance(output, dict):
        raise ValueError("Model forward output must be a dictionary")
    return cast(dict[str, torch.Tensor], output)


def _iter_subgraph_forward_passes(
    *,
    model: nn.Module,
    graph: nx.Graph,
    nodes: Sequence[str],
    assigned_positive_edges: frozenset[tuple[str, str]] | None = None,
    assigned_negative_edges: frozenset[tuple[str, str]] | None = None,
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
    pair_batch_size: int,
    device: torch.device,
    embedding_repository: EmbeddingRepository | None = None,
) -> Iterator[tuple[SubgraphPairChunk, dict[str, torch.Tensor], torch.Tensor]]:
    """Yield device batches and squeezed logits for one sampled subgraph."""
    for chunk in iter_subgraph_pair_chunks(
        graph=graph,
        nodes=nodes,
        assigned_positive_edges=assigned_positive_edges,
        assigned_negative_edges=assigned_negative_edges,
        cache_dir=cache_dir,
        embedding_index=embedding_index,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        pair_batch_size=pair_batch_size,
        embedding_repository=embedding_repository,
    ):
        batch = _move_chunk_to_device(chunk=chunk, device=device)
        logits = _squeeze_binary_logits(_forward_model(model=model, batch=batch)["logits"])
        yield chunk, batch, logits


def _concat_logits_and_pairs(
    *,
    model: nn.Module,
    graph: nx.Graph,
    nodes: Sequence[str],
    assigned_positive_edges: frozenset[tuple[str, str]] | None = None,
    assigned_negative_edges: frozenset[tuple[str, str]] | None = None,
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
    pair_batch_size: int,
    device: torch.device,
    embedding_repository: EmbeddingRepository | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward one sampled subgraph and collect all pair logits and labels."""
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    bce_labels_list: list[torch.Tensor] = []
    bce_mask_list: list[torch.Tensor] = []
    pair_index_a_list: list[torch.Tensor] = []
    pair_index_b_list: list[torch.Tensor] = []

    for _, batch, logits in _iter_subgraph_forward_passes(
        model=model,
        graph=graph,
        nodes=nodes,
        assigned_positive_edges=assigned_positive_edges,
        assigned_negative_edges=assigned_negative_edges,
        cache_dir=cache_dir,
        embedding_index=embedding_index,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        pair_batch_size=pair_batch_size,
        device=device,
        embedding_repository=embedding_repository,
    ):
        logits_list.append(logits)
        labels_list.append(batch["label"].float())
        bce_labels_list.append(batch["bce_label"].float())
        bce_mask_list.append(batch["bce_mask"].float())
        pair_index_a_list.append(batch["pair_index_a"])
        pair_index_b_list.append(batch["pair_index_b"])

    return (
        torch.cat(logits_list, dim=0),
        torch.cat(labels_list, dim=0),
        torch.cat(bce_labels_list, dim=0),
        torch.cat(bce_mask_list, dim=0),
        torch.cat(pair_index_a_list, dim=0),
        torch.cat(pair_index_b_list, dim=0),
    )


def _cat_padded_pair_embeddings(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate padded pair embedding tensors with a shared sequence length."""
    if not tensors:
        raise ValueError("At least one pair embedding tensor is required")
    max_tokens = max(tensor.size(1) for tensor in tensors)
    padded_tensors = [
        torch.nn.functional.pad(tensor, (0, 0, 0, max_tokens - tensor.size(1)))
        if tensor.size(1) < max_tokens
        else tensor
        for tensor in tensors
    ]
    return torch.cat(padded_tensors, dim=0)


def _batch_subgraph_chunks(
    *,
    chunks: Sequence[SubgraphPairChunk],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Concatenate pair chunks from multiple subgraphs into one model batch."""
    if not chunks:
        raise ValueError("At least one subgraph pair chunk is required")
    batch = {
        "emb_a": _cat_padded_pair_embeddings([chunk.emb_a for chunk in chunks]),
        "emb_b": _cat_padded_pair_embeddings([chunk.emb_b for chunk in chunks]),
        "len_a": torch.cat([chunk.len_a for chunk in chunks], dim=0),
        "len_b": torch.cat([chunk.len_b for chunk in chunks], dim=0),
        "label": torch.cat([chunk.label for chunk in chunks], dim=0),
        "pair_index_a": torch.cat([chunk.pair_index_a for chunk in chunks], dim=0),
        "pair_index_b": torch.cat([chunk.pair_index_b for chunk in chunks], dim=0),
        "bce_label": torch.cat([chunk.bce_label for chunk in chunks], dim=0),
        "bce_mask": torch.cat([chunk.bce_mask for chunk in chunks], dim=0),
    }
    return move_batch_to_device(batch=batch, device=device)


def _forward_subgraph_group(
    *,
    model: nn.Module,
    graph: nx.Graph,
    subgraphs: Sequence[tuple[str, ...]],
    assigned_positive_edges: Sequence[frozenset[tuple[str, str]]],
    assigned_negative_edges: Sequence[frozenset[tuple[str, str]]],
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
    pair_batch_size: int,
    device: torch.device,
    embedding_repository: EmbeddingRepository,
) -> tuple[SubgraphForwardResult, ...]:
    """Forward multiple subgraphs in one concatenated pair batch."""
    grouped_chunks: list[tuple[tuple[str, ...], tuple[SubgraphPairChunk, ...]]] = []
    all_chunks: list[SubgraphPairChunk] = []
    for nodes, subgraph_assigned_positive_edges, subgraph_assigned_negative_edges in zip(
        subgraphs,
        assigned_positive_edges,
        assigned_negative_edges,
        strict=True,
    ):
        chunks = tuple(
            iter_subgraph_pair_chunks(
                graph=graph,
                nodes=nodes,
                assigned_positive_edges=subgraph_assigned_positive_edges,
                assigned_negative_edges=subgraph_assigned_negative_edges,
                cache_dir=cache_dir,
                embedding_index=embedding_index,
                input_dim=input_dim,
                max_sequence_length=max_sequence_length,
                pair_batch_size=pair_batch_size,
                embedding_repository=embedding_repository,
            )
        )
        grouped_chunks.append((tuple(nodes), chunks))
        all_chunks.extend(chunks)

    batch = _batch_subgraph_chunks(chunks=all_chunks, device=device)
    logits = _squeeze_binary_logits(_forward_model(model=model, batch=batch)["logits"])
    results: list[SubgraphForwardResult] = []
    logit_offset = 0
    for nodes, chunks in grouped_chunks:
        pair_count = sum(int(chunk.label.numel()) for chunk in chunks)
        next_offset = logit_offset + pair_count
        results.append(
            SubgraphForwardResult(
                nodes=nodes,
                logits=logits[logit_offset:next_offset],
                labels=torch.cat([chunk.label for chunk in chunks], dim=0).to(device).float(),
                bce_labels=torch.cat([chunk.bce_label for chunk in chunks], dim=0)
                .to(device)
                .float(),
                bce_mask=torch.cat([chunk.bce_mask for chunk in chunks], dim=0).to(device).float(),
                pair_index_a=torch.cat([chunk.pair_index_a for chunk in chunks], dim=0).to(device),
                pair_index_b=torch.cat([chunk.pair_index_b for chunk in chunks], dim=0).to(device),
            )
        )
        logit_offset = next_offset
    return tuple(results)


def _subgraph_adjacencies(
    *,
    num_nodes: int,
    logits: torch.Tensor,
    labels: torch.Tensor,
    pair_index_a: torch.Tensor,
    pair_index_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build predicted and target adjacency matrices for one subgraph."""
    pred_adjacency = build_symmetric_adjacency(
        num_nodes=num_nodes,
        pair_index_a=pair_index_a,
        pair_index_b=pair_index_b,
        pair_probabilities=torch.sigmoid(logits),
    )
    target_adjacency = build_symmetric_adjacency(
        num_nodes=num_nodes,
        pair_index_a=pair_index_a,
        pair_index_b=pair_index_b,
        pair_probabilities=labels,
    )
    return pred_adjacency, target_adjacency


def _build_optimizer(config: ConfigDict, model: nn.Module) -> Optimizer:
    """Build the fine-tuning optimizer."""
    finetune_cfg = _topology_finetune_config(config)
    optimizer_cfg = finetune_cfg.get("optimizer", {})
    if not isinstance(optimizer_cfg, dict):
        raise ValueError("topology_finetune.optimizer must be a mapping")
    return torch.optim.AdamW(
        params=[parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=as_float(optimizer_cfg.get("lr", 1e-5), "topology_finetune.optimizer.lr"),
        weight_decay=as_float(
            optimizer_cfg.get("weight_decay", 0.0),
            "topology_finetune.optimizer.weight_decay",
        ),
    )


def _predict_hard_subgraph(
    *,
    model: nn.Module,
    graph: nx.Graph,
    nodes: Sequence[str],
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
    pair_batch_size: int,
    threshold: float,
    device: torch.device,
    embedding_repository: EmbeddingRepository | None = None,
) -> nx.Graph:
    """Predict one hard-thresholded node-induced subgraph."""
    node_tuple = tuple(nodes)
    pred_subgraph = nx.Graph()
    pred_subgraph.add_nodes_from(node_tuple)
    for chunk, _, logits in _iter_subgraph_forward_passes(
        model=model,
        graph=graph,
        nodes=nodes,
        cache_dir=cache_dir,
        embedding_index=embedding_index,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        pair_batch_size=pair_batch_size,
        device=device,
        embedding_repository=embedding_repository,
    ):
        probabilities = torch.sigmoid(logits).detach().cpu().tolist()
        rows = chunk.pair_index_a.tolist()
        cols = chunk.pair_index_b.tolist()
        for row_idx, col_idx, probability in zip(rows, cols, probabilities, strict=True):
            if float(probability) >= threshold:
                pred_subgraph.add_edge(node_tuple[row_idx], node_tuple[col_idx])
    return pred_subgraph


def _chunked_pair_records(
    *,
    pair_records: Sequence[InternalValidationPairRecord],
    batch_size: int,
) -> Iterator[Sequence[InternalValidationPairRecord]]:
    """Yield contiguous slices of pair records."""
    for start in range(0, len(pair_records), batch_size):
        yield pair_records[start : start + batch_size]


def _chunked_indexed_pair_records(
    *,
    indexed_pair_records: Sequence[tuple[int, InternalValidationPairRecord]],
    batch_size: int,
) -> Iterator[Sequence[tuple[int, InternalValidationPairRecord]]]:
    """Yield contiguous slices of globally indexed pair records."""
    for start in range(0, len(indexed_pair_records), batch_size):
        yield indexed_pair_records[start : start + batch_size]


def _validation_pair_batch(
    *,
    pair_records: Sequence[InternalValidationPairRecord],
    embedding_repository: EmbeddingRepository,
) -> dict[str, torch.Tensor]:
    """Materialize one internal-validation pair batch on CPU."""
    unique_proteins = sorted(
        {record.protein_a for record in pair_records}
        | {record.protein_b for record in pair_records}
    )
    embeddings = embedding_repository.get_many(unique_proteins)
    emb_a = torch.nn.utils.rnn.pad_sequence(
        [embeddings[record.protein_a] for record in pair_records],
        batch_first=True,
    )
    emb_b = torch.nn.utils.rnn.pad_sequence(
        [embeddings[record.protein_b] for record in pair_records],
        batch_first=True,
    )
    return {
        "emb_a": emb_a,
        "emb_b": emb_b,
        "len_a": torch.tensor(
            [embeddings[record.protein_a].size(0) for record in pair_records],
            dtype=torch.long,
        ),
        "len_b": torch.tensor(
            [embeddings[record.protein_b].size(0) for record in pair_records],
            dtype=torch.long,
        ),
    }


def _local_internal_validation_pair_records(
    *,
    bucket: InternalValidationNodeBucketPlan,
    accelerator: AcceleratorLike,
) -> tuple[tuple[int, InternalValidationPairRecord], ...]:
    """Return the rank-local slice of internal-validation pair records."""
    if not accelerator.use_distributed:
        return tuple(enumerate(bucket.pair_records))
    return tuple(
        (global_index, record)
        for global_index, record in enumerate(bucket.pair_records)
        if record.subgraph_index % accelerator.num_processes == accelerator.process_index
    )


def _ordered_internal_validation_predictions(
    *,
    local_indices: Sequence[int],
    local_predictions: Sequence[int],
    total_records: int,
    accelerator: AcceleratorLike,
) -> list[int]:
    """Gather sharded internal-validation predictions and restore global order."""
    if len(local_indices) != len(local_predictions):
        raise ValueError("local_indices and local_predictions must have matching lengths")

    if not accelerator.use_distributed:
        ordered: list[int | None] = [None] * total_records
        for index, prediction in zip(local_indices, local_predictions, strict=True):
            ordered[int(index)] = int(prediction)
    else:
        local_length = torch.tensor(
            [len(local_indices)],
            dtype=torch.long,
            device=accelerator.device,
        )
        gathered_lengths = accelerator.gather(local_length).detach().cpu().tolist()
        padded_indices = accelerator.pad_across_processes(
            torch.tensor(local_indices, dtype=torch.long, device=accelerator.device),
            dim=0,
            pad_index=-1,
        )
        padded_predictions = accelerator.pad_across_processes(
            torch.tensor(local_predictions, dtype=torch.long, device=accelerator.device),
            dim=0,
            pad_index=0,
        )
        gathered_indices = accelerator.gather(padded_indices).detach().cpu()
        gathered_predictions = accelerator.gather(padded_predictions).detach().cpu()
        ordered = [None] * total_records
        max_length = max(gathered_lengths, default=0)
        offset = 0
        for shard_length in gathered_lengths:
            shard_indices = gathered_indices[offset : offset + max_length][:shard_length].tolist()
            shard_predictions = gathered_predictions[offset : offset + max_length][
                :shard_length
            ].tolist()
            for index, prediction in zip(shard_indices, shard_predictions, strict=True):
                if index < 0 or index >= total_records:
                    raise ValueError(f"Internal-validation pair index out of bounds: {index}")
                if ordered[index] is not None:
                    raise ValueError(
                        f"Duplicate internal-validation prediction for pair index {index}"
                    )
                ordered[index] = int(prediction)
            offset += max_length

    missing_indices = [index for index, prediction in enumerate(ordered) if prediction is None]
    if missing_indices:
        preview = ", ".join(str(index) for index in missing_indices[:10])
        raise ValueError(f"Missing internal-validation predictions for indices: {preview}")
    return [int(prediction) for prediction in ordered if prediction is not None]


def _evaluate_internal_validation_subgraphs(
    *,
    model: nn.Module,
    validation_plan: InternalValidationPlan,
    embedding_repository: EmbeddingRepository,
    inference_batch_size: int,
    threshold: float,
    device: torch.device,
    accelerator: AcceleratorLike,
    compute_spectral_stats: bool = False,
) -> dict[str, float]:
    """Compute internal subgraph topology metrics on fixed validation samples."""
    pred_graphs_by_size: dict[int, list[nx.Graph]] = {}
    target_graphs_by_size: dict[int, list[nx.Graph]] = {}

    with torch.no_grad():
        for bucket in validation_plan.buckets:
            local_pair_records = _local_internal_validation_pair_records(
                bucket=bucket,
                accelerator=accelerator,
            )
            local_pair_indices: list[int] = []
            local_pair_predictions: list[int] = []
            for indexed_pair_batch in _chunked_indexed_pair_records(
                indexed_pair_records=local_pair_records,
                batch_size=inference_batch_size,
            ):
                pair_batch = [record for _, record in indexed_pair_batch]
                prepared_batch = move_batch_to_device(
                    batch=_validation_pair_batch(
                        pair_records=pair_batch,
                        embedding_repository=embedding_repository,
                    ),
                    device=device,
                )
                with accelerator.autocast():
                    logits = _squeeze_binary_logits(
                        _forward_model(model=model, batch=prepared_batch)["logits"]
                    )
                probabilities = torch.sigmoid(logits).detach().cpu().tolist()
                for (global_index, _), probability in zip(
                    indexed_pair_batch,
                    probabilities,
                    strict=True,
                ):
                    local_pair_indices.append(global_index)
                    local_pair_predictions.append(int(float(probability) >= threshold))

            ordered_predictions = _ordered_internal_validation_predictions(
                local_indices=local_pair_indices,
                local_predictions=local_pair_predictions,
                total_records=len(bucket.pair_records),
                accelerator=accelerator,
            )
            if accelerator.use_distributed and not accelerator.is_main_process:
                continue

            pred_subgraphs = [nx.Graph() for _ in bucket.sampled_subgraphs]
            for subgraph, nodes in zip(pred_subgraphs, bucket.sampled_subgraphs, strict=True):
                subgraph.add_nodes_from(nodes)
            for record, prediction in zip(bucket.pair_records, ordered_predictions, strict=True):
                if prediction <= 0:
                    continue
                pred_subgraphs[record.subgraph_index].add_edge(
                    bucket.sampled_subgraphs[record.subgraph_index][record.pair_index_a],
                    bucket.sampled_subgraphs[record.subgraph_index][record.pair_index_b],
                )
            pred_graphs_by_size[bucket.node_size] = pred_subgraphs
            target_graphs_by_size[bucket.node_size] = list(bucket.target_subgraphs)

    if accelerator.use_distributed and not accelerator.is_main_process:
        return _empty_internal_validation_summary()

    result = evaluate_graph_samples(
        pred_graphs_by_size=pred_graphs_by_size,
        gt_graphs_by_size=target_graphs_by_size,
        include_spectral_stats=compute_spectral_stats,
    )
    summary = cast(Mapping[str, float], result["summary"])
    return {
        metric_name: float(summary[metric_name]) for metric_name in INTERNAL_VALIDATION_SUMMARY_KEYS
    }


def _masked_bce_loss(
    *,
    logits: torch.Tensor,
    bce_labels: torch.Tensor,
    bce_mask: torch.Tensor,
    config: ConfigDict,
) -> torch.Tensor:
    """Compute BCE loss over the masked supervision pairs."""
    per_pair_bce = binary_classification_loss(
        logits=logits,
        labels=bce_labels,
        loss_config=_build_loss_config(get_section(config, "training_config")),
        reduction="none",
    )
    bce_mask_sum = bce_mask.sum()
    if float(bce_mask_sum.detach().item()) <= 0.0:
        return torch.zeros((), dtype=per_pair_bce.dtype, device=per_pair_bce.device)
    return (per_pair_bce * bce_mask).sum() / bce_mask_sum


def _initialize_train_aggregates(
    epoch_plan_size: int,
    epoch_plan: EdgeCoverEpochPlan,
) -> dict[str, float]:
    """Create aggregate storage for one topology fine-tuning epoch."""
    return {
        "bce": 0.0,
        "graph_similarity": 0.0,
        "relative_density": 0.0,
        "degree_mmd": 0.0,
        "clustering_mmd": 0.0,
        "total": 0.0,
        "planned_subgraphs": float(epoch_plan_size),
        "covered_positive_edges": float(epoch_plan.covered_positive_edges),
        "total_positive_edges": float(epoch_plan.total_positive_edges),
        "positive_edge_coverage_ratio": float(epoch_plan.positive_edge_coverage_ratio),
        "mean_positive_edge_reuse": float(epoch_plan.mean_positive_edge_reuse),
    }


def _update_train_aggregates(
    *,
    aggregates: dict[str, float],
    bce_loss: torch.Tensor,
    topology_losses: Mapping[str, torch.Tensor],
    total_loss: torch.Tensor,
) -> None:
    """Accumulate detached per-subgraph losses into epoch aggregates."""
    aggregates["bce"] += float(bce_loss.detach().item())
    aggregates["graph_similarity"] += float(topology_losses["graph_similarity"].detach().item())
    aggregates["relative_density"] += float(topology_losses["relative_density"].detach().item())
    aggregates["degree_mmd"] += float(topology_losses["degree_mmd"].detach().item())
    aggregates["clustering_mmd"] += float(topology_losses["clustering_mmd"].detach().item())
    aggregates["total"] += float(total_loss.detach().item())


def _zero_topology_losses(*, reference: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return zero-valued topology losses on the same device and dtype as ``reference``."""
    zero = torch.zeros((), dtype=reference.dtype, device=reference.device)
    return {
        "graph_similarity": zero,
        "relative_density": zero,
        "degree_mmd": zero,
        "clustering_mmd": zero,
        "total_topology": zero,
    }


def _group_topology_objectives(
    *,
    topology_losses: Mapping[str, torch.Tensor],
    loss_weights: TopologyLossWeights,
) -> dict[str, torch.Tensor]:
    """Group normalized topology losses into density and shape objectives."""
    density_objective = loss_weights.beta * topology_losses["relative_density"]
    shape_objective = (
        loss_weights.alpha * topology_losses["graph_similarity"]
        + loss_weights.gamma * topology_losses["degree_mmd"]
        + loss_weights.delta * topology_losses["clustering_mmd"]
    )
    return {
        "density": density_objective,
        "shape": shape_objective,
    }


def _task_weighted_total_loss(
    *,
    bce_loss: torch.Tensor,
    topology_objectives: Mapping[str, torch.Tensor],
    task_weights: Mapping[str, float],
) -> torch.Tensor:
    """Return the BCE + grouped-topology total under task weights."""
    total_loss = task_weights.get("bce", 1.0) * bce_loss
    total_loss = total_loss + task_weights.get("density", 1.0) * topology_objectives["density"]
    total_loss = total_loss + task_weights.get("shape", 1.0) * topology_objectives["shape"]
    return total_loss


def _gradnorm_reference_parameters(
    *,
    model: nn.Module,
    accelerator: AcceleratorLike,
) -> tuple[nn.Parameter, ...]:
    """Return the parameters used to measure grouped GradNorm magnitudes."""
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    unwrapped_model = model
    if callable(unwrap_model):
        unwrapped_model = cast(nn.Module, unwrap_model(model))
    output_head = getattr(unwrapped_model, "output_head", None)
    if isinstance(output_head, nn.Module):
        params = tuple(parameter for parameter in output_head.parameters() if parameter.requires_grad)
        if params:
            return params
    return tuple(parameter for parameter in unwrapped_model.parameters() if parameter.requires_grad)


def _graph_positive_edge_probability(graph: nx.Graph) -> float:
    """Return the train-graph positive-edge prior probability."""
    num_nodes = graph.number_of_nodes()
    if num_nodes < 2:
        raise ValueError("Topology scratch initialization requires at least 2 nodes")
    possible_edges = num_nodes * (num_nodes - 1) / 2.0
    return float(graph.number_of_edges() / possible_edges)


def _initialize_scratch_output_bias(
    *,
    model: nn.Module,
    train_graph: nx.Graph,
    accelerator: AcceleratorLike,
) -> float:
    """Initialize scratch-mode output bias from the train-graph edge prior."""
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    unwrapped_model = model
    if callable(unwrap_model):
        unwrapped_model = cast(nn.Module, unwrap_model(model))
    return initialize_output_head_bias_with_prior(
        unwrapped_model,
        positive_edge_probability=_graph_positive_edge_probability(train_graph),
    )


def _average_train_aggregates(
    *,
    aggregates: Mapping[str, float],
    num_subgraphs: int,
) -> dict[str, float]:
    """Average train loss metrics while preserving epoch coverage counters."""
    denominator = float(max(1, num_subgraphs))
    return {
        name: (value / denominator if name in TRAIN_LOSS_KEYS else value)
        for name, value in aggregates.items()
    }


def _local_subgraphs_for_rank(
    *,
    subgraphs: Sequence[tuple[str, ...]],
    distributed_context: DistributedContext,
) -> tuple[tuple[str, ...], ...]:
    """Return the rank-local slice of a shared topology epoch plan."""
    if not distributed_context.is_distributed:
        return tuple(subgraphs)
    return tuple(subgraphs[distributed_context.rank :: distributed_context.world_size])


def _local_edge_assignments_for_rank(
    *,
    assigned_edges: Sequence[frozenset[tuple[str, str]]],
    distributed_context: DistributedContext,
) -> tuple[frozenset[tuple[str, str]], ...]:
    """Return the rank-local assigned-edge slice."""
    if not distributed_context.is_distributed:
        return tuple(assigned_edges)
    return tuple(assigned_edges[distributed_context.rank :: distributed_context.world_size])


def _local_subgraph_tasks_for_rank(
    *,
    subgraphs: Sequence[tuple[str, ...]],
    assigned_positive_edges: Sequence[frozenset[tuple[str, str]]],
    assigned_negative_edges: Sequence[frozenset[tuple[str, str]]],
    distributed_context: DistributedContext,
) -> tuple[LocalSubgraphTask, ...]:
    """Return rank-local train tasks padded to equal DDP backward participation."""
    local_subgraphs = _local_subgraphs_for_rank(
        subgraphs=subgraphs,
        distributed_context=distributed_context,
    )
    local_positive_edges = _local_edge_assignments_for_rank(
        assigned_edges=assigned_positive_edges,
        distributed_context=distributed_context,
    )
    local_negative_edges = _local_edge_assignments_for_rank(
        assigned_edges=assigned_negative_edges,
        distributed_context=distributed_context,
    )
    tasks = [
        LocalSubgraphTask(
            nodes=nodes,
            assigned_positive_edges=positive_edges,
            assigned_negative_edges=negative_edges,
        )
        for nodes, positive_edges, negative_edges in zip(
            local_subgraphs,
            local_positive_edges,
            local_negative_edges,
            strict=True,
        )
    ]
    if not distributed_context.is_distributed or not subgraphs:
        return tuple(tasks)

    local_counts = [
        len(subgraphs[rank :: distributed_context.world_size])
        for rank in range(distributed_context.world_size)
    ]
    target_count = max(local_counts)
    padding_count = target_count - len(tasks)
    if padding_count <= 0:
        return tuple(tasks)

    padding_task = LocalSubgraphTask(
        nodes=tuple(subgraphs[0]),
        assigned_positive_edges=assigned_positive_edges[0],
        assigned_negative_edges=assigned_negative_edges[0],
        is_padding=True,
    )
    tasks.extend(padding_task for _ in range(padding_count))
    return tuple(tasks)


def _reduce_train_stats(
    *,
    accelerator: AcceleratorLike,
    device: torch.device,
    train_stats: Mapping[str, float],
    global_subgraph_count: int,
) -> dict[str, float]:
    """Reduce sharded train stats into one global epoch summary."""
    reduced_loss_sums = reduce_scalar_mapping(
        accelerator,
        {key: train_stats[key] for key in TRAIN_LOSS_KEYS},
        device=device,
        reduction="sum",
    )
    denominator = float(max(1, global_subgraph_count))
    reduced_train_stats = {key: value / denominator for key, value in reduced_loss_sums.items()}
    reduced_train_stats.update(
        {
            "planned_subgraphs": float(global_subgraph_count),
            "covered_positive_edges": float(train_stats["covered_positive_edges"]),
            "total_positive_edges": float(train_stats["total_positive_edges"]),
            "positive_edge_coverage_ratio": float(train_stats["positive_edge_coverage_ratio"]),
            "mean_positive_edge_reuse": float(train_stats["mean_positive_edge_reuse"]),
            "topology_loss_scale": float(train_stats.get("topology_loss_scale", 1.0)),
        }
    )
    reduced_train_stats.update(
        reduce_scalar_mapping(
            accelerator,
            {
                "edge_cover_sampling_s": train_stats["edge_cover_sampling_s"],
                "train_forward_backward_s": train_stats["train_forward_backward_s"],
            },
            device=device,
            reduction="mean",
        )
    )
    return reduced_train_stats


def _validation_topology_loss(
    *,
    loss_weights: TopologyLossWeights,
    internal_val_topology_stats: Mapping[str, float],
) -> float:
    """Return the weighted hard-metric topology penalty used in validation monitoring."""
    graph_similarity_loss = 1.0 - float(internal_val_topology_stats["graph_sim"])
    relative_density_loss = (float(internal_val_topology_stats["relative_density"]) - 1.0) ** 2
    degree_mmd = float(internal_val_topology_stats["deg_dist_mmd"])
    clustering_mmd = float(internal_val_topology_stats["cc_mmd"])
    return (
        loss_weights.alpha * graph_similarity_loss
        + loss_weights.beta * relative_density_loss
        + loss_weights.gamma * degree_mmd
        + loss_weights.delta * clustering_mmd
    )


def _fit_epoch(
    *,
    config: ConfigDict,
    model: nn.Module,
    device: torch.device,
    graph: nx.Graph,
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    optimizer: Optimizer,
    epoch_index: int,
    epoch_seed: int,
    input_dim: int,
    max_sequence_length: int,
    loss_weights: TopologyLossWeights,
    pair_batch_size: int,
    use_amp: bool,
    accelerator: AcceleratorLike,
    embedding_repository: EmbeddingRepository,
    negative_lookup: ExplicitNegativePairLookup | None,
    distributed_context: DistributedContext,
    loss_weight_schedule: TopologyLossWeightSchedule | None = None,
    loss_normalization: TopologyLossNormalizationConfig | None = None,
    gradnorm: TopologyGradNormConfig | None = None,
    adaptive_loss_state: TopologyAdaptiveLossState | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """Run one fine-tuning epoch over sampled train subgraphs."""
    del use_amp
    finetune_cfg = _topology_finetune_config(config)
    min_nodes, max_nodes = _resolve_sampling_node_bounds(finetune_cfg)
    subgraphs_per_epoch = as_int(
        finetune_cfg.get("subgraphs_per_epoch", 0),
        "topology_finetune.subgraphs_per_epoch",
    )
    strategy = as_str(finetune_cfg.get("strategy", "mixed"), "topology_finetune.strategy")
    negative_ratio = _resolve_bce_negative_ratio(finetune_cfg)
    gradient_accumulation_steps = _resolve_gradient_accumulation_steps(finetune_cfg)
    subgraphs_per_forward = _resolve_subgraphs_per_forward(finetune_cfg)
    edge_chunk_size = _resolve_edge_chunk_size(
        finetune_cfg=finetune_cfg,
        max_nodes=max_nodes,
    )
    sampling_start = time.perf_counter()
    epoch_plan = sample_edge_cover_subgraphs(
        graph=graph,
        num_subgraphs=subgraphs_per_epoch,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        strategy=strategy,
        seed=epoch_seed,
        edge_chunk_size=edge_chunk_size,
        negative_lookup=negative_lookup,
        negative_ratio=negative_ratio,
    )
    edge_cover_sampling_seconds = time.perf_counter() - sampling_start
    sampled_subgraphs = epoch_plan.subgraphs
    assigned_positive_edges = epoch_plan.assigned_positive_edges
    assigned_negative_edges = epoch_plan.assigned_negative_edges or tuple(
        frozenset() for _ in sampled_subgraphs
    )
    local_tasks = _local_subgraph_tasks_for_rank(
        subgraphs=sampled_subgraphs,
        assigned_positive_edges=assigned_positive_edges,
        assigned_negative_edges=assigned_negative_edges,
        distributed_context=distributed_context,
    )
    aggregates = _initialize_train_aggregates(len(sampled_subgraphs), epoch_plan)
    train_forward_backward_seconds = 0.0
    resolved_loss_weight_schedule = loss_weight_schedule or TopologyLossWeightSchedule()
    resolved_loss_normalization = loss_normalization or TopologyLossNormalizationConfig()
    resolved_gradnorm = gradnorm or TopologyGradNormConfig()
    resolved_adaptive_loss_state = adaptive_loss_state or TopologyAdaptiveLossState()
    current_topology_loss_scale = topology_loss_scale(
        epoch=epoch_index,
        schedule=resolved_loss_weight_schedule,
    )
    effective_loss_weights = replace(
        loss_weights,
        alpha=loss_weights.alpha * current_topology_loss_scale,
        beta=loss_weights.beta * current_topology_loss_scale,
        gamma=loss_weights.gamma * current_topology_loss_scale,
        delta=loss_weights.delta * current_topology_loss_scale,
    )
    skip_topology_during_warmup = current_topology_loss_scale <= 0.0
    real_local_subgraphs = sum(1 for task in local_tasks if not task.is_padding)
    total_subgraphs = max(1, real_local_subgraphs)
    completed_real_subgraphs = 0

    model.train()
    gradnorm_reference_parameters = (
        _gradnorm_reference_parameters(model=model, accelerator=accelerator)
        if resolved_gradnorm.enabled
        else tuple()
    )
    if local_tasks:
        optimizer.zero_grad(set_to_none=True)
    if subgraphs_per_forward == 1:
        for task_index, task in enumerate(local_tasks):
            step_start = time.perf_counter()
            with accelerator.autocast():
                (
                    logits,
                    labels,
                    bce_labels,
                    bce_mask,
                    pair_index_a,
                    pair_index_b,
                ) = _concat_logits_and_pairs(
                    model=model,
                    graph=graph,
                    nodes=task.nodes,
                    assigned_positive_edges=task.assigned_positive_edges,
                    assigned_negative_edges=task.assigned_negative_edges,
                    cache_dir=cache_dir,
                    embedding_index=embedding_index,
                    input_dim=input_dim,
                    max_sequence_length=max_sequence_length,
                    pair_batch_size=pair_batch_size,
                    device=device,
                    embedding_repository=embedding_repository,
                )
                bce_loss = _masked_bce_loss(
                    logits=logits,
                    bce_labels=bce_labels,
                    bce_mask=bce_mask,
                    config=config,
                )
                if skip_topology_during_warmup:
                    topology_losses = _zero_topology_losses(reference=bce_loss)
                    total_loss = bce_loss
                else:
                    pred_adjacency, target_adjacency = _subgraph_adjacencies(
                        num_nodes=len(task.nodes),
                        logits=logits,
                        labels=labels,
                        pair_index_a=pair_index_a,
                        pair_index_b=pair_index_b,
                    )
                    topology_losses = compute_topology_losses(
                        weights=effective_loss_weights,
                        pred_adjacency=pred_adjacency,
                        target_adjacency=target_adjacency,
                        num_nodes=len(task.nodes),
                        pair_index_a=pair_index_a,
                        pair_index_b=pair_index_b,
                        pred_pair_probabilities=torch.sigmoid(logits),
                        target_pair_probabilities=labels,
                    )
                    normalized_topology_losses = normalize_topology_loss_terms(
                        raw_terms={
                            "graph_similarity": topology_losses["graph_similarity"],
                            "relative_density": topology_losses["relative_density"],
                            "degree_mmd": topology_losses["degree_mmd"],
                            "clustering_mmd": topology_losses["clustering_mmd"],
                        },
                        state=resolved_adaptive_loss_state,
                        config=resolved_loss_normalization,
                    )
                    topology_objectives = _group_topology_objectives(
                        topology_losses=normalized_topology_losses,
                        loss_weights=effective_loss_weights,
                    )
                    task_weights = update_gradnorm_task_weights(
                        task_losses={
                            "bce": bce_loss,
                            "density": topology_objectives["density"],
                            "shape": topology_objectives["shape"],
                        },
                        state=resolved_adaptive_loss_state,
                        reference_parameters=gradnorm_reference_parameters,
                        config=resolved_gradnorm,
                    )
                    total_loss = _task_weighted_total_loss(
                        bce_loss=bce_loss,
                        topology_objectives=topology_objectives,
                        task_weights=task_weights,
                    )
                backward_loss = total_loss * 0.0 if task.is_padding else total_loss

            current_window_start = (task_index // gradient_accumulation_steps) * (
                gradient_accumulation_steps
            )
            current_window_size = min(
                gradient_accumulation_steps,
                len(local_tasks) - current_window_start,
            )
            accelerator.backward(backward_loss / float(current_window_size))
            is_accumulation_boundary = (task_index + 1) % gradient_accumulation_steps == 0
            is_last_subgraph = task_index + 1 == len(local_tasks)
            if is_accumulation_boundary or is_last_subgraph:
                optimizer.step()
                if not is_last_subgraph:
                    optimizer.zero_grad(set_to_none=True)
            train_forward_backward_seconds += time.perf_counter() - step_start
            if not task.is_padding:
                completed_real_subgraphs += 1
                _update_train_aggregates(
                    aggregates=aggregates,
                    bce_loss=bce_loss,
                    topology_losses=topology_losses,
                    total_loss=total_loss,
                )
                log_epoch_progress(
                    logger,
                    epoch=epoch_index + 1,
                    step=completed_real_subgraphs,
                    total_steps=total_subgraphs,
                    loss=aggregates["total"] / float(completed_real_subgraphs),
                )
    else:
        for window_start in range(0, len(local_tasks), gradient_accumulation_steps):
            window_end = min(window_start + gradient_accumulation_steps, len(local_tasks))
            current_window_size = window_end - window_start
            for group_start in range(window_start, window_end, subgraphs_per_forward):
                step_start = time.perf_counter()
                group_end = min(group_start + subgraphs_per_forward, window_end)
                group_tasks = local_tasks[group_start:group_end]
                with accelerator.autocast():
                    forward_results = _forward_subgraph_group(
                        model=model,
                        graph=graph,
                        subgraphs=tuple(task.nodes for task in group_tasks),
                        assigned_positive_edges=tuple(
                            task.assigned_positive_edges for task in group_tasks
                        ),
                        assigned_negative_edges=tuple(
                            task.assigned_negative_edges for task in group_tasks
                        ),
                        cache_dir=cache_dir,
                        embedding_index=embedding_index,
                        input_dim=input_dim,
                        max_sequence_length=max_sequence_length,
                        pair_batch_size=pair_batch_size,
                        device=device,
                        embedding_repository=embedding_repository,
                    )
                    group_loss = torch.zeros((), dtype=torch.float32, device=device)
                    group_updates: list[
                        tuple[
                            LocalSubgraphTask,
                            torch.Tensor,
                            Mapping[str, torch.Tensor],
                            torch.Tensor,
                        ]
                    ] = []
                    for task, result in zip(group_tasks, forward_results, strict=True):
                        bce_loss = _masked_bce_loss(
                            logits=result.logits,
                            bce_labels=result.bce_labels,
                            bce_mask=result.bce_mask,
                            config=config,
                        )
                        if skip_topology_during_warmup:
                            topology_losses = _zero_topology_losses(reference=bce_loss)
                            total_loss = bce_loss
                        else:
                            pred_adjacency, target_adjacency = _subgraph_adjacencies(
                                num_nodes=len(result.nodes),
                                logits=result.logits,
                                labels=result.labels,
                                pair_index_a=result.pair_index_a,
                                pair_index_b=result.pair_index_b,
                            )
                            topology_losses = compute_topology_losses(
                                weights=effective_loss_weights,
                                pred_adjacency=pred_adjacency,
                                target_adjacency=target_adjacency,
                                num_nodes=len(result.nodes),
                                pair_index_a=result.pair_index_a,
                                pair_index_b=result.pair_index_b,
                                pred_pair_probabilities=torch.sigmoid(result.logits),
                                target_pair_probabilities=result.labels,
                            )
                            normalized_topology_losses = normalize_topology_loss_terms(
                                raw_terms={
                                    "graph_similarity": topology_losses["graph_similarity"],
                                    "relative_density": topology_losses["relative_density"],
                                    "degree_mmd": topology_losses["degree_mmd"],
                                    "clustering_mmd": topology_losses["clustering_mmd"],
                                },
                                state=resolved_adaptive_loss_state,
                                config=resolved_loss_normalization,
                            )
                            topology_objectives = _group_topology_objectives(
                                topology_losses=normalized_topology_losses,
                                loss_weights=effective_loss_weights,
                            )
                            task_weights = update_gradnorm_task_weights(
                                task_losses={
                                    "bce": bce_loss,
                                    "density": topology_objectives["density"],
                                    "shape": topology_objectives["shape"],
                                },
                                state=resolved_adaptive_loss_state,
                                reference_parameters=gradnorm_reference_parameters,
                                config=resolved_gradnorm,
                            )
                            total_loss = _task_weighted_total_loss(
                                bce_loss=bce_loss,
                                topology_objectives=topology_objectives,
                                task_weights=task_weights,
                            )
                        task_loss = total_loss * 0.0 if task.is_padding else total_loss
                        group_loss = group_loss + task_loss
                        group_updates.append((task, bce_loss, topology_losses, total_loss))

                accelerator.backward(group_loss / float(current_window_size))
                is_last_window = window_end == len(local_tasks)
                is_last_group_in_window = group_end == window_end
                if is_last_group_in_window:
                    optimizer.step()
                    if not is_last_window:
                        optimizer.zero_grad(set_to_none=True)
                train_forward_backward_seconds += time.perf_counter() - step_start
                for task, bce_loss, topology_losses, total_loss in group_updates:
                    if task.is_padding:
                        continue
                    completed_real_subgraphs += 1
                    _update_train_aggregates(
                        aggregates=aggregates,
                        bce_loss=bce_loss,
                        topology_losses=topology_losses,
                        total_loss=total_loss,
                    )
                    log_epoch_progress(
                        logger,
                        epoch=epoch_index + 1,
                        step=completed_real_subgraphs,
                        total_steps=total_subgraphs,
                        loss=aggregates["total"] / float(completed_real_subgraphs),
                    )

    aggregates["edge_cover_sampling_s"] = edge_cover_sampling_seconds
    aggregates["train_forward_backward_s"] = train_forward_backward_seconds
    aggregates["topology_loss_scale"] = current_topology_loss_scale
    return aggregates


def _validate_embedding_cache(
    *,
    graph: nx.Graph,
    embedding_index: Mapping[str, str],
) -> None:
    """Fail fast when topology fine-tuning cannot materialize graph proteins."""
    missing_graph_nodes = sorted(
        node_id for node_id in graph.nodes if node_id not in embedding_index
    )
    if not missing_graph_nodes:
        return
    preview = ", ".join(missing_graph_nodes[:10])
    raise FileNotFoundError(
        "Embedding cache is missing train-graph proteins required by topology_finetune: "
        f"{preview} (missing={len(missing_graph_nodes)})"
    )


def _prepare_topology_finetune_stage_context(
    *,
    config: ConfigDict,
    model: nn.Module,
    device: torch.device,
    log_dir: Path,
    model_dir: Path,
    distributed_context: DistributedContext,
    accelerator: AcceleratorLike,
) -> TopologyFinetuneStageContext:
    """Build shared runtime state for topology fine-tuning."""
    run_cfg = get_section(config, "run_config")
    data_cfg = get_section(config, "data_config")
    model_cfg = get_section(config, "model_config")
    training_cfg = get_section(config, "training_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    finetune_cfg = _topology_finetune_config(config)

    input_dim = as_int(model_cfg.get("input_dim", 0), "model_config.input_dim")
    max_sequence_length = as_int(
        data_cfg.get("max_sequence_length", 64),
        "data_config.max_sequence_length",
    )
    pair_batch_size = as_int(
        finetune_cfg.get("pair_batch_size", training_cfg.get("batch_size", 8)),
        "topology_finetune.pair_batch_size",
    )
    internal_validation_inference_batch_size = _resolve_internal_validation_inference_batch_size(
        finetune_cfg
    )
    internal_validation_compute_spectral_stats = (
        _resolve_internal_validation_compute_spectral_stats(finetune_cfg)
    )
    epochs = as_int(
        finetune_cfg.get("epochs", training_cfg.get("epochs", 1)),
        "topology_finetune.epochs",
    )
    patience = as_int(
        finetune_cfg.get(
            "early_stopping_patience",
            training_cfg.get("early_stopping_patience", 5),
        ),
        "topology_finetune.early_stopping_patience",
    )
    monitor_metric = as_str(
        finetune_cfg.get("monitor_metric", "val_graph_sim"),
        "topology_finetune.monitor_metric",
    )
    run_seed = as_int(run_cfg.get("seed", 0), "run_config.seed")
    use_amp = device.type == "cuda" and as_bool(
        get_section(config, "device_config").get("use_mixed_precision", False),
        "device_config.use_mixed_precision",
    )

    train_path = Path(str(dataloader_cfg.get("train_dataset", "")))
    valid_path = Path(str(dataloader_cfg.get("valid_dataset", "")))
    train_graph, internal_val_graph = _load_supervision_graphs(config=config)
    train_negative_lookup = _load_train_negative_lookup(config=config)
    allow_embedding_generation = (
        dist.is_available() and dist.is_initialized()
        if distributed_context.is_distributed
        else True
    ) or distributed_context.is_main_process
    embedding_cache = ensure_embeddings_ready(
        config=config,
        split_paths=[train_path, valid_path],
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        allow_generation=allow_embedding_generation,
        extra_protein_ids=sorted(train_graph.nodes),
    )
    if distributed_context.is_distributed:
        accelerator.wait_for_everyone()
    _validate_embedding_cache(graph=train_graph, embedding_index=embedding_cache.index)
    internal_validation_node_sets = _build_internal_validation_node_sets(
        finetune_cfg=finetune_cfg,
        graph=internal_val_graph,
        seed=run_seed + 100_000,
    )
    internal_validation_plan = build_internal_validation_plan(
        graph=internal_val_graph,
        sampled_subgraphs=internal_validation_node_sets,
    )
    embedding_repository = EmbeddingRepository(
        cache_dir=embedding_cache.cache_dir,
        embedding_index=embedding_cache.index,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        max_cache_bytes=_resolve_embedding_cache_max_bytes(finetune_cfg),
    )
    if internal_validation_plan.protein_ids:
        embedding_repository.preload(sorted(internal_validation_plan.protein_ids))

    evaluator = Evaluator(
        metrics=["auprc"],
        loss_config=_build_loss_config(training_cfg),
        use_amp=use_amp,
        accelerator=accelerator,
    )
    return TopologyFinetuneStageContext(
        train_graph=train_graph,
        internal_val_graph=internal_val_graph,
        train_negative_lookup=train_negative_lookup,
        cache_dir=embedding_cache.cache_dir,
        embedding_index=embedding_cache.index,
        embedding_repository=embedding_repository,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        pair_batch_size=pair_batch_size,
        internal_validation_inference_batch_size=internal_validation_inference_batch_size,
        internal_validation_compute_spectral_stats=internal_validation_compute_spectral_stats,
        epochs=epochs,
        run_seed=run_seed,
        use_amp=use_amp,
        optimizer=_build_optimizer(config=config, model=model),
        evaluator=evaluator,
        loss_weights=_parse_loss_weights(config),
        loss_weight_schedule=_parse_loss_weight_schedule(finetune_cfg),
        loss_normalization=_parse_loss_normalization_config(finetune_cfg),
        gradnorm=_parse_gradnorm_config(finetune_cfg),
        adaptive_loss_state=TopologyAdaptiveLossState(),
        monitor_metric=monitor_metric,
        early_stopping=EarlyStopping(
            patience=patience,
            mode=_resolve_monitor_mode(monitor_metric),
        ),
        best_checkpoint_path=model_dir / "best_model.pth",
        metrics_path=log_dir / "topology_finetune_metrics.json",
        csv_path=log_dir / "topology_finetune_step.csv",
        internal_validation_node_sets=internal_validation_node_sets,
        internal_validation_plan=internal_validation_plan,
    )


def _evaluate_validation_epoch(
    *,
    config: ConfigDict,
    model: nn.Module,
    dataloaders: dict[str, DataLoader[dict[str, object]]],
    device: torch.device,
    context: TopologyFinetuneStageContext,
    loss_weights: TopologyLossWeights,
) -> ValidationEpochResult:
    """Run pairwise validation and sampled topology validation for one epoch."""
    model.eval()
    with torch.no_grad():
        val_pair_start = time.perf_counter()
        labels, probabilities, average_loss = context.evaluator.collect_probabilities_and_labels(
            model=model,
            data_loader=dataloaders["valid"],
            device=device,
        )
        val_pair_stats = context.evaluator.metrics_from_outputs(
            labels=labels,
            probabilities=probabilities,
            average_loss=average_loss,
            prefix="val",
        )
        val_pair_pass_seconds = time.perf_counter() - val_pair_start
    threshold_start = time.perf_counter()
    decision_threshold, _ = _resolve_internal_validation_threshold(
        config=config,
        labels=labels,
        probabilities=probabilities,
    )
    threshold_resolution_seconds = time.perf_counter() - threshold_start
    internal_validation_start = time.perf_counter()
    internal_val_topology_stats = _evaluate_internal_validation_subgraphs(
        model=model,
        validation_plan=context.internal_validation_plan,
        embedding_repository=context.embedding_repository,
        inference_batch_size=context.internal_validation_inference_batch_size,
        threshold=decision_threshold,
        device=device,
        accelerator=context.evaluator.accelerator,
        compute_spectral_stats=context.internal_validation_compute_spectral_stats,
    )
    internal_validation_seconds = time.perf_counter() - internal_validation_start
    val_pair_loss = float(val_pair_stats.get("val_loss", 0.0))
    val_topology_loss = _validation_topology_loss(
        loss_weights=loss_weights,
        internal_val_topology_stats=internal_val_topology_stats,
    )
    return ValidationEpochResult(
        decision_threshold=decision_threshold,
        val_pair_stats=val_pair_stats,
        internal_val_topology_stats=internal_val_topology_stats,
        val_pair_pass_seconds=val_pair_pass_seconds,
        threshold_resolution_seconds=threshold_resolution_seconds,
        internal_validation_seconds=internal_validation_seconds,
        val_topology_loss=val_topology_loss,
        val_total_loss=val_pair_loss + val_topology_loss,
    )


def _build_epoch_csv_row(
    *,
    epoch: int,
    epoch_seconds: float,
    train_stats: Mapping[str, float],
    validation_result: ValidationEpochResult,
    optimizer: Optimizer,
    peak_gpu_mem_mb: float,
) -> dict[str, float | int | str]:
    """Build the persisted CSV row for one fine-tuning epoch."""
    internal_val_topology_stats = validation_result.internal_val_topology_stats
    val_pair_stats = validation_result.val_pair_stats
    return {
        "Epoch": epoch,
        "Epoch Time": epoch_seconds,
        "Train BCE Loss": train_stats["bce"],
        "Train GS Loss": train_stats["graph_similarity"],
        "Train RD Loss": train_stats["relative_density"],
        "Train Deg MMD": train_stats["degree_mmd"],
        "Train Clus MMD": train_stats["clustering_mmd"],
        "Train Total Loss": train_stats["total"],
        "Val Pair Loss": float(val_pair_stats.get("val_loss", 0.0)),
        "Val Topology Loss": validation_result.val_topology_loss,
        "Val Loss": validation_result.val_total_loss,
        "Val auprc": float(val_pair_stats.get("val_auprc", 0.0)),
        "Internal Val graph_sim": internal_val_topology_stats["graph_sim"],
        "Internal Val relative_density": internal_val_topology_stats["relative_density"],
        "Internal Val deg_dist_mmd": internal_val_topology_stats["deg_dist_mmd"],
        "Internal Val cc_mmd": internal_val_topology_stats["cc_mmd"],
        "Planned Subgraphs": int(train_stats["planned_subgraphs"]),
        "Covered Positive Edges": int(train_stats["covered_positive_edges"]),
        "Total Positive Edges": int(train_stats["total_positive_edges"]),
        "Positive Edge Coverage Ratio": train_stats["positive_edge_coverage_ratio"],
        "Mean Positive Edge Reuse": train_stats["mean_positive_edge_reuse"],
        "edge_cover_sampling_s": train_stats["edge_cover_sampling_s"],
        "train_forward_backward_s": train_stats["train_forward_backward_s"],
        "val_pair_pass_s": validation_result.val_pair_pass_seconds,
        "val_threshold_s": validation_result.threshold_resolution_seconds,
        "internal_val_topology_s": validation_result.internal_validation_seconds,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "Topology Loss Scale": train_stats["topology_loss_scale"],
        "Learning Rate": float(optimizer.param_groups[0]["lr"]),
    }


def _build_best_metrics_payload(
    *,
    epoch: int,
    monitor_metric: str,
    monitor_value: float,
    train_stats: Mapping[str, float],
    validation_result: ValidationEpochResult,
) -> dict[str, float | str]:
    """Build the JSON payload for the best topology fine-tuning checkpoint."""
    val_pair_stats = validation_result.val_pair_stats
    internal_val_topology_stats = validation_result.internal_val_topology_stats
    return {
        "epoch": float(epoch),
        "monitor_metric": monitor_metric,
        "monitor_value": monitor_value,
        "val_loss": validation_result.val_total_loss,
        "val_pair_loss": float(val_pair_stats.get("val_loss", 0.0)),
        "val_topology_loss": validation_result.val_topology_loss,
        "val_total_loss": validation_result.val_total_loss,
        "val_auprc": float(val_pair_stats.get("val_auprc", 0.0)),
        "internal_val_graph_sim": internal_val_topology_stats["graph_sim"],
        "internal_val_relative_density": internal_val_topology_stats["relative_density"],
        "internal_val_deg_dist_mmd": internal_val_topology_stats["deg_dist_mmd"],
        "internal_val_cc_mmd": internal_val_topology_stats["cc_mmd"],
        "planned_subgraphs": train_stats["planned_subgraphs"],
        "covered_positive_edges": train_stats["covered_positive_edges"],
        "total_positive_edges": train_stats["total_positive_edges"],
        "positive_edge_coverage_ratio": train_stats["positive_edge_coverage_ratio"],
        "mean_positive_edge_reuse": train_stats["mean_positive_edge_reuse"],
    }


def run_topology_finetuning_stage(
    runtime: PipelineRuntime,
    model: nn.Module,
    dataloaders: dict[str, DataLoader[dict[str, object]]],
    *,
    checkpoint_path: Path | None,
) -> Path:
    """Fine-tune a pairwise scorer with PRING graph-topology losses."""
    config = runtime.config.raw
    device = runtime.device
    model_name, _ = extract_model_kwargs(config)
    run_id = runtime.stage_run_id("topology_finetune")
    paths = runtime.stage_paths("topology_finetune")
    log_dir = paths.log_dir
    model_dir = paths.model_dir
    logger = runtime.stage_logger("topology_finetune", log_dir / "log.log")
    finetune_cfg = _topology_finetune_config(config)
    init_mode = _resolve_init_mode(finetune_cfg)
    checkpoint_path_resolved = Path(checkpoint_path) if checkpoint_path is not None else None
    if runtime.is_main_process:
        log_stage_event(
            logger,
            "stage_start",
            run_id=run_id,
            checkpoint=checkpoint_path_resolved,
            init_mode=init_mode,
        )
    if init_mode == "warm_start":
        if checkpoint_path_resolved is None:
            raise ValueError(
                "checkpoint_path is required when topology_finetune.init_mode='warm_start'"
            )
        runtime.load_checkpoint(model, checkpoint_path_resolved)

    context = _prepare_topology_finetune_stage_context(
        config=config,
        model=model,
        device=device,
        log_dir=log_dir,
        model_dir=model_dir,
        distributed_context=runtime.distributed,
        accelerator=runtime.accelerator,
    )
    if init_mode == "scratch":
        applied_bias = _initialize_scratch_output_bias(
            model=model,
            train_graph=context.train_graph,
            accelerator=runtime.accelerator,
        )
        if runtime.is_main_process:
            log_stage_event(
                logger,
                "scratch_output_bias_initialized",
                positive_edge_probability=_graph_positive_edge_probability(context.train_graph),
                bias=applied_bias,
            )
    stage_model, prepared_optimizer = cast(
        tuple[nn.Module, Optimizer],
        runtime.accelerator.prepare(model, context.optimizer),
    )
    context = replace(context, optimizer=prepared_optimizer)
    best_metrics: dict[str, float | str] = {}

    if runtime.is_main_process:
        log_stage_event(
            logger,
            "finetune_config",
            epochs=context.epochs,
            monitor=context.monitor_metric,
            internal_validation_subgraphs=context.internal_validation_plan.total_subgraphs,
            internal_validation_node_sizes=sorted(context.internal_validation_node_sets),
            internal_validation_pairs=context.internal_validation_plan.total_pairs,
            pair_batch_size=context.pair_batch_size,
            internal_validation_inference_batch_size=(
                context.internal_validation_inference_batch_size
            ),
            internal_validation_compute_spectral_stats=(
                context.internal_validation_compute_spectral_stats
            ),
        )
        if "validation_subgraphs" in finetune_cfg:
            log_stage_event(
                logger,
                "legacy_validation_subgraphs_ignored",
                configured=finetune_cfg.get("validation_subgraphs"),
                effective_internal_validation_subgraphs=context.internal_validation_plan.total_subgraphs,
                reason="fixed_topology_eval_buckets",
            )

    for epoch in range(context.epochs):
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_stats = _fit_epoch(
            config=config,
            model=stage_model,
            device=device,
            graph=context.train_graph,
            cache_dir=context.cache_dir,
            embedding_index=context.embedding_index,
            optimizer=context.optimizer,
            epoch_index=epoch,
            epoch_seed=_resolve_epoch_seed(
                run_seed=context.run_seed,
                epoch_index=epoch,
                distributed_context=runtime.distributed,
            ),
            input_dim=context.input_dim,
            max_sequence_length=context.max_sequence_length,
            loss_weights=context.loss_weights,
            loss_weight_schedule=context.loss_weight_schedule,
            loss_normalization=context.loss_normalization,
            gradnorm=context.gradnorm,
            adaptive_loss_state=context.adaptive_loss_state,
            pair_batch_size=context.pair_batch_size,
            use_amp=context.use_amp,
            accelerator=runtime.accelerator,
            embedding_repository=context.embedding_repository,
            negative_lookup=context.train_negative_lookup,
            distributed_context=runtime.distributed,
            logger=logger,
        )
        train_stats = _reduce_train_stats(
            accelerator=runtime.accelerator,
            device=device,
            train_stats=train_stats,
            global_subgraph_count=int(train_stats["planned_subgraphs"]),
        )
        effective_validation_loss_weights = replace(
            context.loss_weights,
            alpha=context.loss_weights.alpha * float(train_stats["topology_loss_scale"]),
            beta=context.loss_weights.beta * float(train_stats["topology_loss_scale"]),
            gamma=context.loss_weights.gamma * float(train_stats["topology_loss_scale"]),
            delta=context.loss_weights.delta * float(train_stats["topology_loss_scale"]),
        )

        # Every rank must execute the validation path before the next collective.
        # Restricting validation to rank 0 lets other ranks reach `_sync_flag()`
        # early and deadlock inside NCCL while rank 0 is still evaluating.
        validation_result = _evaluate_validation_epoch(
            config=config,
            model=stage_model,
            dataloaders=dataloaders,
            device=device,
            context=context,
            loss_weights=effective_validation_loss_weights,
        )
        peak_gpu_mem_mb = (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
            if device.type == "cuda"
            else 0.0
        )

        monitor_value = _resolve_monitor_value(
            monitor_metric=context.monitor_metric,
            val_pair_stats=validation_result.val_pair_stats,
            internal_val_topology_stats=validation_result.internal_val_topology_stats,
            val_total_loss=validation_result.val_total_loss,
        )
        should_stop = False
        save_best_checkpoint = False
        if runtime.is_main_process:
            csv_row: dict[str, float | int | str] = _build_epoch_csv_row(
                epoch=epoch + 1,
                epoch_seconds=time.perf_counter() - epoch_start,
                train_stats=train_stats,
                validation_result=validation_result,
                optimizer=context.optimizer,
                peak_gpu_mem_mb=peak_gpu_mem_mb,
            )
            append_csv_row(
                csv_path=context.csv_path,
                row=csv_row,
                fieldnames=TOPOLOGY_FINETUNE_CSV_COLUMNS,
            )
            improved, should_stop = context.early_stopping.update(monitor_value)
            save_best_checkpoint = improved
        save_best_checkpoint = _sync_flag(runtime, save_best_checkpoint)
        if save_best_checkpoint:
            runtime.save_checkpoint(stage_model, context.best_checkpoint_path)
            if runtime.is_main_process:
                best_metrics = _build_best_metrics_payload(
                    epoch=epoch + 1,
                    monitor_metric=context.monitor_metric,
                    monitor_value=monitor_value,
                    train_stats=train_stats,
                    validation_result=validation_result,
                )
                context.metrics_path.write_text(
                    json.dumps(best_metrics, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                log_stage_event(
                    logger,
                    "best_saved",
                    epoch=epoch + 1,
                    monitor=context.monitor_metric,
                    value=monitor_value,
                )
        if runtime.is_main_process:
            log_stage_event(
                logger,
                "epoch_done",
                epoch=epoch + 1,
                train_loss=train_stats["total"],
                val_auprc=float(validation_result.val_pair_stats.get("val_auprc", 0.0)),
                internal_val_graph_sim=validation_result.internal_val_topology_stats["graph_sim"],
                planned_subgraphs=int(train_stats["planned_subgraphs"]),
                covered_positive_edges=int(train_stats["covered_positive_edges"]),
                total_positive_edges=int(train_stats["total_positive_edges"]),
                positive_edge_coverage_ratio=train_stats["positive_edge_coverage_ratio"],
                mean_positive_edge_reuse=train_stats["mean_positive_edge_reuse"],
                edge_cover_sampling_s=train_stats["edge_cover_sampling_s"],
                train_forward_backward_s=train_stats["train_forward_backward_s"],
                val_pair_pass_s=validation_result.val_pair_pass_seconds,
                val_threshold_s=validation_result.threshold_resolution_seconds,
                internal_val_topology_s=validation_result.internal_validation_seconds,
                peak_gpu_mem_mb=peak_gpu_mem_mb,
                topo_loss_scale=train_stats["topology_loss_scale"],
            )

        if runtime.is_distributed:
            stop_flag = torch.tensor([1 if should_stop else 0], device=device, dtype=torch.int64)
            dist.broadcast(stop_flag, src=0)
            should_stop = bool(int(stop_flag.item()))
        if should_stop:
            if runtime.is_main_process:
                log_stage_event(logger, "early_stop", epoch=epoch + 1)
            break

    fallback_save = runtime.is_main_process and not context.best_checkpoint_path.exists()
    if _sync_flag(runtime, fallback_save):
        runtime.save_checkpoint(stage_model, context.best_checkpoint_path)
    if runtime.is_main_process and fallback_save and not best_metrics:
        context.metrics_path.write_text(
            json.dumps(
                {"monitor_metric": context.monitor_metric, "monitor_value": 0.0},
                indent=2,
            ),
            encoding="utf-8",
        )
    if runtime.is_main_process:
        log_stage_event(logger, "stage_done", run_id=run_id)
    runtime.barrier()
    return context.best_checkpoint_path


def _sync_flag(runtime: PipelineRuntime, flag: bool) -> bool:
    """Return a flag that is true on all ranks when any rank reports true."""
    if not runtime.is_distributed:
        return flag
    flag_tensor = torch.tensor([1 if flag else 0], device=runtime.device, dtype=torch.int64)
    reduced = runtime.accelerator.reduce(flag_tensor, reduction="sum")
    return bool(int(reduced.item()) > 0)
