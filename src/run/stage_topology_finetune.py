"""Graph-topology fine-tuning stage for PRING training subgraphs."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
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
from src.run.stage_evaluate import _resolve_decision_threshold
from src.run.stage_train import (
    _build_loss_config,
    _build_stage_runtime,
    _load_checkpoint,
    _save_checkpoint,
)
from src.topology.finetune_data import (
    TOPOLOGY_EVAL_NODE_SIZES,
    TOPOLOGY_EVAL_SAMPLES_PER_SIZE,
    EdgeCoverEpochPlan,
    ExplicitNegativePairLookup,
    SubgraphPairChunk,
    build_explicit_negative_lookup,
    build_pair_supervision_graph,
    iter_subgraph_pair_chunks,
    load_split_node_ids,
    sample_edge_cover_subgraphs,
    sample_topology_evaluation_subgraphs,
)
from src.topology.finetune_losses import (
    TopologyLossWeights,
    build_symmetric_adjacency,
    compute_topology_losses,
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
from src.utils.distributed import DistributedContext, distributed_barrier
from src.utils.early_stop import EarlyStopping
from src.utils.logging import append_csv_row, log_stage_event
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


@dataclass(frozen=True)
class TopologyFinetuneStageContext:
    """Runtime artifacts reused across topology fine-tuning epochs."""

    train_graph: nx.Graph
    internal_val_graph: nx.Graph
    train_negative_lookup: ExplicitNegativePairLookup
    cache_dir: Path
    embedding_index: Mapping[str, str]
    input_dim: int
    max_sequence_length: int
    pair_batch_size: int
    epochs: int
    run_seed: int
    use_amp: bool
    scaler: torch.amp.GradScaler
    optimizer: Optimizer
    evaluator: Evaluator
    threshold_probe: Evaluator
    loss_weights: TopologyLossWeights
    overlap_penalty: float
    monitor_metric: str
    early_stopping: EarlyStopping
    best_checkpoint_path: Path
    metrics_path: Path
    csv_path: Path
    internal_validation_node_sets: Mapping[int, Sequence[tuple[str, ...]]]


@dataclass(frozen=True)
class ValidationEpochResult:
    """Validation outputs used for monitoring and artifact writing."""

    decision_threshold: float
    val_pair_stats: Mapping[str, float]
    internal_val_topology_stats: Mapping[str, float]


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
) -> float:
    """Resolve the scalar value used for checkpoint selection and early stopping."""
    return float(
        {
            "val_loss": float(val_pair_stats.get("val_loss", 0.0)),
            "internal_val_graph_sim": internal_val_topology_stats["graph_sim"],
            "val_graph_sim": internal_val_topology_stats["graph_sim"],
            "internal_val_relative_density": -abs(
                internal_val_topology_stats["relative_density"] - 1.0
            ),
            "val_relative_density": -abs(internal_val_topology_stats["relative_density"] - 1.0),
            "val_auprc": float(val_pair_stats.get("val_auprc", 0.0)),
        }.get(monitor_metric, internal_val_topology_stats["graph_sim"])
    )


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
    )


def _resolve_sampling_node_bounds(finetune_cfg: ConfigDict) -> tuple[int, int]:
    """Resolve subgraph node limits for topology fine-tuning."""
    max_nodes = as_int(finetune_cfg.get("max_nodes", 20), "topology_finetune.max_nodes")
    min_nodes = as_int(finetune_cfg.get("min_nodes", max_nodes), "topology_finetune.min_nodes")
    if min_nodes <= 0:
        raise ValueError("topology_finetune.min_nodes must be positive")
    if max_nodes <= 0:
        raise ValueError("topology_finetune.max_nodes must be positive")
    if min_nodes > max_nodes:
        raise ValueError("topology_finetune.min_nodes must be <= topology_finetune.max_nodes")
    return min_nodes, max_nodes


def _resolve_overlap_penalty(finetune_cfg: ConfigDict) -> float:
    """Return overlap penalty for edge-cover sampling."""
    overlap_penalty_raw = finetune_cfg.get("overlap_penalty")
    if overlap_penalty_raw is None:
        epoch_sampling_cfg = finetune_cfg.get("epoch_sampling", {})
        if epoch_sampling_cfg is None:
            epoch_sampling_cfg = {}
        if not isinstance(epoch_sampling_cfg, dict):
            raise ValueError("topology_finetune.epoch_sampling must be a mapping")
        overlap_penalty_raw = epoch_sampling_cfg.get("overlap_penalty", 0.5)

    overlap_penalty = as_float(
        overlap_penalty_raw,
        "topology_finetune.overlap_penalty",
    )
    if overlap_penalty < 0.0:
        raise ValueError("topology_finetune.overlap_penalty must be >= 0")
    return overlap_penalty


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
    """Return the per-subgraph negative-to-positive BCE ratio."""
    negative_ratio = as_int(
        finetune_cfg.get("bce_negative_ratio", 5),
        "topology_finetune.bce_negative_ratio",
    )
    if negative_ratio < 0:
        raise ValueError("topology_finetune.bce_negative_ratio must be >= 0")
    return negative_ratio


def _build_internal_validation_node_sets(
    *,
    finetune_cfg: ConfigDict,
    graph: nx.Graph,
    seed: int,
) -> dict[int, list[tuple[str, ...]]]:
    """Build topology-evaluate-style node buckets for internal validation."""
    del finetune_cfg
    return sample_topology_evaluation_subgraphs(
        graph=graph,
        seed=seed,
        strategy="mixed",
        node_sizes=TOPOLOGY_EVAL_NODE_SIZES,
        samples_per_size=TOPOLOGY_EVAL_SAMPLES_PER_SIZE,
    )


def _resolve_internal_validation_threshold(
    *,
    config: ConfigDict,
    evaluator: Evaluator,
    model: nn.Module,
    dataloaders: dict[str, DataLoader[dict[str, object]]],
    device: torch.device,
) -> tuple[float, str]:
    """Resolve the hard threshold used for internal topology validation."""
    finetune_cfg = _topology_finetune_config(config)
    evaluate_cfg = config.get("evaluate", {})
    if not isinstance(evaluate_cfg, dict):
        raise ValueError("evaluate must be a mapping")

    threshold_cfg: ConfigDict = {
        "decision_threshold": finetune_cfg.get(
            "decision_threshold",
            cast(ConfigDict, evaluate_cfg).get("decision_threshold", 0.5),
        )
    }
    return _resolve_decision_threshold(
        eval_cfg=threshold_cfg,
        evaluator=evaluator,
        model=model,
        dataloaders=dataloaders,
        device=device,
    )


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
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
    pair_batch_size: int,
    device: torch.device,
    negative_lookup: ExplicitNegativePairLookup | None = None,
    negative_ratio: int = 1,
    seed: int | None = None,
) -> Iterator[tuple[SubgraphPairChunk, dict[str, torch.Tensor], torch.Tensor]]:
    """Yield device batches and squeezed logits for one sampled subgraph."""
    for chunk in iter_subgraph_pair_chunks(
        graph=graph,
        nodes=nodes,
        cache_dir=cache_dir,
        embedding_index=embedding_index,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        pair_batch_size=pair_batch_size,
        negative_lookup=negative_lookup,
        negative_ratio=negative_ratio,
        seed=seed,
    ):
        batch = _move_chunk_to_device(chunk=chunk, device=device)
        logits = _squeeze_binary_logits(_forward_model(model=model, batch=batch)["logits"])
        yield chunk, batch, logits


def _concat_logits_and_pairs(
    *,
    model: nn.Module,
    graph: nx.Graph,
    nodes: Sequence[str],
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
    pair_batch_size: int,
    device: torch.device,
    negative_lookup: ExplicitNegativePairLookup | None = None,
    negative_ratio: int = 1,
    seed: int | None = None,
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
        cache_dir=cache_dir,
        embedding_index=embedding_index,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        pair_batch_size=pair_batch_size,
        device=device,
        negative_lookup=negative_lookup,
        negative_ratio=negative_ratio,
        seed=seed,
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
    ):
        probabilities = torch.sigmoid(logits).detach().cpu().tolist()
        rows = chunk.pair_index_a.tolist()
        cols = chunk.pair_index_b.tolist()
        for row_idx, col_idx, probability in zip(rows, cols, probabilities, strict=True):
            if float(probability) >= threshold:
                pred_subgraph.add_edge(node_tuple[row_idx], node_tuple[col_idx])
    return pred_subgraph


def _evaluate_internal_validation_subgraphs(
    *,
    model: nn.Module,
    graph: nx.Graph,
    sampled_subgraphs: Mapping[int, Sequence[tuple[str, ...]]],
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
    pair_batch_size: int,
    threshold: float,
    device: torch.device,
) -> dict[str, float]:
    """Compute internal subgraph topology metrics on fixed validation samples."""
    pred_graphs_by_size: dict[int, list[nx.Graph]] = {}
    target_graphs_by_size: dict[int, list[nx.Graph]] = {}

    with torch.no_grad():
        for node_size, node_sets in sampled_subgraphs.items():
            pred_graphs_by_size[node_size] = []
            target_graphs_by_size[node_size] = []
            for nodes in node_sets:
                pred_subgraph = _predict_hard_subgraph(
                    model=model,
                    graph=graph,
                    nodes=nodes,
                    cache_dir=cache_dir,
                    embedding_index=embedding_index,
                    input_dim=input_dim,
                    max_sequence_length=max_sequence_length,
                    pair_batch_size=pair_batch_size,
                    threshold=threshold,
                    device=device,
                )
                target_subgraph = graph.subgraph(nodes).copy()
                pred_graphs_by_size[node_size].append(pred_subgraph)
                target_graphs_by_size[node_size].append(target_subgraph)

    result = evaluate_graph_samples(
        pred_graphs_by_size=pred_graphs_by_size,
        gt_graphs_by_size=target_graphs_by_size,
    )
    return cast(dict[str, float], result["summary"])


def _all_reduce_mean(
    *,
    value: float,
    distributed_context: DistributedContext,
    device: torch.device,
) -> float:
    """Average one scalar across all distributed ranks."""
    if not distributed_context.is_distributed or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(max(1, distributed_context.world_size))
    return float(tensor.item())


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
    rank_seed: int,
    input_dim: int,
    max_sequence_length: int,
    loss_weights: TopologyLossWeights,
    pair_batch_size: int,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    negative_lookup: ExplicitNegativePairLookup | None,
) -> dict[str, float]:
    """Run one fine-tuning epoch over sampled train subgraphs."""
    finetune_cfg = _topology_finetune_config(config)
    min_nodes, max_nodes = _resolve_sampling_node_bounds(finetune_cfg)
    subgraphs_per_epoch = as_int(
        finetune_cfg.get("subgraphs_per_epoch", 0),
        "topology_finetune.subgraphs_per_epoch",
    )
    strategy = as_str(finetune_cfg.get("strategy", "mixed"), "topology_finetune.strategy")
    overlap_penalty = _resolve_overlap_penalty(finetune_cfg)
    negative_ratio = _resolve_bce_negative_ratio(finetune_cfg)
    epoch_plan = sample_edge_cover_subgraphs(
        graph=graph,
        num_subgraphs=subgraphs_per_epoch,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        strategy=strategy,
        seed=rank_seed + epoch_index,
        overlap_penalty=overlap_penalty,
    )
    sampled_subgraphs = epoch_plan.subgraphs
    aggregates = _initialize_train_aggregates(len(sampled_subgraphs), epoch_plan)

    model.train()
    for subgraph_index, nodes in enumerate(sampled_subgraphs):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
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
                nodes=nodes,
                cache_dir=cache_dir,
                embedding_index=embedding_index,
                input_dim=input_dim,
                max_sequence_length=max_sequence_length,
                pair_batch_size=pair_batch_size,
                device=device,
                negative_lookup=negative_lookup,
                negative_ratio=negative_ratio,
                seed=(rank_seed + epoch_index) * 1000 + subgraph_index,
            )
            bce_loss = _masked_bce_loss(
                logits=logits,
                bce_labels=bce_labels,
                bce_mask=bce_mask,
                config=config,
            )
            pred_adjacency, target_adjacency = _subgraph_adjacencies(
                num_nodes=len(nodes),
                logits=logits,
                labels=labels,
                pair_index_a=pair_index_a,
                pair_index_b=pair_index_b,
            )
            topology_losses = compute_topology_losses(
                pred_adjacency=pred_adjacency,
                target_adjacency=target_adjacency,
                weights=loss_weights,
            )
            total_loss = bce_loss + topology_losses["total_topology"]

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        _update_train_aggregates(
            aggregates=aggregates,
            bce_loss=bce_loss,
            topology_losses=topology_losses,
            total_loss=total_loss,
        )

    return _average_train_aggregates(aggregates=aggregates, num_subgraphs=len(sampled_subgraphs))


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
        distributed_barrier(distributed_context)
    _validate_embedding_cache(graph=train_graph, embedding_index=embedding_cache.index)

    evaluator = Evaluator(
        metrics=["auprc"],
        loss_config=_build_loss_config(training_cfg),
        use_amp=use_amp,
    )
    threshold_probe = Evaluator(
        metrics=["f1"],
        loss_config=_build_loss_config(training_cfg),
        use_amp=use_amp,
    )
    overlap_penalty = _resolve_overlap_penalty(finetune_cfg)
    return TopologyFinetuneStageContext(
        train_graph=train_graph,
        internal_val_graph=internal_val_graph,
        train_negative_lookup=train_negative_lookup,
        cache_dir=embedding_cache.cache_dir,
        embedding_index=embedding_cache.index,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        pair_batch_size=pair_batch_size,
        epochs=epochs,
        run_seed=run_seed,
        use_amp=use_amp,
        scaler=torch.amp.GradScaler("cuda", enabled=use_amp),
        optimizer=_build_optimizer(config=config, model=model),
        evaluator=evaluator,
        threshold_probe=threshold_probe,
        loss_weights=_parse_loss_weights(config),
        overlap_penalty=overlap_penalty,
        monitor_metric=monitor_metric,
        early_stopping=EarlyStopping(
            patience=patience,
            mode=_resolve_monitor_mode(monitor_metric),
        ),
        best_checkpoint_path=model_dir / "best_model.pth",
        metrics_path=log_dir / "topology_finetune_metrics.json",
        csv_path=log_dir / "topology_finetune_step.csv",
        internal_validation_node_sets=_build_internal_validation_node_sets(
            finetune_cfg=finetune_cfg,
            graph=internal_val_graph,
            seed=run_seed + 100_000,
        ),
    )


def _evaluate_validation_epoch(
    *,
    config: ConfigDict,
    model: nn.Module,
    dataloaders: dict[str, DataLoader[dict[str, object]]],
    device: torch.device,
    context: TopologyFinetuneStageContext,
) -> ValidationEpochResult:
    """Run pairwise validation and sampled topology validation for one epoch."""
    model.eval()
    with torch.no_grad():
        val_pair_stats = context.evaluator.evaluate(
            model=model,
            data_loader=dataloaders["valid"],
            device=device,
            prefix="val",
        )
    decision_threshold, _ = _resolve_internal_validation_threshold(
        config=config,
        evaluator=context.threshold_probe,
        model=model,
        dataloaders=dataloaders,
        device=device,
    )
    internal_val_topology_stats = _evaluate_internal_validation_subgraphs(
        model=model,
        graph=context.internal_val_graph,
        sampled_subgraphs=context.internal_validation_node_sets,
        cache_dir=context.cache_dir,
        embedding_index=context.embedding_index,
        input_dim=context.input_dim,
        max_sequence_length=context.max_sequence_length,
        pair_batch_size=context.pair_batch_size,
        threshold=decision_threshold,
        device=device,
    )
    return ValidationEpochResult(
        decision_threshold=decision_threshold,
        val_pair_stats=val_pair_stats,
        internal_val_topology_stats=internal_val_topology_stats,
    )


def _build_epoch_csv_row(
    *,
    epoch: int,
    epoch_seconds: float,
    train_stats: Mapping[str, float],
    validation_result: ValidationEpochResult,
    optimizer: Optimizer,
) -> dict[str, float | int]:
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
        "Val Loss": float(val_pair_stats.get("val_loss", 0.0)),
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
        "val_loss": float(val_pair_stats.get("val_loss", 0.0)),
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
    config: ConfigDict,
    model: nn.Module,
    device: torch.device,
    dataloaders: dict[str, DataLoader[dict[str, object]]],
    run_id: str,
    checkpoint_path: Path | None,
    distributed_context: DistributedContext,
) -> Path:
    """Fine-tune a pairwise scorer with PRING graph-topology losses."""
    model_name, _ = extract_model_kwargs(config)
    log_dir, model_dir, logger = _build_stage_runtime(
        model_name=model_name,
        stage="topology_finetune",
        run_id=run_id,
        distributed_context=distributed_context,
    )
    finetune_cfg = _topology_finetune_config(config)
    init_mode = _resolve_init_mode(finetune_cfg)
    checkpoint_path_resolved = Path(checkpoint_path) if checkpoint_path is not None else None
    if distributed_context.is_main_process:
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
        _load_checkpoint(model=model, checkpoint_path=checkpoint_path_resolved, device=device)

    context = _prepare_topology_finetune_stage_context(
        config=config,
        model=model,
        device=device,
        log_dir=log_dir,
        model_dir=model_dir,
        distributed_context=distributed_context,
    )
    best_metrics: dict[str, float | str] = {}

    if distributed_context.is_main_process:
        log_stage_event(
            logger,
            "finetune_config",
            epochs=context.epochs,
            monitor=context.monitor_metric,
            internal_validation_subgraphs=sum(
                len(node_sets) for node_sets in context.internal_validation_node_sets.values()
            ),
            internal_validation_node_sizes=sorted(context.internal_validation_node_sets),
            pair_batch_size=context.pair_batch_size,
            overlap_penalty=context.overlap_penalty,
        )

    for epoch in range(context.epochs):
        epoch_start = time.perf_counter()
        train_stats = _fit_epoch(
            config=config,
            model=model,
            device=device,
            graph=context.train_graph,
            cache_dir=context.cache_dir,
            embedding_index=context.embedding_index,
            optimizer=context.optimizer,
            epoch_index=epoch,
            rank_seed=context.run_seed + 1000 * distributed_context.rank,
            input_dim=context.input_dim,
            max_sequence_length=context.max_sequence_length,
            loss_weights=context.loss_weights,
            pair_batch_size=context.pair_batch_size,
            use_amp=context.use_amp,
            scaler=context.scaler,
            negative_lookup=context.train_negative_lookup,
        )
        train_stats = {
            name: _all_reduce_mean(
                value=value,
                distributed_context=distributed_context,
                device=device,
            )
            for name, value in train_stats.items()
        }

        validation_result = _evaluate_validation_epoch(
            config=config,
            model=model,
            dataloaders=dataloaders,
            device=device,
            context=context,
        )

        monitor_value = _resolve_monitor_value(
            monitor_metric=context.monitor_metric,
            val_pair_stats=validation_result.val_pair_stats,
            internal_val_topology_stats=validation_result.internal_val_topology_stats,
        )
        should_stop = False
        if distributed_context.is_main_process:
            append_csv_row(
                csv_path=context.csv_path,
                row=_build_epoch_csv_row(
                    epoch=epoch + 1,
                    epoch_seconds=time.perf_counter() - epoch_start,
                    train_stats=train_stats,
                    validation_result=validation_result,
                    optimizer=context.optimizer,
                ),
                fieldnames=TOPOLOGY_FINETUNE_CSV_COLUMNS,
            )
            improved, should_stop = context.early_stopping.update(monitor_value)
            if improved:
                _save_checkpoint(model=model, checkpoint_path=context.best_checkpoint_path)
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
            )

        if distributed_context.is_distributed:
            stop_flag = torch.tensor([1 if should_stop else 0], device=device, dtype=torch.int64)
            dist.broadcast(stop_flag, src=0)
            should_stop = bool(int(stop_flag.item()))
        if should_stop:
            if distributed_context.is_main_process:
                log_stage_event(logger, "early_stop", epoch=epoch + 1)
            break

    if distributed_context.is_main_process and not context.best_checkpoint_path.exists():
        _save_checkpoint(model=model, checkpoint_path=context.best_checkpoint_path)
        if not best_metrics:
            context.metrics_path.write_text(
                json.dumps(
                    {"monitor_metric": context.monitor_metric, "monitor_value": 0.0},
                    indent=2,
                ),
                encoding="utf-8",
            )
    if distributed_context.is_main_process:
        log_stage_event(logger, "stage_done", run_id=run_id)
    distributed_barrier(distributed_context)
    return context.best_checkpoint_path
