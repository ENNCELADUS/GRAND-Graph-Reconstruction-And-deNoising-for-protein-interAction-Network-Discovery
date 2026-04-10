"""Graph-topology fine-tuning stage for PRING training subgraphs."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from src.embed import ensure_embeddings_ready
from src.evaluate import Evaluator
from src.run.stage_train import (
    _build_loss_config,
    _build_stage_runtime,
    _load_checkpoint,
    _save_checkpoint,
)
from src.topology.finetune_data import (
    SubgraphPairChunk,
    build_pair_supervision_graph,
    iter_subgraph_pair_chunks,
    load_split_node_ids,
    sample_training_subgraphs,
)
from src.topology.finetune_losses import (
    TopologyLossWeights,
    build_symmetric_adjacency,
    compute_topology_losses,
)
from src.topology.metrics import (
    clustering_stats,
    compute_graph_similarity,
    compute_mmd,
    compute_relative_density,
    degree_distribution,
)
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
    "Learning Rate",
]
MAX_TOPOLOGY_FINETUNE_SUBGRAPH_NODES = 20


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
            "val_relative_density": -abs(
                internal_val_topology_stats["relative_density"] - 1.0
            ),
            "val_auprc": float(val_pair_stats.get("val_auprc", 0.0)),
        }.get(monitor_metric, internal_val_topology_stats["graph_sim"])
    )


def _load_supervision_graphs(*, config: ConfigDict) -> tuple[nx.Graph, nx.Graph]:
    """Build train/validation supervision graphs without leaking validation edges."""
    data_cfg = get_section(config, "data_config")
    benchmark_cfg = get_section(data_cfg, "benchmark")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    processed_dir = Path(str(benchmark_cfg.get("processed_dir", "")))
    species = as_str(benchmark_cfg.get("species", "human"), "data_config.benchmark.species")
    split_strategy = as_str(
        benchmark_cfg.get("split_strategy", "BFS"),
        "data_config.benchmark.split_strategy",
    ).upper()
    split_path = processed_dir / f"{species}_{split_strategy}_split.pkl"
    train_nodes = load_split_node_ids(split_path=split_path, split_name="train")
    train_graph = build_pair_supervision_graph(
        pair_path=Path(str(dataloader_cfg.get("train_dataset", ""))),
        node_ids=train_nodes,
    )
    internal_val_graph = build_pair_supervision_graph(
        pair_path=Path(str(dataloader_cfg.get("valid_dataset", ""))),
        node_ids=train_nodes,
    )
    return train_graph, internal_val_graph


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
    """Resolve memory-bounded subgraph node limits for topology fine-tuning."""
    max_nodes = min(
        as_int(finetune_cfg.get("max_nodes", 20), "topology_finetune.max_nodes"),
        MAX_TOPOLOGY_FINETUNE_SUBGRAPH_NODES,
    )
    min_nodes = min(
        as_int(finetune_cfg.get("min_nodes", max_nodes), "topology_finetune.min_nodes"),
        max_nodes,
    )
    return min_nodes, max_nodes


def _move_chunk_to_device(
    *,
    chunk: SubgraphPairChunk,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert a chunk object into model-ready tensors on device."""
    return {
        "emb_a": chunk.emb_a.to(device),
        "emb_b": chunk.emb_b.to(device),
        "len_a": chunk.len_a.to(device),
        "len_b": chunk.len_b.to(device),
        "label": chunk.label.to(device),
        "pair_index_a": chunk.pair_index_a.to(device),
        "pair_index_b": chunk.pair_index_b.to(device),
    }


def _forward_model(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Execute one forward pass under the standard model output contract."""
    if model.training and torch.is_grad_enabled():
        emb_a = batch.get("emb_a")
        emb_b = batch.get("emb_b")
        len_a = batch.get("len_a")
        len_b = batch.get("len_b")
        if (
            emb_a is not None
            and emb_b is not None
            and len_a is not None
            and len_b is not None
        ):
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward one sampled subgraph and collect all pair logits and labels."""
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    pair_index_a_list: list[torch.Tensor] = []
    pair_index_b_list: list[torch.Tensor] = []

    for chunk in iter_subgraph_pair_chunks(
        graph=graph,
        nodes=nodes,
        cache_dir=cache_dir,
        embedding_index=embedding_index,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        pair_batch_size=pair_batch_size,
    ):
        batch = _move_chunk_to_device(chunk=chunk, device=device)
        output = _forward_model(model=model, batch=batch)
        logits = output["logits"]
        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        logits_list.append(logits)
        labels_list.append(batch["label"].float())
        pair_index_a_list.append(batch["pair_index_a"])
        pair_index_b_list.append(batch["pair_index_b"])

    return (
        torch.cat(logits_list, dim=0),
        torch.cat(labels_list, dim=0),
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
    logits, _, pair_index_a, pair_index_b = _concat_logits_and_pairs(
        model=model,
        graph=graph,
        nodes=nodes,
        cache_dir=cache_dir,
        embedding_index=embedding_index,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        pair_batch_size=pair_batch_size,
        device=device,
    )
    node_tuple = tuple(nodes)
    pred_subgraph = nx.Graph()
    pred_subgraph.add_nodes_from(node_tuple)
    probabilities = torch.sigmoid(logits).detach().cpu()
    rows = pair_index_a.detach().cpu().tolist()
    cols = pair_index_b.detach().cpu().tolist()
    for row_idx, col_idx, probability in zip(rows, cols, probabilities.tolist(), strict=True):
        if probability >= threshold:
            pred_subgraph.add_edge(node_tuple[row_idx], node_tuple[col_idx])
    return pred_subgraph


def _evaluate_internal_validation_subgraphs(
    *,
    model: nn.Module,
    graph: nx.Graph,
    sampled_subgraphs: Sequence[tuple[str, ...]],
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
    pair_batch_size: int,
    threshold: float,
    device: torch.device,
) -> dict[str, float]:
    """Compute internal subgraph topology metrics on fixed validation samples."""
    graph_sim_values: list[float] = []
    density_values: list[float] = []
    pred_degree_histograms: list[np.ndarray] = []
    target_degree_histograms: list[np.ndarray] = []
    pred_graphs: list[nx.Graph] = []
    target_graphs: list[nx.Graph] = []

    with torch.no_grad():
        for nodes in sampled_subgraphs:
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
            graph_sim_values.append(
                compute_graph_similarity(
                    pred_graph=pred_subgraph,
                    gt_graph=target_subgraph,
                )
            )
            density_values.append(
                compute_relative_density(
                    pred_graph=pred_subgraph,
                    gt_graph=target_subgraph,
                )
            )
            pred_deg_hist, target_deg_hist = degree_distribution(
                pred_graph=pred_subgraph,
                gt_graph=target_subgraph,
            )
            pred_degree_histograms.append(pred_deg_hist)
            target_degree_histograms.append(target_deg_hist)
            pred_graphs.append(pred_subgraph)
            target_graphs.append(target_subgraph)

    return {
        "graph_sim": float(np.mean(graph_sim_values)) if graph_sim_values else 0.0,
        "relative_density": float(np.mean(density_values)) if density_values else 0.0,
        "deg_dist_mmd": compute_mmd(pred_degree_histograms, target_degree_histograms),
        "cc_mmd": clustering_stats(target_graphs, pred_graphs),
    }


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
) -> dict[str, float]:
    """Run one fine-tuning epoch over sampled train subgraphs."""
    finetune_cfg = _topology_finetune_config(config)
    min_nodes, max_nodes = _resolve_sampling_node_bounds(finetune_cfg)
    sampled_subgraphs = sample_training_subgraphs(
        graph=graph,
        num_subgraphs=as_int(
            finetune_cfg.get("subgraphs_per_epoch", 16),
            "topology_finetune.subgraphs_per_epoch",
        ),
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        strategy=as_str(finetune_cfg.get("strategy", "mixed"), "topology_finetune.strategy"),
        seed=rank_seed + epoch_index,
    )
    loss_cfg = _build_loss_config(get_section(config, "training_config"))
    aggregates = {
        "bce": 0.0,
        "graph_similarity": 0.0,
        "relative_density": 0.0,
        "degree_mmd": 0.0,
        "clustering_mmd": 0.0,
        "total": 0.0,
    }

    model.train()
    for nodes in sampled_subgraphs:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, labels, pair_index_a, pair_index_b = _concat_logits_and_pairs(
                model=model,
                graph=graph,
                nodes=nodes,
                cache_dir=cache_dir,
                embedding_index=embedding_index,
                input_dim=input_dim,
                max_sequence_length=max_sequence_length,
                pair_batch_size=pair_batch_size,
                device=device,
            )
            bce_loss = binary_classification_loss(
                logits=logits,
                labels=labels,
                loss_config=loss_cfg,
                reduction="mean",
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
        aggregates["bce"] += float(bce_loss.detach().item())
        aggregates["graph_similarity"] += float(topology_losses["graph_similarity"].detach().item())
        aggregates["relative_density"] += float(topology_losses["relative_density"].detach().item())
        aggregates["degree_mmd"] += float(topology_losses["degree_mmd"].detach().item())
        aggregates["clustering_mmd"] += float(topology_losses["clustering_mmd"].detach().item())
        aggregates["total"] += float(total_loss.detach().item())

    denominator = float(max(1, len(sampled_subgraphs)))
    return {name: value / denominator for name, value in aggregates.items()}


def run_topology_finetuning_stage(
    config: ConfigDict,
    model: nn.Module,
    device: torch.device,
    dataloaders: dict[str, DataLoader[dict[str, object]]],
    run_id: str,
    checkpoint_path: Path,
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
    checkpoint_path = Path(checkpoint_path)
    if distributed_context.is_main_process:
        log_stage_event(logger, "stage_start", run_id=run_id, checkpoint=checkpoint_path)
    _load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)

    finetune_cfg = _topology_finetune_config(config)
    run_cfg = get_section(config, "run_config")
    data_cfg = get_section(config, "data_config")
    model_cfg = get_section(config, "model_config")
    training_cfg = get_section(config, "training_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    input_dim = as_int(model_cfg.get("input_dim", 0), "model_config.input_dim")
    max_sequence_length = as_int(
        data_cfg.get("max_sequence_length", 64),
        "data_config.max_sequence_length",
    )
    pair_batch_size = as_int(
        finetune_cfg.get("pair_batch_size", training_cfg.get("batch_size", 8)),
        "topology_finetune.pair_batch_size",
    )
    decision_threshold = as_float(
        finetune_cfg.get("decision_threshold", 0.5),
        "topology_finetune.decision_threshold",
    )
    epochs = as_int(
        finetune_cfg.get("epochs", training_cfg.get("epochs", 1)),
        "topology_finetune.epochs",
    )
    monitor_metric = as_str(
        finetune_cfg.get("monitor_metric", "val_graph_sim"),
        "topology_finetune.monitor_metric",
    )
    patience = as_int(
        finetune_cfg.get(
            "early_stopping_patience",
            training_cfg.get("early_stopping_patience", 5),
        ),
        "topology_finetune.early_stopping_patience",
    )
    use_amp = device.type == "cuda" and as_bool(
        get_section(config, "device_config").get("use_mixed_precision", False),
        "device_config.use_mixed_precision",
    )

    train_path = Path(str(dataloader_cfg.get("train_dataset", "")))
    valid_path = Path(str(dataloader_cfg.get("valid_dataset", "")))
    train_graph, internal_val_graph = _load_supervision_graphs(config=config)
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
    missing_graph_nodes = sorted(
        node_id for node_id in train_graph.nodes if node_id not in embedding_cache.index
    )
    if missing_graph_nodes:
        preview = ", ".join(missing_graph_nodes[:10])
        raise FileNotFoundError(
            "Embedding cache is missing train-graph proteins required by topology_finetune: "
            f"{preview} (missing={len(missing_graph_nodes)})"
        )
    optimizer = _build_optimizer(config=config, model=model)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    evaluator = Evaluator(
        metrics=["auprc"],
        loss_config=_build_loss_config(training_cfg),
        use_amp=use_amp,
    )
    loss_weights = _parse_loss_weights(config)
    early_stopping = EarlyStopping(
        patience=patience,
        mode=_resolve_monitor_mode(monitor_metric),
    )
    best_checkpoint_path = model_dir / "best_model.pth"
    metrics_path = log_dir / "topology_finetune_metrics.json"
    csv_path = log_dir / "topology_finetune_step.csv"
    best_metrics: dict[str, float] = {}
    min_nodes, max_nodes = _resolve_sampling_node_bounds(finetune_cfg)
    sampled_internal_val_subgraphs = sample_training_subgraphs(
        graph=internal_val_graph,
        num_subgraphs=as_int(
            finetune_cfg.get("validation_subgraphs", 8),
            "topology_finetune.validation_subgraphs",
        ),
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        strategy=as_str(finetune_cfg.get("strategy", "mixed"), "topology_finetune.strategy"),
        seed=as_int(run_cfg.get("seed", 0), "run_config.seed") + 100_000,
    )

    if distributed_context.is_main_process:
        log_stage_event(
            logger,
            "finetune_config",
            epochs=epochs,
            monitor=monitor_metric,
            subgraphs_per_epoch=finetune_cfg.get("subgraphs_per_epoch", 16),
            internal_validation_subgraphs=len(sampled_internal_val_subgraphs),
            pair_batch_size=pair_batch_size,
        )

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        train_stats = _fit_epoch(
            config=config,
            model=model,
            device=device,
            graph=train_graph,
            cache_dir=embedding_cache.cache_dir,
            embedding_index=embedding_cache.index,
            optimizer=optimizer,
            epoch_index=epoch,
            rank_seed=as_int(run_cfg.get("seed", 0), "run_config.seed")
            + 1000 * distributed_context.rank,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
            loss_weights=loss_weights,
            pair_batch_size=pair_batch_size,
            use_amp=use_amp,
            scaler=scaler,
        )
        train_stats = {
            name: _all_reduce_mean(
                value=value,
                distributed_context=distributed_context,
                device=device,
            )
            for name, value in train_stats.items()
        }

        model.eval()
        with torch.no_grad():
            val_pair_stats = evaluator.evaluate(
                model=model,
                data_loader=dataloaders["valid"],
                device=device,
                prefix="val",
            )
        internal_val_topology_stats = _evaluate_internal_validation_subgraphs(
            model=model,
            graph=internal_val_graph,
            sampled_subgraphs=sampled_internal_val_subgraphs,
            cache_dir=embedding_cache.cache_dir,
            embedding_index=embedding_cache.index,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
            pair_batch_size=pair_batch_size,
            threshold=decision_threshold,
            device=device,
        )

        monitor_value = _resolve_monitor_value(
            monitor_metric=monitor_metric,
            val_pair_stats=val_pair_stats,
            internal_val_topology_stats=internal_val_topology_stats,
        )
        should_stop = False
        if distributed_context.is_main_process:
            append_csv_row(
                csv_path=csv_path,
                row={
                    "Epoch": epoch + 1,
                    "Epoch Time": time.perf_counter() - epoch_start,
                    "Train BCE Loss": train_stats["bce"],
                    "Train GS Loss": train_stats["graph_similarity"],
                    "Train RD Loss": train_stats["relative_density"],
                    "Train Deg MMD": train_stats["degree_mmd"],
                    "Train Clus MMD": train_stats["clustering_mmd"],
                    "Train Total Loss": train_stats["total"],
                    "Val Loss": float(val_pair_stats.get("val_loss", 0.0)),
                    "Val auprc": float(val_pair_stats.get("val_auprc", 0.0)),
                    "Internal Val graph_sim": internal_val_topology_stats["graph_sim"],
                    "Internal Val relative_density": internal_val_topology_stats[
                        "relative_density"
                    ],
                    "Internal Val deg_dist_mmd": internal_val_topology_stats[
                        "deg_dist_mmd"
                    ],
                    "Internal Val cc_mmd": internal_val_topology_stats["cc_mmd"],
                    "Learning Rate": float(optimizer.param_groups[0]["lr"]),
                },
                fieldnames=TOPOLOGY_FINETUNE_CSV_COLUMNS,
            )
            improved, should_stop = early_stopping.update(monitor_value)
            if improved:
                _save_checkpoint(model=model, checkpoint_path=best_checkpoint_path)
                best_metrics = {
                    "epoch": float(epoch + 1),
                    "monitor_metric": monitor_metric,
                    "monitor_value": monitor_value,
                    "val_loss": float(val_pair_stats.get("val_loss", 0.0)),
                    "val_auprc": float(val_pair_stats.get("val_auprc", 0.0)),
                    "internal_val_graph_sim": internal_val_topology_stats["graph_sim"],
                    "internal_val_relative_density": internal_val_topology_stats[
                        "relative_density"
                    ],
                    "internal_val_deg_dist_mmd": internal_val_topology_stats[
                        "deg_dist_mmd"
                    ],
                    "internal_val_cc_mmd": internal_val_topology_stats["cc_mmd"],
                }
                metrics_path.write_text(
                    json.dumps(best_metrics, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                log_stage_event(
                    logger,
                    "best_saved",
                    epoch=epoch + 1,
                    monitor=monitor_metric,
                    value=monitor_value,
                )
            log_stage_event(
                logger,
                "epoch_done",
                epoch=epoch + 1,
                train_loss=train_stats["total"],
                val_auprc=float(val_pair_stats.get("val_auprc", 0.0)),
                internal_val_graph_sim=internal_val_topology_stats["graph_sim"],
            )

        if distributed_context.is_distributed:
            stop_flag = torch.tensor([1 if should_stop else 0], device=device, dtype=torch.int64)
            dist.broadcast(stop_flag, src=0)
            should_stop = bool(int(stop_flag.item()))
        if should_stop:
            if distributed_context.is_main_process:
                log_stage_event(logger, "early_stop", epoch=epoch + 1)
            break

    if distributed_context.is_main_process and not best_checkpoint_path.exists():
        _save_checkpoint(model=model, checkpoint_path=best_checkpoint_path)
        if not best_metrics:
            metrics_path.write_text(
                json.dumps({"monitor_metric": monitor_metric, "monitor_value": 0.0}, indent=2),
                encoding="utf-8",
            )
    if distributed_context.is_main_process:
        log_stage_event(logger, "stage_done", run_id=run_id)
    distributed_barrier(distributed_context)
    return best_checkpoint_path
