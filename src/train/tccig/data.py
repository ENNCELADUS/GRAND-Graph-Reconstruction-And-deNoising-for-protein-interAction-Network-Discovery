"""Runtime data context construction for TCCIG training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import networkx as nx
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer

from src.embed import ensure_embeddings_ready
from src.evaluate import Evaluator
from src.pipeline.runtime import AcceleratorLike, DistributedContext
from src.pipeline.stages.train import _build_loss_config
from src.topology.finetune_data import (
    EmbeddingRepository,
    ExplicitNegativePairLookup,
    InternalValidationPlan,
)
from src.topology.supervision import load_supervision_graphs, load_train_negative_lookup
from src.train.tccig.config import (
    TCCIGTrainConfig,
    build_tccig_optimizer,
    tccig_train_config,
)
from src.train.tccig.graph_prior import GraphPriorArtifacts, build_graph_prior_artifacts
from src.train.topology import shared as topology_train
from src.utils.config import ConfigDict, as_bool, as_int, get_section
from src.utils.early_stop import EarlyStopping


@dataclass(frozen=True)
class TCCIGDataContext:
    """Runtime data and artifacts reused across TCCIG epochs."""

    train_graph: nx.Graph
    internal_val_graph: nx.Graph
    train_negative_lookup: ExplicitNegativePairLookup
    embedding_repository: EmbeddingRepository
    input_dim: int
    max_sequence_length: int
    evaluator: Evaluator
    internal_validation_node_sets: Mapping[int, Sequence[tuple[str, ...]]]
    internal_validation_plan: InternalValidationPlan
    optimizer: Optimizer
    early_stopping: EarlyStopping
    best_checkpoint_path: Path
    metrics_path: Path
    csv_path: Path
    density_prior_probability: float
    density_prior_source: str
    density_prior_bias: float
    graph_prior_artifacts: GraphPriorArtifacts | None


def prepare_tccig_data_context(
    *,
    config: ConfigDict,
    train_cfg: TCCIGTrainConfig,
    model: nn.Module,
    device: torch.device,
    log_dir: Path,
    model_dir: Path,
    distributed_context: DistributedContext,
    accelerator: AcceleratorLike,
) -> TCCIGDataContext:
    """Build shared runtime state for TCCIG training."""
    raw_train_cfg = tccig_train_config(config)
    data_cfg = get_section(config, "data_config")
    model_cfg = get_section(config, "model_config")
    training_cfg = get_section(config, "training_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")

    input_dim = as_int(model_cfg.get("input_dim", 0), "model_config.input_dim")
    max_sequence_length = as_int(
        data_cfg.get("max_sequence_length", 64),
        "data_config.max_sequence_length",
    )
    train_path = Path(str(dataloader_cfg.get("train_dataset", "")))
    valid_path = Path(str(dataloader_cfg.get("valid_dataset", "")))
    train_graph, internal_val_graph = load_supervision_graphs(
        config=config,
        stage_cfg=raw_train_cfg,
        stage_name="tccig_train",
    )
    train_negative_lookup = load_train_negative_lookup(
        config=config,
        stage_cfg=raw_train_cfg,
        stage_name="tccig_train",
    )
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
    topology_train._validate_embedding_cache(
        graph=train_graph,
        embedding_index=embedding_cache.index,
    )
    internal_validation_node_sets = topology_train._build_internal_validation_node_sets(
        finetune_cfg=raw_train_cfg,
        stage_name="tccig_train",
        graph=internal_val_graph,
        seed=train_cfg.run_seed + 100_000,
    )
    internal_validation_plan = topology_train.build_internal_validation_plan(
        graph=internal_val_graph,
        sampled_subgraphs=internal_validation_node_sets,
    )
    embedding_repository = EmbeddingRepository(
        cache_dir=embedding_cache.cache_dir,
        embedding_index=embedding_cache.index,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        max_cache_bytes=topology_train._resolve_embedding_cache_max_bytes(
            raw_train_cfg,
            stage_name="tccig_train",
        ),
    )
    if internal_validation_plan.protein_ids:
        embedding_repository.preload(sorted(internal_validation_plan.protein_ids))
    graph_prior_artifacts = _prepare_graph_prior_artifacts(
        train_cfg=train_cfg,
        train_graph=train_graph,
        embedding_repository=embedding_repository,
        model_dir=model_dir,
        device=device,
        accelerator=accelerator,
        distributed_context=distributed_context,
    )

    density_prior_probability, density_prior_source = _tccig_density_prior(
        train_graph=train_graph,
        train_negative_lookup=train_negative_lookup,
        negative_ratio=train_cfg.negative_ratio,
    )
    density_prior_bias = _initialize_tccig_density_bias(
        model=model,
        positive_edge_probability=density_prior_probability,
        accelerator=accelerator,
    )
    evaluator = Evaluator(
        metrics=["auprc"],
        loss_config=_build_loss_config(training_cfg),
        use_amp=device.type == "cuda"
        and as_bool(
            get_section(config, "device_config").get("use_mixed_precision", False),
            "device_config.use_mixed_precision",
        ),
        accelerator=accelerator,
    )
    return TCCIGDataContext(
        train_graph=train_graph,
        internal_val_graph=internal_val_graph,
        train_negative_lookup=train_negative_lookup,
        embedding_repository=embedding_repository,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        evaluator=evaluator,
        internal_validation_node_sets=internal_validation_node_sets,
        internal_validation_plan=internal_validation_plan,
        optimizer=build_tccig_optimizer(train_cfg, model),
        early_stopping=EarlyStopping(
            patience=train_cfg.early_stopping_patience,
            mode=topology_train._resolve_monitor_mode(train_cfg.monitor_metric),
        ),
        best_checkpoint_path=model_dir / "best_model.pth",
        metrics_path=log_dir / "tccig_train_metrics.json",
        csv_path=log_dir / "tccig_train_step.csv",
        density_prior_probability=density_prior_probability,
        density_prior_source=density_prior_source,
        density_prior_bias=density_prior_bias,
        graph_prior_artifacts=graph_prior_artifacts,
    )


def _mean_pool_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """Return a mean-pooled cached embedding."""
    if embedding.dim() != 2:
        raise ValueError("cached protein embeddings must have shape (seq_len, input_dim)")
    return embedding.float().mean(dim=0)


def _prepare_graph_prior_artifacts(
    *,
    train_cfg: TCCIGTrainConfig,
    train_graph: nx.Graph,
    embedding_repository: EmbeddingRepository,
    model_dir: Path,
    device: torch.device,
    accelerator: AcceleratorLike,
    distributed_context: DistributedContext,
) -> GraphPriorArtifacts | None:
    """Build or load offline train-only graph-prior artifacts."""
    teacher_cfg = train_cfg.graph_prior_teacher
    if not teacher_cfg.enabled:
        return None
    artifact_dir = model_dir / teacher_cfg.artifact_dir
    artifact_path = artifact_dir / "graph_prior_artifacts.pt"
    if teacher_cfg.reuse_artifacts and artifact_path.exists():
        return GraphPriorArtifacts.load(artifact_dir)
    should_build = distributed_context.is_main_process
    if should_build:
        node_features = {
            protein_id: _mean_pool_embedding(embedding_repository.get(protein_id))
            for protein_id in sorted(train_graph.nodes)
        }
        artifacts = build_graph_prior_artifacts(
            graph=train_graph,
            node_features=node_features,
            hidden_dim=teacher_cfg.hidden_dim,
            num_layers=teacher_cfg.num_layers,
            decoder_hidden_dim=teacher_cfg.decoder_hidden_dim,
            dropout=teacher_cfg.dropout,
            epochs=teacher_cfg.epochs,
            mask_ratio=teacher_cfg.mask_ratio,
            negative_ratio=teacher_cfg.negative_ratio,
            seed=train_cfg.run_seed,
            device=device,
        )
        artifacts.save(artifact_dir)
    if distributed_context.is_distributed:
        accelerator.wait_for_everyone()
    return GraphPriorArtifacts.load(artifact_dir)


def _graph_positive_edge_probability(graph: nx.Graph) -> float:
    """Return the full train-graph positive-edge density."""
    num_nodes = graph.number_of_nodes()
    if num_nodes < 2:
        raise ValueError("TCCIG density-bias initialization requires at least 2 train nodes")
    possible_edges = num_nodes * (num_nodes - 1) / 2.0
    return float(graph.number_of_edges() / possible_edges)


def _supervised_positive_probability(
    *,
    train_graph: nx.Graph,
    train_negative_lookup: ExplicitNegativePairLookup,
    negative_ratio: int,
) -> float | None:
    """Return the supervised BCE positive rate when explicit negatives exist."""
    positive_count = int(train_graph.number_of_edges())
    if positive_count <= 0 or negative_ratio <= 0 or not train_negative_lookup.negative_pairs:
        return None
    negative_count = min(len(train_negative_lookup.negative_pairs), positive_count * negative_ratio)
    if negative_count <= 0:
        return None
    return float(positive_count / float(positive_count + negative_count))


def _tccig_density_prior(
    *,
    train_graph: nx.Graph,
    train_negative_lookup: ExplicitNegativePairLookup,
    negative_ratio: int,
) -> tuple[float, str]:
    """Return the graph-density prior for TCCIG density-bias initialization."""
    del train_negative_lookup, negative_ratio
    return _graph_positive_edge_probability(train_graph), "graph_density"


def _initialize_tccig_density_bias(
    *,
    model: nn.Module,
    positive_edge_probability: float,
    accelerator: AcceleratorLike,
) -> float:
    """Initialize a TCCIG model's density-bias head from a sparse prior."""
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    unwrapped_model = cast(nn.Module, unwrap_model(model)) if callable(unwrap_model) else model
    initializer = getattr(unwrapped_model, "initialize_density_bias_with_prior", None)
    if not callable(initializer):
        raise ValueError("tccig_train requires initialize_density_bias_with_prior on the model")
    bias_value = float(initializer(positive_edge_probability))
    if not math.isfinite(bias_value):
        raise ValueError("TCCIG density-bias initialization produced a non-finite bias")
    return bias_value
