"""Runtime data context construction for TCCIG training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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
    )
