"""Student optimization loop for the TCCIG training stage."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import networkx as nx
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer

from src.pipeline.runtime import AcceleratorLike, DistributedContext
from src.topology.finetune_data import (
    EdgeCoverEpochPlan,
    EmbeddingRepository,
    ExplicitNegativePairLookup,
    sample_edge_cover_subgraphs,
)
from src.topology.losses import TCCIGLossWeights, topology_loss_scale
from src.train.tccig.config import TCCIGTrainConfig
from src.train.tccig.graph_prior import GraphPriorArtifacts
from src.train.tccig.retrieval import compute_retrieval_losses, mine_hard_negative_pairs
from src.train.tccig.teacher import OnlineTCCIGTeacher
from src.train.topology import shared as topology_train
from src.utils.config import ConfigDict
from src.utils.logging import log_epoch_progress


class TCCIGStudentTrainer:
    """Train TCCIG student epochs over sampled feature-only graph forwards."""

    def __init__(
        self,
        *,
        train_cfg: TCCIGTrainConfig,
        raw_config: ConfigDict,
        model: nn.Module,
        graph: nx.Graph,
        optimizer: Optimizer,
        device: torch.device,
        accelerator: AcceleratorLike,
        embedding_repository: EmbeddingRepository,
        negative_lookup: ExplicitNegativePairLookup,
        distributed_context: DistributedContext,
        model_dir: Path,
        teacher: OnlineTCCIGTeacher | None,
        graph_prior_artifacts: GraphPriorArtifacts | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.train_cfg = train_cfg
        self.raw_config = raw_config
        self.model = model
        self.graph = graph
        self.optimizer = optimizer
        self.device = device
        self.accelerator = accelerator
        self.embedding_repository = embedding_repository
        self.negative_lookup = negative_lookup
        self.distributed_context = distributed_context
        self.model_dir = model_dir
        self.teacher = teacher
        self.graph_prior_artifacts = graph_prior_artifacts
        self.logger = logger

    def fit_epoch(self, *, epoch_index: int, epoch_seed: int) -> dict[str, float]:
        """Fit one TCCIG training epoch and return unreduced train stats."""
        sampling_start = time.perf_counter()
        epoch_plan = sample_edge_cover_subgraphs(
            graph=self.graph,
            num_subgraphs=self.train_cfg.subgraphs_per_epoch,
            min_nodes=min(self.train_cfg.node_sizes),
            max_nodes=max(self.train_cfg.node_sizes),
            strategy=self.train_cfg.strategy,
            seed=epoch_seed,
            edge_chunk_size=self.train_cfg.edge_chunk_size,
            node_sizes=self.train_cfg.node_sizes if len(self.train_cfg.node_sizes) > 1 else None,
            negative_lookup=self.negative_lookup,
            negative_ratio=self.train_cfg.negative_ratio,
        )
        edge_cover_sampling_seconds = time.perf_counter() - sampling_start
        local_tasks = _local_tasks(
            epoch_plan=epoch_plan,
            distributed_context=self.distributed_context,
        )
        aggregates = topology_train._initialize_train_aggregates(
            len(epoch_plan.subgraphs),
            epoch_plan,
            self.train_cfg.negative_ratio,
        )
        current_topology_loss_scale = topology_loss_scale(
            epoch=epoch_index,
            schedule=self.train_cfg.loss_weight_schedule,
        )
        effective_loss_weights = replace(
            self.train_cfg.loss_weights,
            density=self.train_cfg.loss_weights.density * current_topology_loss_scale,
            degree=self.train_cfg.loss_weights.degree * current_topology_loss_scale,
            clustering=(
                self.train_cfg.loss_weights.clustering * current_topology_loss_scale
                if self.train_cfg.compute_clustering_mmd
                else 0.0
            ),
        )
        train_forward_backward_seconds = self._fit_tccig_epoch(
            epoch_index=epoch_index,
            epoch_seed=epoch_seed,
            local_tasks=local_tasks,
            aggregates=aggregates,
            loss_weights=effective_loss_weights,
        )
        aggregates["edge_cover_sampling_s"] = edge_cover_sampling_seconds
        aggregates["train_forward_backward_s"] = train_forward_backward_seconds
        aggregates["topology_loss_scale"] = current_topology_loss_scale
        return aggregates

    def _fit_tccig_epoch(
        self,
        *,
        epoch_index: int,
        epoch_seed: int,
        local_tasks: Sequence[topology_train.LocalSubgraphTask],
        aggregates: dict[str, float],
        loss_weights: TCCIGLossWeights,
    ) -> float:
        graph_model = topology_train._unwrap_model_for_detached_forward(
            model=self.model,
            accelerator=self.accelerator,
        )
        forward_graph = getattr(graph_model, "forward_graph", None)
        if not callable(forward_graph):
            raise ValueError("TCCIG student training requires a model with forward_graph")
        total_subgraphs = max(1, sum(1 for task in local_tasks if not task.is_padding))
        completed_real_subgraphs = 0
        train_forward_backward_seconds = 0.0
        hard_negative_records: list[dict[str, float | int | str]] = []
        self.model.train()
        if self.teacher is not None:
            self.teacher.teacher.train()

        if local_tasks:
            self.optimizer.zero_grad(set_to_none=True)
        for task_index, task in enumerate(local_tasks):
            step_start = time.perf_counter()
            current_window_start = (task_index // self.train_cfg.gradient_accumulation_steps) * (
                self.train_cfg.gradient_accumulation_steps
            )
            current_window_end = min(
                current_window_start + self.train_cfg.gradient_accumulation_steps,
                len(local_tasks),
            )
            current_window_size = current_window_end - current_window_start
            is_window_boundary = task_index + 1 == current_window_end
            with topology_train._manual_accumulation_context(
                accelerator=self.accelerator,
                model=self.model,
                sync_gradients=is_window_boundary,
            ):
                protein_embeddings, protein_lengths = _load_graph_inputs(
                    nodes=task.nodes,
                    embedding_repository=self.embedding_repository,
                    device=self.device,
                )
                output = cast(
                    dict[str, object],
                    forward_graph(
                        protein_embeddings=protein_embeddings,
                        protein_lengths=protein_lengths,
                    ),
                )
                encoded = cast(Mapping[str, torch.Tensor], output["encoded"])
                logits = topology_train._squeeze_binary_logits(cast(torch.Tensor, output["logits"]))
                candidate_pairs = cast(torch.Tensor, output["candidate_pairs"])
                labels, bce_labels, bce_mask = _candidate_supervision(
                    graph=self.graph,
                    nodes=task.nodes,
                    candidate_pairs=candidate_pairs,
                    assigned_positive_edges=task.assigned_positive_edges,
                    assigned_negative_edges=task.assigned_negative_edges,
                    device=self.device,
                )
                positive_edges = _positive_edge_index_for_nodes(
                    graph=self.graph,
                    nodes=task.nodes,
                    device=self.device,
                )
                should_distill = (
                    self.teacher is not None
                    and loss_weights.teacher != 0.0
                    and positive_edges.size(1) > 0
                )
                if should_distill and self.teacher is not None:
                    node_features = _masked_mean_pool_embeddings(
                        protein_embeddings=protein_embeddings,
                        protein_lengths=protein_lengths,
                    )
                    _ = self.teacher.train_and_score(
                        node_features=node_features,
                        positive_edges=positive_edges,
                        candidate_pairs=candidate_pairs,
                        seed=epoch_seed + task_index,
                        device=self.device,
                        accelerator=self.accelerator,
                        loss_scale=0.0 if task.is_padding else 1.0,
                    )
                degree_targets = _degree_targets_for_nodes(
                    graph=self.graph,
                    nodes=task.nodes,
                    device=self.device,
                )
                struct_targets = _struct_targets_for_nodes(
                    artifacts=self.graph_prior_artifacts,
                    nodes=tuple(task.nodes),
                    device=self.device,
                    expected_dim=encoded["struct"].size(1),
                )
                retrieval_score_matrix = cast(torch.Tensor, output["retrieval_score_matrix"])
                mined_pairs = torch.empty((2, 0), dtype=torch.long, device=self.device)
                if (
                    not task.is_padding
                    and self._should_refresh_hard_negative_cache(epoch_index)
                    and len(hard_negative_records)
                    < self.train_cfg.hard_negative_mining.max_pairs_per_epoch
                ):
                    remaining_cache_budget = (
                        self.train_cfg.hard_negative_mining.max_pairs_per_epoch
                        - len(hard_negative_records)
                    )
                    mined_pairs = mine_hard_negative_pairs(
                        score_matrix=retrieval_score_matrix.detach(),
                        known_positive_pairs=positive_edges,
                        top_k=self.train_cfg.hard_negative_mining.top_k,
                        max_pairs=remaining_cache_budget,
                    )
                    hard_negative_records.extend(
                        _hard_negative_cache_records(
                            nodes=task.nodes,
                            pairs=mined_pairs,
                            score_matrix=retrieval_score_matrix.detach(),
                        )
                    )
                supervised_mask = bce_mask > 0.0
                if mined_pairs.numel() > 0:
                    hard_negative_mask = _candidate_pair_membership_mask(
                        candidate_pairs=candidate_pairs,
                        selected_pairs=mined_pairs,
                    )
                    supervised_mask = supervised_mask | hard_negative_mask
                    bce_labels = bce_labels.masked_fill(hard_negative_mask, 0.0)
                supervised_logits = logits[supervised_mask]
                supervised_labels = bce_labels[supervised_mask]
                retrieval_losses = compute_retrieval_losses(
                    retrieval_score_matrix=retrieval_score_matrix,
                    positive_pairs=positive_edges,
                    known_positive_pairs=positive_edges,
                    candidate_logits=supervised_logits,
                    candidate_labels=supervised_labels,
                    degree_predictions=cast(torch.Tensor, output["degree_predictions"]),
                    degree_targets=degree_targets,
                    struct_predictions=encoded["struct"],
                    struct_targets=struct_targets,
                    hard_negative_pairs=mined_pairs,
                    retrieval_weight=self.train_cfg.retrieval.retrieval_weight,
                    reranker_weight=(
                        self.train_cfg.reranker.weight if self.train_cfg.reranker.enabled else 0.0
                    ),
                    degree_weight=self.train_cfg.graph_prior_teacher.degree_weight,
                    struct_weight=(
                        self.train_cfg.graph_prior_teacher.struct_weight
                        if struct_targets is not None
                        else 0.0
                    ),
                    hard_negative_weight=(
                        self.train_cfg.hard_negative_mining.weight
                        if self.train_cfg.hard_negative_mining.enabled
                        else 0.0
                    ),
                    temperature=self.train_cfg.retrieval.temperature,
                    reranker_negative_temperature=(
                        self.train_cfg.reranker.adaptive_negative_temperature
                    ),
                )
                losses = _retrieval_losses_for_legacy_aggregates(retrieval_losses)
                total_loss = losses["total"]
                backward_loss = total_loss * 0.0 if task.is_padding else total_loss
                topology_losses = _topology_losses_for_aggregates(
                    losses=losses,
                    reference=losses["edge"],
                )

                self.accelerator.backward(backward_loss / float(current_window_size))
                if is_window_boundary:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            train_forward_backward_seconds += time.perf_counter() - step_start
            if not task.is_padding:
                completed_real_subgraphs += 1
                topology_train._update_train_aggregates(
                    aggregates=aggregates,
                    bce_loss=losses["edge"],
                    topology_losses=topology_losses,
                    total_loss=total_loss,
                )
                log_epoch_progress(
                    self.logger,
                    epoch=epoch_index + 1,
                    step=completed_real_subgraphs,
                    total_steps=total_subgraphs,
                    loss=aggregates["total"] / float(completed_real_subgraphs),
                )
        self._write_hard_negative_cache(
            epoch_index=epoch_index,
            records=hard_negative_records,
        )
        return train_forward_backward_seconds

    def _should_refresh_hard_negative_cache(self, epoch_index: int) -> bool:
        """Return whether this epoch should emit hard-negative mining artifacts."""
        cfg = self.train_cfg.hard_negative_mining
        if not cfg.enabled:
            return False
        refresh_every = max(1, cfg.refresh_every_epochs)
        return epoch_index % refresh_every == 0

    def _write_hard_negative_cache(
        self,
        *,
        epoch_index: int,
        records: Sequence[Mapping[str, float | int | str]],
    ) -> None:
        """Persist rank-local mined hard negatives for audit/reuse."""
        if not records:
            return
        cache_dir = self.model_dir / "hard_negative_caches"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = (
            cache_dir
            / f"epoch_{epoch_index + 1:04d}_rank_{self.distributed_context.rank:03d}.pt"
        )
        torch.save(
            {
                "epoch": epoch_index + 1,
                "rank": self.distributed_context.rank,
                "world_size": self.distributed_context.world_size,
                "records": list(records),
            },
            cache_path,
        )


def _load_graph_inputs(
    *,
    nodes: Sequence[str],
    embedding_repository: EmbeddingRepository,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one sampled protein set for TCCIG graph forward."""
    node_tuple = tuple(nodes)
    if len(node_tuple) < 2:
        raise ValueError("TCCIG graph training requires at least two proteins")
    embeddings = embedding_repository.get_many(node_tuple)
    embedding_tensors = [embeddings[protein_id] for protein_id in node_tuple]
    protein_embeddings = pad_sequence(embedding_tensors, batch_first=True).to(device)
    protein_lengths = torch.tensor(
        [embedding.size(0) for embedding in embedding_tensors],
        dtype=torch.long,
        device=device,
    )
    return protein_embeddings, protein_lengths


def _masked_mean_pool_embeddings(
    *,
    protein_embeddings: torch.Tensor,
    protein_lengths: torch.Tensor,
) -> torch.Tensor:
    """Pool cached token embeddings for the topology teacher."""
    clipped_lengths = protein_lengths.to(device=protein_embeddings.device).clamp(
        min=1,
        max=protein_embeddings.size(1),
    )
    token_ids = torch.arange(protein_embeddings.size(1), device=protein_embeddings.device)
    mask = token_ids.unsqueeze(0) < clipped_lengths.unsqueeze(1)
    weighted = protein_embeddings * mask.unsqueeze(-1).to(dtype=protein_embeddings.dtype)
    return weighted.sum(dim=1) / clipped_lengths.unsqueeze(-1).to(dtype=protein_embeddings.dtype)


def _canonical_node_pair(node_a: str, node_b: str) -> tuple[str, str]:
    return (node_a, node_b) if node_a <= node_b else (node_b, node_a)


def _candidate_supervision(
    *,
    graph: nx.Graph,
    nodes: Sequence[str],
    candidate_pairs: torch.Tensor,
    assigned_positive_edges: frozenset[tuple[str, str]],
    assigned_negative_edges: frozenset[tuple[str, str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build topology labels and sparse BCE masks for TCCIG candidates."""
    node_tuple = tuple(nodes)
    labels: list[float] = []
    bce_labels: list[float] = []
    bce_mask: list[float] = []
    has_assigned_supervision = bool(assigned_positive_edges or assigned_negative_edges)
    for source_index, target_index in candidate_pairs.t().detach().cpu().tolist():
        protein_a = node_tuple[int(source_index)]
        protein_b = node_tuple[int(target_index)]
        pair = _canonical_node_pair(protein_a, protein_b)
        topology_label = 1.0 if graph.has_edge(protein_a, protein_b) else 0.0
        pair_is_assigned_positive = pair in assigned_positive_edges
        pair_is_assigned_negative = pair in assigned_negative_edges
        labels.append(topology_label)
        bce_labels.append(
            1.0
            if pair_is_assigned_positive
            else 0.0
            if has_assigned_supervision
            else topology_label
        )
        bce_mask.append(
            1.0
            if not has_assigned_supervision
            or pair_is_assigned_positive
            or pair_is_assigned_negative
            else 0.0
        )
    return (
        torch.tensor(labels, dtype=torch.float32, device=device),
        torch.tensor(bce_labels, dtype=torch.float32, device=device),
        torch.tensor(bce_mask, dtype=torch.float32, device=device),
    )


def _hard_negative_cache_records(
    *,
    nodes: Sequence[str],
    pairs: torch.Tensor,
    score_matrix: torch.Tensor,
) -> list[dict[str, float | int | str]]:
    """Convert mined local-index pairs into stable cache records."""
    if pairs.numel() == 0:
        return []
    node_tuple = tuple(nodes)
    local_pairs = pairs.detach().cpu().long()
    local_scores = score_matrix.detach().float().cpu()
    records: list[dict[str, float | int | str]] = []
    for rank, (source_index, target_index) in enumerate(local_pairs.t().tolist()):
        source = int(source_index)
        target = int(target_index)
        if source >= len(node_tuple) or target >= len(node_tuple):
            continue
        records.append(
            {
                "rank": rank,
                "protein_a": node_tuple[source],
                "protein_b": node_tuple[target],
                "score": float(local_scores[source, target].item()),
            }
        )
    return records


def _candidate_pair_membership_mask(
    *,
    candidate_pairs: torch.Tensor,
    selected_pairs: torch.Tensor,
) -> torch.Tensor:
    """Return a mask for candidate pairs that match selected upper-triangle pairs."""
    mask = torch.zeros(candidate_pairs.size(1), dtype=torch.bool, device=candidate_pairs.device)
    if selected_pairs.numel() == 0 or candidate_pairs.numel() == 0:
        return mask
    selected = {
        (int(source), int(target))
        for source, target in selected_pairs.detach().cpu().long().t().tolist()
    }
    if not selected:
        return mask
    values = [
        (int(min(source, target)), int(max(source, target))) in selected
        for source, target in candidate_pairs.detach().cpu().long().t().tolist()
    ]
    return torch.tensor(values, dtype=torch.bool, device=candidate_pairs.device)


def _positive_edge_index_for_nodes(
    *,
    graph: nx.Graph,
    nodes: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Return subgraph positive edges as upper-triangle local node indices."""
    node_to_index = {node: index for index, node in enumerate(nodes)}
    edges: list[tuple[int, int]] = []
    for protein_a, protein_b in graph.subgraph(nodes).edges():
        index_a = node_to_index[protein_a]
        index_b = node_to_index[protein_b]
        if index_a == index_b:
            continue
        edges.append((min(index_a, index_b), max(index_a, index_b)))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(sorted(set(edges)), dtype=torch.long, device=device).t().contiguous()


def _degree_targets_for_nodes(
    *,
    graph: nx.Graph,
    nodes: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Return train-graph degree targets for a sampled protein set."""
    return torch.tensor(
        [float(graph.degree(node)) for node in nodes],
        dtype=torch.float32,
        device=device,
    )


def _struct_targets_for_nodes(
    *,
    artifacts: GraphPriorArtifacts | None,
    nodes: tuple[str, ...],
    device: torch.device,
    expected_dim: int,
) -> torch.Tensor | None:
    """Return structural distillation targets when artifact dimensions match."""
    if artifacts is None:
        return None
    artifact_index = {protein_id: row for row, protein_id in enumerate(artifacts.protein_ids)}
    if any(node not in artifact_index for node in nodes):
        return None
    targets = artifacts.structural_embeddings[
        torch.tensor([artifact_index[node] for node in nodes], dtype=torch.long)
    ]
    if targets.size(1) != expected_dim:
        return None
    return targets.to(device=device)


def _retrieval_losses_for_legacy_aggregates(
    retrieval_losses: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Expose retrieval losses through the existing TCCIG CSV aggregate keys."""
    reference = retrieval_losses["total"]
    zero = reference.sum() * 0.0
    return {
        "edge": retrieval_losses["reranker"],
        "teacher": zero,
        "budget": zero,
        "relative_density": zero,
        "degree_mmd": retrieval_losses["degree"],
        "clustering_mmd": zero,
        "rank": retrieval_losses["retrieval"],
        "module": zero,
        "spectral": zero,
        "calibration": zero,
        "sparse": retrieval_losses["hard_negative"],
        "total": retrieval_losses["total"],
    }


def _topology_losses_for_aggregates(
    *,
    losses: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map TCCIG loss terms onto the existing topology-train CSV columns."""
    zero_losses = topology_train._zero_topology_losses(reference=reference)
    zero_losses["relative_density"] = losses["relative_density"]
    zero_losses["degree_mmd"] = losses["degree_mmd"]
    zero_losses["clustering_mmd"] = losses["clustering_mmd"]
    zero_losses["total_topology"] = (
        losses["relative_density"] + losses["degree_mmd"] + losses["clustering_mmd"]
    )
    return zero_losses


def _local_tasks(
    *,
    epoch_plan: EdgeCoverEpochPlan,
    distributed_context: DistributedContext,
) -> tuple[topology_train.LocalSubgraphTask, ...]:
    assigned_negative_edges = epoch_plan.assigned_negative_edges or tuple(
        frozenset() for _ in epoch_plan.subgraphs
    )
    return topology_train._local_subgraph_tasks_for_rank(
        subgraphs=epoch_plan.subgraphs,
        assigned_positive_edges=epoch_plan.assigned_positive_edges,
        assigned_negative_edges=assigned_negative_edges,
        distributed_context=distributed_context,
    )
