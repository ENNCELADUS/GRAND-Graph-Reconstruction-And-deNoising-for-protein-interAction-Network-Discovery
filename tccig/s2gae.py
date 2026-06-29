"""S2GAE-style residual denoiser for the standalone TCCIG pipeline."""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import SupportsFloat, cast

import networkx as nx
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score
from src.embed import load_cached_embedding
from src.topology.finetune_data import (
    TOPOLOGY_EVAL_NODE_SIZES,
    InternalValidationPlan,
)
from src.topology.finetune_losses import (
    TopologyLossWeights,
    TopologyLossWeightSchedule,
    compute_topology_losses,
    topology_loss_scale,
)
from src.topology.metrics import evaluate_graph_samples
from src.train.config import LossConfig
from src.utils.losses import binary_classification_loss
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from tccig.prepare import (
    CandidatePair,
    EdgeSamplingConfig,
    EdgeTargetDataset,
    GraphRule,
    RefineRequest,
    SplitBundle,
    TCCIGRuntime,
    TrainRefinerRequest,
    canonical_edge,
    classify_scorer_error_targets,
    collate_edge_targets,
    edges_from_rule,
    ordered_probabilities_from_indexed_rows,
    parse_edge_sampling_config,
    sample_epoch_edge_targets,
    write_json,
)
from tccig.topology_subset import (
    TopologySubgraphEpochChunk,
    TopologySubgraphPlan,
    TopologySubsetPlan,
    TopologySubsetSamplerConfig,
    _all_local_pairs,
    compute_subset_bias_diagnostic,
    group_epoch_samples_by_subgraph,
    sample_epoch_topology_subset,
    select_diagnostic_subgraphs,
)

LOGGER = logging.getLogger(__name__)

SUPPORTED_MONITOR_METRICS = {
    "val_topology_loss",
    "internal_val_graph_sim",
    "val_graph_sim",
    "internal_val_relative_density",
    "val_relative_density",
    "val_auprc",
}
INTERNAL_VALIDATION_SUMMARY_KEYS = ("graph_sim", "relative_density", "deg_dist_mmd", "cc_mmd")
SUPPORTED_ENCODER_AGGREGATIONS = {"add", "mean", "max"}
TCCIG_TRAIN_CSV_COLUMNS = [
    "Epoch",
    "Epoch Time",
    "Train Loss",
    "Train BCE Loss",
    "Train Residual Anchor Loss",
    "Train Weighted Residual Anchor Loss",
    "Train Gradient Norm",
    "Val auprc",
    "Val Topology Loss",
    "Internal Val graph_sim",
    "Internal Val relative_density",
    "Internal Val deg_dist_mmd",
    "Internal Val cc_mmd",
    "Selected Rule Type",
    "Selected Rule Positive Edges",
    "Monitor Metric",
    "Monitor Value",
    "Peak GPU Mem MB",
    "Learning Rate",
]


@dataclass(frozen=True)
class S2GAEConfig:
    """Parsed S2GAE refiner configuration."""

    encoder: str
    input_dim: int
    hidden_dim: int
    num_layers: int
    decoder_hidden_dim: int
    decoder_layers: int
    dropout: float
    epochs: int
    batch_size: int
    edge_sampling: EdgeSamplingConfig
    loss_config: LossConfig
    residual_weight: float
    residual_scale: float | None
    encoder_aggr: str
    layer_norm: bool
    monitor_metric: str
    topology_validation: S2GAETopologyValidationConfig
    optimizer: S2GAEOptimizerConfig
    scheduler: S2GAESchedulerConfig
    optimization: S2GAEOptimizationConfig
    embedding_cache_dir: Path
    embedding_index_path: Path
    max_sequence_length: int | None
    checkpoint_path: Path
    log_dir: Path
    residual_anchor: S2GAEResidualAnchorConfig
    topology_training: S2GAETopologyTrainingConfig


@dataclass
class S2GAERefinerState:
    """In-memory and on-disk state for a trained S2GAE refiner."""

    model: S2GAERefiner
    config: S2GAEConfig
    best_validation_auprc: float
    best_monitor_value: float
    selected_rule: GraphRule | None
    selected_rule_payload: dict[str, object] | None
    epochs_trained: int


@dataclass(frozen=True)
class S2GAEOptimizerConfig:
    """Optimizer hyperparameters for the S2GAE refiner."""

    optimizer_type: str
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float


@dataclass(frozen=True)
class S2GAESchedulerConfig:
    """Scheduler configuration for the S2GAE refiner."""

    scheduler_type: str


@dataclass(frozen=True)
class S2GAEOptimizationConfig:
    """Backward and optimization-loop controls."""

    gradient_clip_norm: float | None


@dataclass(frozen=True)
class S2GAEResidualAnchorConfig:
    """Residual anchor form and weight for the training loss."""

    form: str
    weight: float


@dataclass(frozen=True)
class S2GAETopologyTrainingConfig:
    """Differentiable topology-loss controls for the training objective."""

    enabled: bool
    node_sizes: tuple[int, ...]
    samples_per_size: int
    strategy: str
    seed: int
    coverage_augmentation: bool
    topology_weight: float
    weights: TopologyLossWeights
    warmup_epochs: int
    ramp_epochs: int
    schedule: str
    subset: TopologySubsetSamplerConfig


@dataclass(frozen=True)
class S2GAETopologyLossWeights:
    """Weights for hard-metric validation topology loss."""

    alpha: float
    beta: float
    gamma: float
    delta: float


@dataclass(frozen=True)
class S2GAETopologyValidationConfig:
    """Controls for validation-time hard topology evaluation."""

    enabled: bool
    inference_batch_size: int
    compute_clustering_mmd: bool
    losses: S2GAETopologyLossWeights


@dataclass(frozen=True)
class S2GAELossTerms:
    """S2GAE training loss components."""

    bce: torch.Tensor
    residual_anchor: torch.Tensor
    weighted_residual_anchor: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class ValidationTopologyRuleEvaluation:
    """Topology validation result for one selected hard-graph rule."""

    rule: GraphRule
    validation_metrics: dict[str, float | int]
    rule_payload: Mapping[str, object]


class S2GAERefiner(nn.Module):
    """PyG GNN encoder plus official S2GAE-style cross-layer decoder."""

    def __init__(
        self,
        *,
        encoder: str,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        decoder_hidden_dim: int,
        decoder_layers: int,
        dropout: float,
        encoder_aggr: str = "add",
        layer_norm: bool = False,
        residual_scale: float | None = None,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.dropout = dropout
        self.encoder_name = encoder.lower()
        self.residual_scale = residual_scale
        self.convs = nn.ModuleList()
        graph_conv = _load_graph_conv()
        for layer_index in range(num_layers):
            in_channels = input_dim if layer_index == 0 else hidden_dim
            if self.encoder_name != "graphconv":
                raise ValueError("refiner.encoder must be 'graphconv'")
            self.convs.append(graph_conv(in_channels, hidden_dim, aggr=encoder_aggr))
        self.feature_norm = nn.LayerNorm(input_dim) if layer_norm else None
        self.hidden_norms = (
            nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_layers))
            if layer_norm
            else None
        )
        self.decoder = CrossLayerDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_layers=decoder_layers,
            dropout=dropout,
        )

    def encode(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return one hidden representation per GNN layer."""
        hidden_states: list[torch.Tensor] = []
        x = node_features if self.feature_norm is None else self.feature_norm(node_features)
        for layer_index, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if self.hidden_norms is not None:
                x = self.hidden_norms[layer_index](x)
            if layer_index < len(self.convs) - 1:
                x = functional.relu(x)
                x = functional.dropout(x, p=self.dropout, training=self.training)
            hidden_states.append(x)
        return hidden_states

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        pair_index: torch.Tensor,
        pairwise_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return refined logits and residual deltas for candidate pairs."""
        hidden_states = self.encode(
            node_features=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        return self.decode(
            hidden_states=hidden_states,
            pair_index=pair_index,
            pairwise_probabilities=pairwise_probabilities,
        )

    def decode(
        self,
        *,
        hidden_states: Sequence[torch.Tensor],
        pair_index: torch.Tensor,
        pairwise_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return refined logits and the applied residual for already-encoded states."""
        raw_delta = self.decoder(hidden_states=hidden_states, pair_index=pair_index)
        applied_delta = _bounded_residual(raw_delta, self.residual_scale)
        refined_logits = torch.logit(_clamp_probabilities(pairwise_probabilities)) + applied_delta
        return refined_logits, applied_delta


class CrossLayerDecoder(nn.Module):
    """S2GAE link decoder using all source/destination layer products."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        decoder_hidden_dim: int,
        decoder_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if decoder_layers <= 0:
            raise ValueError("decoder_layers must be positive")
        self.dropout = dropout
        input_dim = hidden_dim * num_layers * num_layers + hidden_dim
        layers: list[nn.Linear] = []
        if decoder_layers == 1:
            layers.append(nn.Linear(input_dim, 1))
        else:
            layers.append(nn.Linear(input_dim, decoder_hidden_dim))
            for _ in range(decoder_layers - 2):
                layers.append(nn.Linear(decoder_hidden_dim, decoder_hidden_dim))
            layers.append(nn.Linear(decoder_hidden_dim, 1))
        self.layers = nn.ModuleList(layers)
        self.reset_output_layer()

    def reset_output_layer(self) -> None:
        """Initialize the residual decoder to preserve pairwise logits."""
        output_layer = cast(nn.Linear, self.layers[-1])
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

    def forward(
        self,
        *,
        hidden_states: Sequence[torch.Tensor],
        pair_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode candidate-pair residual logits from cross-layer products."""
        src_index = pair_index[0]
        dst_index = pair_index[1]
        products: list[torch.Tensor] = []
        for src_hidden in hidden_states:
            src_values = src_hidden[src_index]
            for dst_hidden in hidden_states:
                products.append(src_values * dst_hidden[dst_index])
        final_src_values = hidden_states[-1][src_index]
        final_dst_values = hidden_states[-1][dst_index]
        products.append(torch.abs(final_src_values - final_dst_values))
        x = torch.cat(products, dim=1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        output_layer = cast(nn.Linear, self.layers[-1])
        return cast(torch.Tensor, output_layer(x).squeeze(-1))


def _clamp_probabilities(pairwise_probabilities: torch.Tensor) -> torch.Tensor:
    return pairwise_probabilities.clamp(min=1.0e-6, max=1.0 - 1.0e-6)


def _bounded_residual(delta_logits: torch.Tensor, residual_scale: float | None) -> torch.Tensor:
    """Soft-clamp the residual to ±residual_scale, keeping unit slope near zero.

    Uses ``s * tanh(x / s)`` rather than ``s * tanh(x)`` so a small raw delta passes
    through near-unchanged (slope 1 at the origin) instead of being amplified by ``s``,
    while still saturating at ±s. With ``residual_scale=None`` the residual is unbounded.
    """
    if residual_scale is None:
        return delta_logits
    return residual_scale * torch.tanh(delta_logits / residual_scale)


def asymmetric_residual_anchor(delta_logits: torch.Tensor) -> torch.Tensor:
    """Penalize only upward (edge-adding) residual pushes; deletion is free.

    The symmetric ``delta.pow(2).mean()`` anchor pulls the refiner toward identity
    and equally discourages negative (edge-deleting) deltas. This one-sided variant
    leaves ``delta < 0`` unpenalized so the topology loss can drive pruning.
    """
    return delta_logits.clamp(min=0.0).pow(2).mean()


def residual_refined_logits(
    pairwise_probabilities: torch.Tensor,
    delta_logits: torch.Tensor,
    residual_scale: float | None = None,
) -> torch.Tensor:
    """Add the (optionally bounded) S2GAE residual to clamped pairwise logits."""
    applied_delta = _bounded_residual(delta_logits, residual_scale)
    return torch.logit(_clamp_probabilities(pairwise_probabilities)) + applied_delta


def s2gae_loss_terms(
    *,
    refined_logits: torch.Tensor,
    labels: torch.Tensor,
    delta_logits: torch.Tensor,
    loss_config: LossConfig,
    residual_weight: float,
    anchor_form: str = "symmetric",
) -> S2GAELossTerms:
    """Compute supervised denoising BCE plus the configured residual anchor."""
    bce = binary_classification_loss(
        logits=refined_logits,
        labels=labels,
        loss_config=loss_config,
    )
    if anchor_form == "asymmetric_relu":
        residual_anchor = asymmetric_residual_anchor(delta_logits)
    elif anchor_form == "symmetric":
        residual_anchor = delta_logits.pow(2).mean()
    else:
        raise ValueError(f"Unsupported residual anchor form: {anchor_form}")
    weighted_residual_anchor = residual_weight * residual_anchor
    return S2GAELossTerms(
        bce=bce,
        residual_anchor=residual_anchor,
        weighted_residual_anchor=weighted_residual_anchor,
        total=bce + weighted_residual_anchor,
    )


def _pair_lookup(all_pairs: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Map each query pair column to its index in ``all_pairs`` (undirected match)."""
    node_count = int(all_pairs.max().item()) + 1 if all_pairs.numel() else 1
    all_min = torch.minimum(all_pairs[0], all_pairs[1])
    all_max = torch.maximum(all_pairs[0], all_pairs[1])
    all_codes = all_min * node_count + all_max
    query_min = torch.minimum(query[0], query[1])
    query_max = torch.maximum(query[0], query[1])
    query_codes = query_min * node_count + query_max
    # Sort the reference codes once and binary-search each query code into it.
    order = torch.argsort(all_codes)
    sorted_codes = all_codes[order]
    positions = torch.searchsorted(sorted_codes, query_codes)
    positions = positions.clamp(max=sorted_codes.numel() - 1)
    if not bool(torch.all(sorted_codes[positions] == query_codes)):
        raise ValueError("topology plan pair absent from split graph pair_index")
    return order[positions]


def topology_plan_loss(
    *,
    refiner: S2GAERefiner,
    graph: _SplitGraph,
    plan: InternalValidationPlan,
    node_index: Mapping[str, int],
    weights: TopologyLossWeights,
    include_clustering_mmd: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Differentiable topology loss over all plan buckets (per-epoch full plan)."""
    device = graph.node_features.device
    hidden_states = refiner.encode(
        node_features=graph.node_features,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
    )
    totals = {
        "graph_sim": 0.0,
        "relative_density": 0.0,
        "degree_mmd": 0.0,
        "clustering_mmd": 0.0,
    }
    total_loss = graph.node_features.new_zeros(())
    bucket_count = 0
    for bucket in plan.buckets:
        for subgraph_index, nodes in enumerate(bucket.sampled_subgraphs):
            records = [r for r in bucket.pair_records if r.subgraph_index == subgraph_index]
            if len(records) == 0 or len(nodes) < 2:
                continue
            global_pairs = (
                torch.tensor(
                    [[node_index[r.protein_a], node_index[r.protein_b]] for r in records],
                    dtype=torch.long,
                    device=device,
                )
                .t()
                .contiguous()
            )
            refined_logits, _ = refiner.decode(
                hidden_states=hidden_states,
                pair_index=global_pairs,
                pairwise_probabilities=graph.pairwise_probabilities[
                    _pair_lookup(graph.pair_index, global_pairs)
                ],
            )
            pred = torch.sigmoid(refined_logits)
            target_graph = bucket.target_subgraphs[subgraph_index]
            target = torch.tensor(
                [1.0 if target_graph.has_edge(r.protein_a, r.protein_b) else 0.0 for r in records],
                dtype=torch.float32,
                device=device,
            )
            pair_a = torch.tensor(
                [r.pair_index_a for r in records], dtype=torch.long, device=device
            )
            pair_b = torch.tensor(
                [r.pair_index_b for r in records], dtype=torch.long, device=device
            )
            terms = compute_topology_losses(
                weights=weights,
                num_nodes=len(nodes),
                pair_index_a=pair_a,
                pair_index_b=pair_b,
                pred_pair_probabilities=pred,
                target_pair_probabilities=target,
                include_clustering_mmd=include_clustering_mmd,
            )
            total_loss = total_loss + terms["total_topology"]
            for key, term_key in (
                ("graph_sim", "graph_similarity"),
                ("relative_density", "relative_density"),
                ("degree_mmd", "degree_mmd"),
                ("clustering_mmd", "clustering_mmd"),
            ):
                totals[key] += float(terms[term_key].detach().item())
            bucket_count += 1
    if bucket_count > 0:
        total_loss = total_loss / bucket_count
        totals = {key: value / bucket_count for key, value in totals.items()}
    components = {**totals, "total": float(total_loss.detach().item())}
    return total_loss, components


def topology_subset_chunk_loss(
    *,
    refined_logits: torch.Tensor,
    chunk: TopologySubgraphEpochChunk,
    weights: TopologyLossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted topology loss for one epoch subgraph chunk."""
    device = refined_logits.device
    pred = torch.sigmoid(refined_logits)
    pair_a = torch.tensor(
        [sample.local_index_a for sample in chunk.samples], dtype=torch.long, device=device
    )
    pair_b = torch.tensor(
        [sample.local_index_b for sample in chunk.samples], dtype=torch.long, device=device
    )
    target = torch.tensor(
        [sample.target for sample in chunk.samples], dtype=torch.float32, device=device
    )
    pair_weights = torch.tensor(
        [1.0 / sample.pi_total for sample in chunk.samples],
        dtype=torch.float32,
        device=device,
    )
    terms = compute_topology_losses(
        weights=weights,
        num_nodes=chunk.node_size,
        pair_index_a=pair_a,
        pair_index_b=pair_b,
        pred_pair_probabilities=pred,
        target_pair_probabilities=target,
        pair_weights=pair_weights,
        include_clustering_mmd=False,
    )
    components = {
        "graph_sim": float(terms["graph_similarity"].detach().cpu().item()),
        "relative_density": float(terms["relative_density"].detach().cpu().item()),
        "degree_mmd": float(terms["degree_mmd"].detach().cpu().item()),
        "clustering_mmd": 0.0,
        "total": float(terms["total_topology"].detach().cpu().item()),
        "sample_count": float(len(chunk.samples)),
    }
    return terms["total_topology"], components


def _size_balanced_chunk_scales(node_sizes: Sequence[int]) -> list[float]:
    """Return per-chunk scales for the mean-over-size, mean-over-subgraphs objective.

    `node_sizes` MUST be the full (pre-shard) epoch chunk list so the per-size counts
    are global. Each scale is `1 / (S * N_s)` with S = number of distinct sizes and
    N_s = number of chunks of that size.
    """
    counts: dict[int, int] = {}
    for size in node_sizes:
        counts[int(size)] = counts.get(int(size), 0) + 1
    active_size_count = len(counts)
    if active_size_count == 0:
        raise ValueError("topology subset chunk list must not be empty")
    return [1.0 / float(active_size_count * counts[int(size)]) for size in node_sizes]


def _shard_chunks_for_rank(
    *,
    node_sizes: Sequence[int],
    rank: int,
    world_size: int,
) -> list[tuple[int, float]]:
    """Return (global_index, global_scale) for the chunks owned by `rank`.

    The full `node_sizes` list is identical on every rank (deterministic epoch
    sampling), so global scales are computed here and the disjoint shard is taken by
    strided global index. The union over ranks covers every chunk exactly once.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    global_scales = _size_balanced_chunk_scales(node_sizes)
    return [
        (global_index, global_scales[global_index])
        for global_index in range(len(node_sizes))
        if global_index % world_size == rank
    ]


def _all_reduce_topology_gradients(refiner: S2GAERefiner, runtime: TCCIGRuntime) -> None:
    """SUM topology gradients across ranks for the Accelerate-unwrapped refiner.

    This is the ONLY custom collective in the topology path; everything else
    (launcher, DDP wrapping, AMP scaler, optimizer.step) stays Accelerate-managed.
    `refiner` is the object returned by `_unwrap_refiner` (i.e. `accelerator.unwrap_model`).

    Every rank MUST reduce the SAME parameter set in the SAME order or the collective
    deadlocks. A rank whose shard was empty this epoch has no `.grad`, so we materialize
    a zero grad for every trainable parameter before reducing — making participation
    uniform. SUM is correct (no `world_size` division) because the shards are disjoint and
    the per-chunk scales are global. The grads may still be AMP-scaled here; that is fine
    because the scale is identical across ranks and Accelerate unscales uniformly at
    clip/step time.
    """
    if not runtime.is_distributed:
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    for parameter in refiner.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)


def _all_reduce_component_sums(
    component_sums: dict[str, float], runtime: TCCIGRuntime
) -> dict[str, float]:
    """SUM-reduce rank-local topology component sums into global totals for logging.

    Each rank accumulated only its own shard's (globally-scaled) component contributions,
    so the per-rank dict is a partial sum of the full objective. A SUM all-reduce over the
    fixed key order reconstructs the same totals every rank would log in single-process
    mode. Detached scalars only — this never touches autograd or the optimizer. No-op when
    not distributed (the local dict is already the full sum).
    """
    if not runtime.is_distributed:
        return component_sums
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return component_sums
    keys = sorted(component_sums)
    buffer = torch.tensor(
        [component_sums[key] for key in keys],
        dtype=torch.float64,
        device=runtime.device,
    )
    torch.distributed.all_reduce(buffer, op=torch.distributed.ReduceOp.SUM)
    reduced = buffer.detach().cpu().tolist()
    return {key: float(value) for key, value in zip(keys, reduced, strict=True)}


def _topology_subset_backward_step(
    *,
    refiner: S2GAERefiner,
    graph: _SplitGraph,
    chunks: Sequence[TopologySubgraphEpochChunk],
    node_index: Mapping[str, int],
    weights: TopologyLossWeights,
    runtime: TCCIGRuntime,
    topology_scale: float,
    topology_weight: float,
) -> dict[str, float]:
    """Run sharded, memory-bounded per-chunk topology backward; return component sums.

    `chunks` is the FULL epoch chunk list, identical on every rank. This rank owns the
    disjoint shard `chunks[rank::world_size]`; each owned chunk is scaled by its global
    `1/(S*N_s)` scale and backpropagated immediately. The encode is recomputed inside the
    loop so peak memory is one chunk's forward graph (see Step 4 constraint 2). Backward
    runs through `runtime.accelerator.backward` so AMP stays managed; one explicit
    all-reduce after the loop yields the exact full-objective gradient (constraint 1).
    """
    # NOTE: the component keys MUST match what the full-plan path produces so the shared
    # epoch_history logging (`topology_components["clustering_mmd"]`, s2gae.py:920) does not
    # KeyError. Training clustering is off (Task 8), so clustering_mmd is a constant 0.0.
    if not chunks:
        return {
            "total": 0.0,
            "graph_sim": 0.0,
            "relative_density": 0.0,
            "degree_mmd": 0.0,
            "clustering_mmd": 0.0,
        }
    rank = runtime.rank if runtime.is_distributed else 0
    world_size = runtime.world_size if runtime.is_distributed else 1
    shard = _shard_chunks_for_rank(
        node_sizes=[chunk.node_size for chunk in chunks],
        rank=rank,
        world_size=world_size,
    )
    component_sums = {
        "total": 0.0,
        "graph_sim": 0.0,
        "relative_density": 0.0,
        "degree_mmd": 0.0,
        "clustering_mmd": 0.0,
    }
    for global_index, scale in shard:
        chunk = chunks[global_index]
        # Recompute encode INSIDE the chunk so its graph is freed by this chunk's
        # backward; do NOT hoist this out of the loop (would need retain_graph -> OOM).
        hidden_states = refiner.encode(
            node_features=graph.node_features,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
        )
        global_pairs = (
            torch.tensor(
                [
                    [node_index[sample.protein_a], node_index[sample.protein_b]]
                    for sample in chunk.samples
                ],
                dtype=torch.long,
                device=graph.node_features.device,
            )
            .t()
            .contiguous()
        )
        refined_logits, _ = refiner.decode(
            hidden_states=hidden_states,
            pair_index=global_pairs,
            pairwise_probabilities=graph.pairwise_probabilities[
                _pair_lookup(graph.pair_index, global_pairs)
            ],
        )
        chunk_loss, components = topology_subset_chunk_loss(
            refined_logits=refined_logits,
            chunk=chunk,
            weights=weights,
        )
        scaled = topology_scale * topology_weight * scale * chunk_loss
        # Accelerate-managed backward (keeps AMP scaler); accumulates into refiner.grad.
        runtime.accelerator.backward(scaled)
        for key in component_sums:
            component_sums[key] += components[key] * scale
    _all_reduce_topology_gradients(refiner, runtime)
    # component_sums are RANK-LOCAL: each rank only walked its own disjoint shard. The
    # gradient is made global by the all-reduce above, but these detached scalars are not,
    # so SUM-reduce them too — otherwise the logged train_topology_* values would report
    # only this rank's shard. No-op in single-process mode.
    return _all_reduce_component_sums(component_sums, runtime)


def _subgraph_bias_diagnostic(
    *,
    refiner: S2GAERefiner,
    graph: _SplitGraph,
    subgraph: TopologySubgraphPlan,
    chunk: TopologySubgraphEpochChunk,
    node_index: Mapping[str, int],
    diagnostic_full_space: Mapping[str, Mapping[str, float]],
) -> dict[str, float] | None:
    """IPW-vs-full-space bias diagnostic for one pre-scored capped subgraph."""
    scorer_probs = diagnostic_full_space.get(subgraph.subgraph_id)
    if scorer_probs is None:
        return None
    full_pairs = _all_local_pairs(tuple(subgraph.nodes))
    if any(row[4] not in scorer_probs for row in full_pairs):
        return None
    if any(row[2] not in node_index or row[3] not in node_index for row in full_pairs):
        return None
    device = graph.node_features.device
    with torch.no_grad():
        hidden_states = refiner.encode(
            node_features=graph.node_features,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
        )
        global_pairs = (
            torch.tensor(
                [[node_index[row[2]], node_index[row[3]]] for row in full_pairs],
                dtype=torch.long,
                device=device,
            )
            .t()
            .contiguous()
        )
        pairwise_probabilities = torch.tensor(
            [float(scorer_probs[row[4]]) for row in full_pairs],
            dtype=torch.float32,
            device=device,
        )
        refined_logits, _ = refiner.decode(
            hidden_states=hidden_states,
            pair_index=global_pairs,
            pairwise_probabilities=pairwise_probabilities,
        )
        full_probabilities = torch.sigmoid(refined_logits).detach().cpu().tolist()
    full_space_probabilities = {
        row[4]: float(prob) for row, prob in zip(full_pairs, full_probabilities, strict=True)
    }
    # IPW subset view: this epoch's drawn pairs for the SAME subgraph. pred is read from
    # the exact full-space decode above (the chunk pairs are a subset of the full set), so
    # estimate and reference share one decode and differ only by sampling + reweighting.
    subset_samples = [
        (sample.pair_id, full_space_probabilities[sample.pair_id], 1.0 / sample.pi_total)
        for sample in chunk.samples
        if sample.pair_id in full_space_probabilities
    ]
    if not subset_samples:
        return None
    return compute_subset_bias_diagnostic(
        node_size=subgraph.node_size,
        full_space_probabilities=full_space_probabilities,
        subset_samples=subset_samples,
    )


def _select_diagnostic_subgraphs(
    *,
    plan: TopologySubsetPlan,
    chunk_by_id: Mapping[str, TopologySubgraphEpochChunk],
    max_node_size: int,
    max_subgraphs: int,
    diagnostic_full_space: Mapping[str, Mapping[str, float]] | None = None,
) -> list[TopologySubgraphPlan]:
    """Return shared-selector diagnostics that have epoch samples and scores."""
    eligible_score_ids = (
        set(diagnostic_full_space)
        if diagnostic_full_space is not None
        else {subgraph.subgraph_id for subgraph in plan.subgraphs}
    )
    return [
        subgraph
        for subgraph in select_diagnostic_subgraphs(
            plan,
            max_node_size=max_node_size,
            max_subgraphs=max_subgraphs,
        )
        if subgraph.subgraph_id in chunk_by_id and subgraph.subgraph_id in eligible_score_ids
    ]


def _topology_bias_diagnostic_step(
    *,
    refiner: S2GAERefiner,
    graph: _SplitGraph,
    plan: TopologySubsetPlan,
    chunks: Sequence[TopologySubgraphEpochChunk],
    node_index: Mapping[str, int],
    diagnostic_full_space: Mapping[str, Mapping[str, float]],
    max_node_size: int,
    max_subgraphs: int,
) -> dict[str, float] | None:
    """Aggregate IPW-vs-full-space bias across a few capped subgraphs of the size mixture.

    Returns ``None`` when no eligible subgraph produced epoch samples this round (e.g.
    every active size exceeds ``max_node_size``). Otherwise returns mean and max relative
    error across the sampled subgraphs (so one badly-biased size cannot hide behind the
    others), plus the subgraph count and total full/subset pair counts. No grad is taken;
    this never touches the optimizer state.
    """
    if max_node_size < 2 or not chunks:
        return None
    chunk_by_id = {chunk.subgraph_id: chunk for chunk in chunks}
    targets = _select_diagnostic_subgraphs(
        plan=plan,
        chunk_by_id=chunk_by_id,
        max_node_size=max_node_size,
        max_subgraphs=max_subgraphs,
        diagnostic_full_space=diagnostic_full_space,
    )
    diagnostics: list[dict[str, float]] = []
    for subgraph in targets:
        per_subgraph = _subgraph_bias_diagnostic(
            refiner=refiner,
            graph=graph,
            subgraph=subgraph,
            chunk=chunk_by_id[subgraph.subgraph_id],
            node_index=node_index,
            diagnostic_full_space=diagnostic_full_space,
        )
        if per_subgraph is not None:
            diagnostics.append(per_subgraph)
    if not diagnostics:
        return None
    count = float(len(diagnostics))
    density_errors = [diag["density_relative_error"] for diag in diagnostics]
    degree_errors = [diag["mean_degree_relative_error"] for diag in diagnostics]
    return {
        "density_relative_error": sum(density_errors) / count,
        "mean_degree_relative_error": sum(degree_errors) / count,
        "max_density_relative_error": max(density_errors),
        "max_mean_degree_relative_error": max(degree_errors),
        "subgraphs": count,
        "full_space_pairs": sum(diag["full_space_pairs"] for diag in diagnostics),
        "subset_pairs": sum(diag["subset_pairs"] for diag in diagnostics),
    }


def build_s2gae_optimizer(
    *,
    model: nn.Module,
    config: S2GAEOptimizerConfig,
) -> Optimizer:
    """Build the configured optimizer over refiner parameters only."""
    if config.optimizer_type != "adamw":
        raise ValueError("refiner.optimizer.type must be 'adamw'")
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )


def apply_gradient_clipping(
    *,
    model: nn.Module,
    gradient_clip_norm: float | None,
) -> float:
    """Clip gradients when configured and return the observed norm."""
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        return 0.0
    if gradient_clip_norm is None:
        return _gradient_norm(parameters)
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
    return float(total_norm.detach().cpu().item())


def _zero_model_loss(model: nn.Module) -> torch.Tensor:
    """Return a zero loss connected to model parameters for empty DDP ranks."""
    loss: torch.Tensor | None = None
    for parameter in model.parameters():
        value = parameter.sum() * 0.0
        loss = value if loss is None else loss + value
    if loss is None:
        raise ValueError("S2GAE model must have trainable parameters")
    return loss


class _S2GAESampledTrainStepModule(nn.Module):
    """DDP-safe sampled-edge train step with per-batch input-edge masking."""

    def __init__(self, *, refiner: S2GAERefiner, cfg: S2GAEConfig) -> None:
        super().__init__()
        self.refiner = refiner
        self.cfg = cfg

    def forward(
        self,
        *,
        graph: _SplitGraph,
        pair_indices: torch.Tensor,
        labels: torch.Tensor,
        mask_input_edges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return batch mean loss and detached metric sums for sampled edge targets."""
        if pair_indices.numel() == 0:
            return (
                _zero_model_loss(self.refiner),
                torch.zeros(5, dtype=torch.float64, device=labels.device),
            )
        masked_graph = _masked_split_graph(
            graph=graph,
            masked_pair_indices=pair_indices[mask_input_edges],
        )
        hidden_states = self.refiner.encode(
            node_features=masked_graph.node_features,
            edge_index=masked_graph.edge_index,
            edge_weight=masked_graph.edge_weight,
        )
        refined_logits, delta = self.refiner.decode(
            hidden_states=hidden_states,
            pair_index=masked_graph.pair_index[:, pair_indices],
            pairwise_probabilities=masked_graph.pairwise_probabilities[pair_indices],
        )
        loss_terms = s2gae_loss_terms(
            refined_logits=refined_logits,
            labels=labels,
            delta_logits=delta,
            loss_config=self.cfg.loss_config,
            residual_weight=self.cfg.residual_anchor.weight,
            anchor_form=self.cfg.residual_anchor.form,
        )
        count = int(pair_indices.numel())
        return (
            loss_terms.total,
            torch.tensor(
                [
                    float(loss_terms.total.detach().item()) * count,
                    float(loss_terms.bce.detach().item()) * count,
                    float(loss_terms.residual_anchor.detach().item()) * count,
                    float(loss_terms.weighted_residual_anchor.detach().item()) * count,
                    float(count),
                ],
                dtype=torch.float64,
                device=labels.device,
            ),
        )


def load_mean_pooled_node_features(
    *,
    protein_ids: Sequence[str],
    cache_dir: Path,
    index_path: Path,
    input_dim: int,
    max_sequence_length: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Load cached ESM3 tensors and mean-pool them into node features."""
    embedding_index = _load_embedding_index(index_path)
    features = [
        load_cached_embedding(
            cache_dir=cache_dir,
            index=embedding_index,
            protein_id=protein_id,
            expected_input_dim=input_dim,
            max_sequence_length=max_sequence_length,
        ).mean(dim=0)
        for protein_id in protein_ids
    ]
    return torch.stack(features, dim=0).to(device)


def _node_index_from_split_bundle(bundle: SplitBundle) -> dict[str, int]:
    """Map protein IDs to the node ordering used by ``_build_split_graph``."""
    node_ids = _collect_node_ids(
        pairs=bundle.pairs,
        graph_edges=bundle.pairwise_graph_edges,
        extra_node_ids=bundle.extra_node_ids,
    )
    return {protein_id: index for index, protein_id in enumerate(node_ids)}


def train_refiner(request: TrainRefinerRequest) -> S2GAERefinerState:
    """Train an S2GAE residual denoiser on pairwise-generated train graphs."""
    cfg = _parse_config(request.config)
    device = torch.device(request.runtime.device)
    model = S2GAERefiner(
        encoder=cfg.encoder,
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        decoder_layers=cfg.decoder_layers,
        dropout=cfg.dropout,
        encoder_aggr=cfg.encoder_aggr,
        layer_norm=cfg.layer_norm,
        residual_scale=cfg.residual_scale,
    ).to(device)
    optimizer = build_s2gae_optimizer(model=model, config=cfg.optimizer)
    train_step = _S2GAESampledTrainStepModule(refiner=model, cfg=cfg).to(device)
    prepared = request.runtime.accelerator.prepare(train_step, optimizer)
    train_step_model, optimizer = _prepared_model_and_optimizer(prepared)

    train_graph = _build_split_graph(request.train, cfg=cfg, device=device)
    validation_graph = _build_split_graph(request.validation, cfg=cfg, device=device)
    validation_topology_graph: _SplitGraph | None = None
    if cfg.topology_validation.enabled:
        if request.validation_topology is None or request.validation_topology_plan is None:
            raise ValueError(
                "refiner.topology_validation.enabled requires validation_topology inputs"
            )
        validation_topology_graph = _build_split_graph(
            request.validation_topology,
            cfg=cfg,
            device=device,
        )
    elif cfg.monitor_metric != "val_auprc":
        raise ValueError(
            f"refiner.monitor_metric={cfg.monitor_metric!r} requires topology_validation.enabled"
        )
    train_topology_graph: _SplitGraph | None = None
    train_topology_node_index: dict[str, int] | None = None
    topology_schedule = TopologyLossWeightSchedule(
        warmup_epochs=cfg.topology_training.warmup_epochs,
        ramp_epochs=cfg.topology_training.ramp_epochs,
        schedule=cfg.topology_training.schedule,
    )
    if cfg.topology_training.enabled:
        if request.train_topology is None or request.train_topology_plan is None:
            raise ValueError("refiner.topology_training.enabled requires train_topology inputs")
        train_topology_graph = _build_split_graph(request.train_topology, cfg=cfg, device=device)
        train_topology_node_index = _node_index_from_split_bundle(request.train_topology)
    if request.train.loss_targets is None:
        raise ValueError("request.train.loss_targets is required for S2GAE training")
    validation_labels = _required_float_tensor(
        request.validation.candidate_labels,
        "request.validation.candidate_labels",
        device=device,
    )
    quadrants = classify_scorer_error_targets(
        pairs=request.train.pairs,
        labels=request.train.loss_targets,
        pairwise_graph_edges=request.train.pairwise_graph_edges,
    )
    quadrant_counts = {name: len(targets) for name, targets in quadrants.items()}

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_selected_rule: GraphRule | None = None
    best_selected_rule_payload: dict[str, object] | None = None
    best_validation_auprc = -math.inf
    best_monitor_value = _initial_monitor_value(cfg.monitor_metric)
    history: list[dict[str, float | int]] = []
    _prepare_tccig_train_csv(csv_path=cfg.log_dir / "tccig_train_step.csv", request=request)
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_step_model.train()
        epoch_targets = sample_epoch_edge_targets(
            quadrants=quadrants,
            sampling=cfg.edge_sampling,
            epoch=epoch,
        )
        loader = DataLoader(
            EdgeTargetDataset(epoch_targets),
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_edge_targets,
        )
        prepared_loader = request.runtime.accelerator.prepare(loader)
        local_loss_sums = torch.zeros(5, dtype=torch.float64, device=device)
        gradient_norm = 0.0
        for batch in cast(Iterable[Mapping[str, torch.Tensor]], prepared_loader):
            optimizer.zero_grad(set_to_none=True)
            batch_loss, batch_sums = cast(
                tuple[torch.Tensor, torch.Tensor],
                train_step_model(
                    graph=train_graph,
                    pair_indices=batch["pair_index"].to(device),
                    labels=batch["label"].to(device),
                    mask_input_edges=batch["mask_input_edge"].to(device),
                ),
            )
            request.runtime.accelerator.backward(batch_loss)
            if cfg.optimization.gradient_clip_norm is None:
                gradient_norm = apply_gradient_clipping(
                    model=train_step_model, gradient_clip_norm=None
                )
            else:
                clipped = request.runtime.accelerator.clip_grad_norm_(
                    train_step_model.parameters(),
                    cfg.optimization.gradient_clip_norm,
                )
                gradient_norm = float(clipped.detach().cpu().item())
            optimizer.step()
            local_loss_sums += batch_sums.detach()
        topology_components: dict[str, float] | None = None
        topology_bias_history: dict[str, float | int] = {}
        topology_scale = 0.0
        if cfg.topology_training.enabled and train_topology_graph is not None:
            assert train_topology_node_index is not None
            assert request.train_topology_plan is not None
            topology_scale = topology_loss_scale(epoch=epoch - 1, schedule=topology_schedule)
            if topology_scale > 0.0 and cfg.topology_training.topology_weight > 0.0:
                optimizer.zero_grad(set_to_none=True)
                topology_refiner = _unwrap_refiner(
                    train_step_model, request.runtime.accelerator
                )
                # The per-epoch topology backward runs on the unwrapped refiner over the
                # full-plan graph, which is identical on every rank. Dropout is the only
                # source of cross-rank nondeterminism, so disable it here to keep DDP
                # parameters in lock-step (identical inputs + no dropout -> identical
                # gradients -> identical optimizer step; no gradient all-reduce needed).
                topology_was_training = topology_refiner.training
                topology_refiner.eval()
                try:
                    if isinstance(request.train_topology_plan, TopologySubsetPlan):
                        epoch_samples = sample_epoch_topology_subset(
                            plan=request.train_topology_plan,
                            epoch=epoch,
                            config=cfg.topology_training.subset,
                        )
                        grouped = group_epoch_samples_by_subgraph(epoch_samples)
                        chunks = tuple(grouped[subgraph_id] for subgraph_id in sorted(grouped))
                        topology_components = _topology_subset_backward_step(
                            refiner=topology_refiner,
                            graph=train_topology_graph,
                            chunks=chunks,
                            node_index=train_topology_node_index,
                            weights=cfg.topology_training.weights,
                            runtime=request.runtime,
                            topology_scale=topology_scale,
                            topology_weight=cfg.topology_training.topology_weight,
                        )
                        diag_cfg = cfg.topology_training.subset
                        diagnostic_full_space = (
                            request.train_topology_diagnostic_full_space or {}
                        )
                        # §9 smoke sanity check: once, on epoch 1 (the first scaled step).
                        if (
                            isinstance(request.train_topology_plan, TopologySubsetPlan)
                            and epoch == 1
                            and request.runtime.is_main_process
                        ):
                            smoke_diag = _topology_bias_diagnostic_step(
                                refiner=topology_refiner,
                                graph=train_topology_graph,
                                plan=request.train_topology_plan,
                                chunks=chunks,
                                node_index=train_topology_node_index,
                                diagnostic_full_space=diagnostic_full_space,
                                max_node_size=diag_cfg.bias_diagnostic_max_node_size,
                                max_subgraphs=diag_cfg.bias_diagnostic_max_subgraphs,
                            )
                            if smoke_diag is not None:
                                LOGGER.info(
                                    "tccig topology smoke sanity check (§9): "
                                    "density_rel_err=%.4f mean_degree_rel_err=%.4f "
                                    "subgraphs=%s full_pairs=%s subset_pairs=%s",
                                    smoke_diag["density_relative_error"],
                                    smoke_diag["mean_degree_relative_error"],
                                    int(smoke_diag["subgraphs"]),
                                    int(smoke_diag["full_space_pairs"]),
                                    int(smoke_diag["subset_pairs"]),
                                )
                                topology_bias_history["topology_bias_density_rel_err"] = (
                                    smoke_diag["density_relative_error"]
                                )
                                topology_bias_history["topology_bias_mean_degree_rel_err"] = (
                                    smoke_diag["mean_degree_relative_error"]
                                )

                        # §3.7 production diagnostic: every N epochs across the size mixture.
                        every_n = diag_cfg.bias_diagnostic_every_n_epochs
                        if (
                            isinstance(request.train_topology_plan, TopologySubsetPlan)
                            and every_n > 0
                            and epoch % every_n == 0
                            and request.runtime.is_main_process
                        ):
                            prod_diag = _topology_bias_diagnostic_step(
                                refiner=topology_refiner,
                                graph=train_topology_graph,
                                plan=request.train_topology_plan,
                                chunks=chunks,
                                node_index=train_topology_node_index,
                                diagnostic_full_space=diagnostic_full_space,
                                max_node_size=diag_cfg.bias_diagnostic_max_node_size,
                                max_subgraphs=diag_cfg.bias_diagnostic_max_subgraphs,
                            )
                            if prod_diag is not None:
                                LOGGER.info(
                                    "tccig topology bias diagnostic (§3.7) epoch=%s: "
                                    "mean_density_rel_err=%.4f mean_degree_rel_err=%.4f "
                                    "max_density_rel_err=%.4f subgraphs=%s",
                                    epoch,
                                    prod_diag["density_relative_error"],
                                    prod_diag["mean_degree_relative_error"],
                                    prod_diag["max_density_relative_error"],
                                    int(prod_diag["subgraphs"]),
                                )
                                topology_bias_history["topology_bias_density_rel_err"] = (
                                    prod_diag["density_relative_error"]
                                )
                                topology_bias_history["topology_bias_mean_degree_rel_err"] = (
                                    prod_diag["mean_degree_relative_error"]
                                )
                                topology_bias_history["topology_bias_max_density_rel_err"] = (
                                    prod_diag["max_density_relative_error"]
                                )

                        if isinstance(request.train_topology_plan, TopologySubsetPlan):
                            topology_bias_history["train_topology_subset_pairs"] = len(
                                epoch_samples
                            )
                            topology_bias_history["train_topology_subset_subgraphs"] = len(chunks)
                    else:
                        topo_loss, topology_components = topology_plan_loss(
                            refiner=topology_refiner,
                            graph=train_topology_graph,
                            plan=cast(InternalValidationPlan, request.train_topology_plan),
                            node_index=train_topology_node_index,
                            weights=cfg.topology_training.weights,
                            include_clustering_mmd=cfg.topology_validation.compute_clustering_mmd,
                        )
                        scaled = topology_scale * cfg.topology_training.topology_weight * topo_loss
                        request.runtime.accelerator.backward(scaled)
                finally:
                    topology_refiner.train(topology_was_training)
                if cfg.optimization.gradient_clip_norm is not None:
                    request.runtime.accelerator.clip_grad_norm_(
                        train_step_model.parameters(), cfg.optimization.gradient_clip_norm
                    )
                optimizer.step()
        validation_model = _unwrap_refiner(train_step_model, request.runtime.accelerator)
        global_loss_sums = request.runtime.accelerator.reduce(local_loss_sums, reduction="sum")
        total_loss = float(global_loss_sums[0].detach().cpu().item())
        total_bce_loss = float(global_loss_sums[1].detach().cpu().item())
        total_residual_anchor_loss = float(global_loss_sums[2].detach().cpu().item())
        total_weighted_residual_anchor_loss = float(global_loss_sums[3].detach().cpu().item())
        global_train_count = int(round(float(global_loss_sums[4].detach().cpu().item())))

        validation_auprc = _validation_auprc(
            model=validation_model,
            graph=validation_graph,
            labels=validation_labels,
            batch_size=cfg.batch_size,
            runtime=request.runtime,
        )
        selected_epoch_rule: GraphRule | None = None
        selected_epoch_rule_payload: dict[str, object] | None = None
        selected_epoch_topology_metrics: dict[str, float | int] | None = None
        if cfg.topology_validation.enabled:
            if (
                validation_topology_graph is None
                or request.validation_topology is None
                or request.validation_topology_plan is None
            ):
                raise RuntimeError("Validation topology graph was not initialized")
            topology_evaluation = _evaluate_validation_topology_rules(
                model=validation_model,
                graph=validation_topology_graph,
                pairs=request.validation_topology.pairs,
                validation_plan=cast(InternalValidationPlan, request.validation_topology_plan),
                rule=request.graph_rule,
                validation_auprc=validation_auprc,
                cfg=cfg,
                runtime=request.runtime,
            )
            selected_epoch_rule = topology_evaluation.rule
            selected_epoch_rule_payload = dict(topology_evaluation.rule_payload)
            selected_epoch_topology_metrics = dict(topology_evaluation.validation_metrics)
            selected_epoch_topology_metrics["epoch"] = epoch
            monitor_value = _resolve_monitor_value(
                monitor_metric=cfg.monitor_metric,
                validation_auprc=validation_auprc,
                topology_metrics=topology_evaluation.validation_metrics,
            )
        else:
            monitor_value = validation_auprc
        epoch_denominator = max(1, global_train_count)
        epoch_history: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": total_loss / epoch_denominator,
            "train_bce_loss": total_bce_loss / epoch_denominator,
            "train_residual_anchor_loss": total_residual_anchor_loss / epoch_denominator,
            "train_weighted_residual_anchor_loss": (
                total_weighted_residual_anchor_loss / epoch_denominator
            ),
            "train_gradient_norm": gradient_norm,
            "learning_rate": _current_learning_rate(optimizer),
            "val_auprc": validation_auprc,
            "monitor_value": monitor_value,
            "peak_gpu_mem_mb": _peak_gpu_memory_mb(device),
            "sampled_edge_targets": len(epoch_targets),
            "train_fp_targets": quadrant_counts["fp"],
            "train_fn_targets": quadrant_counts["fn"],
            "train_tp_targets": quadrant_counts["tp"],
            "train_tn_targets": quadrant_counts["tn"],
        }
        if selected_epoch_topology_metrics is not None:
            metrics = selected_epoch_topology_metrics
            epoch_history.update(
                {
                    "val_topology_loss": float(metrics["val_topology_loss"]),
                    "internal_val_graph_sim": float(metrics["graph_sim"]),
                    "internal_val_relative_density": float(metrics["relative_density"]),
                    "internal_val_deg_dist_mmd": float(metrics["deg_dist_mmd"]),
                    "internal_val_cc_mmd": float(metrics["cc_mmd"]),
                    "selected_rule_positive_edges": float(metrics["positive_edges"]),
                }
            )
        epoch_history["train_topology_scale"] = topology_scale
        if topology_components is not None:
            epoch_history.update(
                {
                    "train_topology_loss": topology_components["total"],
                    "train_topo_graph_sim": topology_components["graph_sim"],
                    "train_topo_relative_density": topology_components["relative_density"],
                    "train_topo_degree_mmd": topology_components["degree_mmd"],
                    "train_topo_clustering_mmd": topology_components["clustering_mmd"],
                }
            )
        epoch_history.update(topology_bias_history)
        history.append(epoch_history)
        if request.runtime.is_main_process:
            epoch_time_s = time.perf_counter() - epoch_start
            _append_tccig_train_csv_row(
                csv_path=cfg.log_dir / "tccig_train_step.csv",
                epoch_history=epoch_history,
                selected_rule=selected_epoch_rule,
                monitor_metric=cfg.monitor_metric,
                epoch_time_s=epoch_time_s,
            )
            _log_epoch_summary(
                epoch_history=epoch_history,
                selected_rule=selected_epoch_rule,
                monitor_metric=cfg.monitor_metric,
                epoch_time_s=epoch_time_s,
            )

        if best_state_dict is None or _is_better_monitor(
            value=monitor_value,
            best_value=best_monitor_value,
            monitor_metric=cfg.monitor_metric,
        ):
            best_monitor_value = monitor_value
            best_validation_auprc = validation_auprc
            best_selected_rule = selected_epoch_rule
            best_selected_rule_payload = selected_epoch_rule_payload
            checkpoint_model = _unwrap_refiner(train_step_model, request.runtime.accelerator)
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in checkpoint_model.state_dict().items()
            }
        if request.runtime.is_main_process:
            _write_training_summary(
                cfg=cfg,
                best_monitor_value=best_monitor_value,
                best_validation_auprc=best_validation_auprc,
                best_selected_rule_payload=best_selected_rule_payload,
                optimizer=optimizer,
                history=history,
            )
        _runtime_barrier(request.runtime)

    if best_state_dict is None:
        raise RuntimeError("S2GAE training did not produce a checkpoint")
    checkpoint_model = _unwrap_refiner(train_step_model, request.runtime.accelerator)
    checkpoint_model.load_state_dict(best_state_dict)
    if request.runtime.is_main_process:
        cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": best_state_dict,
            "config": _config_to_json(cfg),
            "best_validation_auprc": best_validation_auprc,
            "monitor_metric": cfg.monitor_metric,
            "best_monitor_value": best_monitor_value,
            "selected_rule": (
                None if best_selected_rule_payload is None else best_selected_rule_payload
            ),
        }
        request.runtime.accelerator.save(payload, cfg.checkpoint_path, safe_serialization=False)
    _runtime_barrier(request.runtime)
    return S2GAERefinerState(
        model=checkpoint_model,
        config=cfg,
        best_validation_auprc=best_validation_auprc,
        best_monitor_value=best_monitor_value,
        selected_rule=best_selected_rule,
        selected_rule_payload=best_selected_rule_payload,
        epochs_trained=cfg.epochs,
    )


@torch.no_grad()
def predict_refined(request: RefineRequest) -> list[float]:
    """Predict refined probabilities for one split with a trained S2GAE state."""
    if not isinstance(request.refiner_state, S2GAERefinerState):
        raise TypeError("request.refiner_state must be an S2GAERefinerState")
    state = request.refiner_state
    device = torch.device(request.runtime.device)
    state.model.to(device)
    prediction_model = _unwrap_refiner(state.model, request.runtime.accelerator)
    prediction_model.eval()
    graph = _build_prediction_graph(request, cfg=state.config, device=device)
    return _prediction_probabilities(
        model=prediction_model,
        graph=graph,
        batch_size=state.config.batch_size,
        runtime=request.runtime,
    )


@dataclass(frozen=True)
class _SplitGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    pair_index: torch.Tensor
    pairwise_probabilities: torch.Tensor


def _build_split_graph(
    bundle: SplitBundle,
    *,
    cfg: S2GAEConfig,
    device: torch.device,
) -> _SplitGraph:
    return _build_graph(
        pairs=bundle.pairs,
        pairwise_probabilities=bundle.pairwise_probabilities,
        pairwise_graph_edges=bundle.pairwise_graph_edges,
        extra_node_ids=bundle.extra_node_ids,
        cfg=cfg,
        device=device,
    )


def _build_prediction_graph(
    request: RefineRequest,
    *,
    cfg: S2GAEConfig,
    device: torch.device,
) -> _SplitGraph:
    return _build_graph(
        pairs=request.pairs,
        pairwise_probabilities=request.pairwise_probabilities,
        pairwise_graph_edges=request.pairwise_graph_edges,
        cfg=cfg,
        device=device,
    )


def _build_graph(
    *,
    pairs: Sequence[CandidatePair],
    pairwise_probabilities: Sequence[float],
    pairwise_graph_edges: Sequence[tuple[str, str]],
    extra_node_ids: Sequence[str] | None = None,
    cfg: S2GAEConfig,
    device: torch.device,
) -> _SplitGraph:
    if len(pairs) != len(pairwise_probabilities):
        raise ValueError("pairs and pairwise_probabilities must have matching lengths")
    node_ids = _collect_node_ids(
        pairs=pairs,
        graph_edges=pairwise_graph_edges,
        extra_node_ids=extra_node_ids,
    )
    node_to_index = {protein_id: index for index, protein_id in enumerate(node_ids)}
    node_features = load_mean_pooled_node_features(
        protein_ids=node_ids,
        cache_dir=cfg.embedding_cache_dir,
        index_path=cfg.embedding_index_path,
        input_dim=cfg.input_dim,
        max_sequence_length=cfg.max_sequence_length,
        device=device,
    )
    edge_index, edge_weight = _edge_index_and_weight_from_edges(
        pairs=pairs,
        pairwise_probabilities=pairwise_probabilities,
        edges=pairwise_graph_edges,
        node_to_index=node_to_index,
        device=device,
    )
    return _SplitGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        pair_index=_pair_index_from_pairs(pairs=pairs, node_to_index=node_to_index, device=device),
        pairwise_probabilities=torch.tensor(
            [float(value) for value in pairwise_probabilities],
            dtype=torch.float32,
            device=device,
        ),
    )


def _masked_split_graph(*, graph: _SplitGraph, masked_pair_indices: torch.Tensor) -> _SplitGraph:
    """Return a graph with selected undirected pair edges removed from encoder input."""
    if masked_pair_indices.numel() == 0 or graph.edge_index.numel() == 0:
        return graph
    node_count = int(graph.node_features.size(0))
    edge_min = torch.minimum(graph.edge_index[0], graph.edge_index[1])
    edge_max = torch.maximum(graph.edge_index[0], graph.edge_index[1])
    edge_codes = edge_min * node_count + edge_max
    target_pairs = graph.pair_index[:, masked_pair_indices]
    target_min = torch.minimum(target_pairs[0], target_pairs[1])
    target_max = torch.maximum(target_pairs[0], target_pairs[1])
    target_codes = target_min * node_count + target_max
    keep_mask = ~torch.isin(edge_codes, target_codes)
    return _SplitGraph(
        node_features=graph.node_features,
        edge_index=graph.edge_index[:, keep_mask],
        edge_weight=graph.edge_weight[keep_mask],
        pair_index=graph.pair_index,
        pairwise_probabilities=graph.pairwise_probabilities,
    )


def _collect_node_ids(
    *,
    pairs: Sequence[CandidatePair],
    graph_edges: Sequence[tuple[str, str]],
    extra_node_ids: Sequence[str] | None = None,
) -> list[str]:
    protein_ids: set[str] = set()
    for pair in pairs:
        protein_ids.add(pair.protein_a)
        protein_ids.add(pair.protein_b)
    for protein_a, protein_b in graph_edges:
        protein_ids.add(protein_a)
        protein_ids.add(protein_b)
    if extra_node_ids is not None:
        protein_ids.update(str(node_id) for node_id in extra_node_ids)
    if not protein_ids:
        raise ValueError("S2GAE split graph requires at least one protein")
    return sorted(protein_ids)


def _edge_index_and_weight_from_edges(
    *,
    pairs: Sequence[CandidatePair],
    pairwise_probabilities: Sequence[float],
    edges: Sequence[tuple[str, str]],
    node_to_index: Mapping[str, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_weights_by_pair = _edge_weights_by_pair(
        pairs=pairs,
        pairwise_probabilities=pairwise_probabilities,
    )
    edge_columns: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    seen: set[tuple[str, str]] = set()
    for protein_a, protein_b in edges:
        edge = canonical_edge(protein_a, protein_b)
        if edge in seen:
            continue
        seen.add(edge)
        src = node_to_index[protein_a]
        dst = node_to_index[protein_b]
        if src == dst:
            continue
        weight = edge_weights_by_pair[edge]
        edge_columns.append((src, dst))
        edge_weights.append(weight)
        edge_columns.append((dst, src))
        edge_weights.append(weight)
    if edge_columns:
        edge_index = torch.tensor(edge_columns, dtype=torch.long, device=device).t().contiguous()
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32, device=device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_weight = torch.empty((0,), dtype=torch.float32, device=device)
    return edge_index, edge_weight


def _edge_weights_by_pair(
    *,
    pairs: Sequence[CandidatePair],
    pairwise_probabilities: Sequence[float],
) -> dict[tuple[str, str], float]:
    weights: dict[tuple[str, str], float] = {}
    for pair, probability in zip(pairs, pairwise_probabilities, strict=True):
        edge = canonical_edge(pair.protein_a, pair.protein_b)
        value = float(probability)
        weights[edge] = max(value, weights.get(edge, value))
    return weights


def _pair_index_from_pairs(
    *,
    pairs: Sequence[CandidatePair],
    node_to_index: Mapping[str, int],
    device: torch.device,
) -> torch.Tensor:
    indices = [(node_to_index[pair.protein_a], node_to_index[pair.protein_b]) for pair in pairs]
    return torch.tensor(indices, dtype=torch.long, device=device).t().contiguous()


def _validation_auprc(
    *,
    model: S2GAERefiner,
    graph: _SplitGraph,
    labels: torch.Tensor,
    batch_size: int,
    runtime: object,
) -> float:
    all_predictions = _prediction_probabilities(
        model=model,
        graph=graph,
        batch_size=batch_size,
        runtime=runtime,
    )
    all_labels = labels.detach().cpu().numpy()
    if len({float(label) for label in all_labels.tolist()}) < 2:
        return 0.0
    return float(average_precision_score(all_labels, all_predictions))


def _prediction_probabilities(
    *,
    model: S2GAERefiner,
    graph: _SplitGraph,
    batch_size: int,
    runtime: object,
) -> list[float]:
    model.eval()
    device = graph.pairwise_probabilities.device
    total = int(graph.pairwise_probabilities.numel())
    accelerator = getattr(runtime, "accelerator", None)

    loader: Iterable[tuple[torch.Tensor]] = DataLoader(
        TensorDataset(torch.arange(total, dtype=torch.long)),
        batch_size=max(1, batch_size),
        shuffle=False,
    )
    prepare_fn = getattr(accelerator, "prepare", None)
    if callable(prepare_fn):
        loader = cast(Iterable[tuple[torch.Tensor]], prepare_fn(loader))

    gather_fn = getattr(accelerator, "gather_for_metrics", None)
    is_distributed = bool(getattr(runtime, "is_distributed", False))

    indexed_rows: list[torch.Tensor] = []
    with torch.inference_mode():
        hidden_states = model.encode(
            node_features=graph.node_features,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
        )
        for (batch_indices,) in loader:
            batch_indices = batch_indices.to(device)
            refined_logits, _ = model.decode(
                hidden_states=hidden_states,
                pair_index=graph.pair_index[:, batch_indices],
                pairwise_probabilities=graph.pairwise_probabilities[batch_indices],
            )
            batch_rows = torch.stack(
                (
                    batch_indices.to(dtype=torch.float64),
                    torch.sigmoid(refined_logits).to(dtype=torch.float64),
                ),
                dim=1,
            )
            if is_distributed and callable(gather_fn):
                gathered = gather_fn(batch_rows)
                if isinstance(gathered, torch.Tensor):
                    batch_rows = gathered
            indexed_rows.append(batch_rows)
    local_rows = (
        torch.cat(indexed_rows, dim=0)
        if indexed_rows
        else torch.empty((0, 2), dtype=torch.float64, device=device)
    )
    return ordered_probabilities_from_indexed_rows(total=total, rows=local_rows)


def _evaluate_validation_topology_rules(
    *,
    model: S2GAERefiner,
    graph: _SplitGraph,
    pairs: Sequence[CandidatePair],
    validation_plan: InternalValidationPlan,
    rule: GraphRule,
    validation_auprc: float,
    cfg: S2GAEConfig,
    runtime: object,
) -> ValidationTopologyRuleEvaluation:
    refined_probabilities = _prediction_probabilities(
        model=model,
        graph=graph,
        batch_size=cfg.topology_validation.inference_batch_size,
        runtime=runtime,
    )
    if len(refined_probabilities) != len(pairs):
        raise ValueError("validation topology probabilities must match candidate pairs")

    metrics = _validation_topology_metrics(
        validation_plan=validation_plan,
        pairs=pairs,
        probabilities=refined_probabilities,
        rule=rule,
        validation_auprc=validation_auprc,
        cfg=cfg,
    )
    return ValidationTopologyRuleEvaluation(
        rule=rule,
        validation_metrics=metrics,
        rule_payload=rule.to_dict(),
    )


def _validation_topology_metrics(
    *,
    validation_plan: InternalValidationPlan,
    pairs: Sequence[CandidatePair],
    probabilities: Sequence[float],
    rule: GraphRule,
    validation_auprc: float,
    cfg: S2GAEConfig,
) -> dict[str, float | int]:
    selected_edges = set(
        edges_from_rule(
            pairs=list(pairs),
            probabilities=list(probabilities),
            rule=rule,
        )
    )
    pred_graphs_by_size: dict[int, list[nx.Graph]] = {}
    target_graphs_by_size: dict[int, list[nx.Graph]] = {}

    for bucket in validation_plan.buckets:
        pred_subgraphs = [nx.Graph() for _ in bucket.sampled_subgraphs]
        for subgraph, nodes in zip(pred_subgraphs, bucket.sampled_subgraphs, strict=True):
            subgraph.add_nodes_from(nodes)
        for record in bucket.pair_records:
            edge = canonical_edge(record.protein_a, record.protein_b)
            if edge not in selected_edges:
                continue
            pred_subgraphs[record.subgraph_index].add_edge(record.protein_a, record.protein_b)
        pred_graphs_by_size[bucket.node_size] = pred_subgraphs
        target_graphs_by_size[bucket.node_size] = list(bucket.target_subgraphs)

    result = evaluate_graph_samples(
        pred_graphs_by_size=pred_graphs_by_size,
        gt_graphs_by_size=target_graphs_by_size,
        include_spectral_stats=False,
        include_clustering_stats=cfg.topology_validation.compute_clustering_mmd,
    )
    summary = cast(Mapping[str, object], result["summary"])
    metrics = {
        name: float(cast(SupportsFloat, summary[name]))
        for name in INTERNAL_VALIDATION_SUMMARY_KEYS
    }
    metrics["val_topology_loss"] = _validation_topology_loss(
        internal_val_topology_stats=metrics,
        weights=cfg.topology_validation.losses,
        include_clustering_mmd=cfg.topology_validation.compute_clustering_mmd,
    )
    metrics["val_auprc"] = float(validation_auprc)
    metrics["positive_edges"] = int(len(selected_edges))
    return metrics


def _validation_topology_loss(
    *,
    internal_val_topology_stats: Mapping[str, float],
    weights: S2GAETopologyLossWeights,
    include_clustering_mmd: bool,
) -> float:
    graph_similarity_loss = 1.0 - float(internal_val_topology_stats["graph_sim"])
    relative_density_loss = (float(internal_val_topology_stats["relative_density"]) - 1.0) ** 2
    degree_mmd = float(internal_val_topology_stats["deg_dist_mmd"])
    clustering_mmd = float(internal_val_topology_stats["cc_mmd"]) if include_clustering_mmd else 0.0
    return (
        weights.alpha * graph_similarity_loss
        + weights.beta * relative_density_loss
        + weights.gamma * degree_mmd
        + weights.delta * clustering_mmd
    )


def _resolve_monitor_value(
    *,
    monitor_metric: str,
    validation_auprc: float,
    topology_metrics: Mapping[str, float | int],
) -> float:
    if monitor_metric == "val_auprc":
        return float(validation_auprc)
    if monitor_metric == "val_topology_loss":
        return float(topology_metrics["val_topology_loss"])
    if monitor_metric in {"internal_val_graph_sim", "val_graph_sim"}:
        return float(topology_metrics["graph_sim"])
    if monitor_metric in {"internal_val_relative_density", "val_relative_density"}:
        return -abs(float(topology_metrics["relative_density"]) - 1.0)
    raise ValueError(f"Unsupported refiner.monitor_metric: {monitor_metric}")


def _initial_monitor_value(monitor_metric: str) -> float:
    return math.inf if _resolve_monitor_mode(monitor_metric) == "min" else -math.inf


def _resolve_monitor_mode(monitor_metric: str) -> str:
    return "min" if monitor_metric == "val_topology_loss" else "max"


def _is_better_monitor(
    *,
    value: float,
    best_value: float,
    monitor_metric: str,
) -> bool:
    if _resolve_monitor_mode(monitor_metric) == "min":
        return value < best_value
    return value > best_value


def _runtime_barrier(runtime: object) -> None:
    wait_for_everyone = getattr(getattr(runtime, "accelerator", None), "wait_for_everyone", None)
    if callable(wait_for_everyone):
        wait_for_everyone()


def _peak_gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def _prepare_tccig_train_csv(*, csv_path: Path, request: TrainRefinerRequest) -> None:
    if not request.runtime.is_main_process:
        _runtime_barrier(request.runtime)
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TCCIG_TRAIN_CSV_COLUMNS)
        writer.writeheader()
    _runtime_barrier(request.runtime)


def _append_tccig_train_csv_row(
    *,
    csv_path: Path,
    epoch_history: Mapping[str, float | int],
    selected_rule: GraphRule | None,
    monitor_metric: str,
    epoch_time_s: float,
) -> None:
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TCCIG_TRAIN_CSV_COLUMNS)
        writer.writerow(
            {
                "Epoch": int(epoch_history["epoch"]),
                "Epoch Time": epoch_time_s,
                "Train Loss": float(epoch_history["train_loss"]),
                "Train BCE Loss": float(epoch_history["train_bce_loss"]),
                "Train Residual Anchor Loss": float(epoch_history["train_residual_anchor_loss"]),
                "Train Weighted Residual Anchor Loss": float(
                    epoch_history["train_weighted_residual_anchor_loss"]
                ),
                "Train Gradient Norm": float(epoch_history["train_gradient_norm"]),
                "Val auprc": float(epoch_history["val_auprc"]),
                "Val Topology Loss": epoch_history.get("val_topology_loss", ""),
                "Internal Val graph_sim": epoch_history.get("internal_val_graph_sim", ""),
                "Internal Val relative_density": epoch_history.get(
                    "internal_val_relative_density",
                    "",
                ),
                "Internal Val deg_dist_mmd": epoch_history.get(
                    "internal_val_deg_dist_mmd",
                    "",
                ),
                "Internal Val cc_mmd": epoch_history.get("internal_val_cc_mmd", ""),
                "Selected Rule Type": "" if selected_rule is None else selected_rule.type,
                "Selected Rule Positive Edges": epoch_history.get(
                    "selected_rule_positive_edges",
                    "",
                ),
                "Monitor Metric": monitor_metric,
                "Monitor Value": float(epoch_history["monitor_value"]),
                "Peak GPU Mem MB": float(epoch_history["peak_gpu_mem_mb"]),
                "Learning Rate": float(epoch_history["learning_rate"]),
            }
        )


def _log_epoch_summary(
    *,
    epoch_history: Mapping[str, float | int],
    selected_rule: GraphRule | None,
    monitor_metric: str,
    epoch_time_s: float,
) -> None:
    """Emit one Slurm-visible summary line for the completed epoch."""
    LOGGER.info(
        (
            "TCCIG epoch %s complete: time_s=%.2f train_loss=%.6f "
            "train_bce=%.6f val_auprc=%.6f monitor[%s]=%.6f "
            "val_topology_loss=%s rule=%s lr=%.6g peak_gpu_mem_mb=%.1f"
        ),
        int(epoch_history["epoch"]),
        epoch_time_s,
        float(epoch_history["train_loss"]),
        float(epoch_history["train_bce_loss"]),
        float(epoch_history["val_auprc"]),
        monitor_metric,
        float(epoch_history["monitor_value"]),
        _format_optional_epoch_value(epoch_history.get("val_topology_loss")),
        "none" if selected_rule is None else selected_rule.type,
        float(epoch_history["learning_rate"]),
        float(epoch_history["peak_gpu_mem_mb"]),
    )


def _format_optional_epoch_value(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "n/a"


def _write_training_summary(
    *,
    cfg: S2GAEConfig,
    best_monitor_value: float,
    best_validation_auprc: float,
    best_selected_rule_payload: dict[str, object] | None,
    optimizer: Optimizer,
    history: Sequence[Mapping[str, float | int]],
) -> None:
    write_json(
        cfg.log_dir / "training_summary.json",
        {
            "config": _config_to_json(cfg),
            "monitor_metric": cfg.monitor_metric,
            "best_monitor_value": best_monitor_value,
            "best_validation_auprc": best_validation_auprc,
            "selected_rule": best_selected_rule_payload,
            "epochs_trained": len(history),
            "checkpoint_path": str(cfg.checkpoint_path),
            "optimizer": _optimizer_config_to_json(cfg.optimizer),
            "scheduler": {"type": cfg.scheduler.scheduler_type},
            "optimization": _optimization_config_to_json(cfg.optimization),
            "current_learning_rate": _current_learning_rate(optimizer),
            "history": list(history),
        },
    )


def _required_float_tensor(
    values: Sequence[int] | None,
    field_name: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    if values is None:
        raise ValueError(f"{field_name} is required for S2GAE training")
    return torch.tensor([float(value) for value in values], dtype=torch.float32, device=device)


def _prepared_model_and_optimizer(prepared: object) -> tuple[nn.Module, Optimizer]:
    if not isinstance(prepared, tuple) or len(prepared) != 2:
        raise TypeError("accelerator.prepare(model, optimizer) must return a two-item tuple")
    model, optimizer = prepared
    if not isinstance(model, nn.Module):
        raise TypeError("accelerator.prepare returned a non-module model")
    if not all(hasattr(optimizer, name) for name in ("zero_grad", "step", "param_groups")):
        raise TypeError("accelerator.prepare returned a non-optimizer optimizer")
    return model, cast(Optimizer, optimizer)


def _unwrap_model(model: nn.Module, accelerator: object) -> nn.Module:
    module = getattr(model, "module", None)
    if isinstance(module, nn.Module):
        return module
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    if callable(unwrap_model):
        unwrapped = unwrap_model(model)
        if isinstance(unwrapped, nn.Module):
            return unwrapped
    return model


def _unwrap_refiner(model: nn.Module, accelerator: object) -> S2GAERefiner:
    if isinstance(model, S2GAERefiner):
        return model
    if isinstance(model, _S2GAESampledTrainStepModule):
        return model.refiner
    module = getattr(model, "module", None)
    if isinstance(module, S2GAERefiner):
        return module
    if isinstance(module, _S2GAESampledTrainStepModule):
        return module.refiner
    unwrapped = _unwrap_model(model, accelerator)
    if isinstance(unwrapped, S2GAERefiner):
        return unwrapped
    refiner = getattr(unwrapped, "refiner", None)
    if isinstance(refiner, S2GAERefiner):
        return refiner
    raise TypeError("Prepared S2GAE model does not contain an S2GAERefiner")


def _parse_config(config: Mapping[str, object]) -> S2GAEConfig:
    if "learning_rate" in config:
        raise ValueError("refiner.learning_rate is no longer supported; use refiner.optimizer.lr")
    topology_loss_raw = config.get("topology_loss")
    if isinstance(topology_loss_raw, Mapping) and bool(topology_loss_raw.get("enabled", False)):
        raise ValueError(
            "refiner.topology_loss is no longer supported; remove the block "
            "(use refiner.topology_validation for checkpoint selection)"
        )
    run_id = str(config.get("_run_id", "tccig_run"))
    log_root = Path(str(config.get("_log_root", "logs")))
    log_dir = _path(config.get("log_dir"), log_root / "tccig" / run_id)
    checkpoint_path = _path(
        config.get("checkpoint_path"),
        Path("models") / "tccig" / "s2gae" / run_id / "best_model.pt",
    )
    embedding_cache_dir = _required_path(config, "embedding_cache_dir")
    embedding_index_path = _path(
        config.get("embedding_index_path"),
        embedding_cache_dir / "index.json",
    )
    max_sequence_length_raw = config.get("max_sequence_length")
    max_sequence_length = (
        None
        if max_sequence_length_raw is None
        else _positive_int(max_sequence_length_raw, "refiner.max_sequence_length")
    )
    monitor_metric = str(config.get("monitor_metric", "val_auprc"))
    if monitor_metric not in SUPPORTED_MONITOR_METRICS:
        raise ValueError(
            f"refiner.monitor_metric must be one of {sorted(SUPPORTED_MONITOR_METRICS)}"
        )
    encoder = str(config.get("encoder", "graphconv")).lower()
    if encoder != "graphconv":
        raise ValueError("refiner.encoder must be 'graphconv'")
    encoder_aggr = str(config.get("encoder_aggr", "mean")).lower()
    if encoder_aggr not in SUPPORTED_ENCODER_AGGREGATIONS:
        raise ValueError(
            f"refiner.encoder_aggr must be one of {sorted(SUPPORTED_ENCODER_AGGREGATIONS)}"
        )
    return S2GAEConfig(
        encoder=encoder,
        input_dim=_positive_int(config.get("input_dim", 1024), "refiner.input_dim"),
        hidden_dim=_positive_int(config.get("hidden_dim", 128), "refiner.hidden_dim"),
        num_layers=_positive_int(config.get("num_layers", 2), "refiner.num_layers"),
        decoder_hidden_dim=_positive_int(
            config.get("decoder_hidden_dim", 256),
            "refiner.decoder_hidden_dim",
        ),
        decoder_layers=_positive_int(
            config.get("decoder_layers", 2),
            "refiner.decoder_layers",
        ),
        dropout=_probability(config.get("dropout", 0.5), "refiner.dropout"),
        epochs=_positive_int(config.get("epochs", 20), "refiner.epochs"),
        batch_size=_positive_int(config.get("batch_size", 4096), "refiner.batch_size"),
        edge_sampling=parse_edge_sampling_config(config.get("edge_sampling", {})),
        loss_config=_parse_loss_config(config.get("loss", {})),
        residual_weight=_non_negative_float(
            config.get("residual_weight", 0.001),
            "refiner.residual_weight",
        ),
        residual_scale=_optional_positive_float(
            config.get("residual_scale", 4.0),
            "refiner.residual_scale",
        ),
        encoder_aggr=encoder_aggr,
        layer_norm=_bool(config.get("layer_norm", True), "refiner.layer_norm"),
        monitor_metric=monitor_metric,
        topology_validation=_parse_topology_validation_config(
            raw_topology_validation=config.get("topology_validation", {}),
            monitor_metric=monitor_metric,
            default_inference_batch_size=config.get("batch_size", 4096),
        ),
        optimizer=_parse_optimizer_config(config.get("optimizer")),
        scheduler=_parse_scheduler_config(config.get("scheduler")),
        optimization=_parse_optimization_config(config.get("optimization")),
        embedding_cache_dir=embedding_cache_dir,
        embedding_index_path=embedding_index_path,
        max_sequence_length=max_sequence_length,
        checkpoint_path=checkpoint_path,
        log_dir=log_dir,
        residual_anchor=_parse_residual_anchor_config(
            config.get("residual_anchor"),
            fallback_weight=_non_negative_float(
                config.get("residual_weight", 0.001), "refiner.residual_weight"
            ),
        ),
        topology_training=_parse_topology_training_config(config.get("topology_training")),
    )


def _config_to_json(cfg: S2GAEConfig) -> dict[str, object]:
    return {
        "encoder": cfg.encoder,
        "input_dim": cfg.input_dim,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "decoder_hidden_dim": cfg.decoder_hidden_dim,
        "decoder_layers": cfg.decoder_layers,
        "dropout": cfg.dropout,
        "encoder_aggr": cfg.encoder_aggr,
        "layer_norm": cfg.layer_norm,
        "residual_scale": cfg.residual_scale,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "edge_sampling": {
            "hard_fraction": cfg.edge_sampling.hard_fraction,
            "easy_anchor_fraction": cfg.edge_sampling.easy_anchor_fraction,
            "seed": cfg.edge_sampling.seed,
            "reshuffle_easy_each_epoch": cfg.edge_sampling.reshuffle_easy_each_epoch,
        },
        "loss": {
            "type": cfg.loss_config.loss_type,
            "pos_weight": cfg.loss_config.pos_weight,
            "label_smoothing": cfg.loss_config.label_smoothing,
        },
        "residual_weight": cfg.residual_weight,
        "monitor_metric": cfg.monitor_metric,
        "topology_validation": {
            "enabled": cfg.topology_validation.enabled,
            "inference_batch_size": cfg.topology_validation.inference_batch_size,
            "compute_clustering_mmd": cfg.topology_validation.compute_clustering_mmd,
            "losses": {
                "alpha": cfg.topology_validation.losses.alpha,
                "beta": cfg.topology_validation.losses.beta,
                "gamma": cfg.topology_validation.losses.gamma,
                "delta": cfg.topology_validation.losses.delta,
            },
        },
        "optimizer": _optimizer_config_to_json(cfg.optimizer),
        "scheduler": {"type": cfg.scheduler.scheduler_type},
        "optimization": _optimization_config_to_json(cfg.optimization),
        "embedding_cache_dir": str(cfg.embedding_cache_dir),
        "embedding_index_path": str(cfg.embedding_index_path),
        "max_sequence_length": cfg.max_sequence_length,
        "checkpoint_path": str(cfg.checkpoint_path),
        "log_dir": str(cfg.log_dir),
        "residual_anchor": {
            "form": cfg.residual_anchor.form,
            "weight": cfg.residual_anchor.weight,
        },
        "topology_training": {
            "enabled": cfg.topology_training.enabled,
            "node_sizes": list(cfg.topology_training.node_sizes),
            "samples_per_size": cfg.topology_training.samples_per_size,
            "strategy": cfg.topology_training.strategy,
            "seed": cfg.topology_training.seed,
            "coverage_augmentation": cfg.topology_training.coverage_augmentation,
            "topology_weight": cfg.topology_training.topology_weight,
            "weights": {
                "alpha": cfg.topology_training.weights.alpha,
                "beta": cfg.topology_training.weights.beta,
                "gamma": cfg.topology_training.weights.gamma,
                "delta": cfg.topology_training.weights.delta,
            },
            "schedule": {
                "warmup_epochs": cfg.topology_training.warmup_epochs,
                "ramp_epochs": cfg.topology_training.ramp_epochs,
                "schedule": cfg.topology_training.schedule,
            },
            "subset": {
                "enabled": cfg.topology_training.subset.enabled,
                "candidate_ratio": cfg.topology_training.subset.candidate_ratio,
                "pool_ratio": cfg.topology_training.subset.pool_ratio,
                "epoch_ratio": cfg.topology_training.subset.epoch_ratio,
                "hard_fraction": cfg.topology_training.subset.hard_fraction,
                "uniform_fraction": cfg.topology_training.subset.uniform_fraction,
                "hard_stratum_fraction": cfg.topology_training.subset.hard_stratum_fraction,
                "seed": cfg.topology_training.subset.seed,
                "max_subgraphs_per_size": (
                    cfg.topology_training.subset.max_subgraphs_per_size
                ),
                "max_labeled_pairs_per_size": (
                    cfg.topology_training.subset.max_labeled_pairs_per_size
                ),
                "bias_diagnostic_every_n_epochs": (
                    cfg.topology_training.subset.bias_diagnostic_every_n_epochs
                ),
                "bias_diagnostic_max_node_size": (
                    cfg.topology_training.subset.bias_diagnostic_max_node_size
                ),
                "bias_diagnostic_max_subgraphs": (
                    cfg.topology_training.subset.bias_diagnostic_max_subgraphs
                ),
            },
        },
    }


def _load_embedding_index(index_path: Path) -> dict[str, str]:
    if not index_path.exists():
        raise FileNotFoundError(f"refiner.embedding_index_path does not exist: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("refiner.embedding_index_path must contain a JSON object")
    index: dict[str, str] = {}
    for protein_id, relative_path in payload.items():
        if not isinstance(protein_id, str) or not isinstance(relative_path, str):
            raise ValueError("refiner.embedding_index_path must map protein IDs to paths")
        index[protein_id] = relative_path
    return index


def _load_graph_conv() -> type[nn.Module]:
    try:
        from torch_geometric.nn import GraphConv
    except ImportError as error:
        raise ImportError(
            "S2GAE refiner requires PyG GraphConv. Run `uv sync --group dev --find-links "
            "https://data.pyg.org/whl/torch-2.10.0+cpu.html` locally, or use the "
            "matching CUDA wheel page on HPC."
        ) from error
    return cast(type[nn.Module], GraphConv)


def _required_path(config: Mapping[str, object], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"refiner.{key} is required")
    return Path(value)


def _path(value: object, default: Path) -> Path:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Path config values must be non-empty strings")
    return Path(value)


def _parse_loss_config(raw_loss: object) -> LossConfig:
    if not isinstance(raw_loss, Mapping):
        raise ValueError("refiner.loss must be a mapping")
    return LossConfig(
        loss_type=str(raw_loss.get("type", "bce_with_logits")),
        pos_weight=_positive_float(raw_loss.get("pos_weight", 1.0), "refiner.loss.pos_weight"),
        label_smoothing=_probability(
            raw_loss.get("label_smoothing", 0.0),
            "refiner.loss.label_smoothing",
        ),
    )


def _parse_topology_validation_config(
    *,
    raw_topology_validation: object,
    monitor_metric: str,
    default_inference_batch_size: object,
) -> S2GAETopologyValidationConfig:
    if not isinstance(raw_topology_validation, Mapping):
        raise ValueError("refiner.topology_validation must be a mapping")
    enabled = (
        _bool(raw_topology_validation.get("enabled"), "refiner.topology_validation.enabled")
        if "enabled" in raw_topology_validation
        else monitor_metric != "val_auprc"
    )
    raw_losses = raw_topology_validation.get("losses", {})
    if not isinstance(raw_losses, Mapping):
        raise ValueError("refiner.topology_validation.losses must be a mapping")
    return S2GAETopologyValidationConfig(
        enabled=enabled,
        inference_batch_size=_positive_int(
            raw_topology_validation.get("inference_batch_size", default_inference_batch_size),
            "refiner.topology_validation.inference_batch_size",
        ),
        compute_clustering_mmd=_bool(
            raw_topology_validation.get("compute_clustering_mmd", True),
            "refiner.topology_validation.compute_clustering_mmd",
        ),
        losses=S2GAETopologyLossWeights(
            alpha=_non_negative_float(
                raw_losses.get("alpha", 0.5),
                "refiner.topology_validation.losses.alpha",
            ),
            beta=_non_negative_float(
                raw_losses.get("beta", 1.0),
                "refiner.topology_validation.losses.beta",
            ),
            gamma=_non_negative_float(
                raw_losses.get("gamma", 0.3),
                "refiner.topology_validation.losses.gamma",
            ),
            delta=_non_negative_float(
                raw_losses.get("delta", 0.3),
                "refiner.topology_validation.losses.delta",
            ),
        ),
    )


def _parse_residual_anchor_config(
    raw: object, *, fallback_weight: float
) -> S2GAEResidualAnchorConfig:
    if raw is None:
        return S2GAEResidualAnchorConfig(form="symmetric", weight=fallback_weight)
    if not isinstance(raw, Mapping):
        raise ValueError("refiner.residual_anchor must be a mapping")
    form = str(raw.get("form", "symmetric"))
    if form not in {"symmetric", "asymmetric_relu"}:
        raise ValueError("refiner.residual_anchor.form must be 'symmetric' or 'asymmetric_relu'")
    return S2GAEResidualAnchorConfig(
        form=form,
        weight=_non_negative_float(raw.get("weight", 0.0), "refiner.residual_anchor.weight"),
    )


def _parse_topology_subset_config(raw: object) -> TopologySubsetSamplerConfig:
    """Parse refiner.topology_training.subset."""
    if raw is None:
        return TopologySubsetSamplerConfig(enabled=False)
    if not isinstance(raw, Mapping):
        raise ValueError("refiner.topology_training.subset must be a mapping")
    cfg = TopologySubsetSamplerConfig(
        enabled=_bool(raw.get("enabled", True), "refiner.topology_training.subset.enabled"),
        candidate_ratio=_positive_int(
            raw.get("candidate_ratio", 20), "refiner.topology_training.subset.candidate_ratio"
        ),
        pool_ratio=_positive_int(
            raw.get("pool_ratio", 10), "refiner.topology_training.subset.pool_ratio"
        ),
        epoch_ratio=_positive_int(
            raw.get("epoch_ratio", 5), "refiner.topology_training.subset.epoch_ratio"
        ),
        hard_fraction=_non_negative_float(
            raw.get("hard_fraction", 0.5), "refiner.topology_training.subset.hard_fraction"
        ),
        uniform_fraction=_non_negative_float(
            raw.get("uniform_fraction", 0.5),
            "refiner.topology_training.subset.uniform_fraction",
        ),
        hard_stratum_fraction=_non_negative_float(
            raw.get("hard_stratum_fraction", 0.2),
            "refiner.topology_training.subset.hard_stratum_fraction",
        ),
        seed=_non_negative_int(raw.get("seed", 0), "refiner.topology_training.subset.seed"),
        max_subgraphs_per_size=_non_negative_int(
            raw.get("max_subgraphs_per_size", 0),
            "refiner.topology_training.subset.max_subgraphs_per_size",
        ),
        max_labeled_pairs_per_size=_non_negative_int(
            raw.get("max_labeled_pairs_per_size", 0),
            "refiner.topology_training.subset.max_labeled_pairs_per_size",
        ),
        bias_diagnostic_every_n_epochs=_non_negative_int(
            raw.get("bias_diagnostic_every_n_epochs", 0),
            "refiner.topology_training.subset.bias_diagnostic_every_n_epochs",
        ),
        bias_diagnostic_max_node_size=_non_negative_int(
            raw.get("bias_diagnostic_max_node_size", 40),
            "refiner.topology_training.subset.bias_diagnostic_max_node_size",
        ),
        bias_diagnostic_max_subgraphs=_non_negative_int(
            raw.get("bias_diagnostic_max_subgraphs", 4),
            "refiner.topology_training.subset.bias_diagnostic_max_subgraphs",
        ),
    )
    cfg.validate()
    return cfg


def _parse_topology_training_config(raw: object) -> S2GAETopologyTrainingConfig:
    if raw is None or not isinstance(raw, Mapping):
        return S2GAETopologyTrainingConfig(
            enabled=False,
            node_sizes=TOPOLOGY_EVAL_NODE_SIZES,
            samples_per_size=20,
            strategy="mixed",
            seed=0,
            coverage_augmentation=True,
            topology_weight=0.0,
            weights=TopologyLossWeights(),
            warmup_epochs=0,
            ramp_epochs=0,
            schedule="linear",
            subset=TopologySubsetSamplerConfig(enabled=False),
        )
    raw_weights = raw.get("weights", {})
    if not isinstance(raw_weights, Mapping):
        raise ValueError("refiner.topology_training.weights must be a mapping")
    raw_schedule = raw.get("schedule", {})
    if not isinstance(raw_schedule, Mapping):
        raise ValueError("refiner.topology_training.schedule must be a mapping")
    return S2GAETopologyTrainingConfig(
        enabled=_bool(raw.get("enabled", False), "refiner.topology_training.enabled"),
        node_sizes=tuple(
            _int_sequence(
                raw.get("node_sizes", list(TOPOLOGY_EVAL_NODE_SIZES)),
                "refiner.topology_training.node_sizes",
            )
        ),
        samples_per_size=_positive_int(
            raw.get("samples_per_size", 20), "refiner.topology_training.samples_per_size"
        ),
        strategy=str(raw.get("strategy", "mixed")),
        seed=_non_negative_int(raw.get("seed", 0), "refiner.topology_training.seed"),
        coverage_augmentation=_bool(
            raw.get("coverage_augmentation", True),
            "refiner.topology_training.coverage_augmentation",
        ),
        topology_weight=_non_negative_float(
            raw.get("topology_weight", 1.0), "refiner.topology_training.topology_weight"
        ),
        weights=TopologyLossWeights(
            alpha=_non_negative_float(raw_weights.get("alpha", 1.0), "...alpha"),
            beta=_non_negative_float(raw_weights.get("beta", 8.0), "...beta"),
            gamma=_non_negative_float(raw_weights.get("gamma", 0.5), "...gamma"),
            delta=_non_negative_float(raw_weights.get("delta", 0.1), "...delta"),
        ),
        warmup_epochs=_non_negative_int(raw_schedule.get("warmup_epochs", 0), "...warmup_epochs"),
        ramp_epochs=_non_negative_int(raw_schedule.get("ramp_epochs", 0), "...ramp_epochs"),
        schedule=str(raw_schedule.get("schedule", "linear")),
        subset=_parse_topology_subset_config(raw.get("subset")),
    )


def _parse_optimizer_config(raw_optimizer: object) -> S2GAEOptimizerConfig:
    if not isinstance(raw_optimizer, Mapping):
        raise ValueError("refiner.optimizer must be a mapping")
    optimizer_type = str(raw_optimizer.get("type", "")).lower()
    if optimizer_type != "adamw":
        raise ValueError("refiner.optimizer.type must be 'adamw'")
    return S2GAEOptimizerConfig(
        optimizer_type=optimizer_type,
        lr=_positive_float(raw_optimizer.get("lr"), "refiner.optimizer.lr"),
        weight_decay=_non_negative_float(
            raw_optimizer.get("weight_decay", 0.0),
            "refiner.optimizer.weight_decay",
        ),
        beta1=_exclusive_probability(
            raw_optimizer.get("beta1", 0.9),
            "refiner.optimizer.beta1",
        ),
        beta2=_exclusive_probability(
            raw_optimizer.get("beta2", 0.999),
            "refiner.optimizer.beta2",
        ),
        eps=_positive_float(raw_optimizer.get("eps", 1e-8), "refiner.optimizer.eps"),
    )


def _parse_scheduler_config(raw_scheduler: object) -> S2GAESchedulerConfig:
    if not isinstance(raw_scheduler, Mapping):
        raise ValueError("refiner.scheduler must be a mapping")
    scheduler_type = str(raw_scheduler.get("type", "")).lower()
    if scheduler_type != "none":
        raise ValueError("refiner.scheduler.type must be 'none'")
    return S2GAESchedulerConfig(scheduler_type=scheduler_type)


def _parse_optimization_config(raw_optimization: object) -> S2GAEOptimizationConfig:
    if not isinstance(raw_optimization, Mapping):
        raise ValueError("refiner.optimization must be a mapping")
    return S2GAEOptimizationConfig(
        gradient_clip_norm=_optional_positive_float(
            raw_optimization.get("gradient_clip_norm", 1.0),
            "refiner.optimization.gradient_clip_norm",
        )
    )


def _optimizer_config_to_json(config: S2GAEOptimizerConfig) -> dict[str, object]:
    return {
        "type": config.optimizer_type,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "eps": config.eps,
    }


def _optimization_config_to_json(config: S2GAEOptimizationConfig) -> dict[str, object]:
    return {"gradient_clip_norm": config.gradient_clip_norm}


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(cast(int | str, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _positive_float(value: object, field_name: str) -> float:
    try:
        parsed = float(cast(float | str, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a positive float") from error
    if parsed <= 0.0 or math.isnan(parsed) or math.isinf(parsed):
        raise ValueError(f"{field_name} must be a positive float")
    return parsed


def _optional_positive_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    parsed = _non_negative_float(value, field_name)
    return None if parsed == 0.0 else parsed


def _non_negative_float(value: object, field_name: str) -> float:
    try:
        parsed = float(cast(float | str, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a non-negative float") from error
    if parsed < 0.0 or math.isnan(parsed) or math.isinf(parsed):
        raise ValueError(f"{field_name} must be a non-negative float")
    return parsed


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        parsed = int(cast(int | str, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a non-negative integer") from error
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _int_sequence(value: object, field_name: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence of integers")
    result: list[int] = []
    for i, item in enumerate(value):
        result.append(_non_negative_int(item, f"{field_name}[{i}]"))
    return result


def _probability(value: object, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name)
    if parsed >= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1)")
    return parsed


def _exclusive_probability(value: object, field_name: str) -> float:
    parsed = _positive_float(value, field_name)
    if parsed >= 1.0:
        raise ValueError(f"{field_name} must be in (0, 1)")
    return parsed


def _bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean")


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    norms = [grad.detach().norm(2).to(device) for grad in grads]
    return float(torch.norm(torch.stack(norms), 2).detach().cpu().item())


def _current_learning_rate(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])
