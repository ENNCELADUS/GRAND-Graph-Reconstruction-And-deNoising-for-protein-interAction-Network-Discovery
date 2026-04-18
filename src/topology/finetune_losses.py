"""Differentiable graph-topology losses for PRING-style fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi

import torch

EPSILON = 1.0e-8
DEFAULT_DEGREE_BINS = 64
DEFAULT_CLUSTERING_BINS = 100


@dataclass(frozen=True)
class TopologyLossWeights:
    """Graph-topology loss weights and kernel settings."""

    alpha: float = 0.5
    beta: float = 1.0
    gamma: float = 0.3
    delta: float = 0.3
    histogram_sigma: float = 1.0
    degree_bins: int = DEFAULT_DEGREE_BINS
    clustering_bins: int = DEFAULT_CLUSTERING_BINS


@dataclass(frozen=True)
class TopologyLossWeightSchedule:
    """Epoch-wise schedule for scaling topology loss weights."""

    warmup_epochs: int = 0
    ramp_epochs: int = 0
    schedule: str = "linear"


def topology_loss_scale(*, epoch: int, schedule: TopologyLossWeightSchedule) -> float:
    """Return the topology-loss scale for a zero-based epoch index."""
    if epoch < 0:
        raise ValueError("epoch must be >= 0")
    if schedule.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0")
    if schedule.ramp_epochs < 0:
        raise ValueError("ramp_epochs must be >= 0")
    schedule_name = schedule.schedule.lower()
    if schedule_name not in {"linear", "cosine"}:
        raise ValueError("schedule must be 'linear' or 'cosine'")

    if epoch < schedule.warmup_epochs:
        return 0.0

    ramp_epoch = epoch - schedule.warmup_epochs
    if schedule.ramp_epochs == 0 or ramp_epoch >= schedule.ramp_epochs:
        return 1.0

    progress = ramp_epoch / float(schedule.ramp_epochs)
    if schedule_name == "cosine":
        return 0.5 * (1.0 - cos(pi * progress))
    return progress


def build_symmetric_adjacency(
    *,
    num_nodes: int,
    pair_index_a: torch.Tensor,
    pair_index_b: torch.Tensor,
    pair_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Scatter upper-triangle pair probabilities into a symmetric adjacency matrix."""
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")
    adjacency = pair_probabilities.new_zeros((num_nodes, num_nodes))
    adjacency[pair_index_a, pair_index_b] = pair_probabilities
    adjacency[pair_index_b, pair_index_a] = pair_probabilities
    adjacency.fill_diagonal_(0.0)
    return adjacency


def soft_graph_similarity_loss(
    *,
    pred_adjacency: torch.Tensor,
    target_adjacency: torch.Tensor,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Differentiable surrogate of official PRING GS loss, i.e. ``1 - GS``."""
    difference = torch.abs(pred_adjacency - target_adjacency).sum()
    denominator = pred_adjacency.sum() + target_adjacency.sum()
    return torch.where(
        denominator > eps,
        difference / (denominator + eps),
        torch.zeros_like(difference),
    )


def _pairwise_graph_similarity_loss(
    *,
    pred_pair_probabilities: torch.Tensor,
    target_pair_probabilities: torch.Tensor,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Differentiable graph similarity loss over upper-triangle pair vectors."""
    difference = torch.abs(pred_pair_probabilities - target_pair_probabilities).sum()
    denominator = pred_pair_probabilities.sum() + target_pair_probabilities.sum()
    return torch.where(
        denominator > eps,
        difference / (denominator + eps),
        torch.zeros_like(difference),
    )


def soft_relative_density_loss(
    *,
    pred_adjacency: torch.Tensor,
    target_adjacency: torch.Tensor,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Squared deviation of differentiable relative density from 1."""
    num_nodes = pred_adjacency.size(0)
    if num_nodes < 2:
        raise ValueError("Adjacency matrices must have at least 2 nodes")
    normalizer = float(num_nodes * (num_nodes - 1))
    pred_density = pred_adjacency.sum() / normalizer
    target_density = target_adjacency.sum() / normalizer
    if float(target_density.detach().item()) <= eps:
        return pred_density.square()
    relative_density = pred_density / (target_density + eps)
    return (relative_density - 1.0).square()


def _pairwise_relative_density_loss(
    *,
    num_nodes: int,
    pred_pair_probabilities: torch.Tensor,
    target_pair_probabilities: torch.Tensor,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Squared deviation of relative density computed from pair vectors."""
    if num_nodes < 2:
        raise ValueError("num_nodes must be >= 2")
    normalizer = float(num_nodes * (num_nodes - 1))
    pred_density = (2.0 * pred_pair_probabilities.sum()) / normalizer
    target_density = (2.0 * target_pair_probabilities.sum()) / normalizer
    if float(target_density.detach().item()) <= eps:
        return pred_density.square()
    relative_density = pred_density / (target_density + eps)
    return (relative_density - 1.0).square()


def _soft_histogram(
    values: torch.Tensor,
    *,
    centers: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Build a normalized soft histogram with Gaussian bin assignment."""
    if sigma <= 0.0:
        raise ValueError("histogram sigma must be positive")
    scaled_distance = (values.unsqueeze(-1) - centers.unsqueeze(0)) / sigma
    histogram = torch.exp(-0.5 * scaled_distance.square()).sum(dim=0)
    return histogram / (histogram.sum() + EPSILON)


def _soft_histogram_mmd(
    pred_histogram: torch.Tensor,
    target_histogram: torch.Tensor,
    *,
    sigma: float,
) -> torch.Tensor:
    """One-sample Gaussian-TV MMD matching ``src.topology.metrics.compute_mmd``."""
    total_variation = 0.5 * torch.abs(pred_histogram - target_histogram).sum()
    kernel_value = torch.exp(-(total_variation.square()) / (2.0 * sigma * sigma))
    return 2.0 - 2.0 * kernel_value


def _soft_degrees(adjacency: torch.Tensor) -> torch.Tensor:
    """Return soft node degrees from a weighted adjacency matrix."""
    return adjacency.sum(dim=1)


def _pairwise_soft_degrees(
    *,
    num_nodes: int,
    pair_index_a: torch.Tensor,
    pair_index_b: torch.Tensor,
    pair_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return soft node degrees from upper-triangle pair probabilities."""
    degrees = pair_probabilities.new_zeros((num_nodes,))
    degrees.scatter_add_(0, pair_index_a, pair_probabilities)
    degrees.scatter_add_(0, pair_index_b, pair_probabilities)
    return degrees


def _soft_clustering_coefficients(
    adjacency: torch.Tensor,
    *,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Return differentiable local clustering coefficients for weighted graphs."""
    degrees = _soft_degrees(adjacency)
    triangle_mass = ((adjacency @ adjacency) * adjacency).sum(dim=1)
    denominator = degrees * (degrees - 1.0)
    clustering = torch.where(
        denominator > eps,
        triangle_mass / (denominator + eps),
        torch.zeros_like(triangle_mass),
    )
    return torch.clamp(clustering, min=0.0, max=1.0)


def _degree_distribution_mmd(
    *,
    pred_adjacency: torch.Tensor,
    target_adjacency: torch.Tensor,
    weights: TopologyLossWeights,
) -> torch.Tensor:
    """MMD between soft degree distributions."""
    num_nodes = pred_adjacency.size(0)
    centers = torch.linspace(
        0.0,
        float(max(1, num_nodes - 1)),
        steps=max(2, weights.degree_bins),
        device=pred_adjacency.device,
        dtype=pred_adjacency.dtype,
    )
    pred_histogram = _soft_histogram(
        _soft_degrees(pred_adjacency),
        centers=centers,
        sigma=weights.histogram_sigma,
    )
    target_histogram = _soft_histogram(
        _soft_degrees(target_adjacency),
        centers=centers,
        sigma=weights.histogram_sigma,
    )
    return _soft_histogram_mmd(
        pred_histogram=pred_histogram,
        target_histogram=target_histogram,
        sigma=weights.histogram_sigma,
    )


def _degree_distribution_mmd_from_pairs(
    *,
    num_nodes: int,
    pair_index_a: torch.Tensor,
    pair_index_b: torch.Tensor,
    pred_pair_probabilities: torch.Tensor,
    target_pair_probabilities: torch.Tensor,
    weights: TopologyLossWeights,
) -> torch.Tensor:
    """MMD between soft degree distributions built from pair vectors."""
    centers = torch.linspace(
        0.0,
        float(max(1, num_nodes - 1)),
        steps=max(2, weights.degree_bins),
        device=pred_pair_probabilities.device,
        dtype=pred_pair_probabilities.dtype,
    )
    pred_histogram = _soft_histogram(
        _pairwise_soft_degrees(
            num_nodes=num_nodes,
            pair_index_a=pair_index_a,
            pair_index_b=pair_index_b,
            pair_probabilities=pred_pair_probabilities,
        ),
        centers=centers,
        sigma=weights.histogram_sigma,
    )
    target_histogram = _soft_histogram(
        _pairwise_soft_degrees(
            num_nodes=num_nodes,
            pair_index_a=pair_index_a,
            pair_index_b=pair_index_b,
            pair_probabilities=target_pair_probabilities,
        ),
        centers=centers,
        sigma=weights.histogram_sigma,
    )
    return _soft_histogram_mmd(
        pred_histogram=pred_histogram,
        target_histogram=target_histogram,
        sigma=weights.histogram_sigma,
    )


def _clustering_distribution_mmd(
    *,
    pred_adjacency: torch.Tensor,
    target_adjacency: torch.Tensor,
    weights: TopologyLossWeights,
) -> torch.Tensor:
    """MMD between soft clustering-coefficient distributions."""
    centers = torch.linspace(
        0.0,
        1.0,
        steps=max(2, weights.clustering_bins),
        device=pred_adjacency.device,
        dtype=pred_adjacency.dtype,
    )
    pred_histogram = _soft_histogram(
        _soft_clustering_coefficients(pred_adjacency),
        centers=centers,
        sigma=max(weights.histogram_sigma / max(1, weights.clustering_bins), EPSILON),
    )
    target_histogram = _soft_histogram(
        _soft_clustering_coefficients(target_adjacency),
        centers=centers,
        sigma=max(weights.histogram_sigma / max(1, weights.clustering_bins), EPSILON),
    )
    return _soft_histogram_mmd(
        pred_histogram=pred_histogram,
        target_histogram=target_histogram,
        sigma=weights.histogram_sigma,
    )


def compute_topology_losses(
    *,
    pred_adjacency: torch.Tensor | None = None,
    target_adjacency: torch.Tensor | None = None,
    weights: TopologyLossWeights,
    num_nodes: int | None = None,
    pair_index_a: torch.Tensor | None = None,
    pair_index_b: torch.Tensor | None = None,
    pred_pair_probabilities: torch.Tensor | None = None,
    target_pair_probabilities: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return weighted graph-topology loss terms and their total."""
    use_pairwise_path = all(
        value is not None
        for value in (
            num_nodes,
            pair_index_a,
            pair_index_b,
            pred_pair_probabilities,
            target_pair_probabilities,
        )
    )
    if use_pairwise_path:
        assert num_nodes is not None
        assert pair_index_a is not None
        assert pair_index_b is not None
        assert pred_pair_probabilities is not None
        assert target_pair_probabilities is not None
        graph_similarity = _pairwise_graph_similarity_loss(
            pred_pair_probabilities=pred_pair_probabilities,
            target_pair_probabilities=target_pair_probabilities,
        )
        relative_density = _pairwise_relative_density_loss(
            num_nodes=num_nodes,
            pred_pair_probabilities=pred_pair_probabilities,
            target_pair_probabilities=target_pair_probabilities,
        )
        degree_mmd = _degree_distribution_mmd_from_pairs(
            num_nodes=num_nodes,
            pair_index_a=pair_index_a,
            pair_index_b=pair_index_b,
            pred_pair_probabilities=pred_pair_probabilities,
            target_pair_probabilities=target_pair_probabilities,
            weights=weights,
        )
    else:
        if pred_adjacency is None or target_adjacency is None:
            raise ValueError(
                "compute_topology_losses requires either adjacency matrices or "
                "the pairwise topology inputs"
            )
        graph_similarity = soft_graph_similarity_loss(
            pred_adjacency=pred_adjacency,
            target_adjacency=target_adjacency,
        )
        relative_density = soft_relative_density_loss(
            pred_adjacency=pred_adjacency,
            target_adjacency=target_adjacency,
        )
        degree_mmd = _degree_distribution_mmd(
            pred_adjacency=pred_adjacency,
            target_adjacency=target_adjacency,
            weights=weights,
        )

    if pred_adjacency is None:
        assert num_nodes is not None
        assert pair_index_a is not None
        assert pair_index_b is not None
        assert pred_pair_probabilities is not None
        pred_adjacency = build_symmetric_adjacency(
            num_nodes=num_nodes,
            pair_index_a=pair_index_a,
            pair_index_b=pair_index_b,
            pair_probabilities=pred_pair_probabilities,
        )
    if target_adjacency is None:
        assert num_nodes is not None
        assert pair_index_a is not None
        assert pair_index_b is not None
        assert target_pair_probabilities is not None
        target_adjacency = build_symmetric_adjacency(
            num_nodes=num_nodes,
            pair_index_a=pair_index_a,
            pair_index_b=pair_index_b,
            pair_probabilities=target_pair_probabilities,
        )
    clustering_mmd = _clustering_distribution_mmd(
        pred_adjacency=pred_adjacency,
        target_adjacency=target_adjacency,
        weights=weights,
    )
    total_topology = (
        weights.alpha * graph_similarity
        + weights.beta * relative_density
        + weights.gamma * degree_mmd
        + weights.delta * clustering_mmd
    )
    return {
        "graph_similarity": graph_similarity,
        "relative_density": relative_density,
        "degree_mmd": degree_mmd,
        "clustering_mmd": clustering_mmd,
        "total_topology": total_topology,
    }
