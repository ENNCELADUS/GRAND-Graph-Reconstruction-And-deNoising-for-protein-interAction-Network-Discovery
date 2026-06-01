"""Retrieval, reconstruction-universe, and graph-assembly helpers for TCCIG."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional


def canonical_pair(node_a: str, node_b: str) -> tuple[str, str]:
    """Return a stable undirected protein-pair key."""
    return (node_a, node_b) if node_a <= node_b else (node_b, node_a)


@dataclass(frozen=True)
class TCCIGReconstructionUniverse:
    """Full candidate universe for PRING-style graph reconstruction."""

    records: tuple[tuple[str, str], ...]
    labels: torch.Tensor

    @property
    def positive_count(self) -> int:
        """Return the number of positive records."""
        return int(self.labels.sum().item())

    @property
    def positive_density(self) -> float:
        """Return positive edge density over the candidate universe."""
        if self.labels.numel() == 0:
            return 0.0
        return float(self.positive_count / float(self.labels.numel()))

    def write_records(self, path: Path) -> None:
        """Write the universe in PRING pair-file format."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            for (protein_a, protein_b), label in zip(
                self.records,
                self.labels.detach().cpu().tolist(),
                strict=True,
            ):
                writer.writerow([protein_a, protein_b, int(label)])


def build_full_reconstruction_universe(
    *,
    protein_ids: Sequence[str],
    positive_edges: Iterable[tuple[str, str]],
) -> TCCIGReconstructionUniverse:
    """Build all unordered protein pairs and label known positives."""
    sorted_ids = tuple(sorted(set(protein_ids)))
    positives = {canonical_pair(node_a, node_b) for node_a, node_b in positive_edges}
    records: list[tuple[str, str]] = []
    labels: list[float] = []
    for source_index, protein_a in enumerate(sorted_ids):
        for protein_b in sorted_ids[source_index + 1 :]:
            pair = canonical_pair(protein_a, protein_b)
            records.append(pair)
            labels.append(1.0 if pair in positives else 0.0)
    return TCCIGReconstructionUniverse(
        records=tuple(records),
        labels=torch.tensor(labels, dtype=torch.float32),
    )


def false_negative_mask(
    *,
    num_nodes: int,
    known_positive_pairs: torch.Tensor,
) -> torch.Tensor:
    """Return a matrix mask for entries that must not act as negatives."""
    mask = torch.eye(num_nodes, dtype=torch.bool, device=known_positive_pairs.device)
    if known_positive_pairs.numel() == 0:
        return mask
    pairs = _normalize_pair_tensor(known_positive_pairs)
    mask[pairs[0], pairs[1]] = True
    mask[pairs[1], pairs[0]] = True
    return mask


def _normalize_pair_tensor(pairs: torch.Tensor) -> torch.Tensor:
    """Normalize pair tensors to shape ``(2, n)``."""
    if pairs.dim() != 2:
        raise ValueError("pairs must have shape (2, n) or (n, 2)")
    if pairs.size(0) == 2:
        normalized = pairs
    elif pairs.size(1) == 2:
        normalized = pairs.t().contiguous()
    else:
        raise ValueError("pairs must have shape (2, n) or (n, 2)")
    return normalized.to(dtype=torch.long)


def _bidirectional_positive_rows(positive_pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query and positive target indices for both undirected directions."""
    pairs = _normalize_pair_tensor(positive_pairs)
    if pairs.numel() == 0:
        empty = pairs.new_empty((0,))
        return empty, empty
    query_indices = torch.cat([pairs[0], pairs[1]], dim=0)
    target_indices = torch.cat([pairs[1], pairs[0]], dim=0)
    return query_indices, target_indices


def retrieval_infonce_loss(
    *,
    score_matrix: torch.Tensor,
    positive_pairs: torch.Tensor,
    known_positive_pairs: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute symmetric InfoNCE with false-negative masking."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    query_indices, target_indices = _bidirectional_positive_rows(positive_pairs)
    if query_indices.numel() == 0:
        return score_matrix.sum() * 0.0
    logits = score_matrix.float() / float(temperature)
    protected = false_negative_mask(
        num_nodes=score_matrix.size(0),
        known_positive_pairs=known_positive_pairs.to(device=score_matrix.device),
    )
    row_logits = logits[query_indices].clone()
    row_mask = protected[query_indices]
    row_mask[torch.arange(row_mask.size(0), device=row_mask.device), target_indices] = False
    row_logits = row_logits.masked_fill(row_mask, torch.finfo(row_logits.dtype).min)
    return functional.cross_entropy(row_logits, target_indices)


def adaptive_weighted_bce(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    negative_temperature: float = 4.0,
) -> torch.Tensor:
    """Return BCE where currently hard negatives receive larger weight."""
    loss_logits = logits.float()
    resolved_labels = labels.to(device=loss_logits.device, dtype=loss_logits.dtype)
    per_pair = functional.binary_cross_entropy_with_logits(
        loss_logits,
        resolved_labels,
        reduction="none",
    )
    weights = torch.ones_like(per_pair)
    negative_mask = resolved_labels <= 0.0
    if bool(negative_mask.any()):
        negative_scores = loss_logits[negative_mask]
        negative_weights = torch.softmax(negative_scores * negative_temperature, dim=0)
        negative_weights = negative_weights * float(negative_scores.numel())
        weights[negative_mask] = negative_weights
    return (per_pair * weights).mean() if per_pair.numel() else loss_logits.sum() * 0.0


def compute_retrieval_losses(
    *,
    retrieval_score_matrix: torch.Tensor,
    positive_pairs: torch.Tensor,
    known_positive_pairs: torch.Tensor,
    candidate_logits: torch.Tensor,
    candidate_labels: torch.Tensor,
    degree_predictions: torch.Tensor | None = None,
    degree_targets: torch.Tensor | None = None,
    struct_predictions: torch.Tensor | None = None,
    struct_targets: torch.Tensor | None = None,
    retrieval_weight: float = 1.0,
    reranker_weight: float = 1.0,
    degree_weight: float = 0.0,
    struct_weight: float = 0.0,
    temperature: float = 0.07,
) -> dict[str, torch.Tensor]:
    """Compute enabled graph-prior retrieval TCCIG training losses."""
    reference = candidate_logits.float()
    zero = reference.sum() * 0.0
    retrieval = (
        retrieval_infonce_loss(
            score_matrix=retrieval_score_matrix,
            positive_pairs=positive_pairs,
            known_positive_pairs=known_positive_pairs,
            temperature=temperature,
        )
        if retrieval_weight != 0.0
        else zero
    )
    reranker = (
        adaptive_weighted_bce(logits=reference, labels=candidate_labels)
        if reranker_weight != 0.0
        else zero
    )
    if degree_weight != 0.0:
        if degree_predictions is None or degree_targets is None:
            raise ValueError("degree targets are required when degree loss is enabled")
        degree = functional.smooth_l1_loss(
            torch.log1p(degree_predictions.float()),
            torch.log1p(degree_targets.to(device=degree_predictions.device).float()),
        )
    else:
        degree = zero
    if struct_weight != 0.0:
        if struct_predictions is None or struct_targets is None:
            raise ValueError("struct targets are required when struct loss is enabled")
        struct = functional.mse_loss(
            struct_predictions.float(),
            struct_targets.to(device=struct_predictions.device).float(),
        )
    else:
        struct = zero
    total = (
        retrieval_weight * retrieval
        + reranker_weight * reranker
        + degree_weight * degree
        + struct_weight * struct
    )
    return {
        "retrieval": retrieval,
        "reranker": reranker,
        "degree": degree,
        "struct": struct,
        "total": total,
    }


def hybrid_degree_capped_topk(
    *,
    candidate_pairs: torch.Tensor,
    scores: torch.Tensor,
    edge_budget: int,
    degree_cap: torch.Tensor,
    cap_slack: float = 1.0,
) -> torch.Tensor:
    """Select top-scoring edges under a soft per-node degree cap."""
    pairs = _normalize_pair_tensor(candidate_pairs)
    if pairs.size(1) != scores.numel():
        raise ValueError("candidate pair count must match score count")
    selected = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    if edge_budget <= 0 or scores.numel() == 0:
        return selected
    degree_counts = torch.zeros_like(degree_cap, dtype=torch.float32, device=scores.device)
    caps = torch.ceil(degree_cap.to(device=scores.device, dtype=torch.float32) + cap_slack)
    order = torch.argsort(scores.float(), descending=True)
    selected_count = 0
    deferred: list[int] = []
    for index_tensor in order:
        index = int(index_tensor.item())
        source = int(pairs[0, index].item())
        target = int(pairs[1, index].item())
        if degree_counts[source] < caps[source] and degree_counts[target] < caps[target]:
            selected[index] = True
            degree_counts[source] += 1.0
            degree_counts[target] += 1.0
            selected_count += 1
            if selected_count >= edge_budget:
                return selected
        else:
            deferred.append(index)
    for index in deferred:
        if selected_count >= edge_budget:
            break
        selected[index] = True
        selected_count += 1
    return selected


def mine_hard_negative_pairs(
    *,
    score_matrix: torch.Tensor,
    known_positive_pairs: torch.Tensor,
    top_k: int,
    max_pairs: int,
) -> torch.Tensor:
    """Mine high-scoring non-positive pairs from an exact retrieval score matrix."""
    if top_k <= 0 or max_pairs <= 0 or score_matrix.size(0) < 2:
        return torch.empty((2, 0), dtype=torch.long, device=score_matrix.device)
    protected = false_negative_mask(
        num_nodes=score_matrix.size(0),
        known_positive_pairs=known_positive_pairs.to(device=score_matrix.device),
    )
    scores = score_matrix.float().clone()
    scores = scores.masked_fill(protected, torch.finfo(scores.dtype).min)
    resolved_top_k = min(top_k, max(1, score_matrix.size(1) - 1))
    partners = torch.topk(scores, k=resolved_top_k, dim=1).indices
    sources = torch.arange(score_matrix.size(0), device=score_matrix.device).unsqueeze(1)
    sources = sources.expand_as(partners)
    pairs = torch.stack([sources.reshape(-1), partners.reshape(-1)], dim=0)
    pairs = pairs[:, pairs[0] != pairs[1]]
    if pairs.numel() == 0:
        return pairs
    canonical = torch.stack(
        [torch.minimum(pairs[0], pairs[1]), torch.maximum(pairs[0], pairs[1])],
        dim=0,
    )
    unique_pairs = torch.unique(canonical.t(), dim=0)
    pair_scores = score_matrix[unique_pairs[:, 0], unique_pairs[:, 1]]
    order = torch.argsort(pair_scores, descending=True)
    selected = unique_pairs[order[:max_pairs]]
    return cast(torch.Tensor, selected.t().contiguous())
