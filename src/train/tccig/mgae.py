"""PyTorch MGAE-style train-only topology teacher for TCCIG."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


@dataclass(frozen=True)
class MaskedEdges:
    """Visible and masked edge split for one teacher corruption step."""

    visible_positive_edges: torch.Tensor
    masked_positive_edges: torch.Tensor
    visible_edge_index: torch.Tensor


@dataclass(frozen=True)
class MGAETeacherStepOutput:
    """Outputs from one MGAE teacher optimization step."""

    loss: torch.Tensor
    positive_logits: torch.Tensor
    negative_logits: torch.Tensor
    masked_positive_edges: torch.Tensor
    negative_edges: torch.Tensor


def _canonical_upper_edges(edges: torch.Tensor) -> torch.Tensor:
    """Normalize edge indices to upper-triangle undirected pairs."""
    if edges.dim() != 2:
        raise ValueError("positive_edges must have shape (2, e) or (e, 2)")
    normalized = edges if edges.size(0) == 2 else edges.t().contiguous()
    if normalized.size(0) != 2:
        raise ValueError("positive_edges must have shape (2, e) or (e, 2)")
    src = torch.minimum(normalized[0], normalized[1])
    dst = torch.maximum(normalized[0], normalized[1])
    keep = src != dst
    if not bool(keep.any()):
        return normalized.new_empty((2, 0))
    pairs = torch.stack([src[keep], dst[keep]], dim=0)
    unique_pairs = torch.unique(pairs.t(), dim=0)
    sort_keys = unique_pairs[:, 0] * (int(unique_pairs[:, 1].max().item()) + 1)
    sort_keys = sort_keys + unique_pairs[:, 1]
    order = torch.argsort(sort_keys)
    return unique_pairs[order].t().contiguous()


def _undirected_with_self_loops(
    *,
    upper_edges: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Build a message-passing edge index from upper-triangle edges."""
    device = upper_edges.device
    loops = torch.arange(num_nodes, device=device, dtype=torch.long)
    self_loops = torch.stack([loops, loops], dim=0)
    if upper_edges.numel() == 0:
        return self_loops
    reversed_edges = torch.stack([upper_edges[1], upper_edges[0]], dim=0)
    return torch.cat([upper_edges, reversed_edges, self_loops], dim=1)


def mask_positive_edges(
    *,
    positive_edges: torch.Tensor,
    num_nodes: int,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> MaskedEdges:
    """Randomly mask positive edges before teacher reconstruction."""
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in (0, 1)")
    edges = _canonical_upper_edges(positive_edges).to(dtype=torch.long)
    edge_count = edges.size(1)
    if edge_count == 0:
        visible = edges
        masked = edges
    else:
        mask_count = min(edge_count, max(1, int(round(edge_count * mask_ratio))))
        permutation = torch.randperm(edge_count, generator=generator, device=edges.device)
        mask_indices = permutation[:mask_count]
        visible_indices = permutation[mask_count:]
        masked = edges[:, mask_indices]
        visible = edges[:, visible_indices]
    return MaskedEdges(
        visible_positive_edges=visible,
        masked_positive_edges=masked,
        visible_edge_index=_undirected_with_self_loops(
            upper_edges=visible,
            num_nodes=num_nodes,
        ),
    )


def sample_negative_edges(
    *,
    positive_edges: torch.Tensor,
    num_nodes: int,
    count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample upper-triangle non-edges after positive masking."""
    if count <= 0 or num_nodes < 2:
        return positive_edges.new_empty((2, 0), dtype=torch.long)
    positives = _canonical_upper_edges(positive_edges).to(dtype=torch.long)
    positive_keys = {(int(src), int(dst)) for src, dst in positives.t().detach().cpu().tolist()}
    candidates: list[tuple[int, int]] = []
    for src in range(num_nodes):
        for dst in range(src + 1, num_nodes):
            if (src, dst) not in positive_keys:
                candidates.append((src, dst))
    if not candidates:
        return positives.new_empty((2, 0), dtype=torch.long)
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=positive_edges.device).t()
    take_count = min(count, candidate_tensor.size(1))
    permutation = torch.randperm(
        candidate_tensor.size(1),
        generator=generator,
        device=positive_edges.device,
    )
    return candidate_tensor[:, permutation[:take_count]]


class DenseGraphEncoder(nn.Module):
    """Small dense graph encoder used for sampled PRING subgraphs."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = dropout
        layers: list[nn.Linear] = []
        for layer_index in range(num_layers):
            in_dim = input_dim if layer_index == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def _normalized_adjacency(
        *,
        edge_index: torch.Tensor,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build symmetric degree-normalized dense adjacency."""
        adjacency = torch.zeros((num_nodes, num_nodes), device=edge_index.device, dtype=dtype)
        if edge_index.numel() > 0:
            adjacency[edge_index[0], edge_index[1]] = 1.0
        degree = adjacency.sum(dim=1).clamp(min=1.0)
        inv_sqrt_degree = degree.rsqrt()
        return inv_sqrt_degree.unsqueeze(1) * adjacency * inv_sqrt_degree.unsqueeze(0)

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> list[torch.Tensor]:
        """Encode node features and return every layer representation."""
        norm_adj = self._normalized_adjacency(
            edge_index=edge_index,
            num_nodes=node_features.size(0),
            dtype=node_features.dtype,
        )
        h = node_features
        outputs: list[torch.Tensor] = []
        for layer_index, layer in enumerate(self.layers):
            h = layer(norm_adj @ h)
            if layer_index != len(self.layers) - 1:
                h = functional.gelu(h)
                h = functional.dropout(h, p=self.dropout, training=self.training)
            outputs.append(h)
        return outputs


class CrossLayerLinkDecoder(nn.Module):
    """MGAE-style cross-layer elementwise-product link decoder."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_encoder_layers: int,
        decoder_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = hidden_dim * num_encoder_layers * num_encoder_layers
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, 1),
        )

    @staticmethod
    def _cross_layer_features(
        representations: list[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Build all source-layer by target-layer product features."""
        src = edge_index[0]
        dst = edge_index[1]
        features: list[torch.Tensor] = []
        for src_repr in representations:
            for dst_repr in representations:
                features.append(src_repr[src] * dst_repr[dst])
        if not features:
            raise ValueError("representations must be non-empty")
        return torch.cat(features, dim=-1)

    def forward(
        self,
        representations: list[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Score edge candidates from cross-layer node representations."""
        if edge_index.size(1) == 0:
            return representations[0].new_zeros((0,))
        features = self._cross_layer_features(representations, edge_index)
        return self.decoder(features).squeeze(-1)


class MGAETeacher(nn.Module):
    """MGAE-style masked-edge reconstruction teacher."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        decoder_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        resolved_decoder_hidden_dim = decoder_hidden_dim or hidden_dim
        self.encoder = DenseGraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = CrossLayerLinkDecoder(
            hidden_dim=hidden_dim,
            num_encoder_layers=num_layers,
            decoder_hidden_dim=resolved_decoder_hidden_dim,
            dropout=dropout,
        )

    def score_pairs(
        self,
        *,
        node_features: torch.Tensor,
        visible_positive_edges: torch.Tensor,
        candidate_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate edges with a graph-aware train-only encoder."""
        edge_index = _undirected_with_self_loops(
            upper_edges=_canonical_upper_edges(visible_positive_edges),
            num_nodes=node_features.size(0),
        )
        representations = self.encoder(node_features, edge_index)
        return self.decoder(representations, candidate_edges)

    def training_step(
        self,
        *,
        node_features: torch.Tensor,
        positive_edges: torch.Tensor,
        mask_ratio: float,
        negative_ratio: int,
        generator: torch.Generator | None = None,
    ) -> MGAETeacherStepOutput:
        """Run one masked-edge reconstruction step and return the differentiable loss."""
        masked = mask_positive_edges(
            positive_edges=positive_edges,
            num_nodes=node_features.size(0),
            mask_ratio=mask_ratio,
            generator=generator,
        )
        negative_count = masked.masked_positive_edges.size(1) * max(1, negative_ratio)
        negative_edges = sample_negative_edges(
            positive_edges=positive_edges,
            num_nodes=node_features.size(0),
            count=negative_count,
            generator=generator,
        )
        representations = self.encoder(node_features, masked.visible_edge_index)
        positive_logits = self.decoder(representations, masked.masked_positive_edges)
        negative_logits = self.decoder(representations, negative_edges)
        logits = torch.cat([positive_logits, negative_logits], dim=0)
        labels = torch.cat(
            [
                torch.ones_like(positive_logits),
                torch.zeros_like(negative_logits),
            ],
            dim=0,
        )
        if logits.numel() == 0:
            loss = node_features.sum() * 0.0
        else:
            loss = functional.binary_cross_entropy_with_logits(logits, labels)
        return MGAETeacherStepOutput(
            loss=loss,
            positive_logits=positive_logits,
            negative_logits=negative_logits,
            masked_positive_edges=masked.masked_positive_edges,
            negative_edges=negative_edges,
        )
