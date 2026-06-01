"""Offline graph-prior teacher artifacts for graph-prior retrieval TCCIG."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch

from src.train.tccig.mgae import MGAETeacher


@dataclass(frozen=True)
class GraphPriorArtifacts:
    """Frozen train-only graph-prior targets for TCCIG student training."""

    protein_ids: tuple[str, ...]
    structural_embeddings: torch.Tensor
    degree_targets: torch.Tensor
    edge_prior_pairs: torch.Tensor
    edge_prior_probabilities: torch.Tensor

    def save(self, directory: Path) -> None:
        """Persist graph-prior artifacts."""
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protein_ids": self.protein_ids,
                "structural_embeddings": self.structural_embeddings.cpu(),
                "degree_targets": self.degree_targets.cpu(),
                "edge_prior_pairs": self.edge_prior_pairs.cpu(),
                "edge_prior_probabilities": self.edge_prior_probabilities.cpu(),
            },
            directory / "graph_prior_artifacts.pt",
        )
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "num_proteins": len(self.protein_ids),
                    "num_edge_priors": int(self.edge_prior_pairs.size(1)),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> GraphPriorArtifacts:
        """Load graph-prior artifacts from disk."""
        payload = torch.load(directory / "graph_prior_artifacts.pt", map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("graph prior artifact payload must be a mapping")
        return cls(
            protein_ids=tuple(str(value) for value in payload["protein_ids"]),
            structural_embeddings=payload["structural_embeddings"].float(),
            degree_targets=payload["degree_targets"].float(),
            edge_prior_pairs=payload["edge_prior_pairs"].long(),
            edge_prior_probabilities=payload["edge_prior_probabilities"].float(),
        )

    def index_for(self, protein_ids: tuple[str, ...], *, device: torch.device) -> torch.Tensor:
        """Return artifact row indices for the requested proteins."""
        index = {protein_id: row for row, protein_id in enumerate(self.protein_ids)}
        return torch.tensor([index[protein_id] for protein_id in protein_ids], device=device)


def _node_feature_matrix(
    *,
    protein_ids: tuple[str, ...],
    node_features: Mapping[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Stack node features in protein-id order."""
    return torch.stack([node_features[protein_id].float() for protein_id in protein_ids], dim=0).to(
        device
    )


def _graph_edges_as_indices(
    *,
    graph: nx.Graph,
    protein_ids: tuple[str, ...],
    device: torch.device,
) -> torch.Tensor:
    """Return upper-triangle graph edges in local node-index space."""
    node_to_index = {protein_id: index for index, protein_id in enumerate(protein_ids)}
    edges: list[tuple[int, int]] = []
    for protein_a, protein_b in graph.edges():
        if protein_a not in node_to_index or protein_b not in node_to_index:
            continue
        index_a = node_to_index[protein_a]
        index_b = node_to_index[protein_b]
        if index_a == index_b:
            continue
        edges.append((min(index_a, index_b), max(index_a, index_b)))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(sorted(set(edges)), dtype=torch.long, device=device).t().contiguous()


def build_graph_prior_artifacts(
    *,
    graph: nx.Graph,
    node_features: Mapping[str, torch.Tensor],
    hidden_dim: int,
    num_layers: int,
    decoder_hidden_dim: int,
    dropout: float,
    epochs: int,
    mask_ratio: float,
    negative_ratio: int,
    seed: int,
    device: torch.device,
) -> GraphPriorArtifacts:
    """Train a pure-PyTorch MGAE/S2-lite teacher and return frozen targets."""
    protein_ids = tuple(sorted(node for node in graph.nodes if node in node_features))
    if not protein_ids:
        raise ValueError("graph-prior teacher requires at least one protein with features")
    features = _node_feature_matrix(
        protein_ids=protein_ids,
        node_features=node_features,
        device=device,
    )
    positive_edges = _graph_edges_as_indices(graph=graph, protein_ids=protein_ids, device=device)
    teacher = MGAETeacher(
        input_dim=features.size(1),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        decoder_hidden_dim=decoder_hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=1.0e-3)
    generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    for epoch in range(max(0, epochs)):
        generator.manual_seed(seed + epoch)
        optimizer.zero_grad(set_to_none=True)
        step = teacher.training_step(
            node_features=features,
            positive_edges=positive_edges,
            mask_ratio=mask_ratio,
            negative_ratio=negative_ratio,
            generator=generator,
        )
        torch.autograd.backward(step.loss)
        optimizer.step()
    teacher.eval()
    with torch.no_grad():
        representations = teacher.encoder(
            features,
            _visible_edge_index_with_self_loops(positive_edges, features.size(0)),
        )
        structural_embeddings = representations[-1].detach().cpu()
        edge_logits = teacher.score_pairs(
            node_features=features,
            visible_positive_edges=positive_edges,
            candidate_edges=positive_edges,
        )
    degree_targets = torch.tensor(
        [float(graph.degree(protein_id)) for protein_id in protein_ids],
        dtype=torch.float32,
    )
    return GraphPriorArtifacts(
        protein_ids=protein_ids,
        structural_embeddings=structural_embeddings,
        degree_targets=degree_targets,
        edge_prior_pairs=positive_edges.detach().cpu(),
        edge_prior_probabilities=torch.sigmoid(edge_logits).detach().cpu(),
    )


def _visible_edge_index_with_self_loops(upper_edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Build a teacher message-passing edge index with self loops."""
    device = upper_edges.device
    loops = torch.arange(num_nodes, device=device, dtype=torch.long)
    self_loops = torch.stack([loops, loops], dim=0)
    if upper_edges.numel() == 0:
        return self_loops
    reversed_edges = torch.stack([upper_edges[1], upper_edges[0]], dim=0)
    return torch.cat([upper_edges, reversed_edges, self_loops], dim=1)
