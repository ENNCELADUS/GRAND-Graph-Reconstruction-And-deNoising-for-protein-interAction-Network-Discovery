"""Unit tests for TCCIG teacher and loss helpers."""

from __future__ import annotations

import networkx as nx
import pytest
import torch
from src.topology.losses import TCCIGLossWeights, compute_tccig_losses
from src.train.tccig.graph_prior import build_graph_prior_artifacts
from src.train.tccig.mgae import MGAETeacher, mask_positive_edges
from src.train.tccig.runner import (
    TCCIG_TRAIN_CSV_COLUMNS,
    _build_tccig_epoch_csv_row,
    _tccig_reconstruction_metrics_payload,
)
from src.train.tccig.teacher import OnlineTCCIGTeacher
from src.train.tccig.trainer import _candidate_reranker_training_logits
from src.train.tccig.validation import _encode_validation_nodes
from src.train.topology import shared as topology_train
from torch import nn


class _WrappedTeacher(nn.Module):
    """DDP-shaped wrapper that hides custom teacher methods."""

    def __init__(self, module: MGAETeacher) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.module(*args, **kwargs)


class _UnwrappingAccelerator:
    """Small accelerator double for wrapped-teacher regression coverage."""

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, _WrappedTeacher) else model

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()


class _EmbeddingRepository:
    """Tiny validation embedding repository double."""

    def __init__(self, embeddings: dict[str, torch.Tensor]) -> None:
        self.embeddings = embeddings

    def get_many(self, protein_ids: tuple[str, ...]) -> dict[str, torch.Tensor]:
        return {protein_id: self.embeddings[protein_id] for protein_id in protein_ids}


class _DataContext:
    """Tiny data-context double exposing only the repository."""

    def __init__(self, embeddings: dict[str, torch.Tensor]) -> None:
        self.embedding_repository = _EmbeddingRepository(embeddings)


def test_validation_node_encoding_batches_by_length_and_restores_order() -> None:
    embeddings = {
        "A": torch.full((6, 2), 1.0),
        "B": torch.full((2, 2), 2.0),
        "C": torch.full((5, 2), 3.0),
        "D": torch.full((3, 2), 4.0),
    }
    batch_lengths: list[list[int]] = []

    def encode_proteins(
        *,
        protein_embeddings: torch.Tensor,
        protein_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_lengths.append([int(length) for length in protein_lengths.tolist()])
        assert protein_embeddings.size(0) <= 2
        return {
            "node": protein_embeddings[:, 0, :],
            "degree": protein_lengths.float().unsqueeze(-1),
        }

    encoded = _encode_validation_nodes(
        data_context=_DataContext(embeddings),  # type: ignore[arg-type]
        protein_ids=("A", "B", "C", "D"),
        device=torch.device("cpu"),
        encode_proteins=encode_proteins,
        node_batch_size=2,
    )

    assert batch_lengths == [[6, 5], [3, 2]]
    assert encoded["node"][:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert encoded["degree"].squeeze(-1).tolist() == [6.0, 2.0, 5.0, 3.0]


def test_tccig_epoch_csv_row_persists_reconstruction_retrieval_metrics() -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    validation_result = topology_train.ValidationEpochResult(
        decision_threshold=0.5,
        val_pair_stats={
            "val_loss": 0.7,
            "val_auprc": 0.4,
            "val_candidate_auprc": 0.31,
            "val_retrieval_recall_at_20": 0.62,
            "val_reconstruction_candidate_count": 20.0,
            "val_reconstruction_positive_count": 5.0,
            "val_composite_score": 0.27,
        },
        internal_val_topology_stats={
            "graph_sim": 0.2,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.3,
            "cc_mmd": 0.4,
        },
        val_pair_pass_seconds=1.0,
        threshold_resolution_seconds=0.0,
        internal_validation_seconds=2.0,
        val_topology_loss=0.1,
        val_total_loss=0.8,
    )
    train_stats = {
        "bce": 1.0,
        "graph_similarity": 0.0,
        "relative_density": 0.0,
        "degree_mmd": 0.0,
        "clustering_mmd": 0.0,
        "total": 1.0,
        "planned_subgraphs": 2.0,
        "covered_positive_edges": 3.0,
        "total_positive_edges": 3.0,
        "positive_edge_coverage_ratio": 1.0,
        "mean_positive_edge_reuse": 1.0,
        "all_subgraph_pairs": 12.0,
        "supervised_pairs": 8.0,
        "bce_positive_pairs": 3.0,
        "bce_target_negative_pairs": 6.0,
        "bce_negative_pairs": 5.0,
        "bce_negative_ratio": 1.67,
        "bce_supervised_fraction": 0.67,
        "edge_cover_sampling_s": 0.1,
        "train_forward_backward_s": 0.2,
        "topology_loss_scale": 1.0,
    }

    row = _build_tccig_epoch_csv_row(
        epoch=1,
        epoch_seconds=3.0,
        train_stats=train_stats,
        validation_result=validation_result,
        optimizer=optimizer,
        peak_gpu_mem_mb=0.0,
    )

    assert "Val Candidate AUPRC" in TCCIG_TRAIN_CSV_COLUMNS
    assert "Val Retrieval Recall@20%" in TCCIG_TRAIN_CSV_COLUMNS
    assert "Val Composite Score" in TCCIG_TRAIN_CSV_COLUMNS
    assert row["Val Candidate AUPRC"] == pytest.approx(0.31)
    assert row["Val Retrieval Recall@20%"] == pytest.approx(0.62)
    assert row["Val Composite Score"] == pytest.approx(0.27)
    payload = _tccig_reconstruction_metrics_payload(validation_result.val_pair_stats)
    assert payload["val_candidate_auprc"] == pytest.approx(0.31)
    assert payload["val_retrieval_recall_at_20"] == pytest.approx(0.62)
    assert payload["val_composite_score"] == pytest.approx(0.27)


def test_candidate_reranker_training_logits_ignore_density_calibration() -> None:
    calibrated_logits = torch.tensor([-5.0, -4.0])
    reranker_logits = torch.tensor([0.2, -0.3])

    selected = _candidate_reranker_training_logits(
        {
            "logits": calibrated_logits,
            "reranker_logits": reranker_logits,
        }
    )

    assert torch.equal(selected, reranker_logits)
    fallback = _candidate_reranker_training_logits({"logits": calibrated_logits})
    assert torch.equal(fallback, calibrated_logits)


def test_mask_positive_edges_masks_requested_fraction_and_keeps_visible_edges() -> None:
    edges = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]], dtype=torch.long)

    masked = mask_positive_edges(
        positive_edges=edges,
        num_nodes=4,
        mask_ratio=0.5,
        generator=torch.Generator().manual_seed(13),
    )

    assert masked.masked_positive_edges.shape == (2, 2)
    assert masked.visible_edge_index.shape[0] == 2
    assert masked.visible_edge_index.shape[1] >= 4
    assert set(map(tuple, masked.masked_positive_edges.t().tolist())).isdisjoint(
        set(map(tuple, masked.visible_positive_edges.t().tolist()))
    )


def test_mgae_teacher_training_step_returns_loss_and_logits() -> None:
    teacher = MGAETeacher(input_dim=8, hidden_dim=12, num_layers=2, dropout=0.0)
    node_features = torch.randn(5, 8)
    positive_edges = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]], dtype=torch.long)

    output = teacher.training_step(
        node_features=node_features,
        positive_edges=positive_edges,
        mask_ratio=0.5,
        negative_ratio=1,
        generator=torch.Generator().manual_seed(47),
    )

    assert output.loss.shape == ()
    assert output.positive_logits.ndim == 1
    assert output.negative_logits.shape == output.positive_logits.shape
    output.loss.backward()


def test_offline_graph_prior_artifacts_include_struct_degree_and_edge_priors() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["P1", "P2", "P3"])
    graph.add_edges_from([("P1", "P2"), ("P2", "P3")])
    node_features = {
        "P1": torch.randn(8),
        "P2": torch.randn(8),
        "P3": torch.randn(8),
    }

    artifacts = build_graph_prior_artifacts(
        graph=graph,
        node_features=node_features,
        hidden_dim=12,
        num_layers=2,
        decoder_hidden_dim=12,
        dropout=0.0,
        epochs=1,
        mask_ratio=0.5,
        negative_ratio=1,
        seed=17,
        device=torch.device("cpu"),
    )

    assert artifacts.protein_ids == ("P1", "P2", "P3")
    assert artifacts.structural_embeddings.shape == (3, 12)
    assert artifacts.degree_targets.tolist() == [1.0, 2.0, 1.0]
    assert artifacts.edge_prior_pairs.shape[0] == 2
    assert artifacts.edge_prior_probabilities.shape == (2,)


def test_online_teacher_scores_when_prepared_teacher_is_wrapped() -> None:
    teacher = MGAETeacher(input_dim=8, hidden_dim=12, num_layers=2, dropout=0.0)
    online_teacher = OnlineTCCIGTeacher(
        teacher=_WrappedTeacher(teacher),  # type: ignore[arg-type]
        optimizer=torch.optim.SGD(teacher.parameters(), lr=1e-3),
        mask_ratio=0.5,
        negative_ratio=1,
    )
    node_features = torch.randn(5, 8)
    positive_edges = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]], dtype=torch.long)
    candidate_pairs = torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.long)

    probabilities = online_teacher.train_and_score(
        node_features=node_features,
        positive_edges=positive_edges,
        candidate_pairs=candidate_pairs,
        seed=31,
        device=torch.device("cpu"),
        accelerator=_UnwrappingAccelerator(),  # type: ignore[arg-type]
        loss_scale=1.0,
    )

    assert probabilities.shape == (3,)
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_tccig_loss_skips_disabled_teacher_inputs() -> None:
    logits = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0])
    pair_index_a = torch.tensor([0, 0, 1])
    pair_index_b = torch.tensor([1, 2, 2])

    losses = compute_tccig_losses(
        logits=logits,
        labels=labels,
        pair_index_a=pair_index_a,
        pair_index_b=pair_index_b,
        num_nodes=3,
        m_hat=torch.tensor(1.5),
        weights=TCCIGLossWeights(
            teacher=0.0,
            budget=0.0,
            density=0.0,
            degree=0.0,
            clustering=0.0,
        ),
    )

    assert losses["teacher"].item() == pytest.approx(0.0)
    assert losses["budget"].item() == pytest.approx(0.0)
    assert losses["total"].requires_grad


def test_tccig_loss_requires_teacher_probabilities_when_enabled() -> None:
    logits = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0])
    pair_index_a = torch.tensor([0, 0, 1])
    pair_index_b = torch.tensor([1, 2, 2])

    with pytest.raises(ValueError, match="teacher_probabilities"):
        compute_tccig_losses(
            logits=logits,
            labels=labels,
            pair_index_a=pair_index_a,
            pair_index_b=pair_index_b,
            num_nodes=3,
            m_hat=torch.tensor(1.5),
            weights=TCCIGLossWeights(teacher=0.1),
        )


def test_tccig_loss_stays_finite_for_saturated_half_precision_logits() -> None:
    num_nodes = 40
    candidate_pairs = torch.triu_indices(num_nodes, num_nodes, offset=1)
    logits = torch.full(
        (candidate_pairs.size(1),),
        -100.0,
        dtype=torch.float16,
        requires_grad=True,
    )
    labels = torch.zeros(candidate_pairs.size(1), dtype=torch.float16)
    labels[:16] = 1.0

    losses = compute_tccig_losses(
        logits=logits,
        labels=labels,
        pair_index_a=candidate_pairs[0],
        pair_index_b=candidate_pairs[1],
        num_nodes=num_nodes,
        m_hat=torch.tensor(0.0, dtype=torch.float16),
        weights=TCCIGLossWeights(
            teacher=0.0,
            budget=0.1,
            density=0.1,
            degree=0.05,
            clustering=0.02,
        ),
    )

    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["relative_density"])
