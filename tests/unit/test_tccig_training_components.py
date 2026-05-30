"""Unit tests for TCCIG teacher and loss helpers."""

from __future__ import annotations

import pytest
import torch
from src.topology.losses import TCCIGLossWeights, compute_tccig_losses
from src.train.tccig.mgae import MGAETeacher, mask_positive_edges
from src.train.tccig.teacher import OnlineTCCIGTeacher
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
