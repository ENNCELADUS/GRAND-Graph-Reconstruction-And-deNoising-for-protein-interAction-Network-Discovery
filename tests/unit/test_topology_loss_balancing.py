"""Unit tests for adaptive topology loss balancing helpers."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from src.topology.loss_balancing import (
    TopologyAdaptiveLossState,
    TopologyGradNormConfig,
    TopologyLossNormalizationConfig,
    initialize_output_head_bias_with_prior,
    normalize_topology_loss_terms,
    update_gradnorm_task_weights,
)


def test_normalize_topology_loss_terms_uses_running_ema_and_clip() -> None:
    state = TopologyAdaptiveLossState()
    config = TopologyLossNormalizationConfig(enabled=True, ema_decay=0.5, clip_value=1.4)

    first = normalize_topology_loss_terms(
        raw_terms={
            "graph_similarity": torch.tensor(2.0, dtype=torch.float32),
            "relative_density": torch.tensor(4.0, dtype=torch.float32),
            "degree_mmd": torch.tensor(1.0, dtype=torch.float32),
            "clustering_mmd": torch.tensor(3.0, dtype=torch.float32),
        },
        state=state,
        config=config,
    )

    second = normalize_topology_loss_terms(
        raw_terms={
            "graph_similarity": torch.tensor(6.0, dtype=torch.float32),
            "relative_density": torch.tensor(4.0, dtype=torch.float32),
            "degree_mmd": torch.tensor(1.0, dtype=torch.float32),
            "clustering_mmd": torch.tensor(3.0, dtype=torch.float32),
        },
        state=state,
        config=config,
    )

    assert first["graph_similarity"].item() == pytest.approx(1.0)
    assert second["graph_similarity"].item() == pytest.approx(1.4)
    assert state.loss_ema["graph_similarity"] == pytest.approx(4.0)


def test_update_gradnorm_task_weights_downweights_density_objective() -> None:
    parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    task_losses = {
        "bce": (parameter - 0.5).square(),
        "density": 10.0 * (parameter + 2.0).square(),
        "shape": 0.5 * parameter.square(),
    }
    state = TopologyAdaptiveLossState(
        initial_task_losses={name: float(loss.detach()) for name, loss in task_losses.items()}
    )

    updated = update_gradnorm_task_weights(
        task_losses=task_losses,
        state=state,
        reference_parameters=(parameter,),
        config=TopologyGradNormConfig(enabled=True, alpha=0.5, learning_rate=0.02),
    )

    assert updated["density"] < 1.0
    assert sum(updated.values()) == pytest.approx(3.0)
    assert updated["bce"] > 0.0
    assert updated["shape"] > 0.0


def test_initialize_output_head_bias_with_prior_sets_last_linear_bias() -> None:
    class _ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_head = nn.Sequential(
                nn.Linear(4, 3),
                nn.ReLU(),
                nn.Linear(3, 1),
            )

    model = _ToyModel()
    applied_bias = initialize_output_head_bias_with_prior(model, positive_edge_probability=0.1)

    final_linear = model.output_head[-1]
    assert isinstance(final_linear, nn.Linear)
    assert applied_bias == pytest.approx(math.log(0.1 / 0.9), rel=1e-6)
    assert final_linear.bias is not None
    assert final_linear.bias.item() == pytest.approx(applied_bias, rel=1e-6)
