"""Adaptive loss-balancing helpers for topology fine-tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence

import torch
from torch import nn

EPSILON = 1.0e-8
DEFAULT_TASK_WEIGHTS = {"bce": 1.0, "density": 1.0, "shape": 1.0}


@dataclass(frozen=True)
class TopologyLossNormalizationConfig:
    """Configuration for detached EMA loss normalization."""

    enabled: bool = False
    ema_decay: float = 0.95
    clip_value: float = 5.0


@dataclass(frozen=True)
class TopologyGradNormConfig:
    """Configuration for grouped GradNorm task reweighting."""

    enabled: bool = False
    alpha: float = 0.5
    learning_rate: float = 0.02
    min_weight: float = 0.2
    max_weight: float = 5.0


@dataclass
class TopologyAdaptiveLossState:
    """Mutable state for EMA normalization and grouped GradNorm."""

    loss_ema: dict[str, float] = field(default_factory=dict)
    task_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_TASK_WEIGHTS),
    )
    initial_task_losses: dict[str, float] | None = None


def normalize_topology_loss_terms(
    *,
    raw_terms: Mapping[str, torch.Tensor],
    state: TopologyAdaptiveLossState,
    config: TopologyLossNormalizationConfig,
) -> dict[str, torch.Tensor]:
    """Return topology terms normalized by detached EMA statistics."""
    if not config.enabled:
        return {name: loss for name, loss in raw_terms.items()}

    normalized: dict[str, torch.Tensor] = {}
    for name, loss in raw_terms.items():
        raw_value = float(loss.detach().item())
        previous = state.loss_ema.get(name)
        ema_value = raw_value if previous is None else (
            config.ema_decay * previous + (1.0 - config.ema_decay) * raw_value
        )
        state.loss_ema[name] = max(ema_value, EPSILON)
        normalized_loss = loss / loss.new_tensor(state.loss_ema[name])
        normalized[name] = torch.clamp(normalized_loss, max=config.clip_value)
    return normalized


def update_gradnorm_task_weights(
    *,
    task_losses: Mapping[str, torch.Tensor],
    state: TopologyAdaptiveLossState,
    reference_parameters: Sequence[nn.Parameter],
    config: TopologyGradNormConfig,
) -> dict[str, float]:
    """Update grouped task weights with one GradNorm-style step."""
    active_task_names = tuple(task_losses)
    current_weights = {
        name: float(state.task_weights.get(name, DEFAULT_TASK_WEIGHTS[name]))
        for name in active_task_names
    }
    if not config.enabled:
        state.task_weights.update(current_weights)
        return current_weights

    params = tuple(reference_parameters)
    if not params:
        state.task_weights.update(current_weights)
        return current_weights

    loss_values = {name: float(loss.detach().item()) for name, loss in task_losses.items()}
    if not all(math.isfinite(value) and value > 0.0 for value in loss_values.values()):
        state.task_weights.update(current_weights)
        return current_weights

    if state.initial_task_losses is None:
        state.initial_task_losses = dict(loss_values)
        state.task_weights.update(current_weights)
        return current_weights

    grad_norms = {
        name: _gradient_norm(loss=loss, reference_parameters=params)
        for name, loss in task_losses.items()
    }
    if not all(math.isfinite(value) and value > 0.0 for value in grad_norms.values()):
        state.task_weights.update(current_weights)
        return current_weights

    weighted_grad_norms = {
        name: current_weights[name] * grad_norms[name] for name in active_task_names
    }
    average_grad_norm = sum(weighted_grad_norms.values()) / float(len(weighted_grad_norms))
    loss_ratios = {
        name: loss_values[name] / max(state.initial_task_losses.get(name, loss_values[name]), EPSILON)
        for name in active_task_names
    }
    mean_loss_ratio = sum(loss_ratios.values()) / float(len(loss_ratios))
    inverse_train_rates = {
        name: loss_ratios[name] / max(mean_loss_ratio, EPSILON) for name in active_task_names
    }

    updated = {}
    for name in active_task_names:
        target_grad_norm = average_grad_norm * (inverse_train_rates[name] ** config.alpha)
        grad_estimate = math.copysign(grad_norms[name], weighted_grad_norms[name] - target_grad_norm)
        candidate = current_weights[name] - config.learning_rate * grad_estimate
        updated[name] = min(config.max_weight, max(config.min_weight, candidate))

    updated = _renormalize_task_weights(updated)
    state.task_weights.update(updated)
    return dict(updated)


def initialize_output_head_bias_with_prior(
    model: nn.Module,
    *,
    positive_edge_probability: float,
) -> float:
    """Initialize the last output-head bias to the logit of a sparse prior."""
    final_linear = _last_output_linear(model)
    if final_linear is None or final_linear.bias is None:
        raise ValueError("model.output_head must expose a final linear layer with bias")

    clipped_probability = min(max(positive_edge_probability, EPSILON), 1.0 - EPSILON)
    bias_value = math.log(clipped_probability / (1.0 - clipped_probability))
    with torch.no_grad():
        final_linear.bias.fill_(bias_value)
    return bias_value


def _gradient_norm(
    *,
    loss: torch.Tensor,
    reference_parameters: Sequence[nn.Parameter],
) -> float:
    grads = torch.autograd.grad(
        loss,
        reference_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared_norm = 0.0
    for grad in grads:
        if grad is None:
            continue
        squared_norm += float(grad.detach().pow(2).sum().item())
    return math.sqrt(max(squared_norm, 0.0))


def _last_output_linear(model: nn.Module) -> nn.Linear | None:
    output_head = getattr(model, "output_head", None)
    if not isinstance(output_head, nn.Module):
        return None

    final_linear: nn.Linear | None = None
    for module in output_head.modules():
        if isinstance(module, nn.Linear):
            final_linear = module
    return final_linear


def _renormalize_task_weights(task_weights: Mapping[str, float]) -> dict[str, float]:
    target_sum = float(len(task_weights))
    weight_sum = sum(task_weights.values())
    if weight_sum <= 0.0:
        return dict(DEFAULT_TASK_WEIGHTS)
    scale = target_sum / weight_sum
    return {name: weight * scale for name, weight in task_weights.items()}


__all__ = [
    "TopologyAdaptiveLossState",
    "TopologyGradNormConfig",
    "TopologyLossNormalizationConfig",
    "initialize_output_head_bias_with_prior",
    "normalize_topology_loss_terms",
    "update_gradnorm_task_weights",
]
