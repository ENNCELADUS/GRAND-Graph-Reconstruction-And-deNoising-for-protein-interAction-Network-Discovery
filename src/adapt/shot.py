"""SHOT domain-adaptation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as functional
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.utils.config import ConfigDict, as_bool, as_float, as_int, as_str, get_section

BatchValue = object
BatchInput = Mapping[str, BatchValue]
_LOW_PRECISION_DTYPES = {torch.float16, torch.bfloat16}


@dataclass(frozen=True)
class ShotOptimizerConfig:
    """Optimizer parameters for SHOT adaptation."""

    optimizer_type: str
    lr: float
    momentum: float
    weight_decay: float


@dataclass(frozen=True)
class ShotSchedulerConfig:
    """Scheduler parameters for SHOT adaptation."""

    scheduler_type: str
    gamma: float
    power: float


@dataclass(frozen=True)
class DomainAdaptationConfig:
    """Config contract for domain adaptation."""

    enabled: bool
    method: str
    target_split: str
    epochs: int
    beta: float
    entropy_weight: float
    diversity_weight: float
    epsilon: float
    freeze_prefixes: tuple[str, ...]
    optimizer: ShotOptimizerConfig
    scheduler: ShotSchedulerConfig


class OutputHeadFeatureHook:
    """Capture pre-classifier features through an output-head forward hook."""

    def __init__(self, model: nn.Module) -> None:
        unwrapped = _unwrap_model(model)
        output_head = getattr(unwrapped, "output_head", None)
        if not isinstance(output_head, nn.Module):
            raise ValueError(
                "SHOT requires model.output_head to exist as an nn.Module for feature extraction"
            )
        self._features: torch.Tensor | None = None
        self._handle = output_head.register_forward_pre_hook(self._hook)

    def _hook(self, module: nn.Module, args: tuple[object, ...]) -> None:
        del module
        if not args:
            raise ValueError("output_head hook received no inputs")
        first = args[0]
        if not isinstance(first, torch.Tensor):
            raise TypeError("output_head input must be a torch.Tensor")
        features = first
        if features.dim() > 2:
            features = features.flatten(start_dim=1)
        self._features = features

    def pop(self) -> torch.Tensor:
        """Return captured features from the latest forward call."""
        if self._features is None:
            raise RuntimeError("SHOT feature hook did not capture output_head inputs")
        features = self._features
        self._features = None
        return features

    def close(self) -> None:
        """Detach hook handle."""
        self._handle.remove()

    def __enter__(self) -> OutputHeadFeatureHook:
        """Return hook instance for context-manager usage."""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close hook handle when leaving a context."""
        del exc_type, exc, traceback
        self.close()


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return cast(nn.Module, model.module)
    return model


def parse_domain_adaptation_config(config: ConfigDict) -> DomainAdaptationConfig:
    """Parse ``training_config.domain_adaptation`` settings with defaults."""
    training_cfg = get_section(config, "training_config")
    adaptation_raw = training_cfg.get("domain_adaptation", {})
    if not isinstance(adaptation_raw, dict):
        raise ValueError("training_config.domain_adaptation must be a mapping")
    adaptation_cfg = adaptation_raw

    enabled = as_bool(adaptation_cfg.get("enabled", True), "domain_adaptation.enabled")
    method = as_str(adaptation_cfg.get("method", "shot"), "domain_adaptation.method").lower()
    if method not in {"none", "shot"}:
        raise ValueError("domain_adaptation.method must be 'none' or 'shot'")
    if enabled and method != "shot":
        raise ValueError("domain_adaptation.enabled=true requires method='shot'")

    target_split = as_str(
        adaptation_cfg.get("target_split", "test"),
        "domain_adaptation.target_split",
    ).lower()
    if target_split != "test":
        raise ValueError("domain_adaptation.target_split must be 'test'")

    freeze_prefixes_raw = adaptation_cfg.get("freeze_prefixes", ["output_head"])
    if not isinstance(freeze_prefixes_raw, list):
        raise ValueError("domain_adaptation.freeze_prefixes must be a list")
    freeze_prefixes = tuple(str(prefix) for prefix in freeze_prefixes_raw)
    if not freeze_prefixes:
        raise ValueError("domain_adaptation.freeze_prefixes must not be empty")

    optimizer_raw = adaptation_cfg.get("optimizer", {})
    if not isinstance(optimizer_raw, dict):
        raise ValueError("domain_adaptation.optimizer must be a mapping")
    optimizer_cfg = ShotOptimizerConfig(
        optimizer_type=as_str(
            optimizer_raw.get("type", "sgd"),
            "domain_adaptation.optimizer.type",
        ).lower(),
        lr=as_float(optimizer_raw.get("lr", 1.0e-4), "domain_adaptation.optimizer.lr"),
        momentum=as_float(
            optimizer_raw.get("momentum", 0.9),
            "domain_adaptation.optimizer.momentum",
        ),
        weight_decay=as_float(
            optimizer_raw.get("weight_decay", 1.0e-3),
            "domain_adaptation.optimizer.weight_decay",
        ),
    )
    if optimizer_cfg.optimizer_type != "sgd":
        raise ValueError("domain_adaptation.optimizer.type must be 'sgd'")

    scheduler_raw = adaptation_cfg.get("scheduler", {})
    if not isinstance(scheduler_raw, dict):
        raise ValueError("domain_adaptation.scheduler must be a mapping")
    scheduler_cfg = ShotSchedulerConfig(
        scheduler_type=as_str(
            scheduler_raw.get("type", "shot_poly"),
            "domain_adaptation.scheduler.type",
        ).lower(),
        gamma=as_float(
            scheduler_raw.get("gamma", 10.0),
            "domain_adaptation.scheduler.gamma",
        ),
        power=as_float(
            scheduler_raw.get("power", 0.75),
            "domain_adaptation.scheduler.power",
        ),
    )
    if scheduler_cfg.scheduler_type not in {"shot_poly", "none"}:
        raise ValueError("domain_adaptation.scheduler.type must be 'shot_poly' or 'none'")

    epochs = as_int(adaptation_cfg.get("epochs", 15), "domain_adaptation.epochs")
    if epochs <= 0:
        raise ValueError("domain_adaptation.epochs must be > 0")
    epsilon = as_float(adaptation_cfg.get("epsilon", 1.0e-5), "domain_adaptation.epsilon")
    if epsilon <= 0.0:
        raise ValueError("domain_adaptation.epsilon must be > 0")

    return DomainAdaptationConfig(
        enabled=enabled,
        method=method,
        target_split=target_split,
        epochs=epochs,
        beta=as_float(adaptation_cfg.get("beta", 0.3), "domain_adaptation.beta"),
        entropy_weight=as_float(
            adaptation_cfg.get("entropy_weight", 1.0),
            "domain_adaptation.entropy_weight",
        ),
        diversity_weight=as_float(
            adaptation_cfg.get("diversity_weight", 1.0),
            "domain_adaptation.diversity_weight",
        ),
        epsilon=epsilon,
        freeze_prefixes=freeze_prefixes,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
    )


def should_run_shot_adaptation(config: ConfigDict) -> bool:
    """Return whether SHOT adaptation is enabled in config."""
    adaptation_cfg = parse_domain_adaptation_config(config)
    return adaptation_cfg.enabled and adaptation_cfg.method == "shot"


def freeze_parameters_by_prefix(model: nn.Module, prefixes: Iterable[str]) -> int:
    """Freeze parameters matching prefixes and keep others trainable.

    Returns:
        Count of trainable parameters after freezing.
    """
    normalized = tuple(prefixes)
    if not normalized:
        raise ValueError("freeze prefixes must not be empty")

    def _matches(name: str) -> bool:
        return any(
            name.startswith(prefix) or name.startswith(f"module.{prefix}") for prefix in normalized
        )

    trainable_count = 0
    for name, parameter in model.named_parameters():
        if _matches(name):
            parameter.requires_grad = False
        else:
            parameter.requires_grad = True
            trainable_count += int(parameter.numel())

    if trainable_count <= 0:
        raise ValueError("SHOT produced zero trainable parameters after applying freeze_prefixes")
    return trainable_count


def logits_to_probabilities(logits: torch.Tensor, epsilon: float = 1.0e-5) -> torch.Tensor:
    """Convert logits to class probabilities (binary or multiclass)."""
    if logits.dim() == 1:
        pos = torch.sigmoid(logits)
        probs = torch.stack([1.0 - pos, pos], dim=1)
        return probs.clamp(min=epsilon, max=1.0 - epsilon)
    if logits.dim() == 2 and logits.size(1) == 1:
        pos = torch.sigmoid(logits.squeeze(1))
        probs = torch.stack([1.0 - pos, pos], dim=1)
        return probs.clamp(min=epsilon, max=1.0 - epsilon)

    probs = torch.softmax(logits, dim=-1)
    return probs.clamp(min=epsilon, max=1.0 - epsilon)


def entropy_loss(probabilities: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Compute per-batch entropy loss used by SHOT."""
    probs = probabilities.clamp(min=epsilon)
    return -torch.mean(torch.sum(probs * torch.log(probs), dim=1))


def diversity_loss(probabilities: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Compute SHOT diversity loss (negative marginal entropy)."""
    mean_probs = probabilities.mean(dim=0).clamp(min=epsilon)
    return torch.sum(mean_probs * torch.log(mean_probs))


def pseudo_label_loss(logits: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
    """Compute pseudo-label supervision loss for binary or multiclass logits."""
    if logits.dim() == 1:
        return functional.binary_cross_entropy_with_logits(logits, pseudo_labels.float())
    if logits.dim() == 2 and logits.size(1) == 1:
        squeezed = logits.squeeze(1)
        return functional.binary_cross_entropy_with_logits(squeezed, pseudo_labels.float())
    return functional.cross_entropy(logits, pseudo_labels.long())


def compute_centroids(
    feature_sums: torch.Tensor,
    class_masses: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Compute per-class centroids from class-weighted feature sums."""
    denom = class_masses.unsqueeze(1).clamp(min=epsilon)
    return feature_sums / denom


def _stable_pair_dtype(left: torch.Tensor, right: torch.Tensor) -> torch.dtype:
    """Choose a common dtype for SHOT similarity math."""
    dtype = torch.promote_types(left.dtype, right.dtype)
    if dtype in _LOW_PRECISION_DTYPES:
        return torch.float32
    return dtype


def assign_pseudo_labels(
    features: torch.Tensor,
    centroids: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Assign pseudo labels by nearest cosine-similarity centroid."""
    compute_dtype = _stable_pair_dtype(features, centroids)
    norm_features = functional.normalize(features.to(dtype=compute_dtype), dim=1, eps=epsilon)
    norm_centroids = functional.normalize(centroids.to(dtype=compute_dtype), dim=1, eps=epsilon)
    similarity = norm_features @ norm_centroids.transpose(0, 1)
    return torch.argmax(similarity, dim=1)
