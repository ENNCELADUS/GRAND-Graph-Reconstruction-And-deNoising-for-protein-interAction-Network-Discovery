"""Unit tests for SHOT adaptation helpers."""

from __future__ import annotations

import pytest
import src.pipeline.stages.adapt as stage_adapt_module
import torch
from src.adapt import (
    OutputHeadFeatureHook,
    assign_pseudo_labels,
    compute_centroids,
    diversity_loss,
    entropy_loss,
    logits_to_probabilities,
    parse_domain_adaptation_config,
    pseudo_label_loss,
    should_run_shot_adaptation,
)
from src.pipeline.runtime import DistributedContext
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _base_config() -> dict[str, object]:
    return {
        "training_config": {
            "domain_adaptation": {
                "enabled": True,
                "method": "shot",
                "target_split": "test",
            }
        }
    }


def test_parse_domain_adaptation_defaults() -> None:
    parsed = parse_domain_adaptation_config(_base_config())

    assert parsed.enabled is True
    assert parsed.method == "shot"
    assert parsed.target_split == "test"
    assert parsed.epochs == 15
    assert parsed.beta == pytest.approx(0.3)
    assert parsed.freeze_prefixes == ("output_head",)
    assert parsed.optimizer.optimizer_type == "sgd"
    assert parsed.scheduler.scheduler_type == "shot_poly"


def test_parse_domain_adaptation_defaults_when_section_missing() -> None:
    parsed = parse_domain_adaptation_config({"training_config": {}})

    assert parsed.enabled is True
    assert parsed.method == "shot"
    assert parsed.target_split == "test"


def test_parse_domain_adaptation_invalid_method_raises() -> None:
    config = _base_config()
    training_cfg = config["training_config"]
    assert isinstance(training_cfg, dict)
    adaptation_cfg = training_cfg["domain_adaptation"]
    assert isinstance(adaptation_cfg, dict)
    adaptation_cfg["method"] = "bad_method"

    with pytest.raises(ValueError, match="domain_adaptation.method"):
        parse_domain_adaptation_config(config)


def test_should_run_shot_adaptation_false_when_disabled() -> None:
    config = _base_config()
    training_cfg = config["training_config"]
    assert isinstance(training_cfg, dict)
    adaptation_cfg = training_cfg["domain_adaptation"]
    assert isinstance(adaptation_cfg, dict)
    adaptation_cfg["enabled"] = False

    assert should_run_shot_adaptation(config) is False


class _DummyHeadModel(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.output_head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.output_head(x)}


class _HalfLogitHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_head = nn.Linear(2, 1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.output_head(x).to(dtype=torch.float16)}


class _TargetFeatureDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        self._features = (
            torch.tensor([1.0, 0.0], dtype=torch.float32),
            torch.tensor([0.0, 1.0], dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self._features[index],
            "label": torch.tensor(float(index), dtype=torch.float32),
        }


@pytest.mark.parametrize("feature_dim", [8, 16, 32, 64])
def test_output_head_feature_hook_captures_batch_aligned_features(feature_dim: int) -> None:
    model = _DummyHeadModel(feature_dim=feature_dim)
    batch = torch.randn(5, feature_dim)

    hook = OutputHeadFeatureHook(model=model)
    try:
        output = model(batch)
        features = hook.pop()
    finally:
        hook.close()

    assert output["logits"].shape == (5, 1)
    assert features.shape == (5, feature_dim)
    assert torch.allclose(features, batch)


def test_binary_probability_and_losses_are_stable() -> None:
    logits = torch.tensor([[0.0], [2.0], [-2.0]], dtype=torch.float32)
    probs = logits_to_probabilities(logits=logits, epsilon=1.0e-5)

    assert probs.shape == (3, 2)
    assert torch.all(probs > 0.0)
    entropy_value = entropy_loss(probabilities=probs, epsilon=1.0e-5)
    diversity_value = diversity_loss(probabilities=probs, epsilon=1.0e-5)
    assert torch.isfinite(entropy_value)
    assert torch.isfinite(diversity_value)


def test_centroid_and_pseudo_labels_handle_zero_mass_class() -> None:
    feature_sums = torch.tensor([[2.0, 2.0], [0.0, 0.0]], dtype=torch.float32)
    class_masses = torch.tensor([2.0, 0.0], dtype=torch.float32)
    centroids = compute_centroids(
        feature_sums=feature_sums,
        class_masses=class_masses,
        epsilon=1.0e-5,
    )

    assert torch.isfinite(centroids).all()
    features = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
    labels = assign_pseudo_labels(features=features, centroids=centroids, epsilon=1.0e-5)
    assert labels.dtype == torch.long
    assert labels.shape == (2,)


def test_assign_pseudo_labels_promotes_mixed_precision_inputs() -> None:
    features = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    centroids = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16)

    labels = assign_pseudo_labels(features=features, centroids=centroids, epsilon=1.0e-5)

    assert torch.equal(labels, torch.tensor([0, 1], dtype=torch.long))


def test_pseudo_label_loss_binary_path() -> None:
    logits = torch.tensor([[0.1], [-0.5], [1.2]], dtype=torch.float32)
    pseudo_labels = torch.tensor([1, 0, 1], dtype=torch.long)
    loss = pseudo_label_loss(logits=logits, pseudo_labels=pseudo_labels)

    assert loss.ndim == 0
    assert float(loss.item()) >= 0.0


def test_compute_global_centroids_promotes_mixed_precision_inputs() -> None:
    config = parse_domain_adaptation_config(_base_config())
    model = _HalfLogitHeadModel()
    loader = DataLoader(_TargetFeatureDataset(), batch_size=2, shuffle=False)
    feature_hook = OutputHeadFeatureHook(model=model)

    try:
        centroids = stage_adapt_module._compute_global_centroids(
            model=model,
            loader=loader,
            feature_hook=feature_hook,
            device=torch.device("cpu"),
            use_amp=False,
            adaptation_config=config,
            accelerator=_FakeAccelerator(),
        )
    finally:
        feature_hook.close()

    expected = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    assert centroids.dtype == torch.float32
    assert torch.allclose(centroids, expected)


class _TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "emb_a": torch.ones((2, 4), dtype=torch.float32) * index,
            "emb_b": torch.ones((2, 4), dtype=torch.float32) * index,
            "label": torch.tensor(float(index % 2), dtype=torch.float32),
        }


class _TinyShotModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.output_head = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )


class _FakeAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.autocast_calls = 0
        self.backward_calls = 0
        self.reduce_calls = 0

    def autocast(self) -> object:
        from contextlib import nullcontext

        self.autocast_calls += 1
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_calls += 1
        loss.backward()

    def reduce(self, value: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        self.reduce_calls += 1
        return value


def test_prepare_shot_optimizer_params_preserves_frozen_requires_grad_for_ddp() -> None:
    model = _TinyShotModel()

    optimizer_params, trainable_count = stage_adapt_module._prepare_shot_optimizer_params(
        model=model,
        prefixes=("output_head",),
        preserve_frozen_requires_grad=True,
    )

    optimizer_param_ids = {id(parameter) for parameter in optimizer_params}
    encoder_param_ids = {id(parameter) for parameter in model.encoder.parameters()}
    output_head_param_ids = {id(parameter) for parameter in model.output_head.parameters()}

    assert trainable_count == sum(parameter.numel() for parameter in model.encoder.parameters())
    assert optimizer_param_ids == encoder_param_ids
    assert optimizer_param_ids.isdisjoint(output_head_param_ids)
    assert all(parameter.requires_grad for parameter in model.output_head.parameters())


def test_prepare_shot_optimizer_params_freezes_prefixes_without_ddp() -> None:
    model = _TinyShotModel()

    optimizer_params, trainable_count = stage_adapt_module._prepare_shot_optimizer_params(
        model=model,
        prefixes=("output_head",),
        preserve_frozen_requires_grad=False,
    )

    optimizer_param_ids = {id(parameter) for parameter in optimizer_params}
    encoder_param_ids = {id(parameter) for parameter in model.encoder.parameters()}
    output_head_param_ids = {id(parameter) for parameter in model.output_head.parameters()}

    assert trainable_count == sum(parameter.numel() for parameter in model.encoder.parameters())
    assert optimizer_param_ids == encoder_param_ids
    assert optimizer_param_ids.isdisjoint(output_head_param_ids)
    assert all(not parameter.requires_grad for parameter in model.output_head.parameters())


def test_build_target_loaders_keeps_plain_loaders_when_accelerator_handles_ddp() -> None:
    config: dict[str, object] = {
        "training_config": {
            "batch_size": 2,
            "domain_adaptation": {
                "enabled": True,
                "method": "shot",
                "target_split": "test",
            },
        },
        "data_config": {
            "dataloader": {
                "num_workers": 0,
                "pin_memory": False,
            }
        },
    }
    adaptation_config = parse_domain_adaptation_config(config)
    base_loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False)
    dataloaders: dict[str, DataLoader[dict[str, object]]] = {"test": base_loader}
    context = DistributedContext(
        ddp_enabled=True,
        is_distributed=True,
        rank=0,
        local_rank=0,
        world_size=2,
    )

    eval_loader, train_loader = stage_adapt_module._build_target_loaders(
        config=config,
        dataloaders=dataloaders,
        adaptation_config=adaptation_config,
        distributed_context=context,
    )

    assert not type(eval_loader.sampler).__name__.endswith("DistributedSampler")
    assert not type(train_loader.sampler).__name__.endswith("DistributedSampler")
    assert train_loader.drop_last is True
    assert eval_loader.drop_last is False


def test_train_one_shot_epoch_uses_accelerator_backward_and_reduce() -> None:
    model = _DummyHeadModel(feature_dim=2)
    loader = DataLoader(_TargetFeatureDataset(), batch_size=1, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    accelerator = _FakeAccelerator()

    epoch_stats = stage_adapt_module._train_one_shot_epoch(
        model=model,
        loader=loader,
        pseudo_labels=torch.tensor([0, 1], dtype=torch.long),
        optimizer=optimizer,
        scheduler=None,
        accelerator=accelerator,
        device=torch.device("cpu"),
        use_amp=False,
        adaptation_config=parse_domain_adaptation_config(_base_config()),
        distributed_context=DistributedContext(ddp_enabled=False, is_distributed=False),
    )

    assert "total_loss" in epoch_stats
    assert accelerator.autocast_calls == len(loader)
    assert accelerator.backward_calls == len(loader)
    assert accelerator.reduce_calls >= 1
