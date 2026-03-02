"""Unit tests for SHOT adaptation helpers."""

from __future__ import annotations

import pytest
import src.run.stage_adapt as stage_adapt_module
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
from src.utils.distributed import DistributedContext
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


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


def test_pseudo_label_loss_binary_path() -> None:
    logits = torch.tensor([[0.1], [-0.5], [1.2]], dtype=torch.float32)
    pseudo_labels = torch.tensor([1, 0, 1], dtype=torch.long)
    loss = pseudo_label_loss(logits=logits, pseudo_labels=pseudo_labels)

    assert loss.ndim == 0
    assert float(loss.item()) >= 0.0


def test_stage_adapt_all_reduce_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    context = DistributedContext(
        ddp_enabled=True,
        is_distributed=True,
        rank=0,
        local_rank=0,
        world_size=2,
    )
    tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)

    monkeypatch.setattr(stage_adapt_module.dist, "is_initialized", lambda: True)

    def fake_all_reduce(value: torch.Tensor, op: object) -> None:
        del op
        value.mul_(2.0)

    monkeypatch.setattr(stage_adapt_module.dist, "all_reduce", fake_all_reduce)

    reduced = stage_adapt_module._all_reduce_sum(tensor, context)
    assert torch.allclose(reduced, torch.tensor([2.0, 4.0], dtype=torch.float32))


class _TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "emb_a": torch.ones((2, 4), dtype=torch.float32) * index,
            "emb_b": torch.ones((2, 4), dtype=torch.float32) * index,
            "label": torch.tensor(float(index % 2), dtype=torch.float32),
        }


def test_build_target_loaders_uses_distributed_sampler_when_ddp_enabled() -> None:
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

    assert isinstance(eval_loader.sampler, DistributedSampler)
    assert isinstance(train_loader.sampler, DistributedSampler)
    assert train_loader.drop_last is True
    assert eval_loader.drop_last is False
