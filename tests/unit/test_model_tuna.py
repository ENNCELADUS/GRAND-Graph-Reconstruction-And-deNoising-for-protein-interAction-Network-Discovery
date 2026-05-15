"""Unit tests for the TUnA-style model."""

from __future__ import annotations

import pytest
import torch
from src.pipeline.stages.train import build_model


def _tuna_model_config(
    *,
    output_type: str = "linear",
    inter_mask_mode: str = "official_block",
    input_dim: int = 8,
    max_sequence_length: int = 512,
) -> dict[str, object]:
    output_head: dict[str, object] = {"type": output_type}
    if output_type == "sngp":
        output_head.update(
            {
                "rffs": 16,
                "gp_cov_momentum": -1.0,
                "gp_ridge_penalty": 1.0,
            }
        )
    return {
        "model": "tuna",
        "input_dim": input_dim,
        "hid_dim": 64,
        "n_layers": 1,
        "n_heads": 8,
        "ff_dim": 256,
        "dropout": 0.2,
        "max_sequence_length": max_sequence_length,
        "activation_function": "swish",
        "exclude_special_tokens": True,
        "inter_mask_mode": inter_mask_mode,
        "random_seed": 47,
        "output_head": output_head,
    }


def _config(
    *,
    output_type: str = "linear",
    inter_mask_mode: str = "official_block",
) -> dict[str, object]:
    return {
        "model_config": _tuna_model_config(
            output_type=output_type,
            inter_mask_mode=inter_mask_mode,
        )
    }


def _make_batch(
    *,
    batch_size: int = 2,
    seq_len_a: int = 7,
    seq_len_b: int = 8,
    input_dim: int = 8,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "emb_a": torch.randn(batch_size, seq_len_a, input_dim),
        "emb_b": torch.randn(batch_size, seq_len_b, input_dim),
        "len_a": torch.tensor([seq_len_a] * batch_size),
        "len_b": torch.tensor([seq_len_b] * batch_size),
        "label": torch.tensor([1.0, 0.0]),
    }


def test_build_model_tuna_via_factory() -> None:
    model = build_model(_config())
    assert model.__class__.__name__ == "TUNA"


@pytest.mark.parametrize("output_type", ["linear", "sngp"])
@pytest.mark.parametrize("inter_mask_mode", ["official_block", "cross_chain"])
def test_tuna_ablation_combinations_forward_logits_shape(
    output_type: str,
    inter_mask_mode: str,
) -> None:
    from src.model.tuna import TUNA

    cfg = _tuna_model_config(output_type=output_type, inter_mask_mode=inter_mask_mode)
    model = TUNA(**cfg)
    model.eval()

    with torch.no_grad():
        out = model(_make_batch())

    assert out["logits"].shape == (2, 1)


def test_tuna_eval_is_ab_ba_symmetric() -> None:
    from src.model.tuna import TUNA

    model = TUNA(**_tuna_model_config(inter_mask_mode="cross_chain"))
    model.eval()
    batch = _make_batch(batch_size=2, seq_len_a=7, seq_len_b=9)
    swapped = {
        "emb_a": batch["emb_b"],
        "emb_b": batch["emb_a"],
        "len_a": batch["len_b"],
        "len_b": batch["len_a"],
    }

    with torch.no_grad():
        logits = model(batch)["logits"]
        swapped_logits = model(swapped)["logits"]

    assert torch.allclose(logits, swapped_logits, atol=1.0e-6)


def test_tuna_uses_official_small_hidden_dimensions() -> None:
    from src.model.tuna import TUNA

    model = TUNA(**_tuna_model_config())

    assert model.hid_dim == 64
    assert model.n_layers == 1
    assert model.ff_dim == 256
    assert len(model.intra_encoder.layers) == 1
    assert len(model.inter_encoder.layers) == 1


def test_tuna_residue_windows_ignore_bos_eos_and_padding() -> None:
    from src.model.tuna import TUNA

    model = TUNA(**_tuna_model_config(input_dim=2))
    model.eval()
    embeddings = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [99.0, 99.0],
            ]
        ]
    )
    windows, lengths = model._residue_windows(embeddings, torch.tensor([5]))

    assert lengths.tolist() == [3]
    assert windows.tolist() == [[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]]


def test_tuna_eval_crop_is_deterministic_and_limited() -> None:
    from src.model.tuna import TUNA

    model = TUNA(**_tuna_model_config(input_dim=1, max_sequence_length=3))
    model.eval()
    embeddings = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    windows, lengths = model._residue_windows(embeddings, torch.tensor([8]))

    assert lengths.tolist() == [3]
    assert windows.squeeze(-1).tolist() == [[1.0, 2.0, 3.0]]


def test_tuna_autocast_keeps_large_embeddings_finite() -> None:
    from src.model.tuna import TUNA

    torch.manual_seed(0)
    model = TUNA(**_tuna_model_config(input_dim=8, inter_mask_mode="cross_chain"))
    model.train()
    batch = {
        "emb_a": torch.randn(2, 7, 8) * 20000.0,
        "emb_b": torch.randn(2, 8, 8) * 20000.0,
        "len_a": torch.tensor([7, 7]),
        "len_b": torch.tensor([8, 8]),
        "label": torch.tensor([1.0, 0.0]),
    }

    with torch.autocast("cpu", dtype=torch.float16):
        output = model(batch)

    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["loss"])


@pytest.mark.parametrize("output_type", ["linear", "sngp"])
@pytest.mark.parametrize("inter_mask_mode", ["official_block", "cross_chain"])
def test_tuna_ablation_combinations_have_no_unused_trainable_parameters(
    output_type: str,
    inter_mask_mode: str,
) -> None:
    from src.model.tuna import TUNA

    model = TUNA(**_tuna_model_config(output_type=output_type, inter_mask_mode=inter_mask_mode))
    model.on_train_epoch_start(epoch_index=0, total_epochs=1)
    logits = model(_make_batch())["logits"]
    logits.sum().backward()

    unused_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused_parameters == []
