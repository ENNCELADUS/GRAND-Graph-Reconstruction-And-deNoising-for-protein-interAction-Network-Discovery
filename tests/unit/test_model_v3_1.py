"""Unit tests for V3_1 model with rich pooling."""

from __future__ import annotations

import torch
from src.pipeline.stages.train import build_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(model_name: str = "v3.1") -> dict[str, object]:
    return {
        "model_config": {
            "model": model_name,
            "input_dim": 8,
            "d_model": 8,
            "encoder_layers": 1,
            "cross_attn_layers": 1,
            "n_heads": 2,
            "mlp_head": {
                "hidden_dims": [8, 4],
                "dropout": 0.1,
                "activation": "gelu",
                "norm": "layernorm",
            },
            "regularization": {
                "dropout": 0.1,
                "token_dropout": 0.0,
                "cross_attention_dropout": 0.1,
                "stochastic_depth": 0.0,
            },
        }
    }


def _make_batch(
    batch_size: int = 2,
    seq_len_a: int = 5,
    seq_len_b: int = 4,
    input_dim: int = 8,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "emb_a": torch.randn(batch_size, seq_len_a, input_dim),
        "emb_b": torch.randn(batch_size, seq_len_b, input_dim),
        "len_a": torch.tensor([seq_len_a] * batch_size),
        "len_b": torch.tensor([seq_len_b] * batch_size),
    }


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


def test_build_model_v3_1_via_factory() -> None:
    """build_model('v3.1') must return a V3_1 instance."""
    model = build_model(_base_config("v3.1"))
    assert model.__class__.__name__ == "V3_1"


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_v3_1_instantiation() -> None:
    """V3_1 must instantiate without error from standard config."""
    from src.model.v3_1 import V3_1

    cfg = _base_config()["model_config"]
    assert isinstance(cfg, dict)
    cfg.pop("model")
    model = V3_1(**cfg)
    assert isinstance(model, torch.nn.Module)


def test_v3_1_uses_shared_rich_pooling_submodule() -> None:
    """InteractionCrossAttention must share rich-pooling parameters for A/B."""
    from src.model.v3_1 import V3_1

    cfg = _base_config()["model_config"]
    assert isinstance(cfg, dict)
    cfg.pop("model")
    model = V3_1(**cfg)
    assert hasattr(model.cross_attention, "pool_a"), "cross_attention must have pool_a"
    assert hasattr(model.cross_attention, "pool_b"), "cross_attention must have pool_b"
    assert model.cross_attention.pool_a is model.cross_attention.pool_b
    assert hasattr(model.cross_attention, "fusion"), "cross_attention must have fusion"


# ---------------------------------------------------------------------------
# Forward pass — output shapes
# ---------------------------------------------------------------------------


def test_v3_1_forward_logits_shape() -> None:
    """Forward pass must return logits of shape (batch, 1)."""
    from src.model.v3_1 import V3_1

    cfg = _base_config()["model_config"]
    assert isinstance(cfg, dict)
    cfg.pop("model")
    model = V3_1(**cfg)
    model.eval()

    batch = _make_batch()
    with torch.no_grad():
        out = model(batch)

    assert "logits" in out
    assert out["logits"].shape == (2, 1), f"Expected (2,1), got {out['logits'].shape}"


def test_v3_1_forward_with_labels_returns_loss() -> None:
    """Forward pass with labels must include a scalar loss."""
    from src.model.v3_1 import V3_1

    cfg = _base_config()["model_config"]
    assert isinstance(cfg, dict)
    cfg.pop("model")
    model = V3_1(**cfg)
    model.eval()

    batch = _make_batch()
    batch["label"] = torch.tensor([1.0, 0.0])
    with torch.no_grad():
        out = model(batch)

    assert "loss" in out
    assert out["loss"].ndim == 0, "loss must be a scalar"


def test_v3_1_forward_no_labels_no_loss_key() -> None:
    """Forward pass without labels must not include 'loss' key."""
    from src.model.v3_1 import V3_1

    cfg = _base_config()["model_config"]
    assert isinstance(cfg, dict)
    cfg.pop("model")
    model = V3_1(**cfg)
    model.eval()

    batch = _make_batch()
    with torch.no_grad():
        out = model(batch)

    assert "loss" not in out


# ---------------------------------------------------------------------------
# RichPooling module
# ---------------------------------------------------------------------------


def test_rich_pooling_output_shape() -> None:
    """RichPooling must return (batch, d_model) regardless of seq_len."""
    from src.model.v3_1 import RichPooling

    d_model = 16
    pool = RichPooling(d_model=d_model, dropout=0.0)
    pool.eval()

    x = torch.randn(3, 7, d_model)
    out = pool(x, padding_mask=None)
    assert out.shape == (3, d_model), f"Expected (3, {d_model}), got {out.shape}"


def test_rich_pooling_respects_padding_mask() -> None:
    """RichPooling output must differ when padding mask zeros out tokens."""
    from src.model.v3_1 import RichPooling

    torch.manual_seed(42)
    d_model = 16
    pool = RichPooling(d_model=d_model, dropout=0.0)
    pool.eval()

    x = torch.randn(1, 6, d_model)
    # No mask
    out_no_mask = pool(x, padding_mask=None)
    # Mask last 3 tokens as padding
    mask = torch.tensor([[False, False, False, True, True, True]])
    out_masked = pool(x, padding_mask=mask)

    assert not torch.allclose(out_no_mask, out_masked), (
        "Masking padding tokens must change the pooled output"
    )


def test_rich_pooling_component_ablation_combinations_forward() -> None:
    """Configured rich-pooling component ablations must keep forward shape stable."""
    from src.model.v3_1 import V3_1

    component_sets = [
        ["esm_cls", "mean", "attn", "max", "gated"],
        ["mean", "attn", "max", "gated"],
        ["esm_cls", "attn", "max", "gated"],
        ["esm_cls", "mean", "max", "gated"],
        ["esm_cls", "mean", "attn", "gated"],
        ["esm_cls", "mean", "attn", "max"],
    ]

    for components in component_sets:
        cfg = _base_config()["model_config"]
        assert isinstance(cfg, dict)
        cfg.pop("model")
        cfg["rich_pooling"] = {"components": components}
        model = V3_1(**cfg)
        model.eval()
        with torch.no_grad():
            out = model(_make_batch(seq_len_a=6, seq_len_b=7))
        assert out["logits"].shape == (2, 1)


def test_rich_pooling_no_attention_ablation_has_no_unused_trainable_parameters() -> None:
    """No-attention rich pooling must not leave trainable parameters unused."""
    from src.model.v3_1 import V3_1

    cfg = _base_config()["model_config"]
    assert isinstance(cfg, dict)
    cfg.pop("model")
    cfg["rich_pooling"] = {"components": ["esm_cls", "mean", "max", "gated"]}
    model = V3_1(**cfg)

    output = model(_make_batch(seq_len_a=6, seq_len_b=7))["logits"]
    output.sum().backward()

    unused_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused_parameters == []


def test_rich_pooling_residue_components_ignore_bos_and_eos() -> None:
    """Residue-only components must exclude ESM BOS and EOS special tokens."""
    from src.model.v3_1 import RichPooling

    torch.manual_seed(7)
    pool = RichPooling(d_model=8, dropout=0.0, components=("mean", "attn", "max"))
    pool.eval()

    x = torch.randn(2, 6, 8)
    changed_specials = x.clone()
    changed_specials[:, 0] = changed_specials[:, 0] + 100.0
    changed_specials[:, 5] = changed_specials[:, 5] - 100.0
    mask = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, False, False, False, True, True],
        ]
    )
    changed_specials[1, 3] = changed_specials[1, 3] - 100.0

    with torch.no_grad():
        out = pool(x, padding_mask=mask)
        changed = pool(changed_specials, padding_mask=mask)

    assert torch.allclose(out, changed)


def test_rich_pooling_cls_component_uses_bos_token() -> None:
    """The esm_cls component must depend on the first ESM special token."""
    from src.model.v3_1 import RichPooling

    torch.manual_seed(11)
    pool = RichPooling(d_model=8, dropout=0.0, components=("esm_cls",))
    pool.eval()

    x = torch.randn(2, 5, 8)
    changed_bos = x.clone()
    changed_bos[:, 0] = changed_bos[:, 0] + 100.0

    with torch.no_grad():
        out = pool(x, padding_mask=None)
        changed = pool(changed_bos, padding_mask=None)

    assert not torch.allclose(out, changed)


def test_rich_pooling_rejects_invalid_components() -> None:
    """Invalid or degenerate rich-pooling component configs must fail fast."""
    from src.model.v3_1 import RichPooling

    invalid_configs = [
        ("gated",),
        ("mean", "mean"),
        ("unknown",),
    ]
    for components in invalid_configs:
        try:
            RichPooling(d_model=8, dropout=0.0, components=components)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for components={components!r}")


# ---------------------------------------------------------------------------
# Padding / variable-length sequences
# ---------------------------------------------------------------------------


def test_v3_1_forward_variable_lengths() -> None:
    """V3_1 must handle variable-length sequences without error."""
    from src.model.v3_1 import V3_1

    cfg = _base_config()["model_config"]
    assert isinstance(cfg, dict)
    cfg.pop("model")
    model = V3_1(**cfg)
    model.eval()

    torch.manual_seed(1)
    batch = {
        "emb_a": torch.randn(2, 5, 8),
        "emb_b": torch.randn(2, 6, 8),
        "len_a": torch.tensor([5, 3]),
        "len_b": torch.tensor([6, 3]),
    }
    with torch.no_grad():
        out = model(batch)

    assert out["logits"].shape == (2, 1)


# ---------------------------------------------------------------------------
# Training mode — gradients flow
# ---------------------------------------------------------------------------


def test_v3_1_gradients_flow() -> None:
    """Loss.backward() must produce non-None gradients for all parameters."""
    from src.model.v3_1 import V3_1

    cfg = _base_config()["model_config"]
    assert isinstance(cfg, dict)
    cfg.pop("model")
    model = V3_1(**cfg)
    model.train()

    batch = _make_batch()
    batch["label"] = torch.tensor([1.0, 0.0])
    out = model(batch)
    out["loss"].backward()

    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"Parameters with no gradient: {no_grad}"
