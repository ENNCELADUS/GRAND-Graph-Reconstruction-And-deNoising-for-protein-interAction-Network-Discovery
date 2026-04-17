"""Unit tests for model construction entrypoints."""

import pytest
from src.pipeline.stages.train import build_model


def _base_config(model_name: str) -> dict[str, object]:
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


def _v5_config() -> dict[str, object]:
    config = _base_config("v5")
    model_config = config["model_config"]
    assert isinstance(model_config, dict)
    model_config["pair_dim"] = 8
    model_config["cnn_dim"] = 8
    model_config["cnn_blocks"] = 1
    model_config["interaction_map"] = {
        "include_pair_features": True,
        "similarity": "cosine",
        "eps": 1.0e-8,
    }
    model_config["pooling"] = {"mode": "max_mean"}
    model_config["regularization"] = {
        "dropout": 0.1,
        "token_dropout": 0.0,
        "cross_attention_dropout": 0.1,
        "stochastic_depth": 0.0,
        "cnn_dropout": 0.0,
    }
    return config


def test_build_model_v3() -> None:
    model = build_model(_base_config("v3"))
    assert model.__class__.__name__ == "V3"


def test_build_model_v4() -> None:
    model = build_model(_base_config("v4"))
    assert model.__class__.__name__ == "V4"


def test_build_model_v5() -> None:
    model = build_model(_v5_config())
    assert model.__class__.__name__ == "V5"


def test_build_model_v6_is_deprecated() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(_base_config("v6"))


def test_build_model_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(_base_config("unknown"))
