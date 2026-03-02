"""Unit tests for optimization search-space helpers."""

from __future__ import annotations

import pytest
from src.optimize.search_space import (
    SearchParameter,
    apply_search_parameters,
    extend_with_nas_lite,
    parse_search_space,
    set_dot_path_value,
)
from src.utils.config import ConfigDict


def _base_config() -> ConfigDict:
    return {
        "run_config": {"stages": ["train"]},
        "device_config": {"ddp_enabled": False},
        "data_config": {},
        "model_config": {
            "model": "v5",
            "d_model": 192,
            "encoder_layers": 2,
            "cross_attn_layers": 2,
        },
        "training_config": {
            "batch_size": 16,
            "optimizer": {"lr": 1.0e-4, "weight_decay": 1.0e-2},
        },
    }


def test_parse_search_space_valid_contract() -> None:
    search_space = parse_search_space(
        [
            {
                "name": "optimizer_lr",
                "path": "training_config.optimizer.lr",
                "type": "float",
                "low": 1.0e-5,
                "high": 1.0e-3,
                "log": True,
            },
            {
                "name": "batch_size",
                "path": "training_config.batch_size",
                "type": "categorical",
                "choices": [8, 16, 32],
            },
        ]
    )

    assert len(search_space) == 2
    assert search_space[0].parameter_type == "float"
    assert search_space[1].choices == (8, 16, 32)


def test_set_dot_path_value_rejects_unknown_path() -> None:
    config = _base_config()

    with pytest.raises(ValueError, match="Unknown config path"):
        set_dot_path_value(config, "training_config.optimizer.not_exists", 1)


def test_apply_search_parameters_returns_copy() -> None:
    base_config = _base_config()
    search_space = [
        SearchParameter(
            name="optimizer_lr",
            path="training_config.optimizer.lr",
            parameter_type="float",
            low=1.0e-5,
            high=1.0e-3,
        )
    ]

    updated = apply_search_parameters(
        base_config=base_config,
        sampled_values={"optimizer_lr": 2.0e-4},
        search_space=search_space,
    )

    training_cfg_updated = updated["training_config"]
    assert isinstance(training_cfg_updated, dict)
    optimizer_cfg_updated = training_cfg_updated["optimizer"]
    assert isinstance(optimizer_cfg_updated, dict)
    assert optimizer_cfg_updated["lr"] == pytest.approx(2.0e-4)

    training_cfg_base = base_config["training_config"]
    assert isinstance(training_cfg_base, dict)
    optimizer_cfg_base = training_cfg_base["optimizer"]
    assert isinstance(optimizer_cfg_base, dict)
    assert optimizer_cfg_base["lr"] == pytest.approx(1.0e-4)


def test_extend_with_nas_lite_adds_architecture_parameters() -> None:
    config = _base_config()
    config["nas_lite"] = {
        "enabled": True,
        "method": "arch_params_hpo",
        "max_candidates": 20,
    }

    base_search_space = parse_search_space(
        [
            {
                "name": "optimizer_lr",
                "path": "training_config.optimizer.lr",
                "type": "float",
                "low": 1.0e-5,
                "high": 1.0e-4,
            }
        ]
    )
    extended = extend_with_nas_lite(root_config=config, base_search_space=base_search_space)
    names = {parameter.name for parameter in extended}

    assert "optimizer_lr" in names
    assert "nas_d_model" in names
    assert "nas_encoder_layers" in names
    assert "nas_cross_attn_layers" in names
