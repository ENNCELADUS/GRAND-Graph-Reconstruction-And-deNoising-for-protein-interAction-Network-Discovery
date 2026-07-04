"""Tests for exp03 TCCIG loss-conflict diagnostic configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tccig import train as tccig_train
from tccig.exp03_configs import (
    EXP03_CALIBRATED_GRID,
    PHASE_A_RUN_IDS,
    PHASE_B_CONFIG_RUN_IDS,
    build_exp03_configs,
    write_exp03_configs,
)
from tccig.s2gae import _parse_config

BASE_CONFIG = Path("configs/tccig/02_balanced_subset.yaml")
COMMITTED_CONFIG_DIR = Path("configs/tccig/exp03")
FIXED_VALIDATION_LOSSES = {"alpha": 1.0, "beta": 8.0, "gamma": 0.5, "delta": 0.0}


def _load_base_config() -> dict[str, object]:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return payload


def _refiner(config: dict[str, object]) -> dict[str, object]:
    value = config["refiner"]
    assert isinstance(value, dict)
    return value


def _topology_training(config: dict[str, object]) -> dict[str, object]:
    value = _refiner(config)["topology_training"]
    assert isinstance(value, dict)
    return value


def _topology_validation(config: dict[str, object]) -> dict[str, object]:
    value = _refiner(config)["topology_validation"]
    assert isinstance(value, dict)
    return value


def _loss(config: dict[str, object]) -> dict[str, object]:
    value = _refiner(config)["loss"]
    assert isinstance(value, dict)
    return value


def _graph_selection(config: dict[str, object]) -> dict[str, object]:
    value = config["graph_selection"]
    assert isinstance(value, dict)
    return value


def _run_id(config: dict[str, object]) -> str:
    run = config["run"]
    assert isinstance(run, dict)
    return str(run["run_id"])


def test_phase_a_configs_keep_validation_monitor_fixed() -> None:
    configs = build_exp03_configs(_load_base_config(), include_phase_b=False)

    assert tuple(configs) == PHASE_A_RUN_IDS
    for run_id, config in configs.items():
        assert _run_id(config) == run_id
        assert _refiner(config)["monitor_metric"] == "val_topology_loss"
        assert _topology_validation(config)["enabled"] is True
        assert _topology_validation(config)["losses"] == FIXED_VALIDATION_LOSSES
        refined_rule = _graph_selection(config)["refined_output_rule"]
        assert refined_rule == {
            "type": "calibrated",
            "objective": "val_topology_loss",
            "grid": list(EXP03_CALIBRATED_GRID),
        }
        assert _graph_selection(config)["rules"] == [{"type": "threshold", "value": 0.5}]


def test_phase_a_training_knobs_match_exp03_matrix() -> None:
    configs = build_exp03_configs(_load_base_config(), include_phase_b=False)

    a1 = _topology_training(configs["03_a1_bce_only"])
    assert a1["enabled"] is False
    assert a1["topo_only_after_epoch"] is None
    assert a1["topology_weight"] == pytest.approx(0.0)
    assert a1["weights"] == FIXED_VALIDATION_LOSSES
    assert _loss(configs["03_a1_bce_only"])["pos_weight"] == pytest.approx(1.0)

    expected_weights = {
        "03_a2_bce_graph_sim": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
        "03_a3_bce_density": {"alpha": 0.0, "beta": 8.0, "gamma": 0.0, "delta": 0.0},
        "03_a4_bce_degree": {"alpha": 0.0, "beta": 0.0, "gamma": 0.5, "delta": 0.0},
        "03_a5_bce_full_topology": FIXED_VALIDATION_LOSSES,
    }
    for run_id, weights in expected_weights.items():
        topology_training = _topology_training(configs[run_id])
        assert topology_training["enabled"] is True
        assert topology_training["topo_only_after_epoch"] is None
        assert topology_training["topology_weight"] == pytest.approx(1.0)
        assert topology_training["weights"] == weights
        assert _loss(configs[run_id])["pos_weight"] == pytest.approx(1.0)


def test_phase_b_config_only_levers_are_named_and_bounded() -> None:
    configs = build_exp03_configs(_load_base_config(), include_phase_b=True)

    assert tuple(configs)[-5:] == PHASE_B_CONFIG_RUN_IDS
    assert _topology_training(configs["03_b1_beta2"])["weights"] == {
        "alpha": 1.0,
        "beta": 2.0,
        "gamma": 0.5,
        "delta": 0.0,
    }
    assert _topology_training(configs["03_b2_beta4"])["weights"] == {
        "alpha": 1.0,
        "beta": 4.0,
        "gamma": 0.5,
        "delta": 0.0,
    }
    assert (
        _topology_training(configs["03_b3_topology_weight_0p5"])["weights"]
        == FIXED_VALIDATION_LOSSES
    )
    assert (
        _topology_training(configs["03_b4_topology_weight_2p0"])["weights"]
        == FIXED_VALIDATION_LOSSES
    )
    assert (
        _topology_training(configs["03_b5_bce_pos_weight_0p5"])["weights"]
        == FIXED_VALIDATION_LOSSES
    )
    assert _topology_training(configs["03_b1_beta2"])["topology_weight"] == pytest.approx(1.0)
    assert _topology_training(configs["03_b2_beta4"])["topology_weight"] == pytest.approx(1.0)
    assert _topology_training(configs["03_b3_topology_weight_0p5"])[
        "topology_weight"
    ] == pytest.approx(0.5)
    assert _topology_training(configs["03_b4_topology_weight_2p0"])[
        "topology_weight"
    ] == pytest.approx(2.0)
    assert _topology_training(configs["03_b5_bce_pos_weight_0p5"])[
        "topology_weight"
    ] == pytest.approx(1.0)
    for run_id in (
        "03_b1_beta2",
        "03_b2_beta4",
        "03_b3_topology_weight_0p5",
        "03_b4_topology_weight_2p0",
    ):
        assert _loss(configs[run_id])["pos_weight"] == pytest.approx(1.0)
    assert _loss(configs["03_b5_bce_pos_weight_0p5"])["pos_weight"] == pytest.approx(0.5)


def _flatten_paths(value: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[tuple[str, ...], object] = {}
    for key, child in value.items():
        flattened.update(_flatten_paths(child, (*prefix, str(key))))
    return flattened


def test_exp03_configs_only_change_intended_paths() -> None:
    base_config = _load_base_config()
    configs = build_exp03_configs(base_config, include_phase_b=True)
    base_paths = _flatten_paths(base_config)
    allowed_changed_paths = {
        ("run", "run_id"),
        ("refiner", "checkpoint_path"),
        ("refiner", "monitor_metric"),
        ("refiner", "topology_validation", "enabled"),
        ("refiner", "topology_validation", "losses", "alpha"),
        ("refiner", "topology_validation", "losses", "beta"),
        ("refiner", "topology_validation", "losses", "gamma"),
        ("refiner", "topology_validation", "losses", "delta"),
        ("refiner", "topology_training", "enabled"),
        ("refiner", "topology_training", "topo_only_after_epoch"),
        ("refiner", "topology_training", "topology_weight"),
        ("refiner", "topology_training", "weights", "alpha"),
        ("refiner", "topology_training", "weights", "beta"),
        ("refiner", "topology_training", "weights", "gamma"),
        ("refiner", "topology_training", "weights", "delta"),
        ("refiner", "loss", "pos_weight"),
        ("graph_selection", "refined_output_rule", "type"),
        ("graph_selection", "refined_output_rule", "objective"),
        ("graph_selection", "refined_output_rule", "grid"),
        ("graph_selection", "rules"),
    }
    for run_id, config in configs.items():
        changed_paths = {
            path for path, value in _flatten_paths(config).items() if base_paths.get(path) != value
        }
        assert changed_paths <= allowed_changed_paths, run_id


def test_generated_exp03_configs_parse_existing_tccig_helpers(tmp_path: Path) -> None:
    output_dir = tmp_path / "configs"
    paths = write_exp03_configs(
        base_config_path=BASE_CONFIG,
        output_dir=output_dir,
        include_phase_b=True,
    )

    assert {path.name for path in paths} == {
        f"{run_id}.yaml" for run_id in (*PHASE_A_RUN_IDS, *PHASE_B_CONFIG_RUN_IDS)
    }
    for path in paths:
        config = tccig_train._load_yaml_config(path)
        parsed_rule = tccig_train._resolve_refined_output_rule_config(config)
        assert parsed_rule.calibrated is True
        refiner_config = dict(config["refiner"])  # type: ignore[arg-type]
        refiner_config["_run_id"] = _run_id(config)
        refiner_config["_log_root"] = "logs"
        parsed_refiner = _parse_config(refiner_config)
        assert parsed_refiner.monitor_metric == "val_topology_loss"
        assert parsed_refiner.topology_validation.losses.beta == pytest.approx(8.0)


def test_committed_exp03_configs_match_generator_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "configs"
    generated_paths = write_exp03_configs(
        base_config_path=BASE_CONFIG,
        output_dir=output_dir,
        include_phase_b=True,
    )

    generated_by_name = {path.name: path for path in generated_paths}
    for run_id in (*PHASE_A_RUN_IDS, *PHASE_B_CONFIG_RUN_IDS):
        file_name = f"{run_id}.yaml"
        generated_path = generated_by_name[file_name]
        committed_path = COMMITTED_CONFIG_DIR / file_name

        with generated_path.open("r", encoding="utf-8") as handle:
            generated_payload = yaml.safe_load(handle)
        with committed_path.open("r", encoding="utf-8") as handle:
            committed_payload = yaml.safe_load(handle)

        assert committed_payload == generated_payload
        assert committed_path.read_text(encoding="utf-8") == generated_path.read_text(
            encoding="utf-8"
        )
