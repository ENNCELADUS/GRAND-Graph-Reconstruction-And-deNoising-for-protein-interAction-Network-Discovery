"""Generate exp03 TCCIG loss-conflict diagnostic configs."""

from __future__ import annotations

import argparse
import copy
import logging
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import cast

import yaml

LOGGER = logging.getLogger(__name__)

EXP03_CALIBRATED_GRID = (0.5, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99)
FIXED_VALIDATION_LOSSES = {"alpha": 1.0, "beta": 8.0, "gamma": 0.5, "delta": 0.0}
PHASE_A_RUN_IDS = (
    "03_a1_bce_only",
    "03_a2_bce_graph_sim",
    "03_a3_bce_density",
    "03_a4_bce_degree",
    "03_a5_bce_full_topology",
)
PHASE_B_CONFIG_RUN_IDS = (
    "03_b1_beta2",
    "03_b2_beta4",
    "03_b3_topology_weight_0p5",
    "03_b4_topology_weight_2p0",
    "03_b5_bce_pos_weight_0p5",
)


def build_exp03_configs(
    base_config: Mapping[str, object],
    *,
    include_phase_b: bool,
) -> dict[str, dict[str, object]]:
    """Build exp03 config payloads from one base config."""
    run_ids = list(PHASE_A_RUN_IDS)
    if include_phase_b:
        run_ids.extend(PHASE_B_CONFIG_RUN_IDS)

    configs: dict[str, dict[str, object]] = {}
    for run_id in run_ids:
        config = _build_base_variant(base_config, run_id)
        _apply_variant(config, run_id)
        configs[run_id] = config
    return configs


def write_exp03_configs(
    *,
    base_config_path: Path,
    output_dir: Path,
    include_phase_b: bool,
) -> list[Path]:
    """Write exp03 YAML configs and return the generated paths."""
    with base_config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML config must be a mapping: {base_config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    configs = build_exp03_configs(
        cast(Mapping[str, object], payload), include_phase_b=include_phase_b
    )
    for run_id, config in configs.items():
        path = output_dir / f"{run_id}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        paths.append(path)
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """Run the exp03 config generator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-b", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in write_exp03_configs(
        base_config_path=args.base,
        output_dir=args.output_dir,
        include_phase_b=args.phase_b,
    ):
        LOGGER.info("%s", path)


def _build_base_variant(base_config: Mapping[str, object], run_id: str) -> dict[str, object]:
    config = copy.deepcopy(dict(base_config))
    run = _section(config, "run")
    refiner = _section(config, "refiner")
    topology_validation = _section(refiner, "topology_validation")
    graph_selection = _section(config, "graph_selection")

    run["run_id"] = run_id
    refiner["checkpoint_path"] = f"models/tccig/s2gae/{run_id}/best_model.pt"
    refiner["monitor_metric"] = "val_topology_loss"
    topology_validation["enabled"] = True
    topology_validation["losses"] = dict(FIXED_VALIDATION_LOSSES)
    graph_selection["refined_output_rule"] = {
        "type": "calibrated",
        "objective": "val_topology_loss",
        "grid": list(EXP03_CALIBRATED_GRID),
    }
    graph_selection["rules"] = [{"type": "threshold", "value": 0.5}]
    return config


def _apply_variant(config: dict[str, object], run_id: str) -> None:
    topology_training = _section(_section(config, "refiner"), "topology_training")
    topology_training["enabled"] = True
    topology_training["topo_only_after_epoch"] = None
    topology_training["topology_weight"] = 1.0
    topology_training["weights"] = dict(FIXED_VALIDATION_LOSSES)
    _section(_section(config, "refiner"), "loss")["pos_weight"] = 1.0

    if run_id == "03_a1_bce_only":
        topology_training["enabled"] = False
        topology_training["topology_weight"] = 0.0
    elif run_id == "03_a2_bce_graph_sim":
        topology_training["weights"] = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}
    elif run_id == "03_a3_bce_density":
        topology_training["weights"] = {"alpha": 0.0, "beta": 8.0, "gamma": 0.0, "delta": 0.0}
    elif run_id == "03_a4_bce_degree":
        topology_training["weights"] = {"alpha": 0.0, "beta": 0.0, "gamma": 0.5, "delta": 0.0}
    elif run_id == "03_b1_beta2":
        topology_training["weights"] = {"alpha": 1.0, "beta": 2.0, "gamma": 0.5, "delta": 0.0}
    elif run_id == "03_b2_beta4":
        topology_training["weights"] = {"alpha": 1.0, "beta": 4.0, "gamma": 0.5, "delta": 0.0}
    elif run_id == "03_b3_topology_weight_0p5":
        topology_training["topology_weight"] = 0.5
    elif run_id == "03_b4_topology_weight_2p0":
        topology_training["topology_weight"] = 2.0
    elif run_id == "03_b5_bce_pos_weight_0p5":
        _section(_section(config, "refiner"), "loss")["pos_weight"] = 0.5


def _section(
    config: MutableMapping[str, object],
    name: str,
) -> MutableMapping[str, object]:
    value = config[name]
    if not isinstance(value, MutableMapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(MutableMapping[str, object], value)


if __name__ == "__main__":
    main()
