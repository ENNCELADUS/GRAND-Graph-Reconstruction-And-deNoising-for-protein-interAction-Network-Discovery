# TCCIG Exp03 Loss-Conflict Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the config, artifact, analysis, and launch-support surface needed to run exp03 as a validation-first BCE-vs-topology diagnostic.

**Architecture:** Keep the model and topology metric definitions unchanged. Add a validation-only train mode so Phase A/B can avoid held-out test artifact generation until a candidate is locked, add a small exp03 config generator so Phase A/B variants are auditable, persist the full calibrated threshold grid that the validation loop already computes, isolate raw pairwise topology-baseline artifacts from refined topology-test artifacts, and add a validation-first analysis script that hides held-out test fields unless a locked candidate is named.

**Tech Stack:** Python 3.10+, PyTorch, Accelerate, NetworkX, PyYAML, pytest, ruff, mypy, Slurm via `scripts/tccig.sh`, repository commands through `uv run` and `rtk`.

---

## Scope Check

The exp03 spec covers one coupled subsystem: standalone TCCIG experiment execution for loss-interaction diagnosis. It spans config materialization, validation-grid artifact persistence, result analysis, and one gated sampler lever. Keeping it in one plan is appropriate because every task supports the same experiment contract and every deliverable is testable without launching the full HPC runs.

This plan does not change validation/test topology metric definitions, the pairwise scorer, ESM caches, or the S2GAE architecture. Phase A and config-only Phase B support are required. The explicit FN/FP hard-quadrant sampler task is gated: execute it only if Phase A analysis selects that lever.

## Fixed Launch Criteria

- Exp02 reference artifact path: `artifacts/exp02_rerun_fix/logs/tccig/02_balanced_subset`.
- Exp02 best validation AUPRC from that artifact: `0.6842201205393263`.
- Phase B validation AUPRC tolerance: no more than a 2% relative drop from exp02, so a candidate must keep `val_auprc >= 0.6705` unless the run report explicitly rejects it before held-out testing.
- Final held-out test metrics are generated only after a candidate is locked by training and validation evidence.
- Phase A/B Slurm launches must set `GRAND_TCCIG_SKIP_TEST_SPLITS=1`; any pairwise/topology test outputs produced before candidate lock are development diagnostics and cannot be cited as clean held-out evidence.

## File Structure

- Modify `tccig/train.py`
  - Add a default-preserving validation-only mode that skips `pairwise_test` and `topology_test` artifact generation.
- Modify `scripts/tccig.sh`
  - Pass the validation-only mode when `GRAND_TCCIG_SKIP_TEST_SPLITS=1`.
- Modify `tests/integration/test_tccig_orchestrator.py`
  - Prove validation-only runs write training/validation artifacts but no held-out pairwise/topology test artifacts.
- Create `tccig/exp03_configs.py`
  - Single owner for generating exp03 YAML configs from `configs/tccig/02_balanced_subset.yaml`.
  - Emits Phase A configs `03_a1` through `03_a5` and config-only Phase B candidate configs `03_b1` through `03_b5`; the operator still launches at most four Phase B runs before review.
  - Keeps `refiner.topology_validation.losses` fixed while changing only training pressure.
- Create `tests/unit/test_tccig_exp03_configs.py`
  - Proves exp03 configs keep calibrated validation fixed and only change intended knobs.
  - Proves generated YAML files parse through existing TCCIG parser helpers.
- Modify `tccig/s2gae.py`
  - Extend validation-grid evaluation to retain every threshold-row metric.
  - Write `logs/tccig/{run_id}/threshold_grid/epoch_XXX.json` and `best_epoch.json`.
- Modify `tests/unit/test_tccig_topology_training.py`
  - Extend existing calibrated-grid unit coverage for all retained threshold rows.
- Modify `tests/integration/test_tccig_orchestrator.py`
  - Add tiny calibrated pipeline assertion that threshold-grid artifacts exist.
- Create `tccig/analyze_exp03.py`
  - Reads training summaries, threshold-grid artifacts, optional held-out outputs, explicit raw-topology baseline artifacts, and exp02 reference artifacts.
  - Writes validation-first `exp03_summary.json`, `exp03_summary.csv`, and `exp03_summary.md`.
- Create `tests/unit/test_tccig_exp03_analysis.py`
  - Uses temporary artifact fixtures to prove validation fields, threshold-grid fields, and held-out gating.
- Modify `tccig/test.py`
  - Add an explicit raw-baseline output directory parameter so raw baseline files cannot overwrite refined `topology_test` files.
- Modify `tccig/raw_pairwise_topology_baseline.py`
  - Use the separate raw-baseline output directory by default.
- Modify `tests/unit/test_tccig_test_export.py`
  - Cover the new raw-baseline output directory behavior while preserving function back-compat.
- Modify `tccig/prepare.py`
  - Gated task only: add explicit FN/FP hard-quadrant sampling ratio.
- Modify `tests/unit/test_tccig_prepare.py`
  - Gated task only: prove default all-FP/all-FN behavior is preserved and configured downsampling is deterministic.
- Create `docs/experiment/tccig/exp03_runbook.md`
  - Operator-facing launch, collection, analysis, Phase B gate, and held-out-test protocol.

## Task 0: Add Validation-Only TCCIG Train Mode

**Files:**
- Modify: `tccig/train.py`
- Modify: `scripts/tccig.sh`
- Modify: `tests/integration/test_tccig_orchestrator.py`

- [ ] **Step 1: Write the failing validation-only orchestrator test**

Append this test to `tests/integration/test_tccig_orchestrator.py`:

```python
def test_tccig_orchestrator_can_skip_heldout_test_artifacts(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, "tiny_validation_only")

    result = run_tccig_pipeline(config, run_test_splits=False)

    run_log_dir = tmp_path / "logs" / "tccig" / "tiny_validation_only"
    manifest = json.loads((run_log_dir / "manifest.json").read_text(encoding="utf-8"))
    assert result.pairwise_metrics == {}
    assert result.topology_metrics == {}
    assert manifest["test_splits_skipped"] is True
    assert (run_log_dir / "training_summary.json").exists()
    assert not (run_log_dir / "pairwise_test").exists()
    assert not (run_log_dir / "topology_test").exists()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/integration/test_tccig_orchestrator.py::test_tccig_orchestrator_can_skip_heldout_test_artifacts -v
```

Expected: FAIL with `TypeError: run_tccig_pipeline() got an unexpected keyword argument 'run_test_splits'`.

- [ ] **Step 3: Implement the default-preserving skip mode**

In `tccig/train.py`, update the function signature:

```python
def run_tccig_pipeline(
    config: Mapping[str, object],
    *,
    build_accelerator_fn: AcceleratorFactory | None = None,
    run_test_splits: bool = True,
) -> TCCIGPipelineResult:
```

After `refined_output_rule = _effective_refined_output_rule(...)`, gate held-out test execution:

```python
    pairwise_metrics: dict[str, float] = {}
    topology_metrics: dict[str, float] = {}
    if run_test_splits:
        pairwise_metrics = tccig_test.run_pairwise_test(
            table=tables["pairwise_test"],
            scorer_cfg=scorer_cfg,
            refiner_cfg=refiner_cfg,
            runtime=runtime,
            cache_dir=cache_dir,
            log_dir=log_dir,
            refiner_state=refiner_state,
            pairwise_input_rule=pairwise_input_rule,
            refined_output_rule=refined_output_rule,
            score_split_fn=_score_split,
        )
        topology_metrics = tccig_test.run_topology_test(
            table=tables["topology_test"],
            processed_dir=processed_dir,
            scorer_cfg=scorer_cfg,
            refiner_cfg=refiner_cfg,
            runtime=runtime,
            cache_dir=cache_dir,
            log_dir=log_dir,
            refiner_state=refiner_state,
            pairwise_input_rule=pairwise_input_rule,
            refined_output_rule=refined_output_rule,
            pairwise_input_payload=pairwise_input_rule.to_dict(),
            score_split_fn=_score_split,
        )
```

Remove the old unconditional `run_pairwise_test` and `run_topology_test` calls. Add the skip marker to the manifest:

```python
        "test_splits_skipped": not run_test_splits,
```

In `main`, add:

```python
    parser.add_argument(
        "--skip-test-splits",
        action="store_true",
        help="train and validate only; do not generate held-out pairwise/topology test artifacts",
    )
```

and call:

```python
    run_tccig_pipeline(
        _load_yaml_config(Path(args.config)),
        run_test_splits=not bool(args.skip_test_splits),
    )
```

In `scripts/tccig.sh`, build optional CLI arguments before the GPU branch:

```bash
TRAIN_EXTRA_ARGS=()
if [ "${GRAND_TCCIG_SKIP_TEST_SPLITS:-0}" = "1" ]; then
  TRAIN_EXTRA_ARGS+=(--skip-test-splits)
fi
```

Pass `"${TRAIN_EXTRA_ARGS[@]}"` after `--config "$CONFIG_PATH"` in both the Accelerate and CPU launch commands.

- [ ] **Step 4: Run focused validation-only and default-behavior tests**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/integration/test_tccig_orchestrator.py::test_tccig_orchestrator_can_skip_heldout_test_artifacts tests/integration/test_tccig_orchestrator.py::test_tccig_orchestrator_runs_concrete_pipeline_and_writes_artifacts -v
```

Expected: PASS. The new mode skips held-out artifacts only when explicitly requested; existing callers still run pairwise/topology tests by default.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add tccig/train.py scripts/tccig.sh tests/integration/test_tccig_orchestrator.py
rtk git commit -m "feat: add validation-only tccig train mode"
```

Expected: commit succeeds with only the files listed above.

## Task 1: Generate Exp03 Configs From One Base Config

**Files:**
- Create: `tccig/exp03_configs.py`
- Create: `tests/unit/test_tccig_exp03_configs.py`
- Create after tests pass: `configs/tccig/exp03/03_a1_bce_only.yaml`
- Create after tests pass: `configs/tccig/exp03/03_a2_bce_graph_sim.yaml`
- Create after tests pass: `configs/tccig/exp03/03_a3_bce_density.yaml`
- Create after tests pass: `configs/tccig/exp03/03_a4_bce_degree.yaml`
- Create after tests pass: `configs/tccig/exp03/03_a5_bce_full_topology.yaml`
- Create after tests pass: `configs/tccig/exp03/03_b1_beta2.yaml`
- Create after tests pass: `configs/tccig/exp03/03_b2_beta4.yaml`
- Create after tests pass: `configs/tccig/exp03/03_b3_topology_weight_0p5.yaml`
- Create after tests pass: `configs/tccig/exp03/03_b4_topology_weight_2p0.yaml`
- Create after tests pass: `configs/tccig/exp03/03_b5_bce_pos_weight_0p5.yaml`

- [ ] **Step 1: Write the failing config-generator tests**

Create `tests/unit/test_tccig_exp03_configs.py`:

```python
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
    assert _topology_training(configs["03_b3_topology_weight_0p5"])[
        "topology_weight"
    ] == pytest.approx(0.5)
    assert _topology_training(configs["03_b4_topology_weight_2p0"])[
        "topology_weight"
    ] == pytest.approx(2.0)
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
            path
            for path, value in _flatten_paths(config).items()
            if base_paths.get(path) != value
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
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_exp03_configs.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tccig.exp03_configs'`.

- [ ] **Step 3: Implement the config generator**

Create `tccig/exp03_configs.py`:

```python
"""Generate exp03 TCCIG loss-conflict diagnostic configs."""

from __future__ import annotations

import argparse
import copy
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

LOGGER = logging.getLogger(__name__)

EXP03_CALIBRATED_GRID: tuple[float, ...] = (
    0.5,
    0.7,
    0.8,
    0.85,
    0.9,
    0.925,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
)
FIXED_VALIDATION_LOSSES: dict[str, float] = {
    "alpha": 1.0,
    "beta": 8.0,
    "gamma": 0.5,
    "delta": 0.0,
}
PHASE_A_RUN_IDS: tuple[str, ...] = (
    "03_a1_bce_only",
    "03_a2_bce_graph_sim",
    "03_a3_bce_density",
    "03_a4_bce_degree",
    "03_a5_bce_full_topology",
)
PHASE_B_CONFIG_RUN_IDS: tuple[str, ...] = (
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
    """Return exp03 configs keyed by run id."""
    configs: dict[str, dict[str, object]] = {}

    configs["03_a1_bce_only"] = _variant(
        base_config,
        run_id="03_a1_bce_only",
        topology_training_enabled=False,
        weights=FIXED_VALIDATION_LOSSES,
        topology_weight=0.0,
        pos_weight=1.0,
    )
    configs["03_a2_bce_graph_sim"] = _variant(
        base_config,
        run_id="03_a2_bce_graph_sim",
        topology_training_enabled=True,
        weights={"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
        topology_weight=1.0,
        pos_weight=1.0,
    )
    configs["03_a3_bce_density"] = _variant(
        base_config,
        run_id="03_a3_bce_density",
        topology_training_enabled=True,
        weights={"alpha": 0.0, "beta": 8.0, "gamma": 0.0, "delta": 0.0},
        topology_weight=1.0,
        pos_weight=1.0,
    )
    configs["03_a4_bce_degree"] = _variant(
        base_config,
        run_id="03_a4_bce_degree",
        topology_training_enabled=True,
        weights={"alpha": 0.0, "beta": 0.0, "gamma": 0.5, "delta": 0.0},
        topology_weight=1.0,
        pos_weight=1.0,
    )
    configs["03_a5_bce_full_topology"] = _variant(
        base_config,
        run_id="03_a5_bce_full_topology",
        topology_training_enabled=True,
        weights=FIXED_VALIDATION_LOSSES,
        topology_weight=1.0,
        pos_weight=1.0,
    )

    if include_phase_b:
        configs["03_b1_beta2"] = _variant(
            base_config,
            run_id="03_b1_beta2",
            topology_training_enabled=True,
            weights={"alpha": 1.0, "beta": 2.0, "gamma": 0.5, "delta": 0.0},
            topology_weight=1.0,
            pos_weight=1.0,
        )
        configs["03_b2_beta4"] = _variant(
            base_config,
            run_id="03_b2_beta4",
            topology_training_enabled=True,
            weights={"alpha": 1.0, "beta": 4.0, "gamma": 0.5, "delta": 0.0},
            topology_weight=1.0,
            pos_weight=1.0,
        )
        configs["03_b3_topology_weight_0p5"] = _variant(
            base_config,
            run_id="03_b3_topology_weight_0p5",
            topology_training_enabled=True,
            weights=FIXED_VALIDATION_LOSSES,
            topology_weight=0.5,
            pos_weight=1.0,
        )
        configs["03_b4_topology_weight_2p0"] = _variant(
            base_config,
            run_id="03_b4_topology_weight_2p0",
            topology_training_enabled=True,
            weights=FIXED_VALIDATION_LOSSES,
            topology_weight=2.0,
            pos_weight=1.0,
        )
        configs["03_b5_bce_pos_weight_0p5"] = _variant(
            base_config,
            run_id="03_b5_bce_pos_weight_0p5",
            topology_training_enabled=True,
            weights=FIXED_VALIDATION_LOSSES,
            topology_weight=1.0,
            pos_weight=0.5,
        )

    return configs


def write_exp03_configs(
    *,
    base_config_path: Path,
    output_dir: Path,
    include_phase_b: bool,
) -> list[Path]:
    """Materialize exp03 YAML files and return written paths."""
    base_config = _load_yaml(base_config_path)
    configs = build_exp03_configs(base_config, include_phase_b=include_phase_b)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for run_id, config in configs.items():
        path = output_dir / f"{run_id}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def _variant(
    base_config: Mapping[str, object],
    *,
    run_id: str,
    topology_training_enabled: bool,
    weights: Mapping[str, float],
    topology_weight: float,
    pos_weight: float,
) -> dict[str, object]:
    config = cast(dict[str, object], copy.deepcopy(dict(base_config)))
    _set_run_id(config, run_id)
    _set_common_validation_contract(config)

    refiner = _mapping(config, "refiner")
    topology_training = _mapping(refiner, "topology_training")
    topology_training["enabled"] = topology_training_enabled
    topology_training["topo_only_after_epoch"] = None
    topology_training["topology_weight"] = float(topology_weight)
    topology_training["weights"] = {name: float(value) for name, value in weights.items()}

    loss = _mapping(refiner, "loss")
    loss["pos_weight"] = float(pos_weight)
    return config


def _set_run_id(config: dict[str, object], run_id: str) -> None:
    run = _mapping(config, "run")
    run["run_id"] = run_id
    refiner = _mapping(config, "refiner")
    refiner["checkpoint_path"] = f"models/tccig/s2gae/{run_id}/best_model.pt"


def _set_common_validation_contract(config: dict[str, object]) -> None:
    refiner = _mapping(config, "refiner")
    refiner["monitor_metric"] = "val_topology_loss"

    topology_validation = _mapping(refiner, "topology_validation")
    topology_validation["enabled"] = True
    topology_validation["losses"] = dict(FIXED_VALIDATION_LOSSES)

    graph_selection = _mapping(config, "graph_selection")
    graph_selection["refined_output_rule"] = {
        "type": "calibrated",
        "objective": "val_topology_loss",
        "grid": list(EXP03_CALIBRATED_GRID),
    }
    graph_selection["rules"] = [{"type": "threshold", "value": 0.5}]


def _mapping(config: Mapping[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return cast(dict[str, object], value)


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return cast(dict[str, object], payload)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for generating exp03 config files."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Generate exp03 TCCIG configs")
    parser.add_argument("--base", type=Path, default=Path("configs/tccig/02_balanced_subset.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("configs/tccig/exp03"))
    parser.add_argument(
        "--phase-b",
        action="store_true",
        help="also write config-only Phase B variants",
    )
    args = parser.parse_args(argv)
    for path in write_exp03_configs(
        base_config_path=args.base,
        output_dir=args.output_dir,
        include_phase_b=bool(args.phase_b),
    ):
        LOGGER.info("%s", path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_exp03_configs.py -v
```

Expected: PASS.

- [ ] **Step 5: Generate committed exp03 config files**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m tccig.exp03_configs --base configs/tccig/02_balanced_subset.yaml --output-dir configs/tccig/exp03 --phase-b
```

Expected: writes ten paths under `configs/tccig/exp03/` and logs each path without using `print`.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add tccig/exp03_configs.py tests/unit/test_tccig_exp03_configs.py configs/tccig/exp03
rtk git commit -m "feat: add exp03 tccig config generator"
```

Expected: commit succeeds with only the files listed above.

## Task 2: Persist Per-Epoch Threshold-Grid Artifacts

**Files:**
- Modify: `tccig/s2gae.py`
- Modify: `tests/unit/test_tccig_topology_training.py`
- Modify: `tests/integration/test_tccig_orchestrator.py`

- [ ] **Step 1: Write failing unit coverage for retained grid rows**

In `tests/unit/test_tccig_topology_training.py`, extend `test_validation_topology_evaluation_selects_grid_argmin` after the current `assert result.rule_payload == ...` block:

```python
    assert result.rule_grid == (
        {
            "rule": {"type": "threshold", "value": 0.5, "source": "validation_calibration"},
            "val_topology_loss": 10.0,
            "graph_sim": 0.6,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.0,
            "cc_mmd": 0.0,
            "positive_edges": 50,
            "val_auprc": 0.42,
        },
        {
            "rule": {"type": "threshold", "value": 0.9, "source": "validation_calibration"},
            "val_topology_loss": 2.0,
            "graph_sim": 1.0,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.0,
            "cc_mmd": 0.0,
            "positive_edges": 90,
            "val_auprc": 0.42,
        },
        {
            "rule": {"type": "threshold", "value": 0.97, "source": "validation_calibration"},
            "val_topology_loss": 5.0,
            "graph_sim": 1.07,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.0,
            "cc_mmd": 0.0,
            "positive_edges": 97,
            "val_auprc": 0.42,
        },
    )
```

Add this integration test to `tests/integration/test_tccig_orchestrator.py` after `test_calibrated_pipeline_persists_epoch_selected_rule_history`:

```python
def test_calibrated_pipeline_writes_threshold_grid_artifacts(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, "calibrated_grid_artifacts")
    refiner_config = config["refiner"]
    assert isinstance(refiner_config, dict)
    refiner_config["monitor_metric"] = "val_topology_loss"
    refiner_config["topology_validation"] = {
        "enabled": True,
        "node_sizes": [2],
        "samples_per_size": 1,
        "strategy": "mixed",
        "seed": 0,
        "inference_batch_size": 4,
        "compute_clustering_mmd": False,
        "losses": {"alpha": 1.0, "beta": 1.0, "gamma": 0.0, "delta": 0.0},
    }
    graph_selection = config["graph_selection"]
    assert isinstance(graph_selection, dict)
    graph_selection["refined_output_rule"] = {
        "type": "calibrated",
        "objective": "val_topology_loss",
        "grid": [0.5, 0.97],
    }

    run_tccig_pipeline(config)

    grid_dir = tmp_path / "logs" / "tccig" / "calibrated_grid_artifacts" / "threshold_grid"
    epoch_payload = json.loads((grid_dir / "epoch_001.json").read_text(encoding="utf-8"))
    best_payload = json.loads((grid_dir / "best_epoch.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (
            tmp_path
            / "logs"
            / "tccig"
            / "calibrated_grid_artifacts"
            / "training_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert epoch_payload["epoch"] == 1
    assert len(epoch_payload["rows"]) == 2
    assert {row["rule"]["value"] for row in epoch_payload["rows"]} == {0.5, 0.97}
    assert epoch_payload["selected_rule"] == summary["history"][0]["selected_rule"]
    assert best_payload == epoch_payload
    assert summary["history"][0]["threshold_grid_path"] == "threshold_grid/epoch_001.json"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_validation_topology_evaluation_selects_grid_argmin tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_writes_threshold_grid_artifacts -v
```

Expected: FAIL because `ValidationTopologyRuleEvaluation` has no `rule_grid` field and no threshold-grid files are written.

- [ ] **Step 3: Retain grid rows in the validation evaluator**

In `tccig/s2gae.py`, replace `ValidationTopologyRuleEvaluation` with:

```python
@dataclass(frozen=True)
class ValidationTopologyRuleEvaluation:
    """Topology validation result for one selected hard-graph rule."""

    rule: GraphRule
    validation_metrics: dict[str, float | int]
    rule_payload: Mapping[str, object]
    rule_grid: tuple[dict[str, object], ...]
```

Before running the new assertions, update every existing `ValidationTopologyRuleEvaluation(...)` construction site in `tccig/s2gae.py` and `tests/unit/test_tccig_topology_training.py`. Existing test doubles that do not care about retained grid rows should pass `rule_grid=()`. Do not leave any positional call ambiguous; use keyword arguments at every construction site.

In `_evaluate_validation_topology_rules`, change the evaluation loop to retain row payloads:

```python
    evaluations: list[ValidationTopologyRuleEvaluation] = []
    grid_rows: list[dict[str, object]] = []
    for rule in rules:
        metrics = _validation_topology_metrics(
            validation_plan=validation_plan,
            pairs=pairs,
            probabilities=refined_probabilities,
            rule=rule,
            validation_auprc=validation_auprc,
            cfg=cfg,
        )
        payload: dict[str, object] = dict(rule.to_dict())
        if rule_payload_source is not None:
            payload["source"] = rule_payload_source
        row: dict[str, object] = {"rule": dict(payload)}
        row.update(metrics)
        grid_rows.append(row)
        evaluations.append(
            ValidationTopologyRuleEvaluation(
                rule=rule,
                validation_metrics=metrics,
                rule_payload=payload,
                rule_grid=(),
            )
        )
    selected = min(
        evaluations,
        key=lambda item: float(item.validation_metrics["val_topology_loss"]),
    )
    return ValidationTopologyRuleEvaluation(
        rule=selected.rule,
        validation_metrics=selected.validation_metrics,
        rule_payload=selected.rule_payload,
        rule_grid=tuple(grid_rows),
    )
```

- [ ] **Step 4: Write threshold-grid JSON artifacts from the train loop**

Add these helpers near `_write_training_summary` in `tccig/s2gae.py`:

```python
def _threshold_grid_relative_path(epoch: int) -> Path:
    return Path("threshold_grid") / f"epoch_{epoch:03d}.json"


def _write_threshold_grid_artifact(
    *,
    log_dir: Path,
    relative_path: Path,
    epoch: int,
    selected_rule_payload: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> None:
    write_json(
        log_dir / relative_path,
        {
            "epoch": int(epoch),
            "selected_rule": dict(selected_rule_payload),
            "rows": [dict(row) for row in rows],
        },
    )
```

In `train_refiner`, add a local before topology validation:

```python
        selected_epoch_threshold_grid: tuple[dict[str, object], ...] | None = None
```

After `selected_epoch_rule_payload = dict(topology_evaluation.rule_payload)`, add:

```python
            selected_epoch_threshold_grid = topology_evaluation.rule_grid
```

Inside the `if selected_epoch_topology_metrics is not None:` history update block, add:

```python
                    "threshold_grid_path": str(_threshold_grid_relative_path(epoch)),
```

Replace the best-check block with an `is_best_epoch` local so the grid artifact and checkpoint selection agree:

```python
        is_best_epoch = best_state_dict is None or _is_better_monitor(
            value=monitor_value,
            best_value=best_monitor_value,
            monitor_metric=cfg.monitor_metric,
        )
        if request.runtime.is_main_process and selected_epoch_threshold_grid is not None:
            relative_path = _threshold_grid_relative_path(epoch)
            assert selected_epoch_rule_payload is not None
            _write_threshold_grid_artifact(
                log_dir=cfg.log_dir,
                relative_path=relative_path,
                epoch=epoch,
                selected_rule_payload=selected_epoch_rule_payload,
                rows=selected_epoch_threshold_grid,
            )
            if is_best_epoch:
                _write_threshold_grid_artifact(
                    log_dir=cfg.log_dir,
                    relative_path=Path("threshold_grid") / "best_epoch.json",
                    epoch=epoch,
                    selected_rule_payload=selected_epoch_rule_payload,
                    rows=selected_epoch_threshold_grid,
                )

        if is_best_epoch:
            best_monitor_value = monitor_value
            best_validation_auprc = validation_auprc
            best_selected_rule = selected_epoch_rule
            best_selected_rule_payload = selected_epoch_rule_payload
            checkpoint_model = _unwrap_refiner(train_step_model, request.runtime.accelerator)
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in checkpoint_model.state_dict().items()
            }
```

- [ ] **Step 5: Run focused tests and verify they pass**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py::test_validation_topology_evaluation_selects_grid_argmin tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_writes_threshold_grid_artifacts -v
```

Expected: PASS.

- [ ] **Step 6: Run the full topology-training unit file**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_topology_training.py -v
```

Expected: PASS. This catches stale `ValidationTopologyRuleEvaluation(...)` test doubles that do not execute in the single focused test.

- [ ] **Step 7: Commit**

Run:

```bash
rtk git add tccig/s2gae.py tests/unit/test_tccig_topology_training.py tests/integration/test_tccig_orchestrator.py
rtk git commit -m "feat: persist tccig validation threshold grids"
```

Expected: commit succeeds with only the files listed above.

## Task 3: Add Validation-First Exp03 Analysis Script

**Files:**
- Create: `tccig/analyze_exp03.py`
- Create: `tests/unit/test_tccig_exp03_analysis.py`

- [ ] **Step 1: Write failing analysis tests**

Create `tests/unit/test_tccig_exp03_analysis.py`:

```python
"""Tests for exp03 validation-first analysis reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tccig.analyze_exp03 import EXP02_REFERENCE_RUN_ID, collect_run_row, write_exp03_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_run_fixture(root: Path, run_id: str, *, with_heldout: bool) -> Path:
    run_dir = root / run_id
    _write_json(
        run_dir / "training_summary.json",
        {
            "monitor_metric": "val_topology_loss",
            "selected_rule": {
                "type": "threshold",
                "value": 0.96,
                "source": "validation_calibration",
            },
            "history": [
                {
                    "epoch": 1,
                    "monitor_value": 9.0,
                    "val_auprc": 0.68,
                    "val_topology_loss": 9.0,
                    "internal_val_graph_sim": 0.70,
                    "internal_val_relative_density": 1.40,
                    "internal_val_deg_dist_mmd": 0.20,
                    "internal_val_cc_mmd": 0.10,
                    "selected_rule_positive_edges": 140,
                    "selected_rule": {
                        "type": "threshold",
                        "value": 0.95,
                        "source": "validation_calibration",
                    },
                    "train_bce_loss": 0.30,
                    "train_topology_loss": 4.5,
                    "train_topo_graph_sim": 0.40,
                    "train_topo_relative_density": 1.4,
                    "train_topo_degree_mmd": 0.2,
                    "sampled_edge_targets": 300,
                    "train_fp_targets": 10,
                    "train_fn_targets": 20,
                },
                {
                    "epoch": 40,
                    "monitor_value": 7.0,
                    "val_auprc": 0.675,
                    "val_topology_loss": 7.0,
                    "internal_val_graph_sim": 0.75,
                    "internal_val_relative_density": 1.08,
                    "internal_val_deg_dist_mmd": 0.15,
                    "internal_val_cc_mmd": 0.08,
                    "selected_rule_positive_edges": 120,
                    "selected_rule": {
                        "type": "threshold",
                        "value": 0.96,
                        "source": "validation_calibration",
                    },
                    "train_bce_loss": 0.25,
                    "train_topology_loss": 3.0,
                    "train_topo_graph_sim": 0.50,
                    "train_topo_relative_density": 1.1,
                    "train_topo_degree_mmd": 0.15,
                    "sampled_edge_targets": 300,
                    "train_fp_targets": 10,
                    "train_fn_targets": 20,
                },
            ],
        },
    )
    _write_json(
        run_dir / "threshold_grid" / "best_epoch.json",
        {
            "epoch": 40,
            "selected_rule": {
                "type": "threshold",
                "value": 0.96,
                "source": "validation_calibration",
            },
            "rows": [
                {"rule": {"type": "threshold", "value": 0.95}, "val_topology_loss": 8.0},
                {"rule": {"type": "threshold", "value": 0.96}, "val_topology_loss": 7.0},
            ],
        },
    )
    if with_heldout:
        _write_json(
            run_dir / "pairwise_test" / "refined_metrics.json",
            {"precision": 0.9, "recall": 0.2, "f1": 0.33, "auprc": 0.7, "auroc": 0.8},
        )
        _write_json(run_dir / "pairwise_test" / "raw_metrics.json", {"auprc": 0.69, "auroc": 0.79})
        _write_json(
            run_dir / "topology_test" / "topology_metrics.json",
            {
                "summary": {
                    "relative_density": 0.95,
                    "graph_sim": 0.8,
                    "deg_dist_mmd": 0.1,
                    "cc_mmd": 0.05,
                },
                "deletion_diagnostics": {"edges_added": 1.0, "edges_deleted": 2.0},
                "protocol": {
                    "candidate_universe": "all_test_ppi.txt",
                    "test_labels_visible_to_model": False,
                },
            },
        )
    return run_dir


def _write_raw_baseline_fixture(root: Path) -> Path:
    artifact_dir = root / "raw_pairwise_topology_baseline"
    _write_json(
        artifact_dir / "topology_metrics.json",
        {
            "summary": {
                "relative_density": 0.88,
                "graph_sim": 0.77,
                "deg_dist_mmd": 0.12,
                "cc_mmd": 0.07,
            },
            "protocol": {
                "candidate_universe": "all_test_ppi.txt",
                "test_labels_visible_to_model": False,
            },
        },
    )
    return artifact_dir


def test_collect_run_row_uses_training_and_threshold_grid_without_heldout(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path, "03_a5_bce_full_topology", with_heldout=True)

    row = collect_run_row(
        run_id="03_a5_bce_full_topology",
        run_dir=run_dir,
        include_heldout=False,
    )

    assert row["run_id"] == "03_a5_bce_full_topology"
    assert row["best_epoch"] == 40
    assert row["selected_threshold"] == 0.96
    assert row["validation_selected_edges"] == 120
    assert row["train_topology_loss_epoch_1"] == 4.5
    assert row["train_topology_loss_epoch_40"] == 3.0
    assert row["threshold_grid_rows"] == 2
    assert row["gate_val_auprc_ok"] is True
    assert row["gate_selected_edges_stable"] is True
    assert row["gate_density_closer_than_exp02"] is True
    assert row["eligible_for_locked_test"] is True
    assert "heldout_refined_precision" not in row


def test_collect_run_row_includes_heldout_only_when_enabled(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path, "03_locked", with_heldout=True)
    raw_baseline_artifact_dir = _write_raw_baseline_fixture(
        tmp_path / "03_locked_raw_pairwise_baseline"
    )

    row = collect_run_row(
        run_id="03_locked",
        run_dir=run_dir,
        include_heldout=True,
        raw_baseline_artifact_dir=raw_baseline_artifact_dir,
    )

    assert row["heldout_refined_precision"] == 0.9
    assert row["heldout_refined_auprc"] == 0.7
    assert row["heldout_raw_auprc"] == 0.69
    assert row["heldout_topology_relative_density"] == 0.95
    assert row["heldout_raw_topology_relative_density"] == 0.88
    assert row["heldout_protocol_candidate_universe"] == "all_test_ppi.txt"
    assert row["heldout_protocol_test_labels_visible_to_model"] is False
    assert row["heldout_edges_added"] == 1.0
    assert row["heldout_edges_deleted"] == 2.0


def test_write_exp03_report_outputs_json_csv_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path / "logs", "03_a5_bce_full_topology", with_heldout=False)
    reference_dir = _write_run_fixture(
        tmp_path / "logs",
        EXP02_REFERENCE_RUN_ID,
        with_heldout=False,
    )
    output_dir = tmp_path / "analysis"
    row = collect_run_row(
        run_id="03_a5_bce_full_topology",
        run_dir=run_dir,
        include_heldout=False,
    )
    reference_row = collect_run_row(
        run_id=EXP02_REFERENCE_RUN_ID,
        run_dir=reference_dir,
        include_heldout=False,
    )

    outputs = write_exp03_report(rows=[reference_row, row], output_dir=output_dir)

    assert outputs["json"].exists()
    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    candidate = next(
        item for item in payload["rows"] if item["run_id"] == "03_a5_bce_full_topology"
    )
    assert candidate["gate_non_worse_reference_metrics"] is True
    csv_rows = list(csv.DictReader(outputs["csv"].open("r", encoding="utf-8")))
    assert csv_rows[1]["run_id"] == "03_a5_bce_full_topology"
    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "Validation-first exp03 summary" in markdown
    assert "Held-out metrics were not included" in markdown
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_exp03_analysis.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tccig.analyze_exp03'`.

- [ ] **Step 3: Implement the analysis module**

Create `tccig/analyze_exp03.py`:

```python
"""Validation-first analysis for TCCIG exp03 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import SupportsFloat, SupportsInt, cast

from tccig.exp03_configs import PHASE_A_RUN_IDS, PHASE_B_CONFIG_RUN_IDS
from tccig.prepare import write_json

EXP02_REFERENCE_RUN_ID = "03_a0_exp02_topo_only_reference"
DEFAULT_EXP02_REFERENCE_DIR = Path("artifacts/exp02_rerun_fix/logs/tccig/02_balanced_subset")
DEFAULT_LOG_ROOT = Path("logs/tccig")
DEFAULT_OUTPUT_DIR = Path("analysis/tccig_exp03")
EXP02_REFERENCE_VAL_AUPRC = 0.6842201205393263
EXP02_REFERENCE_VAL_RELATIVE_DENSITY = 1.1537
VAL_AUPRC_FLOOR = 0.6705

BASE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "best_epoch",
    "selected_threshold",
    "validation_selected_edges",
    "validation_selected_edges_min",
    "validation_selected_edges_max",
    "validation_selected_edges_range",
    "val_topology_loss",
    "val_auprc",
    "internal_val_relative_density",
    "internal_val_graph_sim",
    "internal_val_deg_dist_mmd",
    "internal_val_cc_mmd",
    "train_bce_loss",
    "train_topology_loss_epoch_1",
    "train_topology_loss_epoch_7",
    "train_topology_loss_epoch_40",
    "sampled_edge_targets",
    "train_fp_targets",
    "train_fn_targets",
    "threshold_grid_rows",
    "threshold_grid_best_epoch",
    "val_auprc_floor",
    "gate_val_auprc_ok",
    "gate_selected_edges_stable",
    "gate_density_closer_than_exp02",
    "gate_threshold_grid_present",
    "delta_graph_sim_vs_exp02",
    "delta_deg_mmd_vs_exp02",
    "delta_cc_mmd_vs_exp02",
    "gate_non_worse_reference_metrics",
    "eligible_for_locked_test",
)
HELDOUT_COLUMNS: tuple[str, ...] = (
    "heldout_refined_precision",
    "heldout_refined_recall",
    "heldout_refined_f1",
    "heldout_refined_auprc",
    "heldout_refined_auroc",
    "heldout_raw_auprc",
    "heldout_raw_auroc",
    "heldout_topology_relative_density",
    "heldout_topology_graph_sim",
    "heldout_topology_degree_mmd",
    "heldout_topology_cc_mmd",
    "heldout_edges_added",
    "heldout_edges_deleted",
    "heldout_protocol_candidate_universe",
    "heldout_protocol_test_labels_visible_to_model",
    "heldout_raw_topology_relative_density",
    "heldout_raw_topology_graph_sim",
    "heldout_raw_topology_degree_mmd",
    "heldout_raw_topology_cc_mmd",
    "heldout_raw_protocol_candidate_universe",
    "heldout_raw_protocol_test_labels_visible_to_model",
)


def collect_run_row(
    *,
    run_id: str,
    run_dir: Path,
    include_heldout: bool,
    raw_baseline_artifact_dir: Path | None = None,
) -> dict[str, object]:
    """Collect one exp03 row from a run artifact directory."""
    summary = _load_json(run_dir / "training_summary.json")
    history = cast(list[Mapping[str, object]], summary.get("history", []))
    if not history:
        raise ValueError(f"{run_dir}/training_summary.json has empty history")
    best = _best_history_row(history)
    selected_rule = best.get("selected_rule") or summary.get("selected_rule") or {}
    if not isinstance(selected_rule, Mapping):
        selected_rule = {}
    grid = _load_optional_json(run_dir / "threshold_grid" / "best_epoch.json")

    row: dict[str, object] = {
        "run_id": run_id,
        "best_epoch": _as_int(best, "epoch"),
        "selected_threshold": _selected_threshold(selected_rule),
        "validation_selected_edges": _as_optional_int(best, "selected_rule_positive_edges"),
        "validation_selected_edges_min": _history_int_min(history, "selected_rule_positive_edges"),
        "validation_selected_edges_max": _history_int_max(history, "selected_rule_positive_edges"),
        "validation_selected_edges_range": _history_int_range(
            history,
            "selected_rule_positive_edges",
        ),
        "val_topology_loss": _as_optional_float(best, "val_topology_loss"),
        "val_auprc": _as_optional_float(best, "val_auprc"),
        "internal_val_relative_density": _as_optional_float(best, "internal_val_relative_density"),
        "internal_val_graph_sim": _as_optional_float(best, "internal_val_graph_sim"),
        "internal_val_deg_dist_mmd": _as_optional_float(best, "internal_val_deg_dist_mmd"),
        "internal_val_cc_mmd": _as_optional_float(best, "internal_val_cc_mmd"),
        "train_bce_loss": _as_optional_float(best, "train_bce_loss"),
        "train_topology_loss_epoch_1": _history_float(history, 1, "train_topology_loss"),
        "train_topology_loss_epoch_7": _history_float(history, 7, "train_topology_loss"),
        "train_topology_loss_epoch_40": _history_float(history, 40, "train_topology_loss"),
        "sampled_edge_targets": _as_optional_int(best, "sampled_edge_targets"),
        "train_fp_targets": _as_optional_int(best, "train_fp_targets"),
        "train_fn_targets": _as_optional_int(best, "train_fn_targets"),
        "threshold_grid_rows": _threshold_grid_row_count(grid),
        "threshold_grid_best_epoch": _threshold_grid_epoch(grid),
        "val_auprc_floor": VAL_AUPRC_FLOOR,
    }
    row.update(_validation_gate_fields(row))
    if include_heldout:
        row.update(_heldout_fields(run_dir))
        if raw_baseline_artifact_dir is not None:
            row.update(_raw_topology_baseline_fields(raw_baseline_artifact_dir))
    return row


def write_exp03_report(
    *,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> dict[str, Path]:
    """Write JSON, CSV, and Markdown exp03 reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "exp03_summary.json"
    csv_path = output_dir / "exp03_summary.csv"
    markdown_path = output_dir / "exp03_summary.md"
    normalized_rows = _attach_reference_deltas([dict(row) for row in rows])
    write_json(json_path, {"rows": normalized_rows, "val_auprc_floor": VAL_AUPRC_FLOOR})
    _write_csv(csv_path, normalized_rows)
    markdown_path.write_text(_markdown_report(normalized_rows), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def _best_history_row(history: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return min(history, key=lambda row: float(cast(SupportsFloat, row["monitor_value"])))


def _attach_reference_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    reference = next((row for row in rows if row.get("run_id") == EXP02_REFERENCE_RUN_ID), None)
    if reference is None:
        for row in rows:
            row["delta_graph_sim_vs_exp02"] = ""
            row["delta_deg_mmd_vs_exp02"] = ""
            row["delta_cc_mmd_vs_exp02"] = ""
            row["gate_non_worse_reference_metrics"] = ""
        return rows
    reference_graph_sim = _mapping_float(reference, "internal_val_graph_sim")
    reference_deg_mmd = _mapping_float(reference, "internal_val_deg_dist_mmd")
    reference_cc_mmd = _mapping_float(reference, "internal_val_cc_mmd")
    for row in rows:
        graph_delta = _float_delta(row, "internal_val_graph_sim", reference_graph_sim)
        deg_delta = _float_delta(row, "internal_val_deg_dist_mmd", reference_deg_mmd)
        cc_delta = _float_delta(row, "internal_val_cc_mmd", reference_cc_mmd)
        row["delta_graph_sim_vs_exp02"] = graph_delta
        row["delta_deg_mmd_vs_exp02"] = deg_delta
        row["delta_cc_mmd_vs_exp02"] = cc_delta
        row["gate_non_worse_reference_metrics"] = (
            isinstance(graph_delta, float)
            and graph_delta >= 0.0
            and isinstance(deg_delta, float)
            and deg_delta <= 0.0
            and isinstance(cc_delta, float)
            and cc_delta <= 0.0
        )
    return rows


def _float_delta(
    row: Mapping[str, object],
    key: str,
    reference_value: float | str,
) -> float | str:
    value = _mapping_float(row, key)
    if isinstance(value, float) and isinstance(reference_value, float):
        return value - reference_value
    return ""


def _selected_threshold(selected_rule: Mapping[str, object]) -> float | str:
    value = selected_rule.get("value")
    if isinstance(value, (float, int)):
        return float(value)
    return ""


def _threshold_grid_row_count(payload: Mapping[str, object] | None) -> int | str:
    if payload is None:
        return ""
    rows = payload.get("rows")
    if isinstance(rows, list):
        return len(rows)
    return ""


def _threshold_grid_epoch(payload: Mapping[str, object] | None) -> int | str:
    if payload is None:
        return ""
    value = payload.get("epoch")
    if isinstance(value, int):
        return value
    return ""


def _validation_gate_fields(row: Mapping[str, object]) -> dict[str, object]:
    val_auprc = row.get("val_auprc")
    selected_edges = row.get("validation_selected_edges")
    selected_edges_range = row.get("validation_selected_edges_range")
    relative_density = row.get("internal_val_relative_density")
    threshold_grid_rows = row.get("threshold_grid_rows")
    val_auprc_ok = isinstance(val_auprc, float) and val_auprc >= VAL_AUPRC_FLOOR
    selected_edges_stable = (
        isinstance(selected_edges, int)
        and selected_edges > 0
        and isinstance(selected_edges_range, int)
        and selected_edges_range <= max(1_000, int(0.25 * selected_edges))
    )
    density_closer = (
        isinstance(relative_density, float)
        and abs(relative_density - 1.0) < abs(EXP02_REFERENCE_VAL_RELATIVE_DENSITY - 1.0)
    )
    threshold_grid_present = isinstance(threshold_grid_rows, int) and threshold_grid_rows > 0
    return {
        "gate_val_auprc_ok": val_auprc_ok,
        "gate_selected_edges_stable": selected_edges_stable,
        "gate_density_closer_than_exp02": density_closer,
        "gate_threshold_grid_present": threshold_grid_present,
        "eligible_for_locked_test": (
            val_auprc_ok
            and selected_edges_stable
            and density_closer
            and threshold_grid_present
        ),
    }


def _history_int_values(history: Sequence[Mapping[str, object]], key: str) -> list[int]:
    values: list[int] = []
    for row in history:
        value = row.get(key)
        if isinstance(value, (float, int)):
            values.append(int(value))
    return values


def _history_int_min(history: Sequence[Mapping[str, object]], key: str) -> int | str:
    values = _history_int_values(history, key)
    return min(values) if values else ""


def _history_int_max(history: Sequence[Mapping[str, object]], key: str) -> int | str:
    values = _history_int_values(history, key)
    return max(values) if values else ""


def _history_int_range(history: Sequence[Mapping[str, object]], key: str) -> int | str:
    values = _history_int_values(history, key)
    return max(values) - min(values) if values else ""


def _history_float(
    history: Sequence[Mapping[str, object]],
    epoch: int,
    key: str,
) -> float | str:
    for row in history:
        if int(cast(SupportsInt, row.get("epoch", -1))) == epoch and key in row:
            return float(cast(SupportsFloat, row[key]))
    return ""


def _heldout_fields(run_dir: Path) -> dict[str, object]:
    refined = _load_optional_json(run_dir / "pairwise_test" / "refined_metrics.json") or {}
    raw = _load_optional_json(run_dir / "pairwise_test" / "raw_metrics.json") or {}
    topology = _load_optional_json(run_dir / "topology_test" / "topology_metrics.json") or {}
    summary = topology.get("summary", {}) if isinstance(topology, Mapping) else {}
    deletion = topology.get("deletion_diagnostics", {}) if isinstance(topology, Mapping) else {}
    protocol = topology.get("protocol", {}) if isinstance(topology, Mapping) else {}
    if not isinstance(summary, Mapping):
        summary = {}
    if not isinstance(deletion, Mapping):
        deletion = {}
    if not isinstance(protocol, Mapping):
        protocol = {}
    return {
        "heldout_refined_precision": _mapping_float(refined, "precision"),
        "heldout_refined_recall": _mapping_float(refined, "recall"),
        "heldout_refined_f1": _mapping_float(refined, "f1"),
        "heldout_refined_auprc": _mapping_float(refined, "auprc"),
        "heldout_refined_auroc": _mapping_float(refined, "auroc"),
        "heldout_raw_auprc": _mapping_float(raw, "auprc"),
        "heldout_raw_auroc": _mapping_float(raw, "auroc"),
        "heldout_topology_relative_density": _mapping_float(summary, "relative_density"),
        "heldout_topology_graph_sim": _mapping_float(summary, "graph_sim"),
        "heldout_topology_degree_mmd": _mapping_float(summary, "deg_dist_mmd"),
        "heldout_topology_cc_mmd": _mapping_float(summary, "cc_mmd"),
        "heldout_edges_added": _mapping_float(deletion, "edges_added"),
        "heldout_edges_deleted": _mapping_float(deletion, "edges_deleted"),
        "heldout_protocol_candidate_universe": protocol.get("candidate_universe", ""),
        "heldout_protocol_test_labels_visible_to_model": protocol.get(
            "test_labels_visible_to_model",
            "",
        ),
    }


def _raw_topology_baseline_fields(raw_baseline_artifact_dir: Path) -> dict[str, object]:
    topology = _load_optional_json(raw_baseline_artifact_dir / "topology_metrics.json") or {}
    summary = topology.get("summary", {}) if isinstance(topology, Mapping) else {}
    protocol = topology.get("protocol", {}) if isinstance(topology, Mapping) else {}
    if not isinstance(summary, Mapping):
        summary = {}
    if not isinstance(protocol, Mapping):
        protocol = {}
    return {
        "heldout_raw_topology_relative_density": _mapping_float(summary, "relative_density"),
        "heldout_raw_topology_graph_sim": _mapping_float(summary, "graph_sim"),
        "heldout_raw_topology_degree_mmd": _mapping_float(summary, "deg_dist_mmd"),
        "heldout_raw_topology_cc_mmd": _mapping_float(summary, "cc_mmd"),
        "heldout_raw_protocol_candidate_universe": protocol.get("candidate_universe", ""),
        "heldout_raw_protocol_test_labels_visible_to_model": protocol.get(
            "test_labels_visible_to_model",
            "",
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    columns = list(BASE_COLUMNS)
    if any(any(key in row for key in HELDOUT_COLUMNS) for row in rows):
        columns.extend(HELDOUT_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _markdown_report(rows: Sequence[Mapping[str, object]]) -> str:
    include_heldout = any(any(key in row for key in HELDOUT_COLUMNS) for row in rows)
    lines = [
        "# Validation-first exp03 summary",
        "",
        f"- Exp02 reference validation AUPRC: `{EXP02_REFERENCE_VAL_AUPRC:.10f}`",
        f"- Phase B AUPRC floor: `{VAL_AUPRC_FLOOR:.4f}`",
        f"- Runs summarized: `{len(rows)}`",
        "",
    ]
    if include_heldout:
        lines.append(
            "Held-out metrics are included only for locked runs requested by the operator."
        )
    else:
        lines.append("Held-out metrics were not included.")
    lines.extend(
        [
            "",
            "| run_id | best_epoch | threshold | val_auprc | val_topology_loss | rel_density |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            (
                "| {run_id} | {best_epoch} | {selected_threshold} | {val_auprc} | "
                "{val_topology_loss} | {internal_val_relative_density} |"
            ).format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, object], payload)


def _load_optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _as_int(row: Mapping[str, object], key: str) -> int:
    return int(cast(SupportsInt, row[key]))


def _as_optional_int(row: Mapping[str, object], key: str) -> int | str:
    if key not in row:
        return ""
    return int(cast(SupportsFloat, row[key]))


def _as_optional_float(row: Mapping[str, object], key: str) -> float | str:
    if key not in row:
        return ""
    return float(cast(SupportsFloat, row[key]))


def _mapping_float(row: Mapping[str, object], key: str) -> float | str:
    value = row.get(key)
    if isinstance(value, (float, int)):
        return float(value)
    return ""


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for exp03 analysis."""
    parser = argparse.ArgumentParser(description="Analyze TCCIG exp03 artifacts")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--exp02-reference-dir", type=Path, default=DEFAULT_EXP02_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--locked-run-id", action="append", default=[])
    parser.add_argument(
        "--raw-baseline-run-id",
        default="",
        help="raw topology-baseline run id for the locked candidate",
    )
    parser.add_argument("--include-phase-b", action="store_true")
    args = parser.parse_args(argv)

    rows: list[dict[str, object]] = []
    if args.exp02_reference_dir.exists():
        rows.append(
            collect_run_row(
                run_id=EXP02_REFERENCE_RUN_ID,
                run_dir=args.exp02_reference_dir,
                include_heldout=EXP02_REFERENCE_RUN_ID in args.locked_run_id,
            )
        )
    run_ids = list(PHASE_A_RUN_IDS)
    if args.include_phase_b:
        run_ids.extend(PHASE_B_CONFIG_RUN_IDS)
    for run_id in run_ids:
        run_dir = args.log_root / run_id
        if not run_dir.exists():
            continue
        raw_baseline_artifact_dir = None
        if run_id in args.locked_run_id and args.raw_baseline_run_id:
            raw_baseline_artifact_dir = (
                args.log_root
                / str(args.raw_baseline_run_id)
                / "raw_pairwise_topology_baseline"
            )
        rows.append(
            collect_run_row(
                run_id=run_id,
                run_dir=run_dir,
                include_heldout=run_id in args.locked_run_id,
                raw_baseline_artifact_dir=raw_baseline_artifact_dir,
            )
        )
    write_exp03_report(rows=rows, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_exp03_analysis.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add tccig/analyze_exp03.py tests/unit/test_tccig_exp03_analysis.py
rtk git commit -m "feat: add exp03 validation analysis report"
```

Expected: commit succeeds with only the files listed above.

## Task 4: Isolate Raw Pairwise Topology Baseline Output

**Files:**
- Modify: `tccig/test.py`
- Modify: `tccig/raw_pairwise_topology_baseline.py`
- Modify: `tests/unit/test_tccig_test_export.py`

- [ ] **Step 1: Write failing raw-baseline output-dir tests**

Append this test to `tests/unit/test_tccig_test_export.py`:

```python
def test_run_raw_pairwise_topology_baseline_can_write_separate_output_dir(tmp_path: Path) -> None:
    table = _pair_table()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C", "D"])
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    with (processed_dir / "human_test_graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)
    with (processed_dir / "test_sampled_nodes.pkl").open("wb") as handle:
        pickle.dump({3: [["A", "B", "C"], ["A", "C", "D"]]}, handle)

    def _fake_score_split(**_kwargs: object) -> list[float]:
        return [0.95, 0.10, 0.80, 0.20]

    log_dir = tmp_path / "logs"
    output_dir = log_dir / "raw_pairwise_topology_baseline"
    run_raw_pairwise_topology_baseline(
        table=table,
        processed_dir=processed_dir,
        scorer_cfg={},
        runtime=_runtime(),
        cache_dir=tmp_path / "cache",
        log_dir=log_dir,
        output_dir=output_dir,
        raw_output_rule=GraphRule(type="threshold", value=0.5),
        score_split_fn=_fake_score_split,
    )

    assert (output_dir / "all_test_ppi_pred.txt").exists()
    assert (output_dir / "topology_metrics.json").exists()
    assert not (log_dir / "topology_test" / "topology_metrics.json").exists()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_test_export.py::test_run_raw_pairwise_topology_baseline_can_write_separate_output_dir -v
```

Expected: FAIL with `TypeError: run_raw_pairwise_topology_baseline() got an unexpected keyword argument 'output_dir'`.

- [ ] **Step 3: Add explicit output directory support**

In `tccig/test.py`, update the function signature:

```python
def run_raw_pairwise_topology_baseline(
    *,
    table: PairTable,
    processed_dir: Path,
    scorer_cfg: Mapping[str, object],
    runtime: TCCIGRuntime,
    cache_dir: Path,
    log_dir: Path,
    raw_output_rule: GraphRule,
    score_split_fn: ScoreSplitFn,
    output_dir: Path | None = None,
) -> dict[str, float]:
```

Replace the raw baseline write directory line:

```python
        topology_dir = output_dir if output_dir is not None else log_dir / "topology_test"
```

In `tccig/raw_pairwise_topology_baseline.py`, pass the separate output dir:

```python
    metrics = run_raw_pairwise_topology_baseline(
        table=tables["topology_test"],
        processed_dir=processed_dir,
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
        log_dir=log_dir,
        output_dir=log_dir / "raw_pairwise_topology_baseline",
        raw_output_rule=output_rule,
        score_split_fn=_score_split,
    )
```

Update the manifest payload in `run_baseline` to include the artifact directory:

```python
                "artifact_dir": str(log_dir / "raw_pairwise_topology_baseline"),
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_test_export.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add tccig/test.py tccig/raw_pairwise_topology_baseline.py tests/unit/test_tccig_test_export.py
rtk git commit -m "fix: isolate raw tccig topology baseline artifacts"
```

Expected: commit succeeds with only the files listed above.

## Task 5: Gated FN/FP Hard-Quadrant Sampler Lever

Execute this task only if Phase A analysis points to FN/FP hard-target composition as the next Phase B lever. Skip this task when Phase B uses only `beta`, `topology_weight`, or `pos_weight`.

**Files:**
- Modify: `tccig/prepare.py`
- Modify: `tests/unit/test_tccig_prepare.py`
- Modify after sampler tests pass: `tccig/exp03_configs.py`
- Modify after sampler tests pass: `tests/unit/test_tccig_exp03_configs.py`

- [ ] **Step 1: Write failing sampler tests**

Modify the import block in `tests/unit/test_tccig_prepare.py` to include `parse_edge_sampling_config`:

```python
from tccig.prepare import (
    CandidatePair,
    EdgeSamplingConfig,
    classify_scorer_error_targets,
    ordered_probabilities_from_indexed_rows,
    parse_edge_sampling_config,
    read_pair_table,
    sample_epoch_edge_targets,
    strict_reject_legacy_hooks,
    write_json,
)
```

Append these tests after `test_sample_epoch_edge_targets_keeps_all_hard_and_samples_easy_budget`:

```python
def test_parse_edge_sampling_config_reads_hard_quadrant_ratio() -> None:
    config = parse_edge_sampling_config(
        {
            "hard_fraction": 0.7,
            "easy_anchor_fraction": 0.3,
            "hard_quadrant_ratio": {"fn": 1, "fp": 1},
        }
    )

    assert config.hard_quadrant_ratio == (1, 1)


def test_sample_epoch_edge_targets_downsamples_larger_hard_quadrant_to_ratio() -> None:
    quadrants = classify_scorer_error_targets(
        pairs=[
            CandidatePair("A", "B"),
            CandidatePair("A", "C"),
            CandidatePair("A", "D"),
            CandidatePair("A", "E"),
            CandidatePair("A", "F"),
            CandidatePair("B", "C"),
        ],
        labels=[0, 1, 1, 1, 1, 1],
        pairwise_graph_edges=[("A", "B"), ("B", "C")],
    )

    targets = sample_epoch_edge_targets(
        quadrants=quadrants,
        sampling=EdgeSamplingConfig(
            hard_fraction=0.5,
            easy_anchor_fraction=0.5,
            seed=3,
            hard_quadrant_ratio=(1, 1),
        ),
        epoch=1,
    )

    hard_quadrants = [target.quadrant for target in targets if target.quadrant in {"fp", "fn"}]
    assert hard_quadrants.count("fp") == 1
    assert hard_quadrants.count("fn") == 1


def test_sample_epoch_edge_targets_default_preserves_all_hard_quadrants() -> None:
    quadrants = classify_scorer_error_targets(
        pairs=[
            CandidatePair("A", "B"),
            CandidatePair("A", "C"),
            CandidatePair("A", "D"),
            CandidatePair("A", "E"),
        ],
        labels=[0, 1, 1, 1],
        pairwise_graph_edges=[("A", "B")],
    )

    targets = sample_epoch_edge_targets(
        quadrants=quadrants,
        sampling=EdgeSamplingConfig(hard_fraction=0.5, easy_anchor_fraction=0.5, seed=3),
        epoch=1,
    )

    hard_indices = [target.pair_index for target in targets if target.quadrant in {"fp", "fn"}]
    assert hard_indices == [0, 1, 2, 3]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_prepare.py -k "hard_quadrant_ratio or downsample or default_preserves_all_hard" -v
```

Expected: FAIL because `EdgeSamplingConfig` has no `hard_quadrant_ratio` field.

- [ ] **Step 3: Implement deterministic hard-quadrant downsampling**

In `tccig/prepare.py`, update `EdgeSamplingConfig`:

```python
@dataclass(frozen=True)
class EdgeSamplingConfig:
    """Scorer-error target sampling controls."""

    hard_fraction: float = 0.7
    easy_anchor_fraction: float = 0.3
    seed: int = 0
    reshuffle_easy_each_epoch: bool = True
    hard_quadrant_ratio: tuple[int, int] | None = None
```

In `parse_edge_sampling_config`, parse the ratio before the return:

```python
    hard_quadrant_ratio = _parse_hard_quadrant_ratio(
        raw_config.get("hard_quadrant_ratio"),
    )
```

Then include it in the returned config:

```python
        hard_quadrant_ratio=hard_quadrant_ratio,
```

Add this helper near `parse_edge_sampling_config`:

```python
def _parse_hard_quadrant_ratio(raw_ratio: object) -> tuple[int, int] | None:
    if raw_ratio is None:
        return None
    if not isinstance(raw_ratio, Mapping):
        raise ValueError("refiner.edge_sampling.hard_quadrant_ratio must be a mapping")
    fn = _positive_int(raw_ratio.get("fn"), "refiner.edge_sampling.hard_quadrant_ratio.fn")
    fp = _positive_int(raw_ratio.get("fp"), "refiner.edge_sampling.hard_quadrant_ratio.fp")
    return fn, fp
```

Replace the hard-target line in `sample_epoch_edge_targets`:

```python
    hard_targets = _sample_hard_targets_by_ratio(
        fp_targets=list(quadrants.get("fp", ())),
        fn_targets=list(quadrants.get("fn", ())),
        sampling=sampling,
        epoch=epoch,
    )
```

Add these helpers near `sample_epoch_edge_targets`:

```python
def _sample_hard_targets_by_ratio(
    *,
    fp_targets: list[EdgeTarget],
    fn_targets: list[EdgeTarget],
    sampling: EdgeSamplingConfig,
    epoch: int,
) -> list[EdgeTarget]:
    if sampling.hard_quadrant_ratio is None or not fp_targets or not fn_targets:
        return [*fp_targets, *fn_targets]
    fn_ratio, fp_ratio = sampling.hard_quadrant_ratio
    max_fn_for_fp = math.ceil(len(fp_targets) * fn_ratio / fp_ratio)
    max_fp_for_fn = math.ceil(len(fn_targets) * fp_ratio / fn_ratio)
    if len(fn_targets) > max_fn_for_fp:
        fn_targets = _deterministic_target_sample(
            targets=fn_targets,
            count=max_fn_for_fp,
            seed=sampling.seed + epoch + 1_000_003,
        )
    elif len(fp_targets) > max_fp_for_fn:
        fp_targets = _deterministic_target_sample(
            targets=fp_targets,
            count=max_fp_for_fn,
            seed=sampling.seed + epoch + 2_000_003,
        )
    return [*fp_targets, *fn_targets]


def _deterministic_target_sample(
    *,
    targets: list[EdgeTarget],
    count: int,
    seed: int,
) -> list[EdgeTarget]:
    if count >= len(targets):
        return list(targets)
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(targets), generator=generator)[:count].tolist()
    return [targets[index] for index in sorted(indices)]
```

- [ ] **Step 4: Run sampler tests and verify they pass**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_prepare.py -v
```

Expected: PASS.

- [ ] **Step 5: Add the gated exp03 b6 config**

In `tests/unit/test_tccig_exp03_configs.py`, append:

```python
def test_sampler_phase_b_config_adds_fnfp_ratio_when_requested() -> None:
    configs = build_exp03_configs(
        _load_base_config(),
        include_phase_b=True,
        include_sampler_phase_b=True,
    )

    assert "03_b6_fnfp_1to1_if_sampler_knob_is_implemented" in configs
    edge_sampling = _refiner(configs["03_b6_fnfp_1to1_if_sampler_knob_is_implemented"])["edge_sampling"]
    assert isinstance(edge_sampling, dict)
    assert edge_sampling["hard_quadrant_ratio"] == {"fn": 1, "fp": 1}
```

Update `build_exp03_configs` signature in `tccig/exp03_configs.py`:

```python
def build_exp03_configs(
    base_config: Mapping[str, object],
    *,
    include_phase_b: bool,
    include_sampler_phase_b: bool = False,
) -> dict[str, dict[str, object]]:
```

After `03_b5_bce_pos_weight_0p5`, add:

```python
if include_sampler_phase_b:
    b6 = _variant(
        base_config,
        run_id="03_b6_fnfp_1to1_if_sampler_knob_is_implemented",
        topology_training_enabled=True,
        weights=FIXED_VALIDATION_LOSSES,
        topology_weight=1.0,
        pos_weight=1.0,
    )
    edge_sampling = _mapping(_mapping(b6, "refiner"), "edge_sampling")
    edge_sampling["hard_quadrant_ratio"] = {"fn": 1, "fp": 1}
    configs["03_b6_fnfp_1to1_if_sampler_knob_is_implemented"] = b6
```

Update `write_exp03_configs` and `main` to pass `include_sampler_phase_b`; add this CLI flag:

```python
    parser.add_argument("--sampler-phase-b", action="store_true", help="also write the gated FN:FP b6 config")
```

- [ ] **Step 6: Run config and sampler tests**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_prepare.py tests/unit/test_tccig_exp03_configs.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
rtk git add tccig/prepare.py tccig/exp03_configs.py tests/unit/test_tccig_prepare.py tests/unit/test_tccig_exp03_configs.py
rtk git commit -m "feat: add gated tccig fnfp hard sampling lever"
```

Expected: commit succeeds with only the files listed above.

## Task 6: Add Exp03 Operator Runbook

**Files:**
- Create: `docs/experiment/tccig/exp03_runbook.md`

- [ ] **Step 1: Create the runbook**

Create `docs/experiment/tccig/exp03_runbook.md`:

````markdown
# TCCIG exp03 runbook

## Contract

exp03 is a diagnostic. Phase A and Phase B selection use training dynamics and validation metrics only. Held-out pairwise/topology test metrics are generated only after a candidate is locked.

Treat this runbook and `docs/superpowers/specs/2026-07-04-tccig-exp03-loss-conflict-diagnostic-design.md` as authoritative for exp03. Older threshold wording in `CONTEXT.md`, `tccig/README.md`, or `docs/experiment/tccig/model.md` may still describe fixed `p_refined >= 0.5`; exp03 uses calibrated `val_topology_loss` selection.

## Generate configs

```bash
rtk proxy uv run --locked --no-sync --offline python -m tccig.exp03_configs --base configs/tccig/02_balanced_subset.yaml --output-dir configs/tccig/exp03 --phase-b
```

Use `--sampler-phase-b` only after Phase A selects the FN/FP hard-quadrant sampling lever.

## Pre-launch checks

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/integration/test_tccig_orchestrator.py::test_tccig_orchestrator_can_skip_heldout_test_artifacts -v
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_exp03_configs.py tests/unit/test_tccig_topology_training.py::test_resolve_refined_output_rule_config_accepts_calibrated_grid -v
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_exp03_configs.py::test_exp03_configs_only_change_intended_paths -v
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_writes_threshold_grid_artifacts -v
```

For every Phase A config, confirm:

- `refiner.monitor_metric: val_topology_loss`
- `refiner.topology_validation.enabled: true`
- `refiner.topology_validation.losses: {alpha: 1.0, beta: 8.0, gamma: 0.5, delta: 0.0}`
- `graph_selection.refined_output_rule.type: calibrated`
- `graph_selection.refined_output_rule.objective: val_topology_loss`
- `refiner.topology_training.topo_only_after_epoch: null` for `03_a2` through `03_a5`
- `refiner.topology_training.enabled: false` for `03_a1_bce_only`

## Phase A launch

Submit only `03_a1` through `03_a5`. `03_a0_exp02_topo_only_reference` is the exp02 artifact in `artifacts/exp02_rerun_fix/logs/tccig/02_balanced_subset`.

```bash
GRAND_TCCIG_SKIP_TEST_SPLITS=1 sbatch scripts/tccig.sh configs/tccig/exp03/03_a1_bce_only.yaml
GRAND_TCCIG_SKIP_TEST_SPLITS=1 sbatch scripts/tccig.sh configs/tccig/exp03/03_a2_bce_graph_sim.yaml
GRAND_TCCIG_SKIP_TEST_SPLITS=1 sbatch scripts/tccig.sh configs/tccig/exp03/03_a3_bce_density.yaml
GRAND_TCCIG_SKIP_TEST_SPLITS=1 sbatch scripts/tccig.sh configs/tccig/exp03/03_a4_bce_degree.yaml
GRAND_TCCIG_SKIP_TEST_SPLITS=1 sbatch scripts/tccig.sh configs/tccig/exp03/03_a5_bce_full_topology.yaml
```

Record Slurm job ids and stdout/stderr paths in the experiment tracker before analyzing results. Immediately after each submit, verify the job and log path:

```bash
squeue -j "$JOB_ID" -o "%.18i %.9P %.24j %.8u %.2t %.12M %.12l %.6D %R"
scontrol show job "$JOB_ID"
ls -lh "logs/tccig/slurm_${JOB_ID}.out" "logs/tccig/slurm_${JOB_ID}.err"
tail -n 80 "logs/tccig/slurm_${JOB_ID}.out"
tail -n 80 "logs/tccig/slurm_${JOB_ID}.err"
```

## Phase A analysis

Copy Phase A artifacts back from the HPC root before analysis:

```bash
REMOTE_ROOT=/public/home/wangar2023/grand
LOCAL_ROOT=/Users/richardwang/Documents/grand
mkdir -p "$LOCAL_ROOT/logs/tccig"
scp -r "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/03_a1_bce_only" "$LOCAL_ROOT/logs/tccig/"
scp -r "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/03_a2_bce_graph_sim" "$LOCAL_ROOT/logs/tccig/"
scp -r "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/03_a3_bce_density" "$LOCAL_ROOT/logs/tccig/"
scp -r "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/03_a4_bce_degree" "$LOCAL_ROOT/logs/tccig/"
scp -r "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/03_a5_bce_full_topology" "$LOCAL_ROOT/logs/tccig/"
scp "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/slurm_*.out" "$LOCAL_ROOT/logs/tccig/"
scp "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/slurm_*.err" "$LOCAL_ROOT/logs/tccig/"
```

Before running analysis, verify every Phase A run has validation artifacts and no clean held-out test artifacts:

```bash
test -f logs/tccig/03_a1_bce_only/training_summary.json
test -f logs/tccig/03_a1_bce_only/threshold_grid/best_epoch.json
test -f logs/tccig/03_a2_bce_graph_sim/training_summary.json
test -f logs/tccig/03_a2_bce_graph_sim/threshold_grid/best_epoch.json
test -f logs/tccig/03_a3_bce_density/training_summary.json
test -f logs/tccig/03_a3_bce_density/threshold_grid/best_epoch.json
test -f logs/tccig/03_a4_bce_degree/training_summary.json
test -f logs/tccig/03_a4_bce_degree/threshold_grid/best_epoch.json
test -f logs/tccig/03_a5_bce_full_topology/training_summary.json
test -f logs/tccig/03_a5_bce_full_topology/threshold_grid/best_epoch.json
test ! -d logs/tccig/03_a1_bce_only/pairwise_test
test ! -d logs/tccig/03_a1_bce_only/topology_test
test ! -d logs/tccig/03_a2_bce_graph_sim/pairwise_test
test ! -d logs/tccig/03_a2_bce_graph_sim/topology_test
test ! -d logs/tccig/03_a3_bce_density/pairwise_test
test ! -d logs/tccig/03_a3_bce_density/topology_test
test ! -d logs/tccig/03_a4_bce_degree/pairwise_test
test ! -d logs/tccig/03_a4_bce_degree/topology_test
test ! -d logs/tccig/03_a5_bce_full_topology/pairwise_test
test ! -d logs/tccig/03_a5_bce_full_topology/topology_test
```

Then run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m tccig.analyze_exp03 --log-root logs/tccig --exp02-reference-dir artifacts/exp02_rerun_fix/logs/tccig/02_balanced_subset --output-dir analysis/tccig_exp03
```

Use `analysis/tccig_exp03/exp03_summary.md` to decide whether Phase B is justified. The validation AUPRC floor is `0.6705`.

## Phase B gate

Run Phase B only when all conditions hold:

- Phase A confirms BCE-vs-topology conflict or a clear component-level imbalance.
- At least one Phase A variant improves a topology metric without dropping below validation AUPRC `0.6705`.
- The best Phase A candidate still leaves a validation topology gap, selected-edge instability, or interpretable AUPRC tradeoff worth tuning.

Pick at most two levers from `03_b1` through `03_b5`, and launch at most four Phase B runs before review. Use `03_b6` only after implementing and approving the explicit sampler lever. Submit selected Phase B configs with `GRAND_TCCIG_SKIP_TEST_SPLITS=1` and run the same post-submit Slurm/log checks as Phase A.

## Locked-candidate held-out report

After locking one candidate by validation evidence, run pairwise/topology test once by launching the locked config without `GRAND_TCCIG_SKIP_TEST_SPLITS`. Then run raw pairwise topology baseline into a separate output run id:

```bash
LOCKED_RUN_ID=03_b1_beta2
sbatch scripts/tccig.sh "configs/tccig/exp03/${LOCKED_RUN_ID}.yaml"
GRAND_TCCIG_BASELINE_SOURCE_RUN_ID="$LOCKED_RUN_ID" GRAND_TCCIG_BASELINE_OUTPUT_RUN_ID="${LOCKED_RUN_ID}_raw_pairwise_baseline" sbatch scripts/tccig_pairwise_baseline.sh "configs/tccig/exp03/${LOCKED_RUN_ID}.yaml"
```

Regenerate the report with held-out fields for the locked candidate:

```bash
LOCKED_RUN_ID=03_b1_beta2
RAW_BASELINE_RUN_ID="${LOCKED_RUN_ID}_raw_pairwise_baseline"
rtk proxy uv run --locked --no-sync --offline python -m tccig.analyze_exp03 --log-root logs/tccig --exp02-reference-dir artifacts/exp02_rerun_fix/logs/tccig/02_balanced_subset --output-dir analysis/tccig_exp03 --include-phase-b --locked-run-id "$LOCKED_RUN_ID" --raw-baseline-run-id "$RAW_BASELINE_RUN_ID"
```

The locked report must show `heldout_protocol_candidate_universe=all_test_ppi.txt`, `heldout_protocol_test_labels_visible_to_model=False`, `heldout_raw_protocol_candidate_universe=all_test_ppi.txt`, and `heldout_raw_protocol_test_labels_visible_to_model=False`.

Do not compare raw `0.5` precision directly against refined calibrated-threshold precision as a model-quality claim. Use AUPRC/AUROC and matched operating points.
````

- [ ] **Step 2: Commit**

Run:

```bash
rtk git add docs/experiment/tccig/exp03_runbook.md
rtk git commit -m "docs: add tccig exp03 runbook"
```

Expected: commit succeeds with only `docs/experiment/tccig/exp03_runbook.md`.

## Task 7: Final Verification Before Launch

**Files:**
- Verify: `tccig/exp03_configs.py`
- Verify: `tccig/s2gae.py`
- Verify: `tccig/analyze_exp03.py`
- Verify: `tccig/test.py`
- Verify: `configs/tccig/exp03/*.yaml`
- Verify: `docs/experiment/tccig/exp03_runbook.md`

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_exp03_configs.py tests/unit/test_tccig_exp03_analysis.py tests/unit/test_tccig_topology_training.py tests/unit/test_tccig_test_export.py -v
```

Expected: PASS.

- [ ] **Step 2: Run focused integration tests**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/integration/test_tccig_orchestrator.py::test_tccig_orchestrator_can_skip_heldout_test_artifacts tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_writes_threshold_grid_artifacts tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_uses_selected_rule_for_test_paths_and_manifest tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_persists_epoch_selected_rule_history -v
```

Expected: PASS.

- [ ] **Step 3: Run lint on touched Python files**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline ruff check tccig/exp03_configs.py tccig/analyze_exp03.py tccig/s2gae.py tccig/test.py tccig/raw_pairwise_topology_baseline.py tccig/prepare.py tests/unit/test_tccig_exp03_configs.py tests/unit/test_tccig_exp03_analysis.py tests/unit/test_tccig_topology_training.py tests/unit/test_tccig_test_export.py tests/unit/test_tccig_prepare.py tests/integration/test_tccig_orchestrator.py
```

Expected: PASS. If Task 5 was skipped, omit `tccig/prepare.py` and `tests/unit/test_tccig_prepare.py`.

- [ ] **Step 4: Run config generation and parse check**

Run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m tccig.exp03_configs --base configs/tccig/02_balanced_subset.yaml --output-dir configs/tccig/exp03 --phase-b
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_exp03_configs.py::test_generated_exp03_configs_parse_existing_tccig_helpers -v
```

Expected: PASS and ten YAML files under `configs/tccig/exp03/`.

- [ ] **Step 5: Verify local launch protocol and held-out protocol guards**

Run:

```bash
rtk grep -n "GRAND_TCCIG_SKIP_TEST_SPLITS" scripts/tccig.sh docs/experiment/tccig/exp03_runbook.md
rtk grep -n "heldout_protocol_candidate_universe|heldout_raw_protocol_candidate_universe|test_labels_visible_to_model" tccig/analyze_exp03.py tests/unit/test_tccig_exp03_analysis.py docs/experiment/tccig/exp03_runbook.md
```

Expected: both checks return the validation-only launch path and the held-out protocol fields. The report must expose `candidate_universe=all_test_ppi.txt` and `test_labels_visible_to_model=False` for both refined and raw-topology held-out artifacts.

- [ ] **Step 6: Verify remote launch readiness after syncing to HPC**

Run from the local checkout after pushing or copying the implementation to the remote root:

```bash
ssh wangar2023@10.15.89.192 "cd /public/home/wangar2023/grand && test -d .venv && test -x scripts/tccig.sh && test -f configs/tccig/exp03/03_a1_bce_only.yaml && test -f configs/tccig/exp03/03_b4_topology_weight_2p0.yaml && test -f configs/tccig/exp03/03_b5_bce_pos_weight_0p5.yaml"
ssh wangar2023@10.15.89.192 "cd /public/home/wangar2023/grand && GRAND_TCCIG_SKIP_TEST_SPLITS=1 bash -n scripts/tccig.sh && bash -n scripts/tccig_pairwise_baseline.sh"
```

Expected: both commands exit 0 before any `sbatch` is submitted.

- [ ] **Step 7: Check worktree**

Run:

```bash
rtk git status --short --branch
```

Expected: only intentionally modified files remain. Do not stage unrelated deleted `configs/tccig/02_balanced_subset_smoke.yaml` or untracked `logs/tccig/` unless the user explicitly asks.

## Self-Review

- Spec coverage: Phase A config matrix, fixed validation monitor, Phase B capped levers, threshold-grid artifacts, validation-first analysis, held-out gating, raw baseline separation, sampler gate, run naming, and launch checks are covered.
- Placeholder scan: this plan contains concrete files, tests, commands, and code snippets for every implementation task.
- Type consistency: new public functions are `build_exp03_configs`, `write_exp03_configs`, `collect_run_row`, `write_exp03_report`, and `run_raw_pairwise_topology_baseline(..., output_dir=None)`; tests and snippets use those exact names.
