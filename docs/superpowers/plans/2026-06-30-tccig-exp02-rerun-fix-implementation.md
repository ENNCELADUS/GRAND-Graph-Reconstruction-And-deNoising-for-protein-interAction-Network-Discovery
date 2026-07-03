# TCCIG Exp02 Rerun Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement calibrated refined-output threshold selection/reporting and an opt-in topology-only diagnostic mode for the TCCIG Exp02 rerun.

**Architecture:** Keep `GraphRule` as the executable hard-graph rule (`threshold` only). Parse calibrated refined-output config into a separate train-orchestrator config object, pass its threshold grid into the S2GAE validation loop, persist the selected executable threshold in JSON/checkpoints, and pass that best selected rule into pairwise/topology test export. Add topology-only epochs inside the existing S2GAE train loop by skipping the BCE phase before edge-target sampling while leaving the existing topology backward phase intact.

**Tech Stack:** Python 3.10+, PyTorch, Accelerate, NetworkX, pytest, ruff, mypy, YAML configs via `uv`.

---

## Scope Check

The spec covers two coupled changes in one pipeline:

- Part A changes validation selection and test reporting around the refined-output operating point.
- Part B adds a diagnostic train-loop mode that answers whether the topology gradient can reduce topology loss without BCE.

They should stay in one implementation plan because both changes are needed for a single rerun and both touch `tccig/train.py` and `tccig/s2gae.py`. The tasks below keep the code seams independent: first parsing, then validation-grid selection, then orchestrator plumbing, then topo-only training.

## File Structure

- Modify `tccig/train.py`
  - Parse `graph_selection.refined_output_rule` as either fixed threshold or calibrated grid.
  - Validate calibrated mode against `refiner.monitor_metric` and `refiner.topology_validation.enabled`.
  - Choose the effective refined-output rule after training.
  - Write `ignored_legacy_rules` and the effective rule into the run manifest.
- Modify `tccig/prepare.py`
  - Extend `TrainRefinerRequest` with optional validation graph-rule grid and selected-rule source metadata.
- Modify `tccig/s2gae.py`
  - Evaluate validation topology over one or more hard-threshold rules after a single inference pass.
  - Persist per-epoch nested `selected_rule` in `training_summary.json`.
  - Parse and serialize `topo_only_after_epoch`.
  - Skip BCE edge-target sampling/DataLoader/BCE steps on topo-only epochs.
- Modify `configs/tccig/02_balanced_subset.yaml`
  - Switch the rerun config to calibrated refined-output thresholding.
  - Enable `topo_only_after_epoch: 7`.
- Test `tests/unit/test_tccig_topology_training.py`
  - Parser tests for calibrated refined-output config and topo-only config.
  - Unit tests for validation-grid argmin selection.
  - Unit tests for effective selected-rule helper behavior.
- Test `tests/integration/test_tccig_orchestrator.py`
  - Pipeline plumbing test that calibrated mode passes the checkpoint-selected rule into both test paths and writes `ignored_legacy_rules`.
- Test `tests/integration/test_tccig_topology_training_stage.py`
  - Topo-only integration test that BCE sampling is skipped, BCE/anchor logs are zero, topology loss logs, and the optimizer still steps.

## Task 1: Parse Calibrated Refined-Output Rule Config

**Files:**
- Modify: `tccig/train.py`
- Test: `tests/unit/test_tccig_topology_training.py`

- [ ] **Step 1: Write failing parser tests**

Append these tests near the existing config parser tests in `tests/unit/test_tccig_topology_training.py`, after `test_parse_config_reads_topology_subset_sampler`.

```python
def _calibrated_pipeline_config() -> dict[str, object]:
    return {
        "refiner": {
            "monitor_metric": "val_topology_loss",
            "topology_validation": {"enabled": True},
        },
        "graph_selection": {
            "refined_output_rule": {
                "type": "calibrated",
                "objective": "val_topology_loss",
                "grid": [0.5, 0.9, 0.97],
            }
        },
    }


def test_resolve_refined_output_rule_config_accepts_calibrated_grid() -> None:
    parsed = tccig_train._resolve_refined_output_rule_config(_calibrated_pipeline_config())

    assert parsed.calibrated is True
    assert parsed.objective == "val_topology_loss"
    assert parsed.selected_rule_source == "validation_calibration"
    assert [rule.type for rule in parsed.validation_rules] == ["threshold", "threshold", "threshold"]
    assert [float(rule.value) for rule in parsed.validation_rules] == [0.5, 0.9, 0.97]
    assert parsed.fixed_rule.value == pytest.approx(0.5)


def test_resolve_refined_output_rule_config_preserves_threshold_default() -> None:
    parsed = tccig_train._resolve_refined_output_rule_config({})

    assert parsed.calibrated is False
    assert parsed.objective is None
    assert parsed.selected_rule_source is None
    assert len(parsed.validation_rules) == 1
    assert parsed.validation_rules[0].to_dict() == {"type": "threshold", "value": 0.5}
    assert parsed.fixed_rule.to_dict() == {"type": "threshold", "value": 0.5}


def test_resolve_refined_output_rule_config_rejects_invalid_calibrated_objective() -> None:
    config = _calibrated_pipeline_config()
    graph_selection = config["graph_selection"]
    assert isinstance(graph_selection, dict)
    refined_rule = graph_selection["refined_output_rule"]
    assert isinstance(refined_rule, dict)
    refined_rule["objective"] = "val_auprc"

    with pytest.raises(ValueError, match="objective must be val_topology_loss"):
        tccig_train._resolve_refined_output_rule_config(config)


def test_resolve_refined_output_rule_config_rejects_empty_calibrated_grid() -> None:
    config = _calibrated_pipeline_config()
    graph_selection = config["graph_selection"]
    assert isinstance(graph_selection, dict)
    refined_rule = graph_selection["refined_output_rule"]
    assert isinstance(refined_rule, dict)
    refined_rule["grid"] = []

    with pytest.raises(ValueError, match="grid must not be empty"):
        tccig_train._resolve_refined_output_rule_config(config)


def test_resolve_refined_output_rule_config_requires_topology_monitor_setup() -> None:
    no_topology = _calibrated_pipeline_config()
    refiner = no_topology["refiner"]
    assert isinstance(refiner, dict)
    refiner["topology_validation"] = {"enabled": False}

    with pytest.raises(ValueError, match="topology_validation.enabled"):
        tccig_train._resolve_refined_output_rule_config(no_topology)

    wrong_monitor = _calibrated_pipeline_config()
    refiner = wrong_monitor["refiner"]
    assert isinstance(refiner, dict)
    refiner["monitor_metric"] = "val_auprc"

    with pytest.raises(ValueError, match="monitor_metric"):
        tccig_train._resolve_refined_output_rule_config(wrong_monitor)
```

- [ ] **Step 2: Run parser tests and verify they fail**

Run:

```bash
rtk proxy uv run python -m pytest tests/unit/test_tccig_topology_training.py::test_resolve_refined_output_rule_config_accepts_calibrated_grid tests/unit/test_tccig_topology_training.py::test_resolve_refined_output_rule_config_preserves_threshold_default tests/unit/test_tccig_topology_training.py::test_resolve_refined_output_rule_config_rejects_invalid_calibrated_objective tests/unit/test_tccig_topology_training.py::test_resolve_refined_output_rule_config_rejects_empty_calibrated_grid tests/unit/test_tccig_topology_training.py::test_resolve_refined_output_rule_config_requires_topology_monitor_setup -v
```

Expected: FAIL because `tccig_train._resolve_refined_output_rule_config` does not exist.

- [ ] **Step 3: Add parsed config type and parser helpers**

In `tccig/train.py`, add this import:

```python
from dataclasses import dataclass
```

Add this dataclass near the existing module constants and helper types:

```python
@dataclass(frozen=True)
class RefinedOutputRuleConfig:
    """Parsed refined-output rule config for validation/test graph assembly."""

    calibrated: bool
    fixed_rule: GraphRule
    validation_rules: tuple[GraphRule, ...]
    objective: str | None
    selected_rule_source: str | None
    configured_payload: dict[str, object]
```

Replace the existing `_resolve_refined_output_rule` block with this parser block. Keep `_resolve_refined_output_rule` as a compatibility wrapper for existing tests and callers.

```python
def _resolve_refined_output_rule_config(config: Mapping[str, object]) -> RefinedOutputRuleConfig:
    raw_rule = _graph_selection(config).get(
        "refined_output_rule",
        {"type": "threshold", "value": DEFAULT_GRAPH_THRESHOLD},
    )
    if not isinstance(raw_rule, Mapping):
        raise ValueError("graph_selection.refined_output_rule must be a mapping")
    rule_type = str(raw_rule.get("type", raw_rule.get("mode", "threshold"))).lower()
    if rule_type == "threshold":
        value = _probability(
            raw_rule.get("value", DEFAULT_GRAPH_THRESHOLD),
            "graph_selection.refined_output_rule.value",
        )
        rule = GraphRule(type="threshold", value=value)
        return RefinedOutputRuleConfig(
            calibrated=False,
            fixed_rule=rule,
            validation_rules=(rule,),
            objective=None,
            selected_rule_source=None,
            configured_payload=rule.to_dict(),
        )
    if rule_type == "calibrated":
        _validate_calibrated_refined_output_setup(config)
        objective = str(raw_rule.get("objective", "")).lower()
        if objective != "val_topology_loss":
            raise ValueError(
                "graph_selection.refined_output_rule.objective must be val_topology_loss"
            )
        grid = _probability_sequence(
            raw_rule.get("grid"),
            "graph_selection.refined_output_rule.grid",
        )
        validation_rules = tuple(GraphRule(type="threshold", value=value) for value in grid)
        return RefinedOutputRuleConfig(
            calibrated=True,
            fixed_rule=validation_rules[0],
            validation_rules=validation_rules,
            objective=objective,
            selected_rule_source="validation_calibration",
            configured_payload={
                "type": "calibrated",
                "objective": objective,
                "grid": list(grid),
            },
        )
    raise ValueError("graph_selection.refined_output_rule.type must be threshold or calibrated")


def _resolve_refined_output_rule(config: Mapping[str, object]) -> GraphRule:
    return _resolve_refined_output_rule_config(config).fixed_rule


def _validate_calibrated_refined_output_setup(config: Mapping[str, object]) -> None:
    refiner_cfg = _mapping_section(config, "refiner")
    monitor_metric = str(refiner_cfg.get("monitor_metric", "val_auprc"))
    if monitor_metric != "val_topology_loss":
        raise ValueError(
            "graph_selection.refined_output_rule.type=calibrated requires "
            "refiner.monitor_metric: val_topology_loss"
        )
    topology_validation = refiner_cfg.get("topology_validation", {})
    if not isinstance(topology_validation, Mapping):
        raise ValueError("refiner.topology_validation must be a mapping")
    if not bool(topology_validation.get("enabled", False)):
        raise ValueError(
            "graph_selection.refined_output_rule.type=calibrated requires "
            "refiner.topology_validation.enabled: true"
        )
```

Add this sequence helper beside `_probability`:

```python
def _probability_sequence(value: object, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence")
    parsed = tuple(_probability(item, field_name) for item in value)
    if not parsed:
        raise ValueError(f"{field_name} must not be empty")
    return parsed
```

- [ ] **Step 4: Run parser tests and verify they pass**

Run:

```bash
rtk proxy uv run python -m pytest tests/unit/test_tccig_topology_training.py::test_resolve_refined_output_rule_config_accepts_calibrated_grid tests/unit/test_tccig_topology_training.py::test_resolve_refined_output_rule_config_preserves_threshold_default tests/unit/test_tccig_topology_training.py::test_resolve_refined_output_rule_config_rejects_invalid_calibrated_objective tests/unit/test_tccig_topology_training.py::test_resolve_refined_output_rule_config_rejects_empty_calibrated_grid tests/unit/test_tccig_topology_training.py::test_resolve_refined_output_rule_config_requires_topology_monitor_setup -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
rtk git add tccig/train.py tests/unit/test_tccig_topology_training.py
rtk git commit -m "feat: parse calibrated refined output rule"
```

## Task 2: Evaluate Validation Topology Across a Threshold Grid

**Files:**
- Modify: `tccig/prepare.py`
- Modify: `tccig/s2gae.py`
- Test: `tests/unit/test_tccig_topology_training.py`

- [ ] **Step 1: Write failing validation-grid selection test**

Append this test near the parser tests in `tests/unit/test_tccig_topology_training.py`.

```python
def test_validation_topology_evaluation_selects_grid_argmin(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from tccig.prepare import CandidatePair, GraphRule
    from tccig import s2gae

    prediction_calls = 0

    def fake_prediction_probabilities(**_kwargs: object) -> list[float]:
        nonlocal prediction_calls
        prediction_calls += 1
        return [0.2, 0.8]

    def fake_validation_topology_metrics(**kwargs: object) -> dict[str, float | int]:
        rule = kwargs["rule"]
        assert isinstance(rule, GraphRule)
        losses = {0.5: 10.0, 0.9: 2.0, 0.97: 5.0}
        return {
            "val_topology_loss": losses[float(rule.value)],
            "graph_sim": 0.1 + float(rule.value),
            "relative_density": 1.0,
            "deg_dist_mmd": 0.0,
            "cc_mmd": 0.0,
            "positive_edges": int(float(rule.value) * 100),
            "val_auprc": 0.42,
        }

    monkeypatch.setattr(s2gae, "_prediction_probabilities", fake_prediction_probabilities)
    monkeypatch.setattr(s2gae, "_validation_topology_metrics", fake_validation_topology_metrics)

    cfg = SimpleNamespace(
        topology_validation=SimpleNamespace(inference_batch_size=8),
    )

    result = s2gae._evaluate_validation_topology_rules(
        model=object(),  # type: ignore[arg-type]
        graph=object(),  # type: ignore[arg-type]
        pairs=[
            CandidatePair(protein_a="A", protein_b="B"),
            CandidatePair(protein_a="C", protein_b="D"),
        ],
        validation_plan=object(),  # type: ignore[arg-type]
        rules=(
            GraphRule(type="threshold", value=0.5),
            GraphRule(type="threshold", value=0.9),
            GraphRule(type="threshold", value=0.97),
        ),
        validation_auprc=0.42,
        cfg=cfg,  # type: ignore[arg-type]
        runtime=object(),
        rule_payload_source="validation_calibration",
    )

    assert prediction_calls == 1
    assert result.rule.to_dict() == {"type": "threshold", "value": 0.9}
    assert result.validation_metrics["val_topology_loss"] == pytest.approx(2.0)
    assert result.rule_payload == {
        "type": "threshold",
        "value": 0.9,
        "source": "validation_calibration",
    }
```

- [ ] **Step 2: Run validation-grid test and verify it fails**

Run:

```bash
rtk proxy uv run python -m pytest tests/unit/test_tccig_topology_training.py::test_validation_topology_evaluation_selects_grid_argmin -v
```

Expected: FAIL because `_evaluate_validation_topology_rules` currently accepts `rule=...`, not `rules=...`.

- [ ] **Step 3: Extend `TrainRefinerRequest` with validation rule metadata**

In `tccig/prepare.py`, add these fields to `TrainRefinerRequest` immediately after `graph_rule`.

```python
    validation_graph_rules: Sequence[GraphRule] | None = None
    selected_rule_source: str | None = None
```

The full dataclass should keep all existing fields and become:

```python
@dataclass(frozen=True)
class TrainRefinerRequest:
    """Concrete request for S2GAE refiner training."""

    train: SplitBundle
    validation: SplitBundle
    runtime: TCCIGRuntime
    config: Mapping[str, object]
    graph_rule: GraphRule
    validation_graph_rules: Sequence[GraphRule] | None = None
    selected_rule_source: str | None = None
    validation_topology: SplitBundle | None = None
    validation_topology_plan: object | None = None
    train_topology: SplitBundle | None = None
    train_topology_plan: object | None = None
    train_topology_diagnostic_full_space: Mapping[str, Mapping[str, float]] | None = None
```

- [ ] **Step 4: Replace validation topology rule evaluation helper**

In `tccig/s2gae.py`, replace `_evaluate_validation_topology_rules` with this implementation.

```python
def _evaluate_validation_topology_rules(
    *,
    model: S2GAERefiner,
    graph: _SplitGraph,
    pairs: Sequence[CandidatePair],
    validation_plan: InternalValidationPlan,
    rules: Sequence[GraphRule],
    validation_auprc: float,
    cfg: S2GAEConfig,
    runtime: object,
    rule_payload_source: str | None = None,
) -> ValidationTopologyRuleEvaluation:
    if not rules:
        raise ValueError("validation topology requires at least one graph rule")
    refined_probabilities = _prediction_probabilities(
        model=model,
        graph=graph,
        batch_size=cfg.topology_validation.inference_batch_size,
        runtime=runtime,
    )
    if len(refined_probabilities) != len(pairs):
        raise ValueError("validation topology probabilities must match candidate pairs")

    evaluations: list[ValidationTopologyRuleEvaluation] = []
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
        evaluations.append(
            ValidationTopologyRuleEvaluation(
                rule=rule,
                validation_metrics=metrics,
                rule_payload=payload,
            )
        )
    return min(
        evaluations,
        key=lambda item: float(item.validation_metrics["val_topology_loss"]),
    )
```

- [ ] **Step 5: Update the train loop call site**

In `tccig/s2gae.py`, before the epoch loop or inside the validation topology block, compute the active rule sequence from the request. Add this just before `_evaluate_validation_topology_rules` is called.

```python
            validation_rules = (
                tuple(request.validation_graph_rules)
                if request.validation_graph_rules is not None
                else (request.graph_rule,)
            )
```

Then replace the `_evaluate_validation_topology_rules` call arguments with:

```python
            topology_evaluation = _evaluate_validation_topology_rules(
                model=validation_model,
                graph=validation_topology_graph,
                pairs=request.validation_topology.pairs,
                validation_plan=cast(InternalValidationPlan, request.validation_topology_plan),
                rules=validation_rules,
                validation_auprc=validation_auprc,
                cfg=cfg,
                runtime=request.runtime,
                rule_payload_source=request.selected_rule_source,
            )
```

- [ ] **Step 6: Persist nested selected rule in epoch history**

In `tccig/s2gae.py`, widen the history type and epoch-history type:

```python
history: list[dict[str, object]] = []
```

```python
epoch_history: dict[str, object] = {
```

Inside the existing `if selected_epoch_topology_metrics is not None:` block, add the selected rule object to `epoch_history`.

```python
                    "selected_rule": (
                        None
                        if selected_epoch_rule_payload is None
                        else dict(selected_epoch_rule_payload)
                    ),
```

The update block should include the existing numeric fields and the new nested field:

```python
            epoch_history.update(
                {
                    "val_topology_loss": float(metrics["val_topology_loss"]),
                    "internal_val_graph_sim": float(metrics["graph_sim"]),
                    "internal_val_relative_density": float(metrics["relative_density"]),
                    "internal_val_deg_dist_mmd": float(metrics["deg_dist_mmd"]),
                    "internal_val_cc_mmd": float(metrics["cc_mmd"]),
                    "selected_rule_positive_edges": float(metrics["positive_edges"]),
                    "selected_rule": (
                        None
                        if selected_epoch_rule_payload is None
                        else dict(selected_epoch_rule_payload)
                    ),
                }
            )
```

This nested `selected_rule` is intentionally additive for every run with topology validation enabled, including fixed-threshold mode. It is not calibrated-only. The CSV schema stays unchanged.

- [ ] **Step 7: Update training-summary type annotations**

Find `_write_training_summary` in `tccig/s2gae.py` and change its `history` parameter from `Sequence[Mapping[str, float | int]]` to:

```python
history: Sequence[Mapping[str, object]],
```

Update the typing import at the top of `tccig/s2gae.py`:

```python
from typing import SupportsFloat, SupportsInt, cast
```

Find `_append_tccig_train_csv_row` and `_log_epoch_summary`, change only their `epoch_history` parameter type to:

```python
epoch_history: Mapping[str, object],
```

Add explicit numeric access helpers before `_append_tccig_train_csv_row`:

```python
def _epoch_float(epoch_history: Mapping[str, object], key: str) -> float:
    return float(cast(SupportsFloat, epoch_history[key]))


def _epoch_int(epoch_history: Mapping[str, object], key: str) -> int:
    return int(cast(SupportsInt, epoch_history[key]))
```

Then update every numeric read in `_append_tccig_train_csv_row` to use those helpers. Use this replacement body for the `writer.writerow(...)` call:

```python
        writer.writerow(
            {
                "Epoch": _epoch_int(epoch_history, "epoch"),
                "Epoch Time": epoch_time_s,
                "Train Loss": _epoch_float(epoch_history, "train_loss"),
                "Train BCE Loss": _epoch_float(epoch_history, "train_bce_loss"),
                "Train Residual Anchor Loss": _epoch_float(
                    epoch_history, "train_residual_anchor_loss"
                ),
                "Train Weighted Residual Anchor Loss": _epoch_float(
                    epoch_history, "train_weighted_residual_anchor_loss"
                ),
                "Train Gradient Norm": _epoch_float(epoch_history, "train_gradient_norm"),
                "Val auprc": _epoch_float(epoch_history, "val_auprc"),
                "Val Topology Loss": epoch_history.get("val_topology_loss", ""),
                "Internal Val graph_sim": epoch_history.get("internal_val_graph_sim", ""),
                "Internal Val relative_density": epoch_history.get(
                    "internal_val_relative_density",
                    "",
                ),
                "Internal Val deg_dist_mmd": epoch_history.get(
                    "internal_val_deg_dist_mmd",
                    "",
                ),
                "Internal Val cc_mmd": epoch_history.get("internal_val_cc_mmd", ""),
                "Selected Rule Type": "" if selected_rule is None else selected_rule.type,
                "Selected Rule Positive Edges": epoch_history.get(
                    "selected_rule_positive_edges",
                    "",
                ),
                "Monitor Metric": monitor_metric,
                "Monitor Value": _epoch_float(epoch_history, "monitor_value"),
                "Peak GPU Mem MB": _epoch_float(epoch_history, "peak_gpu_mem_mb"),
                "Learning Rate": _epoch_float(epoch_history, "learning_rate"),
            }
        )
```

Update numeric reads in `_log_epoch_summary` the same way:

```python
        _epoch_int(epoch_history, "epoch"),
        epoch_time_s,
        _epoch_float(epoch_history, "train_loss"),
        _epoch_float(epoch_history, "train_bce_loss"),
        _epoch_float(epoch_history, "val_auprc"),
        monitor_metric,
        _epoch_float(epoch_history, "monitor_value"),
        _format_optional_epoch_value(epoch_history.get("val_topology_loss")),
        "none" if selected_rule is None else selected_rule.type,
        _epoch_float(epoch_history, "learning_rate"),
        _epoch_float(epoch_history, "peak_gpu_mem_mb"),
```

Do not leave `float(epoch_history["..."])` or `int(epoch_history["..."])` on `Mapping[str, object]`; `mypy` will reject those.

- [ ] **Step 8: Run validation-grid tests**

Run:

```bash
rtk proxy uv run python -m pytest tests/unit/test_tccig_topology_training.py::test_validation_topology_evaluation_selects_grid_argmin tests/unit/test_tccig_topology_training.py::test_train_refiner_request_accepts_train_topology_fields -v
```

Expected: PASS.

- [ ] **Step 9: Commit Task 2**

Run:

```bash
rtk git add tccig/prepare.py tccig/s2gae.py tests/unit/test_tccig_topology_training.py
rtk git commit -m "feat: calibrate validation topology threshold"
```

## Task 3: Wire Calibrated Selected Rule Through Pipeline and Manifest

**Files:**
- Modify: `tccig/train.py`
- Test: `tests/unit/test_tccig_topology_training.py`
- Test: `tests/integration/test_tccig_orchestrator.py`

- [ ] **Step 1: Write failing unit tests for effective rule selection**

Append these tests near the refined-output parser tests in `tests/unit/test_tccig_topology_training.py`.

```python
def test_effective_refined_output_rule_uses_checkpoint_rule_for_calibrated() -> None:
    from types import SimpleNamespace
    from tccig.prepare import GraphRule

    config = tccig_train._resolve_refined_output_rule_config(_calibrated_pipeline_config())
    state = SimpleNamespace(selected_rule=GraphRule(type="threshold", value=0.97))

    effective = tccig_train._effective_refined_output_rule(
        refined_rule_config=config,
        refiner_state=state,
    )

    assert effective.to_dict() == {"type": "threshold", "value": 0.97}


def test_effective_refined_output_rule_rejects_missing_calibrated_selected_rule() -> None:
    from types import SimpleNamespace

    config = tccig_train._resolve_refined_output_rule_config(_calibrated_pipeline_config())
    state = SimpleNamespace(selected_rule=None)

    with pytest.raises(RuntimeError, match="selected_rule"):
        tccig_train._effective_refined_output_rule(
            refined_rule_config=config,
            refiner_state=state,
        )


def test_effective_refined_output_rule_keeps_fixed_threshold_config() -> None:
    from types import SimpleNamespace
    from tccig.prepare import GraphRule

    config = tccig_train._resolve_refined_output_rule_config(
        {"graph_selection": {"refined_output_rule": {"type": "threshold", "value": 0.75}}}
    )
    state = SimpleNamespace(selected_rule=GraphRule(type="threshold", value=0.97))

    effective = tccig_train._effective_refined_output_rule(
        refined_rule_config=config,
        refiner_state=state,
    )

    assert effective.to_dict() == {"type": "threshold", "value": 0.75}
```

- [ ] **Step 2: Write failing pipeline plumbing test**

In `tests/integration/test_tccig_orchestrator.py`, add `from types import SimpleNamespace` near the imports.

Append this test after `test_tccig_orchestrator_runs_validation_topology_with_pring_train_test_split`.

```python
def test_calibrated_pipeline_uses_selected_rule_for_test_paths_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tccig.prepare import GraphRule
    import tccig.test as tccig_test

    config = _tiny_config(tmp_path, "calibrated_plumbing")
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
    graph_selection["rules"] = [{"type": "threshold", "value": 0.5}]

    captured: dict[str, object] = {}
    selected_rule = GraphRule(type="threshold", value=0.97)

    def fake_train_refiner(request: object) -> object:
        captured["validation_graph_rules"] = [
            rule.to_dict() for rule in request.validation_graph_rules  # type: ignore[attr-defined]
        ]
        captured["selected_rule_source"] = request.selected_rule_source  # type: ignore[attr-defined]
        return SimpleNamespace(
            selected_rule=selected_rule,
            selected_rule_payload={"type": "threshold", "value": 0.97, "source": "validation_calibration"},
            best_validation_auprc=0.0,
            best_monitor_value=0.0,
        )

    def fake_pairwise_test(**kwargs: object) -> dict[str, float]:
        rule = kwargs["refined_output_rule"]
        assert isinstance(rule, GraphRule)
        captured["pairwise_rule"] = rule.to_dict()
        return {"auprc": 1.0, "auroc": 1.0, "f1": 1.0, "threshold": float(rule.value)}

    def fake_topology_test(**kwargs: object) -> dict[str, float]:
        rule = kwargs["refined_output_rule"]
        assert isinstance(rule, GraphRule)
        captured["topology_rule"] = rule.to_dict()
        return {
            "graph_sim": 1.0,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.0,
            "cc_mmd": 0.0,
            "laplacian_eigen_mmd": 0.0,
        }

    monkeypatch.setattr(s2gae, "train_refiner", fake_train_refiner)
    monkeypatch.setattr(tccig_test, "run_pairwise_test", fake_pairwise_test)
    monkeypatch.setattr(tccig_test, "run_topology_test", fake_topology_test)

    result = run_tccig_pipeline(config)

    assert captured["validation_graph_rules"] == [
        {"type": "threshold", "value": 0.5},
        {"type": "threshold", "value": 0.97},
    ]
    assert captured["selected_rule_source"] == "validation_calibration"
    assert captured["pairwise_rule"] == {"type": "threshold", "value": 0.97}
    assert captured["topology_rule"] == {"type": "threshold", "value": 0.97}
    assert result.refined_output_rule == {"type": "threshold", "value": 0.97}

    manifest = json.loads(
        (tmp_path / "logs" / "tccig" / "calibrated_plumbing" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["refined_output_rule"] == {"type": "threshold", "value": 0.97}
    assert manifest["configured_refined_output_rule"] == {
        "type": "calibrated",
        "objective": "val_topology_loss",
        "grid": [0.5, 0.97],
    }
    assert manifest["ignored_legacy_rules"] == [{"type": "threshold", "value": 0.5}]
```

Append this artifact-level test after `test_calibrated_pipeline_uses_selected_rule_for_test_paths_and_manifest`. Unlike the plumbing test above, this runs the tiny train loop and proves the per-epoch JSON artifact and checkpoint carry the selected calibrated rule.

```python
def test_calibrated_pipeline_persists_epoch_selected_rule_history(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, "calibrated_history")
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
    graph_selection["rules"] = [{"type": "threshold", "value": 0.5}]

    run_tccig_pipeline(config)

    summary_path = tmp_path / "logs" / "tccig" / "calibrated_history" / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = summary["history"][0]

    assert row["selected_rule"]["type"] == "threshold"
    assert row["selected_rule"]["source"] == "validation_calibration"
    assert row["selected_rule"]["value"] in {0.5, 0.97}
    assert summary["selected_rule"] == row["selected_rule"]

    checkpoint_path = tmp_path / "models" / "tccig" / "calibrated_history" / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["selected_rule"] == summary["selected_rule"]
```

- [ ] **Step 3: Run plumbing tests and verify they fail**

Run:

```bash
rtk proxy uv run python -m pytest tests/unit/test_tccig_topology_training.py::test_effective_refined_output_rule_uses_checkpoint_rule_for_calibrated tests/unit/test_tccig_topology_training.py::test_effective_refined_output_rule_rejects_missing_calibrated_selected_rule tests/unit/test_tccig_topology_training.py::test_effective_refined_output_rule_keeps_fixed_threshold_config tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_uses_selected_rule_for_test_paths_and_manifest tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_persists_epoch_selected_rule_history -v
```

Expected: FAIL because `_effective_refined_output_rule` is missing, `run_tccig_pipeline` still passes the configured fixed rule to tests, and epoch history does not yet persist the nested `selected_rule`.

- [ ] **Step 4: Add effective-rule and legacy-rule helpers**

In `tccig/train.py`, add these helpers after `_resolve_refined_output_rule`.

```python
def _effective_refined_output_rule(
    *,
    refined_rule_config: RefinedOutputRuleConfig,
    refiner_state: object,
) -> GraphRule:
    if not refined_rule_config.calibrated:
        return refined_rule_config.fixed_rule
    selected_rule = getattr(refiner_state, "selected_rule", None)
    if not isinstance(selected_rule, GraphRule):
        raise RuntimeError("Calibrated refined-output mode requires refiner_state.selected_rule")
    return selected_rule


def _ignored_legacy_rules_payload(
    *,
    config: Mapping[str, object],
    refined_rule_config: RefinedOutputRuleConfig,
) -> list[dict[str, object]] | None:
    if not refined_rule_config.calibrated:
        return None
    raw_rules = _graph_selection(config).get("rules")
    if raw_rules is None:
        return None
    if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes)):
        raise ValueError("graph_selection.rules must be a sequence")
    payload: list[dict[str, object]] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, Mapping):
            raise ValueError("graph_selection.rules entries must be mappings")
        payload.append(dict(raw_rule))
    return payload
```

- [ ] **Step 5: Wire parsed config through `run_tccig_pipeline`**

In `tccig/train.py`, replace:

```python
    refined_output_rule = _resolve_refined_output_rule(config)
    parsed_rules = parse_rules(
        _graph_selection(config).get("rules", [refined_output_rule.to_dict()])
    )
    graph_rule = parsed_rules[0]
```

with:

```python
    refined_rule_config = _resolve_refined_output_rule_config(config)
    if refined_rule_config.calibrated:
        validation_graph_rules = refined_rule_config.validation_rules
    else:
        validation_graph_rules = tuple(
            parse_rules(
                _graph_selection(config).get(
                    "rules",
                    [refined_rule_config.fixed_rule.to_dict()],
                )
            )
        )
    graph_rule = validation_graph_rules[0]
```

Then pass the grid and source into `TrainRefinerRequest`:

```python
            graph_rule=graph_rule,
            validation_graph_rules=validation_graph_rules,
            selected_rule_source=refined_rule_config.selected_rule_source,
```

Immediately after `refiner_state = s2gae.train_refiner(...)`, add:

```python
    refined_output_rule = _effective_refined_output_rule(
        refined_rule_config=refined_rule_config,
        refiner_state=refiner_state,
    )
```

Keep both test calls passing `refined_output_rule=refined_output_rule`.

Replace the manifest construction with:

```python
    manifest: dict[str, object] = {
        "run_id": run_id,
        "self_pair_rows_dropped": {split: table.self_pair_rows for split, table in tables.items()},
        "pairwise_input_threshold": pairwise_input_payload,
        "refined_output_rule": refined_output_rule.to_dict(),
    }
    if refined_rule_config.calibrated:
        manifest["configured_refined_output_rule"] = dict(refined_rule_config.configured_payload)
        ignored_legacy_rules = _ignored_legacy_rules_payload(
            config=config,
            refined_rule_config=refined_rule_config,
        )
        if ignored_legacy_rules is not None:
            manifest["ignored_legacy_rules"] = ignored_legacy_rules
```

Keep the return object using the effective rule:

```python
        refined_output_rule=refined_output_rule.to_dict(),
```

- [ ] **Step 6: Run plumbing tests and verify they pass**

Run:

```bash
rtk proxy uv run python -m pytest tests/unit/test_tccig_topology_training.py::test_effective_refined_output_rule_uses_checkpoint_rule_for_calibrated tests/unit/test_tccig_topology_training.py::test_effective_refined_output_rule_rejects_missing_calibrated_selected_rule tests/unit/test_tccig_topology_training.py::test_effective_refined_output_rule_keeps_fixed_threshold_config tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_uses_selected_rule_for_test_paths_and_manifest tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_persists_epoch_selected_rule_history -v
```

Expected: PASS.

- [ ] **Step 7: Run unchanged threshold-mode orchestrator test**

Run:

```bash
rtk proxy uv run python -m pytest tests/integration/test_tccig_orchestrator.py::test_tccig_orchestrator_runs_concrete_pipeline_and_writes_artifacts -v
```

Expected: PASS, including `result.refined_output_rule == {"type": "threshold", "value": 0.5}`.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
rtk git add tccig/train.py tests/unit/test_tccig_topology_training.py tests/integration/test_tccig_orchestrator.py
rtk git commit -m "feat: use calibrated selected rule for tccig tests"
```

## Task 4: Add Topology-Only Epoch Config and Training Loop Behavior

**Files:**
- Modify: `tccig/s2gae.py`
- Test: `tests/unit/test_tccig_topology_training.py`
- Test: `tests/integration/test_tccig_topology_training_stage.py`

- [ ] **Step 1: Write failing parser test for topo-only config**

Append this test near `test_parse_config_reads_residual_anchor_and_topology_training` in `tests/unit/test_tccig_topology_training.py`.

```python
def test_parse_config_reads_topology_only_epoch_boundary() -> None:
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    topology_training = config["topology_training"]
    assert isinstance(topology_training, dict)
    topology_training["topo_only_after_epoch"] = 7

    cfg = _parse_config(config)

    assert cfg.topology_training.topo_only_after_epoch == 7
```

Append this default test near `test_parse_config_defaults_topology_training_disabled`.

```python
def test_parse_config_defaults_topology_only_epoch_boundary_off() -> None:
    from tccig.s2gae import _parse_config

    cfg = _parse_config(_base_refiner_config())

    assert cfg.topology_training.topo_only_after_epoch is None
```

- [ ] **Step 2: Write failing topo-only integration test**

In `tests/integration/test_tccig_topology_training_stage.py`, append this test after `test_topology_training_run_deletes_edges_and_logs_topology_loss`.

```python
def test_topology_only_epochs_skip_bce_sampling_and_log_zero_bce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tccig.s2gae as s2gae_module

    config = _topology_training_config(tmp_path, "02_topology_only")
    refiner_config = config["refiner"]
    assert isinstance(refiner_config, dict)
    refiner_config["epochs"] = 1
    topology_training = refiner_config["topology_training"]
    assert isinstance(topology_training, dict)
    topology_training["topo_only_after_epoch"] = 1
    topology_training["schedule"] = {"warmup_epochs": 0, "ramp_epochs": 0, "schedule": "linear"}

    def fail_sample_epoch_edge_targets(**_kwargs: object) -> list[object]:
        raise AssertionError("sample_epoch_edge_targets must not run on topo-only epochs")

    step_calls = 0
    original_step = torch.optim.AdamW.step

    def counting_step(self: torch.optim.AdamW, *args: object, **kwargs: object) -> object:
        nonlocal step_calls
        step_calls += 1
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(s2gae_module, "sample_epoch_edge_targets", fail_sample_epoch_edge_targets)
    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)

    run_tccig_pipeline(config)

    summary = json.loads(
        (
            tmp_path
            / "logs"
            / "tccig"
            / "02_topology_only"
            / "training_summary.json"
        ).read_text(encoding="utf-8")
    )
    row = summary["history"][0]

    assert step_calls >= 1
    assert row["sampled_edge_targets"] == 0
    assert row["train_loss"] == 0.0
    assert row["train_bce_loss"] == 0.0
    assert row["train_residual_anchor_loss"] == 0.0
    assert row["train_weighted_residual_anchor_loss"] == 0.0
    assert row["train_topology_scale"] == 1.0
    assert "train_gradient_norm" in row
    assert "train_topology_loss" in row
```

- [ ] **Step 3: Run topo-only tests and verify they fail**

Run:

```bash
rtk proxy uv run python -m pytest tests/unit/test_tccig_topology_training.py::test_parse_config_reads_topology_only_epoch_boundary tests/unit/test_tccig_topology_training.py::test_parse_config_defaults_topology_only_epoch_boundary_off tests/integration/test_tccig_topology_training_stage.py::test_topology_only_epochs_skip_bce_sampling_and_log_zero_bce -v
```

Expected: FAIL because `topo_only_after_epoch` is not parsed and the BCE sampling phase always runs.

- [ ] **Step 4: Add topo-only config field**

In `tccig/s2gae.py`, add this field to `S2GAETopologyTrainingConfig`:

```python
    topo_only_after_epoch: int | None
```

Update the disabled default return in `_parse_topology_training_config`:

```python
            topo_only_after_epoch=None,
```

Update the enabled return in `_parse_topology_training_config`:

```python
        topo_only_after_epoch=_optional_positive_int(
            raw.get("topo_only_after_epoch"),
            "refiner.topology_training.topo_only_after_epoch",
        ),
```

Add this helper beside `_positive_int` / `_non_negative_int` in `tccig/s2gae.py`:

```python
def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)
```

Update `_config_to_json` under `"topology_training"`:

```python
            "topo_only_after_epoch": cfg.topology_training.topo_only_after_epoch,
```

- [ ] **Step 5: Restructure the BCE phase in the train loop**

In `tccig/s2gae.py`, replace the start of the epoch body from `epoch_targets = sample_epoch_edge_targets(...)` through the BCE loop with this structure.

```python
        topo_only_epoch = (
            cfg.topology_training.topo_only_after_epoch is not None
            and epoch >= cfg.topology_training.topo_only_after_epoch
        )
        epoch_targets: list[EdgeTarget] = []
        local_loss_sums = torch.zeros(5, dtype=torch.float64, device=device)
        gradient_norm = 0.0
        if not topo_only_epoch:
            epoch_targets = sample_epoch_edge_targets(
                quadrants=quadrants,
                sampling=cfg.edge_sampling,
                epoch=epoch,
            )
            loader = DataLoader(
                EdgeTargetDataset(epoch_targets),
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=collate_edge_targets,
            )
            prepared_loader = request.runtime.accelerator.prepare(loader)
            for batch in cast(Iterable[Mapping[str, torch.Tensor]], prepared_loader):
                optimizer.zero_grad(set_to_none=True)
                batch_loss, batch_sums = cast(
                    tuple[torch.Tensor, torch.Tensor],
                    train_step_model(
                        graph=train_graph,
                        pair_indices=batch["pair_index"].to(device),
                        labels=batch["label"].to(device),
                        mask_input_edges=batch["mask_input_edge"].to(device),
                    ),
                )
                request.runtime.accelerator.backward(batch_loss)
                if cfg.optimization.gradient_clip_norm is None:
                    gradient_norm = apply_gradient_clipping(
                        model=train_step_model, gradient_clip_norm=None
                    )
                else:
                    clipped = request.runtime.accelerator.clip_grad_norm_(
                        train_step_model.parameters(),
                        cfg.optimization.gradient_clip_norm,
                    )
                    gradient_norm = float(clipped.detach().cpu().item())
                optimizer.step()
                local_loss_sums += batch_sums.detach()
```

Keep the topology phase that follows structurally unchanged except for indentation alignment, but record the topology-step gradient norm when BCE is skipped. In the topology phase, replace the existing clip-and-step tail:

```python
                if cfg.optimization.gradient_clip_norm is not None:
                    request.runtime.accelerator.clip_grad_norm_(
                        train_step_model.parameters(), cfg.optimization.gradient_clip_norm
                    )
                optimizer.step()
```

with:

```python
                topology_gradient_norm = 0.0
                if cfg.optimization.gradient_clip_norm is not None:
                    clipped = request.runtime.accelerator.clip_grad_norm_(
                        train_step_model.parameters(), cfg.optimization.gradient_clip_norm
                    )
                    topology_gradient_norm = float(clipped.detach().cpu().item())
                else:
                    topology_gradient_norm = apply_gradient_clipping(
                        model=train_step_model,
                        gradient_clip_norm=None,
                    )
                if topo_only_epoch:
                    gradient_norm = topology_gradient_norm
                optimizer.step()
```

This keeps the existing mixed BCE+topology logging behavior unchanged: normal epochs still report the BCE-loop gradient norm. Topology-only epochs report the topology-step norm instead of a misleading default `0.0`.

DDP note for the first HPC rerun: topo-only epochs do not run a forward/backward through the DDP-wrapped `train_step_model`; they only run the existing unwrapped topology backward path. That path already owns the topology-step rank-synchronization/determinism invariant. The single-process integration test will not exercise multi-GPU synchronization, so inspect the first multi-GPU rerun logs for rank hangs/divergence before trusting long runs.

- [ ] **Step 6: Run topo-only tests and verify they pass**

Run:

```bash
rtk proxy uv run python -m pytest tests/unit/test_tccig_topology_training.py::test_parse_config_reads_topology_only_epoch_boundary tests/unit/test_tccig_topology_training.py::test_parse_config_defaults_topology_only_epoch_boundary_off tests/integration/test_tccig_topology_training_stage.py::test_topology_only_epochs_skip_bce_sampling_and_log_zero_bce -v
```

Expected: PASS.

- [ ] **Step 7: Run existing topology-training integration test**

Run:

```bash
rtk proxy uv run python -m pytest tests/integration/test_tccig_topology_training_stage.py::test_topology_training_run_deletes_edges_and_logs_topology_loss -v
```

Expected: PASS, proving default `topo_only_after_epoch: null` leaves existing topology training behavior intact.

- [ ] **Step 8: Commit Task 4**

Run:

```bash
rtk git add tccig/s2gae.py tests/unit/test_tccig_topology_training.py tests/integration/test_tccig_topology_training_stage.py
rtk git commit -m "feat: add topology-only training epochs"
```

## Task 5: Update Exp02 Rerun Config

**Files:**
- Modify: `configs/tccig/02_balanced_subset.yaml`
- Test: `tests/unit/test_tccig_topology_training.py`

- [ ] **Step 1: Write failing config artifact test**

Append this test near `test_training_summary_artifact_persists_topology_subset_block` in `tests/unit/test_tccig_topology_training.py`.

```python
def test_exp02_balanced_subset_config_uses_calibrated_rule_and_topology_only_probe() -> None:
    import yaml

    raw = yaml.safe_load(
        Path("configs/tccig/02_balanced_subset.yaml").read_text(encoding="utf-8")
    )

    graph_selection = raw["graph_selection"]
    refined_rule = graph_selection["refined_output_rule"]
    assert refined_rule == {
        "type": "calibrated",
        "objective": "val_topology_loss",
        "grid": [0.5, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99],
    }

    topology_training = raw["refiner"]["topology_training"]
    assert topology_training["topo_only_after_epoch"] == 7
    assert raw["refiner"]["monitor_metric"] == "val_topology_loss"
    assert raw["refiner"]["topology_validation"]["enabled"] is True
```

- [ ] **Step 2: Run config test and verify it fails**

Run:

```bash
rtk proxy uv run python -m pytest tests/unit/test_tccig_topology_training.py::test_exp02_balanced_subset_config_uses_calibrated_rule_and_topology_only_probe -v
```

Expected: FAIL because the YAML still uses fixed threshold `0.5` and no `topo_only_after_epoch`.

- [ ] **Step 3: Update topology training config**

In `configs/tccig/02_balanced_subset.yaml`, add this key under `refiner.topology_training`, near `topology_weight`.

```yaml
    topo_only_after_epoch: 7
```

- [ ] **Step 4: Update refined-output rule config**

In `configs/tccig/02_balanced_subset.yaml`, replace:

```yaml
  refined_output_rule:
    type: threshold
    value: 0.5
```

with:

```yaml
  refined_output_rule:
    type: calibrated
    objective: val_topology_loss
    grid: [0.5, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99]
```

Leave the existing `graph_selection.rules` block in place for this rerun. It exercises the spec-required `ignored_legacy_rules` manifest behavior and documents that calibrated mode overrides the legacy fixed rule.

- [ ] **Step 5: Run config test and parser smoke test**

Run:

```bash
rtk proxy uv run python -m pytest tests/unit/test_tccig_topology_training.py::test_exp02_balanced_subset_config_uses_calibrated_rule_and_topology_only_probe tests/unit/test_tccig_topology_training.py::test_training_summary_artifact_persists_topology_subset_block -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
rtk git add configs/tccig/02_balanced_subset.yaml tests/unit/test_tccig_topology_training.py
rtk git commit -m "chore: configure exp02 rerun calibration probe"
```

## Task 6: Full Verification and Type/Lint Cleanup

**Files:**
- Modify only files that fail verification.
- Test: touched unit and integration tests.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
rtk proxy uv run python -m pytest tests/unit/test_tccig_topology_training.py tests/unit/test_tccig_test_export.py -v
```

Expected: PASS.

- [ ] **Step 2: Run focused integration tests**

Run:

```bash
rtk proxy uv run python -m pytest tests/integration/test_tccig_orchestrator.py::test_tccig_orchestrator_runs_concrete_pipeline_and_writes_artifacts tests/integration/test_tccig_orchestrator.py::test_tccig_orchestrator_runs_validation_topology_with_pring_train_test_split tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_uses_selected_rule_for_test_paths_and_manifest tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_persists_epoch_selected_rule_history tests/integration/test_tccig_topology_training_stage.py::test_topology_training_run_deletes_edges_and_logs_topology_loss tests/integration/test_tccig_topology_training_stage.py::test_topology_only_epochs_skip_bce_sampling_and_log_zero_bce -v
```

Expected: PASS.

- [ ] **Step 3: Run ruff on touched files**

Run:

```bash
rtk proxy uv run ruff check tccig/s2gae.py tccig/train.py tccig/test.py tccig/prepare.py tests/unit/test_tccig_topology_training.py tests/integration/test_tccig_orchestrator.py tests/integration/test_tccig_topology_training_stage.py
```

Expected: PASS.

- [ ] **Step 4: Run mypy on production modules**

Run:

```bash
rtk proxy uv run mypy tccig src
```

Expected: PASS. If `mypy` flags `epoch_history` object-valued history, keep the nested `selected_rule` field and fix types by widening local annotations to `Mapping[str, object]`; do not remove the JSON field.

- [ ] **Step 5: Run full pytest if focused tests are clean**

Run:

```bash
rtk proxy uv run python -m pytest
```

Expected: PASS.

- [ ] **Step 6: Run post-rerun artifact proof checklist**

Run this after the fixed `02_balanced_subset` HPC rerun finishes and its artifacts are available locally. This is not a pre-rerun local unit-test gate; it should fail against the old fixed-threshold artifacts.

Run:

```bash
rtk proxy uv run python - <<'PY'
import json
from pathlib import Path

import torch

run_id = "02_balanced_subset"
log_dir = Path("logs") / "tccig" / run_id
manifest = json.loads((log_dir / "manifest.json").read_text(encoding="utf-8"))
summary = json.loads((log_dir / "training_summary.json").read_text(encoding="utf-8"))

checkpoint_path = Path(str(summary["checkpoint_path"]))
if not checkpoint_path.is_absolute():
    checkpoint_path = Path.cwd() / checkpoint_path
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

configured_rule = manifest["configured_refined_output_rule"]
assert configured_rule["type"] == "calibrated"
assert configured_rule["objective"] == "val_topology_loss"
assert 0.97 in [float(value) for value in configured_rule["grid"]]
assert manifest["ignored_legacy_rules"] == [{"type": "threshold", "value": 0.5}]

manifest_rule = manifest["refined_output_rule"]
summary_rule = summary["selected_rule"]
checkpoint_rule = checkpoint["selected_rule"]
assert manifest_rule["type"] == summary_rule["type"] == checkpoint_rule["type"]
assert abs(float(manifest_rule["value"]) - float(summary_rule["value"])) < 1e-12
assert checkpoint_rule == summary_rule
assert summary_rule["source"] == "validation_calibration"

history = summary["history"]
topology_rows = [row for row in history if "val_topology_loss" in row]
assert topology_rows, "expected topology-validation history rows"
for row in topology_rows:
    rule = row.get("selected_rule")
    assert rule is not None, f"epoch {row['epoch']} missing selected_rule"
    assert rule["type"] == "threshold"
    assert rule["source"] == "validation_calibration"
    assert float(rule["value"]) in {0.5, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99}

topo_only_rows = [row for row in history if int(row["epoch"]) >= 7]
assert topo_only_rows, "expected topo-only diagnostic rows"
for row in topo_only_rows:
    assert int(row["sampled_edge_targets"]) == 0
    assert float(row["train_loss"]) == 0.0
    assert float(row["train_bce_loss"]) == 0.0
    assert float(row["train_residual_anchor_loss"]) == 0.0
    assert float(row["train_weighted_residual_anchor_loss"]) == 0.0
    assert "train_topology_loss" in row

loss_rows = [row for row in topo_only_rows if "train_topology_loss" in row]
first_loss = float(loss_rows[0]["train_topology_loss"])
last_loss = float(loss_rows[-1]["train_topology_loss"])
if last_loss < first_loss:
    probe = "decreased"
elif last_loss > first_loss:
    probe = "increased"
else:
    probe = "flat"

print(
    json.dumps(
        {
            "effective_refined_output_rule": manifest_rule,
            "checkpoint_selected_rule": checkpoint_rule,
            "topo_only_start_epoch": topo_only_rows[0]["epoch"],
            "topo_only_first_train_topology_loss": first_loss,
            "topo_only_last_train_topology_loss": last_loss,
            "topo_only_probe": probe,
        },
        indent=2,
        sort_keys=True,
    )
)
PY
```

Expected: PASS after the fixed rerun. The printed `topo_only_probe` is the experiment readout: `decreased` supports BCE-vs-topology conflict and FP reweighting next; `flat` or `increased` points back to the topology objective/gradient.

- [ ] **Step 7: Commit verification cleanup**

Run only if Step 1 through Step 5 required cleanup changes:

```bash
rtk git add tccig/s2gae.py tccig/train.py tccig/test.py tccig/prepare.py tests/unit/test_tccig_topology_training.py tests/integration/test_tccig_orchestrator.py tests/integration/test_tccig_topology_training_stage.py configs/tccig/02_balanced_subset.yaml
rtk git commit -m "test: verify exp02 rerun fix"
```

If no cleanup changes were required, skip this commit.

## Self-Review Checklist

- Spec coverage:
  - Calibrated schema, objective validation, grid validation, topology-monitor guards: Task 1.
  - Per-epoch grid evaluation with one inference pass and argmin `val_topology_loss`: Task 2.
  - Per-epoch nested `selected_rule` JSON and best selected rule persistence: Task 2.
  - Test paths consuming best calibrated `selected_rule`: Task 3.
  - `graph_selection.rules` ignored audibly via manifest: Task 3.
  - Topo-only inclusive epoch boundary and BCE sampling/DataLoader skip: Task 4.
  - Exp02 rerun YAML set to calibrated + `topo_only_after_epoch: 7`: Task 5.
  - ruff, mypy `tccig src`, focused tests, full pytest, and post-rerun artifact proof checklist: Task 6.
- Placeholder scan:
  - Banned placeholder phrases do not appear in the task body.
  - Each code-changing step includes concrete code or an exact replacement.
- Type consistency:
  - `RefinedOutputRuleConfig.validation_rules` is always a tuple of executable `GraphRule` objects.
  - `TrainRefinerRequest.validation_graph_rules` is optional and falls back to the legacy single `graph_rule`.
  - `selected_rule_source` is only a payload label; graph construction still receives a `GraphRule`.
  - `training_summary.json.history[*].selected_rule` requires object-valued history typing.
