"""Trial execution helpers that reuse the existing pipeline runner."""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.optimize.search_space import SearchParameter, apply_search_parameters
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_str,
    as_str_list,
    extract_model_kwargs,
    get_section,
)

Direction = str
RunPipelineFn = Callable[[ConfigDict], None]
ALLOWED_STAGES: tuple[str, ...] = ("train", "evaluate")
STAGE_ORDER: dict[str, int] = {stage: index for index, stage in enumerate(ALLOWED_STAGES)}
TRAINING_STAGES: tuple[str, ...] = ("train",)


@dataclass(frozen=True)
class TrialExecutionResult:
    """Single optimization trial result."""

    trial_number: int
    run_id: str
    objective_value: float
    objective_history: tuple[float, ...]
    metric_column: str
    train_csv_path: Path
    checkpoint_path: Path


def execute_trial(
    *,
    base_config: ConfigDict,
    search_space: list[SearchParameter],
    sampled_values: Mapping[str, object],
    study_name: str,
    trial_number: int,
    objective_metric: str,
    direction: Direction,
    execution_cfg: Mapping[str, object],
    run_pipeline_fn: RunPipelineFn,
) -> TrialExecutionResult:
    """Execute one pipeline trial and return objective metrics.

    Args:
        base_config: Baseline root config.
        search_space: Parsed search-space definitions.
        sampled_values: Sampled search values for this trial.
        study_name: Optimization study name.
        trial_number: Zero-based trial index.
        objective_metric: Objective metric (``val_auprc`` or ``auprc``).
        direction: Optimization direction.
        execution_cfg: ``optimization.execution`` mapping.
        run_pipeline_fn: Callable to run one pipeline config.

    Returns:
        Trial execution result with metric history.
    """
    trial_config = apply_search_parameters(
        base_config=base_config,
        sampled_values=sampled_values,
        search_space=search_space,
    )
    trial_base_run_id = build_trial_run_id(study_name=study_name, trial_number=trial_number)
    _patch_trial_runtime_config(
        config=trial_config,
        execution_cfg=execution_cfg,
        run_id=trial_base_run_id,
    )

    run_pipeline_fn(trial_config)

    run_cfg = get_section(trial_config, "run_config")
    objective_stage = _objective_training_stage(run_cfg)
    objective_run_id = _run_id_for_stage(run_cfg=run_cfg, stage=objective_stage)
    model_name, _ = extract_model_kwargs(trial_config)
    train_csv_path = (
        Path("logs") / model_name / objective_stage / objective_run_id / "training_step.csv"
    )
    objective_history, metric_column = read_objective_history(
        csv_path=train_csv_path,
        objective_metric=objective_metric,
    )
    objective_value = pick_objective_value(history=objective_history, direction=direction)
    checkpoint_path = (
        Path("models") / model_name / objective_stage / objective_run_id / "best_model.pth"
    )
    return TrialExecutionResult(
        trial_number=trial_number,
        run_id=objective_run_id,
        objective_value=objective_value,
        objective_history=tuple(objective_history),
        metric_column=metric_column,
        train_csv_path=train_csv_path,
        checkpoint_path=checkpoint_path,
    )


def run_best_full_pipeline(
    *,
    base_config: ConfigDict,
    search_space: list[SearchParameter],
    best_values: Mapping[str, object],
    study_name: str,
    run_pipeline_fn: RunPipelineFn,
) -> str:
    """Run one full staged pipeline using the best searched parameters.

    Args:
        base_config: Baseline root config.
        search_space: Parsed search-space definitions.
        best_values: Best sampled values from the search.
        study_name: Study name for deterministic run IDs.
        run_pipeline_fn: Callable that executes the pipeline.

    Returns:
        Train run ID used for the final staged pipeline run.
    """
    final_config = apply_search_parameters(
        base_config=base_config,
        sampled_values=best_values,
        search_space=search_space,
    )
    run_cfg = get_section(final_config, "run_config")
    run_cfg["stages"] = ["train", "evaluate"]
    run_cfg["train_run_id"] = f"opt_{study_name}_best_train"
    run_cfg["eval_run_id"] = f"opt_{study_name}_best_eval"

    device_cfg = get_section(final_config, "device_config")
    device_cfg["ddp_enabled"] = False
    run_pipeline_fn(final_config)
    return as_str(run_cfg["train_run_id"], "run_config.train_run_id")


def build_trial_run_id(*, study_name: str, trial_number: int) -> str:
    """Build deterministic trial run ID."""
    safe_study = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in study_name
    )
    return f"opt_{safe_study}_t{trial_number:04d}"


def read_objective_history(*, csv_path: Path, objective_metric: str) -> tuple[list[float], str]:
    """Read objective history from ``training_step.csv``.

    Args:
        csv_path: Training CSV path.
        objective_metric: Target objective metric name.

    Returns:
        Tuple of metric history and resolved CSV column name.

    Raises:
        ValueError: If metric column is missing or empty.
    """
    if not csv_path.exists():
        raise ValueError(f"training_step.csv not found: {csv_path}")

    target_header = objective_metric_to_csv_header(objective_metric)
    history: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"training_step.csv has no headers: {csv_path}")
        resolved_header = _resolve_column_name(reader.fieldnames, target_header)
        for row in reader:
            raw_value = row.get(resolved_header)
            if raw_value is None or raw_value == "":
                continue
            history.append(float(raw_value))

    if not history:
        raise ValueError(f"No numeric values found for objective metric: {target_header}")
    return history, resolved_header


def objective_metric_to_csv_header(metric_name: str) -> str:
    """Convert objective metric to ``training_step.csv`` column header."""
    normalized = metric_name.strip().lower().replace(" ", "_")
    if normalized.startswith("val_"):
        normalized = normalized[4:]
    if not normalized:
        raise ValueError("Objective metric must be a non-empty string")
    return f"Val {normalized}"


def pick_objective_value(*, history: list[float], direction: Direction) -> float:
    """Select objective value from history according to optimization direction."""
    if not history:
        raise ValueError("Objective history must not be empty")
    normalized_direction = direction.lower().strip()
    if normalized_direction == "maximize":
        return max(history)
    if normalized_direction == "minimize":
        return min(history)
    raise ValueError("optimization.direction must be 'maximize' or 'minimize'")


def _resolve_column_name(fieldnames: Sequence[str], expected_header: str) -> str:
    """Resolve a CSV column header ignoring case."""
    normalized_field_map = {field.lower(): field for field in fieldnames}
    key = expected_header.lower()
    if key not in normalized_field_map:
        raise ValueError(
            "Metric column "
            f"'{expected_header}' not found in training_step.csv headers: {fieldnames}"
        )
    return normalized_field_map[key]


def _patch_trial_runtime_config(
    *,
    config: ConfigDict,
    execution_cfg: Mapping[str, object],
    run_id: str,
) -> None:
    """Patch runtime keys for one trial."""
    run_cfg = get_section(config, "run_config")
    run_cfg["stages"] = list(_parse_trial_stages(execution_cfg))
    run_cfg["train_run_id"] = run_id
    run_cfg["eval_run_id"] = f"{run_id}_eval"

    device_cfg = get_section(config, "device_config")
    ddp_per_trial = as_bool(
        execution_cfg.get("ddp_per_trial", False),
        "optimization.execution.ddp_per_trial",
    )
    if not ddp_per_trial:
        device_cfg["ddp_enabled"] = False


def _parse_trial_stages(execution_cfg: Mapping[str, object]) -> tuple[str, ...]:
    """Parse and validate optimization execution stages."""
    raw_stages = execution_cfg.get("trial_stages", ["train"])
    stages = tuple(
        stage.lower() for stage in as_str_list(raw_stages, "optimization.execution.trial_stages")
    )
    if not stages:
        raise ValueError("optimization.execution.trial_stages must not be empty")
    if len(set(stages)) != len(stages):
        raise ValueError("optimization.execution.trial_stages must not contain duplicates")
    unsupported = [stage for stage in stages if stage not in STAGE_ORDER]
    if unsupported:
        raise ValueError(
            "optimization.execution.trial_stages contains unsupported stage(s): "
            f"{', '.join(sorted(set(unsupported)))}"
        )
    previous_order = -1
    for stage in stages:
        stage_order = STAGE_ORDER[stage]
        if stage_order < previous_order:
            raise ValueError("optimization.execution.trial_stages must follow: train -> evaluate")
        previous_order = stage_order
    return stages


def _objective_training_stage(run_cfg: ConfigDict) -> str:
    """Return the last configured training stage used for objective logging."""
    raw_stages = run_cfg.get("stages", ["train"])
    stages = tuple(stage.lower() for stage in as_str_list(raw_stages, "run_config.stages"))
    for stage in reversed(stages):
        if stage in TRAINING_STAGES:
            return stage
    raise ValueError(
        "optimization.execution.trial_stages must include 'train' "
        "to produce training_step.csv objective history"
    )


def _run_id_for_stage(run_cfg: ConfigDict, stage: str) -> str:
    """Return configured run_id key for one stage."""
    stage_key_map = {
        "train": "train_run_id",
    }
    run_id_key = stage_key_map.get(stage)
    if run_id_key is None:
        raise ValueError(f"Unsupported objective stage: {stage}")
    run_id = as_str(run_cfg.get(run_id_key, ""), f"run_config.{run_id_key}")
    if not run_id:
        raise ValueError(f"run_config.{run_id_key} must be non-empty")
    return run_id


__all__ = [
    "Direction",
    "RunPipelineFn",
    "TrialExecutionResult",
    "build_trial_run_id",
    "execute_trial",
    "objective_metric_to_csv_header",
    "pick_objective_value",
    "read_objective_history",
    "run_best_full_pipeline",
]
