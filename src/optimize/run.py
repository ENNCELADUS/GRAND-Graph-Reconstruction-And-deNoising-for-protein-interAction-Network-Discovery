"""CLI entrypoint for automated HPO and NAS-lite workflows."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import yaml  # type: ignore[import-untyped]

from src.optimize.backends.optuna_backend import OptunaResult, TrialRecord, run_optuna_optimization
from src.optimize.search_space import extend_with_nas_lite, parse_search_space
from src.optimize.trial_runner import run_best_full_pipeline
from src.utils.config import ConfigDict, as_bool, as_int, as_str, load_config

LOGGER = logging.getLogger(__name__)
PipelineExecuteFn = Callable[[ConfigDict], None]
PIPELINE_EXECUTE_FN: PipelineExecuteFn


def _resolve_pipeline_execute_fn() -> PipelineExecuteFn:
    """Resolve pipeline execute function from runtime module layout."""
    run_module = importlib.import_module("src.run")
    execute_fn = getattr(run_module, "execute_pipeline", None)
    if callable(execute_fn):
        return cast(PipelineExecuteFn, execute_fn)
    return _fallback_pipeline_execute


def _fallback_pipeline_execute(config: ConfigDict) -> None:
    """Fallback to package orchestrator when ``src.run`` script module is unavailable."""
    from src.run.pipeline_orchestrator import execute_pipeline

    cast(PipelineExecuteFn, execute_pipeline)(config)


PIPELINE_EXECUTE_FN = _resolve_pipeline_execute_fn()


def parse_args() -> argparse.Namespace:
    """Parse optimize CLI arguments."""
    parser = argparse.ArgumentParser(description="Run RELIC optimization workflow")
    parser.add_argument("--config", type=str, required=True, help="Path to base config YAML")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Override optimization.backend (optuna only)",
    )
    parser.add_argument(
        "--skip-final-full-pipeline",
        action="store_true",
        help="Skip running best-params full pipeline after HPO",
    )
    return parser.parse_args()


def run_optimization(
    *,
    config: ConfigDict,
    backend_override: str | None,
    skip_final_full_pipeline: bool,
) -> None:
    """Execute optimization according to ``optimization`` config section."""
    optimization_cfg = _resolve_optimization_config(config)
    backend_name = (
        backend_override.lower().strip()
        if isinstance(backend_override, str) and backend_override.strip()
        else as_str(optimization_cfg.get("backend", "optuna"), "optimization.backend").lower()
    )
    if backend_name != "optuna":
        raise ValueError("optimization.backend must be 'optuna'")

    study_name = as_str(optimization_cfg.get("study_name", "relic_hpo"), "optimization.study_name")
    output_dir = Path("artifacts") / "hpo" / study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space = parse_search_space(optimization_cfg.get("search_space", []))
    search_space = extend_with_nas_lite(root_config=config, base_search_space=search_space)
    _cap_trials_by_nas_lite(config=config, optimization_cfg=optimization_cfg)

    result = run_optuna_optimization(
        base_config=config,
        optimization_cfg=optimization_cfg,
        search_space=search_space,
        run_pipeline_fn=PIPELINE_EXECUTE_FN,
    )
    _write_optuna_artifacts(output_dir=output_dir, result=result)
    if not skip_final_full_pipeline:
        best_pipeline_run_id = run_best_full_pipeline(
            base_config=config,
            search_space=search_space,
            best_values=result.best_params,
            study_name=result.study_name,
            run_pipeline_fn=PIPELINE_EXECUTE_FN,
        )
        LOGGER.info("Completed best staged pipeline run: %s", best_pipeline_run_id)


def main() -> None:
    """Run optimize CLI."""
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()
    config = load_config(args.config)
    run_optimization(
        config=config,
        backend_override=args.backend,
        skip_final_full_pipeline=args.skip_final_full_pipeline,
    )


def _resolve_optimization_config(config: ConfigDict) -> ConfigDict:
    """Load and validate ``optimization`` config section."""
    optimization_raw = config.get("optimization")
    if not isinstance(optimization_raw, dict):
        raise ValueError("optimization section is required for src.optimize.run")
    optimization_cfg = cast(ConfigDict, optimization_raw)
    enabled = as_bool(optimization_cfg.get("enabled", False), "optimization.enabled")
    if not enabled:
        raise ValueError("optimization.enabled must be true to run optimization")
    return optimization_cfg


def _cap_trials_by_nas_lite(*, config: ConfigDict, optimization_cfg: ConfigDict) -> None:
    """Cap trial budget when NAS-lite max candidates is configured."""
    nas_cfg_raw = config.get("nas_lite")
    if not isinstance(nas_cfg_raw, dict):
        return
    nas_cfg = cast(ConfigDict, nas_cfg_raw)
    if not as_bool(nas_cfg.get("enabled", False), "nas_lite.enabled"):
        return

    budget_raw = optimization_cfg.get("budget")
    if not isinstance(budget_raw, dict):
        raise ValueError("optimization.budget must be a mapping")
    budget_cfg = cast(ConfigDict, budget_raw)

    n_trials = as_int(budget_cfg.get("n_trials", 20), "optimization.budget.n_trials")
    max_candidates = as_int(nas_cfg.get("max_candidates", n_trials), "nas_lite.max_candidates")
    budget_cfg["n_trials"] = min(n_trials, max_candidates)


def _write_optuna_artifacts(*, output_dir: Path, result: OptunaResult) -> None:
    """Persist Optuna outputs to artifact directory."""
    _write_trials_csv(
        output_path=output_dir / "trials.csv",
        trial_records=list(result.trial_records),
    )
    _write_yaml(
        output_path=output_dir / "best_params.yaml",
        payload={
            "study_name": result.study_name,
            "direction": result.direction,
            "objective_metric": result.objective_metric,
            "best_value": result.best_value,
            "best_params": result.best_params,
        },
    )


def _write_trials_csv(*, output_path: Path, trial_records: list[TrialRecord]) -> None:
    """Write trial records as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["trial_number", "state", "value", "run_id", "params_json"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in trial_records:
            writer.writerow(
                {
                    "trial_number": record.number,
                    "state": record.state,
                    "value": "" if record.value is None else f"{record.value:.8f}",
                    "run_id": "" if record.run_id is None else record.run_id,
                    "params_json": json.dumps(record.params, ensure_ascii=True, sort_keys=True),
                }
            )


def _write_yaml(*, output_path: Path, payload: Mapping[str, object]) -> None:
    """Write one YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)


if __name__ == "__main__":
    main()
