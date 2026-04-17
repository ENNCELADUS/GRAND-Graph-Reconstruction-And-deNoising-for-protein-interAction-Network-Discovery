"""CLI entrypoint for automated HPO and NAS-lite workflows."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import yaml  # type: ignore[import-untyped]

from src.optimize.backends.optuna_backend import OptunaResult, TrialRecord, run_optuna_optimization
from src.optimize.distributed import (
    OptimizationChannel,
    OptimizationCommand,
    build_optimization_channel,
    run_distributed_worker_loop,
)
from src.optimize.search_space import extend_with_nas_lite, parse_search_space
from src.optimize.trial_runner import run_best_full_pipeline
from src.pipeline.engine import execute_pipeline
from src.pipeline.runtime import DistributedContext
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_int,
    as_str,
    as_str_list,
    get_section,
    load_config,
)
from src.utils.logging import generate_run_id

LOGGER = logging.getLogger(__name__)
PipelineExecuteFn = Callable[[ConfigDict], None]
PIPELINE_EXECUTE_FN: PipelineExecuteFn = cast(PipelineExecuteFn, execute_pipeline)


def parse_args() -> argparse.Namespace:
    """Parse optimize CLI arguments."""
    parser = argparse.ArgumentParser(description="Run GRAND optimization workflow")
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
    if "train" not in _configured_run_stages(config):
        PIPELINE_EXECUTE_FN(config)
        return
    backend_name = (
        backend_override.lower().strip()
        if isinstance(backend_override, str) and backend_override.strip()
        else as_str(optimization_cfg.get("backend", "optuna"), "optimization.backend").lower()
    )
    if backend_name != "optuna":
        raise ValueError("optimization.backend must be 'optuna'")

    study_name = as_str(optimization_cfg.get("study_name", "grand_hpo"), "optimization.study_name")
    run_id_prefix = _resolve_optimization_run_id_prefix(optimization_cfg)
    output_dir = Path("artifacts") / "hpo" / study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    search_space = parse_search_space(optimization_cfg.get("search_space", []))
    search_space = extend_with_nas_lite(root_config=config, base_search_space=search_space)
    _cap_trials_by_nas_lite(config=config, optimization_cfg=optimization_cfg)
    execution_cfg = _as_config_dict(optimization_cfg.get("execution", {}), "optimization.execution")

    distributed_context = _initialize_optimization_distributed(execution_cfg=execution_cfg)
    distributed_channel = build_optimization_channel(distributed_context)
    try:
        if distributed_context.is_distributed and not distributed_context.is_main_process:
            run_distributed_worker_loop(
                base_config=config,
                search_space=search_space,
                study_name=study_name,
                run_id_prefix=run_id_prefix,
                objective_metric=as_str(
                    optimization_cfg.get("objective_metric", "val_auprc"),
                    "optimization.objective_metric",
                ),
                direction=as_str(
                    optimization_cfg.get("direction", "maximize"), "optimization.direction"
                ),
                execution_cfg=execution_cfg,
                run_pipeline_fn=PIPELINE_EXECUTE_FN,
                channel=cast("OptimizationChannel", distributed_channel),
            )
            return

        result = run_optuna_optimization(
            base_config=config,
            optimization_cfg=optimization_cfg,
            search_space=search_space,
            run_id_prefix=run_id_prefix,
            run_pipeline_fn=PIPELINE_EXECUTE_FN,
            distributed_channel=distributed_channel,
        )
        _write_optuna_artifacts(output_dir=output_dir, result=result)
        if not skip_final_full_pipeline:
            if distributed_channel is not None:
                distributed_channel.send(
                    OptimizationCommand(
                        kind="run_best_pipeline",
                        best_values=dict(result.best_params),
                    )
                )
            best_pipeline_run_id = run_best_full_pipeline(
                base_config=config,
                search_space=search_space,
                best_values=result.best_params,
                run_id_prefix=run_id_prefix,
                run_pipeline_fn=PIPELINE_EXECUTE_FN,
                ddp_per_trial=distributed_context.is_distributed,
            )
            if distributed_channel is not None:
                distributed_channel.barrier()
            LOGGER.info("Completed best staged pipeline run: %s", best_pipeline_run_id)
        if distributed_channel is not None:
            distributed_channel.send(OptimizationCommand(kind="stop"))
    finally:
        _cleanup_optimization_distributed(distributed_context)


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


def should_run_optimization(config: ConfigDict) -> bool:
    """Return whether launcher scripts should dispatch into optimization mode."""
    optimization_raw = config.get("optimization")
    if not isinstance(optimization_raw, dict):
        return False
    optimization_cfg = cast(ConfigDict, optimization_raw)
    if not as_bool(optimization_cfg.get("enabled", False), "optimization.enabled"):
        return False
    return "train" in _configured_run_stages(config)


def _configured_run_stages(config: ConfigDict) -> tuple[str, ...]:
    """Return normalized configured run stages."""
    run_cfg = get_section(config, "run_config")
    return tuple(
        stage.lower()
        for stage in as_str_list(run_cfg.get("stages", ["train", "evaluate"]), "run_config.stages")
    )


def _resolve_optimization_run_id_prefix(optimization_cfg: ConfigDict) -> str:
    """Resolve timestamp-based run-id prefix for one optimization session."""
    configured_prefix = optimization_cfg.get("run_id_prefix")
    if isinstance(configured_prefix, str) and configured_prefix.strip():
        return configured_prefix.strip()
    return generate_run_id(None)


def _initialize_optimization_distributed(*, execution_cfg: ConfigDict) -> DistributedContext:
    """Initialize one shared process group for distributed optimization sessions."""
    world_size = as_int(os.environ.get("WORLD_SIZE", "1"), "WORLD_SIZE")
    ddp_per_trial = as_bool(
        execution_cfg.get("ddp_per_trial", False),
        "optimization.execution.ddp_per_trial",
    )
    if world_size <= 1:
        return DistributedContext(ddp_enabled=False, is_distributed=False)
    if not ddp_per_trial:
        raise ValueError(
            "Distributed optimization requires optimization.execution.ddp_per_trial=true"
        )
    return _initialize_optimization_process_group()


def _initialize_optimization_process_group() -> DistributedContext:
    """Initialize one optimization-owned process group from torchrun env."""
    if dist.is_initialized():
        return DistributedContext(
            ddp_enabled=True,
            is_distributed=dist.get_world_size() > 1,
            rank=int(dist.get_rank()),
            local_rank=as_int(os.environ.get("LOCAL_RANK", "0"), "LOCAL_RANK"),
            world_size=int(dist.get_world_size()),
            owns_process_group=False,
        )

    world_size = as_int(os.environ.get("WORLD_SIZE", "1"), "WORLD_SIZE")
    rank = as_int(os.environ.get("RANK", "0"), "RANK")
    local_rank = as_int(os.environ.get("LOCAL_RANK", "0"), "LOCAL_RANK")
    if world_size <= 1:
        return DistributedContext(ddp_enabled=True, is_distributed=False)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    LOGGER.info(
        "Initialized optimization process group (backend=%s rank=%d local_rank=%d world_size=%d).",
        backend,
        rank,
        local_rank,
        world_size,
    )
    return DistributedContext(
        ddp_enabled=True,
        is_distributed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        owns_process_group=True,
    )


def _cleanup_optimization_distributed(context: DistributedContext) -> None:
    """Destroy optimization-owned process groups after distributed runs."""
    if context.is_distributed and context.owns_process_group and dist.is_initialized():
        dist.destroy_process_group()


def _as_config_dict(value: object, field_name: str) -> ConfigDict:
    """Return config mapping or raise."""
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return cast(ConfigDict, value)


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
