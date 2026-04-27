"""CLI entrypoint for automated HPO and NAS-lite workflows."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
from src.optimize.search_space import SearchParameter, extend_with_nas_lite, parse_search_space
from src.optimize.trial_runner import (
    RecheckSeedResult,
    execute_recheck_seed,
    run_best_full_pipeline,
)
from src.pipeline.engine import execute_pipeline
from src.pipeline.runtime import DistributedContext
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_float,
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


@dataclass(frozen=True)
class RecheckCandidateSummary:
    """Aggregated post-HPO recheck result for one Optuna candidate."""

    trial_number: int
    original_value: float
    mean_value: float
    std_value: float
    score: float
    params: dict[str, object]
    seed_results: tuple[RecheckSeedResult, ...]


@dataclass(frozen=True)
class RecheckResult:
    """Post-HPO top-k multi-seed recheck result."""

    top_k: int
    seeds: tuple[int, ...]
    std_penalty: float
    direction: str
    best_trial_number: int
    best_params: dict[str, object]
    summaries: tuple[RecheckCandidateSummary, ...]


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
        best_values = dict(result.best_params)
        recheck_result = _run_configured_recheck(
            base_config=config,
            optimization_cfg=optimization_cfg,
            search_space=search_space,
            optuna_result=result,
            run_id_prefix=run_id_prefix,
            execution_cfg=execution_cfg,
            distributed_channel=distributed_channel,
        )
        if recheck_result is not None:
            _write_recheck_artifacts(output_dir=output_dir, result=recheck_result)
            best_values = dict(recheck_result.best_params)
        if not skip_final_full_pipeline:
            if distributed_channel is not None:
                distributed_channel.send(
                    OptimizationCommand(
                        kind="run_best_pipeline",
                        best_values=best_values,
                    )
                )
            best_pipeline_run_id = run_best_full_pipeline(
                base_config=config,
                search_space=search_space,
                best_values=best_values,
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


def _run_configured_recheck(
    *,
    base_config: ConfigDict,
    optimization_cfg: ConfigDict,
    search_space: list[SearchParameter],
    optuna_result: OptunaResult,
    run_id_prefix: str,
    execution_cfg: ConfigDict,
    distributed_channel: OptimizationChannel | None,
) -> RecheckResult | None:
    """Run optional post-HPO top-k multi-seed recheck."""
    recheck_raw = optimization_cfg.get("recheck")
    if recheck_raw is None:
        return None
    recheck_cfg = _as_config_dict(recheck_raw, "optimization.recheck")
    if not as_bool(recheck_cfg.get("enabled", False), "optimization.recheck.enabled"):
        return None

    top_k = as_int(recheck_cfg.get("top_k", 5), "optimization.recheck.top_k")
    if top_k <= 0:
        raise ValueError("optimization.recheck.top_k must be > 0")
    seeds = _parse_seed_list(recheck_cfg.get("seeds", [13, 47, 101]))
    std_penalty = as_float(
        recheck_cfg.get("std_penalty", 0.5),
        "optimization.recheck.std_penalty",
    )
    if std_penalty < 0.0:
        raise ValueError("optimization.recheck.std_penalty must be >= 0")

    top_trials = _select_top_completed_trials(
        trial_records=list(optuna_result.trial_records),
        direction=optuna_result.direction,
        top_k=top_k,
    )
    summaries: list[RecheckCandidateSummary] = []
    for trial_record in top_trials:
        seed_results: list[RecheckSeedResult] = []
        for seed in seeds:
            if distributed_channel is not None:
                distributed_channel.send(
                    OptimizationCommand(
                        kind="run_recheck_trial",
                        trial_number=trial_record.number,
                        seed=seed,
                        sampled_values=dict(trial_record.params),
                    )
                )
            seed_result = execute_recheck_seed(
                base_config=base_config,
                search_space=search_space,
                sampled_values=trial_record.params,
                run_id_prefix=run_id_prefix,
                trial_number=trial_record.number,
                seed=seed,
                objective_metric=optuna_result.objective_metric,
                direction=optuna_result.direction,
                execution_cfg=execution_cfg,
                run_pipeline_fn=PIPELINE_EXECUTE_FN,
            )
            seed_results.append(seed_result)
            if distributed_channel is not None:
                distributed_channel.barrier()

        values = [item.objective_value for item in seed_results]
        mean_value = _mean(values)
        std_value = _population_std(values, mean_value)
        score = _recheck_score(
            mean_value=mean_value,
            std_value=std_value,
            std_penalty=std_penalty,
            direction=optuna_result.direction,
        )
        if trial_record.value is None:
            raise ValueError("Completed trial unexpectedly has no objective value")
        summaries.append(
            RecheckCandidateSummary(
                trial_number=trial_record.number,
                original_value=trial_record.value,
                mean_value=mean_value,
                std_value=std_value,
                score=score,
                params=dict(trial_record.params),
                seed_results=tuple(seed_results),
            )
        )

    ranked = sorted(
        summaries,
        key=lambda item: item.score,
        reverse=optuna_result.direction == "maximize",
    )
    best_summary = ranked[0]
    return RecheckResult(
        top_k=top_k,
        seeds=tuple(seeds),
        std_penalty=std_penalty,
        direction=optuna_result.direction,
        best_trial_number=best_summary.trial_number,
        best_params=dict(best_summary.params),
        summaries=tuple(ranked),
    )


def _select_top_completed_trials(
    *,
    trial_records: list[TrialRecord],
    direction: str,
    top_k: int,
) -> list[TrialRecord]:
    """Return top complete Optuna trials by original objective value."""
    complete_trials = [
        record
        for record in trial_records
        if record.state == "COMPLETE" and record.value is not None
    ]
    if not complete_trials:
        raise ValueError("optimization.recheck requires at least one COMPLETE Optuna trial")
    return sorted(
        complete_trials,
        key=lambda record: cast(float, record.value),
        reverse=direction == "maximize",
    )[:top_k]


def _parse_seed_list(raw_seeds: object) -> tuple[int, ...]:
    """Parse ``optimization.recheck.seeds``."""
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("optimization.recheck.seeds must be a non-empty list")
    seeds: list[int] = []
    for index, raw_seed in enumerate(raw_seeds):
        seed = as_int(raw_seed, f"optimization.recheck.seeds[{index}]")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError("optimization.recheck.seeds must not contain duplicates")
    return tuple(seeds)


def _mean(values: list[float]) -> float:
    """Return arithmetic mean for non-empty values."""
    if not values:
        raise ValueError("Cannot compute mean of empty values")
    return sum(values) / len(values)


def _population_std(values: list[float], mean_value: float) -> float:
    """Return population standard deviation for observed recheck seeds."""
    if not values:
        raise ValueError("Cannot compute standard deviation of empty values")
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _recheck_score(
    *,
    mean_value: float,
    std_value: float,
    std_penalty: float,
    direction: str,
) -> float:
    """Return stability-aware recheck score."""
    if direction == "maximize":
        return mean_value - std_penalty * std_value
    if direction == "minimize":
        return mean_value + std_penalty * std_value
    raise ValueError("optimization.direction must be 'maximize' or 'minimize'")


def _write_recheck_artifacts(*, output_dir: Path, result: RecheckResult) -> None:
    """Persist post-HPO recheck outputs."""
    _write_recheck_summary_csv(
        output_path=output_dir / "recheck_summary.csv",
        summaries=list(result.summaries),
    )
    _write_recheck_seed_csv(
        output_path=output_dir / "recheck_trials.csv",
        summaries=list(result.summaries),
    )
    _write_yaml(
        output_path=output_dir / "rechecked_best_params.yaml",
        payload={
            "selection": "top_k_multi_seed_recheck",
            "top_k": result.top_k,
            "seeds": list(result.seeds),
            "std_penalty": result.std_penalty,
            "direction": result.direction,
            "best_trial_number": result.best_trial_number,
            "best_params": result.best_params,
        },
    )


def _write_recheck_summary_csv(
    *,
    output_path: Path,
    summaries: list[RecheckCandidateSummary],
) -> None:
    """Write candidate-level recheck summaries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "trial_number",
        "original_value",
        "mean_value",
        "std_value",
        "score",
        "params_json",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, summary in enumerate(summaries, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "trial_number": summary.trial_number,
                    "original_value": f"{summary.original_value:.8f}",
                    "mean_value": f"{summary.mean_value:.8f}",
                    "std_value": f"{summary.std_value:.8f}",
                    "score": f"{summary.score:.8f}",
                    "params_json": json.dumps(summary.params, ensure_ascii=True, sort_keys=True),
                }
            )


def _write_recheck_seed_csv(
    *,
    output_path: Path,
    summaries: list[RecheckCandidateSummary],
) -> None:
    """Write seed-level recheck results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_number",
        "seed",
        "run_id",
        "value",
        "metric_column",
        "train_csv_path",
        "checkpoint_path",
        "params_json",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            for seed_result in summary.seed_results:
                writer.writerow(
                    {
                        "trial_number": seed_result.trial_number,
                        "seed": seed_result.seed,
                        "run_id": seed_result.run_id,
                        "value": f"{seed_result.objective_value:.8f}",
                        "metric_column": seed_result.metric_column,
                        "train_csv_path": str(seed_result.train_csv_path),
                        "checkpoint_path": str(seed_result.checkpoint_path),
                        "params_json": json.dumps(
                            seed_result.params,
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                    }
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
