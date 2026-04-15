"""Public pipeline API for imports via ``import src.run``."""

from __future__ import annotations

import logging
import os

from src.run.bootstrap import _rank_from_env, configure_root_logging, parse_args, set_global_seed
from src.run.pipeline_orchestrator import _ddp_find_unused_parameters
from src.run.pipeline_orchestrator import execute_pipeline as _execute_pipeline_impl
from src.run.stage_adapt import ADAPT_CSV_COLUMNS, run_shot_adaptation_stage
from src.run.stage_evaluate import EVAL_CSV_COLUMNS, _metrics_from_config, run_evaluation_stage
from src.run.stage_topology_evaluate import TOPOLOGY_CSV_COLUMNS, run_topology_evaluation_stage
from src.run.stage_topology_finetune import (
    TOPOLOGY_FINETUNE_CSV_COLUMNS,
    run_topology_finetuning_stage,
)
from src.run.stage_train import (
    _training_validation_metrics,
    build_model,
    build_strategy,
    build_trainer,
    run_training_stage,
)
from src.utils.accelerator import build_accelerator
from src.utils.config import ConfigDict, load_config
from src.utils.data_io import build_dataloaders
from src.utils.device import resolve_device

ROOT_LOGGER = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _configure_root_logging() -> None:
    """Configure process-level logging; suppress non-main rank noise."""
    configure_root_logging(logging_module=logging, rank=_rank_from_env())


def execute_pipeline(config: ConfigDict) -> None:
    """Execute pipeline according to configured stages."""
    _execute_pipeline_impl(
        config=config,
        build_dataloaders_fn=build_dataloaders,
        build_model_fn=build_model,
        run_training_stage_fn=run_training_stage,
        run_topology_finetuning_stage_fn=run_topology_finetuning_stage,
        run_adaptation_stage_fn=run_shot_adaptation_stage,
        run_evaluation_stage_fn=run_evaluation_stage,
        run_topology_evaluation_stage_fn=run_topology_evaluation_stage,
        build_accelerator_fn=build_accelerator,
        resolve_device_fn=resolve_device,
    )


def main() -> None:
    """Run CLI entrypoint."""
    _configure_root_logging()
    args = parse_args()
    config = load_config(args.config)
    ROOT_LOGGER.info("Loaded config: %s", args.config)
    execute_pipeline(config=config)


__all__ = [
    "ADAPT_CSV_COLUMNS",
    "EVAL_CSV_COLUMNS",
    "TOPOLOGY_CSV_COLUMNS",
    "TOPOLOGY_FINETUNE_CSV_COLUMNS",
    "_configure_root_logging",
    "_ddp_find_unused_parameters",
    "_metrics_from_config",
    "_training_validation_metrics",
    "build_accelerator",
    "build_dataloaders",
    "build_model",
    "build_strategy",
    "build_trainer",
    "execute_pipeline",
    "logging",
    "main",
    "parse_args",
    "resolve_device",
    "run_shot_adaptation_stage",
    "run_evaluation_stage",
    "run_topology_evaluation_stage",
    "run_topology_finetuning_stage",
    "run_training_stage",
    "set_global_seed",
]
