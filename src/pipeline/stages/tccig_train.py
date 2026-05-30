"""Public TCCIG scratch-training pipeline stage."""

from __future__ import annotations

from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from src.pipeline.runtime import AcceleratorLike, PipelineRuntime
from src.train.tccig.config import parse_tccig_train_config
from src.train.tccig.runner import TCCIGTrainRunner
from src.train.topology import shared as topology_train
from src.utils.config import extract_model_kwargs
from src.utils.logging import log_stage_event


def _graph_model_required(model: nn.Module, accelerator: AcceleratorLike) -> None:
    """Require the graph-forward contract for the TCCIG-only stage."""
    if topology_train._model_supports_graph_forward(model, accelerator):
        return
    raise ValueError("tccig_train requires model_config.model='tccig' with forward_graph support")


def run_tccig_train_stage(
    runtime: PipelineRuntime,
    model: nn.Module,
    dataloaders: dict[str, DataLoader[dict[str, object]]],
) -> Path:
    """Train a TCCIG feature-only graph generator from scratch."""
    config = runtime.config.raw
    model_name, _ = extract_model_kwargs(config)
    if model_name != "tccig":
        raise ValueError("tccig_train requires model_config.model='tccig'")
    _graph_model_required(model, runtime.accelerator)
    parse_tccig_train_config(config)

    run_id = runtime.stage_run_id("tccig_train")
    paths = runtime.stage_paths("tccig_train")
    logger = runtime.stage_logger("tccig_train", paths.log_dir / "log.log")
    if runtime.is_main_process:
        log_stage_event(logger, "stage_start", run_id=run_id, init_mode="scratch")

    best_checkpoint = TCCIGTrainRunner(
        runtime=runtime,
        model=model,
        dataloaders=dataloaders,
        log_dir=paths.log_dir,
        model_dir=paths.model_dir,
        logger=logger,
    ).run()
    if runtime.is_main_process:
        log_stage_event(logger, "stage_done", run_id=run_id)
    return best_checkpoint
