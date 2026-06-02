"""Lifecycle runner for the TCCIG training stage."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.pipeline.runtime import PipelineRuntime
from src.train.tccig.config import TCCIGTrainConfig, parse_tccig_train_config, tccig_train_config
from src.train.tccig.data import TCCIGDataContext, prepare_tccig_data_context
from src.train.tccig.teacher import OnlineTCCIGTeacher
from src.train.tccig.trainer import TCCIGStudentTrainer
from src.train.tccig.validation import TCCIGValidationRunner
from src.train.topology import shared as topology_train
from src.utils.logging import append_csv_row, format_result_payload, log_stage_event

TCCIG_RECONSTRUCTION_CSV_COLUMNS = [
    "Val Candidate AUPRC",
    "Val Retrieval Candidate AUPRC",
    "Val Retrieval Recall@20%",
    "Val Reconstruction Candidate Count",
    "Val Reconstruction Positive Count",
    "Val Composite Score",
]
TCCIG_TRAIN_CSV_COLUMNS = [
    *topology_train.TOPOLOGY_FINETUNE_CSV_COLUMNS,
    *TCCIG_RECONSTRUCTION_CSV_COLUMNS,
]


class TCCIGTrainRunner:
    """Own the TCCIG training lifecycle for one pipeline stage run."""

    def __init__(
        self,
        *,
        runtime: PipelineRuntime,
        model: nn.Module,
        dataloaders: dict[str, DataLoader[dict[str, object]]],
        log_dir: Path,
        model_dir: Path,
        logger: logging.Logger,
    ) -> None:
        self.runtime = runtime
        self.model = model
        self.dataloaders = dataloaders
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.logger = logger

    def run(self) -> Path:
        """Train TCCIG from scratch and return the best checkpoint path."""
        config = self.runtime.config.raw
        device = self.runtime.device
        train_cfg = parse_tccig_train_config(config)
        raw_train_cfg = tccig_train_config(config)

        data_context = prepare_tccig_data_context(
            config=config,
            train_cfg=train_cfg,
            model=self.model,
            device=device,
            log_dir=self.log_dir,
            model_dir=self.model_dir,
            distributed_context=self.runtime.distributed,
            accelerator=self.runtime.accelerator,
        )
        teacher = OnlineTCCIGTeacher.build(
            train_cfg=raw_train_cfg,
            input_dim=data_context.input_dim,
            device=device,
        )
        stage_model, prepared_optimizer = cast(
            tuple[nn.Module, Optimizer],
            self.runtime.accelerator.prepare(self.model, data_context.optimizer),
        )
        data_context = replace(data_context, optimizer=prepared_optimizer)
        if teacher is not None:
            teacher = teacher.prepare(self.runtime.accelerator)
        self._log_config(train_cfg=train_cfg, data_context=data_context)

        trainer = TCCIGStudentTrainer(
            train_cfg=train_cfg,
            raw_config=config,
            model=stage_model,
            graph=data_context.train_graph,
            optimizer=data_context.optimizer,
            device=device,
            accelerator=self.runtime.accelerator,
            embedding_repository=data_context.embedding_repository,
            negative_lookup=data_context.train_negative_lookup,
            distributed_context=self.runtime.distributed,
            model_dir=self.model_dir,
            teacher=teacher,
            graph_prior_artifacts=data_context.graph_prior_artifacts,
            logger=self.logger,
        )
        validation_runner = TCCIGValidationRunner(
            raw_config=config,
            train_cfg=train_cfg,
            data_context=data_context,
            dataloaders=self.dataloaders,
            device=device,
        )
        return self._run_epochs(
            stage_model=stage_model,
            train_cfg=train_cfg,
            data_context=data_context,
            trainer=trainer,
            validation_runner=validation_runner,
        )

    def _log_config(
        self,
        *,
        train_cfg: TCCIGTrainConfig,
        data_context: TCCIGDataContext,
    ) -> None:
        if not self.runtime.is_main_process:
            return
        log_stage_event(
            self.logger,
            "tccig_train_config",
            epochs=train_cfg.epochs,
            monitor=train_cfg.monitor_metric,
            internal_validation_subgraphs=data_context.internal_validation_plan.total_subgraphs,
            internal_validation_node_sizes=sorted(data_context.internal_validation_node_sets),
            internal_validation_pairs=data_context.internal_validation_plan.total_pairs,
            pair_batch_size=train_cfg.pair_batch_size,
            internal_validation_inference_batch_size=(
                train_cfg.internal_validation_inference_batch_size
            ),
            internal_validation_compute_spectral_stats=(
                train_cfg.internal_validation_compute_spectral_stats
            ),
        )
        log_stage_event(
            self.logger,
            "density_bias_initialized",
            positive_edge_probability=data_context.density_prior_probability,
            source=data_context.density_prior_source,
            bias=data_context.density_prior_bias,
        )

    def _run_epochs(
        self,
        *,
        stage_model: nn.Module,
        train_cfg: TCCIGTrainConfig,
        data_context: TCCIGDataContext,
        trainer: TCCIGStudentTrainer,
        validation_runner: TCCIGValidationRunner,
    ) -> Path:
        best_metrics: dict[str, float | str] = {}
        previous_topology_loss_scale: float | None = None
        best_auprc_value = float("-inf")
        best_auprc_epoch = 0

        for epoch in range(train_cfg.epochs):
            epoch_start = time.perf_counter()
            if self.runtime.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.runtime.device)
            train_stats = trainer.fit_epoch(
                epoch_index=epoch,
                epoch_seed=topology_train._resolve_epoch_seed(
                    run_seed=train_cfg.run_seed,
                    epoch_index=epoch,
                    distributed_context=self.runtime.distributed,
                ),
            )
            train_stats = topology_train._reduce_train_stats(
                accelerator=self.runtime.accelerator,
                device=self.runtime.device,
                train_stats=train_stats,
                global_subgraph_count=int(train_stats["planned_subgraphs"]),
            )
            validation_result = validation_runner.evaluate(
                model=stage_model,
                epoch_index=epoch,
                topology_loss_scale_value=float(train_stats["topology_loss_scale"]),
                previous_topology_loss_scale=previous_topology_loss_scale,
            )
            previous_topology_loss_scale = float(train_stats["topology_loss_scale"])
            peak_gpu_mem_mb = (
                float(torch.cuda.max_memory_allocated(self.runtime.device) / (1024.0 * 1024.0))
                if self.runtime.device.type == "cuda"
                else 0.0
            )
            monitor_value = topology_train._resolve_monitor_value(
                monitor_metric=train_cfg.monitor_metric,
                val_pair_stats=validation_result.val_pair_stats,
                internal_val_topology_stats=validation_result.internal_val_topology_stats,
                val_total_loss=validation_result.val_total_loss,
                val_topology_loss=validation_result.val_topology_loss,
            )
            current_auprc = float(validation_result.val_pair_stats.get("val_auprc", 0.0))
            best_auprc_improved = current_auprc > best_auprc_value
            if best_auprc_improved:
                best_auprc_value = current_auprc
                best_auprc_epoch = epoch + 1
            should_stop, saved_metrics = self._persist_epoch_result(
                stage_model=stage_model,
                train_cfg=train_cfg,
                data_context=data_context,
                epoch=epoch,
                epoch_seconds=time.perf_counter() - epoch_start,
                train_stats=train_stats,
                validation_result=validation_result,
                monitor_value=monitor_value,
                peak_gpu_mem_mb=peak_gpu_mem_mb,
                best_auprc_epoch=best_auprc_epoch,
                best_auprc_value=best_auprc_value,
                best_auprc_improved=best_auprc_improved,
            )
            if saved_metrics is not None:
                best_metrics = saved_metrics
            should_stop = self._sync_early_stop(should_stop)
            if should_stop:
                if self.runtime.is_main_process:
                    log_stage_event(self.logger, "early_stop", epoch=epoch + 1)
                break

        self._save_fallback_if_needed(
            stage_model=stage_model,
            data_context=data_context,
            train_cfg=train_cfg,
            best_metrics=best_metrics,
        )
        self.runtime.barrier()
        return data_context.best_checkpoint_path

    def _persist_epoch_result(
        self,
        *,
        stage_model: nn.Module,
        train_cfg: TCCIGTrainConfig,
        data_context: TCCIGDataContext,
        epoch: int,
        epoch_seconds: float,
        train_stats: dict[str, float],
        validation_result: topology_train.ValidationEpochResult,
        monitor_value: float,
        peak_gpu_mem_mb: float,
        best_auprc_epoch: int,
        best_auprc_value: float,
        best_auprc_improved: bool,
    ) -> tuple[bool, dict[str, float | str] | None]:
        should_stop = False
        save_best_checkpoint = False
        if self.runtime.is_main_process:
            append_csv_row(
                csv_path=data_context.csv_path,
                row=_build_tccig_epoch_csv_row(
                    epoch=epoch + 1,
                    epoch_seconds=epoch_seconds,
                    train_stats=train_stats,
                    validation_result=validation_result,
                    optimizer=data_context.optimizer,
                    peak_gpu_mem_mb=peak_gpu_mem_mb,
                ),
                fieldnames=TCCIG_TRAIN_CSV_COLUMNS,
            )
            improved, should_stop = data_context.early_stopping.update(monitor_value)
            save_best_checkpoint = improved
        save_best_checkpoint = topology_train._sync_flag(self.runtime, save_best_checkpoint)

        saved_metrics: dict[str, float | str] | None = None
        if save_best_checkpoint:
            self.runtime.save_checkpoint(stage_model, data_context.best_checkpoint_path)
            if self.runtime.is_main_process:
                saved_metrics = topology_train._build_best_metrics_payload(
                    epoch=epoch + 1,
                    monitor_metric=train_cfg.monitor_metric,
                    monitor_value=monitor_value,
                    train_stats=train_stats,
                    validation_result=validation_result,
                )
                saved_metrics.update(
                    {
                        "best_topology_epoch": float(epoch + 1),
                        "best_topology_auprc": float(
                            validation_result.val_pair_stats.get("val_auprc", 0.0)
                        ),
                        "best_auprc_epoch": float(best_auprc_epoch),
                        "best_auprc": float(best_auprc_value),
                    }
                )
                saved_metrics.update(
                    _tccig_reconstruction_metrics_payload(validation_result.val_pair_stats)
                )
                data_context.metrics_path.write_text(
                    json.dumps(format_result_payload(saved_metrics), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                log_stage_event(
                    self.logger,
                    "best_saved",
                    epoch=epoch + 1,
                    monitor=train_cfg.monitor_metric,
                    value=monitor_value,
                )
        elif self.runtime.is_main_process and data_context.metrics_path.exists():
            self._update_best_auprc_metrics(
                metrics_path=data_context.metrics_path,
                best_auprc_epoch=best_auprc_epoch,
                best_auprc_value=best_auprc_value,
            )
        if self.runtime.is_main_process and best_auprc_improved:
            log_stage_event(
                self.logger,
                "best_auprc_observed",
                epoch=best_auprc_epoch,
                value=best_auprc_value,
            )
        self._log_epoch_done(
            epoch=epoch,
            train_stats=train_stats,
            validation_result=validation_result,
            peak_gpu_mem_mb=peak_gpu_mem_mb,
        )
        return should_stop, saved_metrics

    @staticmethod
    def _update_best_auprc_metrics(
        *,
        metrics_path: Path,
        best_auprc_epoch: int,
        best_auprc_value: float,
    ) -> None:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["best_auprc_epoch"] = float(best_auprc_epoch)
        payload["best_auprc"] = float(best_auprc_value)
        metrics_path.write_text(
            json.dumps(format_result_payload(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _log_epoch_done(
        self,
        *,
        epoch: int,
        train_stats: dict[str, float],
        validation_result: topology_train.ValidationEpochResult,
        peak_gpu_mem_mb: float,
    ) -> None:
        if not self.runtime.is_main_process:
            return
        log_stage_event(
            self.logger,
            "epoch_done",
            epoch=epoch + 1,
            train_loss=train_stats["total"],
            val_auprc=float(validation_result.val_pair_stats.get("val_auprc", 0.0)),
            val_candidate_auprc=float(
                validation_result.val_pair_stats.get("val_candidate_auprc", 0.0)
            ),
            val_retrieval_recall_at_20=float(
                validation_result.val_pair_stats.get("val_retrieval_recall_at_20", 0.0)
            ),
            val_composite_score=float(
                validation_result.val_pair_stats.get("val_composite_score", 0.0)
            ),
            internal_val_graph_sim=validation_result.internal_val_topology_stats["graph_sim"],
            planned_subgraphs=int(train_stats["planned_subgraphs"]),
            covered_positive_edges=int(train_stats["covered_positive_edges"]),
            total_positive_edges=int(train_stats["total_positive_edges"]),
            positive_edge_coverage_ratio=train_stats["positive_edge_coverage_ratio"],
            mean_positive_edge_reuse=train_stats["mean_positive_edge_reuse"],
            all_subgraph_pairs=int(train_stats["all_subgraph_pairs"]),
            supervised_pairs=int(train_stats["supervised_pairs"]),
            bce_positive_pairs=int(train_stats["bce_positive_pairs"]),
            bce_target_negative_pairs=int(train_stats["bce_target_negative_pairs"]),
            bce_negative_pairs=int(train_stats["bce_negative_pairs"]),
            bce_negative_ratio=train_stats["bce_negative_ratio"],
            bce_supervised_fraction=train_stats["bce_supervised_fraction"],
            edge_cover_sampling_s=train_stats["edge_cover_sampling_s"],
            train_forward_backward_s=train_stats["train_forward_backward_s"],
            val_pair_pass_s=validation_result.val_pair_pass_seconds,
            val_threshold_s=validation_result.threshold_resolution_seconds,
            internal_val_topology_s=validation_result.internal_validation_seconds,
            peak_gpu_mem_mb=peak_gpu_mem_mb,
            topo_loss_scale=train_stats["topology_loss_scale"],
        )

    def _sync_early_stop(self, should_stop: bool) -> bool:
        if not self.runtime.is_distributed:
            return should_stop
        stop_flag = torch.tensor(
            [1 if should_stop else 0],
            device=self.runtime.device,
            dtype=torch.int64,
        )
        dist.broadcast(stop_flag, src=0)
        return bool(int(stop_flag.item()))

    def _save_fallback_if_needed(
        self,
        *,
        stage_model: nn.Module,
        data_context: TCCIGDataContext,
        train_cfg: TCCIGTrainConfig,
        best_metrics: dict[str, float | str],
    ) -> None:
        fallback_save = (
            self.runtime.is_main_process and not data_context.best_checkpoint_path.exists()
        )
        if topology_train._sync_flag(self.runtime, fallback_save):
            self.runtime.save_checkpoint(stage_model, data_context.best_checkpoint_path)
        if self.runtime.is_main_process and fallback_save and not best_metrics:
            data_context.metrics_path.write_text(
                json.dumps(
                    format_result_payload(
                        {"monitor_metric": train_cfg.monitor_metric, "monitor_value": 0.0}
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )


def _build_tccig_epoch_csv_row(
    *,
    epoch: int,
    epoch_seconds: float,
    train_stats: dict[str, float],
    validation_result: topology_train.ValidationEpochResult,
    optimizer: Optimizer,
    peak_gpu_mem_mb: float,
) -> dict[str, float | int | str]:
    """Build the persisted CSV row for one graph-prior retrieval TCCIG epoch."""
    row = topology_train._build_epoch_csv_row(
        epoch=epoch,
        epoch_seconds=epoch_seconds,
        train_stats=train_stats,
        validation_result=validation_result,
        optimizer=optimizer,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
    )
    row.update(_tccig_reconstruction_csv_fields(validation_result.val_pair_stats))
    return row


def _tccig_reconstruction_csv_fields(
    val_pair_stats: dict[str, float],
) -> dict[str, float]:
    """Return reconstruction retrieval metrics for TCCIG training CSV artifacts."""
    return {
        "Val Candidate AUPRC": float(val_pair_stats.get("val_candidate_auprc", 0.0)),
        "Val Retrieval Candidate AUPRC": float(
            val_pair_stats.get("val_retrieval_candidate_auprc", 0.0)
        ),
        "Val Retrieval Recall@20%": float(
            val_pair_stats.get("val_retrieval_recall_at_20", 0.0)
        ),
        "Val Reconstruction Candidate Count": float(
            val_pair_stats.get("val_reconstruction_candidate_count", 0.0)
        ),
        "Val Reconstruction Positive Count": float(
            val_pair_stats.get("val_reconstruction_positive_count", 0.0)
        ),
        "Val Composite Score": float(val_pair_stats.get("val_composite_score", 0.0)),
    }


def _tccig_reconstruction_metrics_payload(
    val_pair_stats: dict[str, float],
) -> dict[str, float]:
    """Return reconstruction retrieval metrics for TCCIG best-checkpoint JSON."""
    return {
        "val_candidate_auprc": float(val_pair_stats.get("val_candidate_auprc", 0.0)),
        "val_retrieval_candidate_auprc": float(
            val_pair_stats.get("val_retrieval_candidate_auprc", 0.0)
        ),
        "val_retrieval_recall_at_20": float(
            val_pair_stats.get("val_retrieval_recall_at_20", 0.0)
        ),
        "val_reconstruction_candidate_count": float(
            val_pair_stats.get("val_reconstruction_candidate_count", 0.0)
        ),
        "val_reconstruction_positive_count": float(
            val_pair_stats.get("val_reconstruction_positive_count", 0.0)
        ),
        "val_composite_score": float(val_pair_stats.get("val_composite_score", 0.0)),
    }
