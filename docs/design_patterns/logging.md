# Logging and Artifacts

This document details the logging strategy and artifact structure for GRAND. It serves as a reference for understanding where training metrics, evaluation results, topology outputs, and model checkpoints are stored.

## Execution Context

* Run stages: `train`, `topology_finetune`, `adapt`, `evaluate`, `topology_evaluate` (configured via `run_config.stages`).
* Canonical HPC launchers: `scripts/v3.sh`, `scripts/v4.sh`, `scripts/v5.sh`.
* Centralized loss path: both trainer and evaluator use `training_config.loss` (same `LossConfig` contract).
* DDP behavior: rank 0 writes artifacts/logs; all ranks participate in compute/synchronization via the shared `PipelineRuntime` accelerator.

## Directory Layout

Stage logs and metrics are stored under the `logs/` directory, organized by model architecture, stage, and run ID. Checkpoints are stored separately under `models/` only for stages that write model weights.

### Logging and Checkpoints

*   **Training**: `logs/{model}/train/<run_id>/`
*   **Topology Fine-tune**: `logs/{model}/topology_finetune/<run_id>/`
*   **Adaptation**: `logs/{model}/adapt/<run_id>/`
*   **Evaluation**: `logs/{model}/evaluate/<run_id>/`
*   **Topology Evaluation**: `logs/{model}/topology_evaluate/<run_id>/`
*   **Checkpoints**: `models/{model}/{stage}/<run_id>/best_model.pth` for `train`, `topology_finetune`, and `adapt` only.

**Note**: The `<run_id>` is either provided in the config or automatically generated (timestamped) by the runtime. In distributed mode, run IDs are broadcast from rank 0 to ensure consistency.

**Run ID collision policy**: Treat run IDs as immutable experiment IDs. Reusing
the same run ID can append duplicate rows to CSV summaries and overwrite JSON,
prediction, and log artifacts. When rerunning an experiment for a new fix, use a
new run ID such as `p1_fixed`; if a collision already happened, copy the
newly-written artifacts to a new run directory before restoring the original
tracked run from Git.

## Artifact Types

### 1. `log.log`
*   **Location**: Inside any stage run directory.
*   **Content**: Structured stage events emitted by the orchestrator and stage logic.
*   **Critical events** include:
    *   pipeline runtime resolution (device, distributed context, dataloader readiness),
    *   stage boundaries (`stage_start`, `stage_done`),
    *   checkpoint load/save events,
    *   epoch lifecycle (`epoch_start`, `epoch_done`, early stopping),
    *   evaluation metric and CSV write events.
*   **Heartbeat events**: Training progress logs are emitted at:
    *   step `1`,
    *   every `training_config.logging.heartbeat_every_n_steps`,
    *   final step of each epoch.
*   **Rank behavior**: only rank 0 writes file artifacts and human-readable stage logs.

### 2. `training_step.csv`
*   **Location**: `logs/{model}/train/<run_id>/`
*   **Role**: Structured time-series data for training curves.
*   **Schema (strict order)**:
    *   `Epoch`: Integer epoch number.
    *   `Epoch Time`: Duration of the epoch in seconds.
    *   `Train Loss`: Average training loss.
    *   `Val Loss`: Average validation loss.
    *   `Val {Metric}`: Monitored validation metrics from `training_config.logging.validation_metrics` in configured order.
    *   `Learning Rate`: Current learning rate.

### 3. `evaluate.csv`
*   **Location**: `logs/{model}/evaluate/<run_id>/`
*   **Role**: Final performance report for a model on a test set.
*   **Schema (strict order)**:
    *   `split,auroc,auprc,accuracy,sensitivity,specificity,precision,recall,f1,mcc`
*   **Note**: The evaluator may compute extra metrics internally, but only this fixed schema is persisted.
*   **TCCIG note**: in `evaluate.mode: graph_assembly`, hard metrics are computed
    from top-`m_hat` graph assembly. Fixed `0.5` thresholds and
    validation-calibrated pairwise thresholds are diagnostics, not the graph
    decision rule.

### 4. `topology_finetune_step.csv`
*   **Location**: `logs/{model}/topology_finetune/<run_id>/`
*   **Role**: Per-epoch metrics during topology-aware fine-tuning.
*   **Schema (strict order)**: `Epoch`, `Epoch Time`, train BCE/topology losses, validation pair/topology losses, internal-validation topology metrics, edge-coverage counters, BCE sampling counters, timing fields, peak GPU memory, topology loss scale, and `Learning Rate`.

### 5. Topology Evaluation Artifacts
*   **Location**: `logs/{model}/topology_evaluate/<run_id>/`
*   **Files**:
    *   `topology_metrics.json`: Graph similarity, relative density, MMD distances, and other topology metrics.
    *   `topology_metrics.csv`: Per-node-size topology summary plus aggregate rows.
    *   `graph_eval_results.pkl`: Serialized detailed topology-evaluation output.
    *   `all_test_ppi_pred.txt`: Pairwise predictions in PRING format for graph reconstruction.
*   **TCCIG note**: `topology_metrics.json` records `decision_rule: top_m_hat`
    for graph-forward TCCIG runs and stores fixed-threshold settings only under
    `fixed_threshold_diagnostic`. It also records `debug_assemblies` for
    non-official budget diagnostics such as validation-density and
    oracle-test-density top-K assemblies.

### 6. `best_model.pth`
*   **Location**: `models/{model}/{stage}/<run_id>/best_model.pth` (for `train`, `topology_finetune`, and `adapt` stages).
*   **Role**: The saved state dictionary of the model achieving the best performance on the monitored metric.
*   **Checkpoint I/O**: All checkpoint operations go through `PipelineRuntime.save_checkpoint()` and `load_checkpoint()`, which handle accelerator unwrapping, `wait_for_everyone()` barriers, and main-process-only writes.
*   **Evaluation stages**: `evaluate` and `topology_evaluate` load checkpoints but do not create checkpoint directories because they do not save model weights.
*   **Resume semantics**: Checkpoints contain model weights only. Loading a checkpoint is a warm start; optimizer, scheduler, epoch, and early-stopping state are not restored.

## Checkpoint Policy

The orchestrator controls when checkpoints are saved based on the `run_config.save_best_only` setting:
*   **`true`**: Only the single best checkpoint is kept (`best_model.pth`).
*   **`false`**: A checkpoint is saved at the end of every epoch (e.g., `checkpoint_epoch_001.pth`), in addition to `best_model.pth`.

## Logging Configuration

Training log behavior is configured in `training_config.logging`:

```yaml
training_config:
  logging:
    validation_metrics: ["auprc", "auroc"]
    heartbeat_every_n_steps: 20
```

* `validation_metrics` controls which `Val {Metric}` columns are written to `training_step.csv`.
* `heartbeat_every_n_steps` controls periodic training progress logs in `log.log`.
