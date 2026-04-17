# Logging and Artifacts

This document details the logging strategy and artifact structure for GRAND. It serves as a reference for understanding where training metrics, evaluation results, topology outputs, and model checkpoints are stored.

## Execution Context

* Run stages: `train`, `topology_finetune`, `adapt`, `evaluate`, `topology_evaluate` (configured via `run_config.stages`).
* Canonical HPC launchers: `scripts/v3.sh`, `scripts/v4.sh`, `scripts/v5.sh`.
* Centralized loss path: both trainer and evaluator use `training_config.loss` (same `LossConfig` contract).
* DDP behavior: rank 0 writes artifacts/logs; all ranks participate in compute/synchronization via the shared `PipelineRuntime` accelerator.

## Directory Layout

All artifacts are stored under the `logs/` directory, organized by model architecture, stage, and run ID.

### Logging and Checkpoints

*   **Training**: `logs/{model}/train/<run_id>/`
*   **Topology Fine-tune**: `logs/{model}/topology_finetune/<run_id>/`
*   **Adaptation**: `logs/{model}/adapt/<run_id>/`
*   **Evaluation**: `logs/{model}/evaluate/<run_id>/`
*   **Topology Evaluation**: `logs/{model}/topology_evaluate/<run_id>/`

**Note**: The `<run_id>` is either provided in the config or automatically generated (timestamped) by the runtime. In distributed mode, run IDs are broadcast from rank 0 to ensure consistency.

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

### 4. `topology_finetune.csv`
*   **Location**: `logs/{model}/topology_finetune/<run_id>/`
*   **Role**: Per-epoch metrics during topology-aware fine-tuning.
*   **Schema (strict order)**:
    *   `Epoch,Epoch Time,Train Loss,Val Loss,Val auprc,Val auroc,Learning Rate`

### 5. Topology Evaluation Artifacts
*   **Location**: `logs/{model}/topology_evaluate/<run_id>/`
*   **Files**:
    *   `topology_metrics.json`: Graph similarity, relative density, MMD distances, and other topology metrics.
    *   `all_test_ppi_pred.txt`: Pairwise predictions in PRING format for graph reconstruction.
    *   `topology_evaluate.csv`: Summary metrics row.

### 6. `best_model.pth`
*   **Location**: `logs/{model}/{stage}/<run_id>/best_model.pth` (for `train` and `topology_finetune` stages).
*   **Role**: The saved state dictionary of the model achieving the best performance on the monitored metric.
*   **Checkpoint I/O**: All checkpoint operations go through `PipelineRuntime.save_checkpoint()` and `load_checkpoint()`, which handle accelerator unwrapping, `wait_for_everyone()` barriers, and main-process-only writes.

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
