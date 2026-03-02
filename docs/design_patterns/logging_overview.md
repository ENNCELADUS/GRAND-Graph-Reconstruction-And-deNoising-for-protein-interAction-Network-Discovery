# Logging and Artifacts

This document details the logging strategy and artifact structure for DIPPI. It serves as a reference for understanding where training metrics, evaluation results, and model checkpoints are stored.

## Execution Context

* Run stages: `train`, `evaluate` (configured via `run_config.stages`).
* Canonical HPC launcher: `scripts/v3.sh`.
* Centralized loss path: both trainer and evaluator use `training_config.loss` (same `LossConfig` contract).
* DDP behavior: rank 0 writes artifacts/logs; all ranks still participate in compute/synchronization.

## Directory Layout

All artifacts are stored under the `logs/` and `models/` directories, organized by model architecture and run ID.

### Logging (`logs/`)

*   **Training Logs**: `logs/{model}/train/<run_id>/`
*   **Evaluation Logs**: `logs/{model}/evaluate/<run_id>/`

**Note**: The `<run_id>` is either provided in the config or automatically generated (timestamped) by the orchestrator.

### Checkpoints (`models/`)

*   **Training Models**: `models/{model}/train/<run_id>/`

## Artifact Types

### 1. `log.log`
*   **Location**: Inside any run directory.
*   **Content**: Structured stage events emitted by the orchestrator and trainer.
*   **Critical events** include:
    *   pipeline bootstrap/device resolution/dataloader readiness/model initialization,
    *   stage boundaries (`stage_start`, `stage_complete`),
    *   checkpoint load/save events,
    *   epoch lifecycle (`epoch_start`, `epoch_complete`, early stopping),
    *   evaluation metric and CSV write events.
*   **Heartbeat events**: Training progress logs are emitted at:
    *   step `1`,
    *   every `training_config.logging.heartbeat_every_n_steps`,
    *   final step of each epoch.
*   **Rank behavior**: only rank 0 writes file artifacts and human-readable stage logs.

### 2. `training_step.csv`
*   **Location**: Train directory.
*   **Role**: structured time-series data for training curves.
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

### 4. `best_model.pth`
*   **Location**: `models/{model}/{stage}/{run_id}/`
*   **Role**: The saved state dictionary of the model achieving the best performance on the monitored metric.

## Checkpoint Policy

The orchestrator controls when checkpoints are saved based on the `run_config.save_best_only` setting:
*   **`true`**: Only the single best checkpoint is kept (`best_model.pth`).
*   **`false`**: A checkpoint is saved at the end of every epoch (e.g., `checkpoint_epoch_01.pth`), in addition to `best_model.pth`.

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
