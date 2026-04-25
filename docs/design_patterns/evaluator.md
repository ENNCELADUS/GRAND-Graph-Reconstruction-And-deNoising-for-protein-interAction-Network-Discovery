# Evaluator and Metrics

The **Evaluator** module (`src/evaluate/base.py`) is responsible for assessing model performance. It provides a consistent interface for computing metrics during both training validation and final testing.

Binary PPI predictions use the PRING benchmark setting: probabilities greater than or equal to `0.5` are classified as interacting pairs, and probabilities below `0.5` are classified as non-interacting pairs. Training validation, final evaluation, and topology reconstruction all use this same threshold.

## Core Responsibilities

**Does:**
*   **Metric Calculation**: Computes performance metrics (AUROC, AUPRC, Accuracy, Sensitivity, Specificity, Precision, Recall, F1, MCC) based on model outputs.
*   **Centralized Loss Reporting**: Computes loss with `training_config.loss` via `LossConfig` so train/val/test loss paths share the same objective configuration.
*   **Inference Pass**: Runs a single pass over a data loader to collect logits/probabilities and labels. Uses `accelerator.autocast()` for mixed precision and `accelerator.gather_for_metrics()` for distributed metric gathering.
*   **Stateless Reporting**: Returns a simple dictionary of results (e.g., `{"val_loss": 0.5, "val_auroc": 0.85}`).
*   **Fixed Decision Threshold**: Uses the PRING benchmark convention of probability threshold `0.5` for binary PPI decisions.

**Does NOT:**
*   **State Management**: It does *not* change the model's training mode. The stage orchestrator is responsible for setting `model.eval()` and `torch.no_grad()`.
*   **Logging/I/O**: It does *not* write to files or logs. It returns data to the stage, which handles persistence.
*   **Global Configuration Parsing**: It does *not* parse the full run config; the stage injects selected metric names and loss config.

## Accelerator Requirement

The Evaluator requires a real `AcceleratorLike` instance — there is no optional or fallback path. The accelerator is used for:

*   `accelerator.autocast()` — mixed precision during inference.
*   `accelerator.gather_for_metrics()` — gathering predictions across ranks in distributed mode.
*   Device-aware batch movement via `move_batch_to_device()` and `forward_model()` from `src.pipeline.loops`.

## Configuration Schema

Metrics are defined in the `evaluate` section of the YAML config. `decision_threshold` is fixed at `0.5`; validation-selected thresholds are intentionally unsupported so PRING runs use the same operating point across validation, test, and topology reconstruction.

```yaml
evaluate:
  metrics: [
    "auroc",
    "auprc",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1",
    "mcc",
  ]
  decision_threshold:
    mode: fixed
    value: 0.5
```

## Usage Pattern

### 1. In the Training Stage

The stage orchestrator manages the evaluation context:

```python
# Inside stages/train.py
model.eval()
with torch.no_grad():
    val_stats = evaluator.evaluate(model, dataloaders["valid"], device, prefix="val")
model.train()

# Stage handles CSV logging
append_csv_row(...)
```

### 2. Standalone Evaluation Stage

When `run_config.stages` contains `"evaluate"`, the stage:
1.  Loads the checkpoint via `runtime.load_checkpoint()`.
2.  Resolves the fixed PRING decision threshold (`0.5`).
3.  Instantiates the Evaluator with the runtime's accelerator.
4.  Runs the evaluation pass on the test set.
5.  Writes the result to `logs/{model}/evaluate/{run_id}/evaluate.csv`.

## Output Format

The Evaluator returns a dictionary mapping metric names to values. The stage ensures these are logged consistently.

*   **Console/Log**: `INFO | Validation metrics: val_auroc=0.842, val_recall=0.877`
*   **CSV**:
    *   `training_step.csv`: Appends `Val {Metric}` columns using `training_config.logging.validation_metrics` order.
    *   `evaluate.csv`: Stores final metrics in strict order: `split,auroc,auprc,accuracy,sensitivity,specificity,precision,recall,f1,mcc`.

## DDP and Prefix Behavior

*   Evaluation runs under the shared `PipelineRuntime` accelerator. Rank 0 is responsible for persisted artifacts/logging.
*   Validation keys are prefixed by the stage (`val_*`) when `prefix="val"`.
*   Final evaluation CSV values are persisted without `test_` prefixes.
