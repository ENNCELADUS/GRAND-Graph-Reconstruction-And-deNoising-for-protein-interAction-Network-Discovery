# Evaluator and Metrics

The **Evaluator** module (`src/evaluate/base.py`) is responsible for assessing model performance. It provides a consistent interface for computing metrics during both training validation and final testing.

## Core Responsibilities

**Does:**
*   **Metric Calculation**: Computes performance metrics (e.g., AUROC, Accuracy, F1) based on model outputs.
*   **Centralized Loss Reporting**: Computes loss with `training_config.loss` via `LossConfig` so train/val/test loss paths share the same objective configuration.
*   **Inference Pass**: Runs a single pass over a data loader to collect logits/probabilities and labels.
*   **Stateless Reporting**: Returns a simple dictionary of results (e.g., `{"val_loss": 0.5, "val_auroc": 0.85}`).

**Does NOT:**
*   **State Management**: It does *not* change the model's training mode. The orchestrator is responsible for setting `model.eval()` and `torch.no_grad()`.
*   **Logging/I/O**: It does *not* write to files or logs. It simply returns data to the orchestrator.
*   **Global Configuration Parsing**: It does *not* parse the full run config; the orchestrator injects selected metric names and loss config.

## Configuration Schema

Metrics are defined in the `evaluate` section of the YAML config.

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
```

## Usage Pattern

### 1. In the Orchestrator (Run Loop)

The orchestrator manages the evaluation context:

```python
# Inside run.py
model.eval()
with torch.no_grad():
    # Evaluator performs the pass and computes metrics
    results = evaluator.evaluate(model, val_loader, device)

# Orchestrator handles logging and CSV I/O
append_csv_row(...)
```

### 2. Standalone Evaluation (Evaluate Stage Only)

When `run_config.stages` contains only `["evaluate"]`, the orchestrator:
1.  Loads the configuration and metrics list.
2.  Instantiates the Evaluator.
3.  Loads the checkpoint.
4.  Runs the evaluation pass.
5.  Writes the result to `logs/{model}/evaluate/{run_id}/evaluate.csv`.

## Output Format

The Evaluator returns a dictionary mapping metric names to values. The orchestrator ensures these are logged consistently.

*   **Console/Log**: `INFO | Validation metrics: val_auroc=0.842, val_recall=0.877`
*   **CSV**:
    *   `training_step.csv`: Appends `Val {Metric}` columns using `training_config.logging.validation_metrics` order.
    *   `evaluate.csv`: Stores final metrics in strict order: `split,auroc,auprc,accuracy,sensitivity,specificity,precision,recall,f1,mcc`.

## DDP and Prefix Behavior

*   Evaluation may run under DDP orchestration, but rank 0 is responsible for persisted artifacts/logging.
*   Validation keys are prefixed by the orchestrator (`val_*`) when `prefix="val"`.
*   Final evaluation CSV values are persisted without `test_` prefixes.
