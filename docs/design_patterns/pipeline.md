# Pipeline Architecture

The GRAND pipeline is a single-runtime, config-driven orchestration system. All execution flows through `src/pipeline/`, which owns one `PipelineRuntime` instance backed by a real HuggingFace `Accelerator`. The runtime handles device placement, mixed precision, DDP synchronization, checkpoint I/O, and process coordination. Stages receive the runtime object directly — no fallback builders, no separate device/distributed/accelerator arguments.

## Core Philosophy

1. **Single Runtime**: One `PipelineRuntime` is built once at startup and threaded through every stage. The Accelerator it owns is the only execution orchestrator — there is no second code path for device management, precision, or synchronization.
2. **Config-Driven Execution**: All behaviors — model hyperparameters, training duration, optimization strategies, data paths, and stage selection — are defined in a YAML configuration file.
3. **Stage-Based Workflow**: The pipeline supports explicit stage selection via `run_config.stages`. Stages are independent, composable, and communicate via checkpoints and run IDs.

## Package Layout

```
src/pipeline/
  __init__.py           # public API
  __main__.py           # CLI entry: python -m src.pipeline
  bootstrap.py          # parse_args, set_global_seed, configure_root_logging
  config.py             # PipelineConfig typed view over raw YAML
  runtime.py            # PipelineRuntime, AcceleratorLike, DistributedContext, checkpoint ops
  loops.py              # move_batch_to_device, forward_model, reduce_scalar_mapping, gather
  engine.py             # execute_pipeline orchestrator
  stages/
    train.py            # training stage
    evaluate.py         # evaluation stage
    adapt.py            # SHOT domain adaptation stage
    topology_finetune.py # topology-aware fine-tuning stage
    topology_evaluate.py # topology evaluation stage
```

## PipelineRuntime

The `PipelineRuntime` dataclass is the single object that stages depend on:

```python
@dataclass
class PipelineRuntime:
    config: PipelineConfig
    accelerator: AcceleratorLike
    device: torch.device
    distributed: DistributedContext
    stage_run_ids: dict[str, str]
```

It exposes convenience properties: `is_main_process`, `is_distributed`, `rank`, `world_size`. It also provides checkpoint operations (`save_checkpoint`, `load_checkpoint`), barriers (`barrier`), and stage directory/logger setup (`stage_paths`, `stage_logger`).

## Stage Signatures

Every stage function follows the same pattern:

```python
def run_training_stage(
    runtime: PipelineRuntime,
    model: nn.Module,
    dataloaders: DataLoaderMap,
) -> Path:
```

Stages extract what they need from the runtime: `runtime.device`, `runtime.accelerator`, `runtime.config.raw`, `runtime.stage_run_ids["train"]`, `runtime.is_main_process`. No `ensure_accelerator()` fallback exists — the runtime always provides a real accelerator.

## Pipeline Stages

### 1. Setup and Initialization

Before any stage runs, the engine performs:

*   **Config Loading**: Parses the YAML configuration via `src/pipeline/config.py` into a typed `PipelineConfig`.
*   **Runtime Construction**: `build_runtime()` creates the Accelerator, resolves the device, builds the `DistributedContext`, and generates run IDs for each stage. Run IDs are broadcast from rank 0 in distributed mode.
*   **Data Loading**: Instantiates data loaders using `src/utils/data_io.build_dataloaders()`.
*   **Model Initialization**: Selects and instantiates the model from `src/model/` based on `model_config`.
*   **Stage Logging Bootstrap**: Creates stage loggers and artifact directories so setup events are persisted in `log.log`.

### 2. Train Stage

Trains the model on the configured train/valid split.

1.  The `Trainer` is created with optimizer/scheduler/loss configs and the runtime's accelerator.
2.  The trainer calls `accelerator.prepare()` to wrap model, optimizer, scheduler, and dataloader.
3.  Each epoch: `train_one_epoch()` → validation via `Evaluator` → CSV logging → checkpoint on improvement.
4.  Early stopping is checked via `EarlyStopping`. In distributed mode, the stop flag is reduced across ranks.
5.  Output: `best_model.pth` checkpoint path.

### 3. Topology Fine-tune Stage (optional)

Refines the model on graph-supervision data with topology-aware losses (edge prediction, node degree, graph similarity).

1.  Loads the train-stage checkpoint (warm start) or starts from scratch.
2.  Runs an internal training loop with topology-specific loss computation and edge-cover sampling.
3.  Output: refined checkpoint path.

### 4. Adapt Stage (optional, SHOT)

Runs SHOT domain adaptation when `training_config.domain_adaptation.enabled=true`. Inserted automatically before evaluation.

1.  Loads the upstream checkpoint.
2.  Computes pseudo-labels and centroids on target data.
3.  Optimizes entropy and diversity losses.
4.  Output: adapted checkpoint path.

### 5. Evaluate Stage

Assesses final model performance on the test set.

1.  Loads the best available checkpoint (from adapt, topology_finetune, or train).
2.  Resolves the decision threshold (fixed or `best_f1_on_valid`).
3.  Computes all configured metrics via `Evaluator`.
4.  Writes `evaluate.csv`.

### 6. Topology Evaluate Stage (optional)

Runs PRING-style graph reconstruction and computes topology metrics (graph similarity, relative density, MMD distances).

1.  Loads the best available checkpoint.
2.  Runs inference to produce pairwise predictions.
3.  Reconstructs graphs and computes topology metrics.
4.  Writes `topology_metrics.json` and prediction files.

## Artifact Contracts

* `training_step.csv` strict header order:
  * `Epoch,Epoch Time,Train Loss,Val Loss,Val <metric>...,Learning Rate`
  * `Val <metric>` columns follow `training_config.logging.validation_metrics` order.
* `evaluate.csv` strict header order:
  * `split,auroc,auprc,accuracy,sensitivity,specificity,precision,recall,f1,mcc`
* No `test_` prefixes are used in persisted eval CSV columns.

## Run Stages

Execution is controlled by ordered `run_config.stages`:

*   `["train", "evaluate"]`
*   `["train", "topology_finetune", "evaluate", "topology_evaluate"]`
*   `["evaluate"]` (requires `run_config.load_checkpoint_path`)

Stage ordering is enforced: `train → topology_finetune → evaluate → topology_evaluate`. When `evaluate` is selected and SHOT adaptation is enabled, the `adapt` stage is inserted automatically before evaluation.

## Launcher

* Canonical HPC launchers: `scripts/v3.sh`, `scripts/v4.sh`, `scripts/v5.sh`.
* CLI entry: `python -m src.pipeline --config configs/v5.yaml`
* Legacy script: `python src/run.py --config configs/v5.yaml` (delegates to `src.pipeline`)
* DDP launch: `python -m torch.distributed.run --nproc_per_node=N --module src.pipeline --config ...`
* HPC scripts use `--module src.run` which delegates to `src.pipeline` via `src/run.py`.
