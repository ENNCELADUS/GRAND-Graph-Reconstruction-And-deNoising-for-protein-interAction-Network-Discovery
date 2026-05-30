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
    tccig_train.py      # thin TCCIG training stage wrapper
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
*   **Stage Logging Bootstrap**: Creates stage loggers and log artifact directories so setup events are persisted in `log.log`. Model checkpoint directories are created only for stages that save checkpoints.

### 2. Train Stage

Trains the model on the configured train/valid split.

1.  The `Trainer` is created with optimizer/scheduler/loss configs and the runtime's accelerator.
2.  The trainer calls `accelerator.prepare()` to wrap model, optimizer, scheduler, and dataloader.
3.  Each epoch: `train_one_epoch()` → validation via `Evaluator` → CSV logging → checkpoint on improvement.
4.  Early stopping is checked via `EarlyStopping`. In distributed mode, the stop flag is reduced across ranks.
5.  Output: `best_model.pth` checkpoint path.

### 3. Topology Fine-tune Stage (optional)

Refines the model on graph-supervision data with topology-aware losses (edge prediction, node degree, graph similarity).

1.  Loads the train-stage checkpoint or `run_config.load_checkpoint_path` when `topology_finetune.init_mode: warm_start`; starts from scratch when `init_mode: scratch`.
2.  Runs an internal training loop with topology-specific loss computation and edge-cover sampling.
3.  Output: refined checkpoint path.

Topology fine-tune checkpoints are model-weight checkpoints only. They are suitable for warm-starting a later stage or a new run, but they do not restore optimizer state, scheduler state, epoch counters, or early-stopping state for a true interrupted-run resume.

### 4. TCCIG Train Stage (optional)

Trains the TCCIG feature-only graph generator from scratch.

1.  The public stage wrapper validates `model_config.model: tccig` and the `forward_graph` contract.
2.  `src/train/tccig/runner.py` builds TCCIG data, the optional train-only MGAE teacher, trainer, and validation components.
3.  Each epoch: sampled subgraph training via `TCCIGStudentTrainer` → topology validation → CSV logging → checkpoint on improvement.
4.  Output: `best_model.pth` checkpoint path consumed by `evaluate` and `topology_evaluate`.

TCCIG training is mutually exclusive with `train` and `topology_finetune` in the current stage-order contract.

### 5. Adapt Stage (optional, SHOT)

Runs SHOT domain adaptation when `training_config.domain_adaptation.enabled=true`. Inserted automatically before evaluation.

1.  Loads the upstream checkpoint.
2.  Computes pseudo-labels and centroids on target data.
3.  Optimizes entropy and diversity losses.
4.  Output: adapted checkpoint path.

### 6. Evaluate Stage

Assesses final model performance on the test set.

1.  Loads the best available checkpoint (from adapt, tccig_train, topology_finetune, or train).
2.  Resolves the fixed PRING decision threshold (`0.5`).
3.  Computes all configured metrics via `Evaluator`.
4.  Writes `evaluate.csv`.

### 7. Topology Evaluate Stage (optional)

Runs PRING-style graph reconstruction and computes topology metrics (graph similarity, relative density, MMD distances).

1.  Loads the best available checkpoint.
2.  Runs inference to produce pairwise predictions. TCCIG models use Graph Assembly instead: encode the unique test proteins once, score PRING candidate pairs in chunks, predict `m_hat`, and select top-`m_hat` edges.
3.  Reconstructs graphs and computes topology metrics.
4.  Writes `topology_metrics.json` and prediction files.

## Artifact Contracts

* `logs/{model}/{stage}/{run_id}/` is created for every selected stage.
* `models/{model}/{stage}/{run_id}/` is created only for checkpoint-writing stages: `train`, `topology_finetune`, `tccig_train`, and `adapt`.
* `training_step.csv` strict header order:
  * `Epoch,Epoch Time,Train Loss,Val Loss,Val <metric>...,Learning Rate`
  * `Val <metric>` columns follow `training_config.logging.validation_metrics` order.
* `evaluate.csv` strict header order:
  * `split,auroc,auprc,accuracy,sensitivity,specificity,precision,recall,f1,mcc`
* `topology_finetune_step.csv` records pairwise, topology, coverage, timing, GPU-memory, and learning-rate fields for topology fine-tuning.
* `tccig_train_step.csv` records the analogous TCCIG graph-forward training and topology-validation fields.
* `topology_metrics.json`, `topology_metrics.csv`, and `graph_eval_results.pkl` are the persisted topology-evaluation summary/detail artifacts.
* No `test_` prefixes are used in persisted eval CSV columns.

## Run Stages

Execution is controlled by ordered `run_config.stages`:

*   `["train", "evaluate"]`
*   `["train", "topology_finetune", "evaluate", "topology_evaluate"]`
*   `["tccig_train", "evaluate", "topology_evaluate"]`
*   `["evaluate"]` (requires `run_config.load_checkpoint_path`)

Stage ordering is enforced: `train → topology_finetune → tccig_train → evaluate → topology_evaluate`. In current configs, `tccig_train` cannot be combined with `train` or `topology_finetune`. When `evaluate` is selected and SHOT adaptation is enabled, the `adapt` stage is inserted automatically before evaluation.

## Launcher

* Canonical local entry: `uv run python -m src.run --config configs/v3/v3.yaml`.
* TCCIG local entry: `uv run python -m src.run --config configs/tccig/tccig.yaml`.
* Package entry also works: `uv run python -m src.pipeline --config configs/v3/v3.yaml`.
* Legacy script form works: `uv run python src/run.py --config configs/v3/v3.yaml` (delegates to `src.pipeline`).
* DDP launch: `uv run python -m torch.distributed.run --nproc_per_node=N --module src.run --config ...`.
* Canonical HPC launchers: `scripts/v3.sh`, `scripts/v4.sh`, `scripts/v5.sh`, `scripts/v3_ablation.sh`, `scripts/v3_1_ablation.sh`, `scripts/tuna.sh`, and `scripts/tccig.sh`.
* Launcher scripts inspect `optimization.enabled`; optimized configs dispatch to `--module src.optimize.run`, otherwise they dispatch to `--module src.run`.
