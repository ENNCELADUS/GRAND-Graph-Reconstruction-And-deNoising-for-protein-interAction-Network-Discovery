# Trainer Design

The **Trainer** module is responsible for training-loop execution. The default
`Trainer` is intentionally scoped to pairwise mini-batch classification; model
families with different training contracts live under role-specific packages in
`src/train/`.

## Core Responsibilities

### Pairwise Trainer (`src/train/base.py`)

The `Trainer` class encapsulates the mechanics of updating pairwise classifier
weights. It requires a real `AcceleratorLike` instance — there is no fallback
builder or optional accelerator path.

**Does:**
*   **Execute `train_one_epoch(...)`**: Runs the standard forward-backward pass for a single epoch.
    *   Sets `model.train()`.
    *   Computes loss via `binary_classification_loss()`.
    *   Performs backpropagation through `accelerator.backward(loss)`.
    *   Steps the optimizer and scheduler.
*   **Manage Optimization State**: Builds and rebuilds the `Optimizer` and `Scheduler` on demand. This is critical for strategies that change trainable parameters (e.g., staged unfreezing).
*   **Handle Precision**: Mixed precision is managed by the Accelerator. The trainer uses `accelerator.autocast()` for forward passes and `accelerator.backward()` for gradient scaling — no manual `GradScaler` management.
*   **Prepare Components**: Calls `accelerator.prepare()` to wrap model, optimizer, scheduler, and dataloader for distributed execution.
*   **Expose Hooks**: Provides access to `named_parameters` for strategies to manipulate gradients.

**Does NOT:**
*   **Validation**: Validation logic belongs to the `Evaluator`, called by the stage orchestrator.
*   **Logging**: The trainer emits heartbeat logs to the provided logger, but file creation and stage management are handled by the stage via `src/utils/logging.py`.
*   **Checkpointing**: File I/O for saving models is managed by `PipelineRuntime.save_checkpoint()`.
*   **Global Configuration Parsing**: The Trainer receives only the specific configuration objects (optimizer cfg, scheduler cfg, loss cfg) it needs, not the entire global config.

### TCCIG Training (`src/train/tccig/`)

TCCIG uses a separate training package instead of subclassing the pairwise
`Trainer`. Its contract is graph-forward subgraph training: sample protein
subgraphs, load cached embeddings, call `forward_graph`, optionally update a
train-only MGAE teacher, compute TCCIG/topology losses, validate with topology
metrics, and checkpoint through `PipelineRuntime`.

The student model must keep the inference path feature-only: no target edges,
degrees, communities, neighborhoods, or Laplacian tensors are accepted by
`forward_graph`. Candidate pairs are canonicalized as undirected edges, and the
decoder uses symmetric pair features so graph scores are invariant to protein-set
ordering.

`src/train/tccig/runner.py` owns the TCCIG stage lifecycle. `trainer.py`,
`teacher.py`, `data.py`, `validation.py`, and `config.py` own their respective
roles. Shared topology-training mechanics that are not TCCIG-specific belong in
`src/train/topology/`.

### Strategies (`src/train/strategies/`)

Strategies implement the "How" and "When" of training, particularly for complex finetuning protocols. They follow a callback-style pattern.

**Role:**
*   **Lifecycle Management**: Hooks into training events: `on_train_begin`, `on_epoch_begin`, `on_epoch_end`.
*   **Parameter Control**: Decides which parameters are frozen (`requires_grad=False`) and which are trainable.
*   **Dynamic Optimization**: Triggers the Trainer to rebuild the optimizer and scheduler when the set of trainable parameters changes (e.g., during staged unfreezing).

**Example Strategy: `StagedUnfreeze`**
1.  **Start**: Freezes all layers except the head.
2.  **Epoch N**: Unfreezes the encoder.
3.  **Action**: Calls `trainer.rebuild_optimizer_and_scheduler()` to register the newly trainable parameters with the optimizer.

## Training Loop Logic

Inside `train_one_epoch`:

1.  **Zero Gradients**: `optimizer.zero_grad(set_to_none=True)`.
2.  **Forward Pass**: Compute model output and loss inside `accelerator.autocast()`.
3.  **Backward Pass**: `accelerator.backward(loss)` — the accelerator handles gradient scaling for mixed precision and DDP synchronization.
4.  **Optimizer Step**: `optimizer.step()`.
5.  **Scheduler Step**: Called after the optimizer step.
6.  **Return**: A lightweight dictionary of statistics (e.g., `{"loss": avg_loss, "lr": current_lr}`).

## Architecture

```
src/train/
├── base.py                # Trainer class (requires AcceleratorLike)
├── config.py              # OptimizerConfig, SchedulerConfig, LossConfig
├── topology/              # shared topology-training mechanics
├── tccig/                 # TCCIG-specific training pipeline
└── strategies/
    ├── lifecycle.py       # TrainingStrategy, NoOpStrategy, StagedUnfreezeStrategy
    └── ohem.py            # OHEMSampleStrategy
```
