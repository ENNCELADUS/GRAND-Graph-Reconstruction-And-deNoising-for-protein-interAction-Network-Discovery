# Plan: DeepSpeed ZeRO-2 Backend for Topology Finetune & Evaluate

## Context

The topology finetune stage is the most compute-intensive part of the pipeline — it runs gradient accumulation over sampled subgraphs with multiple loss terms (BCE, graph similarity, density, MMD). Currently the only distributed backend is DDP via HuggingFace Accelerate.

DeepSpeed ZeRO Stage 2 partitions optimizer states and gradients across GPUs, reducing per-GPU memory and enabling larger effective batch sizes. It also provides fused Adam and communication overlap for wall-clock speedups.

The goal is to:
- Write a detailed refactor plan in `docs/topology_finetune/`
- Execute the implementation as a **strict refactor** (no deprecated code, no backup comments)

---

## Deliverable

- Create: `docs/topology_finetune/deepspeed_refactor.md` (full design doc)
- Implement the refactor across the codebase

---

## Files to Modify

| File | Change |
|------|--------|
| `docs/topology_finetune/deepspeed_refactor.md` | NEW — full design doc |
| `pyproject.toml` | Add `deepspeed>=0.16.0` dependency |
| `src/pipeline/config.py` | Add `backend` field to `DeviceSettings` |
| `src/pipeline/runtime.py` | Backend-aware `build_accelerator()`, extend `AcceleratorLike` with `accumulate()` / `no_sync()` |
| `src/pipeline/stages/topology_finetune.py` | Refactor `_fit_epoch()` to use `accumulate()` / `no_sync()`, remove manual grad accumulation |
| `src/pipeline/stages/topology_evaluate.py` | Clean up unused `use_amp` parameter threading |
| `tests/runtime_helpers.py` | Add `accumulate()`, `no_sync()`, `gradient_accumulation_steps` to `NoOpAccelerator` |

---

## Implementation Steps

### Step 1: Write the Design Doc

Create `docs/topology_finetune/deepspeed_refactor.md` with:

- Motivation and backend comparison (DDP vs DeepSpeed ZeRO-2)
- Config schema changes (`device_config.backend`)
- Runtime changes:
  - `build_accelerator()` branching
  - `AcceleratorLike` protocol additions
- Topology finetune training loop refactor:
  - Path 1: `accumulate()`
  - Path 2: `no_sync()`
- Topology evaluate inference changes
- Checkpoint save/load compatibility
- Cleanup inventory (lines to remove)

---

### Step 2: Add Dependency

Add to `pyproject.toml`:

```toml
deepspeed >= 0.16.0
````

---

### Step 3: Config — Add Backend Field

* Add `backend: str` to `DeviceSettings` (default: `"ddp"`)
* Parse from `device_config.backend` in `PipelineConfig.from_dict()`
* Validate: must be `"ddp"` or `"deepspeed"`

---

### Step 4: Runtime — Backend-Aware Accelerator

* Extend `AcceleratorLike` with:

  * `accumulate()`
  * `no_sync()`

* Modify `build_accelerator()`:

  * Accept `backend` param
  * If `"deepspeed"`:

    * Use `DeepSpeedPlugin(zero_stage=2, ...)`
    * Pass `gradient_accumulation_steps`
  * Else use default DDP

* Modify `build_runtime()`:

  * Pass `config.device.backend`

---

### Step 5: Refactor Topology Finetune Training Loop

* Set `accelerator.gradient_accumulation_steps` before epoch loop

#### Path 1: `subgraphs_per_forward == 1`

* Replace manual grad accumulation with:

  ```python
  with accelerator.accumulate(model):
      ...
  ```
* Remove:

  * Loss division
  * Boundary checks
* Always call:

  * `optimizer.step()`
  * `optimizer.zero_grad()`

#### Path 2: `subgraphs_per_forward > 1`

* Use:

  * `accelerator.no_sync(model)` for non-final steps
  * Explicit `optimizer.step()` at window boundaries

* Remove:

  * Manual loss division

#### Additional Cleanup

* Remove initial `optimizer.zero_grad()` before loop
* Handle remainder:

  * Final `optimizer.step()` after loop if needed

---

### Step 6: Clean Up Topology Evaluate

* Remove unused `use_amp` parameter from:

  * `_predict_topology_labels()`
* Remove:

  * `del use_amp`
* Update caller:

  * `run_topology_evaluation_stage()`

---

### Step 7: Update Test Helpers

Update `NoOpAccelerator`:

* Add:

  * `accumulate()` (no-op context manager)
  * `no_sync()` (no-op context manager)
  * `gradient_accumulation_steps: int = 1`

---

### Step 8: Run Tests

```bash
uv run python -m pytest
uv run ruff check --fix .
uv run ruff format .
```

---

## Key Design Decisions

1. **Programmatic DeepSpeed Config**
   No separate JSON — use `DeepSpeedPlugin` kwargs in `build_accelerator()` for centralized config.

2. **Dual Strategy**

   * Path 1 → `accumulate()`
   * Path 2 → `no_sync()`
     Because multi-forward-per-step does not map cleanly to `accumulate()`.

3. **Checkpoint Compatibility**
   ZeRO-2 does **not** partition model parameters → existing checkpoint logic works unchanged.

4. **DDP Padding Behavior**
   Zero-gradient backward from padding remains valid in both DDP and DeepSpeed.

---

## Verification

1. `uv run python -m pytest` → all tests pass
2. `uv run ruff check .` → no lint errors
3. `uv run mypy src` → type checks pass
4. Manual:

   * Verify `NoOpAccelerator` satisfies `AcceleratorLike` (`runtime_checkable`)
