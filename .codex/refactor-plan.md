# Refactor Plan: Collapse Dual-Runtime into Single Accelerator-Owned Pipeline

## Context

The codebase has a dual-runtime design:

- `src/run/`: old layer, 2,283 lines
- `src/pipeline/`: new layer, 1,015 lines

These layers form a circular dependency:

- `engine.py` imports stage functions from `src/run/`
- `src/run/__init__.py` imports the orchestrator from `src/pipeline/`

The current design also has:

- 4 pure re-export shim files
- 3 checkpoint implementations
- 2 model-unwrap implementations
- 2 stage-runtime setup implementations
- 2 barrier implementations
- Stage functions that take 8 positional arguments instead of a single runtime object

The goal is one `PipelineRuntime` that owns the `Accelerator`. All stages should receive that runtime. There should be no fallback `ensure_accelerator()`, no compatibility wrappers, and no circular dependency.

## Target Layout

```text
src/pipeline/                          # canonical home
  __init__.py             (~30 lines)   # public API
  __main__.py             (~26 lines)   # CLI entry
  bootstrap.py            (~54 lines)   # parse_args, seed, logging
  config.py               (~149 lines)  # PipelineConfig typed view
  runtime.py              (~330 lines)  # PipelineRuntime + AcceleratorLike + DistributedContext + checkpoint ops
  loops.py                (~100 lines)  # move_batch, forward_model, reduce, gather
  engine.py               (~280 lines)  # orchestrator; passes runtime to stages
  stages/
    __init__.py           (~10 lines)
    train.py              (~480 lines)
    evaluate.py           (~170 lines)
    adapt.py              (~520 lines)
    topology_finetune.py  (~1180 lines)
    topology_evaluate.py  (~500 lines)

src/run/                               # thin backward-compatible shim
  __init__.py             (~60 lines)   # re-exports from src.pipeline
  __main__.py             (~8 lines)    # delegates to src.pipeline

src/run.py                             # script entrypoint; unchanged
```

## Phases

### Phase 0: Safety Net

- [x] Run the full test suite and record the baseline.
- [x] Add a temporary test that imports every symbol from `src.run.__all__` to lock the public API contract.

### Phase 1: Absorb `accelerator.py` and `distributed.py` Into `runtime.py`

- [x] Move `AcceleratorLike`, `build_accelerator()`, and `distributed_context_from_accelerator()` from `src/utils/accelerator.py` to `src/pipeline/runtime.py`.
- [x] Move `DistributedContext` and `distributed_barrier()` from `src/utils/distributed.py` to `src/pipeline/runtime.py`.
- [x] Delete the original compatibility files instead of keeping 3-line re-export shims.
- [x] Update imports in:
  - `src/train/base.py`
  - `src/evaluate/base.py`
  - `src/pipeline/loops.py`
  - all stage files
- [x] Touch and review these files:
  - `runtime.py`
  - `accelerator.py`
  - `distributed.py`
  - `train/base.py`
  - `evaluate/base.py`
  - `loops.py`
  - all `stage_*.py` files

### Phase 2: Unify Checkpoint, Unwrap, and Stage Runtime

- [x] Delete the following from `src/run/stage_train.py`:
  - `_unwrap_model()`
  - `_save_checkpoint()`
  - `_load_checkpoint()`
  - `_build_stage_runtime()`
  - `_build_stage_logger()`
- [x] Make all stages use `PipelineRuntime` checkpoint, path, and logger methods from `src.pipeline.runtime`.
- [x] Skip the temporary `build_stage_runtime_compat()` adapter and move directly to runtime-first stage calls.
- [x] Replace `distributed_barrier()` usage in `stage_evaluate.py` with `runtime.barrier()`.
- Expected reduction: about 90 lines.

### Phase 3: Simplify Stage Signatures to Accept `PipelineRuntime`

- [x] Use this new pattern:

```python
run_training_stage(runtime, model, dataloaders)
```

Instead of:

```python
run_training_stage(
    stage,
    config,
    model,
    device,
    dataloaders,
    run_id,
    distributed_context,
    accelerator,
)
```

- [x] Make stages extract from the runtime:
  - `runtime.device`
  - `runtime.is_main_process`
  - `runtime.accelerator`
  - `runtime.config.raw`
  - `runtime.stage_run_ids[stage]`
- [x] Remove `ensure_accelerator()` from `loops.py`; runtime always provides an accelerator.
- [x] Update `Trainer.__init__` and `Evaluator.__init__` to require a non-`None` accelerator.
- [x] Simplify `engine.py:execute_pipeline_with_runtime()` call sites.
- [x] Remove the compatibility adapter requirement by never adding it.
- [x] Refactor and test the `train` stage.
- [x] Refactor and test the `evaluate` stage.
- [x] Refactor and test the `adapt` stage.
- [x] Refactor and test the `topology_finetune` stage.
- [x] Refactor and test the `topology_evaluate` stage.
- Expected reduction: about 145 lines.

### Phase 4: Move Stages Into `src/pipeline/stages/`

- [x] Move `src/run/stage_train.py` to `src/pipeline/stages/train.py`.
- [x] Apply the same move for all 5 stages.
- [x] Delete `src/pipeline/topology.py`, the 19-line re-export shim.
- [x] Delete `src/run/bootstrap.py`, the 24-line wrapper.
- [x] Delete `src/run/pipeline_orchestrator.py`, the 40-line wrapper.
- [x] Delete `src/run/__init__.py` and all `src/run/*.py` compatibility aliases.
- [x] Update test imports.
- Expected file count change: 3 files deleted, 1 file created, net -2 files.

### Phase 5: Clean Up Shims and Dead Code

- [x] Grep for remaining imports of:
  - `src.utils.accelerator`
  - `src.utils.distributed`
  - `src.run.stage_*`
- [x] Update `src/optimize/run.py` to import from `src.pipeline.engine`.
- [x] Delete `src/utils/ohem_sample_strategy.py`, the 5-line shim.
- [x] Delete the `_generate_run_id` compatibility wrapper in `runtime.py`.
- [x] Optionally delete the `src/utils/accelerator.py` and `src/utils/distributed.py` shims if no external consumers remain.
- Expected reduction: about 260 lines.

## Expected Impact

| Metric | Before | After |
|---|---:|---:|
| Files in scope | 19 | 13 |
| Lines in scope | ~4,464 | ~3,830 |
| Checkpoint implementations | 3 | 1 |
| Unwrap implementations | 2 | 1 |
| Stage-runtime implementations | 2 | 1 |
| Barrier implementations | 2 | 1 |
| Re-export shim files | 4 | 1 |
| Circular dependencies | 1 | 0 |
| `ensure_accelerator()` calls | 7 | 0 |
| Stage function argument count | 8 | 3 |

## Risk Areas

1. `src/optimize/run.py`: lazy import of `src.run.pipeline_orchestrator.execute_pipeline`. Single call site; update in Phase 5.
2. Test monkeypatching: `test_distributed_utils.py` patches `distributed_module.dist`. Retarget to `src.pipeline.runtime` after Phase 1.
3. `Trainer` and `Evaluator` constructors: currently accept `accelerator=None` and call `ensure_accelerator()`. Phase 3 makes accelerator required. Tests constructing these without an accelerator need a stub.
4. `stage_train.py` training loop: uses `_save_checkpoint` in 3 places with `is_main_process` guards. `runtime.save_checkpoint()` handles barriers internally, but the conditional save logic is stage-specific and must preserve the guard.
5. `stage_topology_finetune.py`: 1,251 lines, largest file. Move is mechanical but needs careful review.

## Verification

After each phase:

1. [x] `uv run python -m pytest tests/unit tests/integration`
2. [x] `uv run ruff check .`
3. [x] `uv run mypy src`
4. [x] Verify `python -m src.pipeline --help` and `python src/run.py --help` entrypoint smoke tests.

---

## Phase 6: Post-Refactor Cleanup (remaining tasks)

These tasks were identified during code review of the completed refactor.

### 6a — Fix duplicate accelerator stubs in tests

- [ ] In `tests/unit/test_trainer_evaluator.py`, delete the local `FakeAccelerator` class (line 156).
- [ ] Import `NoOpAccelerator` from `tests/runtime_helpers.py` instead.
- [ ] Verify `NoOpAccelerator` has all required protocol attributes (`wait_for_everyone`, `unwrap_model`, `save`, distributed properties); add any that are missing.

### 6b — Delete thin wrapper functions in `src/pipeline/stages/adapt.py`

- [ ] Delete `_move_batch_to_device()` (line 54) — one-liner wrapping `move_batch_to_device` from `loops.py`.
- [ ] Delete `_forward_batch_without_labels()` (line 59) — keep the logic but inline it at the call site.
- [ ] Delete `_forward_model()` (line 64) — one-liner wrapping `forward_model` from `loops.py`.
- [ ] Update the 3 call sites (lines 224–227) to call `move_batch_to_device` and `forward_model` from `src.pipeline.loops` directly.

### 6c — Remove `checkpoint_paths` from `PipelineRuntime` (Option A)

- [ ] Remove `checkpoint_paths: dict[str, Path | None]` field from `PipelineRuntime` in `src/pipeline/runtime.py`.
- [ ] Update `src/pipeline/engine.py` to pass checkpoint paths as explicit keyword args to each stage that needs one:
  - `run_evaluation_stage(runtime, model, dataloaders, checkpoint_path=...)`
  - `run_shot_adaptation_stage(runtime, model, dataloaders, checkpoint_path=...)`
  - `run_topology_evaluation_stage(runtime, model, dataloaders, checkpoint_path=...)`
  - `run_topology_finetuning_stage(runtime, model, dataloaders, checkpoint_path=...)`
- [ ] Update each stage function signature to accept `checkpoint_path` as a keyword arg.
- [ ] Update tests that construct `PipelineRuntime` directly.

### 6d — Delete dead `_stage_logger_name` in `src/pipeline/stages/train.py`

- [ ] Delete `_stage_logger_name()` (line 40) — `runtime.stage_logger()` already builds the logger name internally.
- [ ] Verify no other call sites reference it.

### 6e — Delete `src/apps/optimize/` (pure re-export shim)

- [ ] Delete `src/apps/optimize/__init__.py`
- [ ] Delete `src/apps/optimize/__main__.py`
- [ ] Delete `src/apps/optimize/run.py`
- [ ] Delete `src/apps/` directory if it becomes empty.
- [ ] Verify no scripts or tests reference `src.apps`.

### 6f — Fix `src/__init__.py` — add V3_1

- [ ] Add `V3_1` to the import and `__all__` in `src/__init__.py` for consistency with `src/model/__init__.py`.

### 6g — Fix `src/utils/__init__.py` — remove cross-package re-export

- [ ] Remove `OHEMSampleStrategy` and `select_ohem_indices` from `src/utils/__init__.py`.
- [ ] These are already exported from `src/train/__init__.py`; `src/utils` should not reach into `src/train`.
- [ ] Update any consumers that import `OHEMSampleStrategy` from `src.utils` to import from `src.train.strategies.ohem` instead.

### Verification

After all 6x tasks:

1. [ ] `uv run python -m pytest tests/unit tests/integration` — all pass
2. [ ] `uv run ruff check .` — clean
3. [ ] `uv run mypy src` — clean
