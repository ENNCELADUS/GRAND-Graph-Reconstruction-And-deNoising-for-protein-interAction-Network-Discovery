# exp02-rerun-fix design

Date: 2026-06-30
Branch: `exp02-topology-redesign`
Status: design (awaiting implementation plan)

## Context

Run `02_balanced_subset` completed 40 epochs without the prior OOM, but the
hard topology metrics looked catastrophic. Analysis of `artifacts/exp02_rerun`
surfaced **two distinct, independent failures**:

1. **Reporting/selection failure (threshold).** Validation, checkpoint
   selection, and both test paths apply a fixed refined-output threshold of
   `0.5`. An offline sweep (job `929498`) showed `0.5` builds a graph ~32× too
   dense (`val_topology_loss = 7631.5`, `relative_density = 31.9`), while `0.97`
   gives `val_topology_loss = 8.95`, `relative_density = 0.92`. Because the
   monitor used `0.5`, `best_model.pt` is the epoch-1 graph, not the genuinely
   best epoch. The hard threshold is a non-usable operating point for this run.

2. **Training-dynamics failure (the deeper problem).** The differentiable
   `train_topology_loss` — the soft objective the model actually optimizes,
   independent of any hard threshold — **rises monotonically** from `9.71`
   (epoch 3) to `12.17` (epoch 40), and keeps rising after the topology scale
   saturates at `1.0` (epoch 7). Evidence from `training_summary.json`:

   | signal | epoch 3 | epoch 40 |
   |---|---:|---:|
   | `train_topology_loss` | 9.707 | 12.172 |
   | `train_topo_relative_density` | 1.088 | 1.386 |
   | `train_topo_degree_mmd` | 0.589 | 0.672 |
   | `train_bce_loss` | 0.992 | 0.746 |
   | `val_auprc` | 0.685 | 0.739 |
   | selected edges @0.5 | 237,773 | 357,008 |

   Density dominates the loss (~91% at epoch 40: `0.746 + 8·1.386 + 0.5·0.672 =
   12.17`). `train_topology_loss` correlates ~0.9999 with selected edges. BCE
   falls and AUPRC rises while topology, edges, and density all worsen: the BCE
   direction (補 FN — train targets are FN:FP ≈ 5.26×) pushes the refiner into
   an edge-adder, and the topology objective is not winning. The asymmetric
   residual anchor (correct one-sided form, weight `1e-4`, final weighted value
   `~1.3e-4`) is too weak to counter the global positive logit push.

The threshold fix is therefore necessary but **not sufficient** — it makes the
monitor/checkpoint trustworthy, but the training dynamics need a separate probe.

## Goal

One rerun that (A) makes the validation monitor, checkpoint selection, and test
reporting trustworthy under a calibrated operating point, and (B) provides a
gated diagnostic — topology-only epochs after warmup — to answer whether the
topology gradient *alone* can reduce topology loss.

## Scope

**In scope**
- Part A: calibrated refined-output threshold (validation + checkpoint + both
  test paths), per-epoch and best-epoch calibration recorded in JSON.
- Part B: opt-in topology-only epochs after a configured epoch boundary.

**Out of scope (deferred / unchanged)**
- FP/FN reweighting in `sample_epoch_edge_targets`. This is the **Task-2 lever**,
  gated on the Part B probe result. If topology-only epochs reduce topology
  loss, BCE-vs-topology conflict is confirmed and FP reweighting is the next
  experiment. If they do not, the topology gradient/objective itself is the
  suspect and reweighting cannot help.
- The topology loss math, the subset sampler, `topology_subset.py`.
- CSV schema. The JSON `history` in `training_summary.json` already carries
  `train_topology_loss` and the per-epoch topology components; that is
  sufficient for review (confirmed with user). `TCCIG_TRAIN_CSV_COLUMNS` is
  untouched.

## Part A — Calibrated refined-output threshold

### Config schema

Reuse the existing `type` discriminator on `graph_selection.refined_output_rule`
(no new top-level `mode` key). Current production form:

```yaml
graph_selection:
  refined_output_rule:
    type: threshold
    value: 0.5
```

Calibrated form for the rerun:

```yaml
graph_selection:
  refined_output_rule:
    type: calibrated
    objective: val_topology_loss      # only supported objective for now
    grid: [0.5, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99]
```

- `type: threshold` (or absent) → byte-for-byte today's fixed behavior.
- `type: calibrated` → per-epoch grid calibration described below.
- A `mode:` alias may be accepted for back-compat, but the spec's primary path
  and all examples use `type:` to keep the parser from forking.
- `objective` currently must be `val_topology_loss` (the monitor quantity).
  Other values raise a clear `ValueError`.
- `grid` is a fixed coarse list of probabilities in `[0, 1]`; required when
  `type: calibrated`.

### Validation (per epoch)

`_evaluate_validation_topology_rules` already runs inference once and calls
`_validation_topology_metrics` for a single rule. In calibrated mode:

1. Run refined inference once (unchanged).
2. Evaluate `_validation_topology_metrics` for each threshold in `grid` (reusing
   `edges_from_rule` + `evaluate_graph_samples` — the same logic the offline
   `threshold_sweep.py` already validated).
3. Select `argmin objective` (i.e. min `val_topology_loss`).
4. That threshold's metrics become the epoch's reported topology metrics
   (`val_topology_loss`, `graph_sim`, `relative_density`, `deg_dist_mmd`,
   `cc_mmd`, `positive_edges`) and the epoch's `selected_rule`.

Cost: ~`len(grid)`× the validation-metric compute per epoch; inference still
runs once. Grid size is the tuning knob if this is too slow.

### Checkpoint selection

`monitor_value` derives from the calibrated minimum, so `_is_better_monitor`
selects the genuinely-best epoch. `best_model.pt` stores the best epoch's
`selected_rule` payload (as today, via `best_selected_rule_payload`), now
carrying the calibrated threshold rather than a fixed `0.5`.

### Test (both paths consume the calibrated rule)

In calibrated mode, **both** test paths use the best epoch's `selected_rule`
instead of the config's fixed `refined_output_rule.value`:

- `run_pairwise_test` (`test.py:112`): threshold-free metrics (AUPRC/AUROC) stay
  threshold-free; the **refined point metrics** (precision/recall/F1 at a
  threshold) use `refiner_state.selected_rule.value`.
- `run_topology_test` (`test.py:215`): the selected-edge construction uses
  `refiner_state.selected_rule`.

Both consume the *same* calibrated threshold. Test never sweeps. This avoids the
incoherent report where pairwise_test would sit at `0.5` and topology_test at
`0.97`. Plumbing: `run_tccig_pipeline` passes the checkpoint's `selected_rule`
(when `type: calibrated`) as the effective refined-output rule into both test
functions; in `threshold` mode it passes the configured fixed rule as today.

### `graph_selection.rules` handling (no silent ignore)

Today `parse_rules(...)[0]` (the `0.5` in `rules`) is passed to validation
topology. In calibrated mode this list is **ignored for refined-output
selection**, but not silently:

- The grid is the only input to refined-output selection.
- If `graph_selection.rules` is also present, record it in the run manifest
  under `ignored_legacy_rules` so the override is auditable.

### Per-epoch JSON record

Add the full calibrated rule to each epoch's `history` entry (and continue
storing the best rule in `training_summary.json` / checkpoint):

```json
"selected_rule": {
  "type": "threshold",
  "value": 0.97,
  "source": "validation_calibration"
}
```

Today only `selected_rule_positive_edges` is in `history`; this records the
actual threshold chosen each epoch so the calibration trajectory is reviewable.
The best epoch's `selected_rule` continues to be persisted in the summary and
checkpoint as it is today.

## Part B — Topology-only epochs (gated diagnostic)

### Config schema

```yaml
refiner:
  topology_training:
    topo_only_after_epoch: null   # null/absent = off (default)
```

For the rerun, set `topo_only_after_epoch: 7` — epochs with `epoch >= 7` skip
the BCE loop. Rationale: the topology scale only reaches `1.0` at epoch 7 under
the current `warmup_epochs: 1, ramp_epochs: 5` schedule; starting earlier would
probe under a partially-ramped topology gradient and weaken the signal.
Semantics are **inclusive** (`epoch >= N`).

### Mechanics

Within each epoch, the BCE per-batch loop (`s2gae.py:1166-1189`) and the
topology backward step (`s2gae.py:1193+`) are already separate sequential
phases. On a topo-only epoch:

- Skip the BCE per-batch loop entirely. The topology backward step is
  self-contained (its own `zero_grad`/`step`) and runs unchanged.
- The asymmetric residual anchor lives inside the BCE loop, so it is dropped on
  topo-only epochs along with BCE. This yields a clean "does the topology
  gradient *alone* reduce topology loss?" probe.
- `train_bce_loss` (and residual-anchor terms) log `0.0` for the epoch;
  `train_topology_loss` logs its real value. `epoch_denominator = max(1, count)`
  already guards the zero-BCE division, so no metric divides by zero.

Default (`null`) → existing behavior unchanged.

### Reading the probe

In `training_summary.json`, after epoch 7 the trajectory of
`train_topology_loss` with BCE off is the diagnostic:

- **Decreasing** → BCE-vs-topology conflict confirmed; FP reweighting (deferred
  Task 2) is the right next lever.
- **Flat / increasing** → the topology gradient or objective itself is suspect;
  reweighting would not help and a deeper objective/implementation review is
  needed.

## Verification

Unit tests (TDD):
- Calibrated parsing: `type: calibrated` requires `grid` + valid `objective`;
  invalid objective and missing grid raise clear errors; `type: threshold` /
  absent preserves the existing fixed rule.
- Per-epoch calibration picks the grid `argmin val_topology_loss` and writes the
  chosen `selected_rule` (with `source: validation_calibration`) into the epoch
  history.
- Checkpoint stores the best epoch's calibrated `selected_rule`.
- Both `run_pairwise_test` (point metrics) and `run_topology_test` consume
  `refiner_state.selected_rule` in calibrated mode; threshold mode unchanged.
- `ignored_legacy_rules` appears in the manifest when `rules` coexists with a
  calibrated `refined_output_rule`.
- Topo-only: for `epoch >= N` the BCE loop is skipped, `train_bce_loss == 0`,
  the optimizer still steps on the topology loss, and the anchor terms are `0`.
  Default `null` runs the BCE loop on every epoch.

Commands (run inside the project venv):
```bash
source /Users/richardwang/Documents/grand/.venv/bin/activate
uv run python -m pytest
uv run ruff check tccig/s2gae.py tccig/train.py tccig/test.py tccig/prepare.py
uv run mypy src
```

## Expected outcome

A single rerun where the monitor and checkpoint reflect a calibrated, usable
operating point (Part A), and topology runs BCE-free from epoch 7 (Part B) to
reveal whether topology loss can drop at all. The probe result selects the next
experiment: FP reweighting if topology improves under topo-only, or an objective
review if it does not.
