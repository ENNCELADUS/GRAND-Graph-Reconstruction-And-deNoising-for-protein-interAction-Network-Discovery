# exp03 loss-conflict diagnostic design

Date: 2026-07-04
Branch: `exp02-topology-redesign`
Status: design

## Context

The exp02 rerun fixed the first-order reporting problem from the previous
`02_balanced_subset` run. Validation, checkpoint selection, and both test paths
now consume a calibrated refined-output threshold instead of the fixed `0.5`
threshold. The latest copied artifacts in `artifacts/exp02_rerun_latest` show:

| signal | latest exp02 rerun |
|---|---:|
| best epoch | 25 |
| best `val_topology_loss` | 8.1627 |
| best selected threshold | 0.96 |
| best validation selected edges | 12,376 |
| best validation relative density | 1.1537 |
| epoch 7 to 40 `train_topology_loss` | 9.9621 -> 7.2129 |
| refined pairwise precision at `0.96` | 0.9598 |
| refined pairwise recall at `0.96` | 0.1449 |
| topology-test relative density | 0.1609 |

The positive probe result is clear: with BCE disabled after epoch 7, the
topology objective can reduce the differentiable `train_topology_loss`. That
supports the exp02 design's decision rule: the next experiment should focus on
BCE-vs-topology conflict, not on replacing the topology objective immediately.

The remaining problem is also clear. The calibrated threshold chosen by
`val_topology_loss` is usable for validation selection, but it produces a highly
imbalanced pairwise operating point and does not transfer cleanly to topology
test. At threshold `0.96`, refined pairwise precision is high, recall collapses,
and topology test is too sparse. The next run should therefore diagnose whether
the training losses are fighting each other, and whether the topology component
weights are miscalibrated.

## Goal

Run exp03 as a diagnostic ablation before chasing final performance:

1. Determine whether BCE and topology losses conflict under calibrated
   monitoring.
2. Isolate which topology components help or hurt: graph similarity, relative
   density, and degree distribution.
3. Decide whether the current topology weights (`alpha=1, beta=8, gamma=0.5`)
   need adjustment.
4. Test whether a small reweight sweep can improve test behavior without merely
   overfitting validation topology loss.

## Scope

**In scope**

- A minimal Phase A loss-component ablation matrix.
- A gated Phase B reweight sweep only if Phase A shows a tractable conflict.
- Calibrated validation and checkpoint selection kept on for every run.
- Validation and test reporting that records threshold, selected-edge count,
  pairwise precision/recall, topology metrics, and deletion/addition diagnostics.
- One seed first. Additional seeds are deferred until a candidate configuration
  survives the diagnostic gates.

**Out of scope**

- Changing validation/test topology metric definitions.
- Changing the pairwise scorer or ESM embedding cache.
- Replacing the topology objective family.
- Treating exp03 as a final paper-ready benchmark suite.
- Optimizing a new threshold-selection objective before the loss conflict is
  diagnosed. If exp03 finds that all training-side fixes fail to transfer, the
  threshold-selection objective becomes a separate exp04 question.

## Experimental Principle

Keep the selection/reporting machinery fixed while changing only the training
pressure. Every run uses:

```yaml
refiner:
  monitor_metric: val_topology_loss
  topology_validation:
    enabled: true
graph_selection:
  refined_output_rule:
    type: calibrated
    objective: val_topology_loss
    grid: [0.5, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99]
```

This keeps the experiment interpretable: differences in validation/test behavior
come from training objective changes, not from changing the monitor.

## Phase A - Minimum Diagnostic Matrix

Phase A answers "which gradients conflict?" before tuning weights. Each run is
40 epochs unless an early artifact-level failure makes the run invalid.

| Run ID | Variant | BCE phase | Topology training | Weights | Purpose |
|---|---|---|---|---|---|
| exp03_a0 | exp02 topo-only reference | epochs 1-6 only | full topology after warmup | `alpha=1, beta=8, gamma=0.5` | Positive control from latest exp02 artifacts; no rerun required unless code changed. |
| exp03_a1 | BCE-only | on every epoch | disabled, with `topology_weight=0` as fallback | no topology backward | Measures whether BCE alone recreates the dense edge-adder dynamic under calibrated monitoring. |
| exp03_a2 | BCE + graph similarity | on every epoch | graph similarity only | `alpha=1, beta=0, gamma=0` | Tests whether graph similarity conflicts with or supports BCE. |
| exp03_a3 | BCE + density | on every epoch | density only | `alpha=0, beta=8, gamma=0` | Tests whether density alone can counter BCE's edge-adder pressure. |
| exp03_a4 | BCE + degree | on every epoch | degree only | `alpha=0, beta=0, gamma=0.5` | Tests whether degree MMD has independent corrective signal. |
| exp03_a5 | BCE + full topology | on every epoch | full topology | `alpha=1, beta=8, gamma=0.5` | Main conflict test: does full topology still lose against BCE when both are active? |

### Config notes

- For `exp03_a1`, prefer `topology_training.enabled: false` if the pipeline
  still performs calibrated topology validation. If the implementation requires
  the topology-training block to exist, use `topology_weight: 0.0` and
  `topo_only_after_epoch: null`. Either way, topology validation must stay on.
- For `exp03_a2` to `exp03_a5`, set `topo_only_after_epoch: null` so BCE and
  topology are both active after warmup.
- Keep the subset sampler budget, topology validation node sizes, optimizer,
  batch size, residual anchor, and calibrated grid unchanged from exp02.
- Phase A does not require a new FP/FN sampler. It should use the current
  hard-target composition so the first question is whether the existing
  FN-heavy BCE pressure conflicts with each topology component.

## Phase A Readout

For every run, collect a compact comparison table with:

| Metric group | Required fields |
|---|---|
| Selection | best epoch, selected threshold, selected positive edges |
| Validation topology | `val_topology_loss`, `relative_density`, `graph_sim`, `deg_dist_mmd`, `cc_mmd` |
| Training dynamics | `train_bce_loss`, `train_topology_loss` when present, `train_topo_graph_sim`, `train_topo_relative_density`, `train_topo_degree_mmd`, `sampled_edge_targets`, FN/FP counts |
| Pairwise test | AUPRC, AUROC, precision, recall, F1 at selected threshold |
| Topology test | relative density, graph similarity, degree MMD, CC MMD, edges added, edges deleted, deletion precision |

Interpretation rules:

- If `exp03_a1` increases validation/test density or selected edges while
  `exp03_a0` decreases topology loss, BCE-vs-topology conflict is confirmed.
- If `exp03_a3` controls density but destroys recall, density is useful but
  overweighted or too blunt.
- If `exp03_a2` improves pairwise recall while worsening density, graph
  similarity and density are pulling in different directions.
- If `exp03_a4` is weak alone but improves degree metrics without destabilizing
  density, degree should remain as a secondary regularizer.
- If `exp03_a5` resembles BCE-only, topology weight is too weak relative to BCE.
- If `exp03_a5` resembles topo-only and remains sparse on test, the full
  topology objective may overcorrect; Phase B should lower density weight or
  increase BCE pressure carefully.

## Phase B - Gated Small Reweight Sweep

Phase B only runs after Phase A. Its purpose is not a broad hyperparameter
search; it tests the smallest credible correction suggested by Phase A.

### Gate into Phase B

Run Phase B only if all of the following hold:

1. Phase A confirms BCE-vs-topology conflict or a clear component-level
   imbalance.
2. At least one Phase A variant improves a topology metric without making
   pairwise recall unusable.
3. The best Phase A candidate still leaves a measurable validation/test gap or
   precision/recall imbalance worth tuning.

### Candidate Phase B levers

Pick at most two levers based on Phase A:

| Lever | When to use | Candidate settings |
|---|---|---|
| Topology total weight | Full topology loses against BCE | `topology_weight in {0.5, 1.0, 2.0}` around the best Phase A component mix |
| Density weight | Density controls edges but oversparsifies test | `beta in {2, 4, 8}` with fixed `alpha=1`, `gamma=0.5` |
| Positive BCE weight | BCE over-adds positives | `pos_weight in {0.25, 0.5, 1.0}` as a config-only first pass |
| Hard-quadrant sampling weight | FN-heavy hard targets dominate BCE | Add a narrow config knob for FP/FN hard-target sampling weights, then test at most `{FN:FP = 1:1, 2:1}` |

The hard-quadrant sampling lever likely needs a small implementation plan,
because current `sample_epoch_edge_targets` includes all FP and all FN hard
targets and only samples the easy TP/TN anchors. Do not hide this behind
unrelated knobs. If Phase A points to FN pressure as the failure mode, add an
explicit sampler config rather than relying only on `pos_weight`.

### Phase B maximum size

Phase B should be capped at four runs before review. Example if Phase A says
density helps but overcorrects:

| Run ID | Variant |
|---|---|
| exp03_b1 | best Phase A mix with `beta=2` |
| exp03_b2 | best Phase A mix with `beta=4` |
| exp03_b3 | best Phase A mix with `topology_weight=0.5` |
| exp03_b4 | best Phase A mix with `pos_weight=0.5` or explicit FN:FP balance |

After these four runs, stop and analyze. Do not expand into a full sweep without
a new review.

## Success Criteria

exp03 succeeds as a diagnostic if it can answer these questions with artifacts:

1. Does BCE alone push the refiner toward the edge-adder failure mode?
2. Which topology component most directly opposes that pressure?
3. Do the default topology weights overemphasize density relative to pairwise
   recall and test transfer?
4. Is there at least one small reweight direction that improves test topology
   behavior without collapsing refined pairwise recall?

For a candidate configuration to move beyond diagnostics, it should satisfy all
of:

- Better test topology relative density than the exp02 topo-only reference
  (`0.1609`) without returning to a dense graph.
- Refined pairwise recall materially above the exp02 `0.1449` at the selected
  threshold, while precision remains meaningfully above the raw pairwise
  operating point.
- Validation selected threshold and selected-edge count remain stable enough
  that checkpoint selection is not just exploiting a one-epoch threshold quirk.
- Training dynamics support the interpretation: the chosen topology components
  improve for the reason claimed, not by hiding the conflict in the hard
  threshold.

## Failure Interpretation

- If every BCE-on variant recreates the edge-adder or selected-edge blowup,
  the next lever is explicit FN/FP sampling or per-quadrant BCE weighting.
- If density-only improves validation but makes test even sparser, the current
  `val_topology_loss` objective may not transfer; plan exp04 around selection
  objective or validation protocol, not more density tuning.
- If graph similarity alone improves pairwise recall but worsens density, use
  Phase B to add a lighter density term rather than restoring `beta=8`.
- If all topology-component variants fail to improve `train_topology_loss` when
  BCE is on, BCE dominates the topology gradient under the current optimizer
  scale and total topology weight must be raised or alternated.
- If Phase A is noisy or inconclusive, repeat only the most diagnostic two
  variants with a second seed before expanding the matrix.

## Required Artifact Checks

Each run must produce:

- `training_summary.json` with full `history` and best `selected_rule`.
- `manifest.json` with configured calibrated rule and effective refined-output
  rule.
- `pairwise_test/refined_metrics.json`, `pairwise_test/raw_metrics.json`.
- `topology_test/topology_metrics.json`.
- Slurm stdout/stderr paths.

The analysis script for exp03 should emit one table where rows are runs and
columns include at least:

- best epoch
- selected threshold
- validation selected edges
- validation relative density
- `train_topology_loss` at epoch 7 and epoch 40 when present
- refined precision/recall/F1
- topology-test relative density
- topology-test graph similarity
- topology-test degree MMD
- edges added/deleted

## Proposed Run Naming

Use `run_id` values under one exp03 family:

```text
03_a0_exp02_topo_only_reference
03_a1_bce_only
03_a2_bce_graph_sim
03_a3_bce_density
03_a4_bce_degree
03_a5_bce_full_topology
03_b1_beta2
03_b2_beta4
03_b3_topology_weight_0p5
03_b4_bce_pos_weight_0p5
03_b4_fnfp_1to1_if_sampler_knob_is_implemented
```

Use the same suffix in config filenames, log folders, and model checkpoint
folders. Do not overwrite `02_balanced_subset` artifacts.

## Verification Before Launch

Before submitting the Phase A jobs:

1. Confirm every config still parses with calibrated `refined_output_rule`.
2. Confirm `topology_validation.enabled: true` and
   `monitor_metric: val_topology_loss` for every config.
3. Confirm only the intended objective knobs differ across Phase A configs.
4. Run unit tests touched by any required config/parser changes.
5. For a new sampler-weight config, add tests that prove FP/FN ratios change
   deterministically and that the default path preserves current behavior.

## Expected Outcome

The expected outcome is not a final model. The expected outcome is a clear
conflict map:

- BCE-only shows whether the supervised residual objective is the source of
  density drift.
- Single-component topology runs identify whether graph similarity, density, or
  degree is the useful corrective pressure.
- Full-topology BCE-on shows whether the current default weights are strong and
  balanced enough.
- A small Phase B sweep tests whether the diagnosed conflict can be rebalanced
  without changing the monitor.

If Phase B finds a candidate that improves test relative density and pairwise
recall together, then exp04 should repeat that candidate across seeds and
compare against the exp02 topo-only reference. If it does not, exp04 should
target threshold-selection transfer or an explicit FN/FP sampler rather than
continuing blind topology-weight sweeps.
