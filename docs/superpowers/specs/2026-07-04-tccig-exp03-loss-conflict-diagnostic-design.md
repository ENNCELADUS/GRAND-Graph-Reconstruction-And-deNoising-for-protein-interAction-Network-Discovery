# exp03 loss-conflict diagnostic design

Date: 2026-07-04
Branch: `exp02-topology-redesign`
Status: design

## Context

The exp02 rerun fixed the first-order reporting problem from the previous
`02_balanced_subset` run. Validation, checkpoint selection, and both test paths
now consume a calibrated refined-output threshold instead of the fixed `0.5`
threshold. The copied exp02 rerun-fix artifacts in
`artifacts/exp02_rerun_fix` show:

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
4. Test whether a small validation-selected reweight sweep can produce a
   candidate worth one final held-out test evaluation, without using test
   metrics to choose that candidate.

## Scope

**In scope**

- A minimal Phase A loss-component ablation matrix.
- A gated Phase B reweight sweep only if Phase A shows a tractable conflict.
- Calibrated validation and checkpoint selection kept on for every run.
- Validation reporting that records selected threshold, selected-edge count,
  `val_auprc`, topology metrics, and calibration stability.
- A held-out test report only after the Phase B candidate is locked by
  training/validation evidence.
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
- Updating `CONTEXT.md`. Its refined-output-threshold paragraph still describes
  the pre-exp02 fixed `p_refined >= 0.5` behavior; exp03 treats the calibrated
  exp02 behavior and this spec as the current experiment contract.

## Evaluation Hygiene

Phase A and Phase B selection decisions must use only training dynamics and
validation metrics. Pairwise-test and topology-test metrics are held-out
diagnostics. They may be generated for the final locked candidate and for the
exp02 reference comparison, but they must not decide which Phase B levers are
run or which candidate advances.

If test metrics are inspected during exploratory development, label those
results explicitly as development diagnostics and do not present them later as a
clean held-out comparison.

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

This keeps the experiment interpretable: differences in validation behavior and
final held-out behavior come from training objective changes, not from changing
the monitor.

When this spec says "weights" in Phase A or Phase B, it means
`refiner.topology_training.weights` unless explicitly stated otherwise. Keep
`refiner.topology_validation.losses` fixed for every run:

```yaml
refiner:
  topology_validation:
    losses:
      alpha: 1.0
      beta: 8.0
      gamma: 0.5
      delta: 0.0
```

This preserves a common `val_topology_loss` monitor while the training objective
changes.

## Phase A - Minimum Diagnostic Matrix

Phase A is a behavioral loss-interaction diagnostic before tuning weights. Each
run is 40 epochs unless an early artifact-level failure makes the run invalid.

| Run ID | Variant | BCE phase | Topology training | Weights | Purpose |
|---|---|---|---|---|---|
| exp03_a0 | exp02 topo-only reference | epochs 1-6 only | full topology after warmup | `alpha=1, beta=8, gamma=0.5` | Positive control from latest exp02 artifacts; no rerun required unless code changed. |
| exp03_a1 | BCE-only | on every epoch | disabled | no topology backward | Measures whether BCE alone recreates the dense edge-adder dynamic under calibrated monitoring. |
| exp03_a2 | BCE + graph similarity | on every epoch | graph similarity only | `alpha=1, beta=0, gamma=0` | Tests whether graph similarity conflicts with or supports BCE. |
| exp03_a3 | BCE + density | on every epoch | density only | `alpha=0, beta=8, gamma=0` | Tests whether density alone can counter BCE's edge-adder pressure. |
| exp03_a4 | BCE + degree | on every epoch | degree only | `alpha=0, beta=0, gamma=0.5` | Tests whether degree MMD has independent corrective signal. |
| exp03_a5 | BCE + full topology | on every epoch | full topology | `alpha=1, beta=8, gamma=0.5` | Main conflict test: does full topology still lose against BCE when both are active? |

### Config notes

- For `exp03_a1`, set `refiner.topology_training.enabled: false`. Keep
  `refiner.topology_validation.enabled: true`, `monitor_metric:
  val_topology_loss`, and calibrated `refined_output_rule` unchanged.
- For `exp03_a2` to `exp03_a5`, set `topo_only_after_epoch: null` so BCE and
  topology are both active after warmup.
- Keep the subset sampler budget, topology validation node sizes, optimizer,
  batch size, residual anchor, calibrated grid, and
  `refiner.topology_validation.losses` unchanged from exp02.
- Phase A does not require a new FP/FN sampler. It should use the current
  hard-target composition so the first question is whether the existing
  FN-heavy BCE pressure conflicts with each topology component.

## Phase A Readout

For every Phase A/B run, collect a compact training/validation comparison table
with:

| Metric group | Required fields |
|---|---|
| Selection | best epoch, selected threshold, selected positive edges |
| Validation topology | `val_topology_loss`, `internal_val_relative_density`, `internal_val_graph_sim`, `internal_val_deg_dist_mmd`, `internal_val_cc_mmd` |
| Training dynamics | `train_bce_loss`, `train_topology_loss` when present, `train_topo_graph_sim`, `train_topo_relative_density`, `train_topo_degree_mmd`, `sampled_edge_targets`, FN/FP counts |
| Validation calibration stability | per-epoch selected threshold, selected edges, and best-epoch threshold-grid surface |

Use this source map:

| Source | Fields |
|---|---|
| `training_summary.json.history[*]` | `monitor_value`, `selected_rule`, `selected_rule_positive_edges`, `internal_val_*`, train losses, `sampled_edge_targets`, FN/FP/TP/TN counts |
| best-epoch threshold-grid artifact | `val_topology_loss`, selected edges, relative density, graph similarity, and MMDs for every grid threshold |
| final `pairwise_test/*.json` | held-out threshold-free AUPRC/AUROC and final point metrics only after the candidate is locked |
| final `topology_test/topology_metrics.json` | held-out topology summary, `deletion_diagnostics`, `pairwise_input_rule`, `protocol` only after the candidate is locked |

Interpretation rules:

- If `exp03_a1` increases validation density or selected edges while
  `exp03_a0` decreases topology loss, BCE and topology exert opposite behavioral
  pressure.
- If `exp03_a3` controls density but sharply lowers `val_auprc` or drives the
  selected threshold/edge count into an extreme sparse regime, density is useful
  but overweighted or too blunt.
- If `exp03_a2` maintains `val_auprc` while worsening density, graph similarity
  and density are pulling in different directions.
- If `exp03_a4` is weak alone but improves degree metrics without destabilizing
  density, degree should remain as a secondary regularizer.
- If `exp03_a5` resembles BCE-only, topology weight is too weak relative to BCE.
- Confirm BCE-vs-topology conflict only with direct `exp03_a5` evidence:
  compare post-warmup `train_topology_loss` slope, validation selected-edge
  trend, selected threshold, and `val_auprc` against both `exp03_a1` and
  `exp03_a0`.
- If `exp03_a5` resembles topo-only and remains sparse on validation, the full
  topology objective may overcorrect; Phase B should lower density weight or
  increase BCE pressure carefully.

## Phase B - Gated Small Reweight Sweep

Phase B only runs after Phase A. Its purpose is not a broad hyperparameter
search; it tests the smallest credible correction suggested by Phase A.

### Gate into Phase B

Run Phase B only if all of the following hold:

1. Phase A confirms BCE-vs-topology conflict or a clear component-level
   imbalance.
2. At least one Phase A variant improves a topology metric without collapsing
   `val_auprc` or destabilizing selected-edge behavior.
3. The best Phase A candidate still leaves a measurable validation topology gap,
   unstable selected-edge behavior, or validation AUPRC tradeoff worth tuning.

### Candidate Phase B levers

Pick at most two levers based on Phase A:

| Lever | When to use | Candidate settings |
|---|---|---|
| Topology total weight | Full topology loses against BCE | `topology_weight in {0.5, 2.0}` around the Phase A reference; do not rerun `1.0` |
| Density weight | Density controls edges but oversparsifies validation | `beta in {2, 4}` with fixed `alpha=1`, `gamma=0.5`; compare to the Phase A `beta=8` reference |
| Positive BCE weight | BCE over-adds positives | `pos_weight in {0.25, 0.5}` as a config-only first pass |
| Hard-quadrant sampling weight | FN-heavy hard targets dominate BCE | Add a narrow config knob for FP/FN hard-target sampling weights, then test at most `{FN:FP = 1:1, 2:1}` |

The hard-quadrant sampling lever likely needs a small implementation plan,
because current `sample_epoch_edge_targets` includes all FP and all FN hard
targets and only samples the easy TP/TN anchors. Do not hide this behind
unrelated knobs. If Phase A points to FN pressure as the failure mode, add an
explicit sampler config rather than relying only on `pos_weight`.

`pos_weight` is a global positive-label BCE weight. It does not change the
FN/FP hard-target composition. Use it before an explicit sampler only as a
cheap label-pressure probe, not as a replacement for quadrant-balanced sampling.

If implementing explicit hard-quadrant sampling, prefer deterministic
downsampling of the larger hard quadrant to the configured ratio. Preserve the
current default exactly: all FP + all FN hard targets plus sampled TP/TN easy
anchors.

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
   validation AUPRC and selected-edge stability?
4. Is there at least one small reweight direction that improves validation
   topology behavior without collapsing validation ranking quality?

For a candidate configuration to move to final held-out reporting, it should
satisfy all of these training/validation gates:

- Validation relative density closer to 1.0 than the exp02 topo-only reference
  best epoch (`1.1537`) without returning to the dense fixed-threshold failure.
- Non-worse validation graph similarity, degree MMD, and CC MMD relative to the
  exp02 topo-only reference, or a clear component-specific explanation for the
  tradeoff.
- `val_auprc` remains within an acceptable diagnostic tolerance of the exp02
  reference while topology improves; the implementation plan should set the
  numeric tolerance before launch.
- Validation selected threshold and selected-edge count remain stable enough
  that checkpoint selection is not just exploiting a one-epoch threshold quirk.
- Training dynamics support the interpretation: the chosen topology components
  improve for the reason claimed, not by hiding the conflict in the hard
  threshold.

After the candidate is locked by those gates, run held-out pairwise/topology
test once and report:

- Test relative density, graph similarity, degree MMD, CC MMD, and
  deletion/addition diagnostics versus exp02 and a raw pairwise-generated graph
  baseline.
- Refined pairwise AUPRC/AUROC versus raw AUPRC/AUROC.
- Point precision/recall only under matched operating points: selected
  threshold, matched recall, or matched selected-edge count. Do not compare raw
  `0.5` precision directly against refined calibrated-threshold precision as a
  model-quality claim.

## Failure Interpretation

- If every BCE-on variant recreates the edge-adder or selected-edge blowup,
  the next lever is explicit FN/FP sampling or per-quadrant BCE weighting.
- If density-only improves validation density but worsens validation
  graph-similarity/MMD tradeoffs, the current
  `val_topology_loss` objective may not transfer; plan exp04 around selection
  objective or validation protocol, not more density tuning.
- If graph similarity alone preserves ranking quality but worsens density, use
  Phase B to add a lighter density term rather than restoring `beta=8`.
- If all topology-component variants fail to improve `train_topology_loss` when
  BCE is on, BCE-on training overwhelms topology-loss improvement under the
  current optimizer scale and total topology weight must be raised or alternated.
- If Phase A is noisy or inconclusive, repeat only the most diagnostic two
  variants with a second seed before expanding the matrix.

## Required Artifact Checks

Each Phase A/B run must produce:

- `training_summary.json` with full `history` and best `selected_rule`.
- `manifest.json` with configured calibrated rule and effective refined-output
  rule.
- A threshold-grid artifact for at least the best epoch, and preferably each
  epoch, with `val_topology_loss`, selected edges, relative density, graph
  similarity, and MMDs for every grid threshold.
- Slurm stdout/stderr paths.

The final locked candidate and exp02 reference comparison must additionally
produce:

- `pairwise_test/refined_metrics.json`, `pairwise_test/raw_metrics.json`, and
  threshold-free raw/refined AUPRC/AUROC comparison.
- `topology_test/topology_metrics.json`.
- A raw pairwise-generated topology baseline in a separate output directory so
  it cannot overwrite refined `topology_test` artifacts.
- `topology_metrics.json.protocol.test_labels_visible_to_model: false`,
  `candidate_universe: all_test_ppi.txt`, and evidence that
  `human_test_graph.pkl` is used only for metrics.

The analysis script for exp03 should emit one table where rows are runs and
columns include at least:

- best epoch
- selected threshold
- validation selected edges
- validation relative density
- `train_topology_loss` at epoch 7 and epoch 40 when present
- validation topology metrics
- final-heldout refined precision/recall/F1 only after candidate lock
- final-heldout topology-test relative density
- final-heldout topology-test graph similarity
- final-heldout topology-test degree MMD
- final-heldout edges added/deleted

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
03_b5_fnfp_1to1_if_sampler_knob_is_implemented
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

If Phase B finds a validation-selected candidate whose final held-out report
also improves topology metrics and pairwise ranking tradeoffs, then exp04 should
repeat that candidate across seeds and compare against the exp02 topo-only
reference. If the validation-selected candidate fails only at held-out test
transfer, exp04 should target threshold-selection transfer or validation
protocol rather than continuing blind topology-weight sweeps.
