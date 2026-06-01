# TCCIG Current Repo Issues

Date: 2026-05-31
Scope: `/Users/richardwang/Documents/grand`

This note records the TCCIG diagnosis around the `tccig_scratch` run and the
P0/P1/P2 fixed follow-ups through 2026-06-01. Future agents should not treat the
poor results as a single model-capacity problem. The first failure mode was that
training, normal evaluation, internal topology validation, and official
`topology_evaluate` evaluated different objects and used different decision
rules. After P0/P1 fixed evaluation/reporting semantics, P2 showed that
graph-density prior initialization and teacher disabling do not fix score
saturation, weak ranking, or graph reconstruction.

## Baseline Runtime Paths Before P0

- Training uses `forward_graph()` on sampled 60-80 node subgraphs and performs
  all-pairs graph reconstruction.
- Normal `evaluate` uses pairwise `forward()` and applies a fixed threshold of
  `0.5`.
- Internal topology validation runs `forward_graph()` on each validation sampled
  subgraph independently, then applies per-subgraph top-`m_hat`.
- Official `topology_evaluate` scores the whole test candidate universe once,
  predicts one global edge budget, and applies global top-`m_hat`.

Because of this, normal `evaluate` and `topology_evaluate` are not expected to
match. More importantly, internal topology validation is also not isomorphic to
official `topology_evaluate`, so checkpoint selection is unreliable.

## P0 Fixed Result Update

The P0 fixed eval-only run is archived in commit `baf4087` with
`configs/tccig/p0_fixed_eval_only.yaml` and results under
`logs/tccig/{evaluate,topology_evaluate,tccig_train}/p0_fixed/`. It evaluates
`models/tccig/tccig_train/p0_fixed/best_model.pth` using graph assembly
semantics for both `evaluate` and `topology_evaluate`.

Pairwise-style CSV comparison:

| run | AUROC | AUPRC | accuracy | specificity | precision | recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tccig_scratch` | 0.625 | 0.644 | 0.500 | 0.000 | 0.500 | 1.000 | 0.667 | 0.000 |
| `p0_fixed` | 0.581 | 0.082 | 0.948 | 0.958 | 0.055 | 0.180 | 0.084 | 0.078 |

Topology summary comparison:

| run | graph_sim | relative_density | deg_dist_mmd | cc_mmd | laplacian_eigen_mmd |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tccig_scratch` | 0.159 | 0.402 | 37.962 | 15.404 | 34.264 |
| `p0_fixed` | 0.191 | 0.572 | 26.239 | 9.804 | 21.923 |

Graph assembly diagnostics for the successful P0 fixed rerun:

- `candidate_count = 2033136`
- `record_count = 2035153`
- `m_hat = 88478.148`
- `selected_edges = 88478`
- probability quantiles: min `0.000`, mean `0.981`, p50 `0.984`, p90 `0.990`,
  p95 `0.991`, max `1.000`

Interpretation: P0 fixed confirms that the scratch `evaluate.csv` hard metrics
were dominated by the fixed-threshold all-positive decision rule, not by a
usable graph reconstruction. The unified graph-assembly path improves official
topology metrics and makes specificity nonzero, but graph similarity remains
low and precision/recall are still poor. The original inference therefore holds
in a narrower form: evaluation semantics were a real blocker, but the model is
still limited by saturated probabilities, weak ranking over the full candidate
universe, and density-prior / edge-budget calibration.

## P1 Fixed Result Update

The P1 fixed-threshold cleanup run is archived in commits `e9cefbc` and
`51e462a` with results under
`logs/tccig/{evaluate,topology_evaluate,tccig_train}/p1_fixed/`. The original
HPC job reused run id `p0_fixed`, so the covered logs were first copied into
`p1_fixed` and then the tracked `p0_fixed` logs were restored from Git. The
summary CSV files in `p1_fixed` contain only the 2026-06-01 run segment; the raw
`log.log` files still preserve the full collision history for auditability.

Pairwise-style graph-assembly CSV result:

| run | AUROC | AUPRC | accuracy | specificity | precision | recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `p0_fixed` | 0.581 | 0.082 | 0.948 | 0.958 | 0.055 | 0.180 | 0.084 | 0.078 |
| `p1_fixed` | 0.581 | 0.082 | 0.948 | 0.958 | 0.055 | 0.180 | 0.084 | 0.078 |

Topology summary comparison:

| run | graph_sim | relative_density | deg_dist_mmd | cc_mmd | laplacian_eigen_mmd |
| --- | ---: | ---: | ---: | ---: | ---: |
| `p0_fixed` | 0.191 | 0.572 | 26.239 | 9.804 | 21.923 |
| `p1_fixed` | 0.191 | 0.572 | 26.193 | 9.779 | 21.940 |

P1 graph assembly remains top-`m_hat`: `m_hat = 88478.148`, `selected_edges =
88478`, `candidate_count = 2033136`, and probability quantiles remain saturated
with mean `0.981`, p50 `0.984`, p95 `0.991`, and max `1.000`. The new diagnostic
field records `threshold_mode = validation_mcc`, `threshold_value = 0.993`,
validation MCC `0.271`, validation F1 `0.574`, Youden `0.261`, predicted-positive
rate `0.367`, and validation positive rate `0.500`.

Interpretation: P1 fixed the reporting semantics, not the graph generator. It
confirms that fixed threshold `0.5` should be treated as diagnostic only and
that official TCCIG topology evaluation is governed by `decision_rule =
top_m_hat`. The unchanged graph similarity, density, and saturated probability
quantiles keep P2/P3/P4 as the next meaningful model-quality fixes.

## P2 Fixed Result Update

The P2 density-prior / edge-budget observability run is archived in commit
`49c78d4` with results under
`logs/tccig/{evaluate,topology_evaluate,tccig_train}/p2_fixed/`. The run used
canonical `configs/tccig/01.yaml` after setting all run ids to `p2_fixed`,
initializing the TCCIG density bias from train-graph density, and disabling the
online teacher for the Run B ablation. Slurm job `920857` completed on
2026-06-01 without `Traceback`, CUDA OOM, or `Loss: nan`.

Training diagnostics confirm that the P2 code path was active:
`Density Bias Initialized` reports source `graph_density`, positive edge
probability `0.001`, and bias `-6.502`. Training early-stopped at epoch 12.
The checkpoint selected by `val_topology_loss` was epoch 8, while the best
validation AUPRC was epoch 6.

Pairwise-style graph-assembly CSV result:

| run | AUROC | AUPRC | accuracy | specificity | precision | recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `p1_fixed` | 0.581 | 0.082 | 0.948 | 0.958 | 0.055 | 0.180 | 0.084 | 0.078 |
| `p2_fixed` | 0.549 | 0.049 | 0.964 | 0.975 | 0.069 | 0.136 | 0.091 | 0.079 |

Topology summary comparison:

| run | graph_sim | relative_density | deg_dist_mmd | cc_mmd | laplacian_eigen_mmd |
| --- | ---: | ---: | ---: | ---: | ---: |
| `p1_fixed` | 0.191 | 0.572 | 26.193 | 9.779 | 21.940 |
| `p2_fixed` | 0.160 | 0.364 | 38.883 | 17.277 | 35.020 |

P2 graph assembly diagnostics:

- `n_nodes = 2017`
- `full_pair_count = 2033136`
- `candidate_count = 2033136`
- `record_count = 2035153`
- `m_hat = 53701.148`
- `m_hat / candidate_count = 0.026`
- `m_hat / full_pair_count = 0.026`
- `selected_edges = 53701`
- probability quantiles: min `0.000`, mean `0.999`, p50 `1.000`, p90 `1.000`,
  p95 `1.000`, max `1.000`

P2 debug assemblies:

| assembly | budget | selected_edges | graph_sim | relative_density | deg_dist_mmd | cc_mmd | laplacian_eigen_mmd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| model `m_hat` | 53701.148 | 53701 | 0.160 | 0.364 | 38.883 | 17.277 | 35.020 |
| validation density | 761.712 | 762 | 0.040 | 0.024 | 78.078 | 37.639 | 78.231 |
| oracle test density | 27063.000 | 27063 | 0.135 | 0.225 | 50.413 | 22.002 | 47.007 |

Interpretation: P2 validates the observability fix but does not improve model
quality. Graph-density initialization and teacher disabling reduced the learned
edge budget (`88478 -> 53701`) but made score saturation worse
(`mean = 0.999`, median/p90/p95 all `1.000`) and degraded topology metrics.
The oracle-density debug assembly is especially important: even with a
test-density budget, graph similarity falls to `0.135`, so the failure is not
only "how many edges to select." The full-candidate ranking/localization and
decoder calibration are poor. Treat P2 as evidence that density-prior correction
alone is not the main model-quality fix.

## Baseline Observed Symptoms

- Normal evaluation reports approximately:
  - accuracy `0.500`
  - AUPRC `0.644`
  - AUROC `0.625`
  - F1 `0.667`
  - precision `0.500`
  - recall `1.000`
  - specificity `0.000`
  - MCC `0.000`
- The fixed `0.5` threshold is effectively unusable for this checkpoint:
  probabilities are saturated, with validation probability mean around `0.967`,
  median around `0.971`, P95 around `0.988`, and threshold-positive fraction
  equal to `1.000`.
- Official `topology_evaluate` is graph-level poor:
  - `graph_sim = 0.159`
  - `relative_density = 0.402`
  - `deg_dist_mmd = 37.962`
  - `cc_mmd = 15.404`
  - `laplacian_eigen_mmd = 34.264`
- Training loss decreases and pairwise ranking improves, but topology metrics do
  not improve in a stable way. The model is learning some ranking signal, but it
  is not producing realistic graph reconstructions.

## Root Causes From Baseline Diagnosis

1. Evaluation semantics differ.
   Normal `evaluate` uses pairwise `forward()` plus fixed threshold `0.5`, while
   `topology_evaluate` uses graph `forward_graph()` plus global top-`m_hat`.

2. Internal validation differs from official topology evaluation.
   Internal validation uses per-subgraph top-`m_hat`; official evaluation uses
   global top-`m_hat` over the candidate universe. This can overfill each sampled
   subgraph internally while the official assembled graph remains under-dense in
   sampled regions.

3. Internal diagnostics mix assembly rules.
   Graph metrics use top-`m_hat` predictions, but some diagnostics and edge-count
   ratios use fixed-threshold predictions. The CSV can therefore mix two
   different graph assembly interpretations.

4. Checkpoint monitoring does not optimize graph similarity.
   Current `monitor_metric` is `val_topology_loss`, but TCCIG validation has
   graph-similarity weight effectively excluded when `alpha = 0.0`. The best
   topology checkpoint, best AUPRC checkpoint, and best internal graph-sim epoch
   can therefore diverge.

5. The current teacher is an online train-only MGAE teacher, not a stable
   S2GAE/MaskGAE/Bandana-style teacher.
   It is updated inside the student step and then immediately used for
   distillation, which makes the teacher target noisy and non-stationary.

6. Decoder logits saturate.
   `pair_score`, `hub_score`, `lowrank_score`, `module_score`, and
   `density_bias` are added directly. The low-rank dot product is not scaled by
   `sqrt(dim)`, and structural branches do not have small learnable gates. This
   likely contributes to sigmoid saturation.

7. Density prior uses supervised BCE positive rate instead of graph density.
   The initialized positive probability is about `0.239`, while the validation
   target edge fraction is closer to `0.038`. This is too dense for graph
   reconstruction and makes calibration harder. P2 fixed this initialization
   bug, but the P2 result shows the remaining failure is not explained by the
   prior alone.

8. Some configured losses are placeholders.
   In `compute_tccig_losses()`, `rank`, `module`, and `spectral` are currently
   zero-valued placeholders. Do not treat configurations enabling those terms as
   real topology supervision until the losses are implemented or explicitly
   marked as unavailable.

## Priority Fixes

### P0: Unify Evaluation Semantics

Status after `p0_fixed`: implemented for the canonical TCCIG eval-only path and
used to rerun the checkpoint. Keep these requirements as invariants for future
configs and result interpretation.

- Add graph-mode evaluation for TCCIG so normal evaluation can score candidate
  records using `forward_graph()` probabilities.
- Report ranking metrics from graph-mode probabilities and hard metrics from the
  same graph assembly rule.
- Make internal validation mimic official `topology_evaluate`: score all
  validation candidate records globally, assemble a single global top-`m_hat`
  validation graph, then evaluate sampled validation subgraphs from that graph.
- Split diagnostics into explicit fixed-threshold and top-`m_hat` fields:
  - `top_m_pred_edges_total`
  - `top_m_pred_edge_fraction`
  - `top_m_pred_target_edge_ratio`
  - `fixed_threshold_pred_edges_total`
  - `fixed_threshold_pred_edge_fraction`
  - `fixed_threshold_pred_target_edge_ratio`

### P1: Stop Treating Fixed 0.5 as the Main Decision Rule

Status after `p1_fixed`: implemented for TCCIG reporting and pairwise
diagnostics. Keep top-`m_hat` as the graph assembly rule; use
validation-calibrated thresholds only for pairwise diagnostic classification.
Do not interpret the unchanged graph metrics as a failed model-quality fix,
because P1 did not modify training, density priors, edge budgets, or decoder
calibration.

- Fixed threshold `0.5` is diagnostic only for current TCCIG checkpoints.
- Add validation-calibrated threshold modes for pairwise classification, such as
  validation F1, MCC, or Youden index.
- Keep graph assembly budget separate from probability threshold. For topology
  evaluation, prefer top-`m_hat` or validation-density top-K.
- Avoid logging `Decision Threshold: 0.500` as though it controls TCCIG official
  topology evaluation when the actual path uses top-`m_hat`.

### P2: Fix Density Prior and Edge-Budget Observability

Status after `p2_fixed`: implemented and archived. Keep the diagnostics as part
of future TCCIG runs, but do not expect graph-density initialization alone to
fix score saturation or graph reconstruction.

- Initialize graph density from train graph density:
  `abs(E_train) / comb(abs(V_train), 2)`, not supervised BCE positive rate.
- Log edge-budget diagnostics in topology evaluation:
  - `n_nodes`
  - `full_pair_count`
  - `candidate_count`
  - `m_hat`
  - `m_hat / candidate_count`
  - `m_hat / full_pair_count`
  - selected edge count
  - probability min/mean/p50/p90/p95/max
- Add debug assemblies:
  - model `m_hat` top-K
  - validation-density top-K
  - oracle-test-density top-K, diagnostic only

### P3: Adjust Loss Schedule and Checkpoint Monitor

- P2 reinforces this priority: the epoch selected by `val_topology_loss` did
  not translate into better official topology metrics, and the best validation
  AUPRC occurred at a different epoch.
- Until a composite monitor exists, consider monitoring `val_auprc` for
  checkpointing instead of `val_topology_loss`.
- Use warmup/ramp for topology losses.
- Reduce teacher weight initially.
- Temporarily disable clustering if it is noisy or expensive.
- Consider a composite monitor that combines AUPRC, graph similarity, relative
  density error, degree MMD, and clustering MMD.

### P4: Fix Decoder Saturation

- P2 makes this the next highest-leverage model-quality fix. Probability
  quantiles collapsed harder after graph-density prior initialization, so
  budget changes alone are insufficient.
- Scale low-rank dot product by `sqrt(lowrank_dim)`.
- Add small learnable gates for `hub_score`, `lowrank_score`, and
  `module_score`, initialized around `0.1`.
- Use small initialization for structural heads if needed.
- Success criterion: probability quantiles should stop collapsing near `1.0`,
  specificity should become non-zero under a tuned threshold, and calibration
  metrics should improve.

### P5: Simplify Teacher Before Adding More Pretext Complexity

- The P2 teacher-disabled ablation did not improve official graph metrics; do
  not treat the online teacher as the primary cause of the current failure.
- Then try a pretrained frozen teacher.
- Only after evaluation semantics, density prior, edge-budget diagnostics, and
  decoder calibration are corrected should S2GAE/MaskGAE/Bandana-style teacher
  variants be implemented.

## Recommended Experiment Queue

1. Run A: evaluation-only fixes.
   Goal: determine whether the main failure is ranking, edge budget, or assembly
   semantics without changing training.

2. Run B: teacher disabled plus graph-density prior.
   Status: completed as `p2_fixed`. It reduced `m_hat` but worsened saturation,
   AUPRC, and topology metrics.

3. Run D: decoder scale/gate.
   Goal: improve calibration and reduce sigmoid saturation. P2 indicates this
   should move ahead of further density-prior tuning.

4. Run C: warmup/ramp plus slightly higher learning rate.
   Goal: make optimization smoother and prevent topology losses from dominating
   too early, after decoder saturation is addressed or at least instrumented.

5. Run E: frozen teacher distillation.
   Goal: test whether a stable teacher improves graph similarity or degree MMD
   without significantly harming AUPRC.

## Minimal Execution Order

1. Make graph-mode validation and official topology evaluation isomorphic.
2. Separate threshold calibration from graph assembly budget.
3. Initialize density from graph density.
4. Persist probability and `m_hat` diagnostics.
5. Reduce or freeze teacher influence.
6. Add decoder scaling/gating.
7. Only then add richer teacher pretext tasks.

Short explanation of the current result: the model is not completely failing,
because pairwise ranking improves. The main failure is graph reconstruction and
calibration: fixed threshold predicts almost everything positive, official graph
assembly is under-dense in sampled regions, and internal validation does not
faithfully predict official `topology_evaluate`.
