# TCCIG Current Repo Issues

Date: 2026-05-31
Scope: `/Users/richardwang/Documents/grand`

This note records the current TCCIG diagnosis so future agents do not treat the
latest poor results as a single model-capacity problem. The main issue is that
training, normal evaluation, internal topology validation, and official
`topology_evaluate` currently evaluate different objects and use different
decision rules.

## Current Runtime Paths

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

## Observed Symptoms

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

## Main Root Causes

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
   reconstruction and makes calibration harder.

8. Some configured losses are placeholders.
   In `compute_tccig_losses()`, `rank`, `module`, and `spectral` are currently
   zero-valued placeholders. Do not treat configurations enabling those terms as
   real topology supervision until the losses are implemented or explicitly
   marked as unavailable.

## Priority Fixes

### P0: Unify Evaluation Semantics

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

- Fixed threshold `0.5` is diagnostic only for current TCCIG checkpoints.
- Add validation-calibrated threshold modes for pairwise classification, such as
  validation F1, MCC, or Youden index.
- Keep graph assembly budget separate from probability threshold. For topology
  evaluation, prefer top-`m_hat` or validation-density top-K.
- Avoid logging `Decision Threshold: 0.500` as though it controls TCCIG official
  topology evaluation when the actual path uses top-`m_hat`.

### P2: Fix Density Prior and Edge-Budget Observability

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

- Until a composite monitor exists, consider monitoring `val_auprc` for
  checkpointing instead of `val_topology_loss`.
- Use warmup/ramp for topology losses.
- Reduce teacher weight initially.
- Temporarily disable clustering if it is noisy or expensive.
- Consider a composite monitor that combines AUPRC, graph similarity, relative
  density error, degree MMD, and clustering MMD.

### P4: Fix Decoder Saturation

- Scale low-rank dot product by `sqrt(lowrank_dim)`.
- Add small learnable gates for `hub_score`, `lowrank_score`, and
  `module_score`, initialized around `0.1`.
- Use small initialization for structural heads if needed.
- Success criterion: probability quantiles should stop collapsing near `1.0`,
  specificity should become non-zero under a tuned threshold, and calibration
  metrics should improve.

### P5: Simplify Teacher Before Adding More Pretext Complexity

- Run an ablation with teacher loss disabled.
- Then try a pretrained frozen teacher.
- Only after evaluation semantics, density prior, edge-budget diagnostics, and
  decoder calibration are corrected should S2GAE/MaskGAE/Bandana-style teacher
  variants be implemented.

## Recommended Experiment Queue

1. Run A: evaluation-only fixes.
   Goal: determine whether the main failure is ranking, edge budget, or assembly
   semantics without changing training.

2. Run B: teacher disabled plus graph-density prior.
   Goal: reduce probability saturation and make `m_hat` closer to target edge
   counts without losing more than about 2-3 AUPRC points.

3. Run C: warmup/ramp plus slightly higher learning rate.
   Goal: make optimization smoother and prevent topology losses from dominating
   too early.

4. Run D: decoder scale/gate.
   Goal: improve calibration and reduce sigmoid saturation.

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
