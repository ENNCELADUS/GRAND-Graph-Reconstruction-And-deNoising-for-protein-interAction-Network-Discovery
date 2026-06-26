# Experiment Plan

**Problem**: Strong pairwise PPI scorers (frozen v3.1 `pair_context_gated_abba_no_cross`) classify individual pairs well but assemble into over-dense, topologically incoherent interactomes on the PRING graph-reconstruction benchmark.
**Method Thesis**: A lightweight S2GAE-style residual denoiser, trained only on the frozen scorer's error edges over a precision-thresholded noisy graph `G_pairwise`, can correct topology (degree/clustering/spectral structure) without retraining the scorer and without sacrificing pairwise quality.
**Date**: 2026-06-26 (rev 3)

This is the TCCIG experiment plan (`configs/tccig/01.yaml`, `configs/tccig/02.yaml`, `scripts/tccig.sh`). Its job is to establish that the refiner path runs end-to-end, beats the frozen-scorer baseline on PRING topology metrics at matched/controlled pairwise quality, and that the residual-denoiser design choices are justified. The claim map below frames run `01` (BCE-only residual denoiser, topology used only for checkpoint selection) and run `02` (topology-conditioned **training** loss + asymmetric anchor, so the refiner can actively prune) as a staged story — `02` extends `01` rather than re-deriving it. **rev 3** adds the Run 02 mechanism, claim C3, blocks B4–B6, and milestones G1/M5/M6; Run 01 (C1/C2, B1–B3, M0–M4) is unchanged below.

## Run 01 Attempt 1 — FAILED (slurm 928779, 2026-06-25/26): diagnosed and fixed

The first full training launch ran to 40 epochs without crashing but produced a **degenerate refined output**, so its metrics are void. Recorded here so the re-run is not mistaken for a fresh start.

- **Symptom**: exported refined graph collapsed — `human_test_ppi_pred.csv` had 85.1% of rows exactly `0.0`; refined test AUPRC `0.490` (≈ random) vs the raw frozen scorer's `0.792` on the same pairs. Raw-vs-refined Spearman `-0.045` (the refiner destroyed the scorer's ranking instead of refining it).
- **Root cause (numerical instability, not DDP/IO/selection)**: the GraphConv encoder used unnormalized **sum** aggregation, so on the dense `G_pairwise` (~1.1M positive edges at epoch 1) node activations scaled with degree; the cross-layer **product** decoder then squared that magnitude; `residual_weight=1e-3` was far too weak to anchor the residual. Epoch-1 `train_residual_anchor_loss=7.3e14`, gradient norm `4.7e14`. The residual swamped the pairwise logit → sigmoid saturated to 0/1.
- **Aggravating factor**: `monitor_metric=val_topology_loss` selected epoch 13 (`val_auprc=0.120`, near the run's worst) because that epoch's relative-density proxy looked best while pairwise ranking was already broken. This is a selection weakness but **not** the primary cause — every epoch's refiner was net-negative vs raw.

**Fix applied (in code, verified by tests):**
- Encoder aggregation switched to **mean** (`refiner.encoder_aggr: mean`), degree-invariant.
- **LayerNorm** on input features and between conv layers (`refiner.layer_norm: true`).
- Residual **bounded** via `residual_scale * tanh(delta / residual_scale)` with `refiner.residual_scale: 4.0` — unit slope near zero (identity-like for small residuals), saturating at ±4 logits so the refiner can never swamp the raw scorer.
- New diagnostics so a future failure is one-glance debuggable: `run_pairwise_test` now writes `raw_metrics.json` + `refined_metrics.json` and a `raw_probability` column alongside `refined_probability`. Raw metrics use the scorer's natural `0.5` decision threshold.
- `monitor_metric` **kept** at `val_topology_loss` (the C1 topology claim depends on it; see Risks).

Tests: bounded-residual anchor stays `≤ scale²`; mean+LayerNorm keeps hidden states O(1) on a dense hub graph; raw/refined export verified. Stabilization defaults are now baked into `configs/tccig/01.yaml`.

## Run 02 — Topology-conditioned training loss (the method advance)

Run 01's refiner only ever sees topology through the **checkpoint monitor** (`val_topology_loss`); its *gradient* is pure pairwise BCE + a symmetric `delta.pow(2)` anchor. That symmetric anchor penalizes edge **deletion** exactly as hard as edge addition, and BCE on scorer-error edges gives no direct signal to remove a topologically-spurious-but-individually-plausible edge. So 01 can realistically only *add* structure or leave the graph alone — it cannot learn to **prune** the over-dense `G_pairwise` that is the whole problem PRING exposes.

Run 02 closes that gap with three coupled changes (implemented in `tccig/s2gae.py`, `tccig/train.py`, `tccig/test.py`; config `configs/tccig/02.yaml`; verified by `tests/unit/test_tccig_topology_training.py`, `tests/unit/test_tccig_deletion_diagnostics.py`, `tests/integration/test_tccig_topology_training_stage.py`):

- **Differentiable topology loss in the training gradient.** Each epoch, after the per-batch BCE+anchor pass, one full-plan topology backward runs over a train-side `InternalValidationPlan` (node-bucket subgraphs, sizes 20–200) via `src.topology.finetune_losses.compute_topology_losses` (the same differentiable surrogate the monitor uses: `graph_similarity`, `relative_density`, `degree_mmd`, `clustering_mmd`). Train and monitor now optimize the **same** objective family (`weights: {alpha: 1.0, beta: 8.0, gamma: 0.5, delta: 0.1}`).
- **Asymmetric, deletion-free residual anchor.** `relu(delta).pow(2).mean()` replaces the symmetric anchor (`residual_anchor.form: asymmetric_relu`, `weight: 1.0e-4`, 10× below 01's 1e-3). Upward (edge-adding) pushes are still anchored toward the scorer; downward (deletion) deltas are free, so the topology loss can drive pruning without fighting the anchor.
- **Warmup-ramped topology weight.** `topology_loss_scale` holds the topology term at 0 for `warmup_epochs: 5`, ramps linearly over `ramp_epochs: 10`, then holds at 1.0 — the refiner first learns a stable pairwise denoiser (avoiding a cold-start topology collapse reminiscent of Attempt 1) before topology pressure engages.

Coverage guarantee: the train plan augments node-bucket subgraphs so **every GT positive train edge** is covered by at least one bucket (`positive_edge_coverage == 1.0` asserted, else fail-fast). New per-epoch history fields: `train_topology_scale`, `train_topology_loss`, `train_topo_graph_sim`, `train_topo_relative_density`, `train_topo_degree_mmd`, `train_topo_clustering_mmd`. New test-time deletion diagnostics in `topology_metrics.json`: `edges_added`, `edges_deleted`, `net_edge_delta`, `deletion_precision` (the last now computed against GT-graph membership, not a raw-prob proxy).

**Two findings fixed before launch (from code review):** (1) the per-epoch topology backward runs the unwrapped refiner in `eval()` mode so dropout (`0.2`) cannot make ranks diverge under DDP — the full-plan graph is identical on every rank, so disabling dropout keeps gradients/parameters in lock-step without an all-reduce on that path; (2) `deletion_precision` derives ground-truth labels from `human_test_graph.pkl` (already loaded for the official metric) instead of the raw-prob<0.5 fallback, so a high-confidence raw edge that is *not* a true PPI correctly counts as a good deletion. Labels feed only the post-hoc diagnostic — `test_labels_visible_to_model: False` stays accurate.

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|------------------------------|---------------|
| **C1 (Primary)**: The S2GAE residual denoiser improves PRING topology fidelity over the raw frozen v3.1 pairwise graph at comparable pairwise quality. | This is the whole thesis — fix the over-dense graph failure mode PRING exposes, without touching the scorer. | On PRING Human/BFS topology test: refined graph beats `pairwise_baseline` on graph_sim and relative_density (toward 1.0) and lowers deg/cc/spectral MMD, while pairwise F1/AUPRC on `human_test_ppi.txt` does not materially drop. | B1, B2 |
| **C2 (Supporting)**: The gain is a genuine topology correction from the residual + scorer-error training, not just a different operating threshold on the same scorer scores. | Rules out the "it's only a threshold move" anti-claim — the most likely reviewer objection. | Frozen scorer swept across thresholds cannot match the refined graph's topology metrics at equal edge density / equal pairwise precision. | B2, B3 |
| **C3 (Run 02, Primary for `02`)**: Putting the differentiable topology loss in the training gradient (+ deletion-free anchor + warmup) lets the refiner actively **prune** spurious edges, beating the BCE-only run-01 refiner on PRING topology — driven by deletion, not just addition. | This is the run-02 method advance. 01 can only add/keep; 02 must show that learned pruning improves topology beyond what 01 achieves, at controlled pairwise quality. | On PRING Human/BFS: `02` beats `01` on graph_sim and relative_density (toward 1.0) with lower deg/cc/spectral MMD, **and** `edges_deleted > 0` with `net_edge_delta < 0` and high `deletion_precision`, at pairwise AUPRC ≥ raw floor. | B4, B5, B6 |

**Anti-claims to rule out**
- "The improvement is only a threshold change on the existing scorer scores." → B2 (threshold sweep of the baseline at matched density).
- "The residual connection / scorer-error sampling is decoration; plain BCE on all train pairs would do the same." → B3 (deletion study, deferred to config `02`).
- "Topology gains come at a large pairwise cost." → B1 reports both metric families at the fixed operating point.
- "The refiner just collapses the scorer's signal" (the Attempt-1 failure mode). → G0 stability gate proves refined ranking is preserved (refined AUPRC ≥ raw floor) before any expensive run.
- **(Run 02)** "The topology-loss gain is just the deletion-free anchor letting the threshold drift down — any sparser graph would score better." → B5 (matched-density comparison: `02` beats `01` and the baseline threshold sweep at equal edge density) + `deletion_precision` (deletions are GT-negative, not indiscriminate).
- **(Run 02)** "The differentiable topology term does nothing; the asymmetric anchor alone explains the pruning." → B6 ablation (`02` vs `02` with `topology_weight: 0` but asymmetric anchor kept).
- **(Run 02)** "Topology pruning destroys pairwise quality." → B4 reports refined AUPRC/F1 vs raw floor at the `02` operating point; warmup + AUPRC-floor guard protect against a cold-start collapse.

## Paper Storyline
- **Main paper must prove**: C1 (refiner > raw scorer on PRING topology at controlled pairwise quality), C2 (not a threshold artifact), and **C3 (the topology-conditioned training loss enables learned pruning that beats the BCE-only refiner)**. C1/C2 establish the refiner; C3 is the method advance that makes the contribution about *learning topology*, not just denoising.
- **Appendix can support**: per-node-size metric tables (node_sizes 20–200), training/validation topology-loss curves (now including `train_topology_*` from `02`), threshold-sensitivity of the refined output rule, deletion-diagnostics breakdown (`edges_added`/`edges_deleted`/`net_edge_delta`/`deletion_precision`), the warmup-schedule sensitivity, and the Attempt-1 instability + fix as a methods/ablation note (raw-vs-refined preservation).
- **Experiments intentionally cut (for runs 01–02)**: cross-species transfer, function-oriented PRING tasks, swapping the pairwise backbone, K-fold OOF scoring of the train graph, and a full warmup/anchor-weight grid (one default + one ablation point each). These are post-establishment extensions.

## Experiment Blocks

### Block 1: Main anchor result — refiner vs frozen scorer on PRING Human/BFS
- **Claim tested**: C1.
- **Why this block exists**: It is the headline result. If the refiner does not beat the raw pairwise graph on topology at the fixed operating point, the method does not work.
- **Dataset / split / task**: PRING Human, BFS split (`data/PRING/human/BFS`). Pairwise test on `human_test_ppi.txt`; topology reconstruction on the `all_test_ppi.txt` candidate universe with `human_test_graph.pkl` + `test_sampled_nodes.pkl` as metrics-only ground truth.
- **Compared systems**:
  1. **Frozen v3.1 pairwise baseline** — pinned artifact at `logs/tccig/pairwise_baseline` (raw scorer graph, no refiner). Now also recomputable from the run's own `raw_metrics.json` + `raw_probability` column.
  2. **TCCIG refiner (stabilized `01`)** — frozen scorer → `G_pairwise` (τ_pair at validation precision ≥ 0.8) → S2GAE residual denoiser (mean-agg, LayerNorm, tanh-bounded residual) → refined output at `p_refined ≥ 0.5`.
- **Metrics**: Decisive — topology `graph_sim` (↑), `relative_density` (→1.0), `deg_dist_mmd` / `cc_mmd` / `laplacian_eigen_mmd` (↓). Secondary — pairwise `precision`, `recall`, `f1`, `auprc` on `human_test_ppi.txt` (read directly from `refined_metrics.json` vs `raw_metrics.json`).
- **Setup details**: GraphConv encoder (mean aggregation), input_dim 1536, hidden 128, 2 layers, LayerNorm on; cross-layer decoder, 256 hidden × 2 layers; residual bounded `residual_scale=4.0`, residual_weight 1e-3; 40 epochs; AdamW lr 1e-4; bf16; DDP on 4× A40 via `scripts/tccig.sh`. Checkpoint monitored on `val_topology_loss`. ESM3 1024 embedding cache. Single seed for run 01.
- **Success criterion**: Refined graph beats the baseline on graph_sim and at least two of the three MMD metrics, with `relative_density` moving toward 1.0, and pairwise F1/AUPRC within a small margin (no large pairwise regression). **Hard floor**: refined test AUPRC must be ≥ raw AUPRC − small margin; a collapse like Attempt 1 (refined ≪ raw) is an automatic fail.
- **Failure interpretation**: If topology does not improve but training is stable, either τ_pair makes `G_pairwise` too clean/too sparse, or the residual anchor over-constrains the refiner. Inspect input vs refined edge counts and validation topology curves. If refined AUPRC collapses again, re-open the stability investigation (the fix did not hold at scale).
- **Table / figure target**: Main Table 1 (rows: baseline, refiner; columns: pairwise + topology metrics). Per-node-size breakdown → appendix.
- **Priority**: MUST-RUN.

### Block 2: Novelty / anti-threshold isolation — frozen scorer threshold sweep
- **Claim tested**: C2 (and reinforces C1).
- **Why this block exists**: The cheapest reviewer rebuttal is "you just re-thresholded the same scores." This block proves the refiner reaches topology that no single threshold on the raw scorer can.
- **Dataset / split / task**: Same PRING Human/BFS topology test universe.
- **Compared systems**: Frozen v3.1 scorer thresholded at a sweep of values (recover several densities), vs the single refined graph from Block 1.
- **Metrics**: Topology metrics as a function of edge density / pairwise precision; overlay the refiner's single operating point.
- **Setup details**: Reuse cached scorer scores (`data/tccig/score_cache/01/scores`) and the run's `raw_probability` column — no retraining. Pure post-hoc thresholding + metric recompute.
- **Success criterion**: At the density (or pairwise precision) matching the refined graph, the best baseline threshold still has worse graph_sim / higher MMD than the refiner.
- **Failure interpretation**: If a baseline threshold matches the refiner, the contribution collapses to calibration — the residual denoiser is not adding topological structure and the method needs rethinking (e.g., stronger topology objective).
- **Table / figure target**: Figure 1 (topology metric vs density curve, baseline sweep + refiner point).
- **Priority**: MUST-RUN (analysis-only; no extra training).

### Block 3: Simplicity / design-deletion check (deferred to config `02`)
- **Claim tested**: anti-claim — residual connection + scorer-error edge sampling are load-bearing, not decorative.
- **Why this block exists**: Defends the design against "plain BCE on all train pairs, no residual, would do the same."
- **Dataset / split / task**: Same PRING Human/BFS.
- **Compared systems**: (a) full method (stabilized run 01); (b) − residual anchor (residual_weight 0); (c) − scorer-error sampling (exhaustive train-pair BCE, the pre-ADR-002 contract); (d) − cross-layer decoder (simple inner-product decode). All inherit the stabilization knobs (mean-agg, LayerNorm, bounded residual) so the deletion isolates the intended component, not numerical stability.
- **Metrics**: Same topology + pairwise families; report deltas vs full method.
- **Setup details**: Fork `01.yaml` into `02.yaml`+ variants, one knob changed each. Same seed/budget.
- **Success criterion**: Removing each component measurably hurts at least one decisive metric.
- **Failure interpretation**: If a deletion matches the full method, drop that component for a simpler paper.
- **Table / figure target**: Ablation Table 2.
- **Priority**: NICE-TO-HAVE for run 01 (this is the next experiment, not this one).

### Stabilization ablation (optional appendix, config `02`)
- **Claim tested**: the stabilization choices (mean-agg, LayerNorm, bounded residual) are necessary, not cosmetic — a reviewer-facing record of the Attempt-1 failure.
- **Compared systems**: stabilized full method vs each stabilization knob reverted (sum aggregation; LayerNorm off; `residual_scale` unbounded). Expect training divergence / refined collapse on revert.
- **Metrics**: `train_residual_anchor_loss` trajectory, refined vs raw AUPRC.
- **Priority**: NICE-TO-HAVE. Only if a reviewer questions the stabilization design.

### Block 4 (Run 02): Topology-loss main result — trained-to-prune refiner vs Run 01 vs raw scorer
- **Claim tested**: C3 (and re-tests C1 under the new objective).
- **Why this block exists**: The headline Run 02 result. Run 01's BCE-only refiner can only add edges (the symmetric anchor penalizes deletion); on an over-dense `G_pairwise` the topology fix it can deliver is structurally limited. Block 4 asks whether a refiner whose training gradient *contains* the differentiable topology surrogate, with a deletion-free anchor, actually prunes and thereby improves topology beyond Run 01.
- **Dataset / split / task**: Same PRING Human/BFS topology test universe as Block 1. Identical operating-point definition (τ_pair at val precision ≥ 0.8 input; `p_refined ≥ 0.5` output).
- **Compared systems** (three-way, all at the fixed operating point):
  1. **Frozen v3.1 pairwise baseline** (raw scorer graph) — same artifact as Block 1.
  2. **Run 01 refiner** (BCE + symmetric anchor; topology used only for checkpoint selection) — the M2 result.
  3. **Run 02 refiner** (`02.yaml`: BCE + **asymmetric_relu** anchor `weight 1e-4` + per-epoch full-plan differentiable topology backward, warmup 5 / ramp 10, `weights α/β/γ/δ = 1/8/0.5/0.1`).
- **Metrics**: Decisive — topology `graph_sim` (↑), `relative_density` (→1.0), `deg/cc/spectral MMD` (↓), **plus the new `deletion_diagnostics`** (`edges_deleted`, `net_edge_delta`, `deletion_precision`). Secondary — pairwise `precision/recall/f1/auprc` from `refined_metrics.json`. The deletion diagnostics are the mechanism evidence that distinguishes C3 from C1.
- **Setup details**: `configs/tccig/02.yaml`, forked from stabilized `01.yaml` (inherits mean-agg, LayerNorm, bounded residual, all Block-1 hyperparameters). `monitor_metric` stays `val_topology_loss`. Per-epoch topology backward runs in **eval mode on the unwrapped refiner** (no dropout) so the full-plan graph — identical on every rank — yields identical gradients and keeps DDP parameters in lock-step. Coverage augmentation guarantees `positive_edge_coverage == 1.0` (asserts or fails fast). 40 epochs, 4× A40 DDP, single seed (matched to Run 01 for a fair head-to-head).
- **Success criterion**: Run 02 strictly improves on Run 01 on at least one decisive topology metric (lower density / higher graph_sim) *with* `edges_deleted > 0` and `net_edge_delta < 0` (it genuinely prunes), while pairwise AUPRC stays ≥ raw floor − margin. `deletion_precision` (GT-derived) is meaningfully above the prune-everything baseline.
- **Failure interpretation**: If Run 02 ≈ Run 01 on topology, the topology gradient is not buying anything beyond checkpoint selection — either the warmup ramp is too conservative, β (degree term) is too low, or the anchor still over-constrains; retune `topology_weight` / `weights` (cheap config forks) before concluding the training-loss path is unnecessary. If Run 02 prunes but pairwise AUPRC drops below floor, the topology term is over-pruning true edges — raise the anchor weight or β/δ balance.
- **Table / figure target**: Main Table 1 gains a third row (Run 02); new Table 1b = deletion diagnostics (added/deleted/net/precision) across the three systems.
- **Priority**: MUST-RUN (this is the Run 02 deliverable).

### Block 5 (Run 02): Deletion-mechanism isolation — asymmetric anchor vs warmup vs full topology loss
- **Claim tested**: C3 anti-claims — that pruning comes from the topology gradient, not merely from making deletion cheap, and that warmup is load-bearing.
- **Why this block exists**: Run 02 changes three things at once (asymmetric anchor, topology backward, warmup ramp). A reviewer will ask which one causes the pruning. This block separates them with one-knob forks of `02.yaml`.
- **Dataset / split / task**: Same PRING Human/BFS topology test.
- **Compared systems** (one-knob deltas from `02.yaml`):
  1. **Anchor-only**: asymmetric_relu anchor ON, `topology_weight = 0` (deletion is free but no topology gradient). Tests "does cheap deletion alone prune usefully?"
  2. **Topology-only**: symmetric anchor (Run 01's), topology backward ON. Tests "does the topology gradient prune even when the anchor fights deletion?"
  3. **No-warmup**: full `02.yaml` but `warmup_epochs = 0, ramp_epochs = 0` (topology pressure from epoch 1). Tests whether warmup matters for stability/quality vs immediate pressure.
  4. **Full `02.yaml`** (reference, = Block 4 Run 02).
- **Metrics**: `edges_deleted`, `net_edge_delta`, `deletion_precision`, decisive topology metrics, pairwise AUPRC floor; for No-warmup also the early-epoch `train_loss`/`train_topology_scale` trajectory.
- **Setup details**: Four config forks, one knob each, same seed/budget as Block 4. All reuse the same cached scores and train-topology plan.
- **Success criterion**: Full `02.yaml` ≥ each single-knob variant on the decisive topology metric; specifically Topology-only should still prune (isolating the gradient as the driver) and Full should beat Anchor-only (isolating that cheap deletion alone is insufficient).
- **Failure interpretation**: If Anchor-only matches Full, the topology gradient is decoration — collapse the method to "Run 01 + asymmetric anchor" for a simpler paper. If No-warmup matches Full, drop the warmup schedule (one fewer hyperparameter).
- **Table / figure target**: Ablation Table 3 (deletion-mechanism isolation).
- **Priority**: NICE-TO-HAVE (runs only after Block 4 confirms C3; defends it against the "which knob" objection).

### Block 6 (Run 02): Topology-loss-weight / schedule sensitivity
- **Claim tested**: C3 robustness — the pruning result is not a single-point hyperparameter artifact.
- **Why this block exists**: `topology_weight`, the `β` degree term, and the warmup/ramp lengths are the most likely overfit knobs. A small sweep shows the result is stable across a sensible range.
- **Dataset / split / task**: Same PRING Human/BFS topology test.
- **Compared systems**: `02.yaml` with `topology_weight ∈ {0.5, 1.0, 2.0}` and (optionally) `β ∈ {4, 8, 16}`; a short grid, not a full cross-product.
- **Metrics**: decisive topology metrics + deletion diagnostics + pairwise AUPRC floor as functions of the swept knob.
- **Setup details**: Config forks, reuse cached scores. Prefer fewer points run to convergence over a dense grid.
- **Success criterion**: Topology improvement over Run 01 holds across the swept range; pairwise AUPRC stays above floor; identifies whether the chosen defaults sit in a stable region.
- **Failure interpretation**: If only one knob setting works, the result is fragile — report the sensitivity honestly and pick the most defensible operating point.
- **Table / figure target**: Appendix figure (topology metric + deletion_precision vs topology_weight).
- **Priority**: NICE-TO-HAVE (appendix robustness).

**Frontier-necessity block**: Intentionally skipped. TCCIG's refiner is a small GNN, not a frontier (LLM/diffusion/RL) primitive. The only large model is the frozen ESM-3-based v3.1 scorer, which is an input boundary, not a claimed contribution. No frontier-necessity experiment is needed; state this explicitly in the paper.

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 — Sanity | Pipeline runs end-to-end; metric/IO contract correct; scorer cache populates; self-pairs dropped; new raw/refined artifacts present | Short run: 1–2 epochs of `01.yaml` (or `epochs: 2` fork) on 1 GPU | Pipeline completes, writes `pairwise_test` (`raw_metrics.json` + `refined_metrics.json`) + `topology_test`, no NaNs, edge counts sane | ~1–2 GPU-h | Embedding cache / PyG CUDA wheel mismatch; τ_pair precision target unreachable on val |
| **G0 — Stability gate (NEW, blocks M2)** | Prove the fix holds at real scale before the 4-day run: training is stable and the refiner preserves the scorer's ranking | Short run: 3–5 epochs of stabilized `01.yaml` on 1 GPU, full data | **Stop/go**: (1) `train_loss` and `train_bce` trend down across epochs; (2) `train_residual_anchor_loss` stays bounded (O(10s), not 1e8+); (3) refined val AUPRC ≥ raw val AUPRC − 0.05 (no collapse); (4) refined output not >50% a single saturated value | ~1–3 GPU-h | Fix validated on CPU fixtures only; dense real `G_pairwise` could still misbehave. If gate fails, do NOT launch M2 — return to stabilization |
| M1 — Baseline | Confirm frozen pairwise baseline artifact is loadable and its metrics reproduce | Load `logs/tccig/pairwise_baseline`; compute topology metrics; cross-check against `raw_metrics.json` from G0 | Baseline topology + pairwise numbers in hand | <0.5 GPU-h (CPU metrics) | Baseline artifact missing locally — regenerate from frozen scorer if absent |
| M2 — Main method | Full `01.yaml` training run to convergence | 1× full run via `scripts/tccig.sh` (40 epochs, 4× A40 DDP) | C1 satisfied: refiner beats baseline topology at controlled pairwise; refined AUPRC ≥ raw floor | ≤4 days wall (SBATCH `-t 4-00:00:00`); est. 4× A40 GPU-h | Over-constrained residual / wrong τ_pair density; `val_topology_loss` not tracking test topology |
| M3 — Decision | Anti-threshold isolation (Block 2) on cached scores | Post-hoc threshold sweep + metric recompute | C2 satisfied: no baseline threshold matches refiner topology | <1 GPU-h, CPU | Density-matching ambiguity — fix the matching variable (density vs precision) up front |
| M4 — Polish | Per-node-size tables, topology curves, refined-output-threshold sensitivity, stabilization note → appendix | Re-use M2 outputs; optional `02+` deletion forks | Appendix evidence assembled; decide if `02` ablations are needed | low (analysis) + ablation jobs if run | Scope creep into full ablation suite before C1/C2 are locked |
| **G1 — Run 02 gate (NEW, blocks M5)** | Prove the topology backward is stable and actually prunes at real scale before the full Run 02 | Short run: 3–5 epochs of `02.yaml` on 1 GPU, full data | **Stop/go**: (1) `train_topology_loss` finite and trends down once `train_topology_scale > 0`; (2) `train_residual_anchor_loss` stays bounded under the asymmetric anchor; (3) `edges_deleted > 0` / `net_edge_delta < 0` appears (pruning is active); (4) refined val AUPRC ≥ raw − 0.05 (deletion is not destroying true edges) | ~1–3 GPU-h | Topology backward OOM/throughput after coverage augmentation (deferred open item); warmup mis-set so no pruning pressure in the short window — set `warmup_epochs` low for the gate run |
| M5 — Run 02 main | Full `02.yaml` training to convergence (trained-to-prune refiner) | 1× full run via `scripts/tccig.sh` with `--config configs/tccig/02.yaml` (40 epochs, 4× A40 DDP) | C3 satisfied: Run 02 beats Run 01 on a decisive topology metric with `net_edge_delta < 0` and AUPRC ≥ raw floor | ≤4 days wall; est. 4× A40 GPU-h | Topology gradient ≈ no-op vs Run 01 (retune `topology_weight`/`weights`); over-pruning below AUPRC floor |
| M6 — Run 02 decision | Deletion-mechanism isolation (Block 5): anchor-only / topology-only / no-warmup forks | 3× short-to-full forks of `02.yaml`, one knob each | C3 anti-claims ruled out: topology-only still prunes; Full > anchor-only | ~3× partial runs (less than M5 each if early-stoppable) | Cost of three extra training runs — gate on Block 4 success first |
| M7 — Run 02 polish | Topology-weight/schedule sensitivity (Block 6); deletion-diagnostics table; per-node-size deltas vs Run 01 | Re-use M5 outputs + a small weight sweep | Robustness shown; appendix Run 02 evidence assembled | low (analysis) + small sweep | Sweep scope creep — keep to 3 points per knob |

**Run 01 must-run**: M0, **G0**, M1, M2, M3 (Blocks 1 + 2).
**Run 02 must-run (after Run 01 C1 is established)**: **G1**, M5, M6 (Blocks 4 + 5).
**Nice-to-have**: M4, M7, Block 3 deletion study, Block 6 sensitivity, and the stabilization ablation.

> **Sequencing**: Run 02 (M5+) is gated on Run 01 M2 producing a credible C1 result. C3 is *comparative* — "trained-to-prune beats BCE-only" — so it needs the Run 01 refiner as its baseline. Do not launch M5 before M2 lands, otherwise there is nothing to compare against. The code for Run 02 is already implemented, tested, and reviewed (asymmetric anchor, per-epoch topology backward in eval mode for DDP determinism, GT-derived deletion diagnostics); M5 is a config launch, not new development.

## Compute and Data Budget
- **Total estimated GPU-hours (run 01 core)**: M2 dominates — one 40-epoch DDP job on 4× A40 (job-bounded by the 4-day SBATCH wall, expected far less). M0/G0/M1/M3 are <6 GPU-h combined; M3 is CPU-bound metric recompute. **G0 is the cheapest insurance in the plan**: 1–3 GPU-h spent to avoid wasting a 4-day M2 on another unstable run.
- **Total estimated GPU-hours (run 02)**: M5 is one more 40-epoch DDP job, plus the per-epoch full-plan topology backward overhead (one extra forward/backward over all plan buckets per epoch — the deferred memory/throughput open item; G1 measures it before M5 commits). M6 adds up to three single-knob forks (early-stoppable). M7 is a small (≤3-point) weight sweep + CPU analysis. G1 is the same cheap-insurance pattern as G0. Run 02 reuses Run 01's scorer score cache — no new ESM3 inference.
- **Data preparation needs**: PRING Human/BFS already present under `data/PRING/human/BFS`; ESM3 1024 embedding cache at `data/embeddings/esm3_1024`; scorer score cache auto-written to `data/tccig/score_cache/01/`. Run 02 additionally builds a *train-side* topology plan once per run (node-bucket sampler + GT-positive-edge coverage augmentation, asserted `positive_edge_coverage == 1.0`); this is in-process at train startup, not a separate data-prep step. No new data prep.
- **Human evaluation needs**: None.
- **Biggest bottleneck**: A single full training run + correct ESM3/PyG CUDA environment on HPC; everything downstream (Block 2/3, appendix) is cheap analysis on cached scores or forked configs.

## Risks and Mitigations
- **Risk: the stabilization fix validated on CPU fixtures does not hold on the dense real `G_pairwise`.** → Mitigation: **G0 gate** — a short real-data run with explicit bounded-anchor and no-collapse checks before committing to M2. This is the top risk now and the reason G0 exists.
- **Risk: `val_topology_loss` (checkpoint monitor) selects a topologically-plausible but pairwise-broken epoch** (the Attempt-1 aggravating factor). → Mitigation: monitor kept per the C1 design, but G0 and M2 both enforce a hard refined-AUPRC floor as a guard; if the selected checkpoint fails the floor, switch monitor to `val_auprc` (machinery already supports it) and re-run. Plot monitored metric vs held-out topology at M2.
- **Risk: τ_pair (validation precision ≥ 0.8) yields a `G_pairwise` that is too sparse**, starving the GNN of context. → Mitigation: log input vs refined edge counts at M0/G0; if sparse, relax target_precision or report sensitivity.
- **Risk: residual_scale=4.0 over-bounds the residual**, leaving the refiner unable to make large corrections. → Mitigation: G0 shows whether refined ranking improves over raw at all; if the residual is saturating, raise `residual_scale` (it is config-gated). Cheap to retune before M2.
- **Risk: baseline artifact `logs/tccig/pairwise_baseline` absent** (not present locally). → Mitigation: confirm on HPC at M1; regenerate from the frozen scorer if missing (it is a pinned historical artifact, not pipeline-regenerated).
- **Risk: pairwise quality regresses** when topology improves. → Mitigation: B1 reports both families at the fixed operating point from `raw_metrics.json` / `refined_metrics.json`; if F1/AUPRC drops materially, surface the tradeoff rather than hiding it.

### Run 02 (topology-conditioned training) risks
- **Risk (top Run 02 risk): the per-epoch full-plan topology backward OOMs or is too slow at real scale** after GT-positive coverage augmentation (the spec's deferred open item). → Mitigation: **G1 gate** runs the topology backward on full data for a few epochs first; `coverage_bucket_count` is logged so an explosion is visible. If memory-bound, cap coverage buckets or reduce `samples_per_size`/`node_sizes` before M5.
- **Risk: the topology gradient is a near no-op** — Run 02 matches Run 01, so C3 fails. → Mitigation: Block 5 (anchor-only / topology-only) isolates whether the gradient or the asymmetric anchor drives any pruning; if the gradient is inert, retune `topology_weight`/`weights` (esp. `β`) or conclude the anchor alone suffices and simplify the paper.
- **Risk: over-pruning** — the deletion-free anchor + topology loss delete true edges, dropping refined AUPRC below the raw floor. → Mitigation: G1 and M5 enforce the same refined-AUPRC ≥ raw − 0.05 floor as Run 01; `deletion_precision` (now GT-derived) flags whether deletions hit true PPIs. Raise the asymmetric anchor `weight` or shorten the ramp if over-pruning.
- **Risk: multi-rank DDP divergence on the unwrapped topology backward** (raised in code review). → Mitigation: already fixed — the topology forward runs in `eval()` mode so dropout cannot desync ranks; the full-plan graph is identical per rank. Re-confirm at G1 that multi-GPU and single-GPU runs agree on `train_topology_loss`.
- **Risk: `monitor_metric=val_topology_loss` now co-moves with the training objective**, so checkpoint selection could reward the train-time proxy rather than held-out topology. → Mitigation: selection still uses the non-differentiable eval path on the validation split (not the train plan); keep the refined-AUPRC floor guard and plot monitor vs held-out test topology at M5.

## Final Checklist
- [x] Main paper tables are covered (Table 1 = B1, Figure 1 = B2; Run 02 Table 2 = B4)
- [x] Novelty is isolated (B2 anti-threshold for C1/C2; B5 deletion-mechanism isolation for C3; B3 deletion deferred)
- [x] Simplicity is defended (B5 ablates anchor-only vs topology-only vs no-warmup so each Run 02 knob earns its place; method is still a small GNN)
- [x] Frontier contribution is justified or explicitly not claimed (explicitly NOT claimed — frozen scorer is an input boundary)
- [x] Nice-to-have runs are separated from must-run runs (Run 01: M0/G0/M1–M3 must-run; Run 02: G1/M5/M6 must-run; M4/M7 + sensitivity nice-to-have)
- [x] Attempt-1 failure recorded with root cause + fix; cheap stability gates (G0, G1) inserted before each expensive full run
- [x] Run 02 (topology-conditioned loss) framed as a comparative claim (C3: trained-to-prune beats BCE-only), gated on Run 01 C1, code implemented + tested + review findings fixed
