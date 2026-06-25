# Experiment Plan

**Problem**: Strong pairwise PPI scorers (frozen v3.1 `pair_context_gated_abba_no_cross`) classify individual pairs well but assemble into over-dense, topologically incoherent interactomes on the PRING graph-reconstruction benchmark.
**Method Thesis**: A lightweight S2GAE-style residual denoiser, trained only on the frozen scorer's error edges over a precision-thresholded noisy graph `G_pairwise`, can correct topology (degree/clustering/spectral structure) without retraining the scorer and without sacrificing pairwise quality.
**Date**: 2026-06-25

This is the **first** TCCIG experiment (`configs/tccig/01.yaml`, `scripts/tccig.sh`). Its job is to establish that the refiner path runs end-to-end, beats the frozen-scorer baseline on PRING topology metrics at matched/controlled pairwise quality, and that the residual-denoiser design choices are justified. It is a single-config establishing run, not the full ablation suite — but the plan below frames run `01` inside the claim map so later configs (`02+`) extend it rather than re-deriving the story.

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|------------------------------|---------------|
| **C1 (Primary)**: The S2GAE residual denoiser improves PRING topology fidelity over the raw frozen v3.1 pairwise graph at comparable pairwise quality. | This is the whole thesis — fix the over-dense graph failure mode PRING exposes, without touching the scorer. | On PRING Human/BFS topology test: refined graph beats `pairwise_baseline` on graph_sim and relative_density (toward 1.0) and lowers deg/cc/spectral MMD, while pairwise F1/AUPRC on `human_test_ppi.txt` does not materially drop. | B1, B2 |
| **C2 (Supporting)**: The gain is a genuine topology correction from the residual + scorer-error training, not just a different operating threshold on the same scorer scores. | Rules out the "it's only a threshold move" anti-claim — the most likely reviewer objection. | Frozen scorer swept across thresholds cannot match the refined graph's topology metrics at equal edge density / equal pairwise precision. | B2, B3 |

**Anti-claims to rule out**
- "The improvement is only a threshold change on the existing scorer scores." → B2 (threshold sweep of the baseline at matched density).
- "The residual connection / scorer-error sampling is decoration; plain BCE on all train pairs would do the same." → B3 (deletion study, deferred to config `02`).
- "Topology gains come at a large pairwise cost." → B1 reports both metric families at the fixed operating point.

## Paper Storyline
- **Main paper must prove**: C1 (refiner > raw scorer on PRING topology at controlled pairwise quality) and C2 (not a threshold artifact).
- **Appendix can support**: per-node-size metric tables (node_sizes 20–200), training/validation topology-loss curves, threshold-sensitivity of the refined output rule.
- **Experiments intentionally cut (for run 01)**: cross-species transfer, function-oriented PRING tasks, swapping the pairwise backbone, K-fold OOF scoring of the train graph. These are post-establishment extensions, not part of the first run.

## Experiment Blocks

### Block 1: Main anchor result — refiner vs frozen scorer on PRING Human/BFS
- **Claim tested**: C1.
- **Why this block exists**: It is the headline result. If the refiner does not beat the raw pairwise graph on topology at the fixed operating point, the method does not work.
- **Dataset / split / task**: PRING Human, BFS split (`data/PRING/human/BFS`). Pairwise test on `human_test_ppi.txt`; topology reconstruction on the `all_test_ppi.txt` candidate universe with `human_test_graph.pkl` + `test_sampled_nodes.pkl` as metrics-only ground truth.
- **Compared systems**:
  1. **Frozen v3.1 pairwise baseline** — pinned artifact at `logs/tccig/pairwise_baseline` (raw scorer graph, no refiner).
  2. **TCCIG refiner (this run, `01`)** — frozen scorer → `G_pairwise` (τ_pair at validation precision ≥ 0.8) → S2GAE residual denoiser → refined output at `p_refined ≥ 0.5`.
- **Metrics**: Decisive — topology `graph_sim` (↑), `relative_density` (→1.0), `deg_dist_mmd` / `cc_mmd` / `laplacian_eigen_mmd` (↓). Secondary — pairwise `precision`, `recall`, `f1`, `auprc` on `human_test_ppi.txt`.
- **Setup details**: GraphConv encoder, input_dim 1536 (scorer-pair features), hidden 128, 2 layers; cross-layer decoder, 256 hidden × 2 layers; residual_weight 1e-3; 40 epochs; AdamW lr 1e-4; bf16; DDP on 4× A40 via `scripts/tccig.sh`. Checkpoint monitored on `val_topology_loss`. ESM3 1024 embedding cache. Single seed for run 01.
- **Success criterion**: Refined graph beats the baseline on graph_sim and at least two of the three MMD metrics, with `relative_density` moving toward 1.0, and pairwise F1/AUPRC within a small margin (no large pairwise regression).
- **Failure interpretation**: If topology does not improve, either τ_pair makes `G_pairwise` too clean/too sparse, or the residual anchor (1e-3) over-constrains the refiner to the scorer. Inspect input vs refined edge counts and validation topology curves.
- **Table / figure target**: Main Table 1 (rows: baseline, refiner; columns: pairwise + topology metrics). Per-node-size breakdown → appendix.
- **Priority**: MUST-RUN.

### Block 2: Novelty / anti-threshold isolation — frozen scorer threshold sweep
- **Claim tested**: C2 (and reinforces C1).
- **Why this block exists**: The cheapest reviewer rebuttal is "you just re-thresholded the same scores." This block proves the refiner reaches topology that no single threshold on the raw scorer can.
- **Dataset / split / task**: Same PRING Human/BFS topology test universe.
- **Compared systems**: Frozen v3.1 scorer thresholded at a sweep of values (recover several densities), vs the single refined graph from Block 1.
- **Metrics**: Topology metrics as a function of edge density / pairwise precision; overlay the refiner's single operating point.
- **Setup details**: Reuse cached scorer scores (`data/tccig/score_cache/01/scores`) — no retraining. Pure post-hoc thresholding + metric recompute.
- **Success criterion**: At the density (or pairwise precision) matching the refined graph, the best baseline threshold still has worse graph_sim / higher MMD than the refiner.
- **Failure interpretation**: If a baseline threshold matches the refiner, the contribution collapses to calibration — the residual denoiser is not adding topological structure and the method needs rethinking (e.g., stronger topology objective).
- **Table / figure target**: Figure 1 (topology metric vs density curve, baseline sweep + refiner point).
- **Priority**: MUST-RUN (analysis-only; no extra training).

### Block 3: Simplicity / design-deletion check (deferred to config `02`)
- **Claim tested**: anti-claim — residual connection + scorer-error edge sampling are load-bearing, not decorative.
- **Why this block exists**: Defends the design against "plain BCE on all train pairs, no residual, would do the same."
- **Dataset / split / task**: Same PRING Human/BFS.
- **Compared systems**: (a) full method (run 01); (b) − residual anchor (residual_weight 0); (c) − scorer-error sampling (exhaustive train-pair BCE, the pre-ADR-002 contract); (d) − cross-layer decoder (simple inner-product decode).
- **Metrics**: Same topology + pairwise families; report deltas vs full method.
- **Setup details**: Fork `01.yaml` into `02.yaml`+ variants, one knob changed each. Same seed/budget.
- **Success criterion**: Removing each component measurably hurts at least one decisive metric.
- **Failure interpretation**: If a deletion matches the full method, drop that component for a simpler paper.
- **Table / figure target**: Ablation Table 2.
- **Priority**: NICE-TO-HAVE for run 01 (this is the next experiment, not this one).

**Frontier-necessity block**: Intentionally skipped. TCCIG's refiner is a small GNN, not a frontier (LLM/diffusion/RL) primitive. The only large model is the frozen ESM-3-based v3.1 scorer, which is an input boundary, not a claimed contribution. No frontier-necessity experiment is needed; state this explicitly in the paper.

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 — Sanity | Pipeline runs end-to-end; metric/IO contract correct; scorer cache populates; self-pairs dropped | Short run: 1–2 epochs of `01.yaml` (or `epochs: 2` fork) on 1 GPU | Pipeline completes, writes `pairwise_test` + `topology_test` artifacts, no NaNs, edge counts sane | ~1–2 GPU-h | Embedding cache / PyG CUDA wheel mismatch; τ_pair precision target unreachable on val |
| M1 — Baseline | Confirm frozen pairwise baseline artifact is loadable and its metrics reproduce | Load `logs/tccig/pairwise_baseline`; compute topology metrics on it | Baseline topology + pairwise numbers in hand | <0.5 GPU-h (CPU metrics) | Baseline artifact missing locally — regenerate from frozen scorer if absent |
| M2 — Main method | Full `01.yaml` training run to convergence | 1× full run via `scripts/tccig.sh` (40 epochs, 4× A40 DDP) | C1 satisfied: refiner beats baseline topology at controlled pairwise | ~per-job, ≤4 days wall (SBATCH `-t 4-00:00:00`); est. 4× A40 GPU-h | Over-constrained residual / wrong τ_pair density; val_topology_loss not tracking test topology |
| M3 — Decision | Anti-threshold isolation (Block 2) on cached scores | Post-hoc threshold sweep + metric recompute | C2 satisfied: no baseline threshold matches refiner topology | <1 GPU-h, CPU | Density-matching ambiguity — fix the matching variable (density vs precision) up front |
| M4 — Polish | Per-node-size tables, topology curves, refined-output-threshold sensitivity → appendix | Re-use run 01 outputs; optional `02+` deletion forks | Appendix evidence assembled; decide if `02` ablations are needed | low (analysis) + ablation jobs if run | Scope creep into full ablation suite before C1/C2 are locked |

**Must-run**: M0, M1, M2, M3 (Blocks 1 + 2).
**Nice-to-have**: M4 and Block 3 deletion study (configs `02+`).

## Compute and Data Budget
- **Total estimated GPU-hours (run 01 core)**: M2 dominates — one 40-epoch DDP job on 4× A40 (job-bounded by the 4-day SBATCH wall, expected far less). M0/M1/M3 are <4 GPU-h combined; M3 is CPU-bound metric recompute.
- **Data preparation needs**: PRING Human/BFS already present under `data/PRING/human/BFS`; ESM3 1024 embedding cache at `data/embeddings/esm3_1024`; scorer score cache auto-written to `data/tccig/score_cache/01/`. No new data prep.
- **Human evaluation needs**: None.
- **Biggest bottleneck**: A single full training run + correct ESM3/PyG CUDA environment on HPC; everything downstream (Block 2/3, appendix) is cheap analysis on cached scores or forked configs.

## Risks and Mitigations
- **Risk: τ_pair (validation precision ≥ 0.8) yields a `G_pairwise` that is too sparse**, starving the GNN of context. → Mitigation: log input vs refined edge counts at M0; if sparse, relax target_precision or report sensitivity.
- **Risk: `val_topology_loss` (checkpoint monitor) does not correlate with test topology metrics.** → Mitigation: at M2, plot monitored metric vs held-out topology summary; if decoupled, switch monitor to a hard topology metric (config supports `compute_clustering_mmd`).
- **Risk: residual_weight 1e-3 over-anchors the refiner to the scorer**, suppressing real correction. → Mitigation: Block 3 (a) ablation; quick sensitivity if C1 is weak.
- **Risk: baseline artifact `logs/tccig/pairwise_baseline` absent** (not present locally). → Mitigation: confirm on HPC at M1; regenerate from the frozen scorer if missing (it is a pinned historical artifact, not pipeline-regenerated).
- **Risk: pairwise quality regresses** when topology improves. → Mitigation: B1 reports both families at the fixed operating point; if F1/AUPRC drops materially, surface the tradeoff rather than hiding it.

## Final Checklist
- [x] Main paper tables are covered (Table 1 = B1, Figure 1 = B2)
- [x] Novelty is isolated (B2 anti-threshold; B3 deletion deferred to `02`)
- [x] Simplicity is defended (B3 deletion study planned; method is already a small GNN)
- [x] Frontier contribution is justified or explicitly not claimed (explicitly NOT claimed — frozen scorer is an input boundary)
- [x] Nice-to-have runs are separated from must-run runs (M0–M3 must-run; M4 + `02+` nice-to-have)
