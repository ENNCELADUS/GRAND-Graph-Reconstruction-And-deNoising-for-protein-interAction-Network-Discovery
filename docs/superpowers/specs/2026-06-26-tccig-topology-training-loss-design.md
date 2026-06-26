# TCCIG Run 02 — Topology-Conditioned Training Loss for Learned Edge Deletion

**Date**: 2026-06-26
**Status**: Design (approved in brainstorming)
**Config**: `configs/tccig/02.yaml` (fork of `01.yaml`)
**Prior run**: R004/M2 (`artifacts/tccig_01_20260626`) — stabilization held, C1 not met.

## Problem

The stabilized refiner (run `01`) does not improve PRING topology over the raw frozen
v3.1 pairwise baseline. Analysis of `artifacts/tccig_01_20260626` shows:

- Checkpoint selection (`val_topology_loss`) picked **epoch 1** — the global monitor
  minimum (`best_monitor_value = 831.0154`, `best_validation_auprc = 0.6833`). At epoch 1
  the residual anchor is ~5e-5, so the refiner is effectively the identity: refined ≈ raw
  on pairwise (AUPRC 0.79206 vs 0.79177) and marginally **worse** on all five topology
  metrics (graph_sim 0.33663 vs 0.33683; relative_density 3.163 vs 3.134; deg/cc/spectral
  MMD all slightly higher).
- The refiner does **not prune**. It applies a small uniform positive logit shift to all
  visible pairwise test scores, so the topology graph only gains edges, never removes them.

### Root cause (confirmed in code)

1. **Topology is monitor-only.** `s2gae_loss_terms` (`tccig/s2gae.py:375`) is
   `BCE + residual_anchor`. The topology signal lives only in `_validation_topology_loss`
   (`tccig/s2gae.py:1135`), used for checkpoint selection — it never enters the gradient.
   Training has no reason not to densify.
2. **The objective densifies.** The sampled scorer-error objective is FN-dominated
   (epoch 1: 21,684 FN-to-add vs 4,119 FP-to-delete, ~5:1). The dominant gradient pushes
   scores **up** to recover missing edges. Across training, `selected_rule_positive_edges`
   grows 228k → 527k and internal-val relative_density climbs 11 → 23 while graph_sim
   falls 0.184 → 0.128. Pairwise gain and topology fidelity are anti-correlated; no
   checkpoint on the trajectory improves topology.
3. **The anchor forbids deletion.** `residual_anchor = delta.pow(2).mean()`
   (`tccig/s2gae.py:389`) is a *symmetric* quadratic prior — it penalizes a negative delta
   (deletion) exactly as hard as a positive one, pulling the refiner toward identity.
4. **Deletion is already mechanically possible.** `_bounded_residual`
   (`tccig/s2gae.py:353`) is symmetric `s·tanh(δ/s)`; the residual can push a raw edge's
   logit below the 0.5 threshold. The refiner *can* delete — it is simply never *trained*
   to, and is actively discouraged by the anchor.

The eval topology metric path (`_validation_topology_metrics`, `tccig/s2gae.py:1086`) is
**non-differentiable**: hard threshold → discrete `nx.Graph` → MMD. It cannot be the
training loss. The fix requires a differentiable surrogate that creates downward (deletion)
pressure while staying faithful to the eval metric used for checkpoint selection.

## Thesis

Adding a **differentiable, topology-conditioned loss** to the *training* objective — over
sampled subgraphs, matching refined soft edge probabilities against the true train-graph
topology — gives the refiner a reason to **prune** topologically-spurious edges. Combined
with an **asymmetric residual anchor** (deletion no longer penalized) and a **warmup
schedule** (stable identity-ish start), this should produce a checkpoint that beats the
baseline on topology by removing edges, without collapsing pairwise quality.

## Run Boundary

**In scope (this run / `02.yaml`):**
- Differentiable topology loss in the training step (Approach 1: density + degree + soft
  graph similarity + clustering, balanced/topology-conditioned — **not** generic sparsity).
- Asymmetric residual anchor (B): deletion free, upward push penalized, reduced weight.
- Warmup ramp of the topology loss weight.
- Per-epoch full-plan topology backward over a node-bucket plan with **GT-positive-edge
  coverage augmentation**.
- Deletion diagnostics at test time.

**Out of scope (deferred):**
- Block 2 anti-threshold sweep (C2) — defer until a refined graph differs from baseline.
- Block 3 deletion study (architecture ablations).
- τ_pair density sensitivity sweep — separate cheap CPU analysis, not gating this run.

## Success Bar

All criteria must hold on the **selected checkpoint** (PRING Human/BFS test):

| Criterion | Meaning | Threshold |
|---|---|---|
| `deleted_edges > 0` | refiner actually prunes | removes raw `G_pairwise` edges, not only adds |
| added edges not exploding | not one-directional growth | net edge delta negative or near-flat |
| `graph_sim` improves | topology more like true graph | refined > baseline (0.33683) |
| `relative_density` → 1 | no longer over-dense | refined below baseline 3.134, toward 1.0 |
| pairwise AUPRC/F1 within floor | no reckless deletion of positives | refined AUPRC ≥ raw (0.79177) − small margin |
| deletion precision high | deleted edges are the right ones | deleted edges concentrate in pairwise-negative / low-confidence / topology-spurious region |

## Design

### Components

1. **Train-topology plan builder** (orchestrator, `tccig/train.py`)
   - Mirrors `_build_validation_topology_bundle` (`tccig/train.py:382`) on the **train**
     split graph (`build_pair_supervision_graph`, train node universe).
   - Reuses `sample_topology_evaluation_subgraphs` (mixed strategy) +
     `build_internal_validation_plan` — the eval-faithful primitives.
   - **Does not** reuse `sample_edge_cover_subgraphs`.
   - Built once under the Accelerate runtime (sampler is seeded/deterministic).

2. **Positive-edge coverage augmentation** (new helper)
   - Sample normal train topology buckets first (base buckets).
   - Compute which train GT positive edges are not contained in any bucket's
     `pair_records`.
   - Add extra coverage buckets seeded by the endpoints of uncovered positive edges.
   - Expand those buckets to configured `node_sizes` with the **same** BFS/DFS/random-walk
     machinery used by the base sampler.
   - Coverage guarantee: **GT positive edges only**. Non-edges ride along via the all-pairs
     `pair_records` within each bucket.
   - Assert `positive_edge_coverage == 1.0` after augmentation, else raise (fail-fast).
   - Log `base_bucket_count`, `coverage_bucket_count`, `positive_edge_coverage`.
   - Union of base + coverage buckets → one `InternalValidationPlan` (the TrainTopologyPlan).

3. **Differentiable topology loss term** (training step,
   `_S2GAESampledTrainStepModule` in `tccig/s2gae.py`)
   - Reuse `compute_topology_losses` (`src/topology/finetune_losses.py:352`), pairwise path:
     `(num_nodes, pair_index_a, pair_index_b, pred_pair_probabilities,
     target_pair_probabilities)` per bucket.
   - `pred_pair_probabilities = σ(refined_logit)` from the refiner over the bucket's
     `pair_records`.
   - `target_pair_probabilities` from the true train subgraph: 1.0 for GT positive edges,
     0.0 otherwise.
   - Weighted sum `α·graph_similarity + β·relative_density + γ·degree_mmd + δ·clustering_mmd`
     (the module already mirrors the eval-side weights and the Gaussian-TV MMD kernel in
     `src.topology.metrics.compute_mmd`).

4. **Asymmetric residual anchor** (training step)
   - Replace `delta.pow(2).mean()` with a one-sided penalty: only upward pushes penalized,
     e.g. `relu(delta).pow(2).mean()` (negative/deletion delta free), at a reduced weight.
   - Magnitude stability still provided by the tanh `residual_scale=4.0` bound.

5. **Warmup schedule**
   - Reuse `topology_loss_scale` + `TopologyLossWeightSchedule`
     (`src/topology/finetune_losses.py:39`): `warmup_epochs` of pure BCE + anchor (stable
     identity-ish start), then linear/cosine ramp of the topology weight to 1.0.

### Data flow (per epoch)

```
[once, orchestrator under Accelerate]
  G_train (build_pair_supervision_graph, train split)
    -> sample_topology_evaluation_subgraphs (mixed) ............ base buckets
    -> coverage pass: find uncovered GT+ edges
        -> seed extra buckets from uncovered edge endpoints
        -> expand to node_sizes via same BFS/DFS/RW machinery ... coverage buckets
    -> build_internal_validation_plan(union) .................. TrainTopologyPlan
    -> assert positive_edge_coverage == 1.0
    -> log base_bucket_count, coverage_bucket_count, positive_edge_coverage

[each epoch]
  per-batch: BCE + asymmetric_anchor over sampled scorer-error edges (existing path)
  once per epoch (full plan): for each bucket
       -> encode subgraph -> refined p_ij = sigmoid(refined_logit) over pair_records
       -> target_pair_probs from true train subgraph (GT+ = 1, else 0)
       -> compute_topology_losses (pairwise path) -> topo_loss
       -> scaled by topology_loss_scale(epoch) -> backward
  checkpoint selection: val_topology_loss (unchanged, non-diff eval path)
```

The training step total objective:

```
total = w_bce   * BCE(sampled error edges)
      + w_anchor * asymmetric_anchor(delta)            # deletion free; upward penalized
      + topology_loss_scale(epoch) * w_topo * topology_loss(TrainTopologyPlan buckets)
```

### Logging

- Plan: `base_bucket_count`, `coverage_bucket_count`, `positive_edge_coverage`.
- Per epoch: `train_topology_loss` and components `train_topo_graph_sim`,
  `train_topo_relative_density`, `train_topo_degree_mmd`, `train_topo_clustering_mmd`;
  existing `train_residual_anchor_loss` retained (now asymmetric).
- Test-time deletion diagnostics: `edges_added`, `edges_deleted`, `net_edge_delta`,
  `deletion_precision` (fraction of deleted edges that are pairwise-negative /
  low-confidence). These populate the success-bar table directly.

### Error handling

- Coverage pass asserts `positive_edge_coverage == 1.0` after augmentation or raises — a
  silent coverage gap would void the "full use of edges" guarantee.
- Topology backward guarded for empty-bucket / single-node degenerate cases (sampler
  already enforces `>= 2` nodes).
- Reuse existing `_build_split_graph` / plan validation paths.

### Testing (TDD)

- **Coverage augmentation**: synthetic graph with a known uncovered edge → after the pass
  the edge is covered, `positive_edge_coverage == 1.0`, counts logged correctly.
- **Asymmetric anchor**: negative delta → zero/near-zero penalty; positive delta →
  penalized.
- **Topology term backprops**: gradient flows to refiner params; an over-dense bucket
  yields a negative gradient on weak edges (deletion-pressure sign check).
- **Warmup**: `topology_loss_scale` returns 0 during warmup, ramps to 1.0 after.
- **Integration**: short 2-epoch run produces `edges_deleted > 0` and writes the new
  deletion diagnostics.

### Config surface (`configs/tccig/02.yaml`)

Fork of `01.yaml` with:

```yaml
refiner:
  residual_anchor:
    form: asymmetric_relu      # deletion free; upward push penalized
    weight: <reduced>          # below 01's residual_weight 1e-3
  topology_training:
    enabled: true
    node_sizes: [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    samples_per_size: 20
    strategy: mixed
    seed: 0
    coverage_augmentation: true
    topology_weight: <w_topo>
    weights: { alpha: ..., beta: ..., gamma: ..., delta: ... }
    schedule: { warmup_epochs: ..., ramp_epochs: ..., schedule: linear }
  monitor_metric: val_topology_loss   # unchanged
```

Exact weight/schedule values to be set in the implementation plan.

## Open Items for Implementation Plan

- Concrete default values: `w_topo`, asymmetric anchor `weight`, topology `weights`
  (α/β/γ/δ), `warmup_epochs`, `ramp_epochs`.
- Memory/throughput check for the per-epoch full-plan topology backward (number of buckets
  after coverage augmentation × pairs per bucket).
- Whether the coverage pass needs a cap on coverage-bucket count for very sparse / very
  dense train graphs.
