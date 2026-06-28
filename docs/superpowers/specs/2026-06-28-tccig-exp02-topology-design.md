# TCCIG Experiment 02 — Topology Training Re-run Design

Date: 2026-06-28
Status: Approved (pending spec review)
Reference run: `artifacts/tccig_929091` (job 929091, failed)

## 1. Background and Problem

Experiment 02 trains an S2GAE refiner for protein–protein interaction prediction
with an auxiliary differentiable topology loss. Job `929091` (4× A40, 44.55 GiB
each) failed after 16h15m, having completed only 6 zero-scale warmup epochs. The
first positive-scale topology step (epoch 7) OOM'd.

Confirmed root causes from the artifacts:

1. **Full-plan autograd, replicated per rank.** `topology_plan_loss`
   (`tccig/s2gae.py:469`) iterates the entire plan every topology step: 766
   subgraphs, 12,792,400 candidate pairs. The 200-node bucket alone is 586
   subgraphs and 11,661,400 pairs (91% of all pairs; 708,831 positives ≈ 6%).
   For each subgraph it re-encodes, `argsort`s, and `searchsorted`s the full
   `graph.pair_index` in `_pair_lookup` (`tccig/s2gae.py:450`). The whole
   autograd graph is held at once and runs identically on all 4 ranks on the
   unwrapped refiner with no gradient all-reduce (`tccig/s2gae.py:806–828`). The
   crashing `torch.argsort` (`s2gae.py:460`) is just the last ~200 MiB; the
   per-step working set is structurally ~44 GiB. Alloc-config or batch-size
   tuning cannot fix this.

2. **200-node bucket domination.** `coverage_augmentation=true` expanded that
   bucket from 20 → 586 subgraphs to reach `positive_edge_coverage=1.0`, dumping
   all uncovered positives into the maximum-size bucket.

3. **~16 h silent setup.** `train_topology_plan.json` was written 23:23:08;
   `scores/train_topology.pt` finished 15:20:53. ~15h57m went to running the
   frozen v3.1 scorer over all 12.79M candidate pairs, with the collate path
   re-`torch.load`-ing per-endpoint embeddings (`tccig/train.py:801`) — turning a
   26 GB cache into millions of small reads. No progress logging.

4. **Warmup hid the failure and degraded topology.** Schedule
   `warmup_epochs=5, ramp_epochs=10` keeps `topology_loss_scale=0.0` through
   epoch index 5 (`src/topology/finetune_losses.py:39`), so the OOM path was only
   reached after 16 h of setup + 6 epochs. During pure-BCE warmup every topology
   metric worsened monotonically: `val_topology_loss` 7615.7 → 9235.9 (+21%),
   `relative_density` 31.8 → 34.9, `degree_mmd` 24.4 → 25.2, `graph_sim`
   0.186 → 0.181, `selected_rule_positive_edges` 230,354 → 251,693; `val_auprc`
   rose only 0.683 → 0.691.

## 2. Goals and Non-Goals

**Goals**
- Make a positive-scale topology step fit in 44 GiB and stop replicating it 4×.
- Stop the 200-node bucket from dominating both memory and the objective.
- Keep the training objective a faithful estimator of the full-space topology
  statistics used by validation/test, so the model generalizes to the full
  `n·(n−1)` pair space at eval time.
- Make setup scoring bounded, cache-reusable when valid, and observable.
- Validate the OOM/sharding path cheaply before committing to a 40-epoch run.

**Non-Goals**
- No change to the frozen v3.1 scorer, the BCE supervised objective, or the
  validation/test topology metric definitions (`_validation_topology_metrics`,
  `_validation_topology_loss`, `s2gae.py:1305/1357`). Train must continue to
  estimate those same quantities.
- No unrelated refactoring of the S2GAE encoder/decoder.

## 3. Objective: Hybrid Subset Loss with Inverse-Probability Weighting

Retain the existing graph-level topology losses (`graph_similarity`,
`relative_density`, `degree_mmd`; `clustering` optional via
`include_clustering_mmd`) but evaluate them on a per-epoch **labeled subset** of
each subgraph rather than the full `n·(n−1)` pair space:

- Subset = **all positives** + **sampled negatives** at a configurable
  negatives:positives ratio (default 5:1).
- Each candidate pair carries its **sampling stratum** (`positive` /
  `hard_negative` / `uniform_negative`) and the **actual inclusion probability
  `π_i`** for that stratum (the without-replacement draw rate within the
  stratum). Positives have `π_i = 1`.
- Reweighting uses **per-pair `1/π_i`** (Horvitz–Thompson), never a single
  global rate. A single global `π` is only valid under uniform single-stratum
  sampling and is explicitly not used.

### 3.1 Bias statement (precise)

Inverse-probability weighting makes the **linear** accumulators unbiased
estimators of their full-space values: edge count, density numerator, and the
degree scatter-add sums. The **nonlinear** graph losses — `degree_mmd`
(soft-histogram + Gaussian-TV kernel), `graph_similarity` (ratio of weighted
sums), and `clustering` (the `(A@A)*A` triangle term, cubic in edge variables) —
are computed *from* these weighted estimators and are therefore treated as
**consistent / low-bias approximations**, not strictly unbiased. The
approximation error is bounded empirically (see §8 validation check), not assumed
exact.

### 3.2 Data contract into the loss

The per-epoch subset is materialized as, per subgraph:
`(pair_index, stratum, inclusion_prob, target)`. `topology_plan_loss` consumes
`inclusion_prob` directly and never re-derives sampling rates. The linear
accumulators inside `src/topology/finetune_losses.py`
(`_pairwise_relative_density_loss`, `_pairwise_soft_degrees`,
`_pairwise_graph_similarity_loss`) are extended to accept an optional
per-pair weight `w_i = 1/π_i`; full-space normalizers (`n·(n−1)`) are unchanged.

## 4. Balanced / Budgeted Sampler

Stops the 200-node bucket from dominating memory and objective.

- **Per-size budget**: cap subgraphs-per-size and labeled-pairs-per-size.
  Coverage augmentation may no longer dump all uncovered positives into the
  maximum-size bucket; positive-edge coverage is distributed across sizes under
  the budget.
- **Fixed cached negative pool, per-epoch resampled subset**:
  - `pool_ratio: 10` (negatives per positive cached once per subgraph)
  - `epoch_ratio: 5` (negatives per positive drawn from the pool each epoch)
  - `hard_fraction: 0.5` — hard negatives ranked by cached scorer scores
  - `uniform_fraction: 0.5` — uniform negatives for distribution coverage
  - Only pool pairs are ever scored → bounded, fully cacheable score set.
- The sampler emits, per epoch and per pair, the stratum and `π_i` needed by §3.

## 5. Per-Size Loss Aggregation

Replace `mean(all subgraph losses)` with a size-balanced reduction:

```
loss_by_size[size] = mean(subgraph losses at that size)
total_loss         = mean over sizes of loss_by_size[size]
```

The 200-node bucket cannot dominate the objective regardless of subgraph count.
`S` = number of sizes with a positive global subgraph count.

## 6. Distributed Topology Step (Approach A: shard + reduce)

Shard the selected subgraphs by **global subgraph index** across ranks; each rank
computes a **chunked** backward over only its shard (peak memory becomes
independent of total pair count). Per-size normalization is global so the
objective is identical regardless of how subgraphs are sharded.

### 6.1 Gradient correctness (non-negotiables)

- `S` and `global_count_by_size[size]` are **global and identical on every
  rank**. A rank holding no subgraph of a given size contributes a true `0` to
  that size's summand.
- Only **detached** `loss_sum` / `count` are all-reduced — for logging and for
  the `global_count_by_size` normalizer. The backward'd loss is the rank-local
  **differentiable** sum scaled by the global normalizer. Never all-reduce a
  detached loss and then call backward on it (that severs the graph).

Per-rank differentiable loss:

```
for each size:
  local_loss_sum[size] = sum(differentiable subgraph losses on this rank)
  local_count[size]    = number of subgraphs of that size on this rank

all_reduce(global_count_by_size, SUM)   # detached, normalization only

rank_loss = mean over sizes of ( local_loss_sum[size] / global_count_by_size[size] )
```

### 6.2 Reduction mechanism (default = fork b)

The topology step currently runs on the **unwrapped** refiner
(`s2gae.py:806`), so DDP averaging hooks do not fire.

- **Default (fork b):** keep the unwrapped refiner, backward `rank_loss` as above,
  then **manual `dist.all_reduce(grad, SUM)`** over refiner params before
  `optimizer.step()`. No `world_size` factor. Smaller diff; avoids a second DDP
  reducer pass in the same iteration as the BCE backward.
- **Documented equivalent (fork a):** backward through the **DDP-wrapped** model
  so its hooks average gradients, and multiply `rank_loss` by `world_size` to
  cancel the averaging. Mathematically identical gradients.

The implementation uses fork (b) unless review prefers (a).

## 7. Setup / Scoring Cost

- **Score only the bounded pool** (all positives + fixed negative pool), not the
  12.79M full candidate space.
- **Batch embedding loads** to avoid per-endpoint `torch.load` thrash
  (`tccig/train.py:801`).
- **Reuse existing `scores/train_topology.pt` only if** scorer SHA,
  embedding-index SHA, pair hash, **and pair order** all validate against the new
  manifest; otherwise score the new bounded pool from scratch. The run must be
  correct with an empty/incompatible cache — reuse is a speed optimization, never
  a precondition. (Sampler/cache-key changes may legitimately change pool pair
  order and manifest, invalidating the old artifact; that is acceptable.)
- **Progress logging + upfront cost estimate**: log a pair-count and estimated
  scoring time before scoring starts, and periodic progress during scoring, so a
  long scoring phase is never silent.

## 8. Warmup Schedule

Pure-BCE warmup measurably worsened every topology metric (§1.4). Shorten warmup
so topology pressure engages early:

- Proposed: `warmup_epochs: 1`, `ramp_epochs: 5` (tunable; final values open for
  review).

## 9. Fast Smoke-Test Gate

A tiny config (1–2 node sizes, a few subgraphs, 2 epochs, topology scale forced
`> 0` at epoch 1) that exercises the real positive-scale topology backward and
the sharding path in minutes on the A40/L40 queue, **before** committing to the
full 40-epoch run. Guards against "16 h setup + 6 epochs then crash."

The smoke run also performs the **bias validation check**: on a held-out
subgraph, compare subset-estimated (IPW-reweighted) topology statistics against
the exact full-space statistics, and log the relative error per metric to bound
the §3.1 approximation empirically.

## 10. Components and Interfaces

| Component | Location (current) | Change |
|---|---|---|
| Topology plan / sampler | plan build in `tccig/train.py` + plan dataclasses | Add per-size budget, fixed pool + per-epoch resample, emit `(stratum, π_i)` |
| Pool scoring + cache | `tccig/train.py:283/373`, collate `:801` | Score pool only; batched embedding loads; SHA/hash/order-validated reuse; progress logging |
| `topology_plan_loss` | `tccig/s2gae.py:469` | Consume per-epoch subset + `inclusion_prob`; per-size aggregation; subgraph sharding + chunked backward |
| Linear accumulators | `src/topology/finetune_losses.py` | Optional per-pair weight `w_i = 1/π_i` on density/degree/GS sums |
| Distributed step | `tccig/s2gae.py:806–833` | Shard by global index; detached-only all-reduce of loss_sum/count; manual grad all-reduce (fork b) |
| Schedule | config `topology_training.schedule` | `warmup_epochs: 1`, `ramp_epochs: 5` |
| Smoke config | `configs/` + a small run step | New tiny config + bias validation check |

## 11. Error Handling

- Reject a config where any size's per-size budget yields zero subgraphs while
  `enabled=true` (would make `S` inconsistent across ranks).
- Validate `inclusion_prob ∈ (0, 1]` and that every positive has `π_i = 1`.
- On cache validation failure, log the mismatching field (SHA / pair hash / pair
  order) and fall back to scoring rather than silently using a stale artifact.
- Preserve the existing `_pair_lookup` "pair absent from split graph" guard.

## 12. Testing

- **Unit (loss):** IPW-weighted linear accumulators recover full-space density /
  degree sums on a small graph (within sampling tolerance over repeated draws).
- **Unit (aggregation):** per-size mean is invariant to subgraph count within a
  size; a dominating bucket does not shift `total_loss`.
- **Unit (distributed math):** fork (b) rank-local scaled loss + manual SUM
  all-reduce produces gradients equal (to tolerance) to a single-process
  full-plan reference on a 2-rank gloo/CPU test.
- **Unit (sampler):** emitted `π_i` matches realized inclusion frequencies per
  stratum over many draws; positives always `π_i = 1`.
- **Integration:** smoke config runs a positive-scale topology step end-to-end
  without OOM and logs the bias validation error.
- Target ≥80% coverage on touched modules per repo guidelines.

## 13. Open Items for Review

- Distributed reduction: fork (b) default vs fork (a).
- Warmup numbers (`warmup_epochs`, `ramp_epochs`).
- Default ratios (`pool_ratio`, `epoch_ratio`, `hard_fraction`) and per-size
  budget values.
