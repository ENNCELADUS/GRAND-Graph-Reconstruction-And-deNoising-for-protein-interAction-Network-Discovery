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
`relative_density`, `degree_mmd`; `clustering` **default off** for this re-run,
see §3.3) but evaluate them on a per-epoch **labeled subset** of each subgraph
rather than the full `n·(n−1)` pair space:

- Subset = **all positives** + **sampled negatives** at a configurable
  negatives:positives ratio (default 5:1).
- Each candidate pair carries its **sampling stratum** (`positive` /
  `hard_negative` / `uniform_negative`) and the **actual two-stage inclusion
  probability `π_i`** (defined precisely in §3.3). Positives have `π_i = 1`.
- Reweighting uses **per-pair `1/π_i`** (Horvitz–Thompson), never a single
  global rate. A single global `π` is only valid under uniform single-stratum
  sampling and is explicitly not used.

### 3.3 Two-stage inclusion probability (sampling is conditional on a pool)

Negatives are drawn in two stages — a fixed cached **pool** built once per
subgraph (§4), then a per-epoch **subset** drawn from that pool. The estimator is
only a full-space estimator if **both** stages carry a recorded inclusion
probability. The per-pair total is the product:

```
π_i = π_pool(i) · π_epoch|pool(i)
```

To make `π_pool` and `π_epoch|pool` both well-defined and computable, **hard
negatives use a stochastic hard-stratum** rather than a deterministic top-k:

- Define a **hard stratum** per subgraph = the top fraction of negatives by
  cached scorer score (e.g. top 20%, configurable by fraction or score
  threshold). The uniform stratum is the remaining negatives (the full negative
  frame minus positives).
- **Pool stage** draws without replacement *within each stratum*, so every pooled
  negative has a known `π_pool` = (pool draws in its stratum) / (stratum size).
- **Epoch stage** draws without replacement from the pooled members of each
  stratum, giving a known `π_epoch|pool` = (epoch draws in its stratum) / (pooled
  members of its stratum).
- Per-epoch negative mix: **50% stochastic hard-stratum + 50% uniform random**.

A deterministic top-k hard set has `π_pool ∈ {0,1}` and yields an objective
*conditional on that fixed hard set*, not a full-space estimator — so this re-run
does **not** use deterministic top-k. If a future experiment wants deterministic
hard mining, it must be labelled a biased hard-negative objective, not an IPW
full-space estimator.

### 3.4 Bias statement (precise)

Inverse-probability weighting (with the two-stage `π_i` of §3.3) makes the
**linear** accumulators unbiased estimators of their full-space values: edge
count, density numerator, and the degree scatter-add sums. The **nonlinear**
graph losses — `degree_mmd` (soft-histogram + Gaussian-TV kernel),
`graph_similarity` (ratio of weighted sums), and `clustering` (the `(A@A)*A`
triangle term, cubic in edge variables) — are computed *from* these weighted
estimators and are therefore treated as **consistent / low-bias approximations**,
not strictly unbiased. The approximation error is checked, not assumed exact (see
§3.7 production diagnostic and §9 smoke check).

### 3.5 Clustering disabled for this re-run

`clustering` (`include_clustering_mmd`) is **default off** for Experiment 02. It
is the most nonlinear (cubic) of the terms, the furthest from IPW-unbiased under
subsetting, and the most memory/compute heavy (builds dense adjacency). Re-enable
only after the density/degree/GS path is validated.

### 3.6 Data contract into the loss

The per-epoch subset is materialized per subgraph with enough metadata for
correct resampling, `π_i` auditing, and cache validation — not just
`(pair_index, inclusion_prob, target)`. Each pair record carries:

- `pair_id` — stable global pair identifier (the canonicalized undirected
  endpoint code), independent of subgraph membership.
- `subgraph_id` and `node_size` — a pair may appear in multiple subgraphs; its
  `π_i` and target are per-`(subgraph_id, pair_id)`, not global.
- `stratum` — `positive` / `hard_negative` / `uniform_negative`.
- `pi_pool`, `pi_epoch_given_pool`, `pi_total` — the two-stage probabilities of
  §3.3, stored separately so each stage is auditable; `pi_total` is what the loss
  consumes as `1/π_i`.
- `local_index_a`, `local_index_b` — subgraph-local node indices for the degree
  scatter-add and adjacency build.
- `target` — 1.0 if the ground-truth subgraph has the edge, else 0.0.

`topology_plan_loss` consumes `pi_total` directly and never re-derives sampling
rates. The linear accumulators inside `src/topology/finetune_losses.py`
(`_pairwise_relative_density_loss`, `_pairwise_soft_degrees`,
`_pairwise_graph_similarity_loss`) are extended to accept an optional per-pair
weight `w_i = 1/pi_total`; full-space normalizers (`n·(n−1)`) are unchanged.

### 3.7 Production bias diagnostic

The §9 smoke check catches wiring errors but does **not** bound bias for the
production 20–200-node distribution or the hard/uniform mixture — a single tiny
held-out subgraph is not representative. Therefore, during the real run,
periodically (every `bias_diagnostic_every_n_epochs`, e.g. 5) pick a few **capped**
validation subgraphs (small enough that the full `n·(n−1)` space is affordable),
compute both the IPW-reweighted subset statistics and the exact full-space
statistics, and log the per-metric relative error. This is a running diagnostic
of the §3.4 approximation under the actual size mixture, not a one-off check.

## 4. Balanced / Budgeted Sampler

Stops the 200-node bucket from dominating memory and objective.

- **Per-size budget**: cap subgraphs-per-size and labeled-pairs-per-size.
  Coverage augmentation may no longer dump all uncovered positives into the
  maximum-size bucket; positive-edge coverage is distributed across sizes under
  the budget.
- **Fixed cached negative pool, per-epoch resampled subset** (two-stage, §3.3):
  - `pool_ratio: 10` (negatives per positive cached once per subgraph)
  - `epoch_ratio: 5` (negatives per positive drawn from the pool each epoch)
  - Per-epoch negative mix: **50% stochastic hard-stratum + 50% uniform random**.
  - **Hard stratum** = top fraction of negatives by cached scorer score
    (`hard_stratum_fraction`, e.g. 0.2), sampled **stochastically** (not
    deterministic top-k) so `π_pool` and `π_epoch|pool` are computable per §3.3.
  - **Uniform stratum** = the remaining negatives, for distribution coverage.
  - Both pool and epoch draws are **without replacement within each stratum**, so
    every negative's two-stage `π_i` is a known ratio.
  - Only pool pairs are ever scored → bounded, fully cacheable score set.
- The sampler emits, per epoch and per pair, the full record of §3.6
  (stratum, `pi_pool`, `pi_epoch_given_pool`, `pi_total`, ids, local indices).

## 5. Per-Size Loss Aggregation

Replace `mean(all subgraph losses)` with a size-balanced reduction:

```
loss_by_size[size] = mean(subgraph losses at that size)
total_loss         = mean over sizes of loss_by_size[size]
```

The 200-node bucket cannot dominate the objective regardless of subgraph count.

**Eligible sizes.** `S` = the number of **globally active** sizes, i.e. sizes that
survive graph-size and budget filtering with a positive global subgraph count —
**not** every configured `node_size`. A configured size that yields zero
subgraphs (graph too small, or zero budget) is dropped from the size set and
logged as skipped; it does not contribute a phantom `0` summand and does not abort
the run. `S` and the active-size set are global and identical on every rank (§6.1).

## 6. Distributed Topology Step (Approach A: shard + reduce)

Shard the selected subgraphs by **global subgraph index** across ranks; each rank
computes a **per-chunk backward** over only its shard (peak memory becomes
independent of both total pair count *and* shard size). Per-size normalization is
global so the objective is identical regardless of how subgraphs are sharded.

### 6.0 Per-chunk backward (memory non-negotiable)

Sharding alone removes the 4× replication but **not** the per-shard accumulation:
if a rank builds `local_loss_sum` over its whole shard and backprops once, it
still holds every subgraph's autograd graph until the end, so memory scales with
shard size. Required pattern instead:

```
for chunk in this_rank_shard:                 # chunk = one subgraph (or a small group)
    chunk_loss = topology_loss(chunk)          # differentiable, this chunk only
    scaled = chunk_loss / global_count_by_size[size(chunk)] / S
    backward(scaled)                           # immediately; grads accumulate in .grad
    detached_loss_sum[size] += chunk_loss.detach()   # logging only
    free chunk autograd graph
```

Because gradient contributions are **additive**, per-chunk backward of
`chunk_loss / global_count / S` accumulates into `.grad` exactly the gradient of
the global per-size-mean objective — identical to one backward over the summed
loss, but with peak memory bounded by a single chunk. `global_count_by_size` and
`S` are obtained by a detached all-reduce **before** the backward loop.

### 6.1 Gradient correctness (non-negotiables)

- `S` and `global_count_by_size[size]` are **global and identical on every
  rank**. A rank holding no subgraph of a given size contributes a true `0` to
  that size's summand.
- Only **detached** `loss_sum` / `count` are all-reduced — for logging and for
  the `global_count_by_size` normalizer. The backward'd loss is each chunk's
  rank-local **differentiable** loss scaled by the global normalizer. Never
  all-reduce a detached loss and then call backward on it (that severs the graph).

The step is **two-pass** so the normalizer is known before any backward:

```
# Pass 1 — counts only, no autograd graph retained
for each size:
  local_count[size] = number of subgraphs of that size in this rank's shard
all_reduce(global_count_by_size, SUM)        # detached, normalization only
S = number of sizes with global_count_by_size[size] > 0   # identical on every rank

# Pass 2 — per-chunk differentiable backward (see §6 pattern)
for chunk in this_rank_shard:
  scaled = topology_loss(chunk) / global_count_by_size[size(chunk)] / S
  backward(scaled)                            # immediately; .grad accumulates additively
  detached_loss_sum[size(chunk)] += scaled.detach()   # logging only; free chunk graph
```

Summing `chunk / global_count / S` across all chunks on all ranks reconstructs
`mean_over_sizes( mean_over_subgraphs(loss) )` exactly, so per-chunk backward is
gradient-identical to one backward over the full global objective.

### 6.2 Reduction mechanism (default = fork b)

The topology step currently runs on the **unwrapped** refiner
(`s2gae.py:806`), so DDP averaging hooks do not fire.

- **Default (fork b):** keep the unwrapped refiner, run the §6.1 per-chunk
  backward loop (grads accumulate in `.grad`), then **once after the loop** do a
  manual `dist.all_reduce(grad, SUM)` over refiner params before
  `optimizer.step()`. No `world_size` factor. Smaller diff; avoids a second DDP
  reducer pass in the same iteration as the BCE backward.
- **Documented equivalent (fork a):** run the per-chunk backward through the
  **DDP-wrapped** model so its hooks average gradients, and multiply each chunk's
  `scaled` loss by `world_size` to cancel the averaging. Mathematically identical
  gradients.

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

The smoke run also performs an **implementation sanity check**: on a held-out
subgraph, compare subset-estimated (IPW-reweighted) topology statistics against
the exact full-space statistics and log the per-metric relative error. This
catches wiring/reweighting bugs (wrong `π_i`, missing weight, normalizer
mismatch). It does **not** bound bias for the production 20–200-node distribution
or the hard/uniform mixture — that is the job of the §3.7 production diagnostic,
which runs on capped validation subgraphs across the real size mixture during
training.

## 10. Components and Interfaces

| Component | Location (current) | Change |
|---|---|---|
| Topology plan / sampler | plan build in `tccig/train.py` + plan dataclasses | Per-size budget; fixed pool + per-epoch resample; stochastic hard-stratum (50/50); emit full §3.6 record (`stratum`, `pi_pool`, `pi_epoch_given_pool`, `pi_total`, ids, local indices) |
| Pool scoring + cache | `tccig/train.py:283/373`, collate `:801` | Score pool only; batched embedding loads; SHA/hash/order-validated reuse; progress logging |
| `topology_plan_loss` | `tccig/s2gae.py:469` | Consume per-epoch subset + `pi_total`; per-size aggregation over eligible sizes; subgraph sharding + per-chunk backward; `clustering` off |
| Linear accumulators | `src/topology/finetune_losses.py` | Optional per-pair weight `w_i = 1/pi_total` on density/degree/GS sums |
| Distributed step | `tccig/s2gae.py:806–833` | Shard by global index; detached-only all-reduce of loss_sum/count; per-chunk backward; manual grad all-reduce (fork b) |
| Schedule | config `topology_training.schedule` | `warmup_epochs: 1`, `ramp_epochs: 5` |
| Smoke config + diagnostic | `configs/` + run step | Tiny smoke config (sanity check) + periodic production bias diagnostic (§3.7) |

## 11. Error Handling

- **Eligible-size filtering, not rejection.** Compute the eligible size set after
  graph-size and budget filtering (§5). Drop and **log** any configured size with
  zero subgraphs; abort only if **no** eligible size remains while
  `topology_training.enabled=true`. The eligible set and `S` are derived
  identically on every rank, so a dropped size cannot desynchronize ranks.
- Validate `pi_total ∈ (0, 1]`, `pi_pool ∈ (0,1]`, `pi_epoch_given_pool ∈ (0,1]`,
  `pi_total == pi_pool * pi_epoch_given_pool` (to tolerance), and that every
  positive has `pi_total = 1`.
- On cache validation failure, log the mismatching field (SHA / pair hash / pair
  order) and fall back to scoring rather than silently using a stale artifact.
- Preserve the existing `_pair_lookup` "pair absent from split graph" guard.

## 12. Testing

- **Unit (loss):** IPW-weighted linear accumulators recover full-space density /
  degree sums on a small graph (within sampling tolerance over repeated draws).
- **Unit (aggregation):** per-size mean is invariant to subgraph count within a
  size; a dominating bucket does not shift `total_loss`; eligible-size filtering
  excludes zero-budget sizes consistently across ranks.
- **Unit (distributed math):** fork (b) per-chunk backward + manual SUM
  all-reduce produces gradients equal (to tolerance) to a single-process
  full-plan reference on a 2-rank gloo/CPU test.
- **Unit (sampler):** realized two-stage inclusion frequency per stratum matches
  the recorded `pi_pool · pi_epoch_given_pool` over many draws; positives always
  `pi_total = 1`; `pi_total = pi_pool · pi_epoch_given_pool` holds per record.
- **Integration:** smoke config runs a positive-scale topology step end-to-end
  without OOM and logs the bias sanity check (§9).
- Target ≥80% coverage on touched modules per repo guidelines.

## 13. Decisions and Open Items

**Decided in review**
- Objective: hybrid subset + two-stage IPW reweighting (§3).
- Negatives: fixed cached pool + per-epoch resample, **stochastic hard-stratum
  (50%) + uniform (50%)** — no deterministic top-k (§3.3, §4).
- `clustering` **disabled** for this re-run (§3.5).
- Distributed reduction defaults to **fork (b)** (§6.2).
- Per-size aggregation over **eligible** sizes only; zero-budget sizes dropped and
  logged, not rejected (§5, §11).

**Open for review (tunable, non-blocking)**
- Warmup numbers (`warmup_epochs`, `ramp_epochs`) — proposed 1 / 5.
- `hard_stratum_fraction` (proposed 0.2) and per-size budget caps.
- Default ratios (`pool_ratio: 10`, `epoch_ratio: 5`, neg:pos 5:1).
- `bias_diagnostic_every_n_epochs` (proposed 5) and the capped-subgraph size for
  the §3.7 diagnostic.
- Whether to confirm fork (a) instead of (b).
