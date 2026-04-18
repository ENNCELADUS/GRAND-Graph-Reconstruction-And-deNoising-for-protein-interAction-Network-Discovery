# Topology Finetune: Current Warmup Issues

## Scope

This note tracks the current topology-finetune warmup problems observed in:

- `logs/v3/topology_finetune/20260418_221309`
- `logs/v3/slurm_891873.err`

The older loss-scale plateau from `logs/v3/topology_finetune/20260416_214619`
is mostly addressed by the existing RD/log-ratio, topology normalization,
sparse-prior bias, and grouped GradNorm changes. The active problem is now that
the warmup stage is BCE-only in loss terms, but it still uses the expensive
topology subgraph forward path and does not actually train on the intended
ratio-5 supervised binary signal.

## Current Evidence

Epoch 1 from `20260418_221309`:

| Metric | Value |
|---|---:|
| `Topology Loss Scale` | `0.0` |
| `Planned Subgraphs` | `1540` |
| `Covered Positive Edges` | `48771` |
| `Positive Edge Coverage Ratio` | `1.0` |
| `Train Forward Backward S` | `7622.45` |
| `Val Pair Pass S` | `181.89` |
| `Internal Val Topology S` | `2289.16` |
| `Epoch Time` | `10698.54` |
| `Val AUPRC` | `0.5077` |

Recomputed epoch-plan counts from `configs/v3.yaml`:

| Count | Value |
|---|---:|
| All forwarded subgraph pairs | `2,685,799` |
| BCE-supervised pairs | `58,359` |
| BCE positives | `48,771` |
| BCE negatives | `9,588` |
| BCE-supervised fraction | `2.17%` |
| Forward chunks at `pair_batch_size=32` | `84,884` |

For comparison, normal pairwise training in `logs/v3/train/20260308_002906`
ran epoch 1 in about `705s` and reached validation AUPRC `0.7177`.

## Root Causes

### 1. Warmup skips topology loss, not topology work

The code computes `skip_topology_during_warmup`, but only checks it after
`_concat_logits_and_pairs(...)` has enumerated and forwarded every upper-triangle
pair in the sampled subgraph. With 30-60 node subgraphs, this means the warmup
epoch forwards millions of pairs even though only a small masked subset
contributes to BCE.

Impact:

- warmup runtime is dominated by topology-style all-pairs inference
- most model forward compute has `bce_mask=0`
- speed cannot resemble normal binary classification training

### 2. Ratio-5 negatives collapse under subgraph-local filtering

The ratio-5 train file contains:

| Label | Count |
|---|---:|
| positive | `48,771` |
| negative | `243,855` |

The current assignment only uses explicit negatives whose endpoints already fall
inside each sampled topology subgraph, and each selected negative can be used at
most once. In the current plan this yields only `9,588` negatives.

Impact:

- intended supervised ratio: `1 positive : 5 negatives`
- observed supervised ratio: about `5 positives : 1 negative`
- scratch warmup does not train the classifier under the same distribution that
  the ratio-5 supervision file was built to provide

### 3. Scratch bias uses graph density, while warmup sees supervised pairs

Scratch mode initializes the output bias from the full train-graph density:

- logged positive edge probability: `0.0015`
- logged bias: `-6.5024`

That prior is useful for avoiding dense topology predictions, but it is a poor
starting point for BCE warmup when the warmup objective is supervised pair
classification. After fixing ratio-5 negatives, the supervised positive rate is:

```text
1 / (1 + bce_negative_ratio) = 1 / 6 = 0.1667
logit(0.1667) ~= -1.609
```

If the ratio is changed to 1:4, the positive rate would be `0.2` and the bias
would be `-1.386`. For the current `bce_negative_ratio: 5`, use the actual
supervised ratio or the ratio-5 file counts, not the graph-density prior, for
the warmup classifier bias.

### 4. Topology labels are coupled to BCE assignment

When `assigned_positive_edges` is provided, `_subgraph_pair_tuples(...)` labels
only the assigned chunk positives as topology positives. True graph edges in the
expanded subgraph that were not assigned to this chunk become topology zeros.

This violates the design contract in `baseline.md`:

- topology labels should represent the ground-truth induced subgraph
- BCE labels and masks should represent explicit supervised pairs only

Impact:

- topology losses can optimize against an incomplete or incorrect target graph
- BCE assignment details leak into the topology objective
- post-warmup topology gradients can conflict with the actual graph structure

### 5. Internal topology validation runs during pure warmup

Epoch 1 spent `2289.16s` in internal topology validation even though topology
loss scale was `0.0`. Pair validation is still useful during warmup; full
topology validation is expensive and mostly diagnostic until topology loss turns
on.

## Suggested Fix Design

### Fix 1: Add a supervised-pair warmup path

During `topology_loss_scale == 0`, do not call the all-pairs subgraph forward
path. Build and forward only the assigned supervised pairs:

- assigned positive edges
- assigned explicit negative edges
- no adjacency construction
- no topology labels needed for loss

Expected pair/chunk reduction:

| Path | Forwarded Pairs | Chunks at 32 |
|---|---:|---:|
| current all-pairs warmup | `2,685,799` | `84,884` |
| current masked supervised pairs only | `58,359` | `1,824` |
| fixed ratio-5 supervised pairs only | `292,626` | `9,145` |

The fixed ratio-5 path does more useful BCE work than the current masked path,
but still avoids roughly 90% of the current all-pairs chunks.

### Fix 2: Assign BCE negatives globally

Replace subgraph-local negative filtering with deterministic global assignment:

- shuffle all explicit train negatives once per epoch
- assign `positive_count * bce_negative_ratio` negatives per subgraph or task
- do not reuse negatives within an epoch unless the dataset is exhausted
- log the desired and actual negative counts

Important design note:

Global negatives do not need to be endpoints in the topology subgraph if BCE is
computed through a separate supervised-pair forward path. This is the preferred
design. Injecting all negative endpoints into the topology node set can explode
subgraph size and make topology losses much more expensive.

Use endpoint injection only as a bounded fallback, with a hard cap and metrics
for actual node growth.

### Fix 3: Decouple topology loss pairs from BCE pairs

Use two related but separate forward surfaces:

1. Topology surface:
   - all upper-triangle pairs inside the sampled topology node set
   - topology label is always `graph.has_edge(a, b)`
   - used only when topology loss scale is greater than zero

2. BCE surface:
   - explicitly assigned positive and negative pairs
   - endpoints may be outside the topology node set
   - label is based on assignment, not induced-subgraph membership

For the existing all-pairs helper, change labels so topology always uses ground
truth and BCE uses assignment:

```python
is_true_edge = graph.has_edge(protein_a, protein_b)
topology_label = 1.0 if is_true_edge else 0.0

is_assigned_positive = (
    assigned_positive_edges is not None and pair in assigned_positive_edges
)
is_assigned_negative = (
    assigned_negative_edges is not None and pair in assigned_negative_edges
)

if assigned_positive_edges is not None or assigned_negative_edges is not None:
    bce_label = 1.0 if is_assigned_positive else 0.0
    bce_mask = 1.0 if is_assigned_positive or is_assigned_negative else 0.0
else:
    bce_label = topology_label
    bce_mask = 1.0
```

### Fix 4: Initialize warmup bias from supervised ratio

For scratch warmup, initialize the classifier output bias from the actual
supervised BCE ratio, not graph density:

```python
positive_rate = positives / max(1, positives + negatives)
bias = logit(clamp(positive_rate))
```

Preferred source:

1. Actual positive/negative counts from the supervision file or first epoch plan.
2. Fallback to `1 / (1 + bce_negative_ratio)` when counts are not available.
3. Fallback to graph density only when topology starts immediately without BCE
   warmup.

Log both the source and the resulting bias.

### Fix 5: Reduce warmup validation cost

During pure warmup:

- keep pairwise validation every epoch
- skip internal topology validation, or run it only every N epochs
- always run internal topology validation on the first epoch where topology loss
  scale becomes nonzero

This preserves AUPRC feedback while avoiding multi-thousand-second topology
validation passes before topology optimization starts.

## Implementation Checklist

Status: implemented in this branch. The checklist below is covered by targeted
unit/integration tests plus the full pytest suite and coverage run.

### Data planning

- [x] Add epoch-plan diagnostics: total all-pairs, supervised pairs, positive
      count, negative count, target negative count, actual negative ratio, and
      projected chunk counts.
- [x] Replace subgraph-local negative assignment with deterministic global
      assignment for BCE negatives.
- [x] Verify global assignment consumes available train negatives without
      duplicate negatives in one epoch, up to the configured target ratio.
- [x] Log a warning when there are not enough negatives to satisfy
      `bce_negative_ratio`.

### Forward paths

- [x] Add an iterator/helper for arbitrary supervised pair chunks using
      `EmbeddingRepository`.
- [x] During `topology_loss_scale == 0`, forward only supervised pairs.
- [x] During topology epochs, keep topology all-pairs forward separate from BCE
      supervised-pair forward, or explicitly merge their losses without requiring
      BCE negatives to be in the topology node set.
- [x] Avoid injecting all negative endpoints into topology node sets by default.

### Labels and losses

- [x] Change topology labels to always use `graph.has_edge(a, b)`.
- [x] Change BCE labels and masks to depend only on assigned supervised pairs.
- [x] Keep topology losses zero during warmup and skip adjacency construction on
      the supervised-only warmup path.
- [x] Initialize scratch warmup bias from supervised positive rate.
- [x] Log bias source, positive rate, and bias value.

### Validation

- [x] Keep pair validation every epoch.
- [x] Add config for internal topology validation cadence during warmup.
- [x] Skip or decimate internal topology validation while topology loss scale is
      zero.
- [x] Force internal topology validation when topology loss first becomes
      nonzero.

### Tests

- [x] Unit test that global negative assignment reaches the expected ratio on a
      synthetic graph.
- [x] Unit test that out-of-subgraph negatives are still included in supervised
      BCE chunks.
- [x] Unit test that warmup forwards only supervised pairs.
- [x] Unit test that topology labels use ground-truth graph edges independent of
      assignment.
- [x] Integration test that a tiny topology warmup logs warmup diagnostics and
      verifies supervised pair counts match positive plus negative BCE counts.
- [x] Regression test that internal topology validation can be skipped during
      warmup without breaking checkpointing or early stopping.

## Verification Plan

1. Run a one-epoch warmup smoke test and verify logs show:
   - topology loss scale `0.0`
   - forwarded BCE pairs close to `292,626` for full ratio-5 Human/BFS
   - actual negative ratio close to `5.0`
   - no full all-pairs topology forward during warmup
2. Compare epoch runtime against normal training. Expected target is roughly
   normal pairwise training scale, not the current `~10,699s`.
3. Confirm validation AUPRC moves materially above the random/poor baseline
   instead of staying near `0.50`.
4. Run the first topology-enabled epoch and verify:
   - topology labels reflect the induced ground-truth graph
   - `internal_val_relative_density` trends down
   - `internal_val_graph_sim` and pair AUPRC do not regress sharply

## Success Criteria

- Warmup uses the intended ratio-5 supervised BCE signal.
- Warmup runtime is dominated by useful supervised pair forwards, not masked
  all-pairs topology forwards.
- Validation AUPRC improves normally during warmup.
- Topology loss sees the true induced graph after warmup.
- Internal topology validation no longer consumes large warmup time unless
  explicitly requested.
