# Topology Finetune: Baseline Training Algorithm

## 1. Problem Statement

Train a pairwise PPI scorer so that its predicted interaction graph matches the ground-truth graph not just at the pair level (BCE), but also at the graph-topology level (GS, RD, degree MMD, clustering MMD). The PRING Human/BFS training split contains:

| Quantity | Count |
|---|---|
| Train proteins (nodes) | 8,071 |
| Positive edges (train graph) | 60,972 |
| Explicit negative pairs (ratio-5) | 243,855 |
| Connected components | 640 (largest: 7,347 nodes) |
| Average node degree | 15.1 |

The challenge: topology losses require a subgraph context (adjacency matrix, degree distribution, clustering coefficients), but the model is a pairwise scorer that takes two protein embeddings and outputs a single logit. We must bridge the gap between subgraph-level topology objectives and pair-level forward passes.

## 2. Design Requirements

1. Every positive edge must appear exactly once per epoch (no duplication, full coverage).
2. Explicit negative pairs from the dataset must participate in BCE supervision.
3. Each sampled subgraph must support both BCE loss and topology losses jointly.
4. Under DDP, each rank processes a disjoint partition of the work to achieve real speedup.

## 3. Proposed Algorithm

### 3.1 Epoch Plan: Shuffle-and-Chunk Edge Partition

Instead of sampling subgraphs and hoping they cover edges, we partition edges first and build subgraphs around them.

```
SHUFFLE_CHUNK_EPOCH(graph G, chunk_size C, subgraph_size k, seed s):
    rng = Random(s)
    all_positive_edges = list(G.edges())
    rng.shuffle(all_positive_edges)
    # Partition edges into chunks of size C
    edge_chunks = partition(all_positive_edges, C)

    subgraphs = []
    for chunk in edge_chunks:
        # Collect all nodes touched by this edge chunk
        core_nodes = set()
        for (u, v) in chunk:
            core_nodes.add(u)
            core_nodes.add(v)

        # Expand to target subgraph size via BFS from core_nodes
        if len(core_nodes) <= k:
            expanded = bfs_expand(G, core_nodes, target=k, rng=rng)
            subgraphs.append(expanded)
        else:
            # Core too large: use core_nodes directly (variable-size subgraph)
            subgraphs.append(core_nodes)

    return subgraphs
```

Key properties:
- Every positive edge appears in exactly one chunk, therefore exactly one subgraph.
- No edge duplication across subgraphs within an epoch.
- The BFS expansion ensures subgraphs are locally connected (good for topology metrics).
- Chunk size C controls the tradeoff: smaller C = more subgraphs with fewer core edges each, larger C = fewer subgraphs with denser cores.

Recommended default: `C = k * (k-1) / 4` (roughly half the max possible edges in a k-node subgraph), so each subgraph has a meaningful edge density for topology loss computation.

### 3.2 Negative Pair Inclusion for BCE

For each subgraph with node set V_s:

1. All C(|V_s|, 2) pairs are enumerated (upper triangle).
2. Pairs present in G.edges are positive (topology label = 1, BCE label = 1, BCE mask = 1).
3. Pairs present in the explicit negative set are negative (topology label = 0, BCE label = 0, BCE mask = 1).
4. Remaining pairs: topology label = 0, BCE mask = 0 (no BCE supervision, but they still contribute to topology losses via the adjacency matrix).

This means:
- Topology losses see the full subgraph structure (all pairs contribute to adjacency).
- BCE loss only fires on supervised pairs (positives + explicit negatives that happen to fall within the subgraph).
- No random negative sampling needed -- the explicit negatives from the dataset are used directly.

### 3.3 Joint Loss Computation Per Subgraph

```
For each subgraph S with nodes V_s:
    # Forward pass: score all C(|V_s|, 2) pairs
    logits = model(all_pairs_in_V_s)
    probs = sigmoid(logits)

    # BCE loss (masked)
    bce = masked_bce(logits, bce_labels, bce_mask)

    # Topology losses (use all pairs for adjacency)
    A_pred = scatter_to_adjacency(probs, pair_indices, |V_s|)
    A_true = scatter_to_adjacency(topo_labels, pair_indices, |V_s|)
    topo_loss = alpha * GS(A_pred, A_true)
                + beta * RD(A_pred, A_true)
                + gamma * DegMMD(A_pred, A_true)
                + delta * ClusMMD(A_pred, A_true)

    total_loss = bce + topo_loss
    total_loss.backward()  # accumulate gradients
    # optimizer.step() every N subgraphs (gradient accumulation)
```

### 3.4 DDP Strategy: Partition Subgraphs Across Ranks

The correct DDP approach for this custom dataloader:

```
DISTRIBUTED_EPOCH(subgraphs, world_size, rank):
    # All ranks generate the same epoch plan (same seed)
    all_subgraphs = generate_epoch_plan(seed=epoch_seed)

    # Each rank takes a disjoint slice
    my_subgraphs = all_subgraphs[rank::world_size]

    # Each rank trains on its slice
    for sg in my_subgraphs:
        loss = compute_loss(sg)
        loss.backward()
        # DDP automatically all-reduces gradients at optimizer.step()
        optimizer.step()
```

This gives linear speedup with world_size. The gradient all-reduce in DDP ensures all ranks converge to the same model, while each rank processes 1/world_size of the subgraphs.

Important: the epoch plan must be identical across ranks (same seed), but each rank only processes its assigned slice. This is the standard DDP convention used by DistributedSampler.

## 4. Analysis of Current Implementation

### 4.1 Critical Issues

#### Issue 1: DDP Does Not Parallelize Data (Severity: HIGH)

Location: `topology_finetune.py:179-193` (`_resolve_epoch_seed`) and `topology_finetune.py:873` (the subgraph loop in `_fit_epoch`).

All ranks receive the same seed under DDP, and every rank iterates over ALL subgraphs in the epoch plan. Result:

- 4 GPUs = 4x the compute for the same work, zero data-parallel speedup.
- The only thing DDP does is average gradients across identical computations.

The comment says "all ranks need the same epoch plan so they execute the same number of forward and backward steps before the next collective." This is correct about needing the same plan, but wrong about the conclusion. The standard pattern is: same plan, each rank processes `plan[rank::world_size]`.

#### Issue 2: Edge-Cover Sampling Can Run Indefinitely (Severity: HIGH)

Location: `finetune_data.py:779`.

```python
while len(sampled_subgraphs) < num_subgraphs or uncovered_edges:
```

The `or uncovered_edges` clause forces the loop to continue until ALL 60,972 positive edges are covered, regardless of `num_subgraphs`. With 20-node subgraphs averaging ~62 edges each, full coverage requires ~983+ subgraphs minimum. The greedy node selection (`_pick_next_node`) scans all 8,071 graph nodes per node addition, making this O(983 * 20 * 8071) ~ 159M operations per epoch.

Empirically confirmed: `sample_edge_cover_subgraphs(num_subgraphs=100, ...)` did not complete within 2 minutes on the Human/BFS graph.

#### Issue 3: Positive Edges Are Duplicated Across Subgraphs (Severity: MEDIUM)

The edge-cover algorithm seeds each subgraph from an uncovered edge, then expands greedily. But the induced subgraph of the selected nodes contains ALL edges between those nodes, not just the seed edge. Edges in dense neighborhoods appear in multiple subgraphs. The `mean_positive_edge_reuse` metric tracks this but does not prevent it.

The requirement "one epoch should use all edges exactly once" is violated.

### 4.2 Efficiency Issues

#### Issue 4: No Gradient Accumulation Across Subgraphs

Location: `topology_finetune.py:875-925`.

Each subgraph triggers its own `optimizer.zero_grad()` + `backward()` + `optimizer.step()` cycle. With ~1000 subgraphs per epoch, that is 1000 optimizer steps, each based on a single 20-node subgraph (~190 pairs). This is noisy and has high optimizer overhead.

Fix: accumulate gradients over N subgraphs before stepping.

#### Issue 5: Sequential Subgraph Processing, GPU Underutilization

Subgraphs are processed one at a time in a Python for-loop. Each subgraph only has ~190 pairs with pair_batch_size=16, so the GPU processes 16 pairs at a time. Modern GPUs can handle thousands of pairs per batch.

Fix: batch multiple subgraphs into a single forward pass by concatenating their pair tensors.

#### Issue 6: Fixed 20-Node Subgraphs

With `min_nodes=max_nodes=20`, high-degree hub nodes (degree up to 190) cannot have all their edges covered in a single subgraph. The algorithm needs many overlapping subgraphs to cover hub neighborhoods.

Fix: variable subgraph sizes [30, 60] to adapt to local graph density.

### 4.3 Correctness Concerns

#### Issue 7: BCE Negative Sampling Is Per-Subgraph, Not Per-Epoch

`_subgraph_pair_tuples()` samples negatives independently per subgraph with `negative_ratio`. The same explicit negative pair can be sampled in multiple subgraphs (if both endpoints appear), leading to uneven negative supervision.

#### Issue 8: Topology Losses on 20-Node Subgraphs May Be Noisy

Degree distributions and clustering coefficients computed on 20 nodes have high variance. The topology loss signal may be dominated by sampling noise. Internal validation uses subgraphs up to 200 nodes, but training only uses 20 -- this mismatch means the model optimizes for small-subgraph topology that may not transfer to larger-scale structure.

## 5. Recommended Baseline Configuration

```yaml
topology_finetune:
  epochs: 30
  subgraph_node_range: [30, 60]       # Variable size, larger than current 20
  edge_chunk_size: 200                 # Edges per subgraph core
  gradient_accumulation_steps: 8       # Accumulate over 8 subgraphs
  pair_batch_size: 64                  # Larger batch for GPU utilization
  strategy: "mixed"
  decision_threshold:
    mode: best_f1_on_valid
  monitor_metric: "internal_val_graph_sim"
  early_stopping_patience: 7
  embedding_cache_max_bytes: 4_294_967_296  # 4GB, preload all
  optimizer:
    lr: 5.0e-5
    weight_decay: 0.01
  losses:
    alpha: 0.5    # Graph similarity
    beta: 1.0     # Relative density
    gamma: 0.3    # Degree MMD
    delta: 0.3    # Clustering MMD
```

## 6. Priority Tasks Summary

| Status | Priority | Issue | Fix |
|---|---|---|---|
| [x] | P0 | DDP processes identical data on all ranks | Shard subgraphs: `my_subgraphs = plan[rank::world_size]` |
| [x] | P0 | Edge-cover sampling runs indefinitely | Replace with shuffle-and-chunk: O(|E|) instead of O(|E| * |V| * k) |
| [x] | P1 | Edges duplicated across subgraphs | Partition edges into disjoint chunks before building subgraphs |
| [x] | P1 | No gradient accumulation | Accumulate over N subgraphs before optimizer.step() |
| [ ] | P2 | Fixed 20-node subgraphs | Variable size [30, 60] to adapt to local density |
| [ ] | P2 | Sequential subgraph processing | Batch multiple subgraphs in one forward pass |
| [ ] | P2 | Per-subgraph negative sampling | Pre-assign negatives to subgraphs at epoch plan time |
