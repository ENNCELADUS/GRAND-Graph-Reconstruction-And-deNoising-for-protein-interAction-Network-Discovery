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