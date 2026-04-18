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

---

## 4. Observed Training Dynamics (v3, epochs 1–5)

From `logs/v3/slurm_886526.err` (4×GPU, scratch init, 30 epochs planned):

| Epoch | Train Loss | Val AUPRC | Internal Val Graph Sim |
|---|---|---|---|
| 1 | 6.14 | 0.531 | 0.0793 |
| 2 | 2.72 | 0.547 | 0.0793 |
| 3 | 2.70 | 0.550 | 0.0793 |
| 4 | 2.67 | 0.557 | 0.0793 |
| 5 | 2.67 | 0.567 | 0.0793 |

Key observations:
- Train loss plateaus after epoch 2 (~2.67–2.72 range, barely moving).
- Val AUPRC improves slowly but steadily (+0.036 over 5 epochs).
- **Internal Val Graph Sim is completely frozen at 0.0793 across all 5 epochs** — the topology losses have zero measurable effect on graph structure.

### Root Cause Analysis

**1. Topology losses are drowned out by BCE from scratch.**
The model starts with random weights. BCE loss opens at 32.9 (step 1, epoch 1) and crashes to 6.1 by epoch end, then 2.7 by epoch 2. The topology weights (alpha=0.08, beta=0.03, gamma=0.02, delta=0.02, total=0.15) are too small relative to BCE=2.7 to produce meaningful gradient signal. The topology gradients on near-random predictions are noisy and likely conflict with BCE gradients.

**2. Hard-threshold graph_sim cannot move until probability distribution shifts.**
The internal validation uses `best_f1_on_valid` thresholding to produce binary graphs. Until the model's probability distribution shifts enough to change binary predictions at the chosen threshold, graph_sim stays constant regardless of soft-loss progress.

**3. Gradient conflict in early training.**
When the model is still learning basic pairwise discrimination, topology gradients push weights toward matching graph structure — a signal the model cannot yet interpret reliably. This creates conflicting gradient directions that slow BCE convergence and prevent topology learning simultaneously.

---

## 5. Proposed Fix: Topology Loss Weight Scheduler

### Motivation

The model needs to first learn a reliable pairwise scorer (BCE phase), then be guided toward topology-consistent predictions (topology phase). Applying topology losses from epoch 0 wastes gradient budget and produces noisy signal.

This is a standard curriculum learning pattern: start simple, add complexity once the base task is stable.

### Algorithm

```
TOPOLOGY_LOSS_SCALE(epoch, warmup_epochs, ramp_epochs, schedule):
    if epoch < warmup_epochs:
        return 0.0                          # pure BCE phase
    ramp_epoch = epoch - warmup_epochs
    if ramp_epochs == 0 or ramp_epoch >= ramp_epochs:
        return 1.0                          # full topology weight
    progress = ramp_epoch / ramp_epochs
    if schedule == "cosine":
        return 0.5 * (1 - cos(pi * progress))
    return progress                         # linear ramp
```

Applied per epoch before computing topology losses:

```
scale = TOPOLOGY_LOSS_SCALE(epoch, warmup_epochs, ramp_epochs, schedule)
effective_alpha = alpha * scale
effective_beta  = beta  * scale
effective_gamma = gamma * scale
effective_delta = delta * scale

topo_loss = effective_alpha * GS(...)
          + effective_beta  * RD(...)
          + effective_gamma * DegMMD(...)
          + effective_delta * ClusMMD(...)
```

### Recommended Config

```yaml
topology_finetune:
  loss_weight_schedule:
    warmup_epochs: 5    # epochs 0–4: BCE only, let pairwise scorer stabilize
    ramp_epochs: 5      # epochs 5–9: linear ramp from 0 → full topology weight
    schedule: "linear"  # or "cosine" for smoother activation
```

With 30 total epochs:
- Epochs 1–5: pure BCE, model learns basic pairwise discrimination
- Epochs 6–10: topology losses ramp in gradually
- Epochs 11–30: full joint training with stable BCE baseline

### Expected Behavior

- Train loss should continue dropping during warmup (BCE-only signal is clean).
- Graph sim should start moving around epoch 6–7 once topology losses activate.
- Val AUPRC should not regress during ramp because BCE weight is unchanged.

### Implementation Tasks

- [ ] Add `TopologyLossWeightSchedule` dataclass to `src/topology/finetune_losses.py`
  - Fields: `warmup_epochs: int = 0`, `ramp_epochs: int = 0`, `schedule: str = "linear"`
  - Supported schedules: `"linear"`, `"cosine"`
- [ ] Add `topology_loss_scale(*, epoch: int, schedule: TopologyLossWeightSchedule) -> float` to `finetune_losses.py`
- [ ] Add `_parse_loss_weight_schedule(finetune_cfg) -> TopologyLossWeightSchedule` to `topology_finetune.py`
- [ ] Add `loss_weight_schedule: TopologyLossWeightSchedule` field to `TopologyFinetuneStageContext`
- [ ] In `_fit_epoch`: compute `scale = topology_loss_scale(epoch=epoch_index, schedule=...)`, create scaled `TopologyLossWeights` via `dataclasses.replace` before passing to `compute_topology_losses`
- [ ] Log `topo_loss_scale` in the epoch-done event so it's visible in the SLURM log
- [ ] Update `configs/v3.yaml` with `loss_weight_schedule` block
- [ ] Add unit tests for `topology_loss_scale` (boundary conditions: epoch=0, epoch=warmup-1, epoch=warmup, mid-ramp, post-ramp)

---

## 7. Proposed Fix: Validation Threshold Stability and Soft Monitoring

### Root Cause

The internal validation computes graph_sim by:
1. Running inference on all validation subgraph pairs
2. Picking a decision threshold via `best_f1_on_valid` on the pairwise val set
3. Applying that threshold to produce binary predicted graphs
4. Computing hard graph_sim between predicted and ground-truth graphs

Two failure modes compound each other:

**Threshold instability.** `best_f1_on_valid` searches the full precision-recall curve on the pairwise val set. Early in training, the model outputs near-uniform probabilities. The F1-optimal threshold can land anywhere — often high (0.7–0.9) to suppress false positives — producing a near-empty predicted graph. This threshold shifts each epoch independently of model improvement, so graph_sim can stay frozen or even move in the wrong direction due to threshold drift rather than model change.

**Hard metric insensitivity.** Even when soft probabilities are improving (AUPRC 0.531→0.567 over 5 epochs), no hard edge flips occur at the current threshold. The binary graph is identical epoch-to-epoch, so graph_sim reports the same value. The metric gives zero signal about soft progress that is genuinely happening.

### Fix: Fixed threshold for internal validation during training

Use `mode: fixed, value: 0.5` for the topology finetune stage's decision threshold. A fixed 0.5 threshold is stable across epochs — graph_sim will move whenever the model's probability distribution shifts, not whenever the F1-optimal threshold happens to shift. `best_f1_on_valid` remains the right choice for the final `evaluate` stage where calibration matters, but it is the wrong choice for a training monitor.

```yaml
topology_finetune:
  decision_threshold:
    mode: fixed
    value: 0.5
```

This is already supported by `_resolve_internal_validation_threshold` — no code change needed, only config.

**Why 0.5 specifically:** It is the natural decision boundary for a BCE-trained sigmoid output. The model is explicitly trained to push positive pairs above 0.5 and negative pairs below it. Using 0.5 means graph_sim directly reflects whether the model has learned to discriminate, with no threshold search introducing noise.