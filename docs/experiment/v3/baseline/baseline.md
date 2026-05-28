# Topology Fine-Tuning Baseline: Methodology

This document describes the implemented topology fine-tuning baseline in `src/pipeline/stages/topology_finetune.py`, `src/topology/finetune_data.py`, and `src/topology/finetune_losses.py`. The baseline fine-tunes the pairwise `v3` PPI scorer on edge-cover training subgraphs, using pairwise BCE supervision together with differentiable graph-topology objectives.

## 1. Model and Data

Let the training PPI graph be

$$
G = (V, E^+),
$$

where $V$ is the set of train proteins and $E^+$ is the set of positive train interactions loaded from the topology supervision file. Let $E^-$ denote the explicit negative pairs loaded from the same supervision split. Each protein $u \in V$ has a cached ESM3 embedding sequence $X_u \in \mathbb{R}^{L_u \times d}$.

The pairwise model is a logit scorer

$$
z_{uv} = f_\theta(X_u, X_v),
        \qquad
p_{uv} = \sigma(z_{uv}),
$$

implemented by `V3`: a shared sequence encoder, bidirectional cross-attention over the two encoded proteins, and an MLP output head. The topology fine-tune stage uses the same forward contract as ordinary pairwise training: each model batch contains `emb_a`, `emb_b`, `len_a`, and `len_b`, and the model returns `logits`.

## 2. Edge-Cover Epoch Construction

Each epoch starts by constructing a deterministic edge-cover plan from an epoch seed. All distributed ranks build the same global plan, then each rank takes a rank-local slice.

The sampler first shuffles the positive edge set $E^+$. It partitions the shuffled list into chunks $C_1,\ldots,C_M$, where each chunk is constrained by a target node budget $k_m$. With the default node range, target sizes cycle through $30,40,50,60$. If `edge_chunk_size` is not configured, the requested chunk cap for target size $k$ is

$$
B(k) = \left\lfloor \frac{k(k-1)}{4} \right\rfloor.
$$

For a positive-edge chunk $C_m$, the core node set is the first-seen ordered set of edge endpoints:

$$
R_m = \{u : \exists v,\ (u,v)\in C_m\}.
$$

If $|R_m| \ge k_m$, the subgraph node set is $S_m=R_m$. Otherwise, the implementation expands $R_m$ to $k_m$ nodes using the selected traversal strategy: randomized BFS, randomized DFS, or randomized walk with restart fallback. The topology target for this training item is the node-induced subgraph $G[S_m]$.

Explicit negatives are assigned once per epoch. The negative list $E^-$ is shuffled and streamed across subgraphs. For subgraph $m$, the requested negative count is

$$
|N_m| = r |C_m|,
$$

where $r$ is `topology_finetune.bce_negative_ratio`. The assignment is clipped if the finite explicit-negative pool is exhausted.

### Pseudocode: Epoch Plan

```text
BUILD_EPOCH_PLAN(G, E_minus, node_sizes, strategy, ratio, seed):
    rng = Random(seed)
    edges = shuffle(sorted(E_plus(G)), rng)

    chunks = []
    for target_size in cycle(node_sizes):
        chunk = []
        core_nodes = set()
        while edges remain:
            edge = next edge
            if chunk is nonempty and adding edge exceeds edge or node budget:
                break
            chunk.append(edge)
            core_nodes.update(edge endpoints)
        chunks.append((chunk, target_size))
        if no edges remain:
            break

    negatives = shuffle(sorted(E_minus), rng)
    offset = 0
    plan = []
    for chunk, target_size in chunks:
        nodes = EXPAND_CORE_NODES(G, endpoints(chunk), target_size, strategy, rng)
        neg_count = min(ratio * len(chunk), len(negatives) - offset)
        assigned_negatives = negatives[offset : offset + neg_count]
        offset += neg_count
        plan.append((nodes, assigned_positives=chunk, assigned_negatives))

    return plan
```

This plan gives every positive train edge exactly one assigned BCE-supervision slot per epoch. The node-induced subgraph may contain additional positive edges, but those additional edges are used as topology labels rather than assigned BCE positives.

## 3. Forward Propagation on One Subgraph

For a sampled node set $S=\{v_1,\ldots,v_n\}$, topology supervision enumerates every upper-triangle pair:

$$
\mathcal{P}(S)=\{(i,j):1\le i<j\le n\}.
$$

For each $(i,j)\in\mathcal{P}(S)$, the topology label is

$$
y^{\mathrm{topo}}_{ij} =
\mathbf{1}\{(v_i,v_j)\in E^+\}.
$$

The stage materializes pair batches from cached embeddings, forwards them through $f_\theta$, and concatenates logits back into the subgraph order. The topology forward therefore produces:

$$
\mathbf{z}_S = \{z_{ij}\}_{(i,j)\in\mathcal{P}(S)},\qquad
\mathbf{p}_S = \sigma(\mathbf{z}_S).
$$

BCE supervision uses the assigned edge sets rather than all topology pairs:

$$
\mathcal{B}_S = C_S \cup N_S,
\qquad
y^{\mathrm{bce}}_{uv} =
\begin{cases}
1, & (u,v)\in C_S,\\
0, & (u,v)\in N_S.
\end{cases}
$$

When assigned explicit negatives are present, the implementation runs a separate supervised-pair forward for $\mathcal{B}_S$. This ensures explicit negatives contribute to BCE even when their endpoints are not both inside the topology subgraph $S$. If no explicit negatives are assigned, BCE can be computed from the masked all-pair forward.

## 4. Loss Function

The implemented objective combines masked BCE with differentiable topology terms.

### 4.1 Masked BCE

For supervised pair logits $\{z_\ell\}$, labels $\{y_\ell\}$, and mask $m_\ell$, BCE is

$$
\mathcal{L}_{\mathrm{bce}}
=
\frac{\sum_\ell m_\ell\,
\mathrm{BCEWithLogits}(z_\ell,y_\ell)}
{\sum_\ell m_\ell}.
$$

If no supervised pair is available, the loss is a differentiable zero.

### 4.2 Soft Graph Similarity

The graph-similarity surrogate is computed on upper-triangle pair vectors:

$$
\mathcal{L}_{\mathrm{GS}}
=
\frac{\sum_{(i,j)} |p_{ij}-y^{\mathrm{topo}}_{ij}|}
{\sum_{(i,j)} p_{ij}+\sum_{(i,j)} y^{\mathrm{topo}}_{ij}+\epsilon}.
$$

This is the differentiable analogue of $1-\mathrm{GS}$.

### 4.3 Relative Density

The soft predicted and target densities are

$$
\rho_{\mathrm{pred}}
=
\frac{2\sum_{(i,j)}p_{ij}}{n(n-1)},
\qquad
\rho_{\mathrm{true}}
=
\frac{2\sum_{(i,j)}y^{\mathrm{topo}}_{ij}}{n(n-1)}.
$$

The default relative-density penalty is the smooth-L1 loss on the log density ratio:

$$
\mathcal{L}_{\mathrm{RD}}
=
\mathrm{SmoothL1}
\left(
\log\frac{\rho_{\mathrm{pred}}+\epsilon}{\rho_{\mathrm{true}}+\epsilon},
0
\right).
$$

If the target subgraph has effectively zero density, the implementation falls back to $\rho_{\mathrm{pred}}^2$.

### 4.4 Degree-Distribution MMD

Soft degrees are accumulated directly from upper-triangle probabilities:

$$
d_i^{\mathrm{pred}}=\sum_{j\ne i}p_{ij},
\qquad
d_i^{\mathrm{true}}=\sum_{j\ne i}y^{\mathrm{topo}}_{ij}.
$$

Both degree vectors are converted into normalized Gaussian soft histograms $h_{\mathrm{pred}}$ and $h_{\mathrm{true}}$. The MMD surrogate is

$$
\mathcal{L}_{\mathrm{Deg}}
=
2 - 2\exp
\left(
-\frac{\mathrm{TV}(h_{\mathrm{pred}},h_{\mathrm{true}})^2}
{2\sigma^2}
\right),
$$

where $\mathrm{TV}(a,b)=\frac{1}{2}\|a-b\|_1$.

### 4.5 Clustering-Coefficient MMD

When `compute_clustering_mmd=true`, pair probabilities are scattered into symmetric weighted adjacencies $A_{\mathrm{pred}}$ and $A_{\mathrm{true}}$. The differentiable local clustering coefficient is

$$
c_i(A)
=
\frac{\sum_j (A^2)_{ij}A_{ij}}
{d_i(A)(d_i(A)-1)+\epsilon},
$$

with zero assigned when the denominator is not positive. Predicted and target clustering vectors are converted into soft histograms, then compared with the same MMD surrogate:

$$
\mathcal{L}_{\mathrm{Clus}}
=
\mathrm{MMD}_{\mathrm{soft}}(c(A_{\mathrm{pred}}), c(A_{\mathrm{true}})).
$$

## 5. Scheduled and Balanced Training Objective

Raw topology weights are scaled by an epoch schedule $s_t$:

$$
\tilde{\alpha}_t=s_t\alpha,\quad
\tilde{\beta}_t=s_t\beta,\quad
\tilde{\gamma}_t=s_t\gamma,\quad
\tilde{\delta}_t=s_t\delta.
$$

With the default schedule, $s_t=0$ during warmup, then ramps linearly to $1$. During warmup, topology forward/backward is skipped and the model trains only on assigned BCE pairs.

After warmup, topology terms are normalized by detached EMA statistics:

$$
\bar{\mathcal{L}}_q
=
\mathrm{clip}\left(
\frac{\mathcal{L}_q}{\mathrm{EMA}_q+\epsilon},
\mathrm{max}=c
\right),
\qquad
q\in\{\mathrm{GS},\mathrm{RD},\mathrm{Deg},\mathrm{Clus}\}.
$$

The normalized topology terms are grouped into density and shape objectives:

$$
\mathcal{L}_{\mathrm{density}}
=
\tilde{\beta}_t\bar{\mathcal{L}}_{\mathrm{RD}},
$$

$$
\mathcal{L}_{\mathrm{shape}}
=
\tilde{\alpha}_t\bar{\mathcal{L}}_{\mathrm{GS}}
+
\tilde{\gamma}_t\bar{\mathcal{L}}_{\mathrm{Deg}}
+
\tilde{\delta}_t\bar{\mathcal{L}}_{\mathrm{Clus}}.
$$

Grouped GradNorm maintains task weights
$w_{\mathrm{bce}}, w_{\mathrm{density}}, w_{\mathrm{shape}}$ for the three objectives. The optimized loss is

$$
\mathcal{L}_{\mathrm{total}}
=
w_{\mathrm{bce}}\mathcal{L}_{\mathrm{bce}}
+
w_{\mathrm{density}}\mathcal{L}_{\mathrm{density}}
+
w_{\mathrm{shape}}\mathcal{L}_{\mathrm{shape}}.
$$

GradNorm estimates gradient norms on the model output head when available, otherwise on all trainable parameters. The weights are clipped to the configured range and renormalized to preserve the number of active tasks.

## 6. Backward Propagation and Optimization

The default baseline keeps the full subgraph computational graph for each subgraph or subgraph group. It computes $\mathcal{L}_{\mathrm{total}}$, calls `accelerator.backward`, and steps AdamW at the configured gradient-accumulation boundary.

### Pseudocode: Training Epoch

```text
TRAIN_EPOCH(model, graph, negatives, epoch):
    plan = BUILD_EPOCH_PLAN(graph, negatives, seed=epoch_seed(epoch))
    local_tasks = plan[rank :: world_size]
    local_tasks = PAD_FOR_EQUAL_DDP_BACKWARD_COUNT(local_tasks)

    topology_scale = SCHEDULE(epoch)
    effective_weights = topology_scale * base_topology_weights

    optimizer.zero_grad()
    for accumulation_window in windows(local_tasks):
        window_loss = 0

        for task in accumulation_window:
            if topology_scale == 0:
                z_bce, y_bce = FORWARD_ASSIGNED_SUPERVISED_PAIRS(task)
                bce = MASKED_BCE(z_bce, y_bce)
                total = bce
            else:
                if task has assigned negatives:
                    z_bce, y_bce = FORWARD_ASSIGNED_SUPERVISED_PAIRS(task)
                    bce = MASKED_BCE(z_bce, y_bce)
                else:
                    bce = null

                z_all, y_topo, pair_index = FORWARD_ALL_WITHIN_SUBGRAPH_PAIRS(task.nodes)
                if bce is null:
                    bce = MASKED_BCE(z_all, assigned_pair_mask)

                topo = TOPOLOGY_LOSSES(sigmoid(z_all), y_topo, pair_index)
                topo_norm = EMA_NORMALIZE_AND_CLIP(topo)
                density, shape = GROUP_TOPOLOGY_OBJECTIVES(topo_norm, effective_weights)
                task_weights = GRADNORM_UPDATE(bce, density, shape)
                total = task_weights.bce * bce
                      + task_weights.density * density
                      + task_weights.shape * shape

            if task is DDP padding:
                total = 0 * total
            window_loss += total

        backward(window_loss / len(accumulation_window))
        optimizer.step()
        optimizer.zero_grad()
```

Distributed training uses an identical epoch plan on every rank and assigns tasks by striding. Padding tasks contribute zero loss but still participate in backward synchronization, which keeps DDP collectives aligned.

The implementation also contains an optional `chunked_backward` path for memory-limited runs. That path first collects detached logits without retaining activations, computes proxy gradients with respect to the concatenated logits, then replays pair chunks under the original model and backpropagates the precomputed logit gradients. This mode requires `subgraphs_per_forward=1`, disables clustering MMD, and does not support GradNorm. It is not the default `configs/v3.yaml` baseline.

## 7. Internal Validation and Checkpoint Selection

After each epoch, the stage evaluates pairwise validation metrics and optional internal topology validation. Internal topology validation samples fixed node-size buckets from the validation supervision graph, predicts all within-subgraph pairs with the fixed decision threshold, reconstructs hard graphs, and reports PRING-style summaries:

- graph similarity,
- relative density,
- degree-distribution MMD,
- clustering-coefficient MMD,
- optional spectral statistics.

The default monitor is `internal_val_graph_sim`. The best checkpoint is saved from the topology fine-tune stage and is later consumed by ordinary pairwise evaluation and topology evaluation.
