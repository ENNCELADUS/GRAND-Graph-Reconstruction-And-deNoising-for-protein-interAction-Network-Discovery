## Epoch 1 Timing

| Phase                      | Time              | % of Epoch |
|---------------------------|------------------|------------|
| Edge cover sampling       | 40s              | 0.5%       |
| Train forward/backward    | 7,671s (2h 8m)   | 26%        |
| Val pair pass             | 176s (3m)        | 0.6%       |
| Internal val topology     | 21,814s (6h 3m)  | 73%        |

The internal validation is **3× slower than training** and dominates each epoch at ~6 hours.  
Each epoch takes ~8.3 hours total, meaning **30 epochs ≈ 10.4 days**.

---

## Root Causes & Fixes (by impact)

### 1. Quadratic Pair Explosion in Validation (Biggest Factor)

Validation enumerates all **C(k, 2)** pairs per subgraph.

| Node size k | Pairs per subgraph | × 50 subgraphs | Cumulative % |
|------------|--------------------|----------------|--------------|
| 200        | 19,900             | 995,000        | 26%          |
| 180        | 16,110             | 805,500        | 47%          |
| 160        | 12,720             | 636,000        | 64%          |
| 140        | 9,730              | 486,500        | 77%          |
| ...        | ...                | ...            | ...          |
| **Total**  |                    | **3,822,500**  |              |

Top 3 sizes account for **64% of all inference work**.  
At `batch_size=128` → ~29,863 forward passes.

**Fixes:**
- Reduce `TOPOLOGY_EVAL_SAMPLES_PER_SIZE`: `50 → 20` (or 10) → **2.5× speedup**
- Trim node sizes: `[20, 60, 100, 140, 180]` → **~2× additional reduction**

---

### 2. No DDP Parallelism for Validation

All ranks redundantly run full validation (3.8M pairs).  
With 4 GPUs → **4× wasted compute**.

**Fix:**
- Distribute validation across ranks
- All-gather predictions → compute metrics on rank 0

**Result:** ~**4× speedup**

---

### 3. Expensive CPU Metrics (`evaluate_graph_samples`)

Costs:
- `nx.clustering()` → O(n²) × 500 graphs  
- `eigvalsh()` (200×200 Laplacian) × 500  
- `compute_mmd()` → O(n²)

`spectral_stats` not used in monitoring but computed every epoch.

**Fix:**
- Skip `spectral_stats` during internal validation

---

### 4. Validation Runs Every Epoch

Topology metrics are noisy early in training.

**Fix:**
- Add `internal_validation_frequency`
- Run full validation every N epochs (e.g., 3)

---

### 5. Small Validation Batch Size

Current: `batch_size = 128`  
Under `torch.no_grad()`, memory is available.

**Fix:**
- Increase to `512–1024`

**Result:** ~**1.5–2× speedup (inference phase)**

---

## Combined Impact Estimate

| Optimization                          | Speedup |
|--------------------------------------|---------|
| Distribute validation (4 GPUs)       | ~4×     |
| Reduce samples_per_size (50→20)      | ~2.5×   |
| Skip spectral_stats                  | ~1.3×   |

---

## Final Takeaway

Applying just:
- **Distributed validation**
- **Reduced sampling**

→ reduces internal validation from **6 hours → ~36 minutes per epoch**

Adding frequency control makes validation **negligible amortized cost**.

---

## Highest ROI Changes

1. Distribute validation across ranks  
2. Make `samples_per_size` and `node_sizes` configurable