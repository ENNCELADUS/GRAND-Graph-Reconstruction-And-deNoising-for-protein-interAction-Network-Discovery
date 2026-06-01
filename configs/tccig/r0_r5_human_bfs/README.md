# TCCIG R0-R5 Human BFS Queue

Run with:

```bash
sbatch scripts/tccig.sh configs/tccig/r0_r5_human_bfs
```

The queue preserves `model_config.model: tccig` and the existing TCCIG train,
evaluate, and topology-evaluate stages.

| File | Run ID | Increment |
| --- | --- | --- |
| `00_r0_esm_cosine.yaml` | `gpr_r0_esm_cosine_human_bfs` | Pooled ESM cosine retrieval-only baseline with validation-density assembly. |
| `01_r1_sorf_dual.yaml` | `gpr_r1_sorf_dual_human_bfs` | SORF residue-factorized dual retriever. |
| `02_r2_graph_prior.yaml` | `gpr_r2_graph_prior_human_bfs` | Adds offline graph-prior structural and degree targets. |
| `03_r3_hard_negative.yaml` | `gpr_r3_hard_negative_human_bfs` | Adds exact top-k hard-negative mining and margin loss/cache artifacts. |
| `04_r4_reranker.yaml` | `gpr_r4_reranker_human_bfs` | Adds local candidate reranker; external teacher remains disabled. |
| `05_r5_hybrid_assembly.yaml` | `gpr_r5_hybrid_assembly_human_bfs` | Enables hybrid validation-density plus degree-cap assembly. |

