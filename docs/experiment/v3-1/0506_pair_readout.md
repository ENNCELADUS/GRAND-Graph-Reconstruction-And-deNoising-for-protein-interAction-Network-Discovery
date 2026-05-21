# 0506 Pair Readout Ablation

Source configs: `configs/v3-1/0506/*.yaml`

Aggregation rule: metrics are means across seeds 13, 47, and 101.

## Ablation Definitions

| Run | Architecture change |
|---|---|
| `no_cls` | Archived baseline: residue-only rich pooling with `mean`, attention, `max`, and gated fusion; excludes ESM BOS/CLS. |
| `pair_context_gated` | Replaces default rich pooling with residue-only `mean`, `max`, and pair-conditioned attention; builds pair features and gates branch features before projection. |
| `contact_sketch` | Fuses `no_cls` rich pooling with a latent contact sketch: 64 compressed tokens per protein, 32-d pair grid features, and a 2-block lightweight CNN. |

## Test Metrics

| Run | Seeds | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pair_context_gated` | 13/47/101 | **0.680692** | **0.709697** | **0.622972** | **0.592585** | **0.653374** | **0.631331** | **0.610969** | **0.246680** |
| `no_cls` | 13/47/101 | 0.668268 | 0.689207 | 0.615661 | 0.585030 | 0.646309 | 0.623614 | 0.603363 | 0.232008 |
| `contact_sketch` | 13/47/101 | 0.660158 | 0.682424 | 0.610302 | 0.582136 | 0.638483 | 0.617148 | 0.599008 | 0.221055 |

