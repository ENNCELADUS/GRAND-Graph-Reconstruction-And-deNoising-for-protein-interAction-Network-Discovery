# 0430 Rich Pooling Ablation

Source configs: `configs/v3-1/0430/*.yaml`

Aggregation rule: metrics are means across seeds 13, 47, and 101.

## Ablation Definitions

| Run | Architecture change |
|---|---|
| `full` | Rich pooling with ESM BOS/CLS, residue `mean`, residue attention, residue `max`, and gated fusion. |
| `mean_attn` | Residue-only compact readout using `mean` and attention pooling; removes ESM BOS/CLS, max pooling, and gated fusion. |
| `cls_mean_attn` | Adds ESM BOS/CLS back to `mean_attn`; still excludes max pooling and gated fusion. |
| `no_cls` | Residue-only rich pooling with `mean`, attention, `max`, and gated fusion; excludes ESM BOS/CLS. |
| `no_max` | Keeps ESM BOS/CLS, residue `mean`, attention, and gated fusion; removes max pooling. |
| `no_gated` | Keeps ESM BOS/CLS, residue `mean`, attention, and `max`; removes gated fusion. |

## Test Metrics

| Run | Seeds | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_cls` | 13/47/101 | **0.668268** | 0.689207 | **0.615661** | 0.585030 | 0.646309 | 0.623614 | 0.603363 | **0.232008** |
| `no_max` | 13/47/101 | 0.667569 | **0.689211** | 0.614922 | 0.587567 | 0.642291 | 0.622056 | 0.603944 | 0.230465 |
| `no_gated` | 13/47/101 | 0.663673 | 0.684698 | 0.613296 | **0.591386** | 0.635218 | 0.618818 | **0.604539** | 0.226992 |
| `cls_mean_attn` | 13/47/101 | 0.662627 | 0.687704 | 0.611078 | 0.546983 | **0.675207** | **0.628095** | 0.584139 | 0.224408 |
| `mean_attn` | 13/47/101 | 0.662320 | 0.682360 | 0.610696 | 0.583503 | 0.637904 | 0.617526 | 0.599792 | 0.221909 |
| `full` | 13/47/101 | 0.660118 | 0.682508 | 0.611263 | 0.551319 | 0.671239 | 0.627115 | 0.586053 | 0.224622 |

