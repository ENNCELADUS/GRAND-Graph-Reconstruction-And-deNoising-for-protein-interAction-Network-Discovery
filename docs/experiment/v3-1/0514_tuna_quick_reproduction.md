# 0514 TUnA Quick Reproduction

Source configs: `configs/tuna/0514/*.yaml`

Aggregation rule: each row is seed 47 only.

## Ablation Definitions

| Run | Architecture change |
|---|---|
| `tuna64_linear_official_s47` | TUnA hid=64, 1 layer, spectral norm, AB/BA max, official block-diagonal inter mask, linear head. |
| `tuna64_linear_cross_s47` | Same small TUnA head, but replaces the official block mask with true cross-chain attention. |
| `tuna64_sngp_official_s47` | Official block mask plus diagonal RFF/SNGP-style output head. |
| `tuna64_sngp_cross_s47` | True cross-chain attention plus diagonal RFF/SNGP-style output head. |

## Test Metrics

| Run | Seed | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tuna64_linear_official_s47` | 47 | **0.682707** | **0.702437** | **0.622598** | 0.699220 | 0.545935 | **0.606409** | 0.649516 | **0.248090** |
| `tuna64_linear_cross_s47` | 47 | 0.669220 | 0.672152 | 0.610068 | 0.780586 | 0.439462 | 0.582164 | 0.666930 | 0.234094 |
| `tuna64_sngp_official_s47` | 47 | 0.670443 | 0.689549 | 0.610105 | 0.730000 | 0.490148 | 0.588905 | 0.651906 | 0.226771 |
| `tuna64_sngp_cross_s47` | 47 | 0.647703 | 0.630766 | 0.598296 | **0.841666** | 0.354801 | 0.566194 | **0.676980** | 0.224932 |

