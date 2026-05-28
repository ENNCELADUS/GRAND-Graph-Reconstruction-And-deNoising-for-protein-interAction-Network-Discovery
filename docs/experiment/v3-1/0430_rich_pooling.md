# 0430 Rich Pooling Ablation

Run dates: 2026-04-30

Model card: TBD (add later)

Configs:
- `configs/v3-1/0430/*.yaml`

Logs:
- `logs/v3.1/train/{full,mean_attn,cls_mean_attn,no_cls,no_max,no_gated}_s{13,47,101}/`
- `logs/v3.1/evaluate/{full,mean_attn,cls_mean_attn,no_cls,no_max,no_gated}_s{13,47,101}/`

## Run Setup

Three-seed v3.1 readout-component ablation on the PRING Human BFS split. All rows keep the same ESM3 cache, training objective, OHEM sampling, optimizer, scheduler, split, and fixed test metric suite. Metrics below are means across seeds 13, 47, and 101.

Compared variants:
- `full`: ESM BOS/CLS, residue mean, residue attention, residue max, and gated fusion.
- `mean_attn`: residue mean + attention only.
- `cls_mean_attn`: ESM BOS/CLS + residue mean + attention.
- `no_cls`: residue mean + attention + max + gated fusion, without ESM BOS/CLS.
- `no_max`: removes max pooling.
- `no_gated`: removes gated fusion.

## Test results

| Run | Seeds | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_cls` | 13/47/101 | 0.668 | 0.689 | 0.616 | 0.585 | 0.646 | 0.624 | 0.603 | 0.232 |
| `no_max` | 13/47/101 | 0.668 | 0.689 | 0.615 | 0.588 | 0.642 | 0.622 | 0.604 | 0.230 |
| `no_gated` | 13/47/101 | 0.664 | 0.685 | 0.613 | 0.591 | 0.635 | 0.619 | 0.605 | 0.227 |
| `cls_mean_attn` | 13/47/101 | 0.663 | 0.688 | 0.611 | 0.547 | 0.675 | 0.628 | 0.584 | 0.224 |
| `mean_attn` | 13/47/101 | 0.662 | 0.682 | 0.611 | 0.584 | 0.638 | 0.618 | 0.600 | 0.222 |
| `full` | 13/47/101 | 0.660 | 0.683 | 0.611 | 0.551 | 0.671 | 0.627 | 0.586 | 0.225 |

## Main Readout

`no_cls` is the strongest simple rich-pooling baseline by AUROC, accuracy, and MCC. Removing ESM BOS/CLS helps more than adding more pooled components; `full` underperforms the leaner variants.
