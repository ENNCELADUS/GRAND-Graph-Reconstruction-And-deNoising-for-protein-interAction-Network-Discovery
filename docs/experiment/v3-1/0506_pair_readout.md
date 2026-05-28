# 0506 Pair Readout Ablation

Run dates: 2026-05-06

Model card: TBD (add later)

Configs:
- `configs/v3-1/0506/*.yaml`
- Baseline reference rows reuse `configs/v3-1/0430/no_cls_s{13,47,101}.yaml`

Logs:
- `logs/v3.1/train/{pair_context_gated,contact_sketch}_s{13,47,101}/`
- `logs/v3.1/evaluate/{pair_context_gated,contact_sketch,no_cls}_s{13,47,101}/`

## Run Setup

Three-seed v3.1 pair-readout ablation on PRING Human BFS. The comparison keeps the same data split and training setup as the 0430 rich-pooling sweep, then replaces the readout module.

Compared variants:
- `pair_context_gated`: residue mean/max plus pair-conditioned attention and gated pair features.
- `contact_sketch`: latent contact-sketch fusion with compressed tokens and a lightweight CNN.
- `no_cls`: archived residue-only rich-pooling baseline.

## Test results

| Run | Seeds | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pair_context_gated` | 13/47/101 | 0.680 | 0.709 | 0.622 | 0.589 | 0.656 | 0.632 | 0.609 | 0.245 |
| `no_cls` | 13/47/101 | 0.668 | 0.689 | 0.616 | 0.585 | 0.646 | 0.624 | 0.603 | 0.232 |
| `contact_sketch` | 13/47/101 | 0.660 | 0.682 | 0.610 | 0.582 | 0.638 | 0.617 | 0.599 | 0.221 |

## Main Readout

`pair_context_gated` is the clear three-seed winner and improves over `no_cls` by 0.012 AUROC, 0.020 AUPRC, and 0.013 MCC. `contact_sketch` did not justify its added complexity in this setup.
