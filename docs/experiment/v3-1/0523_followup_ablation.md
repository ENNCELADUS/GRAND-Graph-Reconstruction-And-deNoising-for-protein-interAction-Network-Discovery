# 0523 v3.1 Follow-up Ablation

Run dates: 2026-05-22 to 2026-05-23

Model card: TBD (add later)

Configs:
- `configs/v3-1/0522/*.yaml`
- `configs/v3-1/0523/*.yaml`
- Seed-47 reference rows reuse `configs/v3-1/0517/*.yaml`

Logs:
- `logs/v3.1/train/pair_context_gated_*_{s13,s47,s101}/`
- `logs/v3.1/evaluate/pair_context_gated_*_{s13,s47,s101}/`

## Run Setup

Follow-up sweep after the 0517 seed-47 probe. The batch checks whether the best seed-47 ideas hold across seeds, separates width from spectral normalization, and compares cross-chain interaction against block/no-cross interaction choices.

Rows with seeds 13/47/101 are seed means. Single-seed rows are controls that have not been expanded to a seed sweep.

## Test results

| Run | Seeds | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `abba_no_cross` | 13/47/101 | 0.691 | 0.719 | 0.630 | 0.582 | 0.679 | 0.645 | 0.612 | 0.262 |
| `abba_block_sn_d256` | 13/47/101 | 0.690 | 0.720 | 0.629 | 0.567 | 0.690 | 0.648 | 0.604 | 0.260 |
| `abba_block` | 13/47/101 | 0.688 | 0.719 | 0.628 | 0.553 | 0.702 | 0.651 | 0.597 | 0.259 |
| `abba_block_d256_no_sn` | 13/47/101 | 0.684 | 0.715 | 0.625 | 0.597 | 0.653 | 0.633 | 0.614 | 0.250 |
| `abba_block_sn_d512` | 47 | 0.681 | 0.713 | 0.626 | 0.561 | 0.691 | 0.645 | 0.600 | 0.254 |
| `pair_context_gated_d256_no_sn` | 47 | 0.670 | 0.703 | 0.619 | 0.499 | 0.738 | 0.656 | 0.567 | 0.244 |

## Main Readout

`abba_no_cross` is the strongest seed-mean result by AUROC, accuracy, and MCC. `abba_block_sn_d256` has the best seed-mean AUPRC, but the margin over `abba_block` and `abba_no_cross` is small. The d256 no-spectral-norm control is weaker, so the gain is not just from narrowing width.
