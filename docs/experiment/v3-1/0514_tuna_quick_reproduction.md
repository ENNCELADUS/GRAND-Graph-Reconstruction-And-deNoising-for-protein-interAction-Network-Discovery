# 0514 TUnA Quick Reproduction

Run dates: 2026-05-14

Model card: TBD (add later)

Configs:
- `configs/tuna/0514/*.yaml`

Logs:
- `logs/tuna/train/tuna64_{linear,sngp}_{official,cross}_s47/`
- `logs/tuna/evaluate/tuna64_{linear,sngp}_{official,cross}_s47/`

## Run Setup

Seed-47 quick reproduction of a small TUnA-style transformer head on the PRING Human BFS split. Each run uses hidden size 64, one layer, spectral-normalized attention, AB/BA max aggregation, and either the official block-diagonal inter mask or true cross-chain attention. Output heads compare linear vs diagonal RFF/SNGP-style variants.

## Test results

| Run | Seed | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tuna64_linear_official_s47` | 47 | 0.683 | 0.702 | 0.623 | 0.699 | 0.546 | 0.606 | 0.650 | 0.248 |
| `tuna64_linear_cross_s47` | 47 | 0.669 | 0.672 | 0.610 | 0.781 | 0.439 | 0.582 | 0.667 | 0.234 |
| `tuna64_sngp_official_s47` | 47 | 0.670 | 0.690 | 0.610 | 0.730 | 0.490 | 0.589 | 0.652 | 0.227 |
| `tuna64_sngp_cross_s47` | 47 | 0.648 | 0.631 | 0.598 | 0.842 | 0.355 | 0.566 | 0.677 | 0.225 |

## Main Readout

The official block mask is stronger than true cross-chain attention for this quick reproduction. The linear official-head variant is best by AUROC, AUPRC, accuracy, precision, and MCC; SNGP-style heads did not help in this small setting.
