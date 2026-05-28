# 0517 s47 Architecture Probes

Run dates: 2026-05-17

Model card: TBD (add later)

Configs:
- `configs/v3-1/0517/*.yaml`

Logs:
- `logs/v3.1/train/pair_context_gated_*_s47/`
- `logs/v3.1/evaluate/pair_context_gated_*_s47/`

## Run Setup

Seed-47 v3.1 architecture probe on top of the pair-context gated readout. The batch tests AB/BA order aggregation, interaction-mode changes, and spectral-normalized readout/head widths while keeping the PRING Human BFS data and training recipe fixed.

## Test results

| Run | Seed | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pair_context_gated_abba_block_s47` | 47 | 0.694 | 0.722 | 0.630 | 0.582 | 0.678 | 0.644 | 0.612 | 0.262 |
| `pair_context_gated_abba_no_cross_s47` | 47 | 0.693 | 0.722 | 0.630 | 0.566 | 0.694 | 0.649 | 0.605 | 0.262 |
| `pair_context_gated_abba_s47` | 47 | 0.689 | 0.720 | 0.631 | 0.586 | 0.675 | 0.643 | 0.614 | 0.262 |
| `pair_context_gated_s47` | 47 | 0.689 | 0.718 | 0.625 | 0.633 | 0.618 | 0.624 | 0.628 | 0.251 |
| `pair_context_gated_sn_d64_s47` | 47 | 0.680 | 0.712 | 0.623 | 0.590 | 0.656 | 0.632 | 0.610 | 0.246 |
| `pair_context_gated_sn_d128_s47` | 47 | 0.686 | 0.713 | 0.627 | 0.583 | 0.671 | 0.639 | 0.610 | 0.255 |
| `pair_context_gated_sn_d256_s47` | 47 | 0.687 | 0.715 | 0.627 | 0.615 | 0.639 | 0.630 | 0.623 | 0.254 |
| `pair_context_gated_sn_d512_s47` | 47 | 0.678 | 0.706 | 0.621 | 0.611 | 0.630 | 0.623 | 0.617 | 0.241 |
| `pair_context_gated_sn_d768_s47` | 47 | 0.674 | 0.705 | 0.619 | 0.576 | 0.663 | 0.631 | 0.602 | 0.240 |

## Main Readout

AB/BA aggregation is useful. `abba_block_s47` has the best AUROC/AUPRC, while `abba_no_cross_s47` has the best MCC. Spectral-normalized width probes did not beat the AB/BA interaction-mode variants.
