# V3.1 Baseline Archive: no_cls

Archived on 2026-05-06 as the current best v3.1 architecture before adding new
non-ESM pair-readout ablations.

## Source

- Source configs: `configs/v3-1/0430/no_cls_s*.yaml`
- Source eval logs:
  `wangar2023@10.15.89.192:/public/home/wangar2023/grand/logs/v3.1/evaluate/no_cls_s*/evaluate.csv`

## Architecture

- Model: `v3.1`
- Ablation: `no_cls`
- Rich-pooling components: `[mean, attn, max, gated]`
- Seeds: `13`, `47`, `101`

## Mean Test Metrics

- AUROC: 0.668268
- AUPRC: 0.689207
- Accuracy: 0.615661
- F1: 0.603363
- MCC: 0.232008

`no_max` is effectively tied on AUPRC, but `no_cls` wins the main AUROC, MCC,
and accuracy comparison, so it is archived as the baseline for the next v3.1
architecture ablations.
