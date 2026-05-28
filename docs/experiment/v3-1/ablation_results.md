# v3.1 Ablation Results

This directory records downstream architecture ablations for the PRING Human BFS
split. Unless a document says otherwise, ESM3 cache, loss, OHEM, optimizer,
scheduler, data split, and evaluation metrics are unchanged.

Result source:
`wangar2023@10.15.89.192:/public/home/wangar2023/grand/logs/v3.1/evaluate/*/evaluate.csv`

TUnA quick reproduction source:
`wangar2023@10.15.89.192:/public/home/wangar2023/grand/logs/tuna/evaluate/*/evaluate.csv`

## Experiment Documents

| Document | Experiment group | Main takeaway |
|---|---|---|
| [0430_rich_pooling.md](0430_rich_pooling.md) | Rich-pooling component ablations. | `no_cls` is the best simple rich-pooling baseline. |
| [0506_pair_readout.md](0506_pair_readout.md) | New pair readout modules. | `pair_context_gated` is the strongest three-seed v3.1 readout architecture in this result set. |
| [0514_tuna_quick_reproduction.md](0514_tuna_quick_reproduction.md) | TUnA small-head quick reproduction. | TUnA's official block/no-cross setting is better than true cross-chain attention. |
| [0517_s47_architecture_probes.md](0517_s47_architecture_probes.md) | AB/BA, interaction, width, and spectral-norm probes. | `abba_block` and `abba_no_cross` are the best seed-47 probes. |
| [0523_followup_ablation.md](0523_followup_ablation.md) | Follow-up seed sweeps and width/spectral-norm controls. | `abba_no_cross` has the best seed-mean AUROC/MCC; `abba_block_sn_d256` has the best seed-mean AUPRC. |
