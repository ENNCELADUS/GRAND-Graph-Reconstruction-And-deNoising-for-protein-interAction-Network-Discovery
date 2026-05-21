# 0517 s47 Architecture Probes

Source configs: `configs/v3-1/0517/*.yaml`

Aggregation rule: each row is seed 47 only.

## Ablation Definitions

| Run | Architecture change |
|---|---|
| `pair_context_gated_s47` | Reference run: single AB order, bidirectional cross-chain interaction, pair-context gated readout. |
| `pair_context_gated_abba_s47` | Adds AB/BA max aggregation to the reference while keeping bidirectional cross-chain interaction. |
| `pair_context_gated_abba_no_cross_s47` | Uses AB/BA max aggregation and skips post-encoder cross interaction; only the readout fuses chains. |
| `pair_context_gated_abba_block_s47` | Uses AB/BA max aggregation with block-self interaction; preserves layer budget but prevents A-B token mixing before readout. |
| `pair_context_gated_sn_d64_s47` | Width 64 with spectral norm on pair readout and MLP head. |
| `pair_context_gated_sn_d128_s47` | Width 128 with spectral norm on pair readout and MLP head. |
| `pair_context_gated_sn_d256_s47` | Width 256 with spectral norm on pair readout and MLP head. |
| `pair_context_gated_sn_d512_s47` | Width 512 with spectral norm on pair readout and MLP head. |
| `pair_context_gated_sn_d768_s47` | Width 768 with spectral norm on pair readout and MLP head. |

## Test Metrics

| Run | Seed | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pair_context_gated_abba_block_s47` | 47 | **0.693944** | **0.722210** | 0.630211 | 0.581975 | 0.678472 | 0.644251 | 0.611532 | 0.261667 |
| `pair_context_gated_abba_no_cross_s47` | 47 | 0.693333 | 0.722024 | 0.630045 | 0.566160 | **0.693963** | **0.649237** | 0.604860 | **0.262272** |
| `pair_context_gated_abba_s47` | 47 | 0.689112 | 0.720348 | **0.630599** | 0.586373 | 0.674849 | 0.643407 | 0.613567 | 0.262249 |
| `pair_context_gated_s47` | 47 | 0.688510 | 0.717681 | 0.625444 | **0.632598** | 0.618285 | 0.623793 | **0.628165** | 0.250909 |
| `pair_context_gated_sn_d64_s47` | 47 | 0.679722 | 0.712459 | 0.622967 | 0.589735 | 0.656216 | 0.631854 | 0.610068 | 0.246496 |
| `pair_context_gated_sn_d128_s47` | 47 | 0.686096 | 0.712797 | 0.626959 | 0.583047 | 0.670894 | 0.639318 | 0.609887 | 0.254925 |
| `pair_context_gated_sn_d256_s47` | 47 | 0.687333 | 0.714596 | 0.627107 | 0.614862 | 0.639358 | 0.630422 | 0.622545 | 0.254296 |
| `pair_context_gated_sn_d512_s47` | 47 | 0.677833 | 0.706457 | 0.620694 | 0.611093 | 0.630301 | 0.623182 | 0.617078 | 0.241437 |
| `pair_context_gated_sn_d768_s47` | 47 | 0.674223 | 0.704605 | 0.619456 | 0.575731 | 0.663204 | 0.631040 | 0.602118 | 0.239853 |

