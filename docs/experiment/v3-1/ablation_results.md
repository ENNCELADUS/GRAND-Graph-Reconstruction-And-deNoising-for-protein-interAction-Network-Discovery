# v3.1 Architecture Ablation Results

## Scope

This document records the v3.1 downstream architecture ablations evaluated on the
PRING Human BFS split. The ESM3 cache, training objective, OHEM, optimizer,
scheduler, and evaluation pipeline are unchanged across these runs unless noted
below.

Result source:
`wangar2023@10.15.89.192:/public/home/wangar2023/grand/logs/v3.1/evaluate/*/evaluate.csv`

Aggregation rule: when a run has seeds 13, 47, and 101, the table reports the
mean across the three `test` rows.

## Ablation Definitions

| Run | Config source | Architecture change |
|---|---|---|
| `full` | `configs/v3-1/0430/full_s*.yaml` | Rich pooling with `esm_cls`, residue `mean`, residue attention, residue `max`, and gated fusion. |
| `mean_attn` | `configs/v3-1/0430/mean_attn_s*.yaml` | Residue-only compact readout using `mean` and attention pooling; removes ESM BOS/CLS, max pooling, and gated fusion. |
| `cls_mean_attn` | `configs/v3-1/0430/cls_mean_attn_s*.yaml` | Adds ESM BOS/CLS back to `mean_attn`; still excludes max pooling and gated fusion. |
| `no_cls` | `configs/v3-1/0430/no_cls_s*.yaml` | Best archived baseline: residue-only rich pooling with `mean`, attention, `max`, and gated fusion; excludes ESM BOS/CLS. |
| `no_max` | `configs/v3-1/0430/no_max_s*.yaml` | Keeps ESM BOS/CLS, residue `mean`, attention, and gated fusion; removes max pooling. |
| `no_gated` | `configs/v3-1/0430/no_gated_s*.yaml` | Keeps ESM BOS/CLS, residue `mean`, attention, and `max`; removes gated fusion. |
| `pair_context_gated` | `configs/v3-1/0506/pair_context_gated_s*.yaml` | Replaces default rich-pooling pair readout with residue-only `mean`, `max`, and pair-conditioned attention; builds symmetric pair features and gates branch features before projection. |
| `contact_sketch` | `configs/v3-1/0506/contact_sketch_s*.yaml` | Fuses `no_cls` rich-pooling representation with a latent contact sketch: 64 compressed tokens per protein, 32-d pair grid features, and a 2-block lightweight CNN. |

## Mean Test Metrics

| Run | Seeds | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pair_context_gated` | 13/47/101 | **0.680111** | **0.708888** | **0.622197** | 0.588737 | 0.655674 | **0.631610** | **0.608685** | **0.245478** |
| `no_cls` | 13/47/101 | 0.668268 | 0.689207 | 0.615661 | 0.585030 | 0.646309 | 0.623614 | 0.603363 | 0.232008 |
| `no_max` | 13/47/101 | 0.667569 | 0.689211 | 0.614922 | 0.587567 | 0.642291 | 0.622056 | 0.603944 | 0.230465 |
| `no_gated` | 13/47/101 | 0.663673 | 0.684698 | 0.613296 | **0.591386** | 0.635218 | 0.618818 | 0.604539 | 0.226992 |
| `cls_mean_attn` | 13/47/101 | 0.662627 | 0.687704 | 0.611078 | 0.546983 | **0.675207** | 0.628095 | 0.584139 | 0.224408 |
| `mean_attn` | 13/47/101 | 0.662320 | 0.682360 | 0.610696 | 0.583503 | 0.637904 | 0.617526 | 0.599792 | 0.221909 |
| `contact_sketch` | 13/47/101 | 0.660158 | 0.682424 | 0.610302 | 0.582136 | 0.638483 | 0.617148 | 0.599008 | 0.221055 |
| `full` | 13/47/101 | 0.660118 | 0.682508 | 0.611263 | 0.551319 | 0.671239 | 0.627115 | 0.586053 | 0.224622 |

## Interpretation

`pair_context_gated` is the strongest architecture among the v3.1 runs listed in
this document. Relative to the archived `no_cls` baseline, it improves AUROC by
0.011843, AUPRC by 0.019681, accuracy by 0.006536, F1 by 0.005322, and MCC by
0.013470. This points to pair-conditioned residue attention and gated branch
fusion as useful non-ESM-side modeling capacity.

`no_cls` remains the best simple rich-pooling baseline. Removing ESM BOS/CLS is
helpful compared with `full`, and `no_max` is effectively tied on AUPRC but is
slightly worse on AUROC, accuracy, and MCC.

`contact_sketch` does not help in its current form. Its latent contact-map CNN
fusion underperforms the `no_cls` baseline on all core metrics, suggesting that
the current compressed contact sketch either loses useful sequence-level signal
or introduces noisy structure bias without explicit contact supervision.

Practical next step: treat `pair_context_gated` as the new candidate architecture
for further validation, while keeping `no_cls` as the reproducible baseline.
