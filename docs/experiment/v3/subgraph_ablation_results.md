# v3 Subgraph Size Ablation Results

## Scope

This note records the v3 topology-finetuning ablation over training subgraph
size on PRING Human BFS, seed 47. All completed runs warm-start from
`models/v3/train/20260308_002906/best_model.pth` and use the v3 model with
ESM3 cached embeddings.

Primary result source: `logs/v3/evaluate/*/evaluate.csv` and
`logs/v3/topology_evaluate/*/topology_metrics.csv`.

For comparability, the main table uses fixed decision threshold `0.5`. The
`ws_n20`, `ws_n30`, and `ws_n40` runs also have `best_f1_on_valid` evaluations;
those change threshold-dependent metrics but not AUROC/AUPRC.

## Ablation Definitions

| Run | Training subgraph node range | Config / source |
|---|---:|---|
| `20260308_002906_0.5` | none | Base v3 checkpoint, no topology finetune. |
| `ws_n20_0.5` | 20-20 | `configs/v3/ablations/0407/ws_n20.yaml` |
| `ws_n30_0.5` | 30-30 | `configs/v3/ablations/0407/ws_n30.yaml` |
| `ws_n40_0.5` | 40-40 | `configs/v3/ablations/0407/ws_n40.yaml` |
| `ws_n60` | 60-60 | `logs/v3/topology_finetune/ws_n60/` |
| `ws_n100` | 100-100 | `logs/v3/topology_finetune/ws_n100/` |
| `ws_range_n20_60` | 20-60 | `configs/v3/ablations/0426/ws_range_n20_60.yaml` |
| `ws_range_n20_100` | 20-100 | `configs/v3/ablations/0426/ws_range_n20_100.yaml` |
| `ws_range_n60_100` | 60-100 | `configs/v3/ablations/0426/ws_range_n60_100.yaml` |

## Pairwise Test Metrics

| Run | AUROC | AUPRC | Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260308_002906_0.5` | 0.676074 | 0.690534 | 0.601271 | 0.320585 | 0.882103 | 0.731226 | 0.445746 | 0.244940 |
| `ws_n20_0.5` | 0.680589 | 0.698437 | 0.620325 | 0.447918 | 0.792820 | 0.683854 | 0.541294 | 0.256470 |
| `ws_n30_0.5` | 0.685303 | 0.700604 | 0.617793 | 0.391494 | 0.844209 | 0.715443 | 0.506066 | 0.264333 |
| `ws_n40_0.5` | 0.685163 | 0.702787 | 0.608737 | 0.313823 | 0.903804 | 0.765480 | 0.445149 | 0.269522 |
| `ws_n60` | 0.685138 | 0.705907 | 0.597908 | 0.253520 | 0.942475 | 0.815136 | 0.386753 | 0.270392 |
| `ws_n100` | 0.677902 | 0.697702 | 0.583992 | 0.212837 | 0.955340 | 0.826636 | 0.338515 | 0.251052 |
| `ws_range_n20_60` | **0.686804** | **0.706372** | 0.599904 | 0.262203 | 0.937780 | 0.808293 | 0.395960 | **0.271223** |
| `ws_range_n20_100` | 0.685998 | 0.706046 | 0.596873 | 0.250527 | 0.943399 | 0.815786 | 0.383332 | 0.268927 |
| `ws_range_n60_100` | 0.675251 | 0.696878 | 0.573995 | 0.175664 | 0.972531 | 0.864835 | 0.292015 | 0.245274 |

## Topology Metrics

Topology metrics use the `summary,all` row in each `topology_metrics.csv`.
Higher `graph_sim` is better; `relative_density` is best near 1; MMD metrics are
better when lower.

| Run | Graph sim | Relative density | Degree MMD | Clustering MMD | Laplacian MMD |
|---|---:|---:|---:|---:|---:|
| `20260308_002906_0.5` | 0.309202 | 1.385441 | 0.077928 | 0.103679 | 0.064321 |
| `ws_n20_0.5` | 0.323767 | 2.210997 | 0.159719 | 0.180892 | 0.098124 |
| `ws_n30_0.5` | **0.328467** | 1.660682 | 0.094690 | 0.127610 | 0.072996 |
| `ws_n40_0.5` | 0.328065 | **1.109124** | **0.042056** | **0.065156** | **0.060572** |
| `ws_n60` | 0.318451 | 0.791441 | 4.820103 | 4.319203 | 7.491459 |
| `ws_n100` | 0.298604 | 0.644049 | 7.431137 | 5.740243 | 9.606855 |
| `ws_range_n20_60` | 0.320370 | 0.828493 | 4.656764 | 4.518346 | 7.168434 |
| `ws_range_n20_100` | 0.317305 | 0.764440 | 5.175492 | 4.336275 | 7.778930 |
| `ws_range_n60_100` | 0.280779 | 0.478024 | 14.657775 | 10.346642 | 16.406892 |

## Interpretation

`ws_range_n20_60` is the best pairwise predictor in this sweep: it gives the top
AUROC, AUPRC, and MCC at threshold `0.5`. It improves over the base checkpoint by
+0.010730 AUROC, +0.015839 AUPRC, and +0.026283 MCC.

`ws_n40_0.5` is the cleanest topology compromise. It is close to the best graph
similarity, has relative density closest to 1, and has the lowest degree,
clustering, and Laplacian MMD. Smaller exact subgraphs (`n20`, `n30`) increase
graph similarity but produce over-dense predicted graphs. Larger exact or range
subgraphs (`n60`, `n100`, `n20_60`, `n20_100`, `n60_100`) improve pair ranking in
some cases but produce sparse graphs with much worse distribution MMD.

Practical takeaway: use `ws_range_n20_60` when pairwise AUROC/AUPRC is the
objective, but use `ws_n40_0.5` as the better topology-preserving checkpoint.
Avoid `ws_range_n60_100`; it is worse than the base checkpoint on AUROC and only
barely above base on AUPRC/MCC while badly degrading topology metrics.

## Incomplete Runs

`ws_n80` has partial topology-finetuning logs but no completed
`evaluate.csv` or `topology_metrics.csv`; exclude it from ranking. The `ws_n50`
job in `logs/v3/slurm_896057.err` was cancelled by the time limit before a usable
result file was written.
