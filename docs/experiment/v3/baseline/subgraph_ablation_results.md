# v3 Subgraph Size Ablation

Run dates: 2026-03-08, 2026-04-07, and 2026-04-26

Model card: TBD (add later)

Configs:
- Base checkpoint: `configs/v3/v3_20260308_002906.yaml`
- Fixed-size subgraphs: `configs/v3/ablations/0407/ws_n{20,30,40}.yaml`
- Range subgraphs: `configs/v3/ablations/0426/ws_range_n{20_60,20_100,60_100}.yaml`
- Some completed runs only have pulled log folders and no local YAML in this checkout.

Logs:
- `logs/v3/train/20260308_002906/`
- `logs/v3/topology_finetune/{ws_n20,ws_n30,ws_n40,ws_n60,ws_n100,ws_range_n20_60,ws_range_n20_100,ws_range_n60_100}/`
- `logs/v3/evaluate/{20260308_002906_0.5,ws_n20_0.5,ws_n30_0.5,ws_n40_0.5,ws_n60,ws_n100,ws_range_n20_60,ws_range_n20_100,ws_range_n60_100}/`
- `logs/v3/topology_evaluate/{20260308_002906_0.5,ws_n20_0.5,ws_n30_0.5,ws_n40_0.5,ws_n60,ws_n100,ws_range_n20_60,ws_range_n20_100,ws_range_n60_100}/`

## Run Setup

Seed-47 v3 topology-finetuning ablation over training subgraph size on PRING Human BFS. Completed topology-finetune runs warm-start from `models/v3/train/20260308_002906/best_model.pth`. The table uses fixed threshold `0.5` evaluations for comparability.

Topology columns use the `summary,all` row in `topology_metrics.csv`. Higher `graph_sim` is better; `relative_density` is best near 1; MMD columns are better when lower.

## Test results

| Run | AUROC | AUPRC | MCC | Graph sim | Relative density | Degree MMD | Clustering MMD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `20260308_002906_0.5` | 0.676 | 0.691 | 0.245 | 0.309 | 1.385 | 0.078 | 0.104 |
| `ws_n20_0.5` | 0.681 | 0.698 | 0.256 | 0.324 | 2.211 | 0.160 | 0.181 |
| `ws_n30_0.5` | 0.685 | 0.701 | 0.264 | 0.328 | 1.661 | 0.095 | 0.128 |
| `ws_n40_0.5` | 0.685 | 0.703 | 0.270 | 0.328 | 1.109 | 0.042 | 0.065 |
| `ws_n60` | 0.685 | 0.706 | 0.270 | 0.318 | 0.791 | 4.820 | 4.319 |
| `ws_n100` | 0.678 | 0.698 | 0.251 | 0.299 | 0.644 | 7.431 | 5.740 |
| `ws_range_n20_60` | 0.687 | 0.706 | 0.271 | 0.320 | 0.828 | 4.657 | 4.518 |
| `ws_range_n20_100` | 0.686 | 0.706 | 0.269 | 0.317 | 0.764 | 5.175 | 4.336 |
| `ws_range_n60_100` | 0.675 | 0.697 | 0.245 | 0.281 | 0.478 | 14.658 | 10.347 |

## Main Readout

`ws_range_n20_60` is the best pairwise run by AUROC, AUPRC, and MCC. `ws_n40_0.5` is the best topology-preserving compromise: graph similarity is near the best, relative density is closest to 1, and the distribution MMD values are lowest. Exclude `ws_n80` and `ws_n50` from ranking because the pulled logs do not include complete evaluation and topology-evaluation outputs.
