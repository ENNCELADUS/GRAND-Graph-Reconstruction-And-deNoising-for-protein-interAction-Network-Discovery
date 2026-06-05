# v3 Topology Loss Ablation

Run dates: 2026-04-27

Model card: TBD (add later)

Configs:
- `configs/v3/ablations/0427_loss_ablation/*.yaml`

Logs:
- `logs/v3/topology_finetune/topo_*/`
- `logs/v3/evaluate/topo_*/`
- `logs/v3/topology_evaluate/topo_*/`

## Run Setup

Seed-47 v3 topology-loss-weight ablation on PRING Human BFS. The intended controlled setup holds the topology-finetune subgraph range fixed at `[30, 40]` and changes the relative contribution of graph-similarity, relative-density, degree-MMD, clustering-MMD, and BCE terms.

Rows with local `evaluate.csv` report pairwise AUROC/AUPRC/MCC. Rows without local `evaluate.csv` are retained as topology-only results and use `--` for pairwise metrics. Topology metrics come from `topology_metrics.csv` when present and `topology_metrics.json` otherwise.

## Loss Formula

For one sampled topology-finetune subgraph $S$ with upper-triangle protein pairs $\mathcal{P}(S)$, the model predicts pair probabilities $p_{ij}=\sigma(z_{ij})$ and uses topology labels $y_{ij}\in\{0,1\}$. The implemented training objective is:

$$
\mathcal{L}_{\mathrm{train}}
=
\mathcal{L}_{\mathrm{BCE}}
+ \alpha \mathcal{L}_{\mathrm{GS}}
+ \beta \mathcal{L}_{\mathrm{RD}}
+ \gamma \mathcal{L}_{\mathrm{Deg}}
+ \delta I_{\mathrm{CC}}\mathcal{L}_{\mathrm{CC}} .
$$

In this ablation family, `loss_weight_schedule` has `warmup_epochs: 0` and `ramp_epochs: 0`, loss normalization and GradNorm are not configured, so the schedule scale and grouped task weights are both effectively 1. All configs also set `compute_clustering_mmd: false`, so $I_{\mathrm{CC}}=0$ and clustering-MMD is not backpropagated even when the YAML keeps a nonzero `delta`.

The loss terms are:

$$
\mathcal{L}_{\mathrm{BCE}}
=
\frac{\sum_{\ell} m_\ell\,\mathrm{BCEWithLogits}(z_\ell,y_\ell)}
{\sum_{\ell} m_\ell},
$$

where the supervised BCE pairs are the assigned positive train edges plus explicit negatives sampled at `bce_negative_ratio: 5`.

$$
\mathcal{L}_{\mathrm{GS}}
=
\frac{\sum_{(i,j)\in\mathcal{P}(S)} |p_{ij}-y_{ij}|}
{\sum_{(i,j)\in\mathcal{P}(S)}p_{ij}+\sum_{(i,j)\in\mathcal{P}(S)}y_{ij}+\epsilon}.
$$

$$
\rho_{\mathrm{pred}}=\frac{2\sum_{(i,j)\in\mathcal{P}(S)}p_{ij}}{|S|(|S|-1)},
\qquad
\rho_{\mathrm{true}}=\frac{2\sum_{(i,j)\in\mathcal{P}(S)}y_{ij}}{|S|(|S|-1)}.
$$

With the default `rd_loss_form: log_ratio_huber`,

$$
\mathcal{L}_{\mathrm{RD}}
=
\mathrm{SmoothL1}
\left(
\log\frac{\rho_{\mathrm{pred}}+\epsilon}{\rho_{\mathrm{true}}+\epsilon},
0
\right).
$$

For degree and clustering distribution terms, the implementation builds normalized Gaussian soft histograms and compares them with the Gaussian-TV MMD:

$$
\mathrm{MMD}(h_{\mathrm{pred}},h_{\mathrm{true}})
=
2 - 2\exp
\left(
-\frac{\mathrm{TV}(h_{\mathrm{pred}},h_{\mathrm{true}})^2}{2\sigma^2}
\right),
\qquad
\mathrm{TV}(a,b)=\frac{1}{2}\|a-b\|_1 .
$$

Thus $\mathcal{L}_{\mathrm{Deg}}$ applies this MMD to soft-degree histograms, and $\mathcal{L}_{\mathrm{CC}}$ applies it to soft local-clustering-coefficient histograms.

## Ablation Definitions

All rows warm-start from `models/v3/train/20260308_002906/best_model.pth`, use seed 47, fixed threshold 0.5, `subgraph_node_range: [30, 40]`, and `bce_negative_ratio: 5`.

| Run | Effective training formula | Ablation purpose |
|---|---|---|
| `topo_baseline_n30_40` | $\mathcal{L}_{\mathrm{BCE}} + 0.35\mathcal{L}_{\mathrm{GS}} + 0.45\mathcal{L}_{\mathrm{RD}} + 0.15\mathcal{L}_{\mathrm{Deg}}$ | Control run: balanced graph-similarity, density, and degree-shape topology terms at the fixed `[30,40]` subgraph range. |
| `topo_gs_only` | $\mathcal{L}_{\mathrm{BCE}} + 0.35\mathcal{L}_{\mathrm{GS}}$ | Removes density and distribution-shape terms to test whether graph-similarity alone can control the generated graph. |
| `topo_gs_rd` | $\mathcal{L}_{\mathrm{BCE}} + 0.35\mathcal{L}_{\mathrm{GS}} + 0.45\mathcal{L}_{\mathrm{RD}}$ | Adds relative density back to graph similarity, isolating whether density control is enough without degree MMD. No complete local topology-evaluation result is available. |
| `topo_gs_rd_bce_high` | $\mathcal{L}_{\mathrm{BCE}} + 0.175\mathcal{L}_{\mathrm{GS}} + 0.225\mathcal{L}_{\mathrm{RD}}$ | Halves the topology weights relative to `topo_gs_rd`, increasing the relative influence of BCE supervision. |
| `topo_gs_rd_bce_low` | $\mathcal{L}_{\mathrm{BCE}} + 0.70\mathcal{L}_{\mathrm{GS}} + 0.90\mathcal{L}_{\mathrm{RD}}$ | Doubles the topology weights relative to `topo_gs_rd`, lowering the relative influence of BCE supervision. |
| `topo_gs_rd_deg` | $\mathcal{L}_{\mathrm{BCE}} + 0.35\mathcal{L}_{\mathrm{GS}} + 0.45\mathcal{L}_{\mathrm{RD}} + 0.15\mathcal{L}_{\mathrm{Deg}}$ | Adds degree-distribution MMD on top of graph similarity and density, matching the effective control formula with speed-tuned validation settings. |
| `topo_rd_only` | $\mathcal{L}_{\mathrm{BCE}} + 0.45\mathcal{L}_{\mathrm{RD}}$ | Removes graph-similarity and degree-shape terms to test density-only topology supervision. |

## Test results

| Run | AUROC | AUPRC | MCC | Graph sim | Relative density | Degree MMD | Clustering MMD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_finetune` | 0.676 | 0.691 | 0.245 | 0.309 | 1.385 | 17.174 | 11.813 |
| `topo_baseline_n30_40` | 0.686 | 0.703 | 0.265 | 0.328 | 1.254 | 7.185 | 8.279 |
| `topo_gs_only` | 0.678 | 0.689 | 0.250 | 0.323 | 2.696 | 27.391 | 19.670 |
| `topo_gs_rd_bce_high` | 0.684 | 0.699 | 0.261 | 0.328 | 1.864 | 16.468 | 14.823 |
| `topo_gs_rd_bce_low` | -- | -- | -- | 0.327 | 1.061 | 5.486 | 6.601 |
| `topo_gs_rd_deg` | -- | -- | -- | 0.329 | 1.310 | 7.875 | 9.008 |
| `topo_rd_only` | -- | -- | -- | 0.329 | 1.333 | 8.334 | 9.489 |

## Main Readout

The baseline `[30,40]` loss mix remains the strongest row with complete pairwise metrics and improves normalized MMD over `no_finetune`. Among topology-only rows, `topo_gs_rd_bce_low` gives the best density control and lower distribution MMD than `topo_gs_rd_deg` or `topo_rd_only`, while `topo_gs_rd_deg` and `topo_rd_only` have slightly higher graph similarity. `topo_gs_only` remains over-dense, and `topo_gs_rd_bce_high` improves density relative to graph-similarity-only training but is still worse than the low-BCE topology-only row. `topo_gs_rd` still has no complete local topology-evaluation result, so it is excluded.
