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

Only runs with both `evaluate.csv` and `topology_metrics.csv` are included in the main table.

## Test results

| Run | AUROC | AUPRC | MCC | Graph sim | Relative density | Degree MMD | Clustering MMD |
|---|---:|---:|---:|---:|---:|---:|---:|
| `topo_baseline_n30_40` | 0.686 | 0.703 | 0.265 | 0.328 | 1.254 | 7.185 | 8.279 |
| `topo_gs_only` | 0.678 | 0.689 | 0.250 | 0.323 | 2.696 | 27.391 | 19.670 |
| `topo_gs_rd_bce_high` | 0.684 | 0.699 | 0.261 | 0.328 | 1.864 | 16.468 | 14.823 |

## Main Readout

The baseline `[30,40]` loss mix remains the strongest completed row by pairwise AUROC/AUPRC/MCC and has much better density control than `topo_gs_only` or `topo_gs_rd_bce_high`. `topo_gs_only` over-densifies the predicted graph. `topo_gs_rd_bce_low`, `topo_gs_rd_deg`, `topo_gs_rd`, and `topo_rd_only` have pulled fine-tune logs but no complete local `evaluate.csv`, so they are excluded from ranking.
