# Experiment Tracker

Run config: `configs/tccig/01.yaml` · Launch: `scripts/tccig.sh` · Benchmark: PRING Human/BFS

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | Sanity: pipeline runs end-to-end, IO/metric contract correct, scorer cache populates, self-pairs dropped | `01.yaml` fork, `epochs: 2`, 1 GPU | train/val (short) | edge counts, NaN check, artifact presence | MUST | TODO | Verify `pairwise_test` + `topology_test` artifacts write; check τ_pair reaches precision ≥ 0.8 on val |
| R002 | M1 | Reproduce frozen pairwise baseline + its topology metrics | v3.1 scorer @ `logs/tccig/pairwise_baseline` | test | precision/recall/F1/AUPRC; graph_sim, rel_density, deg/cc/spectral MMD | MUST | TODO | Artifact not present locally — confirm on HPC; regenerate from frozen scorer if absent |
| R003 | M2 | Main method: full refiner training to convergence | `01.yaml`, 40 epochs, 4× A40 DDP | train/val → test | monitor `val_topology_loss`; test pairwise + topology | MUST | TODO | Primary run (Block 1 / C1). ≤4-day SBATCH wall. Confirm monitor tracks test topology |
| R004 | M3 | Anti-threshold isolation: no scorer threshold matches refiner topology | Frozen scorer threshold sweep (post-hoc on cached scores) | test | topology summary at density/precision-matched points | MUST | TODO | Block 2 / C2. Fix matching variable (density vs precision) up front; CPU recompute |
| R005 | M4 | Per-node-size tables, topology curves, refined-output-threshold sensitivity | Re-use R003 outputs | test | per-node-size topology details | NICE | TODO | Appendix evidence; no new training |
| R006 | M4 | Deletion study (a): − residual anchor | `02.yaml` fork, `residual_weight: 0` | train/val → test | topology + pairwise deltas vs R003 | NICE | TODO | Defer to config `02`; only if C1 needs reinforcing |
| R007 | M4 | Deletion study (b): − scorer-error sampling (exhaustive train-pair BCE) | `02.yaml` fork, pre-ADR-002 contract | train/val → test | topology + pairwise deltas vs R003 | NICE | TODO | Defer to config `02` |
| R008 | M4 | Deletion study (c): − cross-layer decoder (inner-product decode) | `02.yaml` fork, simple decoder | train/val → test | topology + pairwise deltas vs R003 | NICE | TODO | Defer to config `02` |

Status legend: TODO · RUNNING · DONE · BLOCKED
