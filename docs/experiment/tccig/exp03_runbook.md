# TCCIG exp03 runbook

## Contract

exp03 is a diagnostic. Phase A and Phase B selection use training dynamics and validation metrics only. Held-out pairwise/topology test metrics are generated only after a candidate is locked.

Treat this runbook and `docs/superpowers/specs/2026-07-04-tccig-exp03-loss-conflict-diagnostic-design.md` as authoritative for exp03. Older threshold wording in `CONTEXT.md`, `tccig/README.md`, or `docs/experiment/tccig/model.md` may still describe fixed `p_refined >= 0.5`; exp03 uses calibrated `val_topology_loss` selection.

## Generate configs

```bash
rtk proxy uv run --locked --no-sync --offline python -m tccig.exp03_configs --base configs/tccig/02_balanced_subset.yaml --output-dir configs/tccig/exp03 --phase-b
```

Use `--sampler-phase-b` only after Phase A selects the FN/FP hard-quadrant sampling lever.

## Pre-launch checks

```bash
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/integration/test_tccig_orchestrator.py::test_tccig_orchestrator_can_skip_heldout_test_artifacts -v
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_exp03_configs.py tests/unit/test_tccig_topology_training.py::test_resolve_refined_output_rule_config_accepts_calibrated_grid -v
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/unit/test_tccig_exp03_configs.py::test_exp03_configs_only_change_intended_paths -v
rtk proxy uv run --locked --no-sync --offline python -m pytest tests/integration/test_tccig_orchestrator.py::test_calibrated_pipeline_writes_threshold_grid_artifacts -v
```

For every Phase A config, confirm:

- `refiner.monitor_metric: val_topology_loss`
- `refiner.topology_validation.enabled: true`
- `refiner.topology_validation.losses: {alpha: 1.0, beta: 8.0, gamma: 0.5, delta: 0.0}`
- `graph_selection.refined_output_rule.type: calibrated`
- `graph_selection.refined_output_rule.objective: val_topology_loss`
- `refiner.topology_training.topo_only_after_epoch: null` for `03_a2` through `03_a5`
- `refiner.topology_training.enabled: false` for `03_a1_bce_only`

## Phase A launch

Submit only `03_a1` through `03_a5`. `03_a0_exp02_topo_only_reference` is the exp02 artifact in `artifacts/exp02_rerun_fix/logs/tccig/02_balanced_subset`.

```bash
GRAND_TCCIG_SKIP_TEST_SPLITS=1 sbatch scripts/tccig.sh configs/tccig/exp03/03_a1_bce_only.yaml
GRAND_TCCIG_SKIP_TEST_SPLITS=1 sbatch scripts/tccig.sh configs/tccig/exp03/03_a2_bce_graph_sim.yaml
GRAND_TCCIG_SKIP_TEST_SPLITS=1 sbatch scripts/tccig.sh configs/tccig/exp03/03_a3_bce_density.yaml
GRAND_TCCIG_SKIP_TEST_SPLITS=1 sbatch scripts/tccig.sh configs/tccig/exp03/03_a4_bce_degree.yaml
GRAND_TCCIG_SKIP_TEST_SPLITS=1 sbatch scripts/tccig.sh configs/tccig/exp03/03_a5_bce_full_topology.yaml
```

Record Slurm job ids and stdout/stderr paths in the experiment tracker before analyzing results. Immediately after each submit, verify the job and log path:

```bash
squeue -j "$JOB_ID" -o "%.18i %.9P %.24j %.8u %.2t %.12M %.12l %.6D %R"
scontrol show job "$JOB_ID"
ls -lh "logs/tccig/slurm_${JOB_ID}.out" "logs/tccig/slurm_${JOB_ID}.err"
tail -n 80 "logs/tccig/slurm_${JOB_ID}.out"
tail -n 80 "logs/tccig/slurm_${JOB_ID}.err"
```

## Phase A analysis

Copy Phase A artifacts back from the HPC root before analysis:

```bash
REMOTE_ROOT=/public/home/wangar2023/grand
LOCAL_ROOT=/Users/richardwang/Documents/grand
mkdir -p "$LOCAL_ROOT/logs/tccig"
scp -r "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/03_a1_bce_only" "$LOCAL_ROOT/logs/tccig/"
scp -r "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/03_a2_bce_graph_sim" "$LOCAL_ROOT/logs/tccig/"
scp -r "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/03_a3_bce_density" "$LOCAL_ROOT/logs/tccig/"
scp -r "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/03_a4_bce_degree" "$LOCAL_ROOT/logs/tccig/"
scp -r "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/03_a5_bce_full_topology" "$LOCAL_ROOT/logs/tccig/"
scp "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/slurm_*.out" "$LOCAL_ROOT/logs/tccig/"
scp "wangar2023@10.15.89.192:${REMOTE_ROOT}/logs/tccig/slurm_*.err" "$LOCAL_ROOT/logs/tccig/"
```

Before running analysis, verify every Phase A run has validation artifacts and no clean held-out test artifacts:

```bash
test -f logs/tccig/03_a1_bce_only/training_summary.json
test -f logs/tccig/03_a1_bce_only/threshold_grid/best_epoch.json
test -f logs/tccig/03_a2_bce_graph_sim/training_summary.json
test -f logs/tccig/03_a2_bce_graph_sim/threshold_grid/best_epoch.json
test -f logs/tccig/03_a3_bce_density/training_summary.json
test -f logs/tccig/03_a3_bce_density/threshold_grid/best_epoch.json
test -f logs/tccig/03_a4_bce_degree/training_summary.json
test -f logs/tccig/03_a4_bce_degree/threshold_grid/best_epoch.json
test -f logs/tccig/03_a5_bce_full_topology/training_summary.json
test -f logs/tccig/03_a5_bce_full_topology/threshold_grid/best_epoch.json
test ! -d logs/tccig/03_a1_bce_only/pairwise_test
test ! -d logs/tccig/03_a1_bce_only/topology_test
test ! -d logs/tccig/03_a2_bce_graph_sim/pairwise_test
test ! -d logs/tccig/03_a2_bce_graph_sim/topology_test
test ! -d logs/tccig/03_a3_bce_density/pairwise_test
test ! -d logs/tccig/03_a3_bce_density/topology_test
test ! -d logs/tccig/03_a4_bce_degree/pairwise_test
test ! -d logs/tccig/03_a4_bce_degree/topology_test
test ! -d logs/tccig/03_a5_bce_full_topology/pairwise_test
test ! -d logs/tccig/03_a5_bce_full_topology/topology_test
```

Then run:

```bash
rtk proxy uv run --locked --no-sync --offline python -m tccig.analyze_exp03 --log-root logs/tccig --exp02-reference-dir artifacts/exp02_rerun_fix/logs/tccig/02_balanced_subset --output-dir analysis/tccig_exp03
```

Use `analysis/tccig_exp03/exp03_summary.md` to decide whether Phase B is justified. The validation AUPRC floor is `0.6705`.

## Phase B gate

Run Phase B only when all conditions hold:

- Phase A confirms BCE-vs-topology conflict or a clear component-level imbalance.
- At least one Phase A variant improves a topology metric without dropping below validation AUPRC `0.6705`.
- The best Phase A candidate still leaves a validation topology gap, selected-edge instability, or interpretable AUPRC tradeoff worth tuning.

Pick at most two levers from `03_b1` through `03_b5`, and launch at most four Phase B runs before review. Use `03_b6` only after implementing and approving the explicit sampler lever. Submit selected Phase B configs with `GRAND_TCCIG_SKIP_TEST_SPLITS=1` and run the same post-submit Slurm/log checks as Phase A.

## Locked-candidate held-out report

After locking one candidate by validation evidence, run pairwise/topology test once by launching the locked config without `GRAND_TCCIG_SKIP_TEST_SPLITS`. Then run raw pairwise topology baseline into a separate output run id:

```bash
LOCKED_RUN_ID=03_b1_beta2
sbatch scripts/tccig.sh "configs/tccig/exp03/${LOCKED_RUN_ID}.yaml"
GRAND_TCCIG_BASELINE_SOURCE_RUN_ID="$LOCKED_RUN_ID" GRAND_TCCIG_BASELINE_OUTPUT_RUN_ID="${LOCKED_RUN_ID}_raw_pairwise_baseline" sbatch scripts/tccig_pairwise_baseline.sh "configs/tccig/exp03/${LOCKED_RUN_ID}.yaml"
```

Regenerate the report with held-out fields for the locked candidate:

```bash
LOCKED_RUN_ID=03_b1_beta2
RAW_BASELINE_RUN_ID="${LOCKED_RUN_ID}_raw_pairwise_baseline"
rtk proxy uv run --locked --no-sync --offline python -m tccig.analyze_exp03 --log-root logs/tccig --exp02-reference-dir artifacts/exp02_rerun_fix/logs/tccig/02_balanced_subset --output-dir analysis/tccig_exp03 --include-phase-b --locked-run-id "$LOCKED_RUN_ID" --raw-baseline-run-id "$RAW_BASELINE_RUN_ID"
```

The locked report must show `heldout_protocol_candidate_universe=all_test_ppi.txt`, `heldout_protocol_test_labels_visible_to_model=False`, `heldout_raw_protocol_candidate_universe=all_test_ppi.txt`, and `heldout_raw_protocol_test_labels_visible_to_model=False`.

Do not compare raw `0.5` precision directly against refined calibrated-threshold precision as a model-quality claim. Use AUPRC/AUROC and matched operating points.
