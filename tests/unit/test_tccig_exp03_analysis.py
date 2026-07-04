"""Tests for exp03 validation-first analysis reporting."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from tccig.analyze_exp03 import (
    EXP02_REFERENCE_RUN_ID,
    PHASE_A_RUN_IDS,
    collect_run_row,
    main,
    write_exp03_report,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_run_fixture(root: Path, run_id: str, *, with_heldout: bool) -> Path:
    run_dir = root / run_id
    _write_json(
        run_dir / "training_summary.json",
        {
            "monitor_metric": "val_topology_loss",
            "selected_rule": {
                "type": "threshold",
                "value": 0.96,
                "source": "validation_calibration",
            },
            "history": [
                {
                    "epoch": 1,
                    "monitor_value": 9.0,
                    "val_auprc": 0.68,
                    "val_topology_loss": 9.0,
                    "internal_val_graph_sim": 0.70,
                    "internal_val_relative_density": 1.40,
                    "internal_val_deg_dist_mmd": 0.20,
                    "internal_val_cc_mmd": 0.10,
                    "selected_rule_positive_edges": 140,
                    "selected_rule": {
                        "type": "threshold",
                        "value": 0.95,
                        "source": "validation_calibration",
                    },
                    "train_bce_loss": 0.30,
                    "train_topology_loss": 4.5,
                    "train_topo_graph_sim": 0.40,
                    "train_topo_relative_density": 1.4,
                    "train_topo_degree_mmd": 0.2,
                    "sampled_edge_targets": 300,
                    "train_fp_targets": 10,
                    "train_fn_targets": 20,
                },
                {
                    "epoch": 40,
                    "monitor_value": 7.0,
                    "val_auprc": 0.675,
                    "val_topology_loss": 7.0,
                    "internal_val_graph_sim": 0.75,
                    "internal_val_relative_density": 1.08,
                    "internal_val_deg_dist_mmd": 0.15,
                    "internal_val_cc_mmd": 0.08,
                    "selected_rule_positive_edges": 120,
                    "selected_rule": {
                        "type": "threshold",
                        "value": 0.96,
                        "source": "validation_calibration",
                    },
                    "train_bce_loss": 0.25,
                    "train_topology_loss": 3.0,
                    "train_topo_graph_sim": 0.50,
                    "train_topo_relative_density": 1.1,
                    "train_topo_degree_mmd": 0.15,
                    "sampled_edge_targets": 300,
                    "train_fp_targets": 10,
                    "train_fn_targets": 20,
                },
            ],
        },
    )
    _write_json(
        run_dir / "threshold_grid" / "best_epoch.json",
        {
            "epoch": 40,
            "selected_rule": {
                "type": "threshold",
                "value": 0.96,
                "source": "validation_calibration",
            },
            "rows": [
                {"rule": {"type": "threshold", "value": 0.95}, "val_topology_loss": 8.0},
                {"rule": {"type": "threshold", "value": 0.96}, "val_topology_loss": 7.0},
            ],
        },
    )
    if with_heldout:
        _write_json(
            run_dir / "pairwise_test" / "refined_metrics.json",
            {"precision": 0.9, "recall": 0.2, "f1": 0.33, "auprc": 0.7, "auroc": 0.8},
        )
        _write_json(run_dir / "pairwise_test" / "raw_metrics.json", {"auprc": 0.69, "auroc": 0.79})
        _write_json(
            run_dir / "topology_test" / "topology_metrics.json",
            {
                "summary": {
                    "relative_density": 0.95,
                    "graph_sim": 0.8,
                    "deg_dist_mmd": 0.1,
                    "cc_mmd": 0.05,
                },
                "deletion_diagnostics": {"edges_added": 1.0, "edges_deleted": 2.0},
                "protocol": {
                    "candidate_universe": "all_test_ppi.txt",
                    "test_labels_visible_to_model": False,
                },
            },
        )
    return run_dir


def _write_raw_baseline_fixture(root: Path) -> Path:
    artifact_dir = root / "raw_pairwise_topology_baseline"
    _write_json(
        artifact_dir / "topology_metrics.json",
        {
            "summary": {
                "relative_density": 0.88,
                "graph_sim": 0.77,
                "deg_dist_mmd": 0.12,
                "cc_mmd": 0.07,
            },
            "protocol": {
                "candidate_universe": "all_test_ppi.txt",
                "test_labels_visible_to_model": False,
            },
        },
    )
    return artifact_dir


def test_collect_run_row_uses_training_and_threshold_grid_without_heldout(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path, "03_a5_bce_full_topology", with_heldout=True)

    row = collect_run_row(
        run_id="03_a5_bce_full_topology",
        run_dir=run_dir,
        include_heldout=False,
    )

    assert row["run_id"] == "03_a5_bce_full_topology"
    assert row["best_epoch"] == 40
    assert row["selected_threshold"] == 0.96
    assert row["validation_selected_edges"] == 120
    assert row["validation_selected_edges_min"] == 120
    assert row["validation_selected_edges_max"] == 140
    assert row["validation_selected_edges_range"] == 20
    assert row["sampled_edge_targets"] == 300
    assert row["train_topology_loss_epoch_1"] == 4.5
    assert row["train_topology_loss_epoch_7"] is None
    assert row["train_topology_loss_epoch_40"] == 3.0
    assert row["threshold_grid_rows"] == 2
    assert row["threshold_grid_best_epoch"] == 40
    assert row["gate_val_auprc_ok"] is True
    assert row["gate_selected_edges_stable"] is True
    assert row["gate_density_closer_than_exp02"] is True
    assert row["eligible_for_locked_test"] is True
    assert not any(key.startswith("heldout_") for key in row)


def test_collect_run_row_includes_heldout_only_when_enabled(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path, "03_locked", with_heldout=True)
    raw_baseline_artifact_dir = _write_raw_baseline_fixture(
        tmp_path / "03_locked_raw_pairwise_baseline"
    )

    row = collect_run_row(
        run_id="03_locked",
        run_dir=run_dir,
        include_heldout=True,
        raw_baseline_artifact_dir=raw_baseline_artifact_dir,
    )

    assert row["heldout_refined_precision"] == 0.9
    assert row["heldout_refined_recall"] == 0.2
    assert row["heldout_refined_f1"] == 0.33
    assert row["heldout_refined_auprc"] == 0.7
    assert row["heldout_refined_auroc"] == 0.8
    assert row["heldout_raw_auprc"] == 0.69
    assert row["heldout_raw_auroc"] == 0.79
    assert row["heldout_topology_relative_density"] == 0.95
    assert row["heldout_topology_graph_sim"] == 0.8
    assert row["heldout_topology_deg_dist_mmd"] == 0.1
    assert row["heldout_topology_cc_mmd"] == 0.05
    assert row["heldout_raw_topology_relative_density"] == 0.88
    assert row["heldout_raw_topology_graph_sim"] == 0.77
    assert row["heldout_raw_topology_deg_dist_mmd"] == 0.12
    assert row["heldout_raw_topology_cc_mmd"] == 0.07
    assert row["heldout_protocol_candidate_universe"] == "all_test_ppi.txt"
    assert row["heldout_protocol_test_labels_visible_to_model"] is False
    assert row["heldout_raw_protocol_candidate_universe"] == "all_test_ppi.txt"
    assert row["heldout_raw_protocol_test_labels_visible_to_model"] is False
    assert row["heldout_edges_added"] == 1.0
    assert row["heldout_edges_deleted"] == 2.0


def test_collect_run_row_fails_when_requested_raw_baseline_metrics_missing(
    tmp_path: Path,
) -> None:
    run_dir = _write_run_fixture(tmp_path, "03_locked", with_heldout=True)
    raw_baseline_artifact_dir = tmp_path / "missing_raw_baseline"
    raw_baseline_artifact_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing raw topology baseline metrics"):
        collect_run_row(
            run_id="03_locked",
            run_dir=run_dir,
            include_heldout=True,
            raw_baseline_artifact_dir=raw_baseline_artifact_dir,
        )


def test_write_exp03_report_outputs_json_csv_and_markdown(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path / "logs", "03_a5_bce_full_topology", with_heldout=False)
    reference_dir = _write_run_fixture(
        tmp_path / "logs",
        EXP02_REFERENCE_RUN_ID,
        with_heldout=False,
    )
    output_dir = tmp_path / "analysis"
    row = collect_run_row(
        run_id="03_a5_bce_full_topology",
        run_dir=run_dir,
        include_heldout=False,
    )
    reference_row = collect_run_row(
        run_id=EXP02_REFERENCE_RUN_ID,
        run_dir=reference_dir,
        include_heldout=False,
    )

    outputs = write_exp03_report(rows=[reference_row, row], output_dir=output_dir)

    assert outputs["json"].exists()
    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    candidate = next(
        item for item in payload["rows"] if item["run_id"] == "03_a5_bce_full_topology"
    )
    assert candidate["gate_non_worse_reference_metrics"] is True
    assert candidate["sampled_edge_targets"] == 300
    csv_reader = csv.DictReader(outputs["csv"].open("r", encoding="utf-8"))
    assert csv_reader.fieldnames is not None
    assert not any(name.startswith("heldout_") for name in csv_reader.fieldnames)
    csv_rows = list(csv_reader)
    assert csv_rows[1]["sampled_edge_targets"] == "300"
    assert csv_rows[1]["train_bce_loss_epoch_1"] == "0.3"
    assert csv_rows[1]["train_topology_loss_epoch_1"] == "4.5"
    assert csv_rows[1]["train_bce_loss_epoch_7"] == ""
    assert csv_rows[1]["train_topology_loss_epoch_7"] == ""
    assert csv_rows[1]["train_bce_loss_epoch_40"] == "0.25"
    assert csv_rows[1]["train_topology_loss_epoch_40"] == "3.0"
    assert csv_rows[1]["run_id"] == "03_a5_bce_full_topology"
    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert markdown.startswith("# Validation-first exp03 summary")
    assert "Held-out metrics were not included" in markdown


def test_write_exp03_report_includes_heldout_csv_headers_when_present(tmp_path: Path) -> None:
    run_dir = _write_run_fixture(tmp_path / "logs", "03_locked", with_heldout=True)
    raw_baseline_artifact_dir = _write_raw_baseline_fixture(
        tmp_path / "logs" / "03_locked_raw_pairwise_baseline"
    )
    row = collect_run_row(
        run_id="03_locked",
        run_dir=run_dir,
        include_heldout=True,
        raw_baseline_artifact_dir=raw_baseline_artifact_dir,
    )

    outputs = write_exp03_report(rows=[row], output_dir=tmp_path / "analysis")

    csv_reader = csv.DictReader(outputs["csv"].open("r", encoding="utf-8"))
    assert csv_reader.fieldnames is not None
    assert "heldout_refined_precision" in csv_reader.fieldnames
    assert "heldout_raw_topology_relative_density" in csv_reader.fieldnames
    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "## Held-out protocol" in markdown
    assert "heldout_protocol_candidate_universe" in markdown
    assert "heldout_protocol_test_labels_visible_to_model" in markdown
    assert "heldout_raw_protocol_candidate_universe" in markdown
    assert "heldout_raw_protocol_test_labels_visible_to_model" in markdown
    assert "all_test_ppi.txt" in markdown
    assert "False" in markdown


def test_main_fails_when_phase_a_run_directory_is_missing(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError,
        match=f"Missing required Phase A run directory: .*{PHASE_A_RUN_IDS[0]}",
    ):
        main(
            [
                "--log-root",
                str(tmp_path / "logs"),
                "--output-dir",
                str(tmp_path / "analysis"),
            ]
        )
