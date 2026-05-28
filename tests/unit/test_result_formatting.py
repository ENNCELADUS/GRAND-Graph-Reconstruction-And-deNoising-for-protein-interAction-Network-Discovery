"""Tests for persisted numeric result formatting."""

from __future__ import annotations

from csv import DictReader
from pathlib import Path

from src.topology.report import write_human_table2_reports
from src.utils.logging import append_csv_row, format_result_payload, format_stage_event


def test_stage_result_outputs_use_three_decimal_places(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"

    append_csv_row(
        csv_path=csv_path,
        row={
            "Epoch": 1,
            "Train Loss": 0.927643,
            "Val auprc": 0.123456,
            "Learning Rate": 5e-5,
        },
        fieldnames=["Epoch", "Train Loss", "Val auprc", "Learning Rate"],
    )

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(DictReader(handle))

    assert rows == [
        {
            "Epoch": "1",
            "Train Loss": "0.928",
            "Val auprc": "0.123",
            "Learning Rate": "5e-05",
        }
    ]

    message = format_stage_event("epoch_done", epoch=1, loss=0.927643, lr=5e-5)
    assert "Loss: 0.928" in message
    assert "LR: 5e-05" in message


def test_human_table2_report_outputs_use_three_decimal_places(tmp_path: Path) -> None:
    strategy_metrics = {
        "BFS": {
            "graph_sim": 0.87654,
            "relative_density": 1.23456,
            "deg_dist_mmd": 0.34567,
            "cc_mmd": 0.45678,
            "laplacian_eigen_mmd": 0.56789,
        },
        "DFS": {
            "graph_sim": 0.11111,
            "relative_density": 1.22222,
            "deg_dist_mmd": 0.33333,
            "cc_mmd": 0.44444,
            "laplacian_eigen_mmd": 0.55555,
        },
        "RANDOM_WALK": {
            "graph_sim": 0.66666,
            "relative_density": 1.77777,
            "deg_dist_mmd": 0.88888,
            "cc_mmd": 0.99999,
            "laplacian_eigen_mmd": 0.12345,
        },
    }

    csv_path, markdown_path = write_human_table2_reports(
        output_dir=tmp_path,
        baselines=[],
        model_name="v3",
        model_category="GRAND",
        strategy_metrics=strategy_metrics,
    )

    csv_text = csv_path.read_text(encoding="utf-8")
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "0.877" in csv_text
    assert "0.877" in markdown_text
    assert "0.87654" not in csv_text
    assert "0.8765" not in markdown_text


def test_json_result_payloads_round_floats_to_three_decimals() -> None:
    payload = {
        "summary": {"auprc": 0.87654, "graph_count": 2},
        "details": {"graph_sim": [0.12345, 0.99999]},
    }

    formatted = format_result_payload(payload)

    assert formatted == {
        "summary": {"auprc": 0.877, "graph_count": 2},
        "details": {"graph_sim": [0.123, 1.0]},
    }
