"""Unit tests for PRING-style topology evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest
from src.run.stage_topology_evaluate import (
    _ordered_predictions_from_shards,
    write_topology_predictions,
)
from src.topology.metrics import (
    compute_graph_similarity,
    compute_relative_density,
    evaluate_predicted_graph,
)
from src.topology.report import (
    build_human_table2_rows,
    load_human_table2_baselines,
    write_human_table2_reports,
)


def test_compute_graph_similarity_matches_official_formula() -> None:
    gt_graph = nx.Graph()
    gt_graph.add_edges_from([("A", "B"), ("B", "C")])
    pred_graph = nx.Graph()
    pred_graph.add_edges_from([("A", "B"), ("A", "C")])

    score = compute_graph_similarity(pred_graph=pred_graph, gt_graph=gt_graph)

    assert score == pytest.approx(0.5)


def test_compute_relative_density_matches_official_formula() -> None:
    gt_graph = nx.Graph()
    gt_graph.add_edges_from([("A", "B"), ("B", "C")])
    pred_graph = nx.Graph()
    pred_graph.add_edges_from([("A", "B")])
    pred_graph.add_node("C")

    relative_density = compute_relative_density(pred_graph=pred_graph, gt_graph=gt_graph)

    assert relative_density == pytest.approx(0.5)


def test_evaluate_predicted_graph_returns_official_summary_shape() -> None:
    gt_graph = nx.Graph()
    gt_graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
    pred_graph = nx.Graph()
    pred_graph.add_edges_from([("A", "B"), ("B", "C")])
    sampled_nodes = {
        3: [["A", "B", "C"], ["B", "C", "D"]],
        4: [["A", "B", "C", "D"]],
    }

    result = evaluate_predicted_graph(
        pred_graph=pred_graph,
        gt_graph=gt_graph,
        test_graph_nodes=sampled_nodes,
    )

    assert set(result.keys()) == {"details", "summary", "per_node_size"}
    assert set(result["details"].keys()) == {
        "graph_sim",
        "relative_density",
        "deg_dist_mmd",
        "cc_mmd",
        "laplacian_eigen_mmd",
    }
    assert set(result["summary"].keys()) == {
        "graph_sim",
        "relative_density",
        "deg_dist_mmd",
        "cc_mmd",
        "laplacian_eigen_mmd",
    }
    assert result["per_node_size"][3]["graph_count"] == 2
    assert result["per_node_size"][4]["graph_count"] == 1


def test_write_topology_predictions_emits_pring_format(tmp_path: Path) -> None:
    output_path = tmp_path / "all_test_ppi_pred.txt"

    write_topology_predictions(
        output_path=output_path,
        records=[("P1", "P2"), ("P3", "P4"), ("P5", "P6")],
        predictions=[1, 0, 1],
    )

    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "P1\tP2\t1",
        "P3\tP4\t0",
        "P5\tP6\t1",
    ]


def test_ordered_predictions_from_shards_restores_original_pair_order() -> None:
    ordered_predictions = _ordered_predictions_from_shards(
        total_records=5,
        shard_payloads=[
            {"indices": [0, 2, 4], "predictions": [1, 0, 1]},
            {"indices": [1, 3], "predictions": [0, 1]},
        ],
    )

    assert ordered_predictions == [1, 0, 0, 1, 1]


def test_ordered_predictions_from_shards_rejects_incomplete_results() -> None:
    with pytest.raises(ValueError, match="Missing topology predictions"):
        _ordered_predictions_from_shards(
            total_records=3,
            shard_payloads=[{"indices": [0, 2], "predictions": [1, 0]}],
        )


def test_build_human_table2_rows_merges_baselines_and_v3() -> None:
    baselines = [
        {
            "category": "Seq. Sim.",
            "model": "SPRINT",
            "metrics": {
                "BFS": {
                    "graph_sim": 0.2,
                    "relative_density": 1.1,
                    "deg_dist_mmd": 5.0,
                    "cc_mmd": 4.0,
                    "laplacian_eigen_mmd": 3.0,
                },
                "DFS": {
                    "graph_sim": 0.1,
                    "relative_density": 1.2,
                    "deg_dist_mmd": 6.0,
                    "cc_mmd": 5.0,
                    "laplacian_eigen_mmd": 4.0,
                },
                "RANDOM_WALK": {
                    "graph_sim": 0.3,
                    "relative_density": 1.0,
                    "deg_dist_mmd": 4.0,
                    "cc_mmd": 3.0,
                    "laplacian_eigen_mmd": 2.0,
                },
            },
        }
    ]
    v3_metrics = {
        "BFS": {
            "graph_sim": 0.9,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.1,
            "cc_mmd": 0.1,
            "laplacian_eigen_mmd": 0.1,
        },
        "DFS": {
            "graph_sim": 0.8,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.2,
            "cc_mmd": 0.2,
            "laplacian_eigen_mmd": 0.2,
        },
        "RANDOM_WALK": {
            "graph_sim": 0.85,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.3,
            "cc_mmd": 0.3,
            "laplacian_eigen_mmd": 0.3,
        },
    }

    rows = build_human_table2_rows(
        baselines=baselines,
        model_name="v3",
        model_category="GRAND",
        strategy_metrics=v3_metrics,
    )

    assert [row["model"] for row in rows] == ["v3", "SPRINT"]
    assert rows[0]["avg_rank"] == 1
    assert rows[1]["avg_rank"] == 2
    assert rows[0]["bfs_graph_sim"] == pytest.approx(0.9)
    assert rows[0]["rw_spectral"] == pytest.approx(0.3)


def test_load_baselines_and_write_reports(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baselines.json"
    baseline_path.write_text(
        json.dumps(
            {
                "source": "unit-test",
                "rows": [
                    {
                        "category": "Seq. Sim.",
                        "model": "SPRINT",
                        "metrics": {
                            "BFS": {
                                "graph_sim": 0.2,
                                "relative_density": 1.1,
                                "deg_dist_mmd": 5.0,
                                "cc_mmd": 4.0,
                                "laplacian_eigen_mmd": 3.0,
                            },
                            "DFS": {
                                "graph_sim": 0.1,
                                "relative_density": 1.2,
                                "deg_dist_mmd": 6.0,
                                "cc_mmd": 5.0,
                                "laplacian_eigen_mmd": 4.0,
                            },
                            "RANDOM_WALK": {
                                "graph_sim": 0.3,
                                "relative_density": 1.0,
                                "deg_dist_mmd": 4.0,
                                "cc_mmd": 3.0,
                                "laplacian_eigen_mmd": 2.0,
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"
    strategy_metrics = {
        "BFS": {
            "graph_sim": 0.9,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.1,
            "cc_mmd": 0.1,
            "laplacian_eigen_mmd": 0.1,
        },
        "DFS": {
            "graph_sim": 0.8,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.2,
            "cc_mmd": 0.2,
            "laplacian_eigen_mmd": 0.2,
        },
        "RANDOM_WALK": {
            "graph_sim": 0.85,
            "relative_density": 1.0,
            "deg_dist_mmd": 0.3,
            "cc_mmd": 0.3,
            "laplacian_eigen_mmd": 0.3,
        },
    }

    baselines = load_human_table2_baselines(baseline_path)
    csv_path, markdown_path = write_human_table2_reports(
        output_dir=output_dir,
        baselines=baselines,
        model_name="v3",
        model_category="GRAND",
        strategy_metrics=strategy_metrics,
    )

    assert csv_path.exists()
    assert markdown_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "v3" in csv_text
    assert "SPRINT" in csv_text
    assert "| v3 |" in markdown_text
