"""Unit tests for TCCIG test-time raw/refined export."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import networkx as nx
from tccig.prepare import GraphRule, LabeledPair, PairTable, TCCIGRuntime
from tccig.test import run_pairwise_test, run_raw_pairwise_topology_baseline


class _SingleProcessAccelerator:
    def wait_for_everyone(self) -> None:  # pragma: no cover - trivial barrier
        return None


def _runtime() -> TCCIGRuntime:
    return TCCIGRuntime(
        accelerator=_SingleProcessAccelerator(),
        device="cpu",
        backend="ddp",
        mixed_precision="no",
        is_distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        is_main_process=True,
    )


def _pair_table() -> PairTable:
    records = (
        LabeledPair(protein_a="A", protein_b="B", label=1),
        LabeledPair(protein_a="A", protein_b="C", label=0),
        LabeledPair(protein_a="B", protein_b="C", label=1),
        LabeledPair(protein_a="C", protein_b="D", label=0),
    )
    return PairTable(split="pairwise_test", path=Path("x"), records=records, self_pair_rows=0)


def test_run_pairwise_test_exports_raw_and_refined(tmp_path: Path, monkeypatch: object) -> None:
    import tccig.test as test_module

    table = _pair_table()
    raw_scores = [0.95, 0.10, 0.80, 0.20]
    refined_scores = [0.0, 0.0, 0.0, 0.0]

    def _fake_score_split(**_kwargs: object) -> list[float]:
        return raw_scores

    def _fake_predict_refined(_request: object) -> list[float]:
        return refined_scores

    monkeypatch.setattr(test_module.s2gae, "predict_refined", _fake_predict_refined)  # type: ignore[attr-defined]

    log_dir = tmp_path / "logs"
    metrics = run_pairwise_test(
        table=table,
        scorer_cfg={},
        refiner_cfg={},
        runtime=_runtime(),
        cache_dir=tmp_path / "cache",
        log_dir=log_dir,
        refiner_state=object(),
        # Input-graph threshold is deliberately != 0.5 so a regression that
        # scores raw metrics at the input threshold (0.85, missing the 0.80
        # positive) is caught: raw scorer decisions must use 0.5.
        pairwise_input_rule=GraphRule(type="threshold", value=0.85),
        refined_output_rule=GraphRule(type="threshold", value=0.5),
        score_split_fn=_fake_score_split,
    )

    pairwise_dir = log_dir / "pairwise_test"
    rows = list(csv.DictReader((pairwise_dir / "human_test_ppi_pred.csv").open()))
    assert rows[0].keys() >= {"raw_probability", "refined_probability"}
    assert [float(r["raw_probability"]) for r in rows] == raw_scores
    assert [float(r["refined_probability"]) for r in rows] == refined_scores

    raw_metrics = json.loads((pairwise_dir / "raw_metrics.json").read_text())
    refined_metrics = json.loads((pairwise_dir / "refined_metrics.json").read_text())
    # Raw metrics must reflect the scorer's own 0.5 decision boundary, not the
    # input-graph construction threshold. At 0.5 both positives (0.95, 0.80)
    # are recovered -> recall 1.0; at the 0.85 input threshold recall is 0.5.
    assert raw_metrics["threshold"] == 0.5
    assert raw_metrics["recall"] == 1.0
    # Raw scorer ranks positives above negatives -> perfect AUPRC; refined is degenerate.
    assert raw_metrics["auprc"] == 1.0
    assert refined_metrics["auprc"] < raw_metrics["auprc"]
    # Returned metrics remain the refined ones for back-compat.
    assert metrics == refined_metrics
    # Back-compat artifact still present and equal to refined metrics.
    assert json.loads((pairwise_dir / "pairwise_metrics.json").read_text()) == refined_metrics


def test_run_raw_pairwise_topology_baseline_exports_artifacts(tmp_path: Path) -> None:
    table = _pair_table()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C", "D"])
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    with (processed_dir / "human_test_graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)
    with (processed_dir / "test_sampled_nodes.pkl").open("wb") as handle:
        pickle.dump({3: [["A", "B", "C"], ["A", "C", "D"]]}, handle)

    def _fake_score_split(**_kwargs: object) -> list[float]:
        return [0.95, 0.10, 0.80, 0.20]

    log_dir = tmp_path / "logs"
    metrics = run_raw_pairwise_topology_baseline(
        table=table,
        processed_dir=processed_dir,
        scorer_cfg={},
        runtime=_runtime(),
        cache_dir=tmp_path / "cache",
        log_dir=log_dir,
        raw_output_rule=GraphRule(type="threshold", value=0.5),
        score_split_fn=_fake_score_split,
    )

    topology_dir = log_dir / "topology_test"
    assert set(metrics) == {
        "graph_sim",
        "relative_density",
        "deg_dist_mmd",
        "cc_mmd",
        "laplacian_eigen_mmd",
    }
    assert (topology_dir / "all_test_ppi_pred.txt").read_text(encoding="utf-8").splitlines() == [
        "A\tB\t1",
        "A\tC\t0",
        "B\tC\t1",
        "C\tD\t0",
    ]
    payload = json.loads((topology_dir / "topology_metrics.json").read_text(encoding="utf-8"))
    assert payload["raw_output_rule"] == {"type": "threshold", "value": 0.5}
    assert (topology_dir / "topology_metrics.csv").exists()


def test_run_raw_pairwise_topology_baseline_can_write_separate_output_dir(
    tmp_path: Path,
) -> None:
    table = _pair_table()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C", "D"])
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    with (processed_dir / "human_test_graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)
    with (processed_dir / "test_sampled_nodes.pkl").open("wb") as handle:
        pickle.dump({3: [["A", "B", "C"], ["A", "C", "D"]]}, handle)

    def _fake_score_split(**_kwargs: object) -> list[float]:
        return [0.95, 0.10, 0.80, 0.20]

    log_dir = tmp_path / "logs"
    output_dir = log_dir / "raw_pairwise_topology_baseline"
    run_raw_pairwise_topology_baseline(
        table=table,
        processed_dir=processed_dir,
        scorer_cfg={},
        runtime=_runtime(),
        cache_dir=tmp_path / "cache",
        log_dir=log_dir,
        output_dir=output_dir,
        raw_output_rule=GraphRule(type="threshold", value=0.5),
        score_split_fn=_fake_score_split,
    )

    assert (output_dir / "all_test_ppi_pred.txt").exists()
    assert (output_dir / "topology_metrics.json").exists()
    assert not (log_dir / "topology_test" / "topology_metrics.json").exists()
