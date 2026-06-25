"""Unit tests for TCCIG test-time raw/refined export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tccig.prepare import GraphRule, LabeledPair, PairTable, TCCIGRuntime
from tccig.test import run_pairwise_test


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
