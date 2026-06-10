"""Unit tests for TCCIG graph decision rules."""

from __future__ import annotations

import pytest
from tccig.prepare import (
    CandidatePair,
    GraphRule,
    edges_from_rule,
)
from tccig.train import (
    binary_metrics_at_threshold,
    parse_rules,
    threshold_for_target_precision,
)


def test_graph_rules_select_threshold_edges() -> None:
    pairs = [
        CandidatePair("A", "B"),
        CandidatePair("A", "C"),
        CandidatePair("B", "C"),
    ]
    probabilities = [0.8, 0.7, 0.2]

    assert edges_from_rule(
        pairs=pairs,
        probabilities=probabilities,
        rule=GraphRule(type="threshold", value=0.75),
    ) == [("A", "B")]


def test_graph_rules_reject_removed_top_m_and_top_k() -> None:
    with pytest.raises(ValueError, match="only support threshold"):
        parse_rules([{"type": "top_m", "m": 1}])
    with pytest.raises(ValueError, match="only support threshold"):
        parse_rules([{"type": "top_k", "k": 1}])


def test_target_precision_threshold_selects_lowest_valid_threshold() -> None:
    threshold, metrics = threshold_for_target_precision(
        probabilities=[0.95, 0.90, 0.80, 0.20],
        labels=[1, 0, 1, 0],
        target_precision=0.8,
    )

    assert threshold == pytest.approx(0.95)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["positive_edges"] == 1


def test_target_precision_threshold_fails_when_unreachable() -> None:
    with pytest.raises(ValueError, match="No scorer threshold reaches target_precision=1.0"):
        threshold_for_target_precision(
            probabilities=[0.8, 0.7],
            labels=[0, 0],
            target_precision=1.0,
        )


def test_binary_metrics_at_threshold_reports_zero_precision_for_empty_graph() -> None:
    metrics = binary_metrics_at_threshold(
        labels=[1, 0],
        probabilities=[0.2, 0.1],
        threshold=0.9,
    )

    assert metrics["precision"] == 0.0
    assert metrics["positive_edges"] == 0
