"""Unit tests for TCCIG graph decision rules."""

from __future__ import annotations

from tccig.io import CandidatePair
from tccig.rules import GraphRule, edges_from_rule, select_rule


def test_graph_rules_select_threshold_top_m_and_top_k_edges() -> None:
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
    assert edges_from_rule(
        pairs=pairs,
        probabilities=probabilities,
        rule=GraphRule(type="top_m", value=2),
    ) == [("A", "B"), ("A", "C")]
    assert edges_from_rule(
        pairs=pairs,
        probabilities=probabilities,
        rule=GraphRule(type="top_k", value=1),
    ) == [("A", "B"), ("A", "C")]


def test_select_rule_prefers_validation_f1_then_sparser_graph() -> None:
    pairs = [CandidatePair("A", "B"), CandidatePair("A", "C")]
    labels = [1, 0]
    probabilities = [0.9, 0.8]

    selected_rule, metrics = select_rule(
        pairs=pairs,
        probabilities=probabilities,
        labels=labels,
        rules=[
            GraphRule(type="threshold", value=0.5),
            GraphRule(type="top_m", value=1),
        ],
    )

    assert selected_rule == GraphRule(type="top_m", value=1)
    assert metrics["f1"] == 1.0
