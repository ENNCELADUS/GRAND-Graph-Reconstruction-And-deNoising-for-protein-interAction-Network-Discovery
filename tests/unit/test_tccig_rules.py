"""Unit tests for TCCIG graph decision rules."""

from __future__ import annotations

import math

import pytest
from tccig.io import CandidatePair
from tccig.rules import (
    GraphRule,
    apply_logit_bias,
    edges_from_rule,
    equivalent_threshold_from_bias,
    parse_rules,
    select_rule,
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
            GraphRule(type="threshold", value=0.85),
        ],
    )

    assert selected_rule == GraphRule(type="threshold", value=0.85)
    assert metrics["f1"] == 1.0


def test_logit_bias_calibration_maps_raw_threshold_to_calibrated_half() -> None:
    raw_threshold = 0.75
    bias = -math.log(raw_threshold / (1.0 - raw_threshold))

    calibrated = apply_logit_bias([raw_threshold], bias)[0]

    assert calibrated == pytest.approx(0.5)
    assert equivalent_threshold_from_bias(bias) == pytest.approx(raw_threshold)
