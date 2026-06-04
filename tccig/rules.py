"""Graph decision rules and calibration helpers for TCCIG scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sklearn.metrics import f1_score, matthews_corrcoef

from tccig.io import CandidatePair, canonical_edge

DEFAULT_GRAPH_THRESHOLD = 0.5
PROBABILITY_EPSILON = 1.0e-6


@dataclass(frozen=True)
class GraphRule:
    """Validation-selected graph assembly rule."""

    type: str
    value: float | int

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a serializable rule payload."""
        return {"type": self.type, "value": float(self.value)}


def parse_rules(raw_rules: object) -> list[GraphRule]:
    """Parse configured threshold-only graph rules."""
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("graph_selection.rules must be a non-empty list")
    rules: list[GraphRule] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("graph_selection.rules entries must be mappings")
        rule_type = str(raw_rule.get("type", "")).lower()
        if rule_type == "threshold":
            rules.append(GraphRule(type=rule_type, value=float(raw_rule.get("value", 0.5))))
        else:
            raise ValueError(
                f"Unsupported graph rule type: {rule_type}; "
                "TCCIG graph rules only support threshold"
            )
    return rules


def edges_from_rule(
    *,
    pairs: list[CandidatePair],
    probabilities: list[float],
    rule: GraphRule,
) -> list[tuple[str, str]]:
    """Select graph edges from candidate probabilities under one rule."""
    if len(pairs) != len(probabilities):
        raise ValueError("pairs and probabilities must have matching lengths")
    if rule.type == "threshold":
        threshold = float(rule.value)
        return [
            canonical_edge(pair.protein_a, pair.protein_b)
            for pair, probability in zip(pairs, probabilities, strict=True)
            if float(probability) >= threshold
        ]
    raise ValueError(f"Unsupported graph rule type: {rule.type}")


def select_rule(
    *,
    pairs: list[CandidatePair],
    probabilities: list[float],
    labels: list[int],
    rules: list[GraphRule],
) -> tuple[GraphRule, dict[str, Any]]:
    """Select the validation rule with the best binary F1 and MCC tie-breaker."""
    if len(pairs) != len(probabilities) or len(pairs) != len(labels):
        raise ValueError("pairs, probabilities, and labels must have matching lengths")
    best_rule: GraphRule | None = None
    best_metrics: dict[str, Any] = {}
    for rule in rules:
        selected_edges = set(edges_from_rule(pairs=pairs, probabilities=probabilities, rule=rule))
        predictions = [
            int(canonical_edge(pair.protein_a, pair.protein_b) in selected_edges) for pair in pairs
        ]
        metrics = {
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, predictions)),
            "positive_edges": int(sum(predictions)),
            "rule": rule.to_dict(),
        }
        if _is_better(metrics, best_metrics):
            best_rule = rule
            best_metrics = metrics
    if best_rule is None:
        raise ValueError("No validation graph rules were evaluated")
    return best_rule, best_metrics


def apply_logit_bias(probabilities: list[float], bias: float) -> list[float]:
    """Apply one global logit bias and return calibrated probabilities."""
    return [_apply_logit_bias_value(probability, bias) for probability in probabilities]


def equivalent_threshold_from_bias(bias: float) -> float:
    """Return the raw probability threshold equivalent to calibrated 0.5."""
    return 1.0 / (1.0 + math.exp(float(bias)))


def calibration_payload(
    *,
    bias: float,
    objective: str,
    validation_metrics: dict[str, float | int] | None = None,
    monitor_metric: str | None = None,
    monitor_value: float | None = None,
) -> dict[str, object]:
    """Return the serializable selected-calibration payload."""
    payload: dict[str, object] = {
        "type": "logit_bias",
        "bias": float(bias),
        "threshold_after_calibration": DEFAULT_GRAPH_THRESHOLD,
        "equivalent_probability_threshold": equivalent_threshold_from_bias(float(bias)),
        "objective": objective,
    }
    if monitor_metric is not None:
        payload["monitor_metric"] = monitor_metric
    if monitor_value is not None:
        payload["monitor_value"] = float(monitor_value)
    if validation_metrics is not None:
        payload["validation_metrics"] = validation_metrics
    return payload


def identity_calibration_payload() -> dict[str, object]:
    """Return the no-op calibration payload used by non-S2GAE test hooks."""
    return calibration_payload(bias=0.0, objective="identity")


def _apply_logit_bias_value(probability: float, bias: float) -> float:
    clamped = min(max(float(probability), PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON)
    logit = math.log(clamped / (1.0 - clamped))
    return 1.0 / (1.0 + math.exp(-(logit + float(bias))))


def _is_better(metrics: dict[str, Any], best_metrics: dict[str, Any]) -> bool:
    """Return whether metrics beat the incumbent validation rule."""
    if not best_metrics:
        return True
    return (
        float(metrics["f1"]),
        float(metrics["mcc"]),
        -int(metrics["positive_edges"]),
    ) > (
        float(best_metrics["f1"]),
        float(best_metrics["mcc"]),
        -int(best_metrics["positive_edges"]),
    )
