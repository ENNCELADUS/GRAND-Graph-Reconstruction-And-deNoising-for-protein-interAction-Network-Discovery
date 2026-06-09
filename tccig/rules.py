"""Graph decision rules for TCCIG scores."""

from __future__ import annotations

from dataclasses import dataclass

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


def threshold_for_target_precision(
    *,
    probabilities: list[float],
    labels: list[int],
    target_precision: float,
) -> tuple[float, dict[str, float | int]]:
    """Return the lowest probability threshold that reaches target precision."""
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have matching lengths")
    if not 0.0 <= target_precision <= 1.0:
        raise ValueError("target_precision must be in [0, 1]")
    if not probabilities:
        raise ValueError("target_precision threshold requires at least one probability")

    best_threshold: float | None = None
    best_metrics: dict[str, float | int] = {}
    for threshold in sorted({float(probability) for probability in probabilities}):
        metrics = binary_metrics_at_threshold(
            labels=labels,
            probabilities=probabilities,
            threshold=threshold,
        )
        if int(metrics["positive_edges"]) <= 0:
            continue
        if float(metrics["precision"]) >= target_precision:
            best_threshold = threshold
            best_metrics = metrics
            break
    if best_threshold is None:
        raise ValueError(
            f"No scorer threshold reaches target_precision={target_precision}"
        )
    return best_threshold, best_metrics


def binary_metrics_at_threshold(
    *,
    labels: list[int],
    probabilities: list[float],
    threshold: float,
) -> dict[str, float | int]:
    """Return binary metrics under a fixed probability threshold."""
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have matching lengths")
    predictions = [int(float(probability) >= float(threshold)) for probability in probabilities]
    positive_edges = sum(predictions)
    true_positive_edges = sum(
        1 for label, pred in zip(labels, predictions, strict=True) if label == 1 and pred
    )
    actual_positive_edges = sum(1 for label in labels if label == 1)
    return {
        "precision": 0.0 if positive_edges == 0 else float(true_positive_edges / positive_edges),
        "recall": float(true_positive_edges / max(1, actual_positive_edges)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "positive_edges": int(positive_edges),
    }
