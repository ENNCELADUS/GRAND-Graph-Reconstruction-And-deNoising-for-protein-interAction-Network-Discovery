"""Graph decision rules for TCCIG pairwise and refined scores."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sklearn.metrics import f1_score, matthews_corrcoef

from tccig.io import CandidatePair, canonical_edge


@dataclass(frozen=True)
class GraphRule:
    """Validation-selected graph assembly rule."""

    type: str
    value: float | int

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a serializable rule payload."""
        if self.type == "threshold":
            return {"type": self.type, "value": float(self.value)}
        key = "k" if self.type == "top_k" else "m"
        return {"type": self.type, key: int(self.value)}


def parse_rules(raw_rules: object) -> list[GraphRule]:
    """Parse configured threshold, top-k, and top-M rules."""
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("graph_selection.rules must be a non-empty list")
    rules: list[GraphRule] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("graph_selection.rules entries must be mappings")
        rule_type = str(raw_rule.get("type", "")).lower()
        if rule_type == "threshold":
            rules.append(GraphRule(type=rule_type, value=float(raw_rule.get("value", 0.5))))
        elif rule_type == "top_k":
            rules.append(GraphRule(type=rule_type, value=int(raw_rule["k"])))
        elif rule_type == "top_m":
            rules.append(GraphRule(type=rule_type, value=int(raw_rule["m"])))
        else:
            raise ValueError(f"Unsupported graph rule type: {rule_type}")
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
    if rule.type == "top_m":
        selected_indices = _top_indices(probabilities=probabilities, limit=int(rule.value))
        return [
            canonical_edge(pairs[index].protein_a, pairs[index].protein_b)
            for index in selected_indices
        ]
    if rule.type == "top_k":
        return _top_k_edges(pairs=pairs, probabilities=probabilities, k=int(rule.value))
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


def _top_indices(*, probabilities: list[float], limit: int) -> list[int]:
    """Return stable top-probability indices."""
    if limit <= 0:
        return []
    ranked = sorted(
        range(len(probabilities)),
        key=lambda index: (-float(probabilities[index]), index),
    )
    return ranked[:limit]


def _top_k_edges(
    *,
    pairs: list[CandidatePair],
    probabilities: list[float],
    k: int,
) -> list[tuple[str, str]]:
    """Return the union of per-node top-k incident edges."""
    if k <= 0:
        return []
    incident: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        probability = float(probabilities[index])
        incident[pair.protein_a].append((probability, index))
        incident[pair.protein_b].append((probability, index))
    selected_indices: set[int] = set()
    for entries in incident.values():
        ranked = sorted(entries, key=lambda item: (-item[0], item[1]))
        selected_indices.update(index for _, index in ranked[:k])
    return [
        canonical_edge(pairs[index].protein_a, pairs[index].protein_b)
        for index in sorted(selected_indices)
    ]


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
