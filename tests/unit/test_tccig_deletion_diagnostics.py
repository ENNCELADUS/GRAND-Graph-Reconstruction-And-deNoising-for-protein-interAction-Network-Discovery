"""Tests for TCCIG refined-graph deletion diagnostics."""

from __future__ import annotations

from tccig.prepare import CandidatePair
from tccig.test import compute_deletion_diagnostics


def test_deletion_diagnostics_counts_added_and_deleted() -> None:
    pairs = [CandidatePair("A", "B"), CandidatePair("B", "C"), CandidatePair("A", "C")]
    raw_probabilities = [0.95, 0.40, 0.30]  # raw edges by 0.5: only (A,B)
    raw_edges = [("A", "B")]
    refined_edges = [("B", "C")]  # deleted (A,B); added (B,C)
    labels = [1, 0, 0]

    diagnostics = compute_deletion_diagnostics(
        raw_edges=raw_edges,
        refined_edges=refined_edges,
        pairs=pairs,
        raw_probabilities=raw_probabilities,
        labels=labels,
    )

    assert diagnostics["edges_deleted"] == 1.0
    assert diagnostics["edges_added"] == 1.0
    assert diagnostics["net_edge_delta"] == 0.0
    # the deleted edge (A,B) has label 1 / raw prob 0.95 -> NOT a good deletion
    assert diagnostics["deletion_precision"] == 0.0


def test_deletion_precision_high_when_deleting_negatives() -> None:
    pairs = [CandidatePair("A", "B"), CandidatePair("B", "C")]
    raw_probabilities = [0.40, 0.95]
    raw_edges = [("A", "B"), ("B", "C")]  # (A,B) is a low-confidence raw edge
    refined_edges = [("B", "C")]  # deleted (A,B), a label-0/low-prob pair
    labels = [0, 1]

    diagnostics = compute_deletion_diagnostics(
        raw_edges=raw_edges,
        refined_edges=refined_edges,
        pairs=pairs,
        raw_probabilities=raw_probabilities,
        labels=labels,
    )

    assert diagnostics["edges_deleted"] == 1.0
    assert diagnostics["deletion_precision"] == 1.0
