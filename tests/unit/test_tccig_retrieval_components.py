"""Unit tests for graph-prior retrieval TCCIG helpers."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
import torch
from src.train.tccig.retrieval import (
    TCCIGReconstructionUniverse,
    build_full_reconstruction_universe,
    compute_retrieval_losses,
    false_negative_mask,
    hybrid_degree_capped_topk,
    mine_hard_negative_pairs,
)


def test_full_reconstruction_universe_labels_validation_positives() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("P1", "P2"), ("P2", "P3")])
    universe = build_full_reconstruction_universe(
        protein_ids=("P1", "P2", "P3"),
        positive_edges=frozenset({("P1", "P2")}),
    )

    assert isinstance(universe, TCCIGReconstructionUniverse)
    assert universe.records == (("P1", "P2"), ("P1", "P3"), ("P2", "P3"))
    assert universe.labels.tolist() == [1.0, 0.0, 0.0]
    assert universe.positive_density == pytest.approx(1.0 / 3.0)


def test_false_negative_mask_keeps_known_edges_out_of_negative_denominator() -> None:
    known_edges = torch.tensor(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=torch.long,
    )

    mask = false_negative_mask(num_nodes=4, known_positive_pairs=known_edges)

    assert mask.shape == (4, 4)
    assert mask[0, 1]
    assert mask[1, 0]
    assert mask[1, 2]
    assert mask[2, 1]
    assert not mask[0, 3]


def test_retrieval_losses_include_infonce_degree_and_reranker_terms() -> None:
    score_matrix = torch.tensor(
        [
            [0.0, 0.6, 0.3],
            [0.5, 0.0, 0.4],
            [0.2, 0.1, 0.0],
        ],
        requires_grad=True,
    )
    positive_pairs = torch.tensor([[0, 1]], dtype=torch.long)
    candidate_logits = torch.tensor([3.0, -1.0, 0.5], requires_grad=True)
    candidate_labels = torch.tensor([1.0, 0.0, 0.0])
    degree_predictions = torch.tensor([1.0, 1.0, 0.2], requires_grad=True)
    degree_targets = torch.tensor([1.0, 1.0, 0.0])

    losses = compute_retrieval_losses(
        retrieval_score_matrix=score_matrix,
        positive_pairs=positive_pairs,
        known_positive_pairs=positive_pairs,
        candidate_logits=candidate_logits,
        candidate_labels=candidate_labels,
        degree_predictions=degree_predictions,
        degree_targets=degree_targets,
        retrieval_weight=1.0,
        reranker_weight=0.5,
        degree_weight=0.25,
    )

    assert losses["retrieval"].item() > 0.0
    assert losses["reranker"].item() > 0.0
    assert losses["degree"].item() >= 0.0
    losses["total"].backward()


def test_retrieval_losses_include_mined_hard_negative_margin() -> None:
    score_matrix = torch.tensor(
        [
            [0.0, 0.9, 1.1],
            [0.9, 0.0, 0.2],
            [1.1, 0.2, 0.0],
        ],
        requires_grad=True,
    )
    positive_pairs = torch.tensor([[0, 1]], dtype=torch.long)
    hard_negative_pairs = torch.tensor([[0, 2]], dtype=torch.long)
    candidate_logits = torch.tensor([0.0], requires_grad=True)
    candidate_labels = torch.tensor([0.0])

    losses = compute_retrieval_losses(
        retrieval_score_matrix=score_matrix,
        positive_pairs=positive_pairs,
        known_positive_pairs=positive_pairs,
        candidate_logits=candidate_logits,
        candidate_labels=candidate_labels,
        hard_negative_pairs=hard_negative_pairs,
        retrieval_weight=0.0,
        reranker_weight=0.0,
        hard_negative_weight=1.0,
    )

    assert losses["hard_negative"].item() > 0.0
    losses["total"].backward()


def test_hybrid_degree_capped_topk_respects_global_budget_and_degree_cap() -> None:
    pairs = torch.tensor(
        [
            [0, 0, 0, 1, 2],
            [1, 2, 3, 2, 3],
        ],
        dtype=torch.long,
    )
    scores = torch.tensor([0.99, 0.98, 0.97, 0.80, 0.79])
    degree_cap = torch.tensor([1.0, 2.0, 2.0, 2.0])

    selected = hybrid_degree_capped_topk(
        candidate_pairs=pairs,
        scores=scores,
        edge_budget=3,
        degree_cap=degree_cap,
        cap_slack=0.0,
    )

    assert selected.tolist() == [True, False, False, True, True]


def test_mine_hard_negative_pairs_returns_high_scoring_unknown_pairs() -> None:
    scores = torch.tensor(
        [
            [0.0, 0.9, 0.8, 0.1],
            [0.9, 0.0, 0.7, 0.6],
            [0.8, 0.7, 0.0, 0.5],
            [0.1, 0.6, 0.5, 0.0],
        ]
    )
    positives = torch.tensor([[0, 1]], dtype=torch.long)

    hard_negatives = mine_hard_negative_pairs(
        score_matrix=scores,
        known_positive_pairs=positives,
        top_k=2,
        max_pairs=2,
    )

    assert hard_negatives.tolist() == [[0, 1], [2, 2]]


def test_reconstruction_universe_can_be_written_as_records(tmp_path: Path) -> None:
    universe = build_full_reconstruction_universe(
        protein_ids=("B", "A"),
        positive_edges=frozenset({("A", "B")}),
    )
    path = tmp_path / "universe.tsv"

    universe.write_records(path)

    assert path.read_text(encoding="utf-8").strip() == "A\tB\t1"
