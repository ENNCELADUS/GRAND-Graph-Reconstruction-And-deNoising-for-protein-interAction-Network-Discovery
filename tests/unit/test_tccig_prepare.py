"""Tests for TCCIG scorer-error edge target preparation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tccig.prepare import (
    CandidatePair,
    EdgeSamplingConfig,
    classify_scorer_error_targets,
    read_pair_table,
    sample_epoch_edge_targets,
    strict_reject_legacy_hooks,
    write_json,
)


def test_classify_scorer_error_targets_assigns_all_quadrants() -> None:
    pairs = [
        CandidatePair("A", "B"),
        CandidatePair("A", "C"),
        CandidatePair("B", "C"),
        CandidatePair("C", "D"),
    ]

    quadrants = classify_scorer_error_targets(
        pairs=pairs,
        labels=[0, 1, 1, 0],
        pairwise_graph_edges=[("A", "B"), ("B", "C")],
    )

    actual_quadrants = {
        name: [target.pair_index for target in targets] for name, targets in quadrants.items()
    }
    assert actual_quadrants == {
        "fp": [0],
        "fn": [1],
        "tp": [2],
        "tn": [3],
    }
    assert quadrants["fp"][0].mask_input_edge
    assert quadrants["fn"][0].mask_input_edge is False


def test_sample_epoch_edge_targets_keeps_all_hard_and_samples_easy_budget() -> None:
    quadrants = classify_scorer_error_targets(
        pairs=[
            CandidatePair("A", "B"),
            CandidatePair("A", "C"),
            CandidatePair("B", "C"),
            CandidatePair("C", "D"),
        ],
        labels=[0, 1, 1, 0],
        pairwise_graph_edges=[("A", "B"), ("B", "C")],
    )

    targets = sample_epoch_edge_targets(
        quadrants=quadrants,
        sampling=EdgeSamplingConfig(hard_fraction=0.5, easy_anchor_fraction=0.5, seed=1),
        epoch=1,
    )

    hard_indices = [target.pair_index for target in targets if target.quadrant in {"fp", "fn"}]
    easy_indices = [target.pair_index for target in targets if target.quadrant in {"tp", "tn"}]
    assert hard_indices == [0, 1]
    assert len(easy_indices) == 2


def test_sample_epoch_edge_targets_ceil_easy_anchor_budget() -> None:
    quadrants = classify_scorer_error_targets(
        pairs=[
            CandidatePair("A", "B"),
            CandidatePair("A", "C"),
        ],
        labels=[0, 1],
        pairwise_graph_edges=[("A", "B"), ("A", "C")],
    )

    targets = sample_epoch_edge_targets(
        quadrants=quadrants,
        sampling=EdgeSamplingConfig(hard_fraction=0.7, easy_anchor_fraction=0.3, seed=1),
        epoch=1,
    )

    assert [target.quadrant for target in targets] == ["fp", "tp"]


def test_sample_epoch_edge_targets_falls_back_when_one_easy_class_is_short() -> None:
    quadrants = classify_scorer_error_targets(
        pairs=[
            CandidatePair("A", "B"),
            CandidatePair("A", "C"),
            CandidatePair("B", "C"),
            CandidatePair("C", "D"),
        ],
        labels=[0, 1, 0, 0],
        pairwise_graph_edges=[("A", "B")],
    )

    targets = sample_epoch_edge_targets(
        quadrants=quadrants,
        sampling=EdgeSamplingConfig(hard_fraction=0.5, easy_anchor_fraction=0.5, seed=2),
        epoch=1,
    )

    easy_quadrants = [target.quadrant for target in targets if target.quadrant in {"tp", "tn"}]
    assert easy_quadrants == ["tn", "tn"]


def test_sample_epoch_edge_targets_reshuffles_easy_anchors_by_epoch() -> None:
    pairs = [CandidatePair("A", chr(ord("B") + index)) for index in range(8)]
    quadrants = classify_scorer_error_targets(
        pairs=pairs,
        labels=[0, 1, 1, 1, 0, 0, 0, 0],
        pairwise_graph_edges=[("A", "B"), ("A", "D"), ("A", "E")],
    )
    sampling = EdgeSamplingConfig(
        hard_fraction=0.5,
        easy_anchor_fraction=0.5,
        seed=4,
        reshuffle_easy_each_epoch=True,
    )

    epoch_one = sample_epoch_edge_targets(quadrants=quadrants, sampling=sampling, epoch=1)
    epoch_two = sample_epoch_edge_targets(quadrants=quadrants, sampling=sampling, epoch=2)

    assert [target.pair_index for target in epoch_one] != [
        target.pair_index for target in epoch_two
    ]


def test_strict_reject_legacy_hooks_fails_fast() -> None:
    for config in (
        {"pairwise_scorer": {"target": "legacy"}},
        {"refiner": {"train_target": "legacy"}},
        {"refiner": {"predict_target": "legacy"}},
    ):
        try:
            strict_reject_legacy_hooks(config)
        except ValueError:
            continue
        raise AssertionError("legacy hook config was accepted")


def test_read_pair_table_hides_labels_for_topology_candidates(tmp_path: Path) -> None:
    pair_path = tmp_path / "all_test_ppi.txt"
    pair_path.write_text(
        "\n".join(
            [
                "A\tA\t1",
                "A\tB\t1",
                "B\tC\t0",
            ]
        ),
        encoding="utf-8",
    )

    table = read_pair_table(path=pair_path, split="topology_test", expose_labels=False)

    assert table.self_pair_rows == 1
    assert table.pairs == [CandidatePair("A", "B"), CandidatePair("B", "C")]
    with pytest.raises(ValueError, match="does not expose labels"):
        _ = table.labels


def test_write_json_uses_stable_json_and_tensor_scalar_values(tmp_path: Path) -> None:
    output_path = tmp_path / "artifact" / "metrics.json"

    write_json(output_path, {"z": 2, "a": 1, "tensor": torch.tensor(3.5)})

    assert output_path.read_text(encoding="utf-8") == (
        '{\n  "a": 1,\n  "tensor": 3.5,\n  "z": 2\n}'
    )
