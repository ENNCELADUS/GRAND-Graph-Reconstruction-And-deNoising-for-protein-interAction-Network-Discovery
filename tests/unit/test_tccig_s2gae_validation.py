"""Tests for TCCIG S2GAE validation monitor semantics."""

from __future__ import annotations

import pytest
from tccig.s2gae import _is_better_monitor, _resolve_monitor_value


def test_val_topology_loss_monitor_minimizes_hard_metric_penalty() -> None:
    assert _is_better_monitor(
        value=0.25,
        best_value=0.50,
        monitor_metric="val_topology_loss",
    )
    assert not _is_better_monitor(
        value=0.75,
        best_value=0.50,
        monitor_metric="val_topology_loss",
    )


def test_topology_monitor_values_match_checkpoint_direction() -> None:
    metrics = {
        "val_topology_loss": 0.4,
        "graph_sim": 0.8,
        "relative_density": 1.2,
    }

    assert _resolve_monitor_value(
        monitor_metric="internal_val_graph_sim",
        validation_auprc=0.1,
        topology_metrics=metrics,
    ) == 0.8
    assert _resolve_monitor_value(
        monitor_metric="internal_val_relative_density",
        validation_auprc=0.1,
        topology_metrics=metrics,
    ) == pytest.approx(-0.2)
