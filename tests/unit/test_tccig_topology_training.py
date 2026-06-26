"""Tests for TCCIG Run 02 topology-conditioned training loss."""

from __future__ import annotations

import pytest
import torch
from tccig.s2gae import asymmetric_residual_anchor


def test_asymmetric_anchor_leaves_deletion_free() -> None:
    negative_delta = torch.tensor([-3.0, -1.0, -0.5])
    assert float(asymmetric_residual_anchor(negative_delta)) == 0.0


def test_asymmetric_anchor_penalizes_upward_push() -> None:
    positive_delta = torch.tensor([2.0, 0.0, -4.0])
    # only +2.0 contributes: (2^2 + 0 + 0) / 3
    assert float(asymmetric_residual_anchor(positive_delta)) == pytest.approx(4.0 / 3.0)
