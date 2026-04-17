"""Unit tests for OHEM selection utility."""

import torch
from src.train.strategies.ohem import OHEMSampleStrategy, select_ohem_indices


def test_select_ohem_indices_keeps_hardest() -> None:
    losses = torch.tensor([0.2, 1.5, 0.7, 2.2, 0.1], dtype=torch.float32)
    selected = select_ohem_indices(losses=losses, keep_ratio=0.4, min_keep=1)
    assert selected.tolist() == [3, 1]


def test_ohem_strategy_respects_target_batch_size() -> None:
    losses = torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32)
    strategy = OHEMSampleStrategy(target_batch_size=2, cap_protein=4)
    selected = strategy.select(losses, epoch_index=0)
    assert selected.numel() == 2


def test_ohem_strategy_honors_warmup_epochs() -> None:
    losses = torch.tensor([0.3, 0.9, 0.1], dtype=torch.float32)
    strategy = OHEMSampleStrategy(target_batch_size=1, cap_protein=4, warmup_epochs=2)
    warmup_selected = strategy.select(losses, epoch_index=0)
    post_warmup_selected = strategy.select(losses, epoch_index=2)
    assert warmup_selected.tolist() == [0, 1, 2]
    assert post_warmup_selected.tolist() == [1]


def test_ohem_strategy_applies_cap_protein_constraint() -> None:
    losses = torch.tensor([3.0, 2.0, 1.0, 0.5], dtype=torch.float32)
    protein_a_ids = torch.tensor([0, 0, 0, 3], dtype=torch.long)
    protein_b_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    strategy = OHEMSampleStrategy(target_batch_size=3, cap_protein=2)
    selected = strategy.select(
        losses=losses,
        epoch_index=0,
        protein_a_ids=protein_a_ids,
        protein_b_ids=protein_b_ids,
    )
    assert selected.tolist() == [0, 1, 3]
