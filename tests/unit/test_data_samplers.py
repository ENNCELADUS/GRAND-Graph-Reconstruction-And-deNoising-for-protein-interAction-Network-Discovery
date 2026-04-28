"""Unit tests for custom data batch samplers."""

from __future__ import annotations

from src.utils.data_samplers import StagedOHEMBatchSampler


def test_staged_ohem_sampler_exposes_explicit_batch_semantics() -> None:
    labels = [1] * 100 + [0] * 100
    sampler = StagedOHEMBatchSampler(
        labels=labels,
        batch_size=32,
        warmup_pos_neg_ratio=1.1,
        warmup_epochs=1,
        pool_multiplier=4,
        shuffle=False,
    )

    assert not hasattr(sampler, "batch_size")
    assert sampler.target_batch_size == 32
    assert sampler.warmup_batch_size == 31
    assert sampler.mining_batch_size == 128
    assert len(next(iter(sampler))) == 31

    sampler.set_epoch(1)
    assert len(next(iter(sampler))) == 128


def test_staged_ohem_sampler_uses_rank_local_pool_class_ratio() -> None:
    labels = [1] * 100 + [0] * 100
    sampler = StagedOHEMBatchSampler(
        labels=labels,
        batch_size=32,
        warmup_epochs=0,
        pool_multiplier=4,
        rank=1,
        world_size=4,
        shuffle=False,
    )

    assert len(sampler.pos_indices) == 25
    assert len(sampler.neg_indices) == 25
    assert sampler._pool_class_counts() == (25, 25)


def test_staged_ohem_sampler_keeps_both_classes_in_imbalanced_mining_pool() -> None:
    labels = [1] + [0] * 99
    sampler = StagedOHEMBatchSampler(
        labels=labels,
        batch_size=2,
        warmup_epochs=0,
        pool_multiplier=1,
        shuffle=False,
    )

    assert sampler._pool_class_counts() == (1, 1)
    pool = next(iter(sampler))
    pool_labels = [labels[index] for index in pool]
    assert sorted(pool_labels) == [0, 1]
