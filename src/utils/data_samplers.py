"""Custom samplers used throughout data loading.

Provides batch samplers used for class-imbalanced training, including
staged OHEM sampling.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence


class ImbalancedBatchSampler:
    """Batch sampler that maintains a target positive-to-negative ratio.

    Each epoch iterates through all positive indices once (without replacement)
    and samples negatives once per epoch (with replacement) to match the
    requested ratio.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        pos_neg_ratio: float = 3.0,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            labels: Binary labels (0 for negative, 1 for positive) for the dataset.
            batch_size: Desired batch size used to derive per-class counts.
            pos_neg_ratio: Number of negatives per positive in each batch.
            shuffle: Whether to shuffle positive indices every epoch.
            drop_last: Drop the final batch if it has fewer positives than expected.
            seed: Optional random seed for reproducibility.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if pos_neg_ratio <= 0:
            raise ValueError("pos_neg_ratio must be positive")
        if len(labels) == 0:
            raise ValueError("labels must be non-empty")

        try:
            processed_labels = [int(label) for label in labels]
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"labels must be convertible to integers: {exc}") from exc

        if any(label not in (0, 1) for label in processed_labels):
            raise ValueError("labels must be binary (0 or 1)")

        self.pos_indices = [idx for idx, label in enumerate(processed_labels) if label]
        self.neg_indices = [idx for idx, label in enumerate(processed_labels) if not label]

        if not self.pos_indices:
            raise ValueError("ImbalancedBatchSampler requires at least one positive sample")
        if not self.neg_indices:
            raise ValueError("ImbalancedBatchSampler requires at least one negative sample")

        self.batch_size = batch_size
        self.pos_neg_ratio = float(pos_neg_ratio)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._rng = random.Random(seed)

        self.pos_per_batch = self._compute_pos_per_batch()
        self.neg_per_batch = self._compute_neg_per_batch(self.pos_per_batch)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of indices with the configured ratio."""
        pos_indices = list(self.pos_indices)
        if self.shuffle:
            self._rng.shuffle(pos_indices)

        # Precompute batches of positives and required negatives for the epoch.
        batches: list[list[int]] = []
        neg_requirements: list[int] = []
        batch_size = self.pos_per_batch

        for start in range(0, len(pos_indices), batch_size):
            pos_batch = pos_indices[start : start + batch_size]
            if len(pos_batch) < batch_size and self.drop_last:
                break
            batches.append(pos_batch)
            neg_requirements.append(self._negatives_for_batch(len(pos_batch)))

        total_negs_needed = sum(neg_requirements)
        neg_pool = (
            self._rng.choices(self.neg_indices, k=total_negs_needed)
            if total_negs_needed > 0
            else []
        )

        neg_offset = 0
        for pos_batch, neg_count in zip(batches, neg_requirements, strict=False):
            neg_batch = neg_pool[neg_offset : neg_offset + neg_count]
            neg_offset += neg_count

            batch = pos_batch + neg_batch
            self._rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        """Return number of batches per epoch based on positive coverage."""
        full_batches, remainder = divmod(len(self.pos_indices), self.pos_per_batch)
        if self.drop_last:
            return full_batches
        return full_batches + (1 if remainder else 0)

    def _compute_pos_per_batch(self) -> int:
        """Derive the number of positives per batch from batch_size and ratio."""
        denom = 1.0 + self.pos_neg_ratio
        raw = self.batch_size / denom
        pos_per_batch = max(1, int(math.floor(raw)))
        return pos_per_batch

    def _compute_neg_per_batch(self, pos_per_batch: int) -> int:
        """Compute negatives per batch based on the ratio."""
        neg_per = int(round(pos_per_batch * self.pos_neg_ratio))
        return max(0, neg_per)

    def _negatives_for_batch(self, positive_count: int) -> int:
        """Scale negatives for partial batches at epoch end."""
        if positive_count <= 0:
            return 0
        if positive_count == self.pos_per_batch:
            return self.neg_per_batch
        scaled = int(math.ceil(positive_count * self.pos_neg_ratio))
        return max(0, scaled)


class StagedOHEMBatchSampler:
    """Batch sampler with staged warmup + online hard example mining (OHEM).

    Warmup (epoch < warmup_epochs):
      - Sample positives without replacement.
      - Sample negatives without replacement using warmup_pos_neg_ratio.

    Mining (epoch >= warmup_epochs):
      - Build a candidate pool of size mining_batch_size = pool_multiplier * batch_size.
      - Stratified sampling preserves the dataset's natural class ratio.
      - The trainer performs read-only scoring and OHEM selection.
    """

    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int,
        warmup_pos_neg_ratio: float = 7.0,
        warmup_epochs: int = 2,
        pool_multiplier: int = 16,
        cap_protein: int = 2,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if warmup_pos_neg_ratio <= 0:
            raise ValueError("warmup_pos_neg_ratio must be positive")
        if len(labels) == 0:
            raise ValueError("labels must be non-empty")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if pool_multiplier <= 0:
            raise ValueError("pool_multiplier must be positive")
        if cap_protein <= 0:
            raise ValueError("cap_protein must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")

        try:
            processed_labels = [int(label) for label in labels]
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"labels must be convertible to integers: {exc}") from exc

        if any(label not in (0, 1) for label in processed_labels):
            raise ValueError("labels must be binary (0 or 1)")

        self.labels = processed_labels
        full_pos_indices = [idx for idx, label in enumerate(processed_labels) if label]
        full_neg_indices = [idx for idx, label in enumerate(processed_labels) if not label]

        self.pos_indices = full_pos_indices[rank::world_size]
        self.neg_indices = full_neg_indices[rank::world_size]

        if not self.pos_indices:
            raise ValueError("StagedOHEMBatchSampler requires at least one positive sample")
        if not self.neg_indices:
            raise ValueError("StagedOHEMBatchSampler requires at least one negative sample")

        self.target_batch_size = int(batch_size)
        self.warmup_pos_neg_ratio = float(warmup_pos_neg_ratio)
        self.warmup_epochs = int(warmup_epochs)
        self.pool_multiplier = int(pool_multiplier)
        self.cap_protein = int(cap_protein)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._rng = random.Random(seed)
        self._epoch = 0

        self.pos_per_batch = self._compute_pos_per_batch()
        self.neg_per_batch = self._compute_neg_per_batch(self.pos_per_batch)
        self.warmup_batch_size = self.pos_per_batch + self.neg_per_batch
        self.mining_batch_size = self.pool_multiplier * self.target_batch_size

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch (0-based)."""
        if epoch < 0:
            raise ValueError("epoch must be >= 0")
        self._epoch = epoch
        if self.seed is not None:
            self._rng = random.Random(self.seed + epoch)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield either warmup batches or mining candidate pools."""
        if self._epoch < self.warmup_epochs:
            yield from self._iter_warmup()
        else:
            yield from self._iter_mining()
        self._epoch += 1

    def __len__(self) -> int:
        """Return current epoch batch count based on the active stage."""
        if self._epoch < self.warmup_epochs:
            return self._warmup_length()
        return self._mining_length()

    def _iter_warmup(self) -> Iterator[list[int]]:
        pos_indices = list(self.pos_indices)
        neg_indices = list(self.neg_indices)
        if self.shuffle:
            self._rng.shuffle(pos_indices)
            self._rng.shuffle(neg_indices)

        neg_offset = 0
        batch_size = self.pos_per_batch

        for start in range(0, len(pos_indices), batch_size):
            pos_batch = pos_indices[start : start + batch_size]
            if len(pos_batch) < batch_size and self.drop_last:
                break
            neg_needed = self._negatives_for_batch(len(pos_batch))
            if neg_offset + neg_needed > len(neg_indices):
                break
            neg_batch = neg_indices[neg_offset : neg_offset + neg_needed]
            neg_offset += neg_needed

            batch = pos_batch + neg_batch
            self._rng.shuffle(batch)
            yield batch

    def _iter_mining(self) -> Iterator[list[int]]:
        pos_indices = list(self.pos_indices)
        neg_indices = list(self.neg_indices)
        if self.shuffle:
            self._rng.shuffle(pos_indices)
            self._rng.shuffle(neg_indices)

        pos_in_pool, neg_in_pool = self._pool_class_counts()
        num_pools = self._mining_length()

        pos_offset = 0
        neg_offset = 0

        for _ in range(num_pools):
            pos_batch = (
                pos_indices[pos_offset : pos_offset + pos_in_pool] if pos_in_pool > 0 else []
            )
            neg_batch = (
                neg_indices[neg_offset : neg_offset + neg_in_pool] if neg_in_pool > 0 else []
            )
            pos_offset += pos_in_pool
            neg_offset += neg_in_pool

            pool = pos_batch + neg_batch
            self._rng.shuffle(pool)
            yield pool

    def _compute_pos_per_batch(self) -> int:
        denom = 1.0 + self.warmup_pos_neg_ratio
        raw = self.target_batch_size / denom
        return max(1, int(math.floor(raw)))

    def _compute_neg_per_batch(self, pos_per_batch: int) -> int:
        neg_per = int(round(pos_per_batch * self.warmup_pos_neg_ratio))
        return max(0, neg_per)

    def _negatives_for_batch(self, positive_count: int) -> int:
        if positive_count <= 0:
            return 0
        if positive_count == self.pos_per_batch:
            return self.neg_per_batch
        scaled = int(math.ceil(positive_count * self.warmup_pos_neg_ratio))
        return max(0, scaled)

    def _pool_class_counts(self) -> tuple[int, int]:
        local_count = len(self.pos_indices) + len(self.neg_indices)
        if local_count == 0:
            return 0, 0
        pos_fraction = len(self.pos_indices) / float(local_count)
        pos_in_pool = int(round(self.mining_batch_size * pos_fraction))
        pos_in_pool = max(0, min(pos_in_pool, len(self.pos_indices)))
        neg_in_pool = self.mining_batch_size - pos_in_pool
        neg_in_pool = max(0, min(neg_in_pool, len(self.neg_indices)))
        if self.mining_batch_size >= 2:
            if len(self.pos_indices) > 0 and pos_in_pool == 0:
                pos_in_pool = 1
            if len(self.neg_indices) > 0 and neg_in_pool == 0:
                neg_in_pool = 1
        if pos_in_pool + neg_in_pool > self.mining_batch_size:
            if pos_in_pool >= neg_in_pool and pos_in_pool > 1:
                pos_in_pool -= pos_in_pool + neg_in_pool - self.mining_batch_size
            elif neg_in_pool > 1:
                neg_in_pool -= pos_in_pool + neg_in_pool - self.mining_batch_size
        if pos_in_pool + neg_in_pool == 0:
            return 0, 0
        if pos_in_pool + neg_in_pool < self.mining_batch_size:
            remaining = self.mining_batch_size - (pos_in_pool + neg_in_pool)
            if len(self.neg_indices) - neg_in_pool >= remaining:
                neg_in_pool += remaining
            elif len(self.pos_indices) - pos_in_pool >= remaining:
                pos_in_pool += remaining
        return pos_in_pool, neg_in_pool

    def _warmup_length(self) -> int:
        if self.pos_per_batch <= 0 or self.neg_per_batch <= 0:
            return 0
        pos_batches = len(self.pos_indices) // self.pos_per_batch
        neg_batches = len(self.neg_indices) // self.neg_per_batch
        return min(pos_batches, neg_batches)

    def _mining_length(self) -> int:
        pos_in_pool, neg_in_pool = self._pool_class_counts()
        pos_limit = math.inf if pos_in_pool == 0 else len(self.pos_indices) // pos_in_pool
        neg_limit = math.inf if neg_in_pool == 0 else len(self.neg_indices) // neg_in_pool
        if pos_limit is math.inf and neg_limit is math.inf:
            return 0
        return int(min(pos_limit, neg_limit))


__all__ = [
    "ImbalancedBatchSampler",
    "StagedOHEMBatchSampler",
]
