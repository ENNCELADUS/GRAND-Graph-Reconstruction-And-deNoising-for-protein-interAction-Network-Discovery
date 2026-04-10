"""Data input utilities for embedding-cache-backed protein pair training."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.embed import EmbeddingCacheManifest, ensure_embeddings_ready, load_cached_embedding
from src.utils.config import ConfigDict, as_bool, as_float, as_int, as_str, get_section
from src.utils.data_samplers import StagedOHEMBatchSampler


@dataclass(frozen=True)
class PPIPairRecord:
    """Single protein-protein interaction record.

    Attributes:
        protein_a: First protein identifier.
        protein_b: Second protein identifier.
        label: Binary interaction label.
    """

    protein_a: str
    protein_b: str
    label: int


def _read_ppi_records(file_path: Path) -> list[PPIPairRecord]:
    """Read tab-separated PPI records.

    Args:
        file_path: Input interaction file path.

    Returns:
        Parsed interaction records.

    Raises:
        ValueError: If no valid records are parsed.
    """
    records: list[PPIPairRecord] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split("\t")]
            if len(parts) < 2:
                continue
            if not parts[0] or not parts[1]:
                continue

            if len(parts) >= 3 and parts[2]:
                try:
                    label = int(float(parts[2]))
                except ValueError:
                    continue
            else:
                label = 1

            records.append(PPIPairRecord(parts[0], parts[1], label))

    if not records:
        raise ValueError(f"No valid PPI records found in {file_path}")
    return records


class PRINGPairDataset(Dataset[dict[str, torch.Tensor]]):
    """PPI dataset that loads precomputed protein embeddings from cache."""

    def __init__(
        self,
        file_path: Path,
        input_dim: int,
        max_sequence_length: int,
        cache_dir: Path,
        embedding_index: dict[str, str],
        *,
        cache_embeddings_in_memory: bool = False,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")

        self._records = _read_ppi_records(file_path=file_path)
        self._input_dim = int(input_dim)
        self._max_sequence_length = int(max_sequence_length)
        self._cache_dir = cache_dir
        self._embedding_index = dict(embedding_index)
        self._cache_embeddings_in_memory = cache_embeddings_in_memory
        self._embedding_cache: dict[str, torch.Tensor] = {}
        proteins = sorted(
            {record.protein_a for record in self._records}
            | {record.protein_b for record in self._records}
        )
        self._protein_to_id = {protein: index for index, protein in enumerate(proteins)}

        required_ids = {record.protein_a for record in self._records}
        required_ids.update(record.protein_b for record in self._records)
        self._required_ids = frozenset(required_ids)
        missing_ids = sorted(
            protein_id for protein_id in required_ids if protein_id not in self._embedding_index
        )
        if missing_ids:
            preview = ", ".join(missing_ids[:10])
            raise FileNotFoundError(
                f"Embedding index is missing proteins required by {file_path}: "
                f"{preview} (missing={len(missing_ids)})"
            )

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self._records)

    def labels(self) -> list[int]:
        """Return binary labels for all records."""
        return [int(record.label) for record in self._records]

    def protein_ids(self, indices: Sequence[int] | None = None) -> list[str]:
        """Return sorted protein IDs referenced by the dataset or a record subset."""
        if indices is None:
            return sorted(self._required_ids)

        protein_ids: set[str] = set()
        for index in indices:
            record = self._records[index]
            protein_ids.add(record.protein_a)
            protein_ids.add(record.protein_b)
        return sorted(protein_ids)

    def pair_records(self) -> list[PPIPairRecord]:
        """Return a copy of parsed pair records."""
        return list(self._records)

    def preload_embeddings(self, protein_ids: Iterable[str] | None = None) -> int:
        """Load selected embeddings into memory and return the cache size."""
        for protein_id in protein_ids or self.protein_ids():
            self._embedding_cache.setdefault(
                protein_id,
                load_cached_embedding(
                    cache_dir=self._cache_dir,
                    index=self._embedding_index,
                    protein_id=protein_id,
                    expected_input_dim=self._input_dim,
                    max_sequence_length=self._max_sequence_length,
                ),
            )
        return len(self._embedding_cache)

    def _load_embedding(self, protein_id: str) -> torch.Tensor:
        """Load one cached embedding tensor for a protein ID."""
        cached_embedding = self._embedding_cache.get(protein_id)
        if cached_embedding is not None:
            return cached_embedding

        embedding = load_cached_embedding(
            cache_dir=self._cache_dir,
            index=self._embedding_index,
            protein_id=protein_id,
            expected_input_dim=self._input_dim,
            max_sequence_length=self._max_sequence_length,
        )
        if self._cache_embeddings_in_memory:
            self._embedding_cache[protein_id] = embedding
        return embedding

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one dataset example."""
        record = self._records[index]
        emb_a = self._load_embedding(record.protein_a)
        emb_b = self._load_embedding(record.protein_b)
        len_a = torch.tensor(emb_a.size(0), dtype=torch.long)
        len_b = torch.tensor(emb_b.size(0), dtype=torch.long)
        label = torch.tensor(float(record.label), dtype=torch.float32)
        protein_a_id = torch.tensor(self._protein_to_id[record.protein_a], dtype=torch.long)
        protein_b_id = torch.tensor(self._protein_to_id[record.protein_b], dtype=torch.long)
        return {
            "emb_a": emb_a,
            "emb_b": emb_b,
            "len_a": len_a,
            "len_b": len_b,
            "label": label,
            "protein_a_id": protein_a_id,
            "protein_b_id": protein_b_id,
        }


def _collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate dataset samples with independent pair-wise sequence padding."""
    emb_a = pad_sequence([sample["emb_a"] for sample in batch], batch_first=True)
    emb_b = pad_sequence([sample["emb_b"] for sample in batch], batch_first=True)
    return {
        "emb_a": emb_a,
        "emb_b": emb_b,
        "len_a": torch.stack([sample["len_a"] for sample in batch], dim=0),
        "len_b": torch.stack([sample["len_b"] for sample in batch], dim=0),
        "label": torch.stack([sample["label"] for sample in batch], dim=0),
        "protein_a_id": torch.stack([sample["protein_a_id"] for sample in batch], dim=0),
        "protein_b_id": torch.stack([sample["protein_b_id"] for sample in batch], dim=0),
    }


def _build_split_loader(
    split_path: Path,
    config: ConfigDict,
    embedding_cache: EmbeddingCacheManifest,
    input_dim: int,
    max_sequence_length: int,
    seed: int,
    shuffle: bool,
    distributed: bool,
    rank: int,
    world_size: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a split-specific data loader."""
    training_cfg = get_section(config, "training_config")
    data_cfg = get_section(config, "data_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")

    batch_size = as_int(training_cfg.get("batch_size", 8), "training_config.batch_size")
    num_workers = as_int(dataloader_cfg.get("num_workers", 0), "data_config.dataloader.num_workers")
    pin_memory = as_bool(
        dataloader_cfg.get("pin_memory", False),
        "data_config.dataloader.pin_memory",
    )
    drop_last = as_bool(dataloader_cfg.get("drop_last", False), "data_config.dataloader.drop_last")

    dataset = PRINGPairDataset(
        file_path=split_path,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        cache_dir=embedding_cache.cache_dir,
        embedding_index=embedding_cache.index,
    )

    sampler: DistributedSampler[dict[str, torch.Tensor]] | None = None
    batch_sampler: StagedOHEMBatchSampler | None = None
    should_shuffle = shuffle
    sampling_raw = dataloader_cfg.get("sampling", {})
    if not isinstance(sampling_raw, dict):
        raise ValueError("data_config.dataloader.sampling must be a mapping")
    sampling_cfg = sampling_raw
    sampling_strategy = as_str(
        sampling_cfg.get("strategy", "none"),
        "data_config.dataloader.sampling.strategy",
    ).lower()
    is_train_loader = shuffle

    if is_train_loader and sampling_strategy == "ohem":
        labels = dataset.labels()
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        natural_ratio = float(neg_count) / float(max(1, pos_count))
        batch_sampler = StagedOHEMBatchSampler(
            labels=labels,
            batch_size=batch_size,
            warmup_pos_neg_ratio=as_float(
                sampling_cfg.get("warmup_pos_neg_ratio", natural_ratio),
                "data_config.dataloader.sampling.warmup_pos_neg_ratio",
            ),
            warmup_epochs=as_int(
                sampling_cfg.get("warmup_epochs", 0),
                "data_config.dataloader.sampling.warmup_epochs",
            ),
            pool_multiplier=as_int(
                sampling_cfg.get("pool_multiplier", 32),
                "data_config.dataloader.sampling.pool_multiplier",
            ),
            cap_protein=as_int(
                sampling_cfg.get("cap_protein", 4),
                "data_config.dataloader.sampling.cap_protein",
            ),
            rank=rank if distributed else 0,
            world_size=world_size if distributed else 1,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        )
        should_shuffle = False
    if distributed and batch_sampler is None:
        sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        should_shuffle = False

    if batch_sampler is not None:
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=_collate_batch,
        )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=should_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=_collate_batch,
    )


def _distributed_barrier_if_initialized() -> None:
    """Run ``dist.barrier`` when a process group is active."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def build_dataloaders(
    config: ConfigDict,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, DataLoader[dict[str, object]]]:
    """Build train/valid/test loaders from the global config.

    Args:
        config: Root configuration mapping.
        distributed: Whether distributed data loading is enabled.
        rank: Global rank for distributed loading.
        world_size: Number of distributed processes.

    Returns:
        Split dataloader mapping with keys ``train``, ``valid``, and ``test``.

    Raises:
        FileNotFoundError: If benchmark root, split files, or embedding files are missing.
        ValueError: If embeddings are invalid for configured model input dimension.
    """
    run_cfg = get_section(config, "run_config")
    data_cfg = get_section(config, "data_config")
    model_cfg = get_section(config, "model_config")
    benchmark_cfg = get_section(data_cfg, "benchmark")
    dataloader_cfg = get_section(data_cfg, "dataloader")

    benchmark_root = Path(str(benchmark_cfg.get("root_dir", "")))
    if not benchmark_root.exists():
        raise FileNotFoundError(f"Benchmark root does not exist: {benchmark_root}")

    train_path = Path(str(dataloader_cfg.get("train_dataset", "")))
    valid_path = Path(str(dataloader_cfg.get("valid_dataset", "")))
    test_path = Path(str(dataloader_cfg.get("test_dataset", "")))
    split_paths = [train_path, valid_path, test_path]
    for split_path in split_paths:
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {split_path}")

    input_dim = as_int(model_cfg.get("input_dim", 0), "model_config.input_dim")
    max_sequence_length = as_int(
        data_cfg.get("max_sequence_length", 64),
        "data_config.max_sequence_length",
    )
    seed = as_int(run_cfg.get("seed", 0), "run_config.seed")

    if distributed:
        distributed_initialized = dist.is_available() and dist.is_initialized()
        allow_generation = True if distributed_initialized else rank == 0
        _distributed_barrier_if_initialized()
        embedding_cache = ensure_embeddings_ready(
            config=config,
            split_paths=split_paths,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
            allow_generation=allow_generation,
        )
        _distributed_barrier_if_initialized()
    else:
        embedding_cache = ensure_embeddings_ready(
            config=config,
            split_paths=split_paths,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
            allow_generation=True,
        )

    return {
        "train": cast(
            DataLoader[dict[str, object]],
            _build_split_loader(
                split_path=train_path,
                config=config,
                embedding_cache=embedding_cache,
                input_dim=input_dim,
                max_sequence_length=max_sequence_length,
                seed=seed,
                shuffle=True,
                distributed=distributed,
                rank=rank,
                world_size=world_size,
            ),
        ),
        "valid": cast(
            DataLoader[dict[str, object]],
            _build_split_loader(
                split_path=valid_path,
                config=config,
                embedding_cache=embedding_cache,
                input_dim=input_dim,
                max_sequence_length=max_sequence_length,
                seed=seed + 1,
                shuffle=False,
                distributed=False,
                rank=rank,
                world_size=world_size,
            ),
        ),
        "test": cast(
            DataLoader[dict[str, object]],
            _build_split_loader(
                split_path=test_path,
                config=config,
                embedding_cache=embedding_cache,
                input_dim=input_dim,
                max_sequence_length=max_sequence_length,
                seed=seed + 2,
                shuffle=False,
                distributed=False,
                rank=rank,
                world_size=world_size,
            ),
        ),
    }
