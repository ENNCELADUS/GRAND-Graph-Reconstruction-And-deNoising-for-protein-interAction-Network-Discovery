"""Unit tests for embedding-backed dataloader construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import src.embed.embed as embed_module
import src.utils.data_io as data_io
import torch
from src.embed import EmbeddingCacheManifest
from src.utils.config import ConfigDict


def _write_split(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a}\t{protein_b}\t{label}\n")


def _write_cache(
    cache_dir: Path,
    embeddings: dict[str, torch.Tensor],
    input_dim: int,
    max_sequence_length: int,
) -> dict[str, str]:
    index: dict[str, str] = {}
    for protein_id, tensor in embeddings.items():
        relative_path = embed_module._embedding_relative_path(protein_id)
        output_path = cache_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, output_path)
        index[protein_id] = relative_path

    metadata = {
        "schema_version": 1,
        "source": "esm3",
        "model_name": "esm3_sm_open_v1",
        "input_dim": input_dim,
        "max_sequence_length": max_sequence_length,
        "format": "torch_pt_per_protein",
    }
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (cache_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return index


def _build_config(
    benchmark_root: Path,
    cache_dir: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    input_dim: int,
    max_sequence_length: int,
    sampling: dict[str, object] | None = None,
) -> ConfigDict:
    dataloader_config: dict[str, object] = {
        "train_dataset": str(train_path),
        "valid_dataset": str(valid_path),
        "test_dataset": str(test_path),
        "num_workers": 0,
        "pin_memory": False,
        "drop_last": False,
    }
    if sampling is not None:
        dataloader_config["sampling"] = sampling

    return {
        "run_config": {"seed": 11},
        "data_config": {
            "benchmark": {
                "root_dir": str(benchmark_root),
                "processed_dir": str(benchmark_root),
            },
            "embeddings": {
                "source": "esm3",
                "cache_dir": str(cache_dir),
                "model_name": "esm3_sm_open_v1",
                "device": "cpu",
            },
            "max_sequence_length": max_sequence_length,
            "dataloader": dataloader_config,
        },
        "model_config": {
            "input_dim": input_dim,
        },
        "training_config": {
            "batch_size": 2,
        },
    }


def test_build_dataloaders_uses_cached_embeddings_and_padding(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    train_path = tmp_path / "train.txt"
    valid_path = tmp_path / "valid.txt"
    test_path = tmp_path / "test.txt"
    _write_split(train_path, [("P1", "P2", 1), ("P3", "P2", 0)])
    _write_split(valid_path, [("P1", "P3", 1)])
    _write_split(test_path, [("P2", "P3", 0)])

    cache_dir = tmp_path / "cache"
    input_dim = 4
    max_sequence_length = 8
    _write_cache(
        cache_dir=cache_dir,
        embeddings={
            "P1": torch.ones((2, input_dim), dtype=torch.float32),
            "P2": torch.full((4, input_dim), 2.0, dtype=torch.float32),
            "P3": torch.full((3, input_dim), 3.0, dtype=torch.float32),
        },
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
    )

    config = _build_config(
        benchmark_root=benchmark_root,
        cache_dir=cache_dir,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
    )
    dataloaders = data_io.build_dataloaders(config=config)

    valid_batch = next(iter(dataloaders["valid"]))
    assert tuple(valid_batch["emb_a"].shape) == (1, 2, input_dim)
    assert tuple(valid_batch["emb_b"].shape) == (1, 3, input_dim)
    assert valid_batch["len_a"].tolist() == [2]
    assert valid_batch["len_b"].tolist() == [3]
    assert valid_batch["label"].tolist() == [1.0]
    assert torch.allclose(valid_batch["emb_a"][0, :2], torch.ones((2, input_dim)))
    assert torch.allclose(valid_batch["emb_b"][0, :3], torch.full((3, input_dim), 3.0))

    train_batch = next(iter(dataloaders["train"]))
    assert tuple(train_batch["emb_a"].shape) == (2, 3, input_dim)
    assert tuple(train_batch["emb_b"].shape) == (2, 4, input_dim)
    assert sorted(train_batch["len_a"].tolist()) == [2, 3]
    assert sorted(train_batch["len_b"].tolist()) == [4, 4]
    assert "protein_a_id" in train_batch
    assert "protein_b_id" in train_batch


def test_build_dataloaders_ohem_uses_pool_batch_sampler(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    train_path = tmp_path / "train.txt"
    valid_path = tmp_path / "valid.txt"
    test_path = tmp_path / "test.txt"
    _write_split(train_path, [("P1", "P2", 1), ("P3", "P4", 1), ("P5", "P6", 0), ("P7", "P8", 0)])
    _write_split(valid_path, [("P1", "P2", 1)])
    _write_split(test_path, [("P5", "P6", 0)])

    cache_dir = tmp_path / "cache"
    input_dim = 4
    max_sequence_length = 8
    _write_cache(
        cache_dir=cache_dir,
        embeddings={
            "P1": torch.ones((2, input_dim), dtype=torch.float32),
            "P2": torch.full((2, input_dim), 2.0, dtype=torch.float32),
            "P3": torch.full((2, input_dim), 3.0, dtype=torch.float32),
            "P4": torch.full((2, input_dim), 4.0, dtype=torch.float32),
            "P5": torch.full((2, input_dim), 5.0, dtype=torch.float32),
            "P6": torch.full((2, input_dim), 6.0, dtype=torch.float32),
            "P7": torch.full((2, input_dim), 7.0, dtype=torch.float32),
            "P8": torch.full((2, input_dim), 8.0, dtype=torch.float32),
        },
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
    )

    config = _build_config(
        benchmark_root=benchmark_root,
        cache_dir=cache_dir,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        sampling={
            "strategy": "ohem",
            "warmup_epochs": 0,
            "pool_multiplier": 2,
            "cap_protein": 4,
        },
    )
    dataloaders = data_io.build_dataloaders(config=config)
    train_batch = next(iter(dataloaders["train"]))
    assert tuple(train_batch["label"].shape) == (4,)


def test_build_dataloaders_calls_ensure_embeddings_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    train_path = tmp_path / "train.txt"
    valid_path = tmp_path / "valid.txt"
    test_path = tmp_path / "test.txt"
    _write_split(train_path, [("A1", "A2", 1)])
    _write_split(valid_path, [("A1", "A2", 1)])
    _write_split(test_path, [("A1", "A2", 1)])

    cache_dir = tmp_path / "cache"
    input_dim = 4
    max_sequence_length = 8
    index = _write_cache(
        cache_dir=cache_dir,
        embeddings={
            "A1": torch.ones((2, input_dim), dtype=torch.float32),
            "A2": torch.ones((3, input_dim), dtype=torch.float32),
        },
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
    )

    called: dict[str, bool] = {"value": False}

    def _fake_ensure_embeddings_ready(
        config: ConfigDict,
        split_paths: list[Path],
        input_dim: int,
        max_sequence_length: int,
        allow_generation: bool = True,
    ) -> EmbeddingCacheManifest:
        del config, split_paths, input_dim, max_sequence_length
        called["value"] = True
        assert allow_generation is True
        return EmbeddingCacheManifest(
            cache_dir=cache_dir,
            index=index,
            required_ids=frozenset({"A1", "A2"}),
        )

    monkeypatch.setattr(data_io, "ensure_embeddings_ready", _fake_ensure_embeddings_ready)

    config = _build_config(
        benchmark_root=benchmark_root,
        cache_dir=cache_dir,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
    )
    data_io.build_dataloaders(config=config)
    assert called["value"] is True


def test_pring_pair_dataset_preloads_unique_embeddings_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_path = tmp_path / "pairs.txt"
    _write_split(
        split_path,
        [("P1", "P2", 1), ("P1", "P3", 0), ("P2", "P3", 1)],
    )

    load_counts: dict[str, int] = {}

    def _fake_load_cached_embedding(
        cache_dir: Path,
        index: dict[str, str],
        protein_id: str,
        expected_input_dim: int | None = None,
        max_sequence_length: int | None = None,
    ) -> torch.Tensor:
        del cache_dir, index, expected_input_dim, max_sequence_length
        load_counts[protein_id] = load_counts.get(protein_id, 0) + 1
        fill_value = float(len(protein_id))
        return torch.full((2, 4), fill_value, dtype=torch.float32)

    monkeypatch.setattr(data_io, "load_cached_embedding", _fake_load_cached_embedding)

    dataset = data_io.PRINGPairDataset(
        file_path=split_path,
        input_dim=4,
        max_sequence_length=8,
        cache_dir=tmp_path / "cache",
        embedding_index={"P1": "p1.pt", "P2": "p2.pt", "P3": "p3.pt"},
        cache_embeddings_in_memory=True,
    )

    cached_count = dataset.preload_embeddings()
    first_item = dataset[0]
    second_item = dataset[1]

    assert cached_count == 3
    assert load_counts == {"P1": 1, "P2": 1, "P3": 1}
    assert tuple(first_item["emb_a"].shape) == (2, 4)
    assert tuple(second_item["emb_b"].shape) == (2, 4)


def test_build_dataloaders_failfast_when_cache_missing_for_non_main_rank(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    train_path = tmp_path / "train.txt"
    valid_path = tmp_path / "valid.txt"
    test_path = tmp_path / "test.txt"
    _write_split(train_path, [("M1", "M2", 1)])
    _write_split(valid_path, [("M1", "M2", 1)])
    _write_split(test_path, [("M1", "M2", 1)])

    cache_dir = tmp_path / "cache"
    config = _build_config(
        benchmark_root=benchmark_root,
        cache_dir=cache_dir,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        input_dim=4,
        max_sequence_length=8,
    )

    with pytest.raises(FileNotFoundError, match="Missing embeddings"):
        data_io.build_dataloaders(config=config, distributed=True, rank=1, world_size=2)


def test_build_dataloaders_failfast_on_dim_mismatch_for_non_main_rank(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    train_path = tmp_path / "train.txt"
    valid_path = tmp_path / "valid.txt"
    test_path = tmp_path / "test.txt"
    _write_split(train_path, [("D1", "D1", 1)])
    _write_split(valid_path, [("D1", "D1", 1)])
    _write_split(test_path, [("D1", "D1", 1)])

    cache_dir = tmp_path / "cache"
    _write_cache(
        cache_dir=cache_dir,
        embeddings={
            "D1": torch.ones((3, 3), dtype=torch.float32),
        },
        input_dim=4,
        max_sequence_length=8,
    )

    config = _build_config(
        benchmark_root=benchmark_root,
        cache_dir=cache_dir,
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        input_dim=4,
        max_sequence_length=8,
    )

    with pytest.raises(ValueError, match="Invalid embeddings"):
        data_io.build_dataloaders(config=config, distributed=True, rank=1, world_size=2)
