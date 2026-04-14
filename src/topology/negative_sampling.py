"""PRING-style explicit negative supervision generation utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RatioSupervisionManifest:
    """Paths for generated explicit-supervision files."""

    train_output_path: Path
    valid_output_path: Path
    test_output_path: Path


def _default_ratio_output_path(*, split_dir: Path, split_name: str, negative_ratio: int) -> Path:
    """Return the standard PRING explicit-supervision output path for one split."""
    return split_dir / f"human_{split_name}_ppi_ratio{negative_ratio}_exclusive.txt"


def _canonical_pair(node_a: str, node_b: str) -> tuple[str, str]:
    return (node_a, node_b) if node_a <= node_b else (node_b, node_a)


def _read_global_positive_pairs(path: Path) -> set[tuple[str, str]]:
    positive_pairs: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            positive_pairs.add(_canonical_pair(parts[0], parts[1]))
    return positive_pairs


def _read_labeled_pairs(path: Path) -> tuple[list[tuple[str, str]], set[tuple[str, str]]]:
    positive_pairs: list[tuple[str, str]] = []
    negative_pairs: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            pair = _canonical_pair(parts[0], parts[1])
            label = int(float(parts[2]))
            if label > 0:
                positive_pairs.append(pair)
            elif label == 0:
                negative_pairs.add(pair)
    return positive_pairs, negative_pairs


def _read_observed_nodes(path: Path) -> set[str]:
    """Return all proteins observed in a labeled PRING pair file."""
    observed_nodes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            if parts[0]:
                observed_nodes.add(parts[0])
            if parts[1]:
                observed_nodes.add(parts[1])
    return observed_nodes


def _resolve_all_positive_pairs(
    *,
    global_positive_path: Path | None,
    train_input_path: Path,
    valid_input_path: Path,
    test_input_path: Path,
) -> set[tuple[str, str]]:
    """Load known positives from the global file, with split-local fallback."""
    if global_positive_path is not None and global_positive_path.exists():
        return _read_global_positive_pairs(global_positive_path)

    all_positive_pairs: set[tuple[str, str]] = set()
    for path in (train_input_path, valid_input_path, test_input_path):
        positive_pairs, _ = _read_labeled_pairs(path)
        all_positive_pairs.update(positive_pairs)
    return all_positive_pairs


def _sample_exclusive_negative_pairs(
    *,
    positive_pairs: list[tuple[str, str]],
    all_positive_pairs: set[tuple[str, str]],
    forbidden_pairs: set[tuple[str, str]],
    negative_ratio: int,
    rng: random.Random,
    candidate_nodes: list[str] | None = None,
) -> list[tuple[str, str]]:
    if negative_ratio <= 0:
        return []
    if not positive_pairs:
        return []

    candidates = candidate_nodes or [node for pair in positive_pairs for node in pair]
    required_negative_count = len(positive_pairs) * negative_ratio
    sampled_negatives: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(required_negative_count * 1000, 10000)
    while len(sampled_negatives) < required_negative_count:
        if attempts >= max_attempts:
            raise ValueError(
                "Unable to sample enough exclusive negative pairs for requested ratio"
            )
        attempts += 1
        node_a = rng.choice(candidates)
        node_b = rng.choice(candidates)
        if node_a == node_b:
            continue
        pair = _canonical_pair(node_a, node_b)
        if pair in all_positive_pairs or pair in forbidden_pairs or pair in sampled_negatives:
            continue
        sampled_negatives.add(pair)
    return sorted(sampled_negatives)


def _write_labeled_pairs(
    *,
    path: Path,
    positive_pairs: list[tuple[str, str]],
    negative_pairs: list[tuple[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for node_a, node_b in positive_pairs:
            handle.write(f"{node_a}\t{node_b}\t1\n")
        for node_a, node_b in negative_pairs:
            handle.write(f"{node_a}\t{node_b}\t0\n")


def write_exclusive_ratio_supervision_files(
    *,
    split_dir: Path,
    global_positive_path: Path | None,
    train_input_path: Path,
    valid_input_path: Path,
    test_input_path: Path,
    negative_ratio: int,
    seed: int,
    train_output_path: Path | None = None,
    valid_output_path: Path | None = None,
    test_output_path: Path | None = None,
) -> RatioSupervisionManifest:
    """Generate PRING-style explicit negatives disjoint from the existing 1:1 files."""
    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive")

    train_positive_pairs, train_negative_pairs = _read_labeled_pairs(train_input_path)
    valid_positive_pairs, valid_negative_pairs = _read_labeled_pairs(valid_input_path)
    test_positive_pairs, test_negative_pairs = _read_labeled_pairs(test_input_path)
    candidate_nodes = sorted(
        _read_observed_nodes(train_input_path)
        | _read_observed_nodes(valid_input_path)
        | _read_observed_nodes(test_input_path)
    )
    all_positive_pairs = _resolve_all_positive_pairs(
        global_positive_path=global_positive_path,
        train_input_path=train_input_path,
        valid_input_path=valid_input_path,
        test_input_path=test_input_path,
    )
    forbidden_pairs = (
        set(train_negative_pairs) | set(valid_negative_pairs) | set(test_negative_pairs)
    )
    rng = random.Random(seed)

    new_train_positive_pairs = sorted(train_positive_pairs)
    new_valid_positive_pairs = sorted(valid_positive_pairs)

    new_train_negative_pairs = _sample_exclusive_negative_pairs(
        positive_pairs=new_train_positive_pairs,
        all_positive_pairs=all_positive_pairs,
        forbidden_pairs=forbidden_pairs,
        negative_ratio=negative_ratio,
        rng=rng,
        candidate_nodes=candidate_nodes,
    )
    forbidden_pairs |= set(new_train_negative_pairs)
    new_valid_negative_pairs = _sample_exclusive_negative_pairs(
        positive_pairs=new_valid_positive_pairs,
        all_positive_pairs=all_positive_pairs,
        forbidden_pairs=forbidden_pairs,
        negative_ratio=negative_ratio,
        rng=rng,
        candidate_nodes=candidate_nodes,
    )
    forbidden_pairs |= set(new_valid_negative_pairs)
    new_test_negative_pairs = _sample_exclusive_negative_pairs(
        positive_pairs=sorted(test_positive_pairs),
        all_positive_pairs=all_positive_pairs,
        forbidden_pairs=forbidden_pairs,
        negative_ratio=negative_ratio,
        rng=rng,
        candidate_nodes=candidate_nodes,
    )

    manifest = RatioSupervisionManifest(
        train_output_path=train_output_path
        or _default_ratio_output_path(
            split_dir=split_dir,
            split_name="train",
            negative_ratio=negative_ratio,
        ),
        valid_output_path=valid_output_path
        or _default_ratio_output_path(
            split_dir=split_dir,
            split_name="val",
            negative_ratio=negative_ratio,
        ),
        test_output_path=test_output_path
        or _default_ratio_output_path(
            split_dir=split_dir,
            split_name="test",
            negative_ratio=negative_ratio,
        ),
    )
    _write_labeled_pairs(
        path=manifest.train_output_path,
        positive_pairs=new_train_positive_pairs,
        negative_pairs=new_train_negative_pairs,
    )
    _write_labeled_pairs(
        path=manifest.valid_output_path,
        positive_pairs=new_valid_positive_pairs,
        negative_pairs=new_valid_negative_pairs,
    )
    _write_labeled_pairs(
        path=manifest.test_output_path,
        positive_pairs=sorted(test_positive_pairs),
        negative_pairs=new_test_negative_pairs,
    )
    return manifest


def ensure_ratio_supervision_files(
    *,
    split_dir: Path,
    global_positive_path: Path | None,
    train_input_path: Path,
    valid_input_path: Path,
    test_input_path: Path,
    negative_ratio: int,
    seed: int,
    train_output_path: Path | None = None,
    valid_output_path: Path | None = None,
    test_output_path: Path | None = None,
) -> RatioSupervisionManifest:
    """Create ratio supervision files on demand and return their manifest."""
    manifest = RatioSupervisionManifest(
        train_output_path=train_output_path
        or _default_ratio_output_path(
            split_dir=split_dir,
            split_name="train",
            negative_ratio=negative_ratio,
        ),
        valid_output_path=valid_output_path
        or _default_ratio_output_path(
            split_dir=split_dir,
            split_name="val",
            negative_ratio=negative_ratio,
        ),
        test_output_path=test_output_path
        or _default_ratio_output_path(
            split_dir=split_dir,
            split_name="test",
            negative_ratio=negative_ratio,
        ),
    )
    if (
        manifest.train_output_path.exists()
        and manifest.valid_output_path.exists()
        and manifest.test_output_path.exists()
    ):
        return manifest

    return write_exclusive_ratio_supervision_files(
        split_dir=split_dir,
        global_positive_path=global_positive_path,
        train_input_path=train_input_path,
        valid_input_path=valid_input_path,
        test_input_path=test_input_path,
        negative_ratio=negative_ratio,
        seed=seed,
        train_output_path=manifest.train_output_path,
        valid_output_path=manifest.valid_output_path,
        test_output_path=manifest.test_output_path,
    )
