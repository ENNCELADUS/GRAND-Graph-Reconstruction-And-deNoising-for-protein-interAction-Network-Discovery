"""Unit tests for PRING-style explicit negative supervision generation."""

from __future__ import annotations

from pathlib import Path

from src.topology.negative_sampling import write_exclusive_ratio_supervision_files


def _read_positive_pairs(path: Path) -> set[tuple[str, str]]:
    positive_pairs: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3 or int(float(parts[2])) <= 0:
                continue
            node_a, node_b = sorted((parts[0], parts[1]))
            positive_pairs.add((node_a, node_b))
    return positive_pairs


def _read_counts(path: Path) -> tuple[int, int]:
    positive_count = 0
    negative_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            if int(float(parts[2])) > 0:
                positive_count += 1
            else:
                negative_count += 1
    return positive_count, negative_count


def _read_negative_pairs(path: Path) -> set[tuple[str, str]]:
    negative_pairs: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3 or int(float(parts[2])) > 0:
                continue
            node_a, node_b = sorted((parts[0], parts[1]))
            negative_pairs.add((node_a, node_b))
    return negative_pairs


def test_write_exclusive_ratio_supervision_files_generates_disjoint_ratio5_splits(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "human" / "BFS"
    split_dir.mkdir(parents=True, exist_ok=True)
    all_positive_path = tmp_path / "human_ppi.txt"
    all_positive_path.write_text(
        "\n".join(
            [
                "P1\tP2",
                "P3\tP4",
                "P5\tP6",
                "P7\tP8",
                "P9\tP10",
                "P11\tP12",
                "P13\tP14",
                "P15\tP16",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (split_dir / "human_train_ppi.txt").write_text(
        "\n".join(
            [
                "P1\tP2\t1",
                "P3\tP4\t1",
                "P1\tP3\t0",
                "P1\tP4\t0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (split_dir / "human_val_ppi.txt").write_text(
        "\n".join(
            [
                "P5\tP6\t1",
                "P7\tP8\t1",
                "P2\tP4\t0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (split_dir / "human_test_ppi.txt").write_text(
        "\n".join(
            [
                "P9\tP10\t1",
                "P11\tP12\t1",
                "P13\tP14\t1",
                "P15\tP16\t1",
                "P1\tP5\t0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = write_exclusive_ratio_supervision_files(
        split_dir=split_dir,
        global_positive_path=all_positive_path,
        train_input_path=split_dir / "human_train_ppi.txt",
        valid_input_path=split_dir / "human_val_ppi.txt",
        test_input_path=split_dir / "human_test_ppi.txt",
        negative_ratio=5,
        seed=23,
    )

    train_pos, train_neg = _read_counts(manifest.train_output_path)
    valid_pos, valid_neg = _read_counts(manifest.valid_output_path)
    test_pos, test_neg = _read_counts(manifest.test_output_path)
    old_negative_pairs = (
        _read_negative_pairs(split_dir / "human_train_ppi.txt")
        | _read_negative_pairs(split_dir / "human_val_ppi.txt")
        | _read_negative_pairs(split_dir / "human_test_ppi.txt")
    )
    new_negative_pairs = (
        _read_negative_pairs(manifest.train_output_path)
        | _read_negative_pairs(manifest.valid_output_path)
        | _read_negative_pairs(manifest.test_output_path)
    )

    assert train_neg == train_pos * 5
    assert valid_neg == valid_pos * 5
    assert test_neg == test_pos * 5
    assert old_negative_pairs.isdisjoint(new_negative_pairs)


def test_write_exclusive_ratio_supervision_files_keeps_validation_positives_out_of_train(
    tmp_path: Path,
) -> None:
    split_dir = tmp_path / "human" / "BFS"
    split_dir.mkdir(parents=True, exist_ok=True)
    all_positive_path = tmp_path / "human_ppi.txt"
    all_positive_path.write_text(
        "\n".join(
            [
                "P1\tP2",
                "P3\tP4",
                "P5\tP6",
                "P7\tP8",
                "P9\tP10",
                "",
            ]
        ),
        encoding="utf-8",
    )
    train_input_path = split_dir / "human_train_ppi.txt"
    valid_input_path = split_dir / "human_val_ppi.txt"
    test_input_path = split_dir / "human_test_ppi.txt"
    train_input_path.write_text(
        "\n".join(
            [
                "P1\tP2\t1",
                "P3\tP4\t1",
                "P1\tP3\t0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    valid_input_path.write_text(
        "\n".join(
            [
                "P5\tP6\t1",
                "P7\tP8\t1",
                "P1\tP4\t0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    test_input_path.write_text(
        "\n".join(
            [
                "P9\tP10\t1",
                "P2\tP5\t0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest = write_exclusive_ratio_supervision_files(
        split_dir=split_dir,
        global_positive_path=all_positive_path,
        train_input_path=train_input_path,
        valid_input_path=valid_input_path,
        test_input_path=test_input_path,
        negative_ratio=5,
        seed=23,
    )

    original_train_positives = _read_positive_pairs(train_input_path)
    original_valid_positives = _read_positive_pairs(valid_input_path)

    assert _read_positive_pairs(manifest.train_output_path) == original_train_positives
    assert _read_positive_pairs(manifest.valid_output_path) == original_valid_positives
    assert _read_positive_pairs(manifest.train_output_path).isdisjoint(original_valid_positives)
