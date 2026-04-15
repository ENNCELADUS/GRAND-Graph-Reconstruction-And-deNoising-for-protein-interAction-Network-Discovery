"""Offline generator for exclusive 1:5 PRING supervision files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.topology.negative_sampling import write_exclusive_ratio_supervision_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/PRING/species_processed_data/human"),
        help="Human PRING processed-data root containing BFS/DFS/RANDOM_WALK subdirectories.",
    )
    parser.add_argument(
        "--global-positive-path",
        type=Path,
        default=Path("data/PRING/species_processed_data/human/human_ppi.txt"),
        help="Path to the global positive PPI list used to reject known positives.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=5,
        help="Explicit negative-to-positive ratio to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed for deterministic PRING-style negative sampling.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate exclusive ratio supervision files for each Human split strategy."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for method in ("BFS", "DFS", "RANDOM_WALK"):
        split_dir = args.root / method
        manifest = write_exclusive_ratio_supervision_files(
            split_dir=split_dir,
            global_positive_path=args.global_positive_path,
            train_input_path=split_dir / "human_train_ppi.txt",
            valid_input_path=split_dir / "human_val_ppi.txt",
            test_input_path=split_dir / "human_test_ppi.txt",
            negative_ratio=args.negative_ratio,
            seed=args.seed,
        )
        logging.info(
            f"{method}: "
            f"{manifest.train_output_path.name}, "
            f"{manifest.valid_output_path.name}, "
            f"{manifest.test_output_path.name}"
        )


if __name__ == "__main__":
    main()
