"""Bootstrap helpers for CLI pipeline execution."""

from __future__ import annotations

import argparse
import logging
import os
import random
from types import ModuleType

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRAND training/evaluation pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rank_from_env() -> int:
    """Parse global rank from environment, defaulting to zero."""
    rank_raw = os.environ.get("RANK", "0")
    try:
        return int(rank_raw)
    except ValueError:
        return 0


def configure_root_logging(logging_module: ModuleType, rank: int) -> None:
    """Configure process-level logging."""
    logging_module.captureWarnings(True)
    if rank == 0:
        logging_module.basicConfig(level=logging.INFO, force=True)
        return
    logging_module.basicConfig(level=logging.CRITICAL, force=True)


__all__ = [
    "configure_root_logging",
    "parse_args",
    "rank_from_env",
    "set_global_seed",
]
