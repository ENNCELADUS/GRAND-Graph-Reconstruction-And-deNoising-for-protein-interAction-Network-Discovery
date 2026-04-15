"""Compatibility wrappers for pipeline bootstrap helpers."""

from __future__ import annotations

from src.pipeline.bootstrap import (
    configure_root_logging,
    parse_args,
    rank_from_env,
    set_global_seed,
)


def _rank_from_env() -> int:
    """Backward-compatible alias for rank lookup."""
    return rank_from_env()


__all__ = [
    "_rank_from_env",
    "configure_root_logging",
    "parse_args",
    "rank_from_env",
    "set_global_seed",
]
