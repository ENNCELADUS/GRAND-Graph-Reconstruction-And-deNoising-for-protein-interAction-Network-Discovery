"""Compatibility app entrypoint for optimization workflows."""

from __future__ import annotations

from src.optimize.run import main, run_optimization, should_run_optimization

__all__ = ["main", "run_optimization", "should_run_optimization"]
