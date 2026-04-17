"""Public API for the accelerator-owned pipeline package."""

from __future__ import annotations

__all__ = [
    "AcceleratorLike",
    "DistributedContext",
    "PipelineRuntime",
    "build_accelerator",
    "build_runtime",
    "execute_pipeline",
    "execute_pipeline_with_runtime",
]


def __getattr__(name: str) -> object:
    """Lazily expose public pipeline API without import-time cycles."""
    if name in {
        "execute_pipeline",
        "execute_pipeline_with_runtime",
    }:
        from src.pipeline import engine

        return getattr(engine, name)
    if name in {
        "AcceleratorLike",
        "DistributedContext",
        "PipelineRuntime",
        "build_accelerator",
        "build_runtime",
    }:
        from src.pipeline import runtime

        return getattr(runtime, name)
    raise AttributeError(name)
