"""Compatibility script entrypoint for ``python src/run.py``."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure absolute imports resolve when invoked as `python src/run.py`.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _main() -> None:
    """Delegate execution to package entrypoint."""
    from src.pipeline.__main__ import main

    main()


if __name__ == "__main__":
    _main()
