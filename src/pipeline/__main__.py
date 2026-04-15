"""Canonical module entrypoint for the accelerator-owned pipeline."""

from __future__ import annotations

import logging
import os

from src.pipeline.bootstrap import configure_root_logging, parse_args, rank_from_env
from src.pipeline.config import load_pipeline_config
from src.pipeline.engine import execute_pipeline

ROOT_LOGGER = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    """Run the pipeline CLI entrypoint."""
    configure_root_logging(logging, rank_from_env())
    args = parse_args()
    config = load_pipeline_config(args.config)
    ROOT_LOGGER.info("Loaded config: %s", args.config)
    execute_pipeline(config.raw)


if __name__ == "__main__":
    main()
