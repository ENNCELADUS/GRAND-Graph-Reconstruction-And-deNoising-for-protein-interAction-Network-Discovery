"""Compatibility wrapper around the new accelerator-owned pipeline engine."""

from __future__ import annotations

import torch

from src.pipeline.engine import (
    ALLOWED_STAGES,
    DEFAULT_STAGES,
    STAGE_ORDER,
    execute_pipeline,
)
from src.pipeline.engine import (
    evaluation_checkpoint_path as _evaluation_checkpoint_path,
)
from src.pipeline.engine import (
    selected_stages as _selected_stages,
)
from src.pipeline.engine import (
    selected_stages_with_adaptation as _selected_stages_with_adaptation,
)
from src.pipeline.engine import (
    topology_finetune_checkpoint_path as _topology_finetune_checkpoint_path,
)
from src.pipeline.runtime import ddp_find_unused_parameters as _ddp_find_unused_parameters
from src.utils.logging import generate_run_id

__all__ = [
    "ALLOWED_STAGES",
    "DEFAULT_STAGES",
    "STAGE_ORDER",
    "_ddp_find_unused_parameters",
    "_evaluation_checkpoint_path",
    "_selected_stages",
    "_selected_stages_with_adaptation",
    "_topology_finetune_checkpoint_path",
    "execute_pipeline",
    "generate_run_id",
    "torch",
]
