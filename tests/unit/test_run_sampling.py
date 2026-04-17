"""Unit tests for sampling-related run wiring."""

from __future__ import annotations

import torch
from src.pipeline.runtime import build_accelerator
from src.pipeline.stages.train import build_trainer
from src.utils.config import ConfigDict
from torch import nn


def test_build_trainer_wires_ohem_warmup_epochs() -> None:
    config: ConfigDict = {
        "training_config": {
            "epochs": 2,
            "optimizer": {"type": "adamw", "lr": 1e-3},
            "scheduler": {"type": "none"},
            "loss": {"type": "bce_with_logits", "pos_weight": 1.0, "label_smoothing": 0.0},
            "logging": {"heartbeat_every_n_steps": 0, "validation_metrics": ["auprc"]},
        },
        "data_config": {
            "dataloader": {
                "sampling": {
                    "strategy": "ohem",
                    "cap_protein": 4,
                    "pool_multiplier": 32,
                    "warmup_epochs": 3,
                }
            }
        },
        "device_config": {
            "use_mixed_precision": False,
        },
    }
    model = nn.Linear(4, 1)
    trainer, _ = build_trainer(
        config=config,
        model=model,
        device=torch.device("cpu"),
        accelerator=build_accelerator(
            requested_device="cpu",
            ddp_enabled=False,
            use_mixed_precision=False,
            find_unused_parameters=False,
        ),
        steps_per_epoch=2,
    )
    assert trainer.ohem_strategy is not None
    assert trainer.ohem_strategy.warmup_epochs == 3
    assert trainer.ohem_strategy.target_batch_size == 8
    assert trainer.ohem_strategy.cap_protein == 4
