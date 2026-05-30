"""Online teacher wrapper for TCCIG training."""

from __future__ import annotations

from typing import cast

import torch
from torch.optim import Optimizer

from src.pipeline.runtime import AcceleratorLike
from src.train.config import OptimizerConfig
from src.train.tccig.config import (
    parse_optimizer_config,
    parse_teacher_training_config,
    teacher_config,
)
from src.train.tccig.mgae import MGAETeacher
from src.utils.config import ConfigDict, as_bool, as_float, as_int


class OnlineTCCIGTeacher:
    """Online train-only MGAE teacher used for TCCIG distillation."""

    def __init__(
        self,
        *,
        teacher: MGAETeacher,
        optimizer: Optimizer,
        mask_ratio: float,
        negative_ratio: int,
    ) -> None:
        self.teacher = teacher
        self.optimizer = optimizer
        self.mask_ratio = mask_ratio
        self.negative_ratio = negative_ratio

    @classmethod
    def build(
        cls,
        *,
        train_cfg: ConfigDict,
        input_dim: int,
        device: torch.device,
    ) -> OnlineTCCIGTeacher | None:
        """Build the online teacher when enabled in config."""
        raw_teacher_cfg = teacher_config(train_cfg)
        if not as_bool(raw_teacher_cfg.get("enabled", True), "tccig_train.teacher.enabled"):
            return None

        hidden_dim = as_int(
            raw_teacher_cfg.get("hidden_dim", min(128, max(16, input_dim * 2))),
            "tccig_train.teacher.hidden_dim",
        )
        num_layers = as_int(
            raw_teacher_cfg.get("num_layers", 2),
            "tccig_train.teacher.num_layers",
        )
        decoder_hidden_dim = as_int(
            raw_teacher_cfg.get("decoder_hidden_dim", hidden_dim),
            "tccig_train.teacher.decoder_hidden_dim",
        )
        dropout = as_float(raw_teacher_cfg.get("dropout", 0.1), "tccig_train.teacher.dropout")
        teacher = MGAETeacher(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            dropout=dropout,
        ).to(device)
        optimizer_cfg = raw_teacher_cfg.get("optimizer", {})
        if optimizer_cfg is None:
            optimizer_cfg = {}
        if not isinstance(optimizer_cfg, dict):
            raise ValueError("tccig_train.teacher.optimizer must be a mapping")
        optimizer_config = parse_optimizer_config(
            optimizer_cfg,
            field_name="tccig_train.teacher.optimizer",
            default_lr=1e-3,
        )
        optimizer = _build_teacher_optimizer(teacher=teacher, optimizer_config=optimizer_config)
        mask_ratio, negative_ratio = parse_teacher_training_config(train_cfg)
        return cls(
            teacher=teacher,
            optimizer=optimizer,
            mask_ratio=mask_ratio,
            negative_ratio=negative_ratio,
        )

    def prepare(self, accelerator: AcceleratorLike) -> OnlineTCCIGTeacher:
        """Prepare teacher components with the active accelerator."""
        prepared_teacher, prepared_optimizer = cast(
            tuple[MGAETeacher, Optimizer],
            accelerator.prepare(self.teacher, self.optimizer),
        )
        return OnlineTCCIGTeacher(
            teacher=prepared_teacher,
            optimizer=prepared_optimizer,
            mask_ratio=self.mask_ratio,
            negative_ratio=self.negative_ratio,
        )

    def train_and_score(
        self,
        *,
        node_features: torch.Tensor,
        positive_edges: torch.Tensor,
        candidate_pairs: torch.Tensor,
        seed: int,
        device: torch.device,
        accelerator: AcceleratorLike,
        loss_scale: float,
    ) -> torch.Tensor:
        """Run one online teacher update and return frozen candidate probabilities."""
        self.teacher.train()
        self.optimizer.zero_grad(set_to_none=True)
        teacher_step = self.teacher.training_step(
            node_features=node_features.detach(),
            positive_edges=positive_edges,
            mask_ratio=self.mask_ratio,
            negative_ratio=self.negative_ratio,
            generator=_torch_generator_for_device(device=device, seed=seed),
        )
        accelerator.backward(teacher_step.loss * loss_scale)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher.score_pairs(
                node_features=node_features.detach(),
                visible_positive_edges=positive_edges,
                candidate_edges=candidate_pairs,
            )
        return torch.sigmoid(teacher_logits)


def _torch_generator_for_device(*, device: torch.device, seed: int) -> torch.Generator:
    """Build a deterministic generator compatible with common tensor devices."""
    generator = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    generator.manual_seed(seed)
    return generator


def _build_teacher_optimizer(
    *,
    teacher: MGAETeacher,
    optimizer_config: OptimizerConfig,
) -> Optimizer:
    optimizer_type = optimizer_config.optimizer_type.lower()
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            params=teacher.parameters(),
            lr=optimizer_config.lr,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            eps=optimizer_config.eps,
            weight_decay=optimizer_config.weight_decay,
        )
    if optimizer_type == "sgd":
        return torch.optim.SGD(
            params=teacher.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer type: {optimizer_config.optimizer_type}")
