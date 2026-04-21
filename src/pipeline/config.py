"""Typed pipeline configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.config import (
    ConfigDict,
    as_bool,
    as_int,
    as_str,
    as_str_list,
    extract_model_kwargs,
    get_section,
    load_config,
)


@dataclass(frozen=True)
class RunSettings:
    """Normalized run configuration values used by orchestration."""

    stages: tuple[str, ...]
    seed: int
    train_run_id: str | None
    topology_finetune_run_id: str | None
    adapt_run_id: str | None
    eval_run_id: str | None
    topology_eval_run_id: str | None
    load_checkpoint_path: str | None
    save_best_only: bool


@dataclass(frozen=True)
class DeviceSettings:
    """Normalized device runtime configuration."""

    requested_device: str
    backend: str
    ddp_enabled: bool
    use_mixed_precision: bool
    find_unused_parameters: bool


@dataclass(frozen=True)
class TrainingSettings:
    """Normalized training flags that influence stage selection."""

    shot_enabled: bool
    shot_method: str


@dataclass(frozen=True)
class PipelineConfig:
    """Typed view over the YAML configuration while preserving the raw payload."""

    raw: ConfigDict
    model_name: str
    run: RunSettings
    device: DeviceSettings
    training: TrainingSettings

    @classmethod
    def from_dict(cls, raw: ConfigDict) -> PipelineConfig:
        """Build a typed pipeline config from the raw config mapping."""
        run_cfg = get_section(raw, "run_config")
        device_cfg = get_section(raw, "device_config")
        training_cfg = get_section(raw, "training_config")
        model_name, _ = extract_model_kwargs(raw)

        domain_adaptation = training_cfg.get("domain_adaptation", {})
        if not isinstance(domain_adaptation, dict):
            domain_adaptation = {}

        return cls(
            raw=raw,
            model_name=model_name,
            run=RunSettings(
                stages=tuple(
                    stage.lower()
                    for stage in as_str_list(
                        run_cfg.get("stages", ["train", "evaluate"]),
                        "run_config.stages",
                    )
                ),
                seed=as_int(run_cfg.get("seed", 0), "run_config.seed"),
                train_run_id=_optional_str(run_cfg.get("train_run_id")),
                topology_finetune_run_id=_optional_str(run_cfg.get("topology_finetune_run_id")),
                adapt_run_id=_optional_str(run_cfg.get("adapt_run_id")),
                eval_run_id=_optional_str(run_cfg.get("eval_run_id")),
                topology_eval_run_id=_optional_str(run_cfg.get("topology_eval_run_id")),
                load_checkpoint_path=_optional_str(run_cfg.get("load_checkpoint_path")),
                save_best_only=as_bool(
                    run_cfg.get("save_best_only", True),
                    "run_config.save_best_only",
                ),
            ),
            device=DeviceSettings(
                requested_device=as_str(
                    device_cfg.get("device", "cpu"),
                    "device_config.device",
                ),
                backend=_device_backend(device_cfg.get("backend", "ddp")),
                ddp_enabled=as_bool(
                    device_cfg.get("ddp_enabled", False),
                    "device_config.ddp_enabled",
                ),
                use_mixed_precision=as_bool(
                    device_cfg.get("use_mixed_precision", False),
                    "device_config.use_mixed_precision",
                ),
                find_unused_parameters=as_bool(
                    device_cfg.get("find_unused_parameters", False),
                    "device_config.find_unused_parameters",
                )
                if device_cfg.get("find_unused_parameters") is not None
                else False,
            ),
            training=TrainingSettings(
                shot_enabled=as_bool(
                    domain_adaptation.get("enabled", False),
                    "training_config.domain_adaptation.enabled",
                ),
                shot_method=as_str(
                    domain_adaptation.get("method", "none"),
                    "training_config.domain_adaptation.method",
                ).lower(),
            ),
        )


def load_pipeline_config(config_path: str | Path) -> PipelineConfig:
    """Load YAML config from disk and project it into typed settings."""
    return PipelineConfig.from_dict(load_config(config_path))


def _optional_str(value: object) -> str | None:
    """Return a non-empty string value or ``None``."""
    if isinstance(value, str) and value:
        return value
    return None


def _device_backend(value: object) -> str:
    """Return validated accelerator backend selection."""
    backend = as_str(value, "device_config.backend").lower()
    if backend not in {"ddp", "deepspeed"}:
        raise ValueError("device_config.backend must be 'ddp' or 'deepspeed'")
    return backend


__all__ = [
    "DeviceSettings",
    "PipelineConfig",
    "RunSettings",
    "TrainingSettings",
    "load_pipeline_config",
]
