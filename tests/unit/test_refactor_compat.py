"""Characterization tests for public import compatibility during refactors."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from src.embed import EmbeddingCacheManifest, ensure_embeddings_ready, load_cached_embedding
from src.train.config import LossConfig as SharedLossConfig
from src.train.strategies.ohem import OHEMSampleStrategy, select_ohem_indices
from src.utils.losses import LossConfig, binary_classification_loss


def test_pipeline_stage_modules_are_canonical_runtime_entrypoints() -> None:
    """Lock the new canonical stage package and runtime-first stage signatures."""
    expected_entrypoints = {
        "src.pipeline.stages.train": "run_training_stage",
        "src.pipeline.stages.evaluate": "run_evaluation_stage",
        "src.pipeline.stages.adapt": "run_shot_adaptation_stage",
        "src.pipeline.stages.topology_finetune": "run_topology_finetuning_stage",
        "src.pipeline.stages.topology_evaluate": "run_topology_evaluation_stage",
    }

    for module_name, function_name in expected_entrypoints.items():
        module = importlib.import_module(module_name)
        signature = inspect.signature(getattr(module, function_name))
        assert list(signature.parameters)[:3] == ["runtime", "model", "dataloaders"]


def test_run_package_has_no_python_compatibility_modules() -> None:
    """Lock removal of the legacy ``src/run`` compatibility package."""
    run_package_dir = Path(__file__).resolve().parents[2] / "src" / "run"

    assert not list(run_package_dir.glob("*.py"))


def test_embed_public_exports_contract() -> None:
    """Lock embed module exports expected by data loading and tests."""
    assert EmbeddingCacheManifest.__name__ == "EmbeddingCacheManifest"
    assert callable(ensure_embeddings_ready)
    assert callable(load_cached_embedding)


def test_legacy_losses_and_ohem_imports_contract() -> None:
    """Lock shared utility exports expected across code and tests."""
    assert LossConfig.__name__ == "LossConfig"
    assert callable(binary_classification_loss)
    assert OHEMSampleStrategy.__name__ == "OHEMSampleStrategy"
    assert callable(select_ohem_indices)


def test_losses_legacy_import_points_to_shared_train_config() -> None:
    """Lock compatibility alias between legacy losses path and shared config."""
    assert LossConfig is SharedLossConfig


def test_ohem_public_import_points_to_train_strategy() -> None:
    """Lock the public OHEM import path to the train strategy module."""
    assert OHEMSampleStrategy.__module__ == "src.train.strategies.ohem"
