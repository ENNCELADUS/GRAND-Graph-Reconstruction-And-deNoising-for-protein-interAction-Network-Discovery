"""Characterization tests for public import compatibility during refactors."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import src
import src.utils as utils
from src.embed import EmbeddingCacheManifest, ensure_embeddings_ready, load_cached_embedding
from src.pipeline.runtime import PipelineRuntime
from src.train.config import LossConfig as SharedLossConfig
from src.train.strategies.ohem import OHEMSampleStrategy, select_ohem_indices
from src.utils.losses import LossConfig, binary_classification_loss


def test_pipeline_stage_modules_are_canonical_runtime_entrypoints() -> None:
    """Lock the new canonical stage package and runtime-first stage signatures."""
    expected_entrypoints = {
        "src.pipeline.stages.train": ("run_training_stage", None),
        "src.pipeline.stages.evaluate": ("run_evaluation_stage", "checkpoint_path"),
        "src.pipeline.stages.adapt": ("run_shot_adaptation_stage", "checkpoint_path"),
        "src.pipeline.stages.topology_finetune": (
            "run_topology_finetuning_stage",
            "checkpoint_path",
        ),
        "src.pipeline.stages.topology_evaluate": (
            "run_topology_evaluation_stage",
            "checkpoint_path",
        ),
    }

    for module_name, (function_name, checkpoint_parameter) in expected_entrypoints.items():
        module = importlib.import_module(module_name)
        signature = inspect.signature(getattr(module, function_name))
        assert list(signature.parameters)[:3] == ["runtime", "model", "dataloaders"]
        if checkpoint_parameter is None:
            assert checkpoint_parameter not in signature.parameters
            continue
        checkpoint_path = signature.parameters[checkpoint_parameter]
        assert checkpoint_path.kind is inspect.Parameter.KEYWORD_ONLY


def test_run_package_has_no_python_compatibility_modules() -> None:
    """Lock removal of the legacy ``src/run`` compatibility package."""
    run_package_dir = Path(__file__).resolve().parents[2] / "src" / "run"

    assert not list(run_package_dir.glob("*.py"))


def test_optimize_app_compatibility_package_is_removed() -> None:
    """Lock removal of the pure re-export optimize app shim."""
    optimize_app_dir = Path(__file__).resolve().parents[2] / "src" / "apps" / "optimize"

    assert not optimize_app_dir.exists()


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


def test_runtime_no_longer_stores_stage_checkpoint_paths() -> None:
    """Lock explicit checkpoint passing from engine to stages."""
    assert "checkpoint_paths" not in PipelineRuntime.__dataclass_fields__


def test_stage_cleanup_helpers_are_removed() -> None:
    """Lock removal of thin wrappers and dead logger-name helper."""
    adapt_module = importlib.import_module("src.pipeline.stages.adapt")
    train_module = importlib.import_module("src.pipeline.stages.train")

    assert not hasattr(adapt_module, "_move_batch_to_device")
    assert not hasattr(adapt_module, "_forward_batch_without_labels")
    assert not hasattr(adapt_module, "_forward_model")
    assert not hasattr(train_module, "_stage_logger_name")


def test_public_package_exports_include_v3_1() -> None:
    """Lock the top-level GRAND export surface to match ``src.model``."""
    assert "V3_1" in src.__all__
    assert getattr(src, "V3_1").__name__ == "V3_1"


def test_utils_package_does_not_reexport_train_strategy_symbols() -> None:
    """Lock ``src.utils`` as a utils-only package without train cross-exports."""
    assert "OHEMSampleStrategy" not in utils.__all__
    assert "select_ohem_indices" not in utils.__all__
    assert not hasattr(utils, "OHEMSampleStrategy")
    assert not hasattr(utils, "select_ohem_indices")
