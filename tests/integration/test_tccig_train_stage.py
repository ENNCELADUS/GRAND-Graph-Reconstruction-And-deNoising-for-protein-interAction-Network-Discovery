"""Integration tests for the TCCIG scratch training stage."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from csv import DictReader
from pathlib import Path
from typing import cast

import networkx as nx
import pytest
import src.pipeline.stages.topology_finetune as topology_finetune_stage
import src.train.tccig.trainer as tccig_trainer_module
import torch
from src.pipeline.runtime import DistributedContext
from src.pipeline.stages.tccig_train import run_tccig_train_stage
from src.pipeline.stages.train import build_model
from src.topology.finetune_data import (
    EdgeCoverEpochPlan,
    EmbeddingRepository,
    ExplicitNegativePairLookup,
)
from src.train.config import OptimizerConfig
from src.train.tccig.config import parse_tccig_train_config
from src.train.tccig.mgae import MGAETeacher
from src.train.tccig.teacher import OnlineTCCIGTeacher
from src.train.tccig.trainer import TCCIGStudentTrainer
from src.train.topology import shared as topology_train
from src.utils.config import ConfigDict, load_config
from src.utils.data_io import build_dataloaders
from tests.integration.test_topology_finetune_stage import _build_finetune_config
from tests.runtime_helpers import NoOpAccelerator, build_stage_runtime
from torch.utils.data import DataLoader


def _build_tccig_train_config(tmp_path: Path) -> ConfigDict:
    config = _build_finetune_config(tmp_path)
    config["model_config"] = {
        "model": "tccig",
        "input_dim": 4,
        "d_model": 8,
        "dropout": 0.0,
        "num_modules": 3,
        "lowrank_dim": 2,
        "candidate_proposer": {"type": "all_pairs"},
        "pair_mlp": {"hidden_dims": [8]},
    }
    run_cfg = cast(ConfigDict, config["run_config"])
    run_cfg["stages"] = ["tccig_train"]
    run_cfg["load_checkpoint_path"] = None
    run_cfg["topology_finetune_run_id"] = None
    run_cfg["tccig_train_run_id"] = "tccig_train_case"

    topology_cfg = cast(ConfigDict, config.pop("topology_finetune"))
    config["tccig_train"] = {
        "init_mode": "scratch",
        "epochs": 0,
        "subgraph_node_range": topology_cfg["subgraph_node_range"],
        "strategy": topology_cfg["strategy"],
        "bce_negative_ratio": topology_cfg["bce_negative_ratio"],
        "pair_batch_size": topology_cfg["pair_batch_size"],
        "decision_threshold": topology_cfg["decision_threshold"],
        "optimizer": topology_cfg["optimizer"],
        "teacher": {"enabled": False},
        "losses": {"edge": 1.0, "teacher": 0.0, "clustering": 0.0},
    }
    return config


def test_run_tccig_train_stage_writes_scratch_artifacts(tmp_path: Path) -> None:
    config = _build_tccig_train_config(tmp_path)
    model = build_model(config)
    dataloaders = build_dataloaders(config=config)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"tccig_train": "tccig_train_case"},
        )
        best_checkpoint = run_tccig_train_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
        )
    finally:
        os.chdir(previous_cwd)

    assert best_checkpoint == Path("models/tccig/tccig_train/tccig_train_case/best_model.pth")
    log_dir = tmp_path / "logs" / "tccig" / "tccig_train" / "tccig_train_case"
    assert (tmp_path / best_checkpoint).exists()
    assert (log_dir / "tccig_train_metrics.json").exists()
    assert (log_dir / "log.log").exists()


def test_run_tccig_train_stage_monitors_tccig_topology_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_tccig_train_config(tmp_path)
    train_cfg = cast(ConfigDict, config["tccig_train"])
    train_cfg["epochs"] = 1
    train_cfg["subgraphs_per_epoch"] = 1
    train_cfg["subgraph_node_range"] = [3, 3]
    train_cfg["monitor_metric"] = "val_topology_loss"
    train_cfg["compute_clustering_mmd"] = False
    train_cfg["internal_validation_compute_clustering_mmd"] = False
    train_cfg["loss_weight_schedule"] = {"warmup_epochs": 0, "ramp_epochs": 0, "schedule": "linear"}
    train_cfg["losses"] = {
        "edge": 1.0,
        "teacher": 0.0,
        "budget": 0.0,
        "density": 0.25,
        "degree": 0.5,
        "clustering": 0.0,
    }
    monkeypatch.setattr(
        topology_train,
        "_evaluate_internal_validation_subgraphs",
        lambda **_: {
            "graph_sim": 1.0,
            "relative_density": 2.0,
            "deg_dist_mmd": 0.5,
            "cc_mmd": 0.0,
        },
    )
    model = build_model(config)
    dataloaders = build_dataloaders(config=config)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"tccig_train": "tccig_monitor_case"},
        )
        run_tccig_train_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
        )
    finally:
        os.chdir(previous_cwd)

    csv_path = (
        tmp_path / "logs" / "tccig" / "tccig_train" / "tccig_monitor_case" / "tccig_train_step.csv"
    )
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(DictReader(handle))

    assert float(rows[0]["Val Topology Loss"]) == pytest.approx(0.5)


def test_tccig_student_trainer_runs_online_teacher_for_padding_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_tccig_train_config(tmp_path)
    train_cfg_raw = cast(ConfigDict, config["tccig_train"])
    train_cfg_raw["subgraphs_per_epoch"] = 3
    train_cfg_raw["subgraph_node_range"] = [3, 3]
    train_cfg_raw["compute_clustering_mmd"] = False
    train_cfg_raw["teacher"] = {"enabled": True}
    train_cfg_raw["losses"] = {
        "edge": 1.0,
        "teacher": 0.1,
        "budget": 0.0,
        "density": 0.0,
        "degree": 0.0,
        "clustering": 0.0,
    }
    graph = nx.Graph()
    graph.add_nodes_from(["P1", "P2", "P3", "P4"])
    graph.add_edges_from([("P1", "P2"), ("P2", "P3"), ("P3", "P4")])
    epoch_plan = EdgeCoverEpochPlan(
        subgraphs=(("P1", "P2", "P3"), ("P2", "P3", "P4"), ("P1", "P3", "P4")),
        assigned_positive_edges=(
            frozenset({("P1", "P2")}),
            frozenset({("P2", "P3")}),
            frozenset({("P3", "P4")}),
        ),
        assigned_negative_edges=(
            frozenset({("P1", "P3")}),
            frozenset({("P2", "P4")}),
            frozenset({("P1", "P4")}),
        ),
        total_positive_edges=3,
        covered_positive_edges=3,
        positive_edge_coverage_ratio=1.0,
        mean_positive_edge_reuse=1.0,
    )
    monkeypatch.setattr(tccig_trainer_module, "sample_edge_cover_subgraphs", lambda **_: epoch_plan)

    class _TeacherStep:
        def __init__(self, loss: torch.Tensor) -> None:
            self.loss = loss

    class _CountingTeacher(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.training_steps = 0

        def training_step(self, **_: object) -> _TeacherStep:
            self.training_steps += 1
            return _TeacherStep(self.scale.sum() * 0.1)

        def score_pairs(
            self,
            *,
            node_features: torch.Tensor,
            visible_positive_edges: torch.Tensor,
            candidate_edges: torch.Tensor,
        ) -> torch.Tensor:
            del visible_positive_edges
            return node_features.new_zeros((candidate_edges.size(1),))

    cache_dir = Path(str(cast(ConfigDict, config["data_config"])["embeddings"]["cache_dir"]))  # type: ignore[index]
    embedding_index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    embedding_repository = EmbeddingRepository(
        cache_dir=cache_dir,
        embedding_index=cast(dict[str, str], embedding_index),
        input_dim=4,
        max_sequence_length=8,
    )
    model = build_model(config)
    distributed_context = DistributedContext(
        ddp_enabled=True,
        is_distributed=True,
        rank=1,
        world_size=2,
    )
    accelerator = NoOpAccelerator(distributed=distributed_context)
    teacher = _CountingTeacher()
    trainer = TCCIGStudentTrainer(
        train_cfg=parse_tccig_train_config(config),
        raw_config=config,
        model=model,
        graph=graph,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        accelerator=accelerator,
        embedding_repository=embedding_repository,
        negative_lookup=ExplicitNegativePairLookup(frozenset(), {}),
        distributed_context=distributed_context,
        teacher=OnlineTCCIGTeacher(
            teacher=cast(MGAETeacher, teacher),
            optimizer=torch.optim.SGD(teacher.parameters(), lr=1e-3),
            mask_ratio=0.7,
            negative_ratio=1,
        ),
        logger=None,
    )

    trainer.fit_epoch(epoch_index=0, epoch_seed=123)

    assert teacher.training_steps == 2
    assert accelerator.backward_calls == 4
    assert accelerator.autocast_calls == 0


def test_topology_finetune_rejects_tccig_graph_forward_model(tmp_path: Path) -> None:
    config = _build_finetune_config(tmp_path)
    config["model_config"] = {
        "model": "tccig",
        "input_dim": 4,
        "d_model": 8,
        "dropout": 0.0,
        "num_modules": 3,
        "lowrank_dim": 2,
        "candidate_proposer": {"type": "all_pairs"},
        "pair_mlp": {"hidden_dims": [8]},
    }
    cast(ConfigDict, config["run_config"])["load_checkpoint_path"] = None
    model = build_model(config)
    dataloaders = build_dataloaders(config=config)
    runtime = build_stage_runtime(
        config,
        stage_run_ids={"topology_finetune": "tccig_rejected"},
    )

    with pytest.raises(ValueError, match="tccig_train"):
        topology_finetune_stage.run_topology_finetuning_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=None,
        )


def test_tccig_config_uses_top_level_train_namespace() -> None:
    config = load_config("configs/tccig/tccig.yaml")
    run_cfg = cast(Mapping[str, object], config["run_config"])
    train_cfg = parse_tccig_train_config(config)

    assert run_cfg["stages"] == ["tccig_train", "evaluate", "topology_evaluate"]
    assert "tccig_train" in config
    assert "topology_finetune" not in config
    assert isinstance(train_cfg.optimizer, OptimizerConfig)
