"""Integration tests for the graph-topology fine-tuning stage."""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import networkx as nx
import pytest
import src.run.stage_topology_finetune as topology_finetune_stage
import torch
from src.embed import EmbeddingCacheManifest
from src.run.stage_topology_finetune import (
    _forward_model,
    _load_supervision_graphs,
    _resolve_sampling_node_bounds,
    run_topology_finetuning_stage,
)
from src.run.stage_train import build_model
from src.utils.config import ConfigDict
from src.utils.data_io import build_dataloaders
from src.utils.distributed import DistributedContext
from torch.utils.data import DataLoader


def _write_split(path: Path, rows: list[tuple[str, str, int]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a}\t{protein_b}\t{label}\n")


def _write_embedding_cache(
    cache_dir: Path,
    embeddings: dict[str, torch.Tensor],
    *,
    input_dim: int,
    max_sequence_length: int,
) -> None:
    index: dict[str, str] = {}
    for protein_id, tensor in embeddings.items():
        relative_path = f"embeddings/{protein_id}.pt"
        output_path = cache_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, output_path)
        index[protein_id] = relative_path
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (cache_dir / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "esm3",
                "model_name": "esm3_sm_open_v1",
                "input_dim": input_dim,
                "max_sequence_length": max_sequence_length,
                "format": "torch_pt_per_protein",
            }
        ),
        encoding="utf-8",
    )


def _build_finetune_config(tmp_path: Path) -> ConfigDict:
    benchmark_root = tmp_path / "benchmark"
    processed_dir = benchmark_root / "human" / "BFS"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "human_train_ppi.txt"
    valid_path = processed_dir / "human_val_ppi.txt"
    test_path = processed_dir / "human_test_ppi.txt"
    _write_split(
        train_path,
        [
            ("P1", "P2", 1),
            ("P1", "P3", 0),
            ("P2", "P3", 1),
            ("P2", "P4", 0),
        ],
    )
    _write_split(valid_path, [("P1", "P2", 1), ("P1", "P4", 0), ("P3", "P4", 1)])
    _write_split(test_path, [("P1", "P2", 1), ("P2", "P4", 0)])

    train_graph = nx.Graph()
    train_graph.add_nodes_from(["P1", "P2", "P3", "P4", "P5"])
    train_graph.add_edges_from([("P1", "P2"), ("P2", "P3"), ("P3", "P4")])
    with (processed_dir / "human_train_graph.pkl").open("wb") as handle:
        pickle.dump(train_graph, handle)
    with (processed_dir / "human_BFS_split.pkl").open("wb") as handle:
        pickle.dump({"train": {"P1", "P2", "P3", "P4", "P5"}, "test": {"PX"}}, handle)

    cache_dir = tmp_path / "cache"
    _write_embedding_cache(
        cache_dir=cache_dir,
        embeddings={
            "P1": torch.ones((2, 4), dtype=torch.float32),
            "P2": torch.full((2, 4), 2.0, dtype=torch.float32),
            "P3": torch.full((2, 4), 3.0, dtype=torch.float32),
            "P4": torch.full((2, 4), 4.0, dtype=torch.float32),
            "P5": torch.full((2, 4), 5.0, dtype=torch.float32),
        },
        input_dim=4,
        max_sequence_length=8,
    )

    return {
        "run_config": {
            "stages": ["topology_finetune"],
            "seed": 11,
            "train_run_id": "train_case",
            "topology_finetune_run_id": "topology_ft_case",
            "adapt_run_id": None,
            "eval_run_id": None,
            "topology_eval_run_id": None,
            "load_checkpoint_path": str(tmp_path / "input_checkpoint.pth"),
            "save_best_only": True,
        },
        "device_config": {
            "device": "cpu",
            "ddp_enabled": False,
            "use_mixed_precision": False,
        },
        "data_config": {
            "benchmark": {
                "name": "PRING",
                "root_dir": str(benchmark_root),
                "processed_dir": str(processed_dir),
                "species": "human",
                "split_strategy": "BFS",
            },
            "embeddings": {
                "source": "esm3",
                "cache_dir": str(cache_dir),
                "model_name": "esm3_sm_open_v1",
                "device": "cpu",
            },
            "max_sequence_length": 8,
            "dataloader": {
                "train_dataset": str(train_path),
                "valid_dataset": str(valid_path),
                "test_dataset": str(test_path),
                "num_workers": 0,
                "pin_memory": False,
                "drop_last": False,
                "sampling": {"strategy": "none"},
            },
        },
        "model_config": {
            "model": "v3",
            "input_dim": 4,
            "d_model": 4,
            "encoder_layers": 1,
            "cross_attn_layers": 1,
            "n_heads": 2,
            "mlp_head": {
                "hidden_dims": [4],
                "dropout": 0.0,
                "activation": "gelu",
                "norm": "layernorm",
            },
            "regularization": {
                "dropout": 0.0,
                "token_dropout": 0.0,
                "cross_attention_dropout": 0.0,
                "stochastic_depth": 0.0,
            },
        },
        "training_config": {
            "batch_size": 2,
            "epochs": 1,
            "monitor_metric": "auprc",
            "logging": {"validation_metrics": ["auprc", "auroc"]},
            "optimizer": {"type": "adamw", "lr": 1e-3, "weight_decay": 0.0},
            "scheduler": {"type": "none"},
            "loss": {"type": "bce_with_logits", "pos_weight": 1.0, "label_smoothing": 0.0},
            "strategy": {"type": "none"},
            "domain_adaptation": {"enabled": False, "method": "none", "target_split": "test"},
        },
        "topology_finetune": {
            "epochs": 1,
            "subgraphs_per_epoch": 3,
            "validation_subgraphs": 2,
            "min_nodes": 3,
            "max_nodes": 4,
            "strategy": "mixed",
            "pair_batch_size": 2,
            "decision_threshold": 0.5,
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
            "losses": {
                "alpha": 0.5,
                "beta": 1.0,
                "gamma": 0.3,
                "delta": 0.3,
                "histogram_sigma": 1.0,
            },
        },
}


def test_load_supervision_graphs_excludes_val_edges_and_keeps_all_train_nodes(
    tmp_path: Path,
) -> None:
    config = _build_finetune_config(tmp_path)

    train_graph, internal_val_graph = _load_supervision_graphs(config=config)

    assert set(train_graph.nodes) == {"P1", "P2", "P3", "P4", "P5"}
    assert {tuple(sorted(edge)) for edge in train_graph.edges} == {
        ("P1", "P2"),
        ("P2", "P3"),
    }
    assert set(internal_val_graph.nodes) == {"P1", "P2", "P3", "P4", "P5"}
    assert {tuple(sorted(edge)) for edge in internal_val_graph.edges} == {
        ("P1", "P2"),
        ("P3", "P4"),
    }


def test_resolve_sampling_node_bounds_caps_subgraphs_to_20_nodes() -> None:
    min_nodes, max_nodes = _resolve_sampling_node_bounds(
        {
            "min_nodes": 64,
            "max_nodes": 200,
        }
    )

    assert min_nodes == 20
    assert max_nodes == 20


def test_forward_model_uses_activation_checkpointing_during_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(4, 1)
    model.train()
    observed_checkpoint_calls: list[bool] = []

    def _fake_checkpoint(
        function: object,
        *args: object,
        use_reentrant: bool,
    ) -> torch.Tensor:
        del use_reentrant
        observed_checkpoint_calls.append(True)
        return function(*args)  # type: ignore[misc]

    class _ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = model

        def forward(
            self,
            emb_a: torch.Tensor,
            emb_b: torch.Tensor,
            len_a: torch.Tensor,
            len_b: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del len_a, len_b
            return {"logits": self.projection((emb_a + emb_b).mean(dim=1))}

    monkeypatch.setattr(topology_finetune_stage, "checkpoint", _fake_checkpoint)

    output = _forward_model(
        model=_ToyModel().train(),
        batch={
            "emb_a": torch.ones((2, 3, 4), dtype=torch.float32),
            "emb_b": torch.ones((2, 3, 4), dtype=torch.float32),
            "len_a": torch.tensor([3, 3], dtype=torch.long),
            "len_b": torch.tensor([3, 3], dtype=torch.long),
            "label": torch.tensor([1.0, 0.0], dtype=torch.float32),
        },
    )
    output["logits"].sum().backward()

    assert observed_checkpoint_calls == [True]


def test_run_topology_finetuning_stage_warm_starts_and_writes_artifacts(tmp_path: Path) -> None:
    config = _build_finetune_config(tmp_path)
    model = build_model(config)
    checkpoint_path = Path(str(config["run_config"]["load_checkpoint_path"]))  # type: ignore[index]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    initial_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    torch.save(initial_state, checkpoint_path)

    dataloaders = build_dataloaders(config=config)
    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        best_checkpoint = run_topology_finetuning_stage(
            config=config,
            model=model,
            device=torch.device("cpu"),
            dataloaders=cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            run_id="topology_ft_case",
            checkpoint_path=checkpoint_path,
            distributed_context=DistributedContext(ddp_enabled=False, is_distributed=False),
        )
    finally:
        os.chdir(previous_cwd)

    log_dir = tmp_path / "logs" / "v3" / "topology_finetune" / "topology_ft_case"
    assert best_checkpoint == Path(
        "models/v3/topology_finetune/topology_ft_case/best_model.pth"
    )
    best_checkpoint_path = tmp_path / best_checkpoint
    assert best_checkpoint_path.exists()
    assert (log_dir / "topology_finetune_step.csv").exists()
    assert (log_dir / "topology_finetune_metrics.json").exists()
    assert (log_dir / "log.log").exists()

    updated_state = torch.load(best_checkpoint_path, map_location="cpu")
    assert any(
        not torch.allclose(initial_state[name], updated_state[name])
        for name in initial_state
    )


def test_run_topology_finetuning_stage_allows_embedding_generation_on_non_main_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    config["topology_finetune"]["epochs"] = 0  # type: ignore[index]
    model = build_model(config)
    checkpoint_path = Path(str(config["run_config"]["load_checkpoint_path"]))  # type: ignore[index]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    dataloaders = build_dataloaders(config=config)
    observed_allow_generation: list[bool] = []

    def _fake_ensure_embeddings_ready(
        config: ConfigDict,
        split_paths: Sequence[Path],
        input_dim: int,
        max_sequence_length: int,
        allow_generation: bool = True,
        extra_protein_ids: Sequence[str] | None = None,
    ) -> EmbeddingCacheManifest:
        del split_paths, input_dim, max_sequence_length, extra_protein_ids
        observed_allow_generation.append(allow_generation)
        return EmbeddingCacheManifest(
            cache_dir=Path(str(config["data_config"]["embeddings"]["cache_dir"])),  # type: ignore[index]
            index={
                protein_id: f"embeddings/{protein_id}.pt"
                for protein_id in ("P1", "P2", "P3", "P4", "P5")
            },
            required_ids=frozenset({"P1", "P2", "P3", "P4", "P5"}),
        )

    monkeypatch.setattr(
        topology_finetune_stage,
        "ensure_embeddings_ready",
        _fake_ensure_embeddings_ready,
    )
    monkeypatch.setattr(topology_finetune_stage.dist, "is_available", lambda: True)
    monkeypatch.setattr(topology_finetune_stage.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(topology_finetune_stage, "distributed_barrier", lambda _: None)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        run_topology_finetuning_stage(
            config=config,
            model=model,
            device=torch.device("cpu"),
            dataloaders=cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            run_id="topology_ft_case",
            checkpoint_path=checkpoint_path,
            distributed_context=DistributedContext(
                ddp_enabled=True,
                is_distributed=True,
                rank=1,
                local_rank=1,
                world_size=4,
            ),
        )
    finally:
        os.chdir(previous_cwd)

    assert observed_allow_generation == [True]
