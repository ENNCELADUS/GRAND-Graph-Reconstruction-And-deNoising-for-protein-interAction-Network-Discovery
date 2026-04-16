"""Integration tests for the graph-topology fine-tuning stage."""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Mapping, Sequence
from csv import DictReader
from pathlib import Path
from typing import cast

import networkx as nx
import pytest
import src.pipeline.stages.topology_finetune as topology_finetune_stage
import torch
from src.embed import EmbeddingCacheManifest
from src.evaluate import Evaluator
from src.pipeline.runtime import DistributedContext
from src.pipeline.stages.topology_finetune import (
    ValidationEpochResult,
    _build_internal_validation_node_sets,
    _evaluate_internal_validation_subgraphs,
    _forward_model,
    _load_supervision_graphs,
    _resolve_internal_validation_threshold,
    _resolve_monitor_mode,
    _resolve_monitor_value,
    _resolve_sampling_node_bounds,
    run_topology_finetuning_stage,
)
from src.pipeline.stages.train import build_model
from src.topology.finetune_data import (
    TOPOLOGY_EVAL_NODE_SIZES,
    TOPOLOGY_EVAL_SAMPLES_PER_SIZE,
    EdgeCoverEpochPlan,
    EmbeddingRepository,
    SubgraphPairChunk,
    build_internal_validation_plan,
)
from src.topology.metrics import evaluate_graph_samples
from src.topology.negative_sampling import prepare_topology_supervision_from_config
from src.utils.config import ConfigDict, load_config
from src.utils.data_io import build_dataloaders
from tests.runtime_helpers import NoOpAccelerator, build_stage_runtime
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
            ("P4", "P5", 0),
            ("P5", "P6", 0),
        ],
    )
    _write_split(
        valid_path,
        [("P1", "P2", 1), ("P1", "P4", 0), ("P3", "P4", 1), ("P3", "P6", 0)],
    )
    _write_split(
        test_path,
        [("P1", "P2", 1), ("P2", "P4", 0), ("P1", "P6", 0), ("P7", "P8", 0), ("P9", "P10", 0)],
    )
    (benchmark_root / "human_ppi.txt").write_text(
        "\n".join(
            [
                "P1\tP2",
                "P2\tP3",
                "P3\tP4",
                "P1\tP4",
                "",
            ]
        ),
        encoding="utf-8",
    )

    train_graph = nx.Graph()
    train_graph.add_nodes_from(["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"])
    train_graph.add_edges_from([("P1", "P2"), ("P2", "P3"), ("P3", "P4")])
    with (processed_dir / "human_train_graph.pkl").open("wb") as handle:
        pickle.dump(train_graph, handle)
    with (processed_dir / "human_BFS_split.pkl").open("wb") as handle:
        pickle.dump(
            {
                "train": {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"},
                "test": {"PX"},
            },
            handle,
        )

    cache_dir = tmp_path / "cache"
    _write_embedding_cache(
        cache_dir=cache_dir,
        embeddings={
            "P1": torch.ones((2, 4), dtype=torch.float32),
            "P2": torch.full((2, 4), 2.0, dtype=torch.float32),
            "P3": torch.full((2, 4), 3.0, dtype=torch.float32),
            "P4": torch.full((2, 4), 4.0, dtype=torch.float32),
            "P5": torch.full((2, 4), 5.0, dtype=torch.float32),
            "P6": torch.full((2, 4), 6.0, dtype=torch.float32),
            "P7": torch.full((2, 4), 7.0, dtype=torch.float32),
            "P8": torch.full((2, 4), 8.0, dtype=torch.float32),
            "P9": torch.full((2, 4), 9.0, dtype=torch.float32),
            "P10": torch.full((2, 4), 10.0, dtype=torch.float32),
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
                "min_nodes": 3,
                "max_nodes": 4,
                "strategy": "mixed",
                "bce_negative_ratio": 0,
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


class _RecordingAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.is_main_process = True
        self.use_distributed = False
        self.process_index = 0
        self.local_process_index = 0
        self.num_processes = 1
        self.mixed_precision = "no"
        self.prepare_calls = 0
        self.autocast_calls = 0
        self.backward_calls = 0
        self.reduce_calls = 0

    def prepare(self, *components: object) -> tuple[object, ...]:
        self.prepare_calls += 1
        return components

    def autocast(self) -> object:
        from contextlib import nullcontext

        self.autocast_calls += 1
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_calls += 1
        loss.backward()

    def reduce(self, value: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        del reduction
        self.reduce_calls += 1
        return value

    def wait_for_everyone(self) -> None:
        return None

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def save(self, obj: object, f: object, safe_serialization: bool = False) -> None:
        del safe_serialization
        torch.save(obj, f)


class _CountingOptimizer:
    def __init__(self) -> None:
        self.step_calls = 0
        self.zero_grad_calls = 0

    def zero_grad(self, set_to_none: bool = False) -> None:
        del set_to_none
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


def test_load_supervision_graphs_excludes_val_edges_and_keeps_all_train_nodes(
    tmp_path: Path,
) -> None:
    config = _build_finetune_config(tmp_path)

    train_graph, internal_val_graph = _load_supervision_graphs(config=config)

    assert set(train_graph.nodes) == {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"}
    assert {tuple(sorted(edge)) for edge in train_graph.edges} == {
        ("P1", "P2"),
        ("P2", "P3"),
    }
    assert set(internal_val_graph.nodes) == {
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
        "P10",
    }
    assert {tuple(sorted(edge)) for edge in internal_val_graph.edges} == {
        ("P1", "P2"),
        ("P3", "P4"),
    }


def test_resolve_sampling_node_bounds_uses_subgraph_node_range() -> None:
    min_nodes, max_nodes = _resolve_sampling_node_bounds(
        {
            "subgraph_node_range": [30, 60],
        }
    )

    assert min_nodes == 30
    assert max_nodes == 60


def test_resolve_sampling_node_bounds_rejects_legacy_min_max_config() -> None:
    with pytest.raises(ValueError, match="subgraph_node_range"):
        _resolve_sampling_node_bounds(
            {
                "min_nodes": 30,
                "max_nodes": 60,
            }
        )


def test_build_internal_validation_node_sets_matches_topology_eval_bucketing() -> None:
    graph = nx.path_graph([f"P{i}" for i in range(1, 221)])

    node_sets = _build_internal_validation_node_sets(
        finetune_cfg={"strategy": "mixed"},
        graph=graph,
        seed=11,
    )

    assert sorted(node_sets) == list(TOPOLOGY_EVAL_NODE_SIZES)
    assert all(
        len(node_sets[node_size]) == TOPOLOGY_EVAL_SAMPLES_PER_SIZE for node_size in node_sets
    )
    assert all(
        all(len(nodes) == node_size for nodes in node_sets[node_size]) for node_size in node_sets
    )


def test_build_internal_validation_node_sets_uses_graph_size_fallback_when_under_20() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["P3", "P1", "P2"])

    node_sets = _build_internal_validation_node_sets(
        finetune_cfg={"strategy": "mixed"},
        graph=graph,
        seed=11,
    )

    assert sorted(node_sets) == [3]
    assert len(node_sets[3]) == TOPOLOGY_EVAL_SAMPLES_PER_SIZE
    assert all(nodes == ("P1", "P2", "P3") for nodes in node_sets[3])


def test_resolve_internal_validation_threshold_uses_validation_selected_mode(
    tmp_path: Path,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["decision_threshold"] = {"mode": "best_f1_on_valid"}
    threshold, mode = _resolve_internal_validation_threshold(
        config=config,
        labels=torch.tensor([0, 1, 1, 0], dtype=torch.long),
        probabilities=torch.tensor([0.1, 0.9, 0.8, 0.2], dtype=torch.float32),
    )

    assert threshold == pytest.approx(0.8)
    assert mode == "best_f1_on_valid"


def test_resolve_monitor_mode_uses_min_for_val_loss() -> None:
    assert _resolve_monitor_mode("val_loss") == "min"
    assert _resolve_monitor_mode("val_auprc") == "max"


def test_resolve_monitor_value_reads_val_loss() -> None:
    monitor_value = _resolve_monitor_value(
        monitor_metric="val_loss",
        val_pair_stats={"val_loss": 0.42, "val_auprc": 0.91},
        internal_val_topology_stats={
            "graph_sim": 0.2,
            "relative_density": 1.1,
            "deg_dist_mmd": 0.3,
            "cc_mmd": 0.4,
        },
    )

    assert monitor_value == pytest.approx(0.42)


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


def test_predict_hard_subgraph_streams_pair_chunks_for_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ToyModel(torch.nn.Module):
        def forward(
            self,
            emb_a: torch.Tensor,
            emb_b: torch.Tensor,
            len_a: torch.Tensor,
            len_b: torch.Tensor,
            **_: object,
        ) -> dict[str, torch.Tensor]:
            del emb_b, len_a, len_b
            if emb_a.size(0) == 2:
                logits = torch.tensor([10.0, -10.0], dtype=torch.float32)
            else:
                logits = torch.tensor([10.0], dtype=torch.float32)
            return {"logits": logits}

    node_tuple = ("P1", "P2", "P3")

    def _chunk_stream(**_: object) -> Sequence[SubgraphPairChunk]:
        return [
            SubgraphPairChunk(
                nodes=node_tuple,
                emb_a=torch.ones((2, 1, 4), dtype=torch.float32),
                emb_b=torch.ones((2, 1, 4), dtype=torch.float32),
                len_a=torch.tensor([1, 1], dtype=torch.long),
                len_b=torch.tensor([1, 1], dtype=torch.long),
                label=torch.tensor([1.0, 0.0], dtype=torch.float32),
                pair_index_a=torch.tensor([0, 0], dtype=torch.long),
                pair_index_b=torch.tensor([1, 2], dtype=torch.long),
            ),
            SubgraphPairChunk(
                nodes=node_tuple,
                emb_a=torch.ones((1, 1, 4), dtype=torch.float32),
                emb_b=torch.ones((1, 1, 4), dtype=torch.float32),
                len_a=torch.tensor([1], dtype=torch.long),
                len_b=torch.tensor([1], dtype=torch.long),
                label=torch.tensor([1.0], dtype=torch.float32),
                pair_index_a=torch.tensor([1], dtype=torch.long),
                pair_index_b=torch.tensor([2], dtype=torch.long),
            ),
        ]

    monkeypatch.setattr(topology_finetune_stage, "iter_subgraph_pair_chunks", _chunk_stream)
    monkeypatch.setattr(
        topology_finetune_stage,
        "_concat_logits_and_pairs",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("streaming inference should not concatenate all pair tensors")
        ),
    )

    pred_subgraph = topology_finetune_stage._predict_hard_subgraph(
        model=_ToyModel().eval(),
        graph=nx.Graph(),
        nodes=node_tuple,
        cache_dir=Path("."),
        embedding_index={protein_id: f"{protein_id}.pt" for protein_id in node_tuple},
        input_dim=4,
        max_sequence_length=8,
        pair_batch_size=2,
        threshold=0.5,
        device=torch.device("cpu"),
    )

    assert sorted(pred_subgraph.edges()) == [("P1", "P2"), ("P2", "P3")]


def test_evaluate_internal_validation_subgraphs_matches_per_subgraph_baseline(
    tmp_path: Path,
) -> None:
    graph = nx.Graph()
    graph.add_edges_from([("P1", "P2"), ("P2", "P3"), ("P2", "P4")])
    cache_dir = tmp_path / "cache"
    _write_embedding_cache(
        cache_dir=cache_dir,
        embeddings={
            "P1": torch.full((2, 4), 1.0, dtype=torch.float32),
            "P2": torch.full((2, 4), 2.0, dtype=torch.float32),
            "P3": torch.full((2, 4), 3.0, dtype=torch.float32),
            "P4": torch.full((2, 4), 4.0, dtype=torch.float32),
        },
        input_dim=4,
        max_sequence_length=8,
    )
    embedding_index = {
        protein_id: f"embeddings/{protein_id}.pt" for protein_id in ("P1", "P2", "P3", "P4")
    }
    validation_plan = build_internal_validation_plan(
        graph=graph,
        sampled_subgraphs={3: [("P1", "P2", "P3"), ("P2", "P3", "P4")]},
    )
    embedding_repository = EmbeddingRepository(
        cache_dir=cache_dir,
        embedding_index=embedding_index,
        input_dim=4,
        max_sequence_length=8,
        max_cache_bytes=1_024,
    )

    class _ToyModel(torch.nn.Module):
        def forward(
            self,
            emb_a: torch.Tensor,
            emb_b: torch.Tensor,
            len_a: torch.Tensor,
            len_b: torch.Tensor,
            **_: object,
        ) -> dict[str, torch.Tensor]:
            del len_a, len_b
            scores = emb_a[:, 0, 0] + emb_b[:, 0, 0]
            return {"logits": (scores - 5.0) * 10.0}

    model = _ToyModel().eval()
    batched_summary = _evaluate_internal_validation_subgraphs(
        model=model,
        validation_plan=validation_plan,
        embedding_repository=embedding_repository,
        inference_batch_size=4,
        threshold=0.5,
        device=torch.device("cpu"),
        accelerator=_RecordingAccelerator(),
    )

    expected_pred_graphs = {
        3: [
            topology_finetune_stage._predict_hard_subgraph(
                model=model,
                graph=graph,
                nodes=nodes,
                cache_dir=cache_dir,
                embedding_index=embedding_index,
                input_dim=4,
                max_sequence_length=8,
                pair_batch_size=2,
                threshold=0.5,
                device=torch.device("cpu"),
                embedding_repository=embedding_repository,
            )
            for nodes in (("P1", "P2", "P3"), ("P2", "P3", "P4"))
        ]
    }
    expected_target_graphs = {
        3: [graph.subgraph(nodes).copy() for nodes in (("P1", "P2", "P3"), ("P2", "P3", "P4"))]
    }
    expected_summary = evaluate_graph_samples(
        pred_graphs_by_size=expected_pred_graphs,
        gt_graphs_by_size=expected_target_graphs,
    )["summary"]

    assert batched_summary == pytest.approx(expected_summary)


def test_run_topology_finetuning_stage_reuses_single_validation_pass_for_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["decision_threshold"] = {"mode": "best_f1_on_valid"}
    topology_cfg["epochs"] = 1
    model = build_model(config)
    checkpoint_path = Path(str(config["run_config"]["load_checkpoint_path"]))  # type: ignore[index]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    dataloaders = build_dataloaders(config=config)
    observed_collect_calls: list[int] = []
    original_collect = Evaluator.collect_probabilities_and_labels

    def _record_collect(
        self: Evaluator,
        model: torch.nn.Module,
        data_loader: DataLoader[Mapping[str, object]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        observed_collect_calls.append(1)
        return original_collect(self, model, data_loader, device)

    def _unexpected_threshold_call(*args: object, **kwargs: object) -> float:
        raise AssertionError("threshold selection should reuse collected validation outputs")

    monkeypatch.setattr(Evaluator, "collect_probabilities_and_labels", _record_collect)
    monkeypatch.setattr(Evaluator, "select_best_f1_threshold", _unexpected_threshold_call)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_finetune": "topology_ft_case"},
        )
        run_topology_finetuning_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        os.chdir(previous_cwd)

    assert observed_collect_calls == [1]


def test_run_topology_finetuning_stage_uses_edge_cover_sampling_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    model = build_model(config)
    checkpoint_path = Path(str(config["run_config"]["load_checkpoint_path"]))  # type: ignore[index]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    dataloaders = build_dataloaders(config=config)
    observed_training_calls: list[dict[str, object]] = []

    def _fake_sample_edge_cover_subgraphs(**kwargs: object) -> EdgeCoverEpochPlan:
        observed_training_calls.append(dict(kwargs))
        return EdgeCoverEpochPlan(
            subgraphs=(("P1", "P2", "P3"),),
            assigned_positive_edges=(frozenset({("P1", "P2"), ("P2", "P3")}),),
            total_positive_edges=2,
            covered_positive_edges=2,
            positive_edge_coverage_ratio=1.0,
            mean_positive_edge_reuse=1.0,
        )

    monkeypatch.setattr(
        topology_finetune_stage,
        "sample_edge_cover_subgraphs",
        _fake_sample_edge_cover_subgraphs,
    )

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_finetune": "topology_ft_case"},
        )
        run_topology_finetuning_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        os.chdir(previous_cwd)

    assert len(observed_training_calls) == 1
    assert observed_training_calls[0]["num_subgraphs"] == 0
    assert "overlap_penalty" not in observed_training_calls[0]
    assert "edge_chunk_size" in observed_training_calls[0]


def test_run_topology_finetuning_stage_warm_starts_and_writes_artifacts(tmp_path: Path) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["subgraphs_per_epoch"] = 6
    topology_cfg["validation_subgraphs"] = 7
    model = build_model(config)
    checkpoint_path = Path(str(config["run_config"]["load_checkpoint_path"]))  # type: ignore[index]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    initial_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    torch.save(initial_state, checkpoint_path)

    dataloaders = build_dataloaders(config=config)
    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_finetune": "topology_ft_case"},
        )
        best_checkpoint = run_topology_finetuning_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        os.chdir(previous_cwd)

    log_dir = tmp_path / "logs" / "v3" / "topology_finetune" / "topology_ft_case"
    assert best_checkpoint == Path("models/v3/topology_finetune/topology_ft_case/best_model.pth")
    best_checkpoint_path = tmp_path / best_checkpoint
    assert best_checkpoint_path.exists()
    assert (log_dir / "topology_finetune_step.csv").exists()
    assert (log_dir / "topology_finetune_metrics.json").exists()
    assert (log_dir / "log.log").exists()
    with (log_dir / "topology_finetune_step.csv").open("r", encoding="utf-8", newline="") as handle:
        header = DictReader(handle).fieldnames
    assert header is not None
    assert "Planned Subgraphs" in header
    assert "Positive Edge Coverage Ratio" in header
    assert "Mean Positive Edge Reuse" in header
    assert "edge_cover_sampling_s" in header
    assert "train_forward_backward_s" in header
    assert "val_pair_pass_s" in header
    assert "val_threshold_s" in header
    assert "internal_val_topology_s" in header
    assert "peak_gpu_mem_mb" in header
    log_text = (log_dir / "log.log").read_text(encoding="utf-8")
    assert "Epoch Progress" in log_text
    assert "Legacy Validation Subgraphs Ignored" in log_text

    updated_state = torch.load(best_checkpoint_path, map_location="cpu")
    assert any(
        not torch.allclose(initial_state[name], updated_state[name]) for name in initial_state
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
                for protein_id in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10")
            },
            required_ids=frozenset({"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"}),
        )

    monkeypatch.setattr(
        topology_finetune_stage,
        "ensure_embeddings_ready",
        _fake_ensure_embeddings_ready,
    )
    monkeypatch.setattr(topology_finetune_stage.dist, "is_available", lambda: True)
    monkeypatch.setattr(topology_finetune_stage.dist, "is_initialized", lambda: True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        distributed_context = DistributedContext(
            ddp_enabled=True,
            is_distributed=True,
            rank=1,
            local_rank=1,
            world_size=4,
        )
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_finetune": "topology_ft_case"},
            distributed=distributed_context,
        )
        run_topology_finetuning_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        os.chdir(previous_cwd)

    assert observed_allow_generation == [True]


def test_run_topology_finetuning_stage_uses_shared_epoch_sampling_seed_under_ddp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["init_mode"] = "scratch"
    topology_cfg["epochs"] = 1

    dataloaders = build_dataloaders(config=config)
    observed_rank_seeds: list[tuple[int, int]] = []
    active_rank = 0

    def _fake_sample_edge_cover_subgraphs(**kwargs: object) -> EdgeCoverEpochPlan:
        observed_rank_seeds.append((active_rank, int(kwargs["seed"])))
        return EdgeCoverEpochPlan(
            subgraphs=(),
            assigned_positive_edges=(),
            total_positive_edges=0,
            covered_positive_edges=0,
            positive_edge_coverage_ratio=1.0,
            mean_positive_edge_reuse=0.0,
        )

    def _fake_evaluate_validation_epoch(**_: object) -> ValidationEpochResult:
        return ValidationEpochResult(
            decision_threshold=0.5,
            val_pair_stats={"val_loss": 0.0, "val_auprc": 0.0},
            internal_val_topology_stats={
                "graph_sim": 0.0,
                "relative_density": 0.0,
                "deg_dist_mmd": 0.0,
                "cc_mmd": 0.0,
            },
            val_pair_pass_seconds=0.0,
            threshold_resolution_seconds=0.0,
            internal_validation_seconds=0.0,
        )

    monkeypatch.setattr(
        topology_finetune_stage,
        "sample_edge_cover_subgraphs",
        _fake_sample_edge_cover_subgraphs,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "_evaluate_validation_epoch",
        _fake_evaluate_validation_epoch,
    )
    monkeypatch.setattr(topology_finetune_stage.dist, "broadcast", lambda tensor, src: None)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        for active_rank in (0, 1):
            runtime = build_stage_runtime(
                config,
                stage_run_ids={"topology_finetune": f"topology_ft_rank_{active_rank}"},
                distributed=DistributedContext(
                    ddp_enabled=True,
                    is_distributed=True,
                    rank=active_rank,
                    local_rank=active_rank,
                    world_size=2,
                ),
            )
            run_topology_finetuning_stage(
                runtime,
                build_model(config),
                cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
                checkpoint_path=None,
            )
    finally:
        os.chdir(previous_cwd)

    assert observed_rank_seeds == [(0, 11), (1, 11)]


def test_run_topology_finetuning_stage_shards_subgraphs_across_ranks_under_ddp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["init_mode"] = "scratch"
    topology_cfg["epochs"] = 1

    dataloaders = build_dataloaders(config=config)
    observed_rank_nodes: list[tuple[int, tuple[str, ...]]] = []
    active_rank = 0

    def _fake_sample_edge_cover_subgraphs(**_: object) -> EdgeCoverEpochPlan:
        return EdgeCoverEpochPlan(
            subgraphs=(
                ("P1", "P2", "P3"),
                ("P2", "P3", "P4"),
                ("P3", "P4", "P5"),
                ("P4", "P5", "P6"),
            ),
            assigned_positive_edges=(
                frozenset({("P1", "P2")}),
                frozenset({("P2", "P3")}),
                frozenset({("P3", "P4")}),
                frozenset({("P4", "P5")}),
            ),
            total_positive_edges=4,
            covered_positive_edges=4,
            positive_edge_coverage_ratio=1.0,
            mean_positive_edge_reuse=1.0,
        )

    def _fake_concat_logits_and_pairs(**kwargs: object) -> tuple[torch.Tensor, ...]:
        nodes = tuple(cast(tuple[str, ...], kwargs["nodes"]))
        observed_rank_nodes.append((active_rank, nodes))
        pair_index_b = 1 if len(nodes) > 1 else 0
        return (
            torch.zeros(1, dtype=torch.float32, requires_grad=True),
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([pair_index_b], dtype=torch.long),
        )

    def _fake_evaluate_validation_epoch(**_: object) -> ValidationEpochResult:
        return ValidationEpochResult(
            decision_threshold=0.5,
            val_pair_stats={"val_loss": 0.0, "val_auprc": 0.0},
            internal_val_topology_stats={
                "graph_sim": 0.0,
                "relative_density": 0.0,
                "deg_dist_mmd": 0.0,
                "cc_mmd": 0.0,
            },
            val_pair_pass_seconds=0.0,
            threshold_resolution_seconds=0.0,
            internal_validation_seconds=0.0,
        )

    monkeypatch.setattr(
        topology_finetune_stage,
        "sample_edge_cover_subgraphs",
        _fake_sample_edge_cover_subgraphs,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "_concat_logits_and_pairs",
        _fake_concat_logits_and_pairs,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "_evaluate_validation_epoch",
        _fake_evaluate_validation_epoch,
    )
    monkeypatch.setattr(topology_finetune_stage.dist, "broadcast", lambda tensor, src: None)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        for active_rank in (0, 1):
            runtime = build_stage_runtime(
                config,
                stage_run_ids={"topology_finetune": f"topology_ft_rank_{active_rank}"},
                distributed=DistributedContext(
                    ddp_enabled=True,
                    is_distributed=True,
                    rank=active_rank,
                    local_rank=active_rank,
                    world_size=2,
                ),
            )
            run_topology_finetuning_stage(
                runtime,
                build_model(config),
                cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
                checkpoint_path=None,
            )
    finally:
        os.chdir(previous_cwd)

    assert observed_rank_nodes == [
        (0, ("P1", "P2", "P3")),
        (0, ("P3", "P4", "P5")),
        (1, ("P2", "P3", "P4")),
        (1, ("P4", "P5", "P6")),
    ]


def test_run_topology_finetuning_stage_runs_validation_only_on_main_rank_under_ddp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["init_mode"] = "scratch"
    topology_cfg["epochs"] = 1

    dataloaders = build_dataloaders(config=config)
    observed_pair_validation_ranks: list[int] = []
    observed_internal_validation_ranks: list[int] = []
    active_rank = 0

    def _fake_fit_epoch(**_: object) -> dict[str, float]:
        return {
            "bce": 0.0,
            "graph_similarity": 0.0,
            "relative_density": 0.0,
            "degree_mmd": 0.0,
            "clustering_mmd": 0.0,
            "total": 0.0,
            "planned_subgraphs": 1.0,
            "covered_positive_edges": 1.0,
            "total_positive_edges": 1.0,
            "positive_edge_coverage_ratio": 1.0,
            "mean_positive_edge_reuse": 1.0,
            "edge_cover_sampling_s": 0.0,
            "train_forward_backward_s": 0.0,
        }

    def _record_collect(
        self: Evaluator,
        *,
        model: torch.nn.Module,
        data_loader: DataLoader[Mapping[str, object]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        del self, model, data_loader, device
        observed_pair_validation_ranks.append(active_rank)
        return (
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([0.1, 0.9], dtype=torch.float32),
            0.0,
        )

    def _record_internal_validation(**_: object) -> dict[str, float]:
        observed_internal_validation_ranks.append(active_rank)
        return {
            "graph_sim": 0.0,
            "relative_density": 0.0,
            "deg_dist_mmd": 0.0,
            "cc_mmd": 0.0,
        }

    monkeypatch.setattr(topology_finetune_stage, "_fit_epoch", _fake_fit_epoch)
    monkeypatch.setattr(Evaluator, "collect_probabilities_and_labels", _record_collect)
    monkeypatch.setattr(
        topology_finetune_stage,
        "_evaluate_internal_validation_subgraphs",
        _record_internal_validation,
    )
    monkeypatch.setattr(topology_finetune_stage.dist, "broadcast", lambda tensor, src: None)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        for active_rank in (0, 1):
            runtime = build_stage_runtime(
                config,
                stage_run_ids={"topology_finetune": f"topology_ft_rank_{active_rank}"},
                distributed=DistributedContext(
                    ddp_enabled=True,
                    is_distributed=True,
                    rank=active_rank,
                    local_rank=active_rank,
                    world_size=2,
                ),
            )
            run_topology_finetuning_stage(
                runtime,
                build_model(config),
                cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
                checkpoint_path=None,
            )
    finally:
        os.chdir(previous_cwd)

    assert observed_pair_validation_ranks == [0]
    assert observed_internal_validation_ranks == [0]


def test_fit_epoch_accumulates_gradients_before_optimizer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["gradient_accumulation_steps"] = 2

    graph = nx.path_graph(["P1", "P2", "P3", "P4"])
    optimizer = _CountingOptimizer()
    accelerator = NoOpAccelerator()

    def _fake_sample_edge_cover_subgraphs(**_: object) -> EdgeCoverEpochPlan:
        return EdgeCoverEpochPlan(
            subgraphs=(("P1", "P2"), ("P2", "P3"), ("P3", "P4")),
            assigned_positive_edges=(
                frozenset({("P1", "P2")}),
                frozenset({("P2", "P3")}),
                frozenset({("P3", "P4")}),
            ),
            total_positive_edges=3,
            covered_positive_edges=3,
            positive_edge_coverage_ratio=1.0,
            mean_positive_edge_reuse=1.0,
        )

    def _fake_concat_logits_and_pairs(**_: object) -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(1, dtype=torch.float32, requires_grad=True),
            torch.ones(1, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32),
            torch.ones(1, dtype=torch.float32),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([1], dtype=torch.long),
        )

    monkeypatch.setattr(
        topology_finetune_stage,
        "sample_edge_cover_subgraphs",
        _fake_sample_edge_cover_subgraphs,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "_concat_logits_and_pairs",
        _fake_concat_logits_and_pairs,
    )

    topology_finetune_stage._fit_epoch(
        config=config,
        model=torch.nn.Linear(1, 1),
        device=torch.device("cpu"),
        graph=graph,
        cache_dir=tmp_path,
        embedding_index={},
        optimizer=cast(torch.optim.Optimizer, optimizer),
        epoch_index=0,
        epoch_seed=11,
        input_dim=4,
        max_sequence_length=8,
        loss_weights=topology_finetune_stage.TopologyLossWeights(
            alpha=0.0,
            beta=0.0,
            gamma=0.0,
            delta=0.0,
        ),
        pair_batch_size=2,
        use_amp=False,
        accelerator=accelerator,
        embedding_repository=cast(EmbeddingRepository, object()),
        negative_lookup=None,
        distributed_context=DistributedContext(ddp_enabled=False, is_distributed=False),
    )

    assert accelerator.backward_calls == 3
    assert optimizer.step_calls == 2
    assert optimizer.zero_grad_calls == 2


def test_run_topology_finetuning_stage_uses_accelerator_runtime(
    tmp_path: Path,
) -> None:
    config = _build_finetune_config(tmp_path)
    model = build_model(config)
    checkpoint_path = Path(str(config["run_config"]["load_checkpoint_path"]))  # type: ignore[index]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    dataloaders = build_dataloaders(config=config)
    accelerator = _RecordingAccelerator()

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_finetune": "topology_ft_case"},
            accelerator=accelerator,
        )
        run_topology_finetuning_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        os.chdir(previous_cwd)

    assert accelerator.prepare_calls >= 1
    assert accelerator.autocast_calls >= 1
    assert accelerator.backward_calls >= 1
    assert accelerator.reduce_calls >= 1


def test_run_topology_finetuning_stage_supports_scratch_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["epochs"] = 0
    topology_cfg["init_mode"] = "scratch"

    model = build_model(config)
    dataloaders = build_dataloaders(config=config)
    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_finetune": "topology_ft_case"},
        )
        best_checkpoint = run_topology_finetuning_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=tmp_path / "missing_checkpoint.pth",
        )
    finally:
        os.chdir(previous_cwd)

    assert best_checkpoint == Path("models/v3/topology_finetune/topology_ft_case/best_model.pth")


def test_run_topology_finetuning_stage_requires_prepared_supervision_files(
    tmp_path: Path,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["init_mode"] = "scratch"
    topology_cfg["supervision_train_dataset"] = str(
        tmp_path / "benchmark" / "human" / "BFS" / "missing_train_ratio5.txt"
    )
    topology_cfg["supervision_valid_dataset"] = str(
        tmp_path / "benchmark" / "human" / "BFS" / "missing_valid_ratio5.txt"
    )

    model = build_model(config)
    dataloaders = build_dataloaders(config=config)

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="prepare them offline"):
            runtime = build_stage_runtime(
                config,
                stage_run_ids={"topology_finetune": "topology_ft_case"},
            )
            run_topology_finetuning_stage(
                runtime,
                model,
                cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
                checkpoint_path=None,
            )
    finally:
        os.chdir(previous_cwd)


def test_prepare_topology_supervision_from_config_generates_missing_ratio_supervision_files(
    tmp_path: Path,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["bce_negative_ratio"] = 5
    topology_cfg["supervision_train_dataset"] = str(
        tmp_path / "benchmark" / "human" / "BFS" / "human_train_ppi_ratio5_exclusive.txt"
    )
    topology_cfg["supervision_valid_dataset"] = str(
        tmp_path / "benchmark" / "human" / "BFS" / "human_val_ppi_ratio5_exclusive.txt"
    )
    manifest = prepare_topology_supervision_from_config(config)

    train_supervision_path = Path(str(topology_cfg["supervision_train_dataset"]))
    valid_supervision_path = Path(str(topology_cfg["supervision_valid_dataset"]))
    assert manifest is not None
    assert train_supervision_path.exists()
    assert valid_supervision_path.exists()


def test_v3_configs_omit_validation_mode() -> None:
    for config_path in (Path("configs/v3.yaml"), Path("configs/v3.1.yaml")):
        config = load_config(config_path)
        topology_cfg = config["topology_finetune"]
        assert isinstance(topology_cfg, dict)
        assert "validation_mode" not in topology_cfg
