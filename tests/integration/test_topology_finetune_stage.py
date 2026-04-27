"""Integration tests for the graph-topology fine-tuning stage."""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
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
            "subgraph_node_range": [3, 4],
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
        self._gradient_accumulation_steps = 1
        self.gradient_accumulation_steps_history = [1]
        self.sync_gradients = True
        self.prepare_calls = 0
        self.autocast_calls = 0
        self.backward_calls = 0
        self.reduce_calls = 0
        self.accumulate_calls = 0
        self.accumulate_steps_seen: list[int] = []
        self.no_sync_calls = 0
        self.step = 0

    @property
    def gradient_accumulation_steps(self) -> int:
        return self._gradient_accumulation_steps

    @gradient_accumulation_steps.setter
    def gradient_accumulation_steps(self, value: int) -> None:
        self._gradient_accumulation_steps = int(value)
        self.gradient_accumulation_steps_history.append(self._gradient_accumulation_steps)

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

    def accumulate(self, *models: torch.nn.Module) -> object:
        from contextlib import nullcontext

        del models
        self.accumulate_calls += 1
        self.step += 1
        steps = max(1, int(self.gradient_accumulation_steps))
        self.accumulate_steps_seen.append(steps)
        self.sync_gradients = self.step % steps == 0
        return nullcontext()

    def no_sync(self, model: torch.nn.Module) -> object:
        from contextlib import nullcontext

        del model
        self.no_sync_calls += 1
        return nullcontext()

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


class _DistributedValidationAccelerator(_RecordingAccelerator):
    def __init__(
        self,
        *,
        peer_indices: Sequence[int],
        peer_predictions: Sequence[int],
    ) -> None:
        super().__init__()
        self.use_distributed = True
        self.is_main_process = True
        self.process_index = 0
        self.local_process_index = 0
        self.num_processes = 2
        self._peer_indices = [int(index) for index in peer_indices]
        self._peer_predictions = [int(prediction) for prediction in peer_predictions]
        self._gather_calls = 0

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        self._gather_calls += 1
        if self._gather_calls == 1:
            return torch.tensor(
                [int(value[0].item()), len(self._peer_indices)],
                dtype=value.dtype,
                device=value.device,
            )
        if self._gather_calls == 2:
            padded_peer = self.pad_across_processes(
                torch.tensor(self._peer_indices, dtype=value.dtype, device=value.device),
                dim=0,
                pad_index=-1,
            )
            return torch.cat((value, padded_peer), dim=0)
        if self._gather_calls == 3:
            padded_peer = self.pad_across_processes(
                torch.tensor(self._peer_predictions, dtype=value.dtype, device=value.device),
                dim=0,
                pad_index=0,
            )
            return torch.cat((value, padded_peer), dim=0)
        raise AssertionError("unexpected gather call")

    def pad_across_processes(
        self,
        value: torch.Tensor,
        dim: int = 0,
        pad_index: int = 0,
        pad_first: bool = False,
    ) -> torch.Tensor:
        del dim, pad_first
        target_length = max(value.numel(), len(self._peer_indices))
        if value.numel() >= target_length:
            return value
        padding = torch.full(
            (target_length - value.numel(),),
            fill_value=pad_index,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat((value, padding), dim=0)


class _CountingOptimizer:
    def __init__(self, *, should_apply: Callable[[], bool] | None = None) -> None:
        self.step_calls = 0
        self.zero_grad_calls = 0
        self._should_apply = should_apply

    def zero_grad(self, set_to_none: bool = False) -> None:
        del set_to_none
        if self._should_apply is not None and not self._should_apply():
            return
        self.zero_grad_calls += 1

    def step(self) -> None:
        if self._should_apply is not None and not self._should_apply():
            return
        self.step_calls += 1


class _AccelerateSemanticsAccelerator(_RecordingAccelerator):
    """Test double that mirrors Accelerate's accumulation behavior."""

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_calls += 1
        scaled_loss = loss / float(max(1, self.gradient_accumulation_steps))
        scaled_loss.backward()

    def accumulate(self, *models: torch.nn.Module) -> object:
        del models
        self.accumulate_calls += 1
        self.step += 1
        steps = max(1, int(self.gradient_accumulation_steps))
        self.accumulate_steps_seen.append(steps)
        self.sync_gradients = self.step % steps == 0
        return nullcontext()


class _ScalarLogitModel(torch.nn.Module):
    """Minimal model carrying a single trainable scalar for gradient assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))


class _GradientCaptureOptimizer:
    """Optimizer stub that records the gradient applied at each step."""

    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self._parameter = parameter
        self.step_calls = 0
        self.zero_grad_calls = 0
        self.step_gradients: list[float] = []

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.zero_grad_calls += 1
        if set_to_none:
            self._parameter.grad = None
            return
        if self._parameter.grad is not None:
            self._parameter.grad.zero_()

    def step(self) -> None:
        gradient = self._parameter.grad
        assert gradient is not None
        self.step_calls += 1
        self.step_gradients.append(float(gradient.detach().item()))


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


def test_build_internal_validation_node_sets_uses_optimized_default_sample_count() -> None:
    graph = nx.path_graph([f"P{i}" for i in range(1, 221)])

    node_sets = _build_internal_validation_node_sets(
        finetune_cfg={"strategy": "mixed"},
        graph=graph,
        seed=11,
    )

    assert sorted(node_sets) == list(TOPOLOGY_EVAL_NODE_SIZES)
    assert all(len(node_sets[node_size]) == 20 for node_size in node_sets)
    assert all(
        all(len(nodes) == node_size for nodes in node_sets[node_size]) for node_size in node_sets
    )


def test_build_internal_validation_node_sets_respects_configured_size_buckets() -> None:
    graph = nx.path_graph([f"P{i}" for i in range(1, 221)])

    node_sets = _build_internal_validation_node_sets(
        finetune_cfg={
            "strategy": "mixed",
            "internal_validation_node_sizes": [40, 80],
            "internal_validation_samples_per_size": 3,
        },
        graph=graph,
        seed=11,
    )

    assert sorted(node_sets) == [40, 80]
    assert all(len(node_sets[node_size]) == 3 for node_size in node_sets)


def test_build_internal_validation_node_sets_uses_graph_size_fallback_when_under_20() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["P3", "P1", "P2"])

    node_sets = _build_internal_validation_node_sets(
        finetune_cfg={"strategy": "mixed"},
        graph=graph,
        seed=11,
    )

    assert sorted(node_sets) == [3]
    assert len(node_sets[3]) == 20
    assert all(nodes == ("P1", "P2", "P3") for nodes in node_sets[3])


def test_resolve_internal_validation_threshold_uses_fixed_pring_threshold(
    tmp_path: Path,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["decision_threshold"] = {"mode": "fixed", "value": 0.5}
    threshold, mode = _resolve_internal_validation_threshold(
        config=config,
    )

    assert threshold == pytest.approx(0.5)
    assert mode == "fixed"


def test_parse_loss_weight_schedule_reads_warmup_ramp_and_schedule(
    tmp_path: Path,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["loss_weight_schedule"] = {
        "warmup_epochs": 5,
        "ramp_epochs": 4,
        "schedule": "cosine",
    }

    schedule = topology_finetune_stage._parse_loss_weight_schedule(topology_cfg)

    assert schedule.warmup_epochs == 5
    assert schedule.ramp_epochs == 4
    assert schedule.schedule == "cosine"


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


def test_validation_topology_loss_matches_weighted_hard_metric_penalties() -> None:
    loss = topology_finetune_stage._validation_topology_loss(
        loss_weights=topology_finetune_stage.TopologyLossWeights(
            alpha=0.5,
            beta=1.0,
            gamma=0.3,
            delta=0.2,
        ),
        internal_val_topology_stats={
            "graph_sim": 0.8,
            "relative_density": 1.1,
            "deg_dist_mmd": 0.4,
            "cc_mmd": 0.5,
        },
    )

    expected = 0.5 * (1.0 - 0.8) + (1.1 - 1.0) ** 2 + 0.3 * 0.4 + 0.2 * 0.5
    assert loss == pytest.approx(expected)


def test_validation_topology_loss_can_skip_clustering_mmd() -> None:
    loss = topology_finetune_stage._validation_topology_loss(
        loss_weights=topology_finetune_stage.TopologyLossWeights(
            alpha=0.5,
            beta=1.0,
            gamma=0.3,
            delta=0.2,
        ),
        internal_val_topology_stats={
            "graph_sim": 0.8,
            "relative_density": 1.1,
            "deg_dist_mmd": 0.4,
            "cc_mmd": 999.0,
        },
        include_clustering_mmd=False,
    )

    expected = 0.5 * (1.0 - 0.8) + (1.1 - 1.0) ** 2 + 0.3 * 0.4
    assert loss == pytest.approx(expected)


def test_internal_validation_can_skip_clustering_mmd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, bool] = {}

    def _record_graph_sample_evaluation(**kwargs: object) -> dict[str, object]:
        captured["include_clustering_stats"] = bool(kwargs["include_clustering_stats"])
        return {
            "summary": {
                "graph_sim": 1.0,
                "relative_density": 1.0,
                "deg_dist_mmd": 0.0,
                "cc_mmd": 0.0,
            }
        }

    monkeypatch.setattr(
        topology_finetune_stage,
        "evaluate_graph_samples",
        _record_graph_sample_evaluation,
    )

    summary = _evaluate_internal_validation_subgraphs(
        model=torch.nn.Identity(),
        validation_plan=topology_finetune_stage.InternalValidationPlan(
            buckets=(),
            protein_ids=frozenset(),
            total_subgraphs=0,
            total_pairs=0,
        ),
        embedding_repository=object(),  # type: ignore[arg-type]
        inference_batch_size=1,
        threshold=0.5,
        device=torch.device("cpu"),
        accelerator=NoOpAccelerator(),
        compute_clustering_mmd=False,
    )

    assert captured["include_clustering_stats"] is False
    assert summary["cc_mmd"] == pytest.approx(0.0)


def test_resolve_monitor_value_prefers_weighted_total_for_val_loss() -> None:
    monitor_value = _resolve_monitor_value(
        monitor_metric="val_loss",
        val_pair_stats={"val_loss": 0.42, "val_auprc": 0.91},
        internal_val_topology_stats={
            "graph_sim": 0.2,
            "relative_density": 1.1,
            "deg_dist_mmd": 0.3,
            "cc_mmd": 0.4,
        },
        val_total_loss=0.77,
    )

    assert monitor_value == pytest.approx(0.77)


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
        include_spectral_stats=False,
    )["summary"]

    assert batched_summary == pytest.approx(expected_summary)


def test_evaluate_internal_validation_subgraphs_shards_rank_local_work_and_merges_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("P1", "P2"),
            ("P2", "P3"),
            ("P3", "P4"),
            ("P4", "P5"),
            ("P5", "P6"),
        ]
    )
    cache_dir = tmp_path / "cache"
    _write_embedding_cache(
        cache_dir=cache_dir,
        embeddings={
            protein_id: torch.full((2, 4), float(index), dtype=torch.float32)
            for index, protein_id in enumerate(("P1", "P2", "P3", "P4", "P5", "P6"), start=1)
        },
        input_dim=4,
        max_sequence_length=8,
    )
    embedding_index = {
        protein_id: f"embeddings/{protein_id}.pt"
        for protein_id in ("P1", "P2", "P3", "P4", "P5", "P6")
    }
    sampled_subgraphs = {
        3: [
            ("P1", "P2", "P3"),
            ("P2", "P3", "P4"),
            ("P3", "P4", "P5"),
            ("P4", "P5", "P6"),
        ]
    }
    validation_plan = build_internal_validation_plan(
        graph=graph,
        sampled_subgraphs=sampled_subgraphs,
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
            return {"logits": (scores - 7.0) * 10.0}

    model = _ToyModel().eval()
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
                pair_batch_size=3,
                threshold=0.5,
                device=torch.device("cpu"),
                embedding_repository=embedding_repository,
            )
            for nodes in sampled_subgraphs[3]
        ]
    }
    expected_target_graphs = {3: [graph.subgraph(nodes).copy() for nodes in sampled_subgraphs[3]]}
    expected_summary = evaluate_graph_samples(
        pred_graphs_by_size=expected_pred_graphs,
        gt_graphs_by_size=expected_target_graphs,
        include_spectral_stats=False,
    )["summary"]

    processed_subgraph_indices: set[int] = set()
    original_validation_pair_batch = topology_finetune_stage._validation_pair_batch

    def _record_validation_pair_batch(
        *,
        pair_records: Sequence[topology_finetune_stage.InternalValidationPairRecord],
        embedding_repository: EmbeddingRepository,
    ) -> dict[str, torch.Tensor]:
        processed_subgraph_indices.update(int(record.subgraph_index) for record in pair_records)
        return original_validation_pair_batch(
            pair_records=pair_records,
            embedding_repository=embedding_repository,
        )

    monkeypatch.setattr(
        topology_finetune_stage,
        "_validation_pair_batch",
        _record_validation_pair_batch,
    )

    bucket = validation_plan.buckets[0]
    peer_indices = [
        index for index, record in enumerate(bucket.pair_records) if record.subgraph_index % 2 == 1
    ]
    peer_predictions = [
        int(
            expected_pred_graphs[3][record.subgraph_index].has_edge(
                record.protein_a,
                record.protein_b,
            )
        )
        for index, record in enumerate(bucket.pair_records)
        if index in peer_indices
    ]
    accelerator = _DistributedValidationAccelerator(
        peer_indices=peer_indices,
        peer_predictions=peer_predictions,
    )

    summary = _evaluate_internal_validation_subgraphs(
        model=model,
        validation_plan=validation_plan,
        embedding_repository=embedding_repository,
        inference_batch_size=16,
        threshold=0.5,
        device=torch.device("cpu"),
        accelerator=accelerator,
        compute_spectral_stats=False,
    )

    assert summary == pytest.approx(expected_summary)
    assert processed_subgraph_indices == {0, 2}


def test_run_topology_finetuning_stage_uses_fixed_threshold_after_validation_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["decision_threshold"] = {"mode": "fixed", "value": 0.5}
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

    monkeypatch.setattr(Evaluator, "collect_probabilities_and_labels", _record_collect)

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
        reader = DictReader(handle)
        header = reader.fieldnames
        rows = list(reader)
    assert header is not None
    assert "Planned Subgraphs" in header
    assert "Positive Edge Coverage Ratio" in header
    assert "Mean Positive Edge Reuse" in header
    assert "All Subgraph Pairs" in header
    assert "Supervised Pairs" in header
    assert "BCE Positive Pairs" in header
    assert "BCE Target Negative Pairs" in header
    assert "BCE Negative Pairs" in header
    assert "BCE Negative Ratio" in header
    assert "BCE Supervised Fraction" in header
    assert "edge_cover_sampling_s" in header
    assert "train_forward_backward_s" in header
    assert "val_pair_pass_s" in header
    assert "val_threshold_s" in header
    assert "internal_val_topology_s" in header
    assert "Topology Loss Scale" in header
    assert "peak_gpu_mem_mb" in header
    assert rows
    first_row = rows[0]
    assert int(first_row["Supervised Pairs"]) == int(first_row["BCE Positive Pairs"]) + int(
        first_row["BCE Negative Pairs"]
    )
    log_text = (log_dir / "log.log").read_text(encoding="utf-8")
    assert "Epoch Progress" in log_text
    assert "Legacy Validation Subgraphs Ignored" in log_text
    assert "Topo Loss Scale" in log_text

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


def test_run_topology_finetuning_stage_runs_validation_on_all_ranks_under_ddp(
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

    assert observed_pair_validation_ranks == [0, 1]
    assert observed_internal_validation_ranks == [0, 1]


def test_fit_epoch_accumulates_gradients_before_optimizer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["gradient_accumulation_steps"] = 2

    graph = nx.path_graph(["P1", "P2", "P3", "P4"])
    accelerator = NoOpAccelerator()
    optimizer = _CountingOptimizer()

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
    assert accelerator.accumulate_calls == 0
    assert accelerator.accumulate_steps_seen == []
    assert accelerator.no_sync_calls == 1
    assert accelerator.gradient_accumulation_steps == 1
    assert optimizer.step_calls == 2
    assert optimizer.zero_grad_calls == 3


def test_fit_epoch_chunked_backward_replays_pair_chunks_without_concat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["chunked_backward"] = True
    topology_cfg["compute_clustering_mmd"] = False
    topology_cfg["gradient_accumulation_steps"] = 1

    graph = nx.Graph()
    graph.add_edges_from([("P1", "P2"), ("P2", "P3")])
    accelerator = NoOpAccelerator()
    optimizer = _CountingOptimizer()
    forward_grad_modes: list[bool] = []

    def _fake_sample_edge_cover_subgraphs(**_: object) -> EdgeCoverEpochPlan:
        return EdgeCoverEpochPlan(
            subgraphs=(("P1", "P2", "P3"),),
            assigned_positive_edges=(frozenset({("P1", "P2")}),),
            assigned_negative_edges=(frozenset(),),
            total_positive_edges=2,
            covered_positive_edges=1,
            positive_edge_coverage_ratio=0.5,
            mean_positive_edge_reuse=1.0,
        )

    def _fake_iter_subgraph_pair_chunks(**kwargs: object) -> Sequence[SubgraphPairChunk]:
        nodes = tuple(cast(Sequence[str], kwargs["nodes"]))
        return (
            SubgraphPairChunk(
                nodes=nodes,
                emb_a=torch.ones((2, 1, 4), dtype=torch.float32),
                emb_b=torch.ones((2, 1, 4), dtype=torch.float32),
                len_a=torch.ones(2, dtype=torch.long),
                len_b=torch.ones(2, dtype=torch.long),
                label=torch.tensor([1.0, 0.0], dtype=torch.float32),
                pair_index_a=torch.tensor([0, 0], dtype=torch.long),
                pair_index_b=torch.tensor([1, 2], dtype=torch.long),
                bce_label=torch.tensor([1.0, 0.0], dtype=torch.float32),
                bce_mask=torch.tensor([1.0, 0.0], dtype=torch.float32),
            ),
            SubgraphPairChunk(
                nodes=nodes,
                emb_a=torch.ones((1, 1, 4), dtype=torch.float32),
                emb_b=torch.ones((1, 1, 4), dtype=torch.float32),
                len_a=torch.ones(1, dtype=torch.long),
                len_b=torch.ones(1, dtype=torch.long),
                label=torch.ones(1, dtype=torch.float32),
                pair_index_a=torch.ones(1, dtype=torch.long),
                pair_index_b=torch.full((1,), 2, dtype=torch.long),
                bce_label=torch.zeros(1, dtype=torch.float32),
                bce_mask=torch.zeros(1, dtype=torch.float32),
            ),
        )

    def _fake_forward_model(
        *,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del model
        forward_grad_modes.append(torch.is_grad_enabled())
        batch_size = int(batch["label"].numel())
        logits = torch.full(
            (batch_size,),
            0.25,
            dtype=torch.float32,
            requires_grad=torch.is_grad_enabled(),
        )
        return {"logits": logits}

    def _unexpected_concat_logits_and_pairs(**_: object) -> tuple[torch.Tensor, ...]:
        raise AssertionError("chunked backward must not concatenate all pair logits")

    monkeypatch.setattr(
        topology_finetune_stage,
        "sample_edge_cover_subgraphs",
        _fake_sample_edge_cover_subgraphs,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "iter_subgraph_pair_chunks",
        _fake_iter_subgraph_pair_chunks,
    )
    monkeypatch.setattr(topology_finetune_stage, "_forward_model", _fake_forward_model)
    monkeypatch.setattr(
        topology_finetune_stage,
        "_concat_logits_and_pairs",
        _unexpected_concat_logits_and_pairs,
    )

    train_stats = topology_finetune_stage._fit_epoch(
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
            alpha=0.2,
            beta=0.3,
            gamma=0.4,
            delta=0.0,
        ),
        pair_batch_size=2,
        use_amp=False,
        accelerator=accelerator,
        embedding_repository=cast(EmbeddingRepository, object()),
        negative_lookup=None,
        distributed_context=DistributedContext(ddp_enabled=False, is_distributed=False),
    )

    assert forward_grad_modes == [False, False, True, True, True]
    assert accelerator.backward_calls == 3
    assert optimizer.step_calls == 1
    assert train_stats["planned_subgraphs"] == pytest.approx(1.0)
    assert train_stats["all_subgraph_pairs"] == pytest.approx(3.0)


def test_chunked_backward_detached_pass_bypasses_ddp_reducer_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["chunked_backward"] = True
    topology_cfg["compute_clustering_mmd"] = False

    class _PairModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        def forward(
            self,
            emb_a: torch.Tensor,
            emb_b: torch.Tensor,
            len_a: torch.Tensor,
            len_b: torch.Tensor,
            **_: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del len_a, len_b
            logits = (emb_a + emb_b).mean(dim=(1, 2)) * self.weight
            return {"logits": logits}

    class _DdpReducerGuard(torch.nn.Module):
        def __init__(self, module: torch.nn.Module) -> None:
            super().__init__()
            self.module = module
            self.unreduced_no_grad_forward_seen = False

        def forward(self, **batch: torch.Tensor) -> dict[str, torch.Tensor]:
            if self.training and not torch.is_grad_enabled():
                self.unreduced_no_grad_forward_seen = True
            if (
                self.training
                and torch.is_grad_enabled()
                and self.unreduced_no_grad_forward_seen
            ):
                raise RuntimeError("Expected to have finished reduction in the prior iteration")
            return self.module(**batch)

    class _UnwrappingAccelerator(_RecordingAccelerator):
        def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
            if isinstance(model, _DdpReducerGuard):
                return model.module
            return model

    def _fake_iter_subgraph_pair_chunks(**kwargs: object) -> Sequence[SubgraphPairChunk]:
        nodes = tuple(cast(Sequence[str], kwargs["nodes"]))
        return (
            SubgraphPairChunk(
                nodes=nodes,
                emb_a=torch.ones((2, 1, 4), dtype=torch.float32),
                emb_b=torch.ones((2, 1, 4), dtype=torch.float32),
                len_a=torch.ones(2, dtype=torch.long),
                len_b=torch.ones(2, dtype=torch.long),
                label=torch.tensor([1.0, 0.0], dtype=torch.float32),
                pair_index_a=torch.tensor([0, 0], dtype=torch.long),
                pair_index_b=torch.tensor([1, 2], dtype=torch.long),
                bce_label=torch.tensor([1.0, 0.0], dtype=torch.float32),
                bce_mask=torch.tensor([1.0, 0.0], dtype=torch.float32),
            ),
        )

    def _fake_iter_supervised_pair_chunks(**_: object) -> Sequence[SubgraphPairChunk]:
        return (
            SubgraphPairChunk(
                nodes=("P1", "P2", "P3"),
                emb_a=torch.ones((1, 1, 4), dtype=torch.float32),
                emb_b=torch.ones((1, 1, 4), dtype=torch.float32),
                len_a=torch.ones(1, dtype=torch.long),
                len_b=torch.ones(1, dtype=torch.long),
                label=torch.ones(1, dtype=torch.float32),
                pair_index_a=torch.zeros(1, dtype=torch.long),
                pair_index_b=torch.ones(1, dtype=torch.long),
                bce_label=torch.ones(1, dtype=torch.float32),
                bce_mask=torch.ones(1, dtype=torch.float32),
            ),
        )

    monkeypatch.setattr(
        topology_finetune_stage,
        "iter_subgraph_pair_chunks",
        _fake_iter_subgraph_pair_chunks,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "iter_supervised_pair_chunks",
        _fake_iter_supervised_pair_chunks,
    )

    wrapped_model = _DdpReducerGuard(_PairModel()).train()
    accelerator = _UnwrappingAccelerator()
    result = topology_finetune_stage._backward_chunked_subgraph_task(
        config=config,
        model=wrapped_model,
        graph=nx.Graph([("P1", "P2"), ("P2", "P3")]),
        task=topology_finetune_stage.LocalSubgraphTask(
            nodes=("P1", "P2", "P3"),
            assigned_positive_edges=frozenset({("P1", "P2")}),
            assigned_negative_edges=frozenset({("P1", "P3")}),
        ),
        cache_dir=tmp_path,
        embedding_index={},
        input_dim=4,
        max_sequence_length=8,
        pair_batch_size=2,
        device=torch.device("cpu"),
        embedding_repository=cast(EmbeddingRepository, object()),
        loss_weights=topology_finetune_stage.TopologyLossWeights(
            alpha=0.2,
            beta=0.3,
            gamma=0.4,
            delta=0.0,
        ),
        loss_normalization=topology_finetune_stage.TopologyLossNormalizationConfig(),
        gradnorm=topology_finetune_stage.TopologyGradNormConfig(),
        adaptive_loss_state=topology_finetune_stage.TopologyAdaptiveLossState(),
        gradnorm_reference_parameters=(),
        current_window_size=1,
        accelerator=accelerator,
    )

    assert not wrapped_model.unreduced_no_grad_forward_seen
    assert result.total_loss.item() > 0.0
    assert accelerator.backward_calls == 3


def test_chunked_backward_replays_pair_chunks_with_same_rng_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["chunked_backward"] = True
    topology_cfg["compute_clustering_mmd"] = False

    class _StochasticPairModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.random_factors: list[tuple[bool, float]] = []

        def forward(
            self,
            emb_a: torch.Tensor,
            emb_b: torch.Tensor,
            len_a: torch.Tensor,
            len_b: torch.Tensor,
            **_: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del emb_b, len_a, len_b
            random_factor = torch.rand((), dtype=emb_a.dtype, device=emb_a.device)
            self.random_factors.append((torch.is_grad_enabled(), float(random_factor.detach())))
            logits = emb_a.mean(dim=(1, 2)) * self.weight * random_factor
            return {"logits": logits}

    def _fake_iter_subgraph_pair_chunks(**kwargs: object) -> Sequence[SubgraphPairChunk]:
        nodes = tuple(cast(Sequence[str], kwargs["nodes"]))
        return (
            SubgraphPairChunk(
                nodes=nodes,
                emb_a=torch.ones((2, 1, 4), dtype=torch.float32),
                emb_b=torch.ones((2, 1, 4), dtype=torch.float32),
                len_a=torch.ones(2, dtype=torch.long),
                len_b=torch.ones(2, dtype=torch.long),
                label=torch.tensor([1.0, 0.0], dtype=torch.float32),
                pair_index_a=torch.tensor([0, 0], dtype=torch.long),
                pair_index_b=torch.tensor([1, 2], dtype=torch.long),
                bce_label=torch.tensor([1.0, 0.0], dtype=torch.float32),
                bce_mask=torch.tensor([1.0, 1.0], dtype=torch.float32),
            ),
        )

    monkeypatch.setattr(
        topology_finetune_stage,
        "iter_subgraph_pair_chunks",
        _fake_iter_subgraph_pair_chunks,
    )

    model = _StochasticPairModel().train()
    torch.manual_seed(123)

    topology_finetune_stage._backward_chunked_subgraph_task(
        config=config,
        model=model,
        graph=nx.Graph([("P1", "P2"), ("P2", "P3")]),
        task=topology_finetune_stage.LocalSubgraphTask(
            nodes=("P1", "P2", "P3"),
            assigned_positive_edges=frozenset({("P1", "P2")}),
            assigned_negative_edges=frozenset(),
        ),
        cache_dir=tmp_path,
        embedding_index={},
        input_dim=4,
        max_sequence_length=8,
        pair_batch_size=2,
        device=torch.device("cpu"),
        embedding_repository=cast(EmbeddingRepository, object()),
        loss_weights=topology_finetune_stage.TopologyLossWeights(
            alpha=0.2,
            beta=0.3,
            gamma=0.4,
            delta=0.0,
        ),
        loss_normalization=topology_finetune_stage.TopologyLossNormalizationConfig(),
        gradnorm=topology_finetune_stage.TopologyGradNormConfig(),
        adaptive_loss_state=topology_finetune_stage.TopologyAdaptiveLossState(),
        gradnorm_reference_parameters=(),
        current_window_size=1,
        accelerator=_RecordingAccelerator(),
    )

    no_grad_factors = [factor for grad_enabled, factor in model.random_factors if not grad_enabled]
    grad_factors = [factor for grad_enabled, factor in model.random_factors if grad_enabled]
    assert no_grad_factors
    assert grad_factors
    assert grad_factors[0] == pytest.approx(no_grad_factors[0])


def test_chunked_backward_sync_boundary_uses_single_ddp_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["chunked_backward"] = True
    topology_cfg["compute_clustering_mmd"] = False

    class _SyncRecordingAccelerator(_RecordingAccelerator):
        def __init__(self) -> None:
            super().__init__()
            self._in_no_sync = False
            self.backward_sync_flags: list[bool] = []

        def backward(self, loss: torch.Tensor) -> None:
            self.backward_calls += 1
            self.backward_sync_flags.append(not self._in_no_sync)
            loss.backward()

        def no_sync(self, model: torch.nn.Module) -> object:
            del model
            self.no_sync_calls += 1
            accelerator = self

            class _NoSyncContext:
                def __enter__(self) -> object:
                    accelerator._in_no_sync = True
                    return self

                def __exit__(
                    self,
                    exc_type: object,
                    exc_value: object,
                    traceback: object,
                ) -> None:
                    del exc_type, exc_value, traceback
                    accelerator._in_no_sync = False

            return _NoSyncContext()

    class _PairModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

        def forward(
            self,
            emb_a: torch.Tensor,
            emb_b: torch.Tensor,
            len_a: torch.Tensor,
            len_b: torch.Tensor,
            **_: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del len_a, len_b
            logits = (emb_a + emb_b).mean(dim=(1, 2)) * self.weight
            return {"logits": logits}

    def _fake_iter_subgraph_pair_chunks(**kwargs: object) -> Sequence[SubgraphPairChunk]:
        nodes = tuple(cast(Sequence[str], kwargs["nodes"]))
        return (
            SubgraphPairChunk(
                nodes=nodes,
                emb_a=torch.ones((1, 1, 4), dtype=torch.float32),
                emb_b=torch.ones((1, 1, 4), dtype=torch.float32),
                len_a=torch.ones(1, dtype=torch.long),
                len_b=torch.ones(1, dtype=torch.long),
                label=torch.ones(1, dtype=torch.float32),
                pair_index_a=torch.zeros(1, dtype=torch.long),
                pair_index_b=torch.ones(1, dtype=torch.long),
                bce_label=torch.ones(1, dtype=torch.float32),
                bce_mask=torch.ones(1, dtype=torch.float32),
            ),
        )

    def _fake_iter_supervised_pair_chunks(**_: object) -> Sequence[SubgraphPairChunk]:
        return (
            SubgraphPairChunk(
                nodes=("P1", "P2", "P3"),
                emb_a=torch.ones((1, 1, 4), dtype=torch.float32),
                emb_b=torch.ones((1, 1, 4), dtype=torch.float32),
                len_a=torch.ones(1, dtype=torch.long),
                len_b=torch.ones(1, dtype=torch.long),
                label=torch.ones(1, dtype=torch.float32),
                pair_index_a=torch.zeros(1, dtype=torch.long),
                pair_index_b=torch.ones(1, dtype=torch.long),
                bce_label=torch.ones(1, dtype=torch.float32),
                bce_mask=torch.ones(1, dtype=torch.float32),
            ),
        )

    monkeypatch.setattr(
        topology_finetune_stage,
        "iter_subgraph_pair_chunks",
        _fake_iter_subgraph_pair_chunks,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "iter_supervised_pair_chunks",
        _fake_iter_supervised_pair_chunks,
    )

    accelerator = _SyncRecordingAccelerator()
    accelerator.sync_gradients = True
    topology_finetune_stage._backward_chunked_subgraph_task(
        config=config,
        model=_PairModel().train(),
        graph=nx.Graph([("P1", "P2"), ("P2", "P3")]),
        task=topology_finetune_stage.LocalSubgraphTask(
            nodes=("P1", "P2", "P3"),
            assigned_positive_edges=frozenset({("P1", "P2")}),
            assigned_negative_edges=frozenset({("P1", "P3")}),
        ),
        cache_dir=tmp_path,
        embedding_index={},
        input_dim=4,
        max_sequence_length=8,
        pair_batch_size=2,
        device=torch.device("cpu"),
        embedding_repository=cast(EmbeddingRepository, object()),
        loss_weights=topology_finetune_stage.TopologyLossWeights(
            alpha=0.2,
            beta=0.3,
            gamma=0.4,
            delta=0.0,
        ),
        loss_normalization=topology_finetune_stage.TopologyLossNormalizationConfig(),
        gradnorm=topology_finetune_stage.TopologyGradNormConfig(),
        adaptive_loss_state=topology_finetune_stage.TopologyAdaptiveLossState(),
        gradnorm_reference_parameters=(),
        current_window_size=1,
        accelerator=accelerator,
    )

    assert accelerator.no_sync_calls == 1
    assert accelerator.backward_sync_flags == [False, False, True]


def test_fit_epoch_flushes_remainder_window_without_leaking_final_gradient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["gradient_accumulation_steps"] = 3

    graph = nx.path_graph(["P1", "P2", "P3", "P4", "P5", "P6"])
    model = _ScalarLogitModel()
    optimizer = _GradientCaptureOptimizer(model.weight)
    accelerator = _AccelerateSemanticsAccelerator()
    task_values = {
        ("P1", "P2"): 1.0,
        ("P2", "P3"): 2.0,
        ("P3", "P4"): 3.0,
        ("P4", "P5"): 4.0,
        ("P5", "P6"): 5.0,
    }

    def _fake_sample_edge_cover_subgraphs(**_: object) -> EdgeCoverEpochPlan:
        return EdgeCoverEpochPlan(
            subgraphs=tuple(task_values),
            assigned_positive_edges=tuple(frozenset({nodes}) for nodes in task_values),
            total_positive_edges=len(task_values),
            covered_positive_edges=len(task_values),
            positive_edge_coverage_ratio=1.0,
            mean_positive_edge_reuse=1.0,
        )

    def _fake_forward_supervised_task(
        *,
        model: torch.nn.Module,
        task: topology_finetune_stage.LocalSubgraphTask,
        **_: object,
    ) -> topology_finetune_stage.SupervisedForwardResult:
        assert isinstance(model, _ScalarLogitModel)
        value = torch.tensor([task_values[task.nodes]], dtype=torch.float32)
        return topology_finetune_stage.SupervisedForwardResult(
            logits=model.weight * value,
            bce_labels=torch.ones(1, dtype=torch.float32),
            bce_mask=torch.ones(1, dtype=torch.float32),
        )

    monkeypatch.setattr(
        topology_finetune_stage,
        "sample_edge_cover_subgraphs",
        _fake_sample_edge_cover_subgraphs,
    )
    monkeypatch.setattr(topology_finetune_stage, "topology_loss_scale", lambda **_: 0.0)
    monkeypatch.setattr(
        topology_finetune_stage,
        "_forward_supervised_task",
        _fake_forward_supervised_task,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "_masked_bce_loss",
        lambda *, logits, **_: logits.sum(),
    )

    topology_finetune_stage._fit_epoch(
        config=config,
        model=model,
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

    assert optimizer.step_gradients == [2.0, 4.5]
    assert optimizer.step_calls == 2
    assert optimizer.zero_grad_calls == 3
    assert model.weight.grad is None


def test_fit_epoch_batches_multiple_subgraphs_into_one_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["gradient_accumulation_steps"] = 4
    topology_cfg["subgraphs_per_forward"] = 2

    graph = nx.path_graph(["P1", "P2", "P3", "P4", "P5"])
    optimizer = _CountingOptimizer()
    accelerator = NoOpAccelerator()
    observed_forward_batch_sizes: list[int] = []

    def _fake_sample_edge_cover_subgraphs(**_: object) -> EdgeCoverEpochPlan:
        return EdgeCoverEpochPlan(
            subgraphs=(("P1", "P2"), ("P2", "P3"), ("P3", "P4"), ("P4", "P5")),
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

    def _fake_iter_subgraph_pair_chunks(**kwargs: object) -> Sequence[SubgraphPairChunk]:
        nodes = tuple(cast(Sequence[str], kwargs["nodes"]))
        return (
            SubgraphPairChunk(
                nodes=nodes,
                emb_a=torch.zeros((1, 1, 4), dtype=torch.float32),
                emb_b=torch.ones((1, 1, 4), dtype=torch.float32),
                len_a=torch.ones(1, dtype=torch.long),
                len_b=torch.ones(1, dtype=torch.long),
                label=torch.ones(1, dtype=torch.float32),
                pair_index_a=torch.zeros(1, dtype=torch.long),
                pair_index_b=torch.ones(1, dtype=torch.long),
                bce_label=torch.ones(1, dtype=torch.float32),
                bce_mask=torch.ones(1, dtype=torch.float32),
            ),
        )

    def _fake_forward_model(
        *,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del model
        batch_size = int(batch["label"].numel())
        observed_forward_batch_sizes.append(batch_size)
        return {"logits": torch.zeros(batch_size, dtype=torch.float32, requires_grad=True)}

    monkeypatch.setattr(
        topology_finetune_stage,
        "sample_edge_cover_subgraphs",
        _fake_sample_edge_cover_subgraphs,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "iter_subgraph_pair_chunks",
        _fake_iter_subgraph_pair_chunks,
    )
    monkeypatch.setattr(topology_finetune_stage, "_forward_model", _fake_forward_model)

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
        pair_batch_size=8,
        use_amp=False,
        accelerator=accelerator,
        embedding_repository=cast(EmbeddingRepository, object()),
        negative_lookup=None,
        distributed_context=DistributedContext(ddp_enabled=False, is_distributed=False),
    )

    assert observed_forward_batch_sizes == [2, 2]
    assert accelerator.accumulate_calls == 0
    assert accelerator.accumulate_steps_seen == []
    assert accelerator.no_sync_calls == 1
    assert accelerator.gradient_accumulation_steps == 1


def test_fit_epoch_grouped_accumulation_scales_by_subgraph_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["gradient_accumulation_steps"] = 4
    topology_cfg["subgraphs_per_forward"] = 2

    graph = nx.path_graph(["P1", "P2", "P3", "P4", "P5"])
    model = _ScalarLogitModel()
    optimizer = _GradientCaptureOptimizer(model.weight)
    accelerator = _AccelerateSemanticsAccelerator()
    task_values = {
        ("P1", "P2"): 1.0,
        ("P2", "P3"): 2.0,
        ("P3", "P4"): 3.0,
        ("P4", "P5"): 4.0,
    }

    def _fake_sample_edge_cover_subgraphs(**_: object) -> EdgeCoverEpochPlan:
        return EdgeCoverEpochPlan(
            subgraphs=tuple(task_values),
            assigned_positive_edges=tuple(frozenset({nodes}) for nodes in task_values),
            total_positive_edges=len(task_values),
            covered_positive_edges=len(task_values),
            positive_edge_coverage_ratio=1.0,
            mean_positive_edge_reuse=1.0,
        )

    def _fake_forward_supervised_group(
        *,
        model: torch.nn.Module,
        tasks: Sequence[topology_finetune_stage.LocalSubgraphTask],
        **_: object,
    ) -> tuple[topology_finetune_stage.SupervisedForwardResult, ...]:
        assert isinstance(model, _ScalarLogitModel)
        return tuple(
            topology_finetune_stage.SupervisedForwardResult(
                logits=model.weight * torch.tensor([task_values[task.nodes]], dtype=torch.float32),
                bce_labels=torch.ones(1, dtype=torch.float32),
                bce_mask=torch.ones(1, dtype=torch.float32),
            )
            for task in tasks
        )

    monkeypatch.setattr(
        topology_finetune_stage,
        "sample_edge_cover_subgraphs",
        _fake_sample_edge_cover_subgraphs,
    )
    monkeypatch.setattr(topology_finetune_stage, "topology_loss_scale", lambda **_: 0.0)
    monkeypatch.setattr(
        topology_finetune_stage,
        "_forward_supervised_group",
        _fake_forward_supervised_group,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "_masked_bce_loss",
        lambda *, logits, **_: logits.sum(),
    )

    topology_finetune_stage._fit_epoch(
        config=config,
        model=model,
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
        pair_batch_size=8,
        use_amp=False,
        accelerator=accelerator,
        embedding_repository=cast(EmbeddingRepository, object()),
        negative_lookup=None,
        distributed_context=DistributedContext(ddp_enabled=False, is_distributed=False),
    )

    assert optimizer.step_gradients == [2.5]
    assert optimizer.step_calls == 1
    assert optimizer.zero_grad_calls == 2
    assert model.weight.grad is None


def test_fit_epoch_accumulates_grouped_partial_window_with_dynamic_window_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["gradient_accumulation_steps"] = 4
    topology_cfg["subgraphs_per_forward"] = 2

    graph = nx.path_graph(["P1", "P2", "P3", "P4", "P5", "P6"])
    optimizer = _CountingOptimizer()
    accelerator = NoOpAccelerator()

    def _fake_sample_edge_cover_subgraphs(**_: object) -> EdgeCoverEpochPlan:
        return EdgeCoverEpochPlan(
            subgraphs=(
                ("P1", "P2"),
                ("P2", "P3"),
                ("P3", "P4"),
                ("P4", "P5"),
                ("P5", "P6"),
            ),
            assigned_positive_edges=(
                frozenset({("P1", "P2")}),
                frozenset({("P2", "P3")}),
                frozenset({("P3", "P4")}),
                frozenset({("P4", "P5")}),
                frozenset({("P5", "P6")}),
            ),
            total_positive_edges=5,
            covered_positive_edges=5,
            positive_edge_coverage_ratio=1.0,
            mean_positive_edge_reuse=1.0,
        )

    def _fake_iter_subgraph_pair_chunks(**kwargs: object) -> Sequence[SubgraphPairChunk]:
        nodes = tuple(cast(Sequence[str], kwargs["nodes"]))
        return (
            SubgraphPairChunk(
                nodes=nodes,
                emb_a=torch.zeros((1, 1, 4), dtype=torch.float32),
                emb_b=torch.ones((1, 1, 4), dtype=torch.float32),
                len_a=torch.ones(1, dtype=torch.long),
                len_b=torch.ones(1, dtype=torch.long),
                label=torch.ones(1, dtype=torch.float32),
                pair_index_a=torch.zeros(1, dtype=torch.long),
                pair_index_b=torch.ones(1, dtype=torch.long),
                bce_label=torch.ones(1, dtype=torch.float32),
                bce_mask=torch.ones(1, dtype=torch.float32),
            ),
        )

    def _fake_forward_model(
        *,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del model
        batch_size = int(batch["label"].numel())
        return {"logits": torch.zeros(batch_size, dtype=torch.float32, requires_grad=True)}

    monkeypatch.setattr(
        topology_finetune_stage,
        "sample_edge_cover_subgraphs",
        _fake_sample_edge_cover_subgraphs,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "iter_subgraph_pair_chunks",
        _fake_iter_subgraph_pair_chunks,
    )
    monkeypatch.setattr(topology_finetune_stage, "_forward_model", _fake_forward_model)

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
        pair_batch_size=8,
        use_amp=False,
        accelerator=accelerator,
        embedding_repository=cast(EmbeddingRepository, object()),
        negative_lookup=None,
        distributed_context=DistributedContext(ddp_enabled=False, is_distributed=False),
    )

    assert accelerator.accumulate_calls == 0
    assert accelerator.accumulate_steps_seen == []
    assert accelerator.no_sync_calls == 1
    assert accelerator.gradient_accumulation_steps == 1
    assert optimizer.step_calls == 2
    assert optimizer.zero_grad_calls == 3


def test_fit_epoch_skips_topology_work_during_warmup_single_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    graph = nx.path_graph(["P1", "P2", "P3"])

    def _fake_sample_edge_cover_subgraphs(**_: object) -> EdgeCoverEpochPlan:
        return EdgeCoverEpochPlan(
            subgraphs=(("P1", "P2"),),
            assigned_positive_edges=(frozenset({("P1", "P2")}),),
            total_positive_edges=1,
            covered_positive_edges=1,
            positive_edge_coverage_ratio=1.0,
            mean_positive_edge_reuse=1.0,
        )

    def _unexpected_concat_logits_and_pairs(**_: object) -> tuple[torch.Tensor, ...]:
        raise AssertionError("warmup should not forward all subgraph pairs")

    def _fake_forward_supervised_task(
        **_: object,
    ) -> topology_finetune_stage.SupervisedForwardResult:
        return topology_finetune_stage.SupervisedForwardResult(
            logits=torch.zeros(1, dtype=torch.float32, requires_grad=True),
            bce_labels=torch.ones(1, dtype=torch.float32),
            bce_mask=torch.ones(1, dtype=torch.float32),
        )

    def _unexpected_subgraph_adjacencies(**_: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("warmup should skip topology adjacency construction")

    def _unexpected_topology_losses(**_: object) -> dict[str, torch.Tensor]:
        raise AssertionError("warmup should skip topology loss computation")

    monkeypatch.setattr(
        topology_finetune_stage,
        "sample_edge_cover_subgraphs",
        _fake_sample_edge_cover_subgraphs,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "_concat_logits_and_pairs",
        _unexpected_concat_logits_and_pairs,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "_forward_supervised_task",
        _fake_forward_supervised_task,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "_subgraph_adjacencies",
        _unexpected_subgraph_adjacencies,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "compute_topology_losses",
        _unexpected_topology_losses,
    )

    train_stats = topology_finetune_stage._fit_epoch(
        config=config,
        model=torch.nn.Linear(1, 1),
        device=torch.device("cpu"),
        graph=graph,
        cache_dir=tmp_path,
        embedding_index={},
        optimizer=cast(torch.optim.Optimizer, _CountingOptimizer()),
        epoch_index=0,
        epoch_seed=11,
        input_dim=4,
        max_sequence_length=8,
        loss_weights=topology_finetune_stage.TopologyLossWeights(
            alpha=0.5,
            beta=1.0,
            gamma=0.3,
            delta=0.2,
        ),
        loss_weight_schedule=topology_finetune_stage.TopologyLossWeightSchedule(
            warmup_epochs=2,
            ramp_epochs=1,
        ),
        pair_batch_size=2,
        use_amp=False,
        accelerator=NoOpAccelerator(),
        embedding_repository=cast(EmbeddingRepository, object()),
        negative_lookup=None,
        distributed_context=DistributedContext(ddp_enabled=False, is_distributed=False),
    )

    assert train_stats["topology_loss_scale"] == pytest.approx(0.0)
    assert train_stats["bce"] > 0.0
    assert train_stats["total"] == pytest.approx(train_stats["bce"])
    assert train_stats["graph_similarity"] == pytest.approx(0.0)
    assert train_stats["relative_density"] == pytest.approx(0.0)
    assert train_stats["degree_mmd"] == pytest.approx(0.0)
    assert train_stats["clustering_mmd"] == pytest.approx(0.0)


def test_fit_epoch_skips_topology_work_during_warmup_grouped_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["gradient_accumulation_steps"] = 2
    topology_cfg["subgraphs_per_forward"] = 2
    graph = nx.path_graph(["P1", "P2", "P3", "P4"])

    def _fake_sample_edge_cover_subgraphs(**_: object) -> EdgeCoverEpochPlan:
        return EdgeCoverEpochPlan(
            subgraphs=(("P1", "P2"), ("P2", "P3")),
            assigned_positive_edges=(
                frozenset({("P1", "P2")}),
                frozenset({("P2", "P3")}),
            ),
            total_positive_edges=2,
            covered_positive_edges=2,
            positive_edge_coverage_ratio=1.0,
            mean_positive_edge_reuse=1.0,
        )

    def _unexpected_forward_subgraph_group(
        **_: object,
    ) -> tuple[
        topology_finetune_stage.SubgraphForwardResult,
        ...,
    ]:
        raise AssertionError("warmup should not forward grouped all-pairs subgraphs")

    def _fake_forward_supervised_group(
        *,
        tasks: Sequence[topology_finetune_stage.LocalSubgraphTask],
        **_: object,
    ) -> tuple[topology_finetune_stage.SupervisedForwardResult, ...]:
        return tuple(
            topology_finetune_stage.SupervisedForwardResult(
                logits=torch.zeros(1, dtype=torch.float32, requires_grad=True),
                bce_labels=torch.ones(1, dtype=torch.float32),
                bce_mask=torch.ones(1, dtype=torch.float32),
            )
            for _ in tasks
        )

    def _unexpected_subgraph_adjacencies(**_: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("warmup should skip topology adjacency construction")

    def _unexpected_topology_losses(**_: object) -> dict[str, torch.Tensor]:
        raise AssertionError("warmup should skip topology loss computation")

    monkeypatch.setattr(
        topology_finetune_stage,
        "sample_edge_cover_subgraphs",
        _fake_sample_edge_cover_subgraphs,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "_forward_subgraph_group",
        _unexpected_forward_subgraph_group,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "_forward_supervised_group",
        _fake_forward_supervised_group,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "_subgraph_adjacencies",
        _unexpected_subgraph_adjacencies,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "compute_topology_losses",
        _unexpected_topology_losses,
    )

    train_stats = topology_finetune_stage._fit_epoch(
        config=config,
        model=torch.nn.Linear(1, 1),
        device=torch.device("cpu"),
        graph=graph,
        cache_dir=tmp_path,
        embedding_index={},
        optimizer=cast(torch.optim.Optimizer, _CountingOptimizer()),
        epoch_index=0,
        epoch_seed=11,
        input_dim=4,
        max_sequence_length=8,
        loss_weights=topology_finetune_stage.TopologyLossWeights(
            alpha=0.5,
            beta=1.0,
            gamma=0.3,
            delta=0.2,
        ),
        loss_weight_schedule=topology_finetune_stage.TopologyLossWeightSchedule(
            warmup_epochs=2,
            ramp_epochs=1,
        ),
        pair_batch_size=8,
        use_amp=False,
        accelerator=NoOpAccelerator(),
        embedding_repository=cast(EmbeddingRepository, object()),
        negative_lookup=None,
        distributed_context=DistributedContext(ddp_enabled=False, is_distributed=False),
    )

    assert train_stats["topology_loss_scale"] == pytest.approx(0.0)
    assert train_stats["bce"] > 0.0
    assert train_stats["total"] == pytest.approx(train_stats["bce"])
    assert train_stats["graph_similarity"] == pytest.approx(0.0)
    assert train_stats["relative_density"] == pytest.approx(0.0)
    assert train_stats["degree_mmd"] == pytest.approx(0.0)
    assert train_stats["clustering_mmd"] == pytest.approx(0.0)


def _run_fit_epoch_for_rank(
    *,
    tmp_path: Path,
    config: ConfigDict,
    graph: nx.Graph,
    distributed_context: DistributedContext,
) -> NoOpAccelerator:
    accelerator = NoOpAccelerator(distributed=distributed_context)
    topology_finetune_stage._fit_epoch(
        config=config,
        model=torch.nn.Linear(1, 1),
        device=torch.device("cpu"),
        graph=graph,
        cache_dir=tmp_path,
        embedding_index={},
        optimizer=cast(torch.optim.Optimizer, _CountingOptimizer()),
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
        pair_batch_size=8,
        use_amp=False,
        accelerator=accelerator,
        embedding_repository=cast(EmbeddingRepository, object()),
        negative_lookup=None,
        distributed_context=distributed_context,
    )
    return accelerator


def test_fit_epoch_pads_uneven_ddp_subgraph_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    graph = nx.path_graph(["P1", "P2", "P3", "P4"])

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

    backward_counts = [
        _run_fit_epoch_for_rank(
            tmp_path=tmp_path,
            config=config,
            graph=graph,
            distributed_context=DistributedContext(
                ddp_enabled=True,
                is_distributed=True,
                rank=rank,
                local_rank=rank,
                world_size=2,
            ),
        ).backward_calls
        for rank in (0, 1)
    ]

    assert backward_counts == [2, 2]


def test_fit_epoch_pads_uneven_ddp_grouped_forward_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["subgraphs_per_forward"] = 2
    topology_cfg["gradient_accumulation_steps"] = 4
    graph = nx.path_graph(["P1", "P2", "P3", "P4", "P5", "P6"])

    def _fake_sample_edge_cover_subgraphs(**_: object) -> EdgeCoverEpochPlan:
        return EdgeCoverEpochPlan(
            subgraphs=(
                ("P1", "P2"),
                ("P2", "P3"),
                ("P3", "P4"),
                ("P4", "P5"),
                ("P5", "P6"),
            ),
            assigned_positive_edges=(
                frozenset({("P1", "P2")}),
                frozenset({("P2", "P3")}),
                frozenset({("P3", "P4")}),
                frozenset({("P4", "P5")}),
                frozenset({("P5", "P6")}),
            ),
            total_positive_edges=5,
            covered_positive_edges=5,
            positive_edge_coverage_ratio=1.0,
            mean_positive_edge_reuse=1.0,
        )

    def _fake_iter_subgraph_pair_chunks(**kwargs: object) -> Sequence[SubgraphPairChunk]:
        nodes = tuple(cast(Sequence[str], kwargs["nodes"]))
        return (
            SubgraphPairChunk(
                nodes=nodes,
                emb_a=torch.zeros((1, 1, 4), dtype=torch.float32),
                emb_b=torch.ones((1, 1, 4), dtype=torch.float32),
                len_a=torch.ones(1, dtype=torch.long),
                len_b=torch.ones(1, dtype=torch.long),
                label=torch.ones(1, dtype=torch.float32),
                pair_index_a=torch.zeros(1, dtype=torch.long),
                pair_index_b=torch.ones(1, dtype=torch.long),
                bce_label=torch.ones(1, dtype=torch.float32),
                bce_mask=torch.ones(1, dtype=torch.float32),
            ),
        )

    def _fake_forward_model(
        *,
        model: torch.nn.Module,
        batch: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del model
        batch_size = int(batch["label"].numel())
        return {"logits": torch.zeros(batch_size, dtype=torch.float32, requires_grad=True)}

    monkeypatch.setattr(
        topology_finetune_stage,
        "sample_edge_cover_subgraphs",
        _fake_sample_edge_cover_subgraphs,
    )
    monkeypatch.setattr(
        topology_finetune_stage,
        "iter_subgraph_pair_chunks",
        _fake_iter_subgraph_pair_chunks,
    )
    monkeypatch.setattr(topology_finetune_stage, "_forward_model", _fake_forward_model)

    backward_counts = [
        _run_fit_epoch_for_rank(
            tmp_path=tmp_path,
            config=config,
            graph=graph,
            distributed_context=DistributedContext(
                ddp_enabled=True,
                is_distributed=True,
                rank=rank,
                local_rank=rank,
                world_size=2,
            ),
        ).backward_calls
        for rank in (0, 1)
    ]

    assert backward_counts == [2, 2]


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
    topology_cfg["bce_negative_ratio"] = 2
    topology_cfg["loss_weight_schedule"] = {
        "warmup_epochs": 1,
        "ramp_epochs": 1,
        "schedule": "linear",
    }

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

    final_linear = model.output_head.layers[-1]
    assert isinstance(final_linear, torch.nn.Linear)
    assert final_linear.bias is not None
    assert final_linear.bias.item() == pytest.approx(
        torch.logit(torch.tensor(1.0 / 3.0, dtype=torch.float32)).item(),
        rel=1e-6,
    )
    assert best_checkpoint == Path("models/v3/topology_finetune/topology_ft_case/best_model.pth")


def test_evaluate_validation_epoch_skips_internal_topology_validation_during_warmup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_finetune_config(tmp_path)
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)
    topology_cfg["internal_validation_warmup_cadence"] = 0
    model = build_model(config)
    dataloaders = build_dataloaders(config=config)
    train_graph, internal_val_graph = _load_supervision_graphs(config=config)
    embedding_index = {
        protein_id: f"embeddings/{protein_id}.pt"
        for protein_id in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10")
    }
    embedding_repository = EmbeddingRepository(
        cache_dir=Path(str(config["data_config"]["embeddings"]["cache_dir"])),  # type: ignore[index]
        embedding_index=embedding_index,
        input_dim=4,
        max_sequence_length=8,
        max_cache_bytes=1_024,
    )
    context = topology_finetune_stage.TopologyFinetuneStageContext(
        train_graph=train_graph,
        internal_val_graph=internal_val_graph,
        train_negative_lookup=topology_finetune_stage.ExplicitNegativePairLookup(
            negative_pairs=frozenset(),
            partners_by_node={},
        ),
        cache_dir=Path(str(config["data_config"]["embeddings"]["cache_dir"])),  # type: ignore[index]
        embedding_index=embedding_index,
        embedding_repository=embedding_repository,
        input_dim=4,
        max_sequence_length=8,
        pair_batch_size=2,
        internal_validation_inference_batch_size=2,
        internal_validation_compute_spectral_stats=False,
        compute_clustering_mmd=True,
        internal_validation_compute_clustering_mmd=True,
        epochs=1,
        run_seed=11,
        use_amp=False,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        evaluator=Evaluator(
            metrics=["auprc"],
            loss_config=topology_finetune_stage._build_loss_config(config["training_config"]),  # type: ignore[arg-type]
            accelerator=NoOpAccelerator(),
        ),
        loss_weights=topology_finetune_stage.TopologyLossWeights(),
        loss_weight_schedule=topology_finetune_stage.TopologyLossWeightSchedule(
            warmup_epochs=1,
            ramp_epochs=1,
        ),
        loss_normalization=topology_finetune_stage.TopologyLossNormalizationConfig(),
        gradnorm=topology_finetune_stage.TopologyGradNormConfig(),
        adaptive_loss_state=topology_finetune_stage.TopologyAdaptiveLossState(),
        monitor_metric="val_auprc",
        early_stopping=topology_finetune_stage.EarlyStopping(patience=1, mode="max"),
        best_checkpoint_path=tmp_path / "best.pth",
        metrics_path=tmp_path / "metrics.json",
        csv_path=tmp_path / "metrics.csv",
        internal_validation_node_sets={},
        internal_validation_plan=build_internal_validation_plan(
            graph=internal_val_graph,
            sampled_subgraphs={3: [("P1", "P2", "P3")]},
        ),
    )

    def _unexpected_internal_validation(**_: object) -> dict[str, float]:
        raise AssertionError("internal topology validation should be skipped during warmup")

    monkeypatch.setattr(
        topology_finetune_stage,
        "_evaluate_internal_validation_subgraphs",
        _unexpected_internal_validation,
    )

    result = topology_finetune_stage._evaluate_validation_epoch(
        config=config,
        model=model,
        dataloaders=cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
        device=torch.device("cpu"),
        context=context,
        loss_weights=topology_finetune_stage.TopologyLossWeights(),
        epoch_index=0,
        topology_loss_scale=0.0,
        previous_topology_loss_scale=None,
    )

    assert result.internal_val_topology_stats == pytest.approx(
        {
            "graph_sim": 0.0,
            "relative_density": 0.0,
            "deg_dist_mmd": 0.0,
            "cc_mmd": 0.0,
        }
    )
    assert result.internal_validation_seconds == pytest.approx(0.0)


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
    for config_path in (Path("configs/v3.yaml"), Path("configs/v3-1/v3.1.yaml")):
        config = load_config(config_path)
        topology_cfg = config["topology_finetune"]
        assert isinstance(topology_cfg, dict)
        assert "validation_mode" not in topology_cfg


def test_v3_1_config_uses_ohem_and_fixed_pooling_hpo_space() -> None:
    config = load_config("configs/v3-1/v3.1.yaml")
    data_cfg = config["data_config"]
    model_cfg = config["model_config"]
    optimization_cfg = config["optimization"]
    assert isinstance(data_cfg, dict)
    assert isinstance(model_cfg, dict)
    assert isinstance(optimization_cfg, dict)

    dataloader_cfg = data_cfg["dataloader"]
    assert isinstance(dataloader_cfg, dict)
    sampling_cfg = dataloader_cfg["sampling"]
    assert isinstance(sampling_cfg, dict)
    assert sampling_cfg["strategy"] == "ohem"

    rich_pooling_cfg = model_cfg["rich_pooling"]
    assert isinstance(rich_pooling_cfg, dict)
    assert rich_pooling_cfg["components"] == ["esm_cls", "mean", "attn", "max", "gated"]

    search_space = optimization_cfg["search_space"]
    assert isinstance(search_space, list)
    names = {entry["name"] for entry in search_space if isinstance(entry, dict)}
    assert "sampling_strategy" not in names
    assert {"ohem_warmup_epochs", "ohem_pool_multiplier", "ohem_cap_protein"} <= names


def test_v3_1_0428_rich_pooling_ablation_configs() -> None:
    expected_components = {
        "full.yaml": ["esm_cls", "mean", "attn", "max", "gated"],
        "no_cls.yaml": ["mean", "attn", "max", "gated"],
        "no_mean.yaml": ["esm_cls", "attn", "max", "gated"],
        "no_attn.yaml": ["esm_cls", "mean", "max", "gated"],
        "no_max.yaml": ["esm_cls", "mean", "attn", "gated"],
        "no_gated.yaml": ["esm_cls", "mean", "attn", "max"],
    }

    config_dir = Path("configs/v3-1/0428")
    assert {path.name for path in config_dir.glob("*.yaml")} == set(expected_components)
    for filename, components in expected_components.items():
        config = load_config(config_dir / filename)
        run_cfg = config["run_config"]
        data_cfg = config["data_config"]
        model_cfg = config["model_config"]
        assert isinstance(run_cfg, dict)
        assert isinstance(data_cfg, dict)
        assert isinstance(model_cfg, dict)

        run_id = filename.removesuffix(".yaml")
        assert run_cfg["stages"] == ["train", "evaluate"]
        assert run_cfg["train_run_id"] == run_id
        assert run_cfg["eval_run_id"] == run_id
        assert "optimization" not in config

        dataloader_cfg = data_cfg["dataloader"]
        assert isinstance(dataloader_cfg, dict)
        sampling_cfg = dataloader_cfg["sampling"]
        assert isinstance(sampling_cfg, dict)
        assert sampling_cfg["strategy"] == "ohem"

        rich_pooling_cfg = model_cfg["rich_pooling"]
        assert isinstance(rich_pooling_cfg, dict)
        assert rich_pooling_cfg["components"] == components


def test_v3_topology_finetune_uses_fixed_threshold_scheduler_and_patience() -> None:
    config = load_config("configs/v3.yaml")
    topology_cfg = config["topology_finetune"]
    assert isinstance(topology_cfg, dict)

    decision_threshold = topology_cfg["decision_threshold"]
    assert isinstance(decision_threshold, dict)
    assert decision_threshold == {"mode": "fixed", "value": 0.5}
    assert topology_cfg["early_stopping_patience"] == 8

    loss_weight_schedule = topology_cfg["loss_weight_schedule"]
    assert isinstance(loss_weight_schedule, dict)
    assert loss_weight_schedule == {
        "warmup_epochs": 5,
        "ramp_epochs": 5,
        "schedule": "linear",
    }

    losses_cfg = topology_cfg["losses"]
    assert isinstance(losses_cfg, dict)
    assert losses_cfg["rd_loss_form"] == "log_ratio_huber"

    normalization_cfg = topology_cfg["loss_normalization"]
    assert isinstance(normalization_cfg, dict)
    assert normalization_cfg == {
        "enabled": True,
        "ema_decay": 0.95,
        "clip_value": 5.0,
    }

    gradnorm_cfg = topology_cfg["gradnorm"]
    assert isinstance(gradnorm_cfg, dict)
    assert gradnorm_cfg == {
        "enabled": True,
        "alpha": 0.5,
        "learning_rate": 0.02,
        "min_weight": 0.2,
        "max_weight": 5.0,
    }


def test_0407_ablation_configs_use_warm_start_val_loss_recipe() -> None:
    expected_epochs = {
        "ws_n20.yaml": 12,
        "ws_n30.yaml": 12,
        "ws_n40.yaml": 20,
        "ws_n50.yaml": 20,
    }
    for config_path in sorted(Path("configs/v3/ablations/0407").glob("ws_n*.yaml")):
        config = load_config(config_path)
        topology_cfg = config["topology_finetune"]
        assert isinstance(topology_cfg, dict)

        decision_threshold = topology_cfg["decision_threshold"]
        assert isinstance(decision_threshold, dict)
        assert topology_cfg["init_mode"] == "warm_start"
        assert topology_cfg["epochs"] == expected_epochs[config_path.name]
        assert topology_cfg["pair_batch_size"] == 32
        assert topology_cfg["internal_validation_inference_batch_size"] == 128
        assert decision_threshold == {"mode": "fixed", "value": 0.5}
        assert topology_cfg["monitor_metric"] == "val_loss"
        assert topology_cfg["early_stopping_patience"] == 4

        optimizer_cfg = topology_cfg["optimizer"]
        assert isinstance(optimizer_cfg, dict)
        assert optimizer_cfg == {"lr": 6.0e-6, "weight_decay": 6.0e-3}

        loss_weight_schedule = topology_cfg["loss_weight_schedule"]
        assert isinstance(loss_weight_schedule, dict)
        assert loss_weight_schedule == {
            "warmup_epochs": 0,
            "ramp_epochs": 0,
            "schedule": "linear",
        }

        losses_cfg = topology_cfg["losses"]
        assert isinstance(losses_cfg, dict)
        assert losses_cfg["alpha"] == pytest.approx(0.35)
        assert losses_cfg["beta"] == pytest.approx(0.45)
        assert losses_cfg["gamma"] == pytest.approx(0.15)
        assert losses_cfg["delta"] == pytest.approx(0.15)
        assert losses_cfg["histogram_sigma"] == pytest.approx(1.25)
        assert losses_cfg["degree_bins"] == 48
        assert losses_cfg["clustering_bins"] == 64

        topology_eval_cfg = config["topology_evaluate"]
        assert isinstance(topology_eval_cfg, dict)
        assert topology_eval_cfg["inference_batch_size"] == 128


def test_0426_large_n_ablation_configs_disable_finetune_clustering_mmd() -> None:
    expected_node_sizes = {
        "ws_n60.yaml": 60,
        "ws_n80.yaml": 80,
        "ws_n100.yaml": 100,
    }

    for filename, node_size in expected_node_sizes.items():
        config_path = Path("configs/v3/ablations/0426") / filename
        config = load_config(config_path)
        run_cfg = config["run_config"]
        topology_cfg = config["topology_finetune"]
        topology_eval_cfg = config["topology_evaluate"]
        assert isinstance(run_cfg, dict)
        assert isinstance(topology_cfg, dict)
        assert isinstance(topology_eval_cfg, dict)

        run_id = f"ws_n{node_size}"
        assert run_cfg["topology_finetune_run_id"] == run_id
        assert run_cfg["eval_run_id"] == run_id
        assert run_cfg["topology_eval_run_id"] == run_id
        assert topology_cfg["subgraph_node_range"] == [node_size, node_size]
        assert topology_cfg["compute_clustering_mmd"] is False
        assert topology_cfg["internal_validation_compute_clustering_mmd"] is False
        assert topology_cfg["pair_batch_size"] == 32
        assert topology_cfg["subgraphs_per_forward"] == 1
        assert topology_eval_cfg["inference_batch_size"] == 128
        assert "compute_clustering_mmd" not in topology_eval_cfg


def test_0426_chunked_backward_configs_cover_large_and_range_n() -> None:
    for filename in ("ws_n80.yaml", "ws_n100.yaml"):
        config = load_config(Path("configs/v3/ablations/0426") / filename)
        topology_cfg = config["topology_finetune"]
        assert isinstance(topology_cfg, dict)
        assert topology_cfg["chunked_backward"] is True
        assert topology_cfg["compute_clustering_mmd"] is False

    expected_ranges = {
        "ws_range_n20_60.yaml": [20, 60],
        "ws_range_n40_80.yaml": [40, 80],
        "ws_range_n20_100.yaml": [20, 100],
    }
    for filename, node_range in expected_ranges.items():
        config = load_config(Path("configs/v3/ablations/0426") / filename)
        run_cfg = config["run_config"]
        topology_cfg = config["topology_finetune"]
        assert isinstance(run_cfg, dict)
        assert isinstance(topology_cfg, dict)

        run_id = filename.removesuffix(".yaml")
        assert run_cfg["topology_finetune_run_id"] == run_id
        assert run_cfg["eval_run_id"] == run_id
        assert run_cfg["topology_eval_run_id"] == run_id
        assert topology_cfg["subgraph_node_range"] == node_range
        assert topology_cfg["chunked_backward"] is True
        assert topology_cfg["compute_clustering_mmd"] is False
        assert topology_cfg["internal_validation_compute_clustering_mmd"] is False
