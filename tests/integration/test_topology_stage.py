"""Integration tests for the topology evaluation stage."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import src.pipeline.stages.topology_evaluate as topology_stage
import torch
from src.pipeline.runtime import DistributedContext
from src.pipeline.stages.evaluate import run_evaluation_stage
from src.pipeline.stages.topology_evaluate import run_topology_evaluation_stage
from src.pipeline.stages.train import build_model
from src.utils.config import ConfigDict
from src.utils.data_io import build_dataloaders
from tests.runtime_helpers import build_stage_runtime
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


def _build_topology_config(tmp_path: Path) -> ConfigDict:
    benchmark_root = tmp_path / "benchmark"
    processed_dir = benchmark_root / "human" / "BFS"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "human_train_ppi.txt"
    valid_path = processed_dir / "human_val_ppi.txt"
    test_path = processed_dir / "human_test_ppi.txt"
    all_test_path = processed_dir / "all_test_ppi.txt"

    _write_split(train_path, [("P1", "P2", 1), ("P1", "P3", 0), ("P2", "P3", 1)])
    _write_split(valid_path, [("P1", "P2", 1), ("P1", "P3", 0)])
    _write_split(test_path, [("P1", "P2", 1), ("P2", "P3", 1)])
    _write_split(all_test_path, [("P1", "P2", 1), ("P1", "P3", 0), ("P2", "P3", 1)])

    gt_graph = cast(object, __import__("networkx").Graph())
    gt_graph.add_edges_from([("P1", "P2"), ("P2", "P3")])
    with (processed_dir / "human_test_graph.pkl").open("wb") as handle:
        pickle.dump(gt_graph, handle)
    with (processed_dir / "test_sampled_nodes.pkl").open("wb") as handle:
        pickle.dump({3: [["P1", "P2", "P3"]]}, handle)
    with (processed_dir / "human_BFS_split.pkl").open("wb") as handle:
        pickle.dump({"train": {"P1", "P2", "P3"}, "test": {"P1", "P2", "P3"}}, handle)

    cache_dir = tmp_path / "cache"
    _write_embedding_cache(
        cache_dir=cache_dir,
        embeddings={
            "P1": torch.ones((2, 4), dtype=torch.float32),
            "P2": torch.full((2, 4), 2.0, dtype=torch.float32),
            "P3": torch.full((2, 4), 3.0, dtype=torch.float32),
        },
        input_dim=4,
        max_sequence_length=8,
    )

    baselines_path = tmp_path / "baselines.json"
    baselines_path.write_text(
        json.dumps(
            {
                "source": "integration-test",
                "rows": [
                    {
                        "category": "Seq. Sim.",
                        "model": "SPRINT",
                        "metrics": {
                            "BFS": {
                                "graph_sim": 0.2,
                                "relative_density": 1.1,
                                "deg_dist_mmd": 5.0,
                                "cc_mmd": 4.0,
                                "laplacian_eigen_mmd": 3.0,
                            },
                            "DFS": {
                                "graph_sim": 0.1,
                                "relative_density": 1.2,
                                "deg_dist_mmd": 6.0,
                                "cc_mmd": 5.0,
                                "laplacian_eigen_mmd": 4.0,
                            },
                            "RANDOM_WALK": {
                                "graph_sim": 0.3,
                                "relative_density": 1.0,
                                "deg_dist_mmd": 4.0,
                                "cc_mmd": 3.0,
                                "laplacian_eigen_mmd": 2.0,
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    return {
        "run_config": {
            "stages": ["topology_evaluate"],
            "seed": 7,
            "train_run_id": "unused_train",
            "adapt_run_id": None,
            "eval_run_id": None,
            "topology_eval_run_id": "topology_case",
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
            "optimizer": {"type": "adamw", "lr": 1e-3},
            "scheduler": {"type": "none"},
            "loss": {"type": "bce_with_logits", "pos_weight": 1.0, "label_smoothing": 0.0},
            "strategy": {"type": "none"},
            "domain_adaptation": {"enabled": False, "method": "none", "target_split": "test"},
        },
        "evaluate": {
            "metrics": ["auprc", "auroc"],
            "decision_threshold": {"mode": "fixed", "value": 0.5},
        },
        "topology_evaluate": {
            "decision_threshold": {"mode": "fixed", "value": 0.5},
            "save_pair_predictions": True,
            "report_baselines": str(baselines_path),
            "inference_batch_size": 2,
        },
    }


def test_run_topology_evaluation_stage_writes_expected_artifacts(tmp_path: Path) -> None:
    config = _build_topology_config(tmp_path)
    previous_cwd = Path.cwd()
    try:
        Path(config["run_config"]["load_checkpoint_path"]).parent.mkdir(parents=True, exist_ok=True)  # type: ignore[index]
        model = build_model(config)
        torch.save(model.state_dict(), Path(str(config["run_config"]["load_checkpoint_path"])))  # type: ignore[index]
        dataloaders = build_dataloaders(config=config)
        checkpoint_path = Path(str(config["run_config"]["load_checkpoint_path"]))  # type: ignore[index]
        __import__("os").chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_evaluate": "topology_case"},
        )
        run_topology_evaluation_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        __import__("os").chdir(previous_cwd)

    log_dir = tmp_path / "logs" / "v3" / "topology_evaluate" / "topology_case"
    assert (log_dir / "all_test_ppi_pred.txt").exists()
    assert (log_dir / "topology_metrics.json").exists()
    assert (log_dir / "topology_metrics.csv").exists()
    assert (log_dir / "graph_eval_results.pkl").exists()

    log_text = (log_dir / "log.log").read_text(encoding="utf-8")
    assert "Decision Threshold" in log_text
    assert "fixed" in log_text
    assert "0.500" in log_text


def test_tccig_graph_assembly_evaluate_uses_all_test_universe_and_writes_diagnostics(
    tmp_path: Path,
) -> None:
    config = _build_topology_config(tmp_path)
    cast(ConfigDict, config["run_config"])["stages"] = ["evaluate"]
    cast(ConfigDict, config["run_config"])["eval_run_id"] = "graph_eval_case"
    cast(ConfigDict, config["model_config"])["model"] = "tccig"
    cast(ConfigDict, config["evaluate"])["mode"] = "graph_assembly"

    class _GraphAssemblyModel(torch.nn.Module):
        def forward(
            self,
            emb_a: torch.Tensor,
            emb_b: torch.Tensor,
            **_: object,
        ) -> dict[str, torch.Tensor]:
            pair_sum = emb_a.mean(dim=(1, 2)) + emb_b.mean(dim=(1, 2))
            probabilities = torch.where(
                pair_sum <= 3.5,
                torch.full_like(pair_sum, 0.7),
                torch.full_like(pair_sum, 0.6),
            )
            return {"logits": torch.logit(probabilities)}

        def forward_graph(self, **_: object) -> dict[str, torch.Tensor]:
            return {}

        def encode_graph_nodes(
            self,
            *,
            protein_embeddings: torch.Tensor,
            protein_lengths: torch.Tensor,
        ) -> torch.Tensor:
            del protein_lengths
            return protein_embeddings.mean(dim=1)

        def edge_budget_from_node_embeddings(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_count: int,
        ) -> torch.Tensor:
            del node_embeddings, candidate_count
            return torch.tensor(2.0)

        def decode_graph_candidates(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_pairs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del node_embeddings
            scores = {
                (0, 1): 0.9,
                (0, 2): 0.1,
                (1, 2): 0.8,
            }
            probabilities = torch.tensor(
                [
                    scores[(int(source), int(target))]
                    for source, target in candidate_pairs.t().tolist()
                ],
                dtype=torch.float32,
            )
            return {"edge_probabilities": probabilities}

    model = _GraphAssemblyModel()
    checkpoint_path = Path(str(cast(ConfigDict, config["run_config"])["load_checkpoint_path"]))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    dataloaders = build_dataloaders(config=config)

    previous_cwd = Path.cwd()
    try:
        __import__("os").chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"evaluate": "graph_eval_case"},
        )
        metrics = run_evaluation_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        __import__("os").chdir(previous_cwd)

    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)

    log_dir = tmp_path / "logs" / "tccig" / "evaluate" / "graph_eval_case"
    evaluate_csv = log_dir / "evaluate.csv"
    diagnostics_path = log_dir / "graph_assembly_diagnostics.json"
    assert evaluate_csv.exists()
    assert diagnostics_path.exists()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["assembly_rule"] == "validation_density"
    assert diagnostics["n_nodes"] == 3
    assert diagnostics["full_pair_count"] == 3
    assert diagnostics["candidate_count"] == 3
    assert diagnostics["selected_edges"] == 2
    assert diagnostics["edge_budget"] == pytest.approx(2.0)
    assert diagnostics["m_hat"] == pytest.approx(2.0)
    assert diagnostics["m_hat_per_candidate"] == pytest.approx(2.0 / 3.0, abs=1.0e-3)
    assert diagnostics["m_hat_per_full_pair"] == pytest.approx(2.0 / 3.0, abs=1.0e-3)
    assert diagnostics["threshold_mode"] == "validation_mcc"
    assert diagnostics["threshold_value"] == pytest.approx(0.7)


def test_tccig_pairwise_evaluate_uses_validation_calibrated_threshold(
    tmp_path: Path,
) -> None:
    config = _build_topology_config(tmp_path)
    data_cfg = cast(ConfigDict, config["data_config"])
    benchmark_cfg = cast(ConfigDict, data_cfg["benchmark"])
    processed_dir = Path(str(benchmark_cfg["processed_dir"]))
    _write_split(
        processed_dir / "human_val_ppi.txt",
        [("P1", "P2", 1), ("P1", "P3", 0), ("P2", "P3", 0)],
    )
    _write_split(
        processed_dir / "human_test_ppi.txt",
        [("P1", "P2", 1), ("P1", "P3", 0)],
    )
    cast(ConfigDict, config["run_config"])["stages"] = ["evaluate"]
    cast(ConfigDict, config["run_config"])["eval_run_id"] = "tccig_pairwise_threshold_case"
    cast(ConfigDict, config["model_config"])["model"] = "tccig"
    cast(ConfigDict, config["evaluate"])["mode"] = "pairwise"
    cast(ConfigDict, config["evaluate"])["tccig_pairwise_threshold"] = {
        "mode": "validation_mcc"
    }

    class _PairwiseTCCIGModel(torch.nn.Module):
        def forward(
            self,
            emb_a: torch.Tensor,
            emb_b: torch.Tensor,
            **_: object,
        ) -> dict[str, torch.Tensor]:
            pair_sum = emb_a.mean(dim=(1, 2)) + emb_b.mean(dim=(1, 2))
            probabilities = torch.where(
                pair_sum <= 3.5,
                torch.full_like(pair_sum, 0.7),
                torch.where(
                    pair_sum <= 4.5,
                    torch.full_like(pair_sum, 0.6),
                    torch.full_like(pair_sum, 0.2),
                ),
            )
            return {"logits": torch.logit(probabilities)}

    model = _PairwiseTCCIGModel()
    checkpoint_path = Path(str(cast(ConfigDict, config["run_config"])["load_checkpoint_path"]))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    dataloaders = build_dataloaders(config=config)

    previous_cwd = Path.cwd()
    try:
        __import__("os").chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"evaluate": "tccig_pairwise_threshold_case"},
        )
        metrics = run_evaluation_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        __import__("os").chdir(previous_cwd)

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)

    log_text = (
        tmp_path
        / "logs"
        / "tccig"
        / "evaluate"
        / "tccig_pairwise_threshold_case"
        / "log.log"
    ).read_text(encoding="utf-8")
    assert "tccig_pairwise_threshold" in log_text
    assert "Threshold Value: 0.700" in log_text


def test_tccig_graph_assembly_evaluate_clamps_infinite_edge_budget(
    tmp_path: Path,
) -> None:
    config = _build_topology_config(tmp_path)
    cast(ConfigDict, config["run_config"])["stages"] = ["evaluate"]
    cast(ConfigDict, config["run_config"])["eval_run_id"] = "graph_eval_inf_budget"
    cast(ConfigDict, config["model_config"])["model"] = "tccig"
    cast(ConfigDict, config["evaluate"])["mode"] = "graph_assembly"

    class _InfiniteBudgetGraphAssemblyModel(torch.nn.Module):
        def forward(
            self,
            emb_a: torch.Tensor,
            emb_b: torch.Tensor,
            **_: object,
        ) -> dict[str, torch.Tensor]:
            pair_sum = emb_a.mean(dim=(1, 2)) + emb_b.mean(dim=(1, 2))
            return {"logits": torch.logit(torch.full_like(pair_sum, 0.5))}

        def forward_graph(self, **_: object) -> dict[str, torch.Tensor]:
            return {}

        def encode_graph_nodes(
            self,
            *,
            protein_embeddings: torch.Tensor,
            protein_lengths: torch.Tensor,
        ) -> torch.Tensor:
            del protein_lengths
            return protein_embeddings.mean(dim=1)

        def edge_budget_from_node_embeddings(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_count: int,
        ) -> torch.Tensor:
            del node_embeddings, candidate_count
            return torch.tensor(float("inf"))

        def decode_graph_candidates(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_pairs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del node_embeddings
            return {"edge_probabilities": torch.full((candidate_pairs.size(1),), 0.5)}

    model = _InfiniteBudgetGraphAssemblyModel()
    checkpoint_path = Path(str(cast(ConfigDict, config["run_config"])["load_checkpoint_path"]))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    dataloaders = build_dataloaders(config=config)

    previous_cwd = Path.cwd()
    try:
        __import__("os").chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"evaluate": "graph_eval_inf_budget"},
        )
        run_evaluation_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        __import__("os").chdir(previous_cwd)

    diagnostics_path = (
        tmp_path
        / "logs"
        / "tccig"
        / "evaluate"
        / "graph_eval_inf_budget"
        / "graph_assembly_diagnostics.json"
    )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["candidate_count"] == 3
    assert diagnostics["selected_edges"] == 3


def test_tccig_graph_assembly_scores_candidates_and_selects_top_budget(
    tmp_path: Path,
) -> None:
    config = _build_topology_config(tmp_path)
    processed_dir = Path(str(config["data_config"]["benchmark"]["processed_dir"]))  # type: ignore[index]
    bundle = topology_stage._build_topology_loader(
        config=config,
        split_path=processed_dir / "all_test_ppi.txt",
    )

    class _GraphAssemblyModel(torch.nn.Module):
        def forward_graph(self, **_: object) -> dict[str, torch.Tensor]:
            return {}

        def encode_graph_nodes(
            self,
            *,
            protein_embeddings: torch.Tensor,
            protein_lengths: torch.Tensor,
        ) -> torch.Tensor:
            del protein_lengths
            return protein_embeddings.mean(dim=1)

        def edge_budget_from_node_embeddings(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_count: int,
        ) -> torch.Tensor:
            del node_embeddings, candidate_count
            return torch.tensor(2.0)

        def decode_graph_candidates(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_pairs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del node_embeddings
            scores = {
                (0, 1): 0.9,
                (0, 2): 0.1,
                (1, 2): 0.8,
            }
            probabilities = torch.tensor(
                [
                    scores[(int(source), int(target))]
                    for source, target in candidate_pairs.t().tolist()
                ],
                dtype=torch.float32,
            )
            return {"edge_probabilities": probabilities}

    predictions = topology_stage._predict_tccig_graph_assembly_labels(
        config=config,
        model=_GraphAssemblyModel(),
        dataset=bundle.dataset,
        records=bundle.records,
        device=torch.device("cpu"),
        accelerator=build_stage_runtime(config).accelerator,
    )

    assert predictions == [1, 0, 1]


def test_tccig_graph_assembly_uses_batched_protein_encoder(tmp_path: Path) -> None:
    config = _build_topology_config(tmp_path)
    cast(ConfigDict, config["topology_evaluate"])["tccig"] = {
        "candidate_batch_size": 8,
        "node_batch_size": 2,
    }
    processed_dir = Path(str(config["data_config"]["benchmark"]["processed_dir"]))  # type: ignore[index]
    bundle = topology_stage._build_topology_loader(
        config=config,
        split_path=processed_dir / "all_test_ppi.txt",
    )

    class _BatchedGraphAssemblyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.observed_batch_size: int | None = None

        def forward_graph(self, **_: object) -> dict[str, torch.Tensor]:
            return {}

        def encode_graph_nodes(self, **_: object) -> torch.Tensor:
            raise AssertionError("graph assembly should use the richer batched encoder")

        def encode_proteins(self, **_: object) -> dict[str, torch.Tensor]:
            raise AssertionError("graph assembly should not build one global padded tensor")

        def encode_proteins_batched(
            self,
            *,
            protein_embeddings: Sequence[torch.Tensor],
            device: torch.device,
            batch_size: int,
        ) -> dict[str, torch.Tensor]:
            self.observed_batch_size = batch_size
            node_embeddings = torch.stack(
                [embedding.to(device).mean(dim=0) for embedding in protein_embeddings],
                dim=0,
            )
            return {"node": node_embeddings}

        def edge_budget_from_node_embeddings(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_count: int,
        ) -> torch.Tensor:
            del node_embeddings, candidate_count
            return torch.tensor(2.0)

        def decode_graph_candidates(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_pairs: torch.Tensor,
            encoded: Mapping[str, torch.Tensor] | None = None,
        ) -> dict[str, torch.Tensor]:
            del node_embeddings, encoded
            scores = {
                (0, 1): 0.9,
                (0, 2): 0.1,
                (1, 2): 0.8,
            }
            probabilities = torch.tensor(
                [
                    scores[(int(source), int(target))]
                    for source, target in candidate_pairs.t().tolist()
                ],
                dtype=torch.float32,
            )
            return {"edge_probabilities": probabilities}

    model = _BatchedGraphAssemblyModel()
    predictions = topology_stage._predict_tccig_graph_assembly_labels(
        config=config,
        model=model,
        dataset=bundle.dataset,
        records=bundle.records,
        device=torch.device("cpu"),
        accelerator=build_stage_runtime(config).accelerator,
    )

    assert model.observed_batch_size == 2
    assert predictions == [1, 0, 1]


def test_tccig_topology_evaluate_writes_graph_assembly_diagnostics(tmp_path: Path) -> None:
    config = _build_topology_config(tmp_path)
    cast(ConfigDict, config["model_config"])["model"] = "tccig"
    processed_dir = Path(str(cast(ConfigDict, config["data_config"])["benchmark"]["processed_dir"]))
    config["tccig_train"] = {
        "supervision_train_dataset": str(processed_dir / "human_train_ppi.txt"),
        "supervision_valid_dataset": str(processed_dir / "human_val_ppi.txt"),
    }

    class _GraphAssemblyModel(torch.nn.Module):
        def forward_graph(self, **_: object) -> dict[str, torch.Tensor]:
            return {}

        def encode_graph_nodes(
            self,
            *,
            protein_embeddings: torch.Tensor,
            protein_lengths: torch.Tensor,
        ) -> torch.Tensor:
            del protein_lengths
            return protein_embeddings.mean(dim=1)

        def edge_budget_from_node_embeddings(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_count: int,
        ) -> torch.Tensor:
            del node_embeddings, candidate_count
            return torch.tensor(2.0)

        def decode_graph_candidates(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_pairs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del node_embeddings
            probabilities = torch.tensor([0.9, 0.1, 0.8], dtype=torch.float32)
            return {"edge_probabilities": probabilities[: candidate_pairs.size(1)]}

    model = _GraphAssemblyModel()
    checkpoint_path = Path(str(cast(ConfigDict, config["run_config"])["load_checkpoint_path"]))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    dataloaders = build_dataloaders(config=config)

    previous_cwd = Path.cwd()
    try:
        __import__("os").chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_evaluate": "topology_case"},
        )
        run_topology_evaluation_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        __import__("os").chdir(previous_cwd)

    log_dir = tmp_path / "logs" / "tccig" / "topology_evaluate" / "topology_case"
    diagnostics_path = log_dir / "graph_assembly_diagnostics.json"
    assert diagnostics_path.exists()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["assembly_rule"] == "validation_density"
    assert diagnostics["n_nodes"] == 3
    assert diagnostics["full_pair_count"] == 3
    assert diagnostics["candidate_count"] == 3
    assert diagnostics["selected_edges"] == 1
    assert diagnostics["m_hat_per_candidate"] == pytest.approx(2.0 / 3.0, abs=1.0e-3)
    assert diagnostics["m_hat_per_full_pair"] == pytest.approx(2.0 / 3.0, abs=1.0e-3)
    assert diagnostics["edge_budget"] == pytest.approx(1.0)

    metrics_payload = json.loads((log_dir / "topology_metrics.json").read_text(encoding="utf-8"))
    assert metrics_payload["graph_assembly"]["assembly_rule"] == "validation_density"
    assert metrics_payload["decision_rule"] == "validation_density"
    assert metrics_payload["fixed_threshold_diagnostic"] == {
        "mode": "fixed",
        "value": 0.5,
    }
    debug_assemblies = metrics_payload["debug_assemblies"]
    assert set(debug_assemblies) == {
        "official",
        "model_m_hat",
        "validation_density",
        "oracle_test_density",
    }
    assert debug_assemblies["official"]["diagnostic_only"] is False
    assert debug_assemblies["model_m_hat"]["diagnostic_only"] is True
    assert debug_assemblies["model_m_hat"]["budget"] == pytest.approx(2.0)
    assert debug_assemblies["validation_density"]["diagnostic_only"] is True
    assert debug_assemblies["validation_density"]["source_density"] == pytest.approx(
        1.0 / 3.0,
        abs=1.0e-3,
    )
    assert debug_assemblies["validation_density"]["budget"] == pytest.approx(1.0)
    assert debug_assemblies["validation_density"]["selected_edges"] == 1
    assert debug_assemblies["oracle_test_density"]["diagnostic_only"] is True
    assert debug_assemblies["oracle_test_density"]["source_density"] == pytest.approx(
        2.0 / 3.0,
        abs=1.0e-3,
    )
    assert debug_assemblies["oracle_test_density"]["budget"] == pytest.approx(2.0)
    assert "summary" in debug_assemblies["validation_density"]


def test_tccig_graph_forward_topology_metrics_skip_non_main_distributed_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_topology_config(tmp_path)
    checkpoint_path = Path(str(cast(ConfigDict, config["run_config"])["load_checkpoint_path"]))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    dataloaders = build_dataloaders(config=config)

    class _GraphAssemblyModel(torch.nn.Module):
        def forward_graph(self, **_: object) -> dict[str, torch.Tensor]:
            return {}

        def encode_graph_nodes(
            self,
            *,
            protein_embeddings: torch.Tensor,
            protein_lengths: torch.Tensor,
        ) -> torch.Tensor:
            del protein_lengths
            return protein_embeddings.mean(dim=1)

        def edge_budget_from_node_embeddings(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_count: int,
        ) -> torch.Tensor:
            del node_embeddings, candidate_count
            return torch.tensor(2.0)

        def decode_graph_candidates(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_pairs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del node_embeddings
            return {"edge_probabilities": torch.ones(candidate_pairs.size(1))}

    model = _GraphAssemblyModel()
    torch.save(model.state_dict(), checkpoint_path)

    def _fail_graph_metrics(**_: object) -> dict[str, object]:
        raise AssertionError("non-main graph-forward rank should skip graph metrics")

    monkeypatch.setattr(topology_stage, "evaluate_predicted_graph", _fail_graph_metrics)

    previous_cwd = Path.cwd()
    try:
        __import__("os").chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_evaluate": "graph_forward_non_main"},
            distributed=DistributedContext(
                ddp_enabled=True,
                is_distributed=True,
                rank=1,
                local_rank=1,
                world_size=2,
            ),
        )
        summary = run_topology_evaluation_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        __import__("os").chdir(previous_cwd)

    assert summary == {}


def test_tccig_graph_assembly_preserves_self_edge_records_as_negatives(
    tmp_path: Path,
) -> None:
    config = _build_topology_config(tmp_path)
    processed_dir = Path(str(config["data_config"]["benchmark"]["processed_dir"]))  # type: ignore[index]
    bundle = topology_stage._build_topology_loader(
        config=config,
        split_path=processed_dir / "all_test_ppi.txt",
    )

    class _GraphAssemblyModel(torch.nn.Module):
        def forward_graph(self, **_: object) -> dict[str, torch.Tensor]:
            return {}

        def encode_graph_nodes(
            self,
            *,
            protein_embeddings: torch.Tensor,
            protein_lengths: torch.Tensor,
        ) -> torch.Tensor:
            del protein_lengths
            return protein_embeddings.mean(dim=1)

        def edge_budget_from_node_embeddings(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_count: int,
        ) -> torch.Tensor:
            del node_embeddings
            assert candidate_count == 2
            return torch.tensor(1.0)

        def decode_graph_candidates(
            self,
            *,
            node_embeddings: torch.Tensor,
            candidate_pairs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            del node_embeddings
            assert not torch.any(candidate_pairs[0] == candidate_pairs[1])
            scores = {
                (0, 1): 0.2,
                (1, 2): 0.8,
            }
            probabilities = torch.tensor(
                [
                    scores[(int(source), int(target))]
                    for source, target in candidate_pairs.t().tolist()
                ],
                dtype=torch.float32,
            )
            return {"edge_probabilities": probabilities}

    predictions = topology_stage._predict_tccig_graph_assembly_labels(
        config=config,
        model=_GraphAssemblyModel(),
        dataset=bundle.dataset,
        records=[("P1", "P1"), ("P1", "P2"), ("P2", "P3")],
        device=torch.device("cpu"),
        accelerator=build_stage_runtime(config).accelerator,
    )

    assert predictions == [0, 0, 1]


def _fake_sharded_topology_result(node_sizes: tuple[int, ...]) -> dict[str, object]:
    details: dict[str, dict[int, list[float] | float]] = {
        "graph_sim": {},
        "relative_density": {},
        "deg_dist_mmd": {},
        "cc_mmd": {},
        "laplacian_eigen_mmd": {},
    }
    per_node_size: dict[int, dict[str, float | int]] = {}
    for node_size in node_sizes:
        graph_sim_values = [node_size / 100.0]
        relative_density_values = [1.0 + (node_size / 1000.0)]
        deg_dist_mmd = node_size / 10.0
        cc_mmd = node_size / 20.0
        laplacian_eigen_mmd = node_size / 40.0
        details["graph_sim"][node_size] = graph_sim_values
        details["relative_density"][node_size] = relative_density_values
        details["deg_dist_mmd"][node_size] = deg_dist_mmd
        details["cc_mmd"][node_size] = cc_mmd
        details["laplacian_eigen_mmd"][node_size] = laplacian_eigen_mmd
        per_node_size[node_size] = {
            "graph_count": 1,
            "graph_sim": graph_sim_values[0],
            "relative_density": relative_density_values[0],
            "deg_dist_mmd": deg_dist_mmd,
            "cc_mmd": cc_mmd,
            "laplacian_eigen_mmd": laplacian_eigen_mmd,
        }
    summary = {
        "graph_sim": float(np.mean([values[0] for values in details["graph_sim"].values()])),
        "relative_density": float(
            np.mean([values[0] for values in details["relative_density"].values()])
        ),
        "deg_dist_mmd": float(np.mean(list(details["deg_dist_mmd"].values()))),
        "cc_mmd": float(np.mean(list(details["cc_mmd"].values()))),
        "laplacian_eigen_mmd": float(np.mean(list(details["laplacian_eigen_mmd"].values()))),
    }
    return {
        "details": details,
        "summary": summary,
        "per_node_size": per_node_size,
    }


def test_run_topology_evaluation_stage_shards_graph_metrics_under_ddp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_topology_config(tmp_path)
    processed_dir = (
        Path(str(config["data_config"]["benchmark"]["processed_dir"]))  # type: ignore[index]
    )
    with (processed_dir / "test_sampled_nodes.pkl").open("wb") as handle:
        pickle.dump(
            {
                20: [["P1"]],
                40: [["P2"]],
                60: [["P3"]],
                80: [["P4"]],
            },
            handle,
        )

    checkpoint_path = Path(str(config["run_config"]["load_checkpoint_path"]))  # type: ignore[index]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(config)
    torch.save(model.state_dict(), checkpoint_path)
    dataloaders = build_dataloaders(config=config)

    monkeypatch.setattr(
        topology_stage,
        "_build_topology_loader",
        lambda **_: (
            cast(DataLoader[dict[str, object]], dataloaders["test"]),
            [("P1", "P2"), ("P1", "P3"), ("P2", "P3")],
            3,
        ),
    )
    monkeypatch.setattr(topology_stage, "_predict_topology_labels", lambda **_: [1, 0, 1])
    monkeypatch.setattr(topology_stage, "_resolve_decision_threshold", lambda **_: (0.5, "fixed"))

    observed_local_node_sizes: list[tuple[int, ...]] = []

    def _record_local_graph_eval(
        *,
        pred_graph: object,
        gt_graph: object,
        test_graph_nodes: object,
    ) -> dict[str, object]:
        del pred_graph, gt_graph
        assert isinstance(test_graph_nodes, dict)
        node_sizes = tuple(sorted(int(node_size) for node_size in test_graph_nodes))
        observed_local_node_sizes.append(node_sizes)
        return _fake_sharded_topology_result(node_sizes)

    def _fake_all_gather_object(
        object_list: list[object | None],
        local_result: object,
    ) -> None:
        object_list[0] = _fake_sharded_topology_result((40, 80))
        object_list[1] = local_result

    monkeypatch.setattr(topology_stage, "evaluate_predicted_graph", _record_local_graph_eval)
    monkeypatch.setattr(topology_stage.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(topology_stage.dist, "all_gather_object", _fake_all_gather_object)

    previous_cwd = Path.cwd()
    try:
        __import__("os").chdir(tmp_path)
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_evaluate": "topology_sharded"},
            distributed=DistributedContext(
                ddp_enabled=True,
                is_distributed=True,
                rank=1,
                local_rank=1,
                world_size=2,
            ),
        )
        summary = run_topology_evaluation_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        __import__("os").chdir(previous_cwd)

    expected_summary = cast(
        dict[str, float],
        _fake_sharded_topology_result((20, 40, 60, 80))["summary"],
    )
    assert observed_local_node_sizes == [(20, 60)]
    assert summary == pytest.approx(expected_summary)


def test_run_topology_evaluation_stage_non_main_rank_computes_topology_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_topology_config(tmp_path)
    checkpoint_path = Path(str(config["run_config"]["load_checkpoint_path"]))  # type: ignore[index]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(config)
    torch.save(model.state_dict(), checkpoint_path)
    dataloaders = build_dataloaders(config=config)

    monkeypatch.setattr(
        topology_stage,
        "_build_topology_loader",
        lambda **_: (
            cast(DataLoader[dict[str, object]], dataloaders["test"]),
            [("P1", "P2"), ("P1", "P3"), ("P2", "P3")],
            3,
        ),
    )
    monkeypatch.setattr(topology_stage, "_predict_topology_labels", lambda **_: [1, 0, 1])

    previous_cwd = Path.cwd()
    try:
        __import__("os").chdir(tmp_path)
        distributed_context = DistributedContext(
            ddp_enabled=True,
            is_distributed=True,
            rank=1,
            local_rank=1,
            world_size=2,
        )
        runtime = build_stage_runtime(
            config,
            stage_run_ids={"topology_evaluate": "topology_non_main"},
            distributed=distributed_context,
        )
        summary = run_topology_evaluation_stage(
            runtime,
            model,
            cast(dict[str, DataLoader[dict[str, object]]], dataloaders),
            checkpoint_path=checkpoint_path,
        )
    finally:
        __import__("os").chdir(previous_cwd)

    assert "graph_sim" in summary
