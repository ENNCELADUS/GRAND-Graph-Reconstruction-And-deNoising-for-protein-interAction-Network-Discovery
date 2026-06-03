"""Integration tests for the standalone TCCIG PRING orchestrator."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx
from tccig.train import PairwiseScoreRequest, RefineRequest, TrainRefinerRequest, run_tccig_pipeline

HOOK_EVENTS: list[tuple[str, object]] = []


def _write_pairs(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for protein_a, protein_b, label in rows:
            handle.write(f"{protein_a}\t{protein_b}\t{label}\n")


def _write_tiny_pring_fixture(processed_dir: Path) -> None:
    _write_pairs(
        processed_dir / "human_train_ppi_ratio5_exclusive.txt",
        [("A", "B", 1), ("A", "A", 1), ("A", "C", 0)],
    )
    _write_pairs(
        processed_dir / "human_val_ppi_ratio5_exclusive.txt",
        [("A", "B", 1), ("B", "B", 1), ("A", "C", 0)],
    )
    _write_pairs(
        processed_dir / "human_test_ppi.txt",
        [("T1", "T2", 1), ("T1", "T1", 1), ("T2", "T3", 0)],
    )
    _write_pairs(
        processed_dir / "all_test_ppi.txt",
        [("T1", "T2", 1), ("T1", "T3", 0), ("T3", "T3", 1)],
    )

    graph = nx.Graph()
    graph.add_nodes_from(["T1", "T2", "T3"])
    graph.add_edge("T1", "T2")
    with (processed_dir / "human_test_graph.pkl").open("wb") as handle:
        pickle.dump(graph, handle)
    with (processed_dir / "test_sampled_nodes.pkl").open("wb") as handle:
        pickle.dump({2: [["T1", "T2"], ["T1", "T3"]]}, handle)


def fake_score_pairs(request: PairwiseScoreRequest) -> list[float]:
    """Fake external pairwise scorer that enforces label-free inputs."""
    assert all(not hasattr(pair, "label") for pair in request.pairs)
    assert all(pair.protein_a != pair.protein_b for pair in request.pairs)
    assert not hasattr(request, "human_test_graph")
    assert not hasattr(request, "test_sampled_nodes")
    HOOK_EVENTS.append(("score", request.split))
    return [0.9 if pair.protein_b in {"B", "T2"} else 0.1 for pair in request.pairs]


def fake_train_refiner(request: TrainRefinerRequest) -> dict[str, str]:
    """Fake trainer that sees train targets but no test graph artifacts."""
    assert request.train.loss_targets == [1, 0]
    assert request.train.candidate_labels == [1, 0]
    assert request.train.graph_edges == [("A", "B")]
    assert not hasattr(request, "human_test_graph")
    assert not hasattr(request, "test_sampled_nodes")
    HOOK_EVENTS.append(("train_refiner", len(request.train.pairs)))
    return {"state": "fake"}


def fake_predict_refined(request: RefineRequest) -> list[float]:
    """Fake refiner that receives pairwise graph inputs but not ground truth graphs."""
    assert all(not hasattr(pair, "label") for pair in request.pairs)
    assert not hasattr(request, "human_test_graph")
    assert not hasattr(request, "test_sampled_nodes")
    HOOK_EVENTS.append(("predict_refined", request.split))
    return list(request.pairwise_probabilities)


def test_tccig_orchestrator_keeps_pring_truth_out_of_model_inputs(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "PRING" / "human" / "BFS"
    _write_tiny_pring_fixture(processed_dir)
    HOOK_EVENTS.clear()

    result = run_tccig_pipeline(
        {
            "run": {"run_id": "tiny", "log_root": str(tmp_path / "logs")},
            "data": {"processed_dir": str(processed_dir)},
            "device": {
                "device": "cpu",
                "backend": "ddp",
                "mixed_precision": False,
            },
            "pairwise_scorer": {
                "target": f"{__name__}:fake_score_pairs",
            },
            "refiner": {
                "train_target": f"{__name__}:fake_train_refiner",
                "predict_target": f"{__name__}:fake_predict_refined",
            },
            "graph_selection": {
                "rules": [
                    {"type": "threshold", "value": 0.5},
                    {"type": "top_k", "k": 1},
                    {"type": "top_m", "m": 1},
                ]
            },
        }
    )

    assert ("score", "train") in HOOK_EVENTS
    assert ("score", "topology_test") in HOOK_EVENTS
    assert ("predict_refined", "topology_test") in HOOK_EVENTS
    assert result.manifest["self_pair_rows_dropped"] == {
        "train": 1,
        "validation": 1,
        "pairwise_test": 1,
        "topology_test": 1,
    }
    assert result.selected_rule["type"] in {"threshold", "top_k", "top_m"}

    topology_log_dir = tmp_path / "logs" / "tccig" / "topology_test" / "tiny"
    assert (topology_log_dir / "all_test_ppi_pred.txt").exists()
    assert (topology_log_dir / "topology_metrics.json").exists()

    scoring_manifest = tmp_path / "logs" / "tccig" / "score" / "tiny" / "train.json"
    assert scoring_manifest.exists()
    assert '"pair_count": 2' in scoring_manifest.read_text(encoding="utf-8")


def test_validation_selected_rule_is_reused_for_topology_test(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "PRING" / "human" / "BFS"
    _write_tiny_pring_fixture(processed_dir)
    HOOK_EVENTS.clear()

    result = run_tccig_pipeline(
        {
            "run": {"run_id": "top_m_case", "log_root": str(tmp_path / "logs")},
            "data": {"processed_dir": str(processed_dir)},
            "device": {"device": "cpu", "backend": "ddp", "mixed_precision": False},
            "pairwise_scorer": {"target": f"{__name__}:fake_score_pairs"},
            "refiner": {
                "train_target": f"{__name__}:fake_train_refiner",
                "predict_target": f"{__name__}:fake_predict_refined",
            },
            "graph_selection": {
                "rules": [
                    {"type": "threshold", "value": 0.95},
                    {"type": "top_m", "m": 1},
                ]
            },
        }
    )

    assert result.selected_rule["type"] == "top_m"
    assert result.pairwise_metrics["f1"] == 1.0
    assert result.pairwise_metrics["mcc"] == 1.0

    selected_rule_path = (
        tmp_path / "logs" / "tccig" / "validation" / "top_m_case" / "selected_rule.json"
    )
    selected_rule = json.loads(selected_rule_path.read_text(encoding="utf-8"))
    assert selected_rule["type"] == "top_m"
    assert selected_rule["m"] == 1

    prediction_path = (
        tmp_path / "logs" / "tccig" / "topology_test" / "top_m_case" / "all_test_ppi_pred.txt"
    )
    predicted_rows = prediction_path.read_text(encoding="utf-8").strip().splitlines()
    assert predicted_rows == ["T1\tT2\t1", "T1\tT3\t0"]


def test_tccig_runtime_wires_deepspeed_backend_without_launching_it(tmp_path: Path) -> None:
    processed_dir = tmp_path / "data" / "PRING" / "human" / "BFS"
    _write_tiny_pring_fixture(processed_dir)
    accelerator_calls: list[dict[str, object]] = []

    class FakeAccelerator:
        device = "cpu"

    def fake_build_accelerator(**kwargs: object) -> FakeAccelerator:
        accelerator_calls.append(dict(kwargs))
        return FakeAccelerator()

    run_tccig_pipeline(
        {
            "run": {"run_id": "runtime_case", "log_root": str(tmp_path / "logs")},
            "data": {"processed_dir": str(processed_dir)},
            "device": {
                "device": "cuda",
                "backend": "deepspeed",
                "mixed_precision": True,
                "find_unused_parameters": True,
            },
            "pairwise_scorer": {"target": f"{__name__}:fake_score_pairs"},
            "refiner": {
                "train_target": f"{__name__}:fake_train_refiner",
                "predict_target": f"{__name__}:fake_predict_refined",
            },
            "graph_selection": {"rules": [{"type": "threshold", "value": 0.5}]},
        },
        build_accelerator_fn=fake_build_accelerator,
    )

    assert accelerator_calls == [
        {
            "requested_device": "cuda",
            "backend": "deepspeed",
            "ddp_enabled": True,
            "use_mixed_precision": True,
            "find_unused_parameters": True,
        }
    ]
