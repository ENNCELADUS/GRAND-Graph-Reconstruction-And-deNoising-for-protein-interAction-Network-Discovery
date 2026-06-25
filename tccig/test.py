"""Test-time evaluation helpers for the concrete TCCIG pipeline."""

from __future__ import annotations

import csv
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

import networkx as nx
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.topology.metrics import evaluate_predicted_graph, reconstruct_graph

from tccig import s2gae
from tccig.prepare import (
    CandidatePair,
    GraphRule,
    PairTable,
    RefineRequest,
    TCCIGRuntime,
    canonical_edge,
    edges_from_rule,
    write_json,
)

TOPOLOGY_METRIC_NAMES = [
    "graph_sim",
    "relative_density",
    "deg_dist_mmd",
    "cc_mmd",
    "laplacian_eigen_mmd",
]
TOPOLOGY_CSV_COLUMNS = ["scope", "node_size", "graph_count", *TOPOLOGY_METRIC_NAMES]
# raw_metrics.json reports the frozen scorer's own decision quality at its natural
# 0.5 probability boundary, independent of the (often target-precision) input-graph
# threshold used to build G_pairwise. Threshold-free AUROC/AUPRC are the real
# raw-vs-refined comparison; this only sets the point-metric operating point.
RAW_SCORER_DECISION_THRESHOLD = 0.5
PAIRWISE_CSV_COLUMNS = [
    "protein_a",
    "protein_b",
    "label",
    "raw_probability",
    "refined_probability",
]


class ScoreSplitFn(Protocol):
    """Pairwise scorer/cache boundary owned by the orchestrator."""

    def __call__(
        self,
        *,
        split: str,
        pairs: Sequence[CandidatePair],
        scorer_cfg: Mapping[str, object],
        runtime: TCCIGRuntime,
        cache_dir: Path,
    ) -> list[float]:
        """Return frozen pairwise scorer probabilities for one split."""


def run_pairwise_test(
    *,
    table: PairTable,
    scorer_cfg: Mapping[str, object],
    refiner_cfg: Mapping[str, object],
    runtime: TCCIGRuntime,
    cache_dir: Path,
    log_dir: Path,
    refiner_state: object,
    pairwise_input_rule: GraphRule,
    refined_output_rule: GraphRule,
    score_split_fn: ScoreSplitFn,
) -> dict[str, float]:
    """Write full-model pairwise-test artifacts and return binary metrics."""
    pairwise_scores = score_split_fn(
        split="pairwise_test",
        pairs=table.pairs,
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
    )
    pairwise_edges = edges_from_rule(
        pairs=table.pairs,
        probabilities=pairwise_scores,
        rule=pairwise_input_rule,
    )
    refined_scores = s2gae.predict_refined(
        RefineRequest(
            split="pairwise_test",
            pairs=table.pairs,
            pairwise_probabilities=pairwise_scores,
            pairwise_graph_edges=pairwise_edges,
            refiner_state=refiner_state,
            runtime=runtime,
            config=refiner_cfg,
        )
    )
    metrics = _binary_metrics(
        labels=table.labels,
        probabilities=refined_scores,
        threshold=float(refined_output_rule.value),
    )
    raw_metrics = _binary_metrics(
        labels=table.labels,
        probabilities=pairwise_scores,
        threshold=RAW_SCORER_DECISION_THRESHOLD,
    )
    if runtime.is_main_process:
        pairwise_dir = log_dir / "pairwise_test"
        _write_pairwise_predictions(
            pairwise_dir / "human_test_ppi_pred.csv",
            table,
            raw_probabilities=pairwise_scores,
            refined_probabilities=refined_scores,
        )
        write_json(pairwise_dir / "raw_metrics.json", raw_metrics)
        write_json(pairwise_dir / "refined_metrics.json", metrics)
        # Back-compat: pairwise_metrics.json mirrors the refined metrics.
        write_json(pairwise_dir / "pairwise_metrics.json", metrics)
    _runtime_barrier(runtime)
    return metrics


def run_topology_test(
    *,
    table: PairTable,
    processed_dir: Path,
    scorer_cfg: Mapping[str, object],
    refiner_cfg: Mapping[str, object],
    runtime: TCCIGRuntime,
    cache_dir: Path,
    log_dir: Path,
    refiner_state: object,
    pairwise_input_rule: GraphRule,
    refined_output_rule: GraphRule,
    pairwise_input_payload: Mapping[str, object],
    score_split_fn: ScoreSplitFn,
) -> dict[str, float]:
    """Write full-model topology-test artifacts and return official metrics."""
    pairwise_scores = score_split_fn(
        split="topology_test",
        pairs=table.pairs,
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
    )
    pairwise_edges = edges_from_rule(
        pairs=table.pairs,
        probabilities=pairwise_scores,
        rule=pairwise_input_rule,
    )
    refined_scores = s2gae.predict_refined(
        RefineRequest(
            split="topology_test",
            pairs=table.pairs,
            pairwise_probabilities=pairwise_scores,
            pairwise_graph_edges=pairwise_edges,
            refiner_state=refiner_state,
            runtime=runtime,
            config=refiner_cfg,
        )
    )
    selected_edges = edges_from_rule(
        pairs=table.pairs,
        probabilities=refined_scores,
        rule=refined_output_rule,
    )
    selected_edge_set = set(selected_edges)
    predictions = [
        int(canonical_edge(pair.protein_a, pair.protein_b) in selected_edge_set)
        for pair in table.pairs
    ]
    with (processed_dir / "human_test_graph.pkl").open("rb") as handle:
        gt_graph = cast(nx.Graph, pickle.load(handle))
    with (processed_dir / "test_sampled_nodes.pkl").open("rb") as handle:
        test_graph_nodes = cast(dict[int, list[list[str]]], pickle.load(handle))
    topology_result = evaluate_predicted_graph(
        pred_graph=reconstruct_graph(selected_edges),
        gt_graph=gt_graph,
        test_graph_nodes=test_graph_nodes,
    )
    summary = {key: float(value) for key, value in topology_result["summary"].items()}
    if runtime.is_main_process:
        topology_dir = log_dir / "topology_test"
        _write_topology_predictions(
            output_path=topology_dir / "all_test_ppi_pred.txt",
            pairs=table.pairs,
            predictions=predictions,
        )
        payload = {
            "summary": summary,
            "per_node_size": _json_safe_per_node_size(topology_result["per_node_size"]),
            "details": _json_safe_details(topology_result["details"]),
            "refined_output_rule": refined_output_rule.to_dict(),
            "pairwise_input_rule": dict(pairwise_input_payload),
            "protocol": {
                "candidate_universe": "all_test_ppi.txt",
                "ground_truth_graph": "human_test_graph.pkl",
                "sampled_nodes": "test_sampled_nodes.pkl",
                "test_labels_visible_to_model": False,
            },
        }
        write_json(topology_dir / "topology_metrics.json", payload)
        _write_topology_metrics_csv(
            csv_path=topology_dir / "topology_metrics.csv",
            per_node_size=cast(dict[int, dict[str, float | int]], topology_result["per_node_size"]),
            summary=summary,
        )
    _runtime_barrier(runtime)
    return summary


def _binary_metrics(
    *,
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> dict[str, float]:
    predictions = [int(float(probability) >= threshold) for probability in probabilities]
    labels_list = [int(label) for label in labels]
    class_count = len(set(labels_list))
    return {
        "accuracy": float(accuracy_score(labels_list, predictions)),
        "precision": float(precision_score(labels_list, predictions, zero_division=0)),
        "recall": float(recall_score(labels_list, predictions, zero_division=0)),
        "f1": float(f1_score(labels_list, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels_list, predictions)),
        "auroc": 0.0 if class_count < 2 else float(roc_auc_score(labels_list, probabilities)),
        "auprc": 0.0
        if class_count < 2
        else float(average_precision_score(labels_list, probabilities)),
        "threshold": float(threshold),
    }


def _write_pairwise_predictions(
    path: Path,
    table: PairTable,
    *,
    raw_probabilities: Sequence[float],
    refined_probabilities: Sequence[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PAIRWISE_CSV_COLUMNS,
        )
        writer.writeheader()
        for pair, label, raw, refined in zip(
            table.pairs,
            table.labels,
            raw_probabilities,
            refined_probabilities,
            strict=True,
        ):
            writer.writerow(
                {
                    "protein_a": pair.protein_a,
                    "protein_b": pair.protein_b,
                    "label": int(label),
                    "raw_probability": float(raw),
                    "refined_probability": float(refined),
                }
            )


def _write_topology_predictions(
    *,
    output_path: Path,
    pairs: Sequence[CandidatePair],
    predictions: Sequence[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pair, prediction in zip(pairs, predictions, strict=True):
            handle.write(f"{pair.protein_a}\t{pair.protein_b}\t{int(prediction)}\n")


def _write_topology_metrics_csv(
    *,
    csv_path: Path,
    per_node_size: Mapping[int, Mapping[str, float | int]],
    summary: Mapping[str, float],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOPOLOGY_CSV_COLUMNS)
        writer.writeheader()
        for node_size in sorted(per_node_size):
            writer.writerow(
                {"scope": "node_size", "node_size": node_size, **per_node_size[node_size]}
            )
        writer.writerow(
            {
                "scope": "summary",
                "node_size": "all",
                "graph_count": sum(int(values["graph_count"]) for values in per_node_size.values()),
                **summary,
            }
        )


def _json_safe_details(
    details: Mapping[str, Mapping[int, Sequence[float] | float]],
) -> dict[str, dict[str, object]]:
    return {
        metric_name: {str(node_size): values for node_size, values in values_by_size.items()}
        for metric_name, values_by_size in details.items()
    }


def _json_safe_per_node_size(
    per_node_size: Mapping[int, Mapping[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    return {str(node_size): dict(values) for node_size, values in per_node_size.items()}


def _runtime_barrier(runtime: TCCIGRuntime) -> None:
    wait_for_everyone = getattr(runtime.accelerator, "wait_for_everyone", None)
    if callable(wait_for_everyone):
        wait_for_everyone()
