"""Standalone PRING-aligned TCCIG IO and orchestration entrypoint."""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import pickle
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import networkx as nx
import yaml  # type: ignore[import-untyped]
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from tccig.io import CandidatePair, PairTable, read_pair_table, write_json
from tccig.rules import GraphRule, edges_from_rule, parse_rules, select_rule


class AcceleratorLike(Protocol):
    """Minimal accelerator surface needed by the standalone scaffold."""

    device: object


class PairwiseScorer(Protocol):
    """Callable pairwise scorer hook."""

    def __call__(self, request: PairwiseScoreRequest) -> Sequence[float]:
        """Score one split of label-free candidate pairs."""


class TrainRefiner(Protocol):
    """Callable refiner training hook."""

    def __call__(self, request: TrainRefinerRequest) -> object:
        """Train or initialize the refiner boundary."""


class PredictRefined(Protocol):
    """Callable refined-score prediction hook."""

    def __call__(self, request: RefineRequest) -> Sequence[float]:
        """Return refined probabilities for candidate pairs."""


@dataclass(frozen=True)
class TCCIGRuntime:
    """Runtime details passed to external hooks."""

    device: str
    backend: str
    mixed_precision: bool
    accelerator: AcceleratorLike


@dataclass(frozen=True)
class PairwiseScoreRequest:
    """Label-free pairwise scoring request."""

    split: str
    pairs: Sequence[CandidatePair]
    runtime: TCCIGRuntime
    config: Mapping[str, object]


@dataclass(frozen=True)
class SplitBundle:
    """Orchestrator-owned split bundle for graph/refiner stages."""

    split: str
    pairs: list[CandidatePair]
    pairwise_probabilities: list[float]
    pairwise_graph_edges: list[tuple[str, str]]
    candidate_labels: list[int] | None = None
    loss_targets: list[int] | None = None
    graph_edges: list[tuple[str, str]] | None = None


@dataclass(frozen=True)
class TrainRefinerRequest:
    """Request passed to the refiner training hook."""

    train: SplitBundle
    validation: SplitBundle
    runtime: TCCIGRuntime
    config: Mapping[str, object]


@dataclass(frozen=True)
class RefineRequest:
    """Label-free refined prediction request."""

    split: str
    pairs: Sequence[CandidatePair]
    pairwise_probabilities: Sequence[float]
    pairwise_graph_edges: Sequence[tuple[str, str]]
    refiner_state: object
    runtime: TCCIGRuntime
    config: Mapping[str, object]


@dataclass(frozen=True)
class TCCIGPipelineResult:
    """High-level result from one standalone TCCIG pipeline run."""

    manifest: dict[str, object]
    selected_rule: dict[str, object]
    pairwise_metrics: dict[str, float]
    topology_metrics: dict[str, float]


def run_tccig_pipeline(
    config: Mapping[str, object],
    *,
    build_accelerator_fn: Callable[..., AcceleratorLike] | None = None,
) -> TCCIGPipelineResult:
    """Run the standalone TCCIG PRING IO/orchestration scaffold."""
    runtime = _build_runtime(
        config=config,
        build_accelerator_fn=build_accelerator_fn or _default_build_accelerator,
    )
    run_id = _run_id(config)
    log_root = _log_root(config)
    tables = _load_tables(config)
    self_pair_rows = {split: table.self_pair_rows for split, table in tables.items()}

    pairwise_scorer = _load_pairwise_scorer(config)
    pairwise_graph_rule = _pairwise_graph_rule(config)
    scorer_cfg = _mapping_section(config, "pairwise_scorer")
    bundles = {
        split: _score_table(
            table=table,
            split=split,
            scorer=pairwise_scorer,
            runtime=runtime,
            scorer_cfg=scorer_cfg,
            pairwise_graph_rule=pairwise_graph_rule,
        )
        for split, table in tables.items()
    }
    _write_scoring_manifests(
        output_dir=log_root / "tccig" / "score" / run_id,
        tables=tables,
        bundles=bundles,
    )

    train_bundle = _with_targets(bundles["train"], tables["train"], include_loss=True)
    validation_bundle = _with_targets(
        bundles["validation"],
        tables["validation"],
        include_loss=False,
    )

    refiner_cfg = _mapping_section(config, "refiner")
    train_refiner = _load_optional_callable(
        refiner_cfg.get("train_target"),
        _not_implemented_train_refiner,
    )
    predict_refined = _load_optional_callable(
        refiner_cfg.get("predict_target"),
        _not_implemented_predict_refined,
    )
    refiner_state = cast(TrainRefiner, train_refiner)(
        TrainRefinerRequest(
            train=train_bundle,
            validation=validation_bundle,
            runtime=runtime,
            config=refiner_cfg,
        )
    )

    validation_refined = _predict_refined_probabilities(
        predict_refined=cast(PredictRefined, predict_refined),
        split="validation",
        bundle=validation_bundle,
        refiner_state=refiner_state,
        runtime=runtime,
        refiner_cfg=refiner_cfg,
    )
    rules = parse_rules(_mapping_section(config, "graph_selection").get("rules"))
    selected_rule, selected_rule_metrics = select_rule(
        pairs=validation_bundle.pairs,
        probabilities=validation_refined,
        labels=tables["validation"].labels,
        rules=rules,
    )
    selected_rule_payload: dict[str, object] = {
        **selected_rule.to_dict(),
        "validation_metrics": selected_rule_metrics,
    }
    selected_rule_path = log_root / "tccig" / "validation" / run_id / "selected_rule.json"
    write_json(selected_rule_path, selected_rule_payload)

    pairwise_metrics = _run_pairwise_test(
        bundle=bundles["pairwise_test"],
        labels=tables["pairwise_test"].labels,
        output_dir=log_root / "tccig" / "pairwise_test" / run_id,
    )
    topology_metrics = _run_topology_test(
        config=config,
        bundle=bundles["topology_test"],
        predict_refined=cast(PredictRefined, predict_refined),
        refiner_state=refiner_state,
        selected_rule=selected_rule,
        runtime=runtime,
        refiner_cfg=refiner_cfg,
        output_dir=log_root / "tccig" / "topology_test" / run_id,
    )

    manifest: dict[str, object] = {
        "run_id": run_id,
        "self_pair_rows_dropped": self_pair_rows,
        "pair_counts": {split: len(table.records) for split, table in tables.items()},
        "selected_rule_path": str(selected_rule_path),
    }
    write_json(log_root / "tccig" / "run" / run_id / "manifest.json", manifest)
    return TCCIGPipelineResult(
        manifest=manifest,
        selected_rule=selected_rule_payload,
        pairwise_metrics=pairwise_metrics,
        topology_metrics=topology_metrics,
    )


def _load_tables(config: Mapping[str, object]) -> dict[str, PairTable]:
    data_cfg = _mapping_section(config, "data")
    processed_dir = Path(str(data_cfg["processed_dir"]))
    return {
        "train": read_pair_table(
            path=processed_dir / "human_train_ppi_ratio5_exclusive.txt",
            split="train",
            expose_labels=True,
        ),
        "validation": read_pair_table(
            path=processed_dir / "human_val_ppi_ratio5_exclusive.txt",
            split="validation",
            expose_labels=True,
        ),
        "pairwise_test": read_pair_table(
            path=processed_dir / "human_test_ppi.txt",
            split="pairwise_test",
            expose_labels=True,
        ),
        "topology_test": read_pair_table(
            path=processed_dir / "all_test_ppi.txt",
            split="topology_test",
            expose_labels=False,
        ),
    }


def _score_table(
    *,
    table: PairTable,
    split: str,
    scorer: PairwiseScorer,
    runtime: TCCIGRuntime,
    scorer_cfg: Mapping[str, object],
    pairwise_graph_rule: GraphRule,
) -> SplitBundle:
    pairs = table.pairs
    raw_scores = scorer(
        PairwiseScoreRequest(
            split=split,
            pairs=pairs,
            runtime=runtime,
            config=scorer_cfg,
        )
    )
    probabilities = _normalize_probabilities(raw_scores)
    graph_edges = edges_from_rule(
        pairs=pairs,
        probabilities=probabilities,
        rule=pairwise_graph_rule,
    )
    return SplitBundle(
        split=split,
        pairs=pairs,
        pairwise_probabilities=probabilities,
        pairwise_graph_edges=graph_edges,
    )


def _write_scoring_manifests(
    *,
    output_dir: Path,
    tables: Mapping[str, PairTable],
    bundles: Mapping[str, SplitBundle],
) -> None:
    """Persist one label-safe scoring manifest per split."""
    for split, table in tables.items():
        bundle = bundles[split]
        write_json(
            output_dir / f"{split}.json",
            {
                "split": split,
                "path": str(table.path),
                "pair_count": len(bundle.pairs),
                "self_pair_rows_dropped": table.self_pair_rows,
                "pairwise_graph_edge_count": len(bundle.pairwise_graph_edges),
            },
        )


def _with_targets(bundle: SplitBundle, table: PairTable, *, include_loss: bool) -> SplitBundle:
    labels = table.labels
    return SplitBundle(
        split=bundle.split,
        pairs=bundle.pairs,
        pairwise_probabilities=bundle.pairwise_probabilities,
        pairwise_graph_edges=bundle.pairwise_graph_edges,
        candidate_labels=labels,
        loss_targets=labels if include_loss else None,
        graph_edges=table.positive_edges,
    )


def _predict_refined_probabilities(
    *,
    predict_refined: PredictRefined,
    split: str,
    bundle: SplitBundle,
    refiner_state: object,
    runtime: TCCIGRuntime,
    refiner_cfg: Mapping[str, object],
) -> list[float]:
    raw_scores = predict_refined(
        RefineRequest(
            split=split,
            pairs=bundle.pairs,
            pairwise_probabilities=bundle.pairwise_probabilities,
            pairwise_graph_edges=bundle.pairwise_graph_edges,
            refiner_state=refiner_state,
            runtime=runtime,
            config=refiner_cfg,
        )
    )
    return _normalize_probabilities(raw_scores)


def _run_pairwise_test(
    *,
    bundle: SplitBundle,
    labels: list[int],
    output_dir: Path,
) -> dict[str, float]:
    metrics = _binary_metrics(labels=labels, probabilities=bundle.pairwise_probabilities)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "pairwise_metrics.json", metrics)
    with (output_dir / "pairwise_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", *metrics.keys()])
        writer.writeheader()
        writer.writerow({"split": "pairwise_test", **metrics})
    return metrics


def _run_topology_test(
    *,
    config: Mapping[str, object],
    bundle: SplitBundle,
    predict_refined: PredictRefined,
    refiner_state: object,
    selected_rule: GraphRule,
    runtime: TCCIGRuntime,
    refiner_cfg: Mapping[str, object],
    output_dir: Path,
) -> dict[str, float]:
    refined_probabilities = _predict_refined_probabilities(
        predict_refined=predict_refined,
        split="topology_test",
        bundle=bundle,
        refiner_state=refiner_state,
        runtime=runtime,
        refiner_cfg=refiner_cfg,
    )
    selected_edges = set(
        edges_from_rule(
            pairs=bundle.pairs,
            probabilities=refined_probabilities,
            rule=selected_rule,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "all_test_ppi_pred.txt").open("w", encoding="utf-8") as handle:
        for pair in bundle.pairs:
            edge = tuple(sorted((pair.protein_a, pair.protein_b)))
            handle.write(f"{pair.protein_a}\t{pair.protein_b}\t{int(edge in selected_edges)}\n")

    data_cfg = _mapping_section(config, "data")
    processed_dir = Path(str(data_cfg["processed_dir"]))
    with (processed_dir / "human_test_graph.pkl").open("rb") as handle:
        gt_graph = cast(nx.Graph, pickle.load(handle))
    with (processed_dir / "test_sampled_nodes.pkl").open("rb") as handle:
        test_sampled_nodes = cast(dict[int, list[list[str]]], pickle.load(handle))
    topology_result = _evaluate_predicted_graph(
        pred_graph=_reconstruct_graph(selected_edges),
        gt_graph=gt_graph,
        test_graph_nodes=test_sampled_nodes,
    )
    raw_summary = cast(Mapping[str, object], topology_result["summary"])
    summary = {name: _object_to_float(value) for name, value in raw_summary.items()}
    write_json(output_dir / "topology_metrics.json", {"summary": summary})
    return summary


def _binary_metrics(*, labels: list[int], probabilities: list[float]) -> dict[str, float]:
    predictions = [int(probability >= 0.5) for probability in probabilities]
    has_both_classes = len(set(labels)) > 1
    return {
        "auroc": _safe_metric(
            lambda: roc_auc_score(labels, probabilities) if has_both_classes else 0.0
        ),
        "auprc": _safe_metric(
            lambda: average_precision_score(labels, probabilities) if has_both_classes else 0.0
        ),
        "accuracy": _safe_metric(lambda: accuracy_score(labels, predictions)),
        "precision": _safe_metric(lambda: precision_score(labels, predictions, zero_division=0)),
        "recall": _safe_metric(lambda: recall_score(labels, predictions, zero_division=0)),
        "f1": _safe_metric(lambda: f1_score(labels, predictions, zero_division=0)),
        "mcc": _safe_metric(lambda: matthews_corrcoef(labels, predictions)),
    }


def _safe_metric(metric_fn: Callable[[], float]) -> float:
    value = float(metric_fn())
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def _object_to_float(value: object) -> float:
    """Convert JSON-like numeric values into float."""
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Expected numeric topology metric value, got {type(value).__name__}")


def _normalize_probabilities(raw_scores: Sequence[float]) -> list[float]:
    values = [float(value) for value in raw_scores]
    if all(0.0 <= value <= 1.0 for value in values):
        return values
    return [1.0 / (1.0 + math.exp(-value)) for value in values]


def _load_pairwise_scorer(config: Mapping[str, object]) -> PairwiseScorer:
    scorer_cfg = _mapping_section(config, "pairwise_scorer")
    return cast(PairwiseScorer, _load_callable(scorer_cfg["target"]))


def _load_optional_callable(value: object, default: Callable[..., object]) -> Callable[..., object]:
    if value is None:
        return default
    return _load_callable(value)


def _load_callable(value: object) -> Callable[..., object]:
    if not isinstance(value, str) or ":" not in value:
        raise ValueError("Hook target must use 'module:function' syntax")
    module_name, function_name = value.split(":", 1)
    module = importlib.import_module(module_name)
    loaded = getattr(module, function_name)
    if not callable(loaded):
        raise TypeError(f"Hook target is not callable: {value}")
    return cast(Callable[..., object], loaded)


def _not_implemented_train_refiner(request: TrainRefinerRequest) -> object:
    del request
    raise NotImplementedError("refiner.train_target is required until model training exists")


def _not_implemented_predict_refined(request: RefineRequest) -> Sequence[float]:
    del request
    raise NotImplementedError("refiner.predict_target is required until model inference exists")


def _build_runtime(
    *,
    config: Mapping[str, object],
    build_accelerator_fn: Callable[..., AcceleratorLike],
) -> TCCIGRuntime:
    device_cfg = _mapping_section(config, "device")
    requested_device = str(device_cfg.get("device", "cpu"))
    backend = str(device_cfg.get("backend", "ddp")).lower()
    mixed_precision = bool(device_cfg.get("mixed_precision", False))
    accelerator = build_accelerator_fn(
        requested_device=requested_device,
        backend=backend,
        ddp_enabled=backend in {"ddp", "deepspeed"},
        use_mixed_precision=mixed_precision,
        find_unused_parameters=bool(device_cfg.get("find_unused_parameters", False)),
    )
    return TCCIGRuntime(
        device=str(accelerator.device),
        backend=backend,
        mixed_precision=mixed_precision,
        accelerator=accelerator,
    )


def _default_build_accelerator(**kwargs: object) -> AcceleratorLike:
    """Lazy-load the repository Accelerator builder for real CLI runs."""
    runtime_module = importlib.import_module("src.pipeline.runtime")
    build_fn = runtime_module.build_accelerator
    return cast(AcceleratorLike, build_fn(**kwargs))


def _reconstruct_graph(edges: set[tuple[str, str]]) -> nx.Graph:
    """Lazy-load the repo PRING graph reconstruction helper."""
    metrics_module = importlib.import_module("src.topology.metrics")
    reconstruct_fn = metrics_module.reconstruct_graph
    return cast(nx.Graph, reconstruct_fn(edges))


def _evaluate_predicted_graph(
    *,
    pred_graph: nx.Graph,
    gt_graph: nx.Graph,
    test_graph_nodes: Mapping[int, list[list[str]]],
) -> Mapping[str, object]:
    """Lazy-load the repo PRING topology metric helper."""
    metrics_module = importlib.import_module("src.topology.metrics")
    evaluate_fn = metrics_module.evaluate_predicted_graph
    return cast(
        Mapping[str, object],
        evaluate_fn(
            pred_graph=pred_graph,
            gt_graph=gt_graph,
            test_graph_nodes=test_graph_nodes,
        ),
    )


def _pairwise_graph_rule(config: Mapping[str, object]) -> GraphRule:
    graph_cfg = _mapping_section(config, "graph_selection")
    raw_rule = graph_cfg.get("pairwise_graph_rule", {"type": "threshold", "value": 0.5})
    return parse_rules([raw_rule])[0]


def _mapping_section(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return cast(Mapping[str, object], value)


def _run_id(config: Mapping[str, object]) -> str:
    run_cfg = _mapping_section(config, "run")
    return str(run_cfg.get("run_id", "tccig_run"))


def _log_root(config: Mapping[str, object]) -> Path:
    run_cfg = _mapping_section(config, "run")
    return Path(str(run_cfg.get("log_root", "logs")))


def _load_yaml_config(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Config root must be a mapping")
    return cast(Mapping[str, object], payload)


def main() -> None:
    """CLI entrypoint for ``uv run python tccig/train.py --config <yaml>``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_tccig_pipeline(_load_yaml_config(args.config))


if __name__ == "__main__":
    main()
