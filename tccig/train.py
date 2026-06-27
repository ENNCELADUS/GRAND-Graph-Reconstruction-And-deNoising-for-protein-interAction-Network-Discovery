"""Concrete Accelerate-backed TCCIG training and evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import networkx as nx
import torch
import yaml  # type: ignore[import-untyped]
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, set_seed
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
)
from src.embed import load_cached_embedding
from src.pipeline.stages.train import build_model
from src.topology.finetune_data import (
    TOPOLOGY_EVAL_NODE_SIZES,
    InternalValidationPlan,
    _canonical_edge,
    _expand_chunk_nodes,
    build_internal_validation_plan,
    build_pair_supervision_graph,
    load_split_node_ids,
    sample_topology_evaluation_subgraphs,
)
from src.topology.plan_cache import (
    load_plan_cache,
    payload_to_plan,
    plan_payload_metadata,
    plan_to_payload,
    write_plan_cache,
)
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from tccig import s2gae
from tccig import test as tccig_test
from tccig.prepare import (
    CandidatePair,
    GraphRule,
    PairScoreDataset,
    PairScoreRecord,
    PairTable,
    SplitBundle,
    TCCIGPipelineResult,
    TCCIGRuntime,
    TrainRefinerRequest,
    collate_pair_score_records,
    edges_from_rule,
    load_pring_tables,
    ordered_probabilities_from_indexed_rows,
    score_cache_metadata,
    strict_reject_legacy_hooks,
    write_json,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_GRAPH_THRESHOLD = 0.5
AcceleratorFactory = Callable[[], object]


def parse_rules(raw_rules: object) -> list[GraphRule]:
    """Parse configured threshold-only graph rules."""
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("graph_selection.rules must be a non-empty list")
    rules: list[GraphRule] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("graph_selection.rules entries must be mappings")
        rule_type = str(raw_rule.get("type", "")).lower()
        if rule_type == "threshold":
            rules.append(GraphRule(type=rule_type, value=float(raw_rule.get("value", 0.5))))
        else:
            raise ValueError(
                f"Unsupported graph rule type: {rule_type}; "
                "TCCIG graph rules only support threshold"
            )
    return rules


def threshold_for_target_precision(
    *,
    probabilities: Sequence[float],
    labels: Sequence[int],
    target_precision: float,
) -> tuple[float, dict[str, float | int]]:
    """Return the lowest probability threshold that reaches target precision."""
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have matching lengths")
    if not 0.0 <= target_precision <= 1.0:
        raise ValueError("target_precision must be in [0, 1]")
    if not probabilities:
        raise ValueError("target_precision threshold requires at least one probability")

    best_threshold: float | None = None
    best_metrics: dict[str, float | int] = {}
    for threshold in sorted({float(probability) for probability in probabilities}):
        metrics = binary_metrics_at_threshold(
            labels=labels,
            probabilities=probabilities,
            threshold=threshold,
        )
        if int(metrics["positive_edges"]) <= 0:
            continue
        if float(metrics["precision"]) >= target_precision:
            best_threshold = threshold
            best_metrics = metrics
            break
    if best_threshold is None:
        raise ValueError(f"No scorer threshold reaches target_precision={target_precision}")
    return best_threshold, best_metrics


def binary_metrics_at_threshold(
    *,
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> dict[str, float | int]:
    """Return binary metrics under a fixed probability threshold."""
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have matching lengths")
    predictions = [int(float(probability) >= float(threshold)) for probability in probabilities]
    positive_edges = sum(predictions)
    true_positive_edges = sum(
        1 for label, pred in zip(labels, predictions, strict=True) if label == 1 and pred
    )
    actual_positive_edges = sum(1 for label in labels if label == 1)
    return {
        "precision": 0.0 if positive_edges == 0 else float(true_positive_edges / positive_edges),
        "recall": float(true_positive_edges / max(1, actual_positive_edges)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "positive_edges": int(positive_edges),
    }


def run_tccig_pipeline(
    config: Mapping[str, object],
    *,
    build_accelerator_fn: AcceleratorFactory | None = None,
) -> TCCIGPipelineResult:
    """Run the concrete scorer -> S2GAE refiner -> test-artifact TCCIG pipeline."""
    strict_reject_legacy_hooks(config)
    runtime = _build_runtime(config=config, build_accelerator_fn=build_accelerator_fn)
    _configure_logging(runtime)
    set_seed(_sampling_seed(config))

    run_id = _run_id(config)
    log_dir = _log_root(config) / "tccig" / run_id
    cache_dir = _cache_root(config) / "score_cache" / run_id
    processed_dir = _required_path(_mapping_section(config, "data"), "processed_dir")
    tables = load_pring_tables(processed_dir)
    scorer_cfg = _mapping_section(config, "pairwise_scorer")
    refiner_cfg = dict(_mapping_section(config, "refiner"))
    refiner_cfg["_run_id"] = run_id
    refiner_cfg["_log_root"] = str(_log_root(config))

    train_scores = _score_split(
        split="train",
        pairs=tables["train"].pairs,
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
    )
    validation_scores = _score_split(
        split="validation",
        pairs=tables["validation"].pairs,
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
    )
    pairwise_input_rule, pairwise_input_payload = _resolve_pairwise_input_rule(
        config=config,
        validation_scores=validation_scores,
        validation_labels=tables["validation"].labels,
    )
    refined_output_rule = _resolve_refined_output_rule(config)
    parsed_rules = parse_rules(
        _graph_selection(config).get("rules", [refined_output_rule.to_dict()])
    )
    graph_rule = parsed_rules[0]

    train_bundle = _bundle_from_table(
        table=tables["train"],
        probabilities=train_scores,
        graph_rule=pairwise_input_rule,
        include_loss_targets=True,
    )
    validation_bundle = _bundle_from_table(
        table=tables["validation"],
        probabilities=validation_scores,
        graph_rule=pairwise_input_rule,
        include_loss_targets=True,
    )
    validation_topology, validation_topology_plan = _build_validation_topology_bundle(
        config=config,
        processed_dir=processed_dir,
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
        pairwise_input_rule=pairwise_input_rule,
    )
    train_topology, train_topology_plan, _train_topo_stats = _build_train_topology_bundle(
        config=config,
        processed_dir=processed_dir,
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
        pairwise_input_rule=pairwise_input_rule,
    )

    refiner_state = s2gae.train_refiner(
        TrainRefinerRequest(
            train=train_bundle,
            validation=validation_bundle,
            runtime=runtime,
            config=refiner_cfg,
            graph_rule=graph_rule,
            validation_topology=validation_topology,
            validation_topology_plan=validation_topology_plan,
            train_topology=train_topology,
            train_topology_plan=train_topology_plan,
        )
    )
    pairwise_metrics = tccig_test.run_pairwise_test(
        table=tables["pairwise_test"],
        scorer_cfg=scorer_cfg,
        refiner_cfg=refiner_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
        log_dir=log_dir,
        refiner_state=refiner_state,
        pairwise_input_rule=pairwise_input_rule,
        refined_output_rule=refined_output_rule,
        score_split_fn=_score_split,
    )
    topology_metrics = tccig_test.run_topology_test(
        table=tables["topology_test"],
        processed_dir=processed_dir,
        scorer_cfg=scorer_cfg,
        refiner_cfg=refiner_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
        log_dir=log_dir,
        refiner_state=refiner_state,
        pairwise_input_rule=pairwise_input_rule,
        refined_output_rule=refined_output_rule,
        pairwise_input_payload=pairwise_input_rule.to_dict(),
        score_split_fn=_score_split,
    )

    manifest = {
        "run_id": run_id,
        "self_pair_rows_dropped": {split: table.self_pair_rows for split, table in tables.items()},
        "pairwise_input_threshold": pairwise_input_payload,
        "refined_output_rule": refined_output_rule.to_dict(),
    }
    if runtime.is_main_process:
        write_json(log_dir / "manifest.json", manifest)
    _runtime_barrier(runtime)
    return TCCIGPipelineResult(
        manifest=manifest,
        pairwise_input_threshold=pairwise_input_payload,
        refined_output_rule=refined_output_rule.to_dict(),
        pairwise_metrics=pairwise_metrics,
        topology_metrics=topology_metrics,
    )


@torch.no_grad()
def score_pairs_with_v3_1(
    *,
    pairs: Sequence[CandidatePair],
    config: Mapping[str, object],
    runtime: TCCIGRuntime,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> list[float]:
    """Score label-free candidate pairs with the frozen v3.1 ABBA/no-cross checkpoint."""
    if not pairs:
        return []
    model_config = _load_v3_1_abba_no_cross_model_config(
        _required_path(config, "model_config_path")
    )
    checkpoint_path = _required_path(config, "checkpoint_path")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")
    embedding_cache_dir = _required_path(config, "embedding_cache_dir")
    embedding_index = _load_embedding_index(embedding_cache_dir / "index.json")
    input_dim = _positive_int(
        cast(Mapping[str, object], model_config["model_config"]).get("input_dim"),
        "pairwise_scorer.model_config.input_dim",
    )
    max_sequence_length = _optional_positive_int(
        config.get("max_sequence_length"),
        "pairwise_scorer.max_sequence_length",
    )

    model = build_model(cast(dict[str, object], model_config))
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state))
    model.to(torch.device(runtime.device))
    model.eval()

    loader = DataLoader(
        PairScoreDataset(pairs),
        batch_size=_positive_int(config.get("batch_size", 512), "pairwise_scorer.batch_size"),
        shuffle=False,
        collate_fn=lambda records: _collate_pair_score_batch(
            records=records,
            cache_dir=embedding_cache_dir,
            embedding_index=embedding_index,
            input_dim=input_dim,
            max_sequence_length=max_sequence_length,
        ),
    )
    prepared = runtime.accelerator.prepare(model, loader)
    model, loader = _prepared_model_and_loader(prepared)

    rows: list[torch.Tensor] = []
    processed = 0
    gather_for_metrics = runtime.accelerator.gather_for_metrics if runtime.is_distributed else None
    for batch_index, batch in enumerate(loader, start=1):
        indices = batch.pop("index").to(runtime.device)
        output = model({key: value.to(runtime.device) for key, value in batch.items()})
        logits = cast(torch.Tensor, output["logits"]).squeeze(-1)
        probabilities = torch.sigmoid(logits).to(dtype=torch.float64)
        batch_rows = torch.stack((indices.to(dtype=torch.float64), probabilities), dim=1)
        if gather_for_metrics is not None:
            gathered = gather_for_metrics(batch_rows)
            if isinstance(gathered, torch.Tensor):
                batch_rows = gathered
        rows.append(batch_rows)
        processed += int(indices.numel())
        if progress_callback is not None:
            progress_callback(
                {
                    "batch_index": batch_index,
                    "processed_pairs": processed,
                    "local_pair_count": len(pairs),
                }
            )

    local_rows = (
        torch.cat(rows, dim=0)
        if rows
        else torch.empty((0, 2), dtype=torch.float64, device=torch.device(runtime.device))
    )
    return ordered_probabilities_from_indexed_rows(total=len(pairs), rows=local_rows)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for ``python -m tccig.train --config ...``."""
    parser = argparse.ArgumentParser(description="Run the concrete TCCIG pipeline")
    parser.add_argument("--config", required=True, help="Path to a TCCIG YAML config")
    args = parser.parse_args(argv)
    run_tccig_pipeline(_load_yaml_config(Path(args.config)))


def _score_split(
    *,
    split: str,
    pairs: Sequence[CandidatePair],
    scorer_cfg: Mapping[str, object],
    runtime: TCCIGRuntime,
    cache_dir: Path,
) -> list[float]:
    metadata = score_cache_metadata(split=split, pairs=pairs, scorer_config=scorer_cfg)
    if _score_cache_enabled(scorer_cfg):
        cached = _load_score_cache(cache_dir=cache_dir, split=split, metadata=metadata)
        if cached is not None:
            return cached

    probabilities = score_pairs_with_v3_1(pairs=pairs, config=scorer_cfg, runtime=runtime)
    if _score_cache_enabled(scorer_cfg) and runtime.is_main_process:
        _write_score_cache(
            cache_dir=cache_dir,
            split=split,
            metadata=metadata,
            probabilities=probabilities,
        )
    _runtime_barrier(runtime)
    if _score_cache_enabled(scorer_cfg):
        cached = _load_score_cache(cache_dir=cache_dir, split=split, metadata=metadata)
        if cached is None:
            raise RuntimeError(f"score cache was not written for split={split}")
        return cached
    return probabilities


def _covered_positive_edges(
    *,
    sampled: Mapping[int, Sequence[tuple[str, ...]]],
    graph: nx.Graph,
) -> set[frozenset[str]]:
    """Return the set of GT positive edges contained in any sampled bucket."""
    covered: set[frozenset[str]] = set()
    for buckets in sampled.values():
        for nodes in buckets:
            node_set = set(nodes)
            for node_a, node_b in graph.subgraph(node_set).edges():
                covered.add(frozenset((node_a, node_b)))
    return covered


def augment_plan_for_positive_edge_coverage(
    *,
    graph: nx.Graph,
    base_sampled: dict[int, list[tuple[str, ...]]],
    node_sizes: Sequence[int],
    strategy: str,
    seed: int,
) -> tuple[dict[int, list[tuple[str, ...]]], dict[str, float | int]]:
    """Add coverage buckets until every GT positive edge appears in some bucket."""
    augmented: dict[int, list[tuple[str, ...]]] = {
        size: list(buckets) for size, buckets in base_sampled.items()
    }
    base_bucket_count = sum(len(buckets) for buckets in augmented.values())
    all_positive = {_canonical_edge(node_a, node_b) for node_a, node_b in graph.edges()}
    target_size = max(node_sizes)
    normalized_strategy = strategy.upper()
    if normalized_strategy not in {"BFS", "DFS", "RANDOM_WALK"}:
        normalized_strategy = "BFS"
    rng = random.Random(seed)

    # Maintain `covered` incrementally: seed it once from the base buckets, then
    # extend it with each newly added coverage bucket's induced edges. This avoids
    # re-scanning every bucket per uncovered edge (the previous quadratic hotspot).
    covered: set[tuple[str, str]] = {
        _canonical_edge(*tuple(sorted(edge)))
        for edge in _covered_positive_edges(sampled=augmented, graph=graph)
    }
    coverage_bucket_count = 0
    for edge in sorted(all_positive - covered):
        if edge in covered:
            continue  # drained by a previously added coverage bucket
        nodes = _expand_chunk_nodes(
            graph=graph,
            edge_chunk=[(edge[0], edge[1])],
            target_size=target_size,
            strategy=normalized_strategy,
            rng=rng,
        )
        augmented.setdefault(target_size, []).append(tuple(sorted(nodes)))
        for node_a, node_b in graph.subgraph(set(nodes)).edges():
            covered.add(_canonical_edge(node_a, node_b))
        coverage_bucket_count += 1

    matched_positive = len(covered & all_positive)
    coverage = 1.0 if not all_positive else matched_positive / len(all_positive)
    if coverage != 1.0:
        raise ValueError(
            f"positive-edge coverage augmentation failed: coverage={coverage:.6f} < 1.0"
        )
    stats: dict[str, float | int] = {
        "base_bucket_count": base_bucket_count,
        "coverage_bucket_count": coverage_bucket_count,
        "positive_edge_coverage": coverage,
    }
    return augmented, stats


def _coverage_stats_from_payload(payload: Mapping[str, object]) -> dict[str, float | int]:
    """Extract persisted coverage stats, defaulting to an empty contract."""
    raw = payload.get("coverage_stats", {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): value for key, value in raw.items()}  # type: ignore[misc]


def _load_or_build_topology_plan(
    *,
    split: str,
    graph: nx.Graph,
    node_sizes: Sequence[int],
    samples_per_size: int,
    seed: int,
    strategy: str,
    coverage_augmentation: bool,
    runtime: TCCIGRuntime,
    cache_dir: Path,
    build_fn: Callable[[], tuple[InternalValidationPlan, dict[str, float | int]]],
) -> tuple[InternalValidationPlan, dict[str, float | int]]:
    """Load the topology plan from cache or build it once on the main rank.

    Mirrors ``_score_split``: all ranks attempt the load; on a miss only the main
    rank builds and writes, then a barrier lets every rank read the result.
    """
    metadata = plan_payload_metadata(
        split=split,
        graph=graph,
        node_sizes=node_sizes,
        samples_per_size=samples_per_size,
        seed=seed,
        strategy=strategy,
        coverage_augmentation=coverage_augmentation,
    )
    cached = load_plan_cache(cache_dir=cache_dir, split=split, metadata=metadata)
    if cached is not None:
        return payload_to_plan(cached, graph=graph), _coverage_stats_from_payload(cached)

    if runtime.is_main_process:
        plan, coverage_stats = build_fn()
        payload = plan_to_payload(plan)
        payload["coverage_stats"] = dict(coverage_stats)
        write_plan_cache(cache_dir=cache_dir, split=split, metadata=metadata, payload=payload)
    _runtime_barrier(runtime)

    reloaded = load_plan_cache(cache_dir=cache_dir, split=split, metadata=metadata)
    if reloaded is None:
        raise RuntimeError(f"topology plan cache was not written for split={split}")
    return payload_to_plan(reloaded, graph=graph), _coverage_stats_from_payload(reloaded)


def _build_train_topology_bundle(
    *,
    config: Mapping[str, object],
    processed_dir: Path,
    scorer_cfg: Mapping[str, object],
    runtime: TCCIGRuntime,
    cache_dir: Path,
    pairwise_input_rule: GraphRule,
) -> tuple[SplitBundle | None, object | None, dict[str, float | int]]:
    refiner_cfg = _mapping_section(config, "refiner")
    topo_cfg = refiner_cfg.get("topology_training", {})
    if not isinstance(topo_cfg, Mapping) or not bool(topo_cfg.get("enabled", False)):
        return None, None, {}
    split_path = processed_dir / "human_BFS_split.pkl"
    node_ids = load_split_node_ids(split_path=split_path, split_name="train")
    train_graph = build_pair_supervision_graph(
        pair_path=processed_dir / "human_train_ppi_ratio5_exclusive.txt",
        node_ids=node_ids,
    )
    node_sizes = _int_sequence(
        topo_cfg.get("node_sizes", TOPOLOGY_EVAL_NODE_SIZES),
        "refiner.topology_training.node_sizes",
    )
    seed = _non_negative_int(topo_cfg.get("seed", 0), "refiner.topology_training.seed")
    strategy = str(topo_cfg.get("strategy", "mixed"))
    samples_per_size = _positive_int(
        topo_cfg.get("samples_per_size", 20),
        "refiner.topology_training.samples_per_size",
    )
    coverage_augmentation = bool(topo_cfg.get("coverage_augmentation", True))

    def _build() -> tuple[InternalValidationPlan, dict[str, float | int]]:
        sampled = sample_topology_evaluation_subgraphs(
            graph=train_graph,
            seed=seed,
            strategy=strategy,
            node_sizes=node_sizes,
            samples_per_size=samples_per_size,
        )
        stats: dict[str, float | int] = {}
        if coverage_augmentation:
            sampled, stats = augment_plan_for_positive_edge_coverage(
                graph=train_graph,
                base_sampled={int(k): list(v) for k, v in sampled.items()},
                node_sizes=node_sizes,
                strategy=strategy,
                seed=seed,
            )
        built_plan = build_internal_validation_plan(graph=train_graph, sampled_subgraphs=sampled)
        return built_plan, stats

    plan, coverage_stats = _load_or_build_topology_plan(
        split="train_topology",
        graph=train_graph,
        node_sizes=node_sizes,
        samples_per_size=samples_per_size,
        seed=seed,
        strategy=strategy,
        coverage_augmentation=coverage_augmentation,
        runtime=runtime,
        cache_dir=cache_dir,
        build_fn=_build,
    )
    if coverage_stats:
        LOGGER.info(
            "tccig train topology coverage: base=%s coverage=%s positive_edge_coverage=%.4f",
            coverage_stats.get("base_bucket_count"),
            coverage_stats.get("coverage_bucket_count"),
            float(coverage_stats.get("positive_edge_coverage", 0.0)),
        )
    pairs = [
        CandidatePair(record.protein_a, record.protein_b)
        for bucket in plan.buckets
        for record in bucket.pair_records
    ]
    probabilities = _score_split(
        split="train_topology",
        pairs=pairs,
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
    )
    pairwise_edges = edges_from_rule(
        pairs=pairs,
        probabilities=probabilities,
        rule=pairwise_input_rule,
    )
    return (
        SplitBundle(
            split="train_topology",
            pairs=pairs,
            pairwise_probabilities=probabilities,
            pairwise_graph_edges=pairwise_edges,
        ),
        plan,
        coverage_stats,
    )


def _build_validation_topology_bundle(
    *,
    config: Mapping[str, object],
    processed_dir: Path,
    scorer_cfg: Mapping[str, object],
    runtime: TCCIGRuntime,
    cache_dir: Path,
    pairwise_input_rule: GraphRule,
) -> tuple[SplitBundle | None, object | None]:
    refiner_cfg = _mapping_section(config, "refiner")
    topology_cfg = refiner_cfg.get("topology_validation", {})
    if not isinstance(topology_cfg, Mapping) or not bool(topology_cfg.get("enabled", False)):
        return None, None
    split_path = processed_dir / "human_BFS_split.pkl"
    node_ids = load_split_node_ids(split_path=split_path, split_name="train")
    validation_graph = build_pair_supervision_graph(
        pair_path=processed_dir / "human_val_ppi_ratio5_exclusive.txt",
        node_ids=node_ids,
    )
    sampled = sample_topology_evaluation_subgraphs(
        graph=validation_graph,
        seed=_non_negative_int(topology_cfg.get("seed", 0), "refiner.topology_validation.seed"),
        strategy=str(topology_cfg.get("strategy", "mixed")),
        node_sizes=_int_sequence(
            topology_cfg.get("node_sizes", TOPOLOGY_EVAL_NODE_SIZES),
            "refiner.topology_validation.node_sizes",
        ),
        samples_per_size=_positive_int(
            topology_cfg.get("samples_per_size", 20),
            "refiner.topology_validation.samples_per_size",
        ),
    )
    plan = build_internal_validation_plan(graph=validation_graph, sampled_subgraphs=sampled)
    pairs = [
        CandidatePair(record.protein_a, record.protein_b)
        for bucket in plan.buckets
        for record in bucket.pair_records
    ]
    probabilities = _score_split(
        split="validation_topology",
        pairs=pairs,
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
    )
    pairwise_edges = edges_from_rule(
        pairs=pairs,
        probabilities=probabilities,
        rule=pairwise_input_rule,
    )
    return (
        SplitBundle(
            split="validation_topology",
            pairs=pairs,
            pairwise_probabilities=probabilities,
            pairwise_graph_edges=pairwise_edges,
        ),
        plan,
    )


def _bundle_from_table(
    *,
    table: PairTable,
    probabilities: list[float],
    graph_rule: GraphRule,
    include_loss_targets: bool,
) -> SplitBundle:
    graph_edges = edges_from_rule(pairs=table.pairs, probabilities=probabilities, rule=graph_rule)
    labels = table.labels
    return SplitBundle(
        split=table.split,
        pairs=table.pairs,
        pairwise_probabilities=probabilities,
        pairwise_graph_edges=graph_edges,
        candidate_labels=labels,
        loss_targets=labels if include_loss_targets else None,
        graph_edges=table.positive_edges,
    )


def _resolve_pairwise_input_rule(
    *,
    config: Mapping[str, object],
    validation_scores: list[float],
    validation_labels: list[int],
) -> tuple[GraphRule, dict[str, object]]:
    graph_selection = _graph_selection(config)
    raw_rule = graph_selection.get("pairwise_input_threshold", None)
    source = "graph_selection.pairwise_input_threshold"
    if raw_rule is None:
        raw_rule = {"mode": "fixed", "value": DEFAULT_GRAPH_THRESHOLD}
        source = "default"
    if not isinstance(raw_rule, Mapping):
        raise ValueError("graph_selection.pairwise_input_threshold must be a mapping")
    mode = str(raw_rule.get("mode", "fixed")).lower()
    if mode == "fixed":
        value = _probability(
            raw_rule.get("value", DEFAULT_GRAPH_THRESHOLD),
            "graph_selection.pairwise_input_threshold.value",
        )
        metrics = binary_metrics_at_threshold(
            labels=validation_labels,
            probabilities=validation_scores,
            threshold=value,
        )
        return (
            GraphRule(type="threshold", value=value),
            {
                "type": "threshold",
                "mode": "fixed",
                "value": value,
                "source": source,
                "metrics": metrics,
            },
        )
    if mode == "target_precision":
        split = str(raw_rule.get("split", "validation")).lower()
        if split != "validation":
            raise ValueError("pairwise_input_threshold.target_precision must use validation split")
        target_precision = _probability(
            raw_rule.get("target_precision", raw_rule.get("value")),
            "graph_selection.pairwise_input_threshold.target_precision",
        )
        threshold, metrics = threshold_for_target_precision(
            probabilities=validation_scores,
            labels=validation_labels,
            target_precision=target_precision,
        )
        return (
            GraphRule(type="threshold", value=threshold),
            {
                "type": "threshold",
                "mode": "target_precision",
                "target_precision": target_precision,
                "value": threshold,
                "source": source,
                "split": "validation",
                "metrics": metrics,
            },
        )
    raise ValueError(
        "graph_selection.pairwise_input_threshold.mode must be fixed or target_precision"
    )


def _resolve_refined_output_rule(config: Mapping[str, object]) -> GraphRule:
    raw_rule = _graph_selection(config).get(
        "refined_output_rule",
        {"type": "threshold", "value": DEFAULT_GRAPH_THRESHOLD},
    )
    if not isinstance(raw_rule, Mapping):
        raise ValueError("graph_selection.refined_output_rule must be a mapping")
    if str(raw_rule.get("type", "threshold")).lower() != "threshold":
        raise ValueError("graph_selection.refined_output_rule only supports threshold")
    return GraphRule(
        type="threshold",
        value=_probability(
            raw_rule.get("value", DEFAULT_GRAPH_THRESHOLD),
            "graph_selection.refined_output_rule.value",
        ),
    )


def _collate_pair_score_batch(
    *,
    records: Sequence[PairScoreRecord],
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int | None,
) -> dict[str, torch.Tensor]:
    raw = collate_pair_score_records(records)
    protein_a = cast(list[str], raw["protein_a"])
    protein_b = cast(list[str], raw["protein_b"])
    embeddings_a = [
        load_cached_embedding(
            cache_dir=cache_dir,
            index=embedding_index,
            protein_id=protein_id,
            expected_input_dim=input_dim,
            max_sequence_length=max_sequence_length,
        )
        for protein_id in protein_a
    ]
    embeddings_b = [
        load_cached_embedding(
            cache_dir=cache_dir,
            index=embedding_index,
            protein_id=protein_id,
            expected_input_dim=input_dim,
            max_sequence_length=max_sequence_length,
        )
        for protein_id in protein_b
    ]
    return {
        "index": cast(torch.Tensor, raw["index"]),
        "emb_a": pad_sequence(embeddings_a, batch_first=True),
        "emb_b": pad_sequence(embeddings_b, batch_first=True),
        "len_a": torch.tensor([tensor.size(0) for tensor in embeddings_a], dtype=torch.long),
        "len_b": torch.tensor([tensor.size(0) for tensor in embeddings_b], dtype=torch.long),
    }


def _build_runtime(
    *,
    config: Mapping[str, object],
    build_accelerator_fn: AcceleratorFactory | None,
) -> TCCIGRuntime:
    if build_accelerator_fn is None:
        device_cfg = _mapping_section(config, "device")
        mixed_precision = _mixed_precision_mode(device_cfg.get("mixed_precision", "no"))
        accelerator = Accelerator(
            cpu=str(device_cfg.get("device", "")).lower() == "cpu",
            mixed_precision=mixed_precision,
            dataloader_config=DataLoaderConfiguration(
                even_batches=True,
                use_seedable_sampler=True,
            ),
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=bool(device_cfg.get("find_unused_parameters", False))
                )
            ],
        )
    else:
        accelerator = build_accelerator_fn()
    device = torch.device(getattr(accelerator, "device", "cpu"))
    world_size = int(getattr(accelerator, "num_processes", 1))
    device_cfg = _mapping_section(config, "device")
    return TCCIGRuntime(
        accelerator=cast(object, accelerator),
        device=str(device),
        backend=str(device_cfg.get("backend", "ddp")),
        mixed_precision=_mixed_precision_mode(device_cfg.get("mixed_precision", "no")),
        is_distributed=world_size > 1 or bool(getattr(accelerator, "use_distributed", False)),
        rank=int(getattr(accelerator, "process_index", 0)),
        local_rank=int(getattr(accelerator, "local_process_index", 0)),
        world_size=world_size,
        is_main_process=bool(getattr(accelerator, "is_main_process", True)),
    )


def _prepared_model_and_loader(prepared: object) -> tuple[nn.Module, object]:
    if not isinstance(prepared, tuple) or len(prepared) != 2:
        raise TypeError("accelerator.prepare(model, loader) must return a two-item tuple")
    model, loader = prepared
    if not isinstance(model, nn.Module):
        raise TypeError("accelerator.prepare returned a non-module model")
    return model, loader


def _load_v3_1_abba_no_cross_model_config(path: Path) -> dict[str, object]:
    payload = _load_yaml_config(path)
    model_cfg = payload.get("model_config")
    if not isinstance(model_cfg, Mapping):
        raise ValueError("pairwise_scorer.model_config_path must contain model_config")
    if str(model_cfg.get("model", "")).lower() != "v3.1":
        raise ValueError("model_config.model must be 'v3.1'")
    interaction = model_cfg.get("interaction", {})
    if not isinstance(interaction, Mapping) or str(interaction.get("mode", "none")) != "none":
        raise ValueError("model_config.interaction.mode must be 'none'")
    pair_readout = model_cfg.get("pair_readout", {})
    if (
        not isinstance(pair_readout, Mapping)
        or str(pair_readout.get("order_aggregation", "")) != "abba_max"
    ):
        raise ValueError("model_config.pair_readout.order_aggregation must be 'abba_max'")
    return dict(payload)


def _score_cache_enabled(scorer_cfg: Mapping[str, object]) -> bool:
    raw = scorer_cfg.get("score_cache", {})
    return not isinstance(raw, Mapping) or bool(raw.get("enabled", True))


def _load_score_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
) -> list[float] | None:
    path = cache_dir / "scores" / f"{split}.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("metadata") != metadata:
        return None
    probabilities = payload.get("probabilities")
    if not isinstance(probabilities, torch.Tensor):
        return None
    return [float(value) for value in probabilities.tolist()]


def _write_score_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
    probabilities: Sequence[float],
) -> None:
    scores_path = cache_dir / "scores" / f"{split}.pt"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": dict(metadata),
            "probabilities": torch.tensor([float(value) for value in probabilities]),
        },
        scores_path,
    )
    write_json(cache_dir / "manifests" / f"{split}.json", dict(metadata))


def _load_embedding_index(index_path: Path) -> dict[str, str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Embedding index must be a JSON object: {index_path}")
    return {str(key): str(value) for key, value in payload.items()}


def _load_yaml_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return cast(dict[str, object], payload)


def _graph_selection(config: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping_section(config, "graph_selection")


def _mapping_section(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = config.get(name, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return section


def _run_id(config: Mapping[str, object]) -> str:
    run_cfg = _mapping_section(config, "run")
    return str(run_cfg.get("run_id", "tccig_run"))


def _log_root(config: Mapping[str, object]) -> Path:
    run_cfg = _mapping_section(config, "run")
    return Path(str(run_cfg.get("log_root", "logs")))


def _cache_root(config: Mapping[str, object]) -> Path:
    run_cfg = _mapping_section(config, "run")
    return Path(str(run_cfg.get("cache_root", "data/tccig")))


def _sampling_seed(config: Mapping[str, object]) -> int:
    refiner_cfg = _mapping_section(config, "refiner")
    edge_sampling = refiner_cfg.get("edge_sampling", {})
    if isinstance(edge_sampling, Mapping):
        return _non_negative_int(edge_sampling.get("seed", 0), "refiner.edge_sampling.seed")
    return 0


def _required_path(config: Mapping[str, object], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return Path(value)


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(cast(int | str, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        parsed = int(cast(int | str, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a non-negative integer") from error
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _optional_positive_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


def _mixed_precision_mode(value: object) -> str:
    if isinstance(value, bool):
        return "fp16" if value else "no"
    mode = str(value).strip().lower()
    if mode in {"no", "fp16", "bf16"}:
        return mode
    raise ValueError("device.mixed_precision must be one of: no, fp16, bf16")


def _probability(value: object, field_name: str) -> float:
    try:
        parsed = float(cast(float | str, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be in [0, 1]") from error
    if math.isnan(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return parsed


def _int_sequence(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence")
    parsed = tuple(_positive_int(item, field_name) for item in value)
    if not parsed:
        raise ValueError(f"{field_name} must not be empty")
    return parsed


def _runtime_barrier(runtime: TCCIGRuntime) -> None:
    wait_for_everyone = getattr(runtime.accelerator, "wait_for_everyone", None)
    if callable(wait_for_everyone):
        wait_for_everyone()


def _configure_logging(runtime: TCCIGRuntime) -> None:
    level = logging.INFO if runtime.is_main_process else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    main()
