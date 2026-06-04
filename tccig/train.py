"""Standalone PRING-aligned TCCIG IO and orchestration entrypoint."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
import math
import pickle
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import networkx as nx
import torch
import torch.distributed as _dist
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
from src.embed import load_cached_embedding
from src.pipeline.stages.train import build_model
from src.topology.finetune_data import (
    TOPOLOGY_EVAL_NODE_SIZES,
    InternalValidationPlan,
    build_internal_validation_plan,
    build_pair_supervision_graph,
    load_split_node_ids,
    sample_topology_evaluation_subgraphs,
)
from torch.nn.utils.rnn import pad_sequence

from tccig.io import CandidatePair, PairTable, canonical_edge, read_pair_table, write_json
from tccig.rules import GraphRule, edges_from_rule, parse_rules, select_rule

LOGGER = logging.getLogger(__name__)

TOPOLOGY_METRIC_NAMES = [
    "graph_sim",
    "relative_density",
    "deg_dist_mmd",
    "cc_mmd",
    "laplacian_eigen_mmd",
]
TOPOLOGY_CSV_COLUMNS = [
    "scope",
    "node_size",
    "graph_count",
    *TOPOLOGY_METRIC_NAMES,
]
ProgressCallback = Callable[[Mapping[str, object]], None]


class AcceleratorLike(Protocol):
    """Minimal accelerator surface needed by the standalone scaffold."""

    device: object

    def prepare(self, *args: object) -> object:
        """Prepare trainable objects for the configured runtime."""
        ...

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate a loss tensor through the configured runtime."""
        ...


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
    is_distributed: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_main_process: bool = True


@dataclass(frozen=True)
class PairwiseScoreRequest:
    """Label-free pairwise scoring request."""

    split: str
    pairs: Sequence[CandidatePair]
    runtime: TCCIGRuntime
    config: Mapping[str, object]
    progress_callback: ProgressCallback | None = None


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
    graph_rules: tuple[GraphRule, ...] = ()
    validation_topology: SplitBundle | None = None
    validation_topology_plan: InternalValidationPlan | None = None


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


def score_pairs_with_v3_1(request: PairwiseScoreRequest) -> list[float]:
    """Score TCCIG candidate pairs with a checkpoint-backed v3.1 pairwise model."""
    model_config_path = _required_path(request.config, "model_config_path", "pairwise_scorer")
    checkpoint_path = _required_path(request.config, "checkpoint_path", "pairwise_scorer")
    embedding_cache_dir = _required_path(
        request.config,
        "embedding_cache_dir",
        "pairwise_scorer",
    )
    batch_size = _positive_int(
        request.config.get("batch_size", 32),
        "pairwise_scorer.batch_size",
    )
    max_sequence_length = _positive_int(
        request.config.get("max_sequence_length"),
        "pairwise_scorer.max_sequence_length",
    )
    progress_every_batches = _positive_int(
        request.config.get("progress_every_batches", 1),
        "pairwise_scorer.progress_every_batches",
    )

    model_config = _load_v3_1_abba_no_cross_model_config(model_config_path)
    input_dim = _positive_int(model_config.get("input_dim"), "model_config.input_dim")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"pairwise_scorer.checkpoint_path does not exist: {checkpoint_path}"
        )

    embedding_index = _load_embedding_index(embedding_cache_dir / "index.json")
    device = torch.device(request.runtime.device)
    model = build_model({"model_config": model_config})
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError("pairwise_scorer.checkpoint_path must contain a model state dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    probabilities: list[float] = []
    with torch.inference_mode():
        for batch_index, batch_pairs in enumerate(
            _chunk_pairs(request.pairs, batch_size),
            start=1,
        ):
            batch = _build_v3_1_pair_batch(
                pairs=batch_pairs,
                cache_dir=embedding_cache_dir,
                embedding_index=embedding_index,
                input_dim=input_dim,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            output = model(batch)
            logits = cast(torch.Tensor, output["logits"])
            reduced_logits = (
                logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            )
            probabilities.extend(torch.sigmoid(reduced_logits).detach().cpu().tolist())
            if request.progress_callback is not None and (
                batch_index % progress_every_batches == 0
                or len(probabilities) == len(request.pairs)
            ):
                request.progress_callback(
                    {
                        "batch_index": batch_index,
                        "processed_pairs": len(probabilities),
                        "local_pair_count": len(request.pairs),
                    }
                )
    return [float(probability) for probability in probabilities]


def _load_v3_1_abba_no_cross_model_config(model_config_path: Path) -> dict[str, object]:
    """Load and validate the v3.1 abba-no-cross pairwise architecture config."""
    payload = _load_yaml_config(model_config_path)
    model_config_raw = payload.get("model_config")
    if not isinstance(model_config_raw, Mapping):
        raise ValueError("pairwise_scorer.model_config_path must contain model_config")
    model_config = dict(model_config_raw)

    _require_config_value(
        model_config.get("model"),
        "v3.1",
        "model_config.model",
    )
    pair_readout = _required_mapping(model_config, "pair_readout", "model_config")
    _require_config_value(
        pair_readout.get("mode"),
        "pair_context_gated",
        "model_config.pair_readout.mode",
    )
    _require_config_value(
        pair_readout.get("order_aggregation"),
        "abba_max",
        "model_config.pair_readout.order_aggregation",
    )
    interaction = _required_mapping(model_config, "interaction", "model_config")
    _require_config_value(
        interaction.get("mode"),
        "none",
        "model_config.interaction.mode",
    )
    return model_config


def _required_mapping(
    config: Mapping[str, object],
    key: str,
    namespace: str,
) -> Mapping[str, object]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{namespace}.{key} must be a mapping")
    return value


def _require_config_value(value: object, expected: str, field_name: str) -> None:
    if str(value).lower() != expected:
        raise ValueError(f"{field_name} must be {expected!r} for the TCCIG v3.1 scorer")


def _required_path(config: Mapping[str, object], key: str, namespace: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{namespace}.{key} is required")
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


def _load_embedding_index(index_path: Path) -> dict[str, str]:
    if not index_path.exists():
        raise FileNotFoundError(f"pairwise_scorer.embedding_cache_dir missing index: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Embedding index must be a JSON object")
    index: dict[str, str] = {}
    for protein_id, relative_path in payload.items():
        if not isinstance(protein_id, str) or not isinstance(relative_path, str):
            raise ValueError("Embedding index must map protein IDs to relative paths")
        index[protein_id] = relative_path
    return index


def _chunk_pairs(
    pairs: Sequence[CandidatePair],
    batch_size: int,
) -> Iterator[Sequence[CandidatePair]]:
    for start in range(0, len(pairs), batch_size):
        yield pairs[start : start + batch_size]


def _build_v3_1_pair_batch(
    *,
    pairs: Sequence[CandidatePair],
    cache_dir: Path,
    embedding_index: Mapping[str, str],
    input_dim: int,
    max_sequence_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    embeddings: dict[str, torch.Tensor] = {}
    for pair in pairs:
        for protein_id in (pair.protein_a, pair.protein_b):
            if protein_id not in embeddings:
                embeddings[protein_id] = load_cached_embedding(
                    cache_dir=cache_dir,
                    index=embedding_index,
                    protein_id=protein_id,
                    expected_input_dim=input_dim,
                    max_sequence_length=max_sequence_length,
                )

    emb_a = pad_sequence([embeddings[pair.protein_a] for pair in pairs], batch_first=True)
    emb_b = pad_sequence([embeddings[pair.protein_b] for pair in pairs], batch_first=True)
    len_a = torch.tensor(
        [embeddings[pair.protein_a].size(0) for pair in pairs],
        dtype=torch.long,
    )
    len_b = torch.tensor(
        [embeddings[pair.protein_b].size(0) for pair in pairs],
        dtype=torch.long,
    )
    return {
        "emb_a": emb_a.to(device),
        "emb_b": emb_b.to(device),
        "len_a": len_a.to(device),
        "len_b": len_b.to(device),
    }


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
    _configure_tccig_logging(runtime)
    run_id = _run_id(config)
    log_root = _log_root(config)
    _preflight_tccig_dimensions(config)
    tables = _load_tables(config)
    self_pair_rows = {split: table.self_pair_rows for split, table in tables.items()}

    pairwise_scorer = _load_pairwise_scorer(config)
    pairwise_graph_rule = _pairwise_graph_rule(config)
    scorer_cfg = _mapping_section(config, "pairwise_scorer")
    rules = tuple(parse_rules(_mapping_section(config, "graph_selection").get("rules")))
    score_dir = log_root / "tccig" / "score" / run_id
    train_scored = _score_table_stage(
        table=tables["train"],
        split="train",
        scorer=pairwise_scorer,
        runtime=runtime,
        scorer_cfg=scorer_cfg,
        pairwise_graph_rule=pairwise_graph_rule,
        output_dir=score_dir,
    )
    validation_scored = _score_table_stage(
        table=tables["validation"],
        split="validation",
        scorer=pairwise_scorer,
        runtime=runtime,
        scorer_cfg=scorer_cfg,
        pairwise_graph_rule=pairwise_graph_rule,
        output_dir=score_dir,
    )

    train_bundle = _with_targets(train_scored, tables["train"], include_loss=True)
    validation_bundle = _with_targets(
        validation_scored,
        tables["validation"],
        include_loss=False,
    )

    refiner_cfg = _refiner_runtime_config(
        config=_mapping_section(config, "refiner"),
        run_id=run_id,
        log_root=log_root,
    )
    validation_topology_bundle: SplitBundle | None = None
    validation_topology_plan: InternalValidationPlan | None = None
    if _topology_validation_enabled(refiner_cfg):
        validation_topology_bundle, validation_topology_plan = _build_validation_topology_bundle(
            config=config,
            refiner_cfg=refiner_cfg,
            scorer=pairwise_scorer,
            runtime=runtime,
            scorer_cfg=scorer_cfg,
            pairwise_graph_rule=pairwise_graph_rule,
            output_dir=score_dir,
        )
        if runtime.is_main_process:
            write_json(
                score_dir / "validation_topology.json",
                {
                    "split": "validation_topology",
                    "pair_count": len(validation_topology_bundle.pairs),
                    "pairwise_graph_edge_count": len(
                        validation_topology_bundle.pairwise_graph_edges
                    ),
                    "validation_topology_subgraphs": validation_topology_plan.total_subgraphs,
                    "validation_topology_pairs": validation_topology_plan.total_pairs,
                    "validation_topology_node_sizes": [
                        bucket.node_size for bucket in validation_topology_plan.buckets
                    ],
                },
            )
        _runtime_barrier(runtime)
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
            graph_rules=rules,
            validation_topology=validation_topology_bundle,
            validation_topology_plan=validation_topology_plan,
        )
    )

    selected_rule, selected_rule_payload = _selected_rule_from_refiner_state(
        refiner_state=refiner_state,
    )
    if selected_rule is None or selected_rule_payload is None:
        selected_rule, selected_rule_payload = _fallback_select_validation_rule(
            validation_bundle=validation_bundle,
            validation_labels=tables["validation"].labels,
            predict_refined=cast(PredictRefined, predict_refined),
            refiner_state=refiner_state,
            runtime=runtime,
            refiner_cfg=refiner_cfg,
            rules=rules,
        )
    selected_rule_path = log_root / "tccig" / "validation" / run_id / "selected_rule.json"
    write_json(selected_rule_path, selected_rule_payload)

    pairwise_test_bundle = _score_table_stage(
        table=tables["pairwise_test"],
        split="pairwise_test",
        scorer=pairwise_scorer,
        runtime=runtime,
        scorer_cfg=scorer_cfg,
        pairwise_graph_rule=pairwise_graph_rule,
        output_dir=score_dir,
    )
    pairwise_metrics = _run_pairwise_test(
        bundle=pairwise_test_bundle,
        labels=tables["pairwise_test"].labels,
        output_dir=log_root / "tccig" / "pairwise_test" / run_id,
    )
    topology_test_bundle = _score_table_stage(
        table=tables["topology_test"],
        split="topology_test",
        scorer=pairwise_scorer,
        runtime=runtime,
        scorer_cfg=scorer_cfg,
        pairwise_graph_rule=pairwise_graph_rule,
        output_dir=score_dir,
    )
    topology_metrics = _run_topology_test(
        config=config,
        bundle=topology_test_bundle,
        predict_refined=cast(PredictRefined, predict_refined),
        refiner_state=refiner_state,
        selected_rule=selected_rule,
        pairwise_graph_rule=pairwise_graph_rule,
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
    output_dir: Path,
) -> SplitBundle:
    return _score_pairs(
        pairs=table.pairs,
        split=split,
        scorer=scorer,
        runtime=runtime,
        scorer_cfg=scorer_cfg,
        pairwise_graph_rule=pairwise_graph_rule,
        output_dir=output_dir,
    )


def _score_table_stage(
    *,
    table: PairTable,
    split: str,
    scorer: PairwiseScorer,
    runtime: TCCIGRuntime,
    scorer_cfg: Mapping[str, object],
    pairwise_graph_rule: GraphRule,
    output_dir: Path,
) -> SplitBundle:
    bundle = _score_table(
        table=table,
        split=split,
        scorer=scorer,
        runtime=runtime,
        scorer_cfg=scorer_cfg,
        pairwise_graph_rule=pairwise_graph_rule,
        output_dir=output_dir,
    )
    if runtime.is_main_process:
        _write_scoring_manifest(output_dir=output_dir, table=table, bundle=bundle)
    _runtime_barrier(runtime)
    return bundle


def _score_pairs(
    *,
    pairs: Sequence[CandidatePair],
    split: str,
    scorer: PairwiseScorer,
    runtime: TCCIGRuntime,
    scorer_cfg: Mapping[str, object],
    pairwise_graph_rule: GraphRule,
    output_dir: Path,
) -> SplitBundle:
    candidate_pairs = list(pairs)
    local_indexed_pairs = _rank_local_indexed_pairs(
        pairs=candidate_pairs,
        runtime=runtime,
    )
    local_pairs = [pair for _, pair in local_indexed_pairs]
    cache_metadata = _score_cache_metadata(
        split=split,
        pairs=candidate_pairs,
        scorer_cfg=scorer_cfg,
    )
    cached_probabilities = _load_cached_pairwise_probabilities(
        output_dir=output_dir,
        split=split,
        expected_metadata=cache_metadata,
    )
    if cached_probabilities is not None:
        graph_edges = edges_from_rule(
            pairs=candidate_pairs,
            probabilities=cached_probabilities,
            rule=pairwise_graph_rule,
        )
        return SplitBundle(
            split=split,
            pairs=candidate_pairs,
            pairwise_probabilities=cached_probabilities,
            pairwise_graph_edges=graph_edges,
        )
    raw_scores = scorer(
        PairwiseScoreRequest(
            split=split,
            pairs=local_pairs,
            runtime=runtime,
            config=scorer_cfg,
        )
    )
    local_probabilities = _normalize_probabilities(raw_scores)
    if len(local_probabilities) != len(local_pairs):
        raise ValueError(
            f"pairwise scorer returned {len(local_probabilities)} scores for "
            f"{len(local_pairs)} rank-local {split} pairs"
        )
    indexed_scores = [
        (pair_index, probability)
        for (pair_index, _), probability in zip(
            local_indexed_pairs,
            local_probabilities,
            strict=True,
        )
    ]
    probabilities = _ordered_scores_from_rank_shards(
        total_pairs=len(candidate_pairs),
        local_indexed_scores=indexed_scores,
        runtime=runtime,
    )
    graph_edges = edges_from_rule(
        pairs=candidate_pairs,
        probabilities=probabilities,
        rule=pairwise_graph_rule,
    )
    _write_pairwise_score_cache(
        output_dir=output_dir,
        split=split,
        metadata=cache_metadata,
        probabilities=probabilities,
        runtime=runtime,
    )
    return SplitBundle(
        split=split,
        pairs=candidate_pairs,
        pairwise_probabilities=probabilities,
        pairwise_graph_edges=graph_edges,
    )


def _score_cache_metadata(
    *,
    split: str,
    pairs: Sequence[CandidatePair],
    scorer_cfg: Mapping[str, object],
) -> dict[str, object] | None:
    if not _score_cache_enabled(scorer_cfg):
        return None
    return {
        "version": 1,
        "split": split,
        "pair_count": len(pairs),
        "pair_hash": _ordered_pair_hash(pairs),
        "scorer": _score_cache_scorer_fingerprint(scorer_cfg),
    }


def _score_cache_enabled(scorer_cfg: Mapping[str, object]) -> bool:
    cache_cfg = scorer_cfg.get("score_cache")
    return isinstance(cache_cfg, Mapping) and bool(cache_cfg.get("enabled", False))


def _score_cache_scorer_fingerprint(scorer_cfg: Mapping[str, object]) -> dict[str, object]:
    return {
        "target": str(scorer_cfg.get("target", "")),
        "model_config_sha256": _optional_config_file_sha256(
            scorer_cfg.get("model_config_path"),
        ),
        "checkpoint_sha256": _optional_config_file_sha256(
            scorer_cfg.get("checkpoint_path"),
        ),
        "embedding_index_sha256": _optional_embedding_index_sha256(scorer_cfg),
        "max_sequence_length": scorer_cfg.get("max_sequence_length"),
    }


def _optional_embedding_index_sha256(scorer_cfg: Mapping[str, object]) -> str | None:
    raw_cache_dir = scorer_cfg.get("embedding_cache_dir")
    if raw_cache_dir is None:
        return None
    return _file_sha256(Path(str(raw_cache_dir)) / "index.json")


def _optional_config_file_sha256(raw_path: object) -> str | None:
    if raw_path is None:
        return None
    return _file_sha256(Path(str(raw_path)))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_pair_hash(pairs: Sequence[CandidatePair]) -> str:
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(pair.protein_a.encode("utf-8"))
        digest.update(b"\0")
        digest.update(pair.protein_b.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_cached_pairwise_probabilities(
    *,
    output_dir: Path,
    split: str,
    expected_metadata: Mapping[str, object] | None,
) -> list[float] | None:
    if expected_metadata is None:
        return None
    cache_path = _score_cache_path(output_dir=output_dir, split=split)
    if not cache_path.exists():
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu")
    except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("metadata") != dict(expected_metadata):
        return None
    probabilities = payload.get("probabilities")
    if not isinstance(probabilities, torch.Tensor):
        return None
    if probabilities.dim() != 1 or int(probabilities.numel()) != int(
        expected_metadata["pair_count"]
    ):
        return None
    return [float(value) for value in probabilities.to(dtype=torch.float32).tolist()]


def _write_pairwise_score_cache(
    *,
    output_dir: Path,
    split: str,
    metadata: Mapping[str, object] | None,
    probabilities: Sequence[float],
    runtime: TCCIGRuntime,
) -> None:
    if metadata is None or not runtime.is_main_process:
        return
    cache_path = _score_cache_path(output_dir=output_dir, split=split)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": dict(metadata),
            "probabilities": torch.tensor(
                [float(probability) for probability in probabilities],
                dtype=torch.float32,
            ),
        },
        cache_path,
    )


def _score_cache_path(*, output_dir: Path, split: str) -> Path:
    return output_dir / "cache" / f"{split}.pt"


def _build_validation_topology_bundle(
    *,
    config: Mapping[str, object],
    refiner_cfg: Mapping[str, object],
    scorer: PairwiseScorer,
    runtime: TCCIGRuntime,
    scorer_cfg: Mapping[str, object],
    pairwise_graph_rule: GraphRule,
    output_dir: Path,
) -> tuple[SplitBundle, InternalValidationPlan]:
    data_cfg = _mapping_section(config, "data")
    processed_dir = Path(str(data_cfg["processed_dir"]))
    train_nodes = load_split_node_ids(
        split_path=processed_dir / "human_BFS_split.pkl",
        split_name="train",
    )
    validation_graph = build_pair_supervision_graph(
        pair_path=processed_dir / "human_val_ppi_ratio5_exclusive.txt",
        node_ids=train_nodes,
    )
    topology_cfg = _mapping_section(refiner_cfg, "topology_validation")
    sampled_subgraphs = sample_topology_evaluation_subgraphs(
        graph=validation_graph,
        seed=_non_negative_int(topology_cfg.get("seed", 0), "refiner.topology_validation.seed"),
        strategy=str(topology_cfg.get("strategy", "mixed")),
        node_sizes=_topology_validation_node_sizes(topology_cfg),
        samples_per_size=_positive_int(
            topology_cfg.get("samples_per_size", 20),
            "refiner.topology_validation.samples_per_size",
        ),
    )
    validation_plan = build_internal_validation_plan(
        graph=validation_graph,
        sampled_subgraphs=sampled_subgraphs,
    )
    bundle = _score_pairs(
        pairs=_unique_validation_topology_pairs(validation_plan),
        split="validation_topology",
        scorer=scorer,
        runtime=runtime,
        scorer_cfg=scorer_cfg,
        pairwise_graph_rule=pairwise_graph_rule,
        output_dir=output_dir,
    )
    return bundle, validation_plan


def _topology_validation_enabled(refiner_cfg: Mapping[str, object]) -> bool:
    monitor_metric = str(refiner_cfg.get("monitor_metric", "val_auprc"))
    topology_cfg = refiner_cfg.get("topology_validation")
    if isinstance(topology_cfg, Mapping) and "enabled" in topology_cfg:
        return bool(topology_cfg["enabled"])
    return monitor_metric != "val_auprc"


def _topology_validation_node_sizes(topology_cfg: Mapping[str, object]) -> tuple[int, ...]:
    raw_node_sizes = topology_cfg.get("node_sizes", TOPOLOGY_EVAL_NODE_SIZES)
    if not isinstance(raw_node_sizes, Sequence) or isinstance(raw_node_sizes, (str, bytes)):
        raise ValueError("refiner.topology_validation.node_sizes must be a sequence")
    node_sizes = [
        _positive_int(node_size, "refiner.topology_validation.node_sizes")
        for node_size in raw_node_sizes
    ]
    if not node_sizes:
        raise ValueError("refiner.topology_validation.node_sizes must not be empty")
    return tuple(dict.fromkeys(node_sizes))


def _unique_validation_topology_pairs(plan: InternalValidationPlan) -> list[CandidatePair]:
    edges: set[tuple[str, str]] = set()
    for bucket in plan.buckets:
        for record in bucket.pair_records:
            edges.add(canonical_edge(record.protein_a, record.protein_b))
    return [CandidatePair(protein_a=edge[0], protein_b=edge[1]) for edge in sorted(edges)]


def _selected_rule_from_refiner_state(
    *,
    refiner_state: object,
) -> tuple[GraphRule | None, dict[str, object] | None]:
    selected_rule = getattr(refiner_state, "selected_rule", None)
    selected_rule_payload = getattr(refiner_state, "selected_rule_payload", None)
    if isinstance(selected_rule, GraphRule) and isinstance(selected_rule_payload, dict):
        return selected_rule, selected_rule_payload
    return None, None


def _rank_local_indexed_pairs(
    *,
    pairs: Sequence[CandidatePair],
    runtime: TCCIGRuntime,
) -> list[tuple[int, CandidatePair]]:
    """Return candidate pairs owned by the current rank in stable file order."""
    if not runtime.is_distributed:
        return list(enumerate(pairs))
    return [
        (pair_index, pair)
        for pair_index, pair in enumerate(pairs)
        if pair_index % runtime.world_size == runtime.rank
    ]


def _ordered_scores_from_rank_shards(
    *,
    total_pairs: int,
    local_indexed_scores: Sequence[tuple[int, float]],
    runtime: TCCIGRuntime,
    gather_fn: Callable[
        [Sequence[tuple[int, float]]],
        Sequence[Sequence[tuple[int, float]]],
    ]
    | None = None,
) -> list[float]:
    """Gather rank-local scores and restore original candidate-pair order."""
    if not runtime.is_distributed:
        return _ordered_scores_from_shards(
            total_pairs=total_pairs,
            shard_payloads=[local_indexed_scores],
        )
    if gather_fn is not None:
        gathered_scores = list(gather_fn(local_indexed_scores))
    elif _dist.is_available() and _dist.is_initialized():
        gathered_payloads: list[list[tuple[int, float]] | None] = [None] * runtime.world_size
        _dist.all_gather_object(gathered_payloads, list(local_indexed_scores))
        gathered_scores = [payload for payload in gathered_payloads if payload is not None]
    else:
        gathered_scores = [local_indexed_scores]
    return _ordered_scores_from_shards(
        total_pairs=total_pairs,
        shard_payloads=gathered_scores,
    )


def _ordered_scores_from_shards(
    *,
    total_pairs: int,
    shard_payloads: Sequence[Sequence[tuple[int, float]]],
) -> list[float]:
    ordered: list[float | None] = [None] * total_pairs
    for shard in shard_payloads:
        for pair_index, score in shard:
            if pair_index < 0 or pair_index >= total_pairs:
                raise ValueError(f"Rank score index out of range: {pair_index}")
            if ordered[pair_index] is not None:
                raise ValueError(f"Duplicate rank score for pair index {pair_index}")
            ordered[pair_index] = float(score)
    missing = [index for index, score in enumerate(ordered) if score is None]
    if missing:
        raise ValueError(f"Missing rank scores for pair indices: {missing[:10]}")
    return [float(score) for score in ordered]


def _fallback_select_validation_rule(
    *,
    validation_bundle: SplitBundle,
    validation_labels: list[int],
    predict_refined: PredictRefined,
    refiner_state: object,
    runtime: TCCIGRuntime,
    refiner_cfg: Mapping[str, object],
    rules: Sequence[GraphRule],
) -> tuple[GraphRule, dict[str, object]]:
    validation_refined = _predict_refined_probabilities(
        predict_refined=predict_refined,
        split="validation",
        bundle=validation_bundle,
        refiner_state=refiner_state,
        runtime=runtime,
        refiner_cfg=refiner_cfg,
    )
    selected_rule, selected_rule_metrics = select_rule(
        pairs=validation_bundle.pairs,
        probabilities=validation_refined,
        labels=validation_labels,
        rules=list(rules),
    )
    return selected_rule, {
        **selected_rule.to_dict(),
        "validation_metrics": selected_rule_metrics,
    }


def _write_scoring_manifest(
    *,
    output_dir: Path,
    table: PairTable,
    bundle: SplitBundle,
) -> None:
    write_json(
        output_dir / f"{bundle.split}.json",
        {
            "split": bundle.split,
            "path": str(table.path),
            "pair_count": len(bundle.pairs),
            "self_pair_rows_dropped": table.self_pair_rows,
            "pairwise_graph_edge_count": len(bundle.pairwise_graph_edges),
        },
    )


def _write_scoring_manifests(
    *,
    output_dir: Path,
    tables: Mapping[str, PairTable],
    bundles: Mapping[str, SplitBundle],
) -> None:
    """Persist one label-safe scoring manifest per split."""
    for split, table in tables.items():
        _write_scoring_manifest(output_dir=output_dir, table=table, bundle=bundles[split])


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
    probabilities = _normalize_probabilities(raw_scores)
    if len(probabilities) != len(bundle.pairs):
        raise ValueError(
            f"refiner returned {len(probabilities)} scores for {len(bundle.pairs)} pairs"
        )
    return probabilities


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
    pairwise_graph_rule: GraphRule,
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
    data_cfg = _mapping_section(config, "data")
    processed_dir = Path(str(data_cfg["processed_dir"]))
    with (processed_dir / "human_test_graph.pkl").open("rb") as handle:
        gt_graph = cast(nx.Graph, pickle.load(handle))
    with (processed_dir / "test_sampled_nodes.pkl").open("rb") as handle:
        test_sampled_nodes = cast(dict[int, list[list[str]]], pickle.load(handle))
    topology_result = _evaluate_predicted_graph_sharded(
        pred_graph=_reconstruct_graph(selected_edges),
        gt_graph=gt_graph,
        test_graph_nodes=test_sampled_nodes,
        runtime=runtime,
    )
    raw_summary = cast(Mapping[str, object], topology_result["summary"])
    summary = {name: _object_to_float(value) for name, value in raw_summary.items()}
    if runtime.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_topology_predictions(
            output_path=output_dir / "all_test_ppi_pred.txt",
            pairs=bundle.pairs,
            selected_edges=selected_edges,
        )
        write_json(
            output_dir / "topology_metrics.json",
            {
                "summary": summary,
                "per_node_size": topology_result["per_node_size"],
                "details": topology_result["details"],
                "selected_rule": selected_rule.to_dict(),
                "pairwise_graph_rule": pairwise_graph_rule.to_dict(),
                "pair_counts": {
                    "candidate_pairs": len(bundle.pairs),
                    "pairwise_graph_edges": len(bundle.pairwise_graph_edges),
                    "refined_positive_edges": len(selected_edges),
                },
                "protocol": {
                    "candidate_universe": "all_test_ppi.txt",
                    "ground_truth_graph": "human_test_graph.pkl",
                    "sampled_nodes": "test_sampled_nodes.pkl",
                    "test_labels_visible_to_model": False,
                },
                "runtime": {
                    "is_distributed": runtime.is_distributed,
                    "rank": runtime.rank,
                    "world_size": runtime.world_size,
                },
            },
        )
        _write_topology_metrics_csv(
            csv_path=output_dir / "topology_metrics.csv",
            per_node_size=cast(dict[int, dict[str, float | int]], topology_result["per_node_size"]),
            summary=summary,
        )
    _runtime_barrier(runtime)
    return summary


def _write_topology_predictions(
    *,
    output_path: Path,
    pairs: Sequence[CandidatePair],
    selected_edges: set[tuple[str, str]],
) -> None:
    """Write PRING hard-label topology predictions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            edge = canonical_edge(pair.protein_a, pair.protein_b)
            handle.write(f"{pair.protein_a}\t{pair.protein_b}\t{int(edge in selected_edges)}\n")


def _write_topology_metrics_csv(
    *,
    csv_path: Path,
    per_node_size: Mapping[int, Mapping[str, float | int]],
    summary: Mapping[str, float],
) -> None:
    """Persist official per-node-size and summary PRING topology metrics."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOPOLOGY_CSV_COLUMNS)
        writer.writeheader()
        for node_size in sorted(int(node_size) for node_size in per_node_size):
            values = per_node_size[node_size]
            writer.writerow(
                {
                    "scope": "node_size",
                    "node_size": node_size,
                    **{name: values.get(name, "") for name in TOPOLOGY_METRIC_NAMES},
                    "graph_count": int(values.get("graph_count", 0)),
                }
            )
        writer.writerow(
            {
                "scope": "summary",
                "node_size": "all",
                "graph_count": sum(
                    int(values.get("graph_count", 0)) for values in per_node_size.values()
                ),
                **{name: summary.get(name, "") for name in TOPOLOGY_METRIC_NAMES},
            }
        )


def _empty_graph_evaluation_result() -> dict[str, Any]:
    """Return an empty graph-evaluation payload for ranks with no node-size buckets."""
    return {"details": {}, "summary": {}, "per_node_size": {}}


def _shard_test_graph_nodes_for_rank(
    *,
    test_graph_nodes: Mapping[int, list[list[str]]],
    runtime: TCCIGRuntime,
) -> dict[int, list[list[str]]]:
    """Return topology-test node-size buckets assigned to the current TCCIG rank."""
    normalized = {
        int(node_size): list(node_lists) for node_size, node_lists in test_graph_nodes.items()
    }
    if not runtime.is_distributed:
        return normalized
    ordered_node_sizes = sorted(normalized, reverse=True)
    local_node_sizes = ordered_node_sizes[runtime.rank :: runtime.world_size]
    return {node_size: normalized[node_size] for node_size in sorted(local_node_sizes)}


def _evaluate_predicted_graph_sharded(
    *,
    pred_graph: nx.Graph,
    gt_graph: nx.Graph,
    test_graph_nodes: Mapping[int, list[list[str]]],
    runtime: TCCIGRuntime,
    gather_fn: Callable[[dict[str, Any]], Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Evaluate PRING topology metrics on rank-local node-size buckets and merge them."""
    if not runtime.is_distributed:
        return dict(
            _evaluate_predicted_graph(
                pred_graph=pred_graph,
                gt_graph=gt_graph,
                test_graph_nodes=test_graph_nodes,
            )
        )
    if gather_fn is None and (not _dist.is_available() or not _dist.is_initialized()):
        return dict(
            _evaluate_predicted_graph(
                pred_graph=pred_graph,
                gt_graph=gt_graph,
                test_graph_nodes=test_graph_nodes,
            )
        )

    local_test_graph_nodes = _shard_test_graph_nodes_for_rank(
        test_graph_nodes=test_graph_nodes,
        runtime=runtime,
    )
    local_result = (
        dict(
            _evaluate_predicted_graph(
                pred_graph=pred_graph,
                gt_graph=gt_graph,
                test_graph_nodes=local_test_graph_nodes,
            )
        )
        if local_test_graph_nodes
        else _empty_graph_evaluation_result()
    )
    if gather_fn is not None:
        gathered_results = list(gather_fn(local_result))
    else:
        gathered_payloads: list[dict[str, Any] | None] = [None] * runtime.world_size
        _dist.all_gather_object(gathered_payloads, local_result)
        gathered_results = [
            cast(Mapping[str, Any], payload)
            for payload in gathered_payloads
            if payload is not None
        ]
    return _merge_graph_sample_evaluations(shard_results=gathered_results)


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


def _preflight_tccig_dimensions(config: Mapping[str, object]) -> None:
    """Fail fast when scorer, refiner, and embedding cache dimensions disagree."""
    scorer_cfg = _mapping_section(config, "pairwise_scorer")
    if scorer_cfg.get("target") != "tccig.train:score_pairs_with_v3_1":
        return

    model_config_path = _required_path(scorer_cfg, "model_config_path", "pairwise_scorer")
    embedding_cache_dir = _required_path(
        scorer_cfg,
        "embedding_cache_dir",
        "pairwise_scorer",
    )
    model_config = _load_v3_1_abba_no_cross_model_config(model_config_path)
    scorer_input_dim = _positive_int(
        model_config.get("input_dim"),
        "pairwise_scorer.model_config.input_dim",
    )
    refiner_cfg = _mapping_section(config, "refiner")
    refiner_input_dim = _preflight_refiner_input_dim(
        refiner_cfg=refiner_cfg,
        scorer_input_dim=scorer_input_dim,
    )
    embedding_cache_dim = _first_embedding_cache_dim(embedding_cache_dir)
    if len({scorer_input_dim, refiner_input_dim, embedding_cache_dim}) == 1:
        return
    raise ValueError(
        "TCCIG preflight dimension mismatch: "
        f"pairwise_scorer.model_config.input_dim={scorer_input_dim}, "
        f"refiner.input_dim={refiner_input_dim}, "
        f"embedding_cache_dim={embedding_cache_dim}"
    )


def _first_embedding_cache_dim(cache_dir: Path) -> int:
    embedding_index = _load_embedding_index(cache_dir / "index.json")
    if not embedding_index:
        raise ValueError("pairwise_scorer.embedding_cache_dir index must not be empty")
    protein_id = sorted(embedding_index)[0]
    embedding = load_cached_embedding(
        cache_dir=cache_dir,
        index=embedding_index,
        protein_id=protein_id,
    )
    return int(embedding.size(1))


def _preflight_refiner_input_dim(
    *,
    refiner_cfg: Mapping[str, object],
    scorer_input_dim: int,
) -> int:
    raw_input_dim = refiner_cfg.get("input_dim")
    if raw_input_dim is not None:
        return _positive_int(raw_input_dim, "refiner.input_dim")
    if refiner_cfg.get("train_target") == "tccig.s2gae:train_refiner":
        return _positive_int(1024, "refiner.input_dim")
    return scorer_input_dim


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
    is_distributed = bool(getattr(accelerator, "use_distributed", False))
    rank = _non_negative_int(getattr(accelerator, "process_index", 0), "accelerator.rank")
    local_rank = _non_negative_int(
        getattr(accelerator, "local_process_index", rank),
        "accelerator.local_rank",
    )
    world_size = _positive_int(
        getattr(accelerator, "num_processes", 1),
        "accelerator.world_size",
    )
    return TCCIGRuntime(
        device=str(accelerator.device),
        backend=backend,
        mixed_precision=mixed_precision,
        accelerator=accelerator,
        is_distributed=is_distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main_process=bool(getattr(accelerator, "is_main_process", rank == 0)),
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
) -> Mapping[str, Any]:
    """Lazy-load the repo PRING topology metric helper."""
    metrics_module = importlib.import_module("src.topology.metrics")
    evaluate_fn = metrics_module.evaluate_predicted_graph
    return cast(
        Mapping[str, Any],
        evaluate_fn(
            pred_graph=pred_graph,
            gt_graph=gt_graph,
            test_graph_nodes=test_graph_nodes,
        ),
    )


def _merge_graph_sample_evaluations(
    *,
    shard_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Lazy-load the repo PRING graph-sample merge helper."""
    metrics_module = importlib.import_module("src.topology.metrics")
    merge_fn = metrics_module.merge_graph_sample_evaluations
    return cast(dict[str, Any], merge_fn(shard_results=shard_results))


def _runtime_barrier(runtime: TCCIGRuntime) -> None:
    """Synchronize TCCIG ranks when the accelerator exposes a barrier."""
    wait_for_everyone = getattr(runtime.accelerator, "wait_for_everyone", None)
    if callable(wait_for_everyone):
        wait_for_everyone()


def _configure_tccig_logging(runtime: TCCIGRuntime) -> None:
    """Configure standalone TCCIG logging for Slurm stdout/stderr."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO if runtime.is_main_process else logging.CRITICAL,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    LOGGER.info(
        "TCCIG runtime initialized: backend=%s distributed=%s rank=%s world_size=%s device=%s",
        runtime.backend,
        runtime.is_distributed,
        runtime.rank,
        runtime.world_size,
        runtime.device,
    )


def _pairwise_graph_rule(config: Mapping[str, object]) -> GraphRule:
    graph_cfg = _mapping_section(config, "graph_selection")
    raw_rule = graph_cfg.get("pairwise_graph_rule", {"type": "threshold", "value": 0.5})
    return parse_rules([raw_rule])[0]


def _refiner_runtime_config(
    *,
    config: Mapping[str, object],
    run_id: str,
    log_root: Path,
) -> Mapping[str, object]:
    """Add orchestrator-owned run context to refiner hook config."""
    return {
        **config,
        "_run_id": run_id,
        "_log_root": str(log_root),
    }


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
