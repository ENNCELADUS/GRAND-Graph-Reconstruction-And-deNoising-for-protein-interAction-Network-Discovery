"""Topology evaluation stage for PRING-style Human graph reconstruction."""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as _dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.embed import ensure_embeddings_ready
from src.evaluate import DEFAULT_DECISION_THRESHOLD
from src.pipeline.loops import (
    forward_model,
    gather_indexed_predictions,
    move_batch_to_device,
)
from src.pipeline.runtime import AcceleratorLike, DistributedContext, PipelineRuntime
from src.pipeline.stages.evaluate import _resolve_decision_threshold
from src.topology import (
    evaluate_predicted_graph,
    load_human_table2_baselines,
    merge_graph_sample_evaluations,
    reconstruct_graph,
    write_human_table2_reports,
)
from src.utils.config import (
    ConfigDict,
    as_bool,
    as_int,
    as_str,
    extract_model_kwargs,
    get_section,
)
from src.utils.data_io import PRINGPairDataset, _collate_batch
from src.utils.logging import append_csv_row, format_result_payload, log_stage_event

dist = _dist

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
EXPECTED_STRATEGIES = {"BFS", "DFS", "RANDOM_WALK"}


@dataclass(frozen=True)
class GraphAssemblyResult:
    """TCCIG graph-assembly predictions and observability diagnostics."""

    predictions: list[int]
    probabilities: list[float]
    m_hat: float
    candidate_count: int
    selected_edges: int
    assembly_rule: str = "top_m_hat"


class TopologyLoaderBundle(tuple):
    """Three-item loader tuple with access to the backing PRING dataset."""

    data_loader: DataLoader[dict[str, object]]
    records: list[tuple[str, str]]
    cached_embedding_count: int
    dataset: PRINGPairDataset

    def __new__(
        cls,
        *,
        data_loader: DataLoader[dict[str, object]],
        records: list[tuple[str, str]],
        cached_embedding_count: int,
        dataset: PRINGPairDataset,
    ) -> TopologyLoaderBundle:
        """Create a backwards-compatible three-item topology loader bundle."""
        value = super().__new__(cls, (data_loader, records, cached_embedding_count))
        value.data_loader = data_loader
        value.records = records
        value.cached_embedding_count = cached_embedding_count
        value.dataset = dataset
        return value


def write_topology_predictions(
    *,
    output_path: Path,
    records: Sequence[tuple[str, str]],
    predictions: Sequence[int],
) -> None:
    """Write PRING-format topology predictions."""
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have the same length")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for (protein_a, protein_b), prediction in zip(records, predictions, strict=True):
            handle.write(f"{protein_a}\t{protein_b}\t{int(prediction)}\n")


def _topology_config(config: ConfigDict) -> ConfigDict:
    """Return topology configuration mapping."""
    topology_cfg = config.get("topology_evaluate", {})
    if not isinstance(topology_cfg, dict):
        raise ValueError("topology_evaluate must be a mapping")
    return cast(ConfigDict, topology_cfg)


def _topology_paths(config: ConfigDict) -> tuple[Path, Path, Path]:
    """Resolve Human topology input paths from processed directory."""
    data_cfg = get_section(config, "data_config")
    benchmark_cfg = get_section(data_cfg, "benchmark")
    processed_dir = Path(str(benchmark_cfg.get("processed_dir", "")))
    species = as_str(benchmark_cfg.get("species", "human"), "data_config.benchmark.species")
    if species.lower() != "human":
        raise ValueError("topology_evaluate currently supports Human PRING topology only")
    all_test_path = processed_dir / "all_test_ppi.txt"
    gt_graph_path = processed_dir / f"{species}_test_graph.pkl"
    sampled_nodes_path = processed_dir / "test_sampled_nodes.pkl"
    for path in (all_test_path, gt_graph_path, sampled_nodes_path):
        if not path.exists():
            raise FileNotFoundError(f"Topology evaluation input not found: {path}")
    return all_test_path, gt_graph_path, sampled_nodes_path


class _IndexedTopologyDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset wrapper that attaches the original pair index to each sample."""

    def __init__(self, dataset: PRINGPairDataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = dict(self._dataset[index])
        item["pair_index"] = torch.tensor(index, dtype=torch.long)
        return item


def _collate_topology_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate topology batches while preserving original sample indices."""
    collated = _collate_batch(batch)
    collated["pair_index"] = torch.stack([sample["pair_index"] for sample in batch], dim=0)
    return collated


def _build_topology_loader(
    *,
    config: ConfigDict,
    split_path: Path,
) -> TopologyLoaderBundle:
    """Build deterministic topology inference loader for embedding-backed models."""
    model_cfg = get_section(config, "model_config")
    data_cfg = get_section(config, "data_config")
    dataloader_cfg = get_section(data_cfg, "dataloader")
    training_cfg = get_section(config, "training_config")
    topology_cfg = _topology_config(config)

    input_dim = as_int(model_cfg.get("input_dim", 0), "model_config.input_dim")
    max_sequence_length = as_int(
        data_cfg.get("max_sequence_length", 64),
        "data_config.max_sequence_length",
    )
    valid_path = Path(str(dataloader_cfg.get("valid_dataset", "")))
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation dataset path not found: {valid_path}")

    embedding_cache = ensure_embeddings_ready(
        config=config,
        split_paths=[valid_path, split_path],
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        allow_generation=True,
    )
    dataset = PRINGPairDataset(
        file_path=split_path,
        input_dim=input_dim,
        max_sequence_length=max_sequence_length,
        cache_dir=embedding_cache.cache_dir,
        embedding_index=embedding_cache.index,
        cache_embeddings_in_memory=as_bool(
            topology_cfg.get("cache_embeddings_in_memory", True),
            "topology_evaluate.cache_embeddings_in_memory",
        ),
    )
    all_records = [(record.protein_a, record.protein_b) for record in dataset.pair_records()]
    preload_embeddings = as_bool(
        topology_cfg.get("preload_embeddings", True),
        "topology_evaluate.preload_embeddings",
    )
    if preload_embeddings:
        dataset.preload_embeddings(dataset.protein_ids())
    batch_size = as_int(
        topology_cfg.get("inference_batch_size", training_cfg.get("batch_size", 8)),
        "topology_evaluate.inference_batch_size",
    )
    num_workers = as_int(
        dataloader_cfg.get("num_workers", 0),
        "data_config.dataloader.num_workers",
    )
    if preload_embeddings:
        num_workers = 0
    loader = DataLoader(
        dataset=_IndexedTopologyDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=as_bool(
            dataloader_cfg.get("pin_memory", False),
            "data_config.dataloader.pin_memory",
        ),
        drop_last=False,
        collate_fn=_collate_topology_batch,
    )
    return TopologyLoaderBundle(
        data_loader=cast(DataLoader[dict[str, object]], loader),
        records=all_records,
        cached_embedding_count=len(dataset._embedding_cache),
        dataset=dataset,
    )


def _predict_topology_labels(
    *,
    model: torch.nn.Module,
    data_loader: DataLoader[dict[str, object]],
    device: torch.device,
    total_records: int,
    decision_threshold: float,
    accelerator: AcceleratorLike,
) -> list[int]:
    """Predict probabilities and thresholded labels for all topology pairs."""
    gathered_indices: list[int] = []
    gathered_predictions: list[int] = []
    with torch.inference_mode():
        for batch in data_loader:
            prepared_batch = move_batch_to_device(batch=batch, device=device)
            batch_index_tensor = cast(torch.Tensor, prepared_batch.pop("pair_index"))
            with accelerator.autocast():
                output = forward_model(model=model, batch=prepared_batch)
            logits = output["logits"]
            reduced_logits = (
                logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            )
            batch_predictions = torch.tensor(
                [
                    int(float(value) >= decision_threshold)
                    for value in torch.sigmoid(reduced_logits).detach().cpu().tolist()
                ],
                dtype=torch.long,
                device=accelerator.device,
            )
            if accelerator.use_distributed:
                batch_index_tensor = accelerator.gather_for_metrics(batch_index_tensor)
                batch_predictions = accelerator.gather_for_metrics(batch_predictions)
            gathered_indices.extend(
                int(index) for index in batch_index_tensor.detach().cpu().tolist()
            )
            gathered_predictions.extend(
                int(prediction) for prediction in batch_predictions.detach().cpu().tolist()
            )
    ordered: list[int | None] = [None] * total_records
    for index, prediction in zip(gathered_indices, gathered_predictions, strict=True):
        ordered[int(index)] = int(prediction)
    missing = [index for index, prediction in enumerate(ordered) if prediction is None]
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        raise ValueError(f"Missing topology predictions for indices: {preview}")
    return [int(prediction) for prediction in ordered if prediction is not None]


def _model_supports_graph_forward(model: torch.nn.Module, accelerator: AcceleratorLike) -> bool:
    """Return whether a model exposes the TCCIG graph-forward contract."""
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    candidate = cast(torch.nn.Module, unwrap_model(model)) if callable(unwrap_model) else model
    return callable(getattr(candidate, "forward_graph", None))


def _unwrap_graph_model(
    *,
    model: torch.nn.Module,
    accelerator: AcceleratorLike,
) -> torch.nn.Module:
    """Return the module object that owns graph-forward helper methods."""
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    return cast(torch.nn.Module, unwrap_model(model)) if callable(unwrap_model) else model


def _tccig_topology_eval_config(config: ConfigDict) -> ConfigDict:
    """Return optional ``topology_evaluate.tccig`` config."""
    topology_cfg = _topology_config(config)
    raw_config = topology_cfg.get("tccig", {})
    if raw_config is None:
        return {}
    if not isinstance(raw_config, dict):
        raise ValueError("topology_evaluate.tccig must be a mapping")
    return cast(ConfigDict, raw_config)


def _resolve_tccig_candidate_batch_size(config: ConfigDict) -> int:
    """Return candidate chunk size for TCCIG graph assembly."""
    topology_cfg = _topology_config(config)
    tccig_cfg = _tccig_topology_eval_config(config)
    training_cfg = get_section(config, "training_config")
    fallback_batch_size = as_int(
        topology_cfg.get("inference_batch_size", training_cfg.get("batch_size", 8)),
        "topology_evaluate.inference_batch_size",
    )
    batch_size = as_int(
        tccig_cfg.get("candidate_batch_size", max(1024, fallback_batch_size)),
        "topology_evaluate.tccig.candidate_batch_size",
    )
    if batch_size <= 0:
        raise ValueError("topology_evaluate.tccig.candidate_batch_size must be > 0")
    return batch_size


def _load_tccig_node_inputs(
    *,
    dataset: PRINGPairDataset,
    records: Sequence[tuple[str, str]],
    device: torch.device,
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    """Load unique topology-evaluation protein embeddings for graph assembly."""
    protein_ids = tuple(sorted({protein for record in records for protein in record}))
    if len(protein_ids) < 2:
        raise ValueError("TCCIG topology evaluation requires at least two proteins")
    embedding_tensors = [dataset._load_embedding(protein_id) for protein_id in protein_ids]
    protein_embeddings = pad_sequence(embedding_tensors, batch_first=True).to(device)
    protein_lengths = torch.tensor(
        [embedding.size(0) for embedding in embedding_tensors],
        dtype=torch.long,
        device=device,
    )
    return protein_ids, protein_embeddings, protein_lengths


def _candidate_pairs_for_records(
    *,
    records: Sequence[tuple[str, str]],
    node_to_index: Mapping[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Map PRING pair records to local graph candidate-pair indices."""
    return torch.tensor(
        [
            [node_to_index[protein_a] for protein_a, _ in records],
            [node_to_index[protein_b] for _, protein_b in records],
        ],
        dtype=torch.long,
        device=device,
    )


def _assemble_top_m_hat_predictions(
    *,
    probabilities: Sequence[float],
    m_hat: float,
) -> list[int]:
    """Select the top-``m_hat`` candidate edges and return hard PRING labels."""
    edge_budget = max(0, min(len(probabilities), int(round(m_hat))))
    predictions = [0 for _ in probabilities]
    if edge_budget == 0:
        return predictions
    top_indices = sorted(
        range(len(probabilities)),
        key=lambda index: (-float(probabilities[index]), index),
    )[:edge_budget]
    for index in top_indices:
        predictions[index] = 1
    return predictions


def _probability_stats(probabilities: Sequence[float]) -> dict[str, float]:
    """Return stable summary statistics for graph-assembly probabilities."""
    if not probabilities:
        return {
            "probability_min": 0.0,
            "probability_mean": 0.0,
            "probability_max": 0.0,
            "probability_p50": 0.0,
            "probability_p90": 0.0,
            "probability_p95": 0.0,
        }
    probability_tensor = torch.tensor(probabilities, dtype=torch.float32)
    return {
        "probability_min": float(torch.min(probability_tensor).item()),
        "probability_mean": float(torch.mean(probability_tensor).item()),
        "probability_max": float(torch.max(probability_tensor).item()),
        "probability_p50": float(torch.quantile(probability_tensor, 0.50).item()),
        "probability_p90": float(torch.quantile(probability_tensor, 0.90).item()),
        "probability_p95": float(torch.quantile(probability_tensor, 0.95).item()),
    }


def graph_assembly_diagnostics(result: GraphAssemblyResult) -> dict[str, float | int | str]:
    """Build the persisted diagnostics payload for a TCCIG graph assembly."""
    return {
        "assembly_rule": result.assembly_rule,
        "m_hat": float(result.m_hat),
        "record_count": len(result.predictions),
        "candidate_count": int(result.candidate_count),
        "selected_edges": int(result.selected_edges),
    } | _probability_stats(result.probabilities)


def write_graph_assembly_diagnostics(
    *,
    output_path: Path,
    result: GraphAssemblyResult,
) -> dict[str, float | int | str]:
    """Persist graph-assembly diagnostics and return the payload."""
    payload = graph_assembly_diagnostics(result)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(format_result_payload(payload), handle, indent=2, sort_keys=True)
    return payload


def _predict_tccig_graph_assembly_result(
    *,
    config: ConfigDict,
    model: torch.nn.Module,
    dataset: PRINGPairDataset,
    records: Sequence[tuple[str, str]],
    device: torch.device,
    accelerator: AcceleratorLike,
) -> GraphAssemblyResult:
    """Run TCCIG Graph Assembly with top-``m_hat`` candidate selection."""
    graph_model = _unwrap_graph_model(model=model, accelerator=accelerator)
    encode_graph_nodes = getattr(graph_model, "encode_graph_nodes", None)
    decode_graph_candidates = getattr(graph_model, "decode_graph_candidates", None)
    edge_budget_from_node_embeddings = getattr(
        graph_model,
        "edge_budget_from_node_embeddings",
        None,
    )
    if not (
        callable(encode_graph_nodes)
        and callable(decode_graph_candidates)
        and callable(edge_budget_from_node_embeddings)
    ):
        raise ValueError(
            "TCCIG topology evaluation requires encode_graph_nodes, "
            "decode_graph_candidates, and edge_budget_from_node_embeddings"
        )

    scorable_records_with_indices = [
        (index, record) for index, record in enumerate(records) if record[0] != record[1]
    ]
    if not scorable_records_with_indices:
        return GraphAssemblyResult(
            predictions=[0] * len(records),
            probabilities=[0.0] * len(records),
            m_hat=0.0,
            candidate_count=0,
            selected_edges=0,
        )

    scorable_indices = [index for index, _ in scorable_records_with_indices]
    scorable_records = [record for _, record in scorable_records_with_indices]
    protein_ids, protein_embeddings, protein_lengths = _load_tccig_node_inputs(
        dataset=dataset,
        records=scorable_records,
        device=device,
    )
    node_to_index = {protein_id: index for index, protein_id in enumerate(protein_ids)}
    candidate_batch_size = _resolve_tccig_candidate_batch_size(config)
    probabilities: list[float] = []
    with torch.inference_mode(), accelerator.autocast():
        node_embeddings = cast(
            torch.Tensor,
            encode_graph_nodes(
                protein_embeddings=protein_embeddings,
                protein_lengths=protein_lengths,
            ),
        )
        m_hat_tensor = cast(
            torch.Tensor,
            edge_budget_from_node_embeddings(
                node_embeddings=node_embeddings,
                candidate_count=len(scorable_records),
            ),
        )
        for start in range(0, len(scorable_records), candidate_batch_size):
            candidate_pairs = _candidate_pairs_for_records(
                records=scorable_records[start : start + candidate_batch_size],
                node_to_index=node_to_index,
                device=device,
            )
            output = cast(
                dict[str, torch.Tensor],
                decode_graph_candidates(
                    node_embeddings=node_embeddings,
                    candidate_pairs=candidate_pairs,
                ),
            )
            probabilities.extend(
                float(value) for value in output["edge_probabilities"].detach().cpu().tolist()
            )
    if len(probabilities) != len(scorable_records):
        raise ValueError("TCCIG graph assembly did not score every topology pair")
    scorable_predictions = _assemble_top_m_hat_predictions(
        probabilities=probabilities,
        m_hat=float(m_hat_tensor.detach().cpu().item()),
    )
    predictions = [0] * len(records)
    full_probabilities = [0.0] * len(records)
    for index, probability, prediction in zip(
        scorable_indices,
        probabilities,
        scorable_predictions,
        strict=True,
    ):
        full_probabilities[index] = probability
        predictions[index] = prediction
    return GraphAssemblyResult(
        predictions=predictions,
        probabilities=full_probabilities,
        m_hat=float(m_hat_tensor.detach().cpu().item()),
        candidate_count=len(scorable_records),
        selected_edges=sum(scorable_predictions),
    )


def _predict_tccig_graph_assembly_labels(
    *,
    config: ConfigDict,
    model: torch.nn.Module,
    dataset: PRINGPairDataset,
    records: Sequence[tuple[str, str]],
    device: torch.device,
    accelerator: AcceleratorLike,
) -> list[int]:
    """Run TCCIG Graph Assembly and return hard labels only."""
    return _predict_tccig_graph_assembly_result(
        config=config,
        model=model,
        dataset=dataset,
        records=records,
        device=device,
        accelerator=accelerator,
    ).predictions


def _ordered_predictions_from_shards(
    *,
    total_records: int,
    shard_payloads: Sequence[Mapping[str, Sequence[int]]],
) -> list[int]:
    """Restore original prediction order from rank-local shards."""
    ordered: list[int | None] = [None] * total_records
    for shard_payload in shard_payloads:
        indices = [int(index) for index in shard_payload["indices"]]
        predictions = [int(prediction) for prediction in shard_payload["predictions"]]
        if len(indices) != len(predictions):
            raise ValueError("Shard indices and predictions must have matching lengths")
        for index, prediction in zip(indices, predictions, strict=True):
            if index < 0 or index >= total_records:
                raise ValueError(f"Shard index out of bounds: {index}")
            if ordered[index] is not None:
                raise ValueError(f"Duplicate prediction for pair index {index}")
            ordered[index] = prediction

    missing_indices = [index for index, prediction in enumerate(ordered) if prediction is None]
    if missing_indices:
        preview = ", ".join(str(index) for index in missing_indices[:10])
        raise ValueError(f"Missing topology predictions for indices: {preview}")
    return [int(prediction) for prediction in ordered if prediction is not None]


def _gather_ordered_predictions(
    *,
    local_indices: Sequence[int],
    local_predictions: Sequence[int],
    total_records: int,
    distributed_context: DistributedContext,
    accelerator: AcceleratorLike,
) -> list[int]:
    """Gather local predictions from all ranks and restore original order on every rank."""
    del distributed_context
    return gather_indexed_predictions(
        accelerator,
        indices=list(local_indices),
        predictions=list(local_predictions),
        total_records=total_records,
    )


def _json_safe_details(
    details: dict[str, dict[int, list[float] | float]],
) -> dict[str, dict[str, Any]]:
    """Convert integer node-size keys into JSON-safe strings."""
    return {
        metric_name: {str(node_size): values for node_size, values in values_by_size.items()}
        for metric_name, values_by_size in details.items()
    }


def _json_safe_per_node_size(
    per_node_size: dict[int, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    """Convert integer node-size keys into JSON-safe strings."""
    return {str(node_size): values for node_size, values in per_node_size.items()}


def _write_topology_metrics_csv(
    *,
    csv_path: Path,
    per_node_size: dict[int, dict[str, float | int]],
    summary: dict[str, float],
) -> None:
    """Persist per-node-size and summary topology metrics."""
    for node_size in sorted(per_node_size):
        row = {"scope": "node_size", "node_size": node_size, **per_node_size[node_size]}
        append_csv_row(csv_path=csv_path, row=row, fieldnames=TOPOLOGY_CSV_COLUMNS)
    append_csv_row(
        csv_path=csv_path,
        row={
            "scope": "summary",
            "node_size": "all",
            "graph_count": sum(int(values["graph_count"]) for values in per_node_size.values()),
            **summary,
        },
        fieldnames=TOPOLOGY_CSV_COLUMNS,
    )


def _latest_strategy_metrics(log_root: Path) -> dict[str, dict[str, float]]:
    """Return the latest topology summaries per strategy when all are available."""
    latest_by_strategy: dict[str, tuple[float, dict[str, Any]]] = {}
    for metrics_path in log_root.glob("*/topology_metrics.json"):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        strategy = str(payload.get("split_strategy", "")).upper()
        if strategy not in EXPECTED_STRATEGIES:
            continue
        modified_time = metrics_path.stat().st_mtime
        if strategy not in latest_by_strategy or modified_time > latest_by_strategy[strategy][0]:
            latest_by_strategy[strategy] = (modified_time, payload)
    if set(latest_by_strategy) != EXPECTED_STRATEGIES:
        return {}
    return {
        strategy: cast(dict[str, float], latest_by_strategy[strategy][1]["summary"])
        for strategy in EXPECTED_STRATEGIES
    }


def _maybe_write_comparison_report(
    *,
    config: ConfigDict,
    model_name: str,
    logger: logging.Logger,
) -> None:
    """Write the PRING-style Human Table 2 comparison if all strategies are available."""
    topology_cfg = _topology_config(config)
    baseline_path_value = topology_cfg.get("report_baselines")
    if not isinstance(baseline_path_value, str) or not baseline_path_value.strip():
        return
    baseline_path = Path(baseline_path_value)
    if not baseline_path.exists():
        log_stage_event(
            logger,
            "comparison_report_skipped",
            reason=f"missing_baselines:{baseline_path}",
        )
        return
    strategy_metrics = _latest_strategy_metrics(Path("logs") / model_name / "topology_evaluate")
    if not strategy_metrics:
        log_stage_event(logger, "comparison_report_skipped", reason="incomplete_strategy_set")
        return
    baselines = load_human_table2_baselines(baseline_path)
    output_dir = Path("artifacts") / "reports" / "pring"
    csv_path, markdown_path = write_human_table2_reports(
        output_dir=output_dir,
        baselines=baselines,
        model_name=model_name,
        model_category="GRAND",
        strategy_metrics=strategy_metrics,
    )
    log_stage_event(
        logger,
        "comparison_report_written",
        csv_path=csv_path,
        markdown_path=markdown_path,
    )


def _empty_graph_evaluation_result() -> dict[str, Any]:
    """Return an empty graph-evaluation payload for ranks with no assigned buckets."""
    return {"details": {}, "summary": {}, "per_node_size": {}}


def _shard_test_graph_nodes_for_rank(
    *,
    test_graph_nodes: Mapping[int, list[list[str]]],
    distributed_context: DistributedContext,
) -> dict[int, list[list[str]]]:
    """Return the node-size buckets assigned to the current rank.

    Buckets are assigned in descending node-size order to reduce the risk that one rank
    receives only the largest and most expensive evaluations.
    """
    if not distributed_context.is_distributed:
        return {
            int(node_size): list(node_lists) for node_size, node_lists in test_graph_nodes.items()
        }

    ordered_node_sizes = sorted((int(node_size) for node_size in test_graph_nodes), reverse=True)
    local_node_sizes = ordered_node_sizes[
        distributed_context.rank :: distributed_context.world_size
    ]
    return {node_size: list(test_graph_nodes[node_size]) for node_size in sorted(local_node_sizes)}


def _evaluate_predicted_graph_sharded(
    *,
    pred_graph: torch.Tensor | object,
    gt_graph: torch.Tensor | object,
    test_graph_nodes: Mapping[int, list[list[str]]],
    distributed_context: DistributedContext,
) -> dict[str, Any]:
    """Evaluate topology metrics on rank-local node-size buckets and merge the results."""
    if (
        not distributed_context.is_distributed
        or not dist.is_available()
        or not dist.is_initialized()
    ):
        return evaluate_predicted_graph(
            pred_graph=cast(Any, pred_graph),
            gt_graph=cast(Any, gt_graph),
            test_graph_nodes=test_graph_nodes,
        )

    local_test_graph_nodes = _shard_test_graph_nodes_for_rank(
        test_graph_nodes=test_graph_nodes,
        distributed_context=distributed_context,
    )
    local_result = (
        evaluate_predicted_graph(
            pred_graph=cast(Any, pred_graph),
            gt_graph=cast(Any, gt_graph),
            test_graph_nodes=local_test_graph_nodes,
        )
        if local_test_graph_nodes
        else _empty_graph_evaluation_result()
    )
    gathered_results: list[dict[str, Any] | None] = [None] * distributed_context.world_size
    dist.all_gather_object(gathered_results, local_result)
    return merge_graph_sample_evaluations(
        shard_results=[
            cast(Mapping[str, Any], shard_result)
            for shard_result in gathered_results
            if shard_result is not None
        ]
    )


def run_topology_evaluation_stage(
    runtime: PipelineRuntime,
    model: torch.nn.Module,
    dataloaders: dict[str, DataLoader[dict[str, object]]],
    *,
    checkpoint_path: Path,
) -> dict[str, float]:
    """Run PRING-style Human topology evaluation and persist artifacts."""
    config = runtime.config.raw
    device = runtime.device
    topology_cfg = _topology_config(config)
    evaluate_cfg = get_section(config, "evaluate")
    checkpoint_path_resolved = Path(checkpoint_path)
    model_name, _ = extract_model_kwargs(config)
    run_id = runtime.stage_run_id("topology_evaluate")
    paths = runtime.stage_paths("topology_evaluate")
    log_dir = paths.log_dir
    logger = runtime.stage_logger("topology_evaluate", log_dir / "log.log")
    if runtime.is_main_process:
        log_stage_event(logger, "stage_start", run_id=run_id, checkpoint=checkpoint_path_resolved)
    runtime.load_checkpoint(model, checkpoint_path_resolved)
    model.eval()

    threshold_cfg: ConfigDict = {
        "decision_threshold": topology_cfg.get(
            "decision_threshold",
            evaluate_cfg.get("decision_threshold", DEFAULT_DECISION_THRESHOLD),
        )
    }
    decision_threshold, threshold_mode = _resolve_decision_threshold(
        eval_cfg=threshold_cfg,
    )
    if runtime.is_main_process:
        log_stage_event(logger, "decision_threshold", mode=threshold_mode, value=decision_threshold)

    all_test_path, gt_graph_path, sampled_nodes_path = _topology_paths(config)
    topology_bundle = _build_topology_loader(
        config=config,
        split_path=all_test_path,
    )
    topology_loader, records, cached_embedding_count = topology_bundle
    topology_loader = cast(
        DataLoader[dict[str, object]],
        runtime.accelerator.prepare(topology_loader),
    )
    if runtime.is_main_process:
        log_stage_event(
            logger,
            "topology_inference_ready",
            pair_count=len(records),
            cached_embedding_count=cached_embedding_count,
            distributed=runtime.is_distributed,
            world_size=runtime.world_size,
        )
    graph_assembly_payload: dict[str, float | int | str] | None = None
    if _model_supports_graph_forward(model, runtime.accelerator):
        graph_assembly_result = _predict_tccig_graph_assembly_result(
            config=config,
            model=model,
            dataset=topology_bundle.dataset,
            records=records,
            device=device,
            accelerator=runtime.accelerator,
        )
        predictions = graph_assembly_result.predictions
        if runtime.is_main_process:
            graph_assembly_payload = write_graph_assembly_diagnostics(
                output_path=log_dir / "graph_assembly_diagnostics.json",
                result=graph_assembly_result,
            )
            log_stage_event(
                logger,
                "tccig_graph_assembly",
                pair_count=len(records),
                assembly_rule=graph_assembly_result.assembly_rule,
                m_hat=graph_assembly_result.m_hat,
                selected_edges=graph_assembly_result.selected_edges,
                candidate_batch_size=_resolve_tccig_candidate_batch_size(config),
            )
            log_stage_event(
                logger,
                "graph_assembly_diagnostics_written",
                path=log_dir / "graph_assembly_diagnostics.json",
            )
    else:
        predictions = _predict_topology_labels(
            model=model,
            data_loader=topology_loader,
            device=device,
            total_records=len(records),
            decision_threshold=decision_threshold,
            accelerator=runtime.accelerator,
        )

    prediction_path = log_dir / "all_test_ppi_pred.txt"
    if runtime.is_main_process and as_bool(
        topology_cfg.get("save_pair_predictions", True),
        "topology_evaluate.save_pair_predictions",
    ):
        write_topology_predictions(
            output_path=prediction_path,
            records=records,
            predictions=predictions,
        )
        log_stage_event(logger, "pair_predictions_written", path=prediction_path)

    predicted_edges = [
        (protein_a, protein_b)
        for (protein_a, protein_b), prediction in zip(records, predictions, strict=True)
        if prediction > 0
    ]
    pred_graph = reconstruct_graph(predicted_edges)
    with gt_graph_path.open("rb") as handle:
        gt_graph = pickle.load(handle)
    with sampled_nodes_path.open("rb") as handle:
        test_graph_nodes = pickle.load(handle)
    topology_result = _evaluate_predicted_graph_sharded(
        pred_graph=pred_graph,
        gt_graph=gt_graph,
        test_graph_nodes=test_graph_nodes,
        distributed_context=runtime.distributed,
    )

    if runtime.is_main_process:
        with (log_dir / "graph_eval_results.pkl").open("wb") as handle:
            pickle.dump(topology_result["details"], handle)
        with (log_dir / "topology_metrics.json").open("w", encoding="utf-8") as handle:
            data_cfg = get_section(config, "data_config")
            benchmark_cfg = get_section(data_cfg, "benchmark")
            payload = {
                "model": model_name,
                "run_id": run_id,
                "species": as_str(
                    benchmark_cfg.get("species", "human"),
                    "data_config.benchmark.species",
                ),
                "split_strategy": as_str(
                    benchmark_cfg.get("split_strategy", "BFS"),
                    "data_config.benchmark.split_strategy",
                ).upper(),
                "decision_threshold": decision_threshold,
                "summary": topology_result["summary"],
                "per_node_size": _json_safe_per_node_size(topology_result["per_node_size"]),
                "details": _json_safe_details(topology_result["details"]),
            }
            if graph_assembly_payload is not None:
                payload["graph_assembly"] = graph_assembly_payload
            json.dump(
                format_result_payload(payload),
                handle,
                indent=2,
                sort_keys=True,
            )
        _write_topology_metrics_csv(
            csv_path=log_dir / "topology_metrics.csv",
            per_node_size=topology_result["per_node_size"],
            summary=topology_result["summary"],
        )
        log_stage_event(logger, "topology_metrics_written", path=log_dir / "topology_metrics.json")
        _maybe_write_comparison_report(config=config, model_name=model_name, logger=logger)
        log_stage_event(logger, "stage_done", run_id=run_id)
    runtime.barrier()
    return cast(dict[str, float], topology_result["summary"])
