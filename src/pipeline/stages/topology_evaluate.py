"""Topology evaluation stage for PRING-style Human graph reconstruction."""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as _dist
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
from src.utils.logging import append_csv_row, log_stage_event

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
) -> tuple[DataLoader[dict[str, object]], list[tuple[str, str]], int]:
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
    return (cast(DataLoader[dict[str, object]], loader), all_records, len(dataset._embedding_cache))


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
    topology_loader, records, cached_embedding_count = _build_topology_loader(
        config=config,
        split_path=all_test_path,
    )
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
            json.dump(
                {
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
                },
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
