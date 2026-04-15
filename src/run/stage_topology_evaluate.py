"""Topology evaluation stage for PRING-style Human graph reconstruction."""

from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from src.embed import ensure_embeddings_ready
from src.evaluate import Evaluator
from src.run.stage_evaluate import _resolve_decision_threshold
from src.run.stage_train import _build_loss_config, _build_stage_runtime, _load_checkpoint
from src.utils.accelerator import AcceleratorLike, LocalAccelerator
from src.topology import (
    evaluate_predicted_graph,
    load_human_table2_baselines,
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
from src.utils.distributed import DistributedContext, distributed_barrier
from src.utils.logging import append_csv_row, log_stage_event

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


def _move_batch_to_device(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    """Move tensor values in a batch to the target device."""
    prepared: dict[str, object] = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        prepared[key] = (
            value.to(device, non_blocking=non_blocking)
            if isinstance(value, torch.Tensor)
            else value
        )
    return prepared


def _forward_model(model: torch.nn.Module, batch: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Execute one forward pass using the evaluator contract."""
    try:
        output = model(**batch)
    except TypeError:
        output = model(batch=batch)
    if not isinstance(output, dict):
        raise ValueError("Model forward output must be a dictionary")
    return cast(dict[str, torch.Tensor], output)


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


def _build_topology_loader(
    *,
    config: ConfigDict,
    split_path: Path,
    distributed_context: DistributedContext,
) -> tuple[DataLoader[dict[str, object]], list[tuple[str, str]], list[int], int]:
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
    local_indices = _rank_shard_indices(
        total_records=len(all_records),
        distributed_context=distributed_context,
    )
    preload_embeddings = as_bool(
        topology_cfg.get("preload_embeddings", True),
        "topology_evaluate.preload_embeddings",
    )
    if preload_embeddings:
        dataset.preload_embeddings(dataset.protein_ids(indices=local_indices))
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
        dataset=Subset(dataset, local_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=as_bool(
            dataloader_cfg.get("pin_memory", False),
            "data_config.dataloader.pin_memory",
        ),
        drop_last=False,
        collate_fn=_collate_batch,
    )
    return (
        cast(DataLoader[dict[str, object]], loader),
        all_records,
        local_indices,
        len(dataset._embedding_cache),
    )


def _rank_shard_indices(
    *,
    total_records: int,
    distributed_context: DistributedContext,
) -> list[int]:
    """Return deterministic record indices assigned to the current rank."""
    if not distributed_context.is_distributed:
        return list(range(total_records))
    return list(range(distributed_context.rank, total_records, distributed_context.world_size))


def _predict_topology_labels(
    *,
    model: torch.nn.Module,
    data_loader: DataLoader[dict[str, object]],
    device: torch.device,
    decision_threshold: float,
    use_amp: bool,
) -> list[int]:
    """Predict probabilities and thresholded labels for all topology pairs."""
    predictions: list[int] = []
    autocast_context = (
        torch.autocast(device_type=device.type, enabled=True)
        if use_amp and device.type in {"cpu", "cuda"}
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        for batch in data_loader:
            prepared_batch = _move_batch_to_device(batch=batch, device=device)
            output = _forward_model(model=model, batch=prepared_batch)
            logits = output["logits"]
            reduced_logits = (
                logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            )
            batch_probabilities = torch.sigmoid(reduced_logits).detach().cpu().tolist()
            predictions.extend(
                int(float(value) >= decision_threshold) for value in batch_probabilities
            )
    return predictions


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
) -> list[int]:
    """Gather local predictions from all ranks and restore original order on every rank."""
    if not distributed_context.is_distributed:
        return [int(prediction) for prediction in local_predictions]
    if not dist.is_initialized():
        raise RuntimeError("Distributed topology evaluation requires an initialized process group")

    gathered_payloads: list[dict[str, list[int]]] = [
        {"indices": [], "predictions": []} for _ in range(distributed_context.world_size)
    ]
    dist.all_gather_object(
        object_list=gathered_payloads,
        obj={
            "indices": [int(index) for index in local_indices],
            "predictions": [int(prediction) for prediction in local_predictions],
        },
    )
    return _ordered_predictions_from_shards(
        total_records=total_records,
        shard_payloads=gathered_payloads,
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


def run_topology_evaluation_stage(
    config: ConfigDict,
    model: torch.nn.Module,
    device: torch.device,
    dataloaders: dict[str, DataLoader[dict[str, object]]],
    run_id: str,
    checkpoint_path: Path,
    distributed_context: DistributedContext,
    accelerator: AcceleratorLike | None = None,
) -> dict[str, float]:
    """Run PRING-style Human topology evaluation and persist artifacts."""
    stage_accelerator = accelerator or LocalAccelerator(device)
    checkpoint_path_resolved = Path(checkpoint_path)
    model_name, _ = extract_model_kwargs(config)
    log_dir, _, logger = _build_stage_runtime(
        model_name=model_name,
        stage="topology_evaluate",
        run_id=run_id,
        distributed_context=distributed_context,
    )
    if distributed_context.is_main_process:
        log_stage_event(logger, "stage_start", run_id=run_id, checkpoint=checkpoint_path_resolved)
    _load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path_resolved,
        device=device,
        accelerator=stage_accelerator,
    )
    model.eval()

    topology_cfg = _topology_config(config)
    evaluate_cfg = get_section(config, "evaluate")
    training_cfg = get_section(config, "training_config")
    device_cfg = get_section(config, "device_config")
    threshold_cfg: ConfigDict = {
        "decision_threshold": topology_cfg.get(
            "decision_threshold",
            evaluate_cfg.get("decision_threshold", 0.5),
        )
    }
    use_amp = device.type == "cuda" and as_bool(
        device_cfg.get("use_mixed_precision", False),
        "device_config.use_mixed_precision",
    )
    threshold_mode = "broadcast"
    decision_threshold = 0.5
    if distributed_context.is_main_process:
        threshold_probe = Evaluator(
            metrics=["f1"],
            loss_config=_build_loss_config(training_cfg),
            use_amp=use_amp,
        )
        decision_threshold, threshold_mode = _resolve_decision_threshold(
            eval_cfg=threshold_cfg,
            evaluator=threshold_probe,
            model=model,
            dataloaders=dataloaders,
            device=device,
        )
        log_stage_event(logger, "decision_threshold", mode=threshold_mode, value=decision_threshold)

    if distributed_context.is_distributed:
        if not dist.is_initialized():
            raise RuntimeError(
                "Distributed topology evaluation requires an initialized process group"
            )
        threshold_tensor = torch.tensor([decision_threshold], dtype=torch.float32, device=device)
        dist.broadcast(threshold_tensor, src=0)
        decision_threshold = float(threshold_tensor.item())

    all_test_path, gt_graph_path, sampled_nodes_path = _topology_paths(config)
    topology_loader, records, local_indices, cached_embedding_count = _build_topology_loader(
        config=config,
        split_path=all_test_path,
        distributed_context=distributed_context,
    )
    if distributed_context.is_main_process:
        log_stage_event(
            logger,
            "topology_inference_ready",
            pair_count=len(records),
            local_pair_count=len(local_indices),
            cached_embedding_count=cached_embedding_count,
            distributed=distributed_context.is_distributed,
            world_size=distributed_context.world_size,
        )
    local_predictions = _predict_topology_labels(
        model=model,
        data_loader=topology_loader,
        device=device,
        decision_threshold=decision_threshold,
        use_amp=use_amp,
    )
    predictions = _gather_ordered_predictions(
        local_indices=local_indices,
        local_predictions=local_predictions,
        total_records=len(records),
        distributed_context=distributed_context,
    )

    prediction_path = log_dir / "all_test_ppi_pred.txt"
    if distributed_context.is_main_process and as_bool(
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
    topology_result = evaluate_predicted_graph(
        pred_graph=pred_graph,
        gt_graph=gt_graph,
        test_graph_nodes=test_graph_nodes,
    )

    if distributed_context.is_main_process:
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
    distributed_barrier(distributed_context)
    return cast(dict[str, float], topology_result["summary"])
