"""Validation-first analysis for TCCIG exp03 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from tccig.exp03_configs import PHASE_A_RUN_IDS, PHASE_B_CONFIG_RUN_IDS
from tccig.prepare import write_json

EXP02_REFERENCE_RUN_ID = "03_a0_exp02_topo_only_reference"
DEFAULT_EXP02_REFERENCE_DIR = Path("artifacts/exp02_rerun_fix/logs/tccig/02_balanced_subset")
DEFAULT_LOG_ROOT = Path("logs/tccig")
DEFAULT_OUTPUT_DIR = Path("analysis/tccig_exp03")
EXP02_REFERENCE_VAL_AUPRC = 0.6842201205393263
EXP02_REFERENCE_VAL_RELATIVE_DENSITY = 1.1537
VAL_AUPRC_FLOOR = 0.6705

EPOCH_COLUMNS = (1, 7, 40)
BASE_COLUMNS = (
    "run_id",
    "best_epoch",
    "selected_threshold",
    "validation_selected_edges",
    "validation_selected_edges_min",
    "validation_selected_edges_max",
    "validation_selected_edges_range",
    "val_topology_loss",
    "val_auprc",
    "internal_val_graph_sim",
    "internal_val_relative_density",
    "internal_val_deg_dist_mmd",
    "internal_val_cc_mmd",
    "sampled_edge_targets",
    "threshold_grid_rows",
    "threshold_grid_best_epoch",
    "val_auprc_floor",
    "gate_val_auprc_ok",
    "gate_selected_edges_stable",
    "gate_density_closer_than_exp02",
    "gate_threshold_grid_present",
    "exp02_reference_val_auprc",
    "exp02_reference_val_relative_density",
    "exp02_delta_val_auprc",
    "exp02_delta_val_topology_loss",
    "exp02_delta_relative_density_abs_error",
    "gate_non_worse_reference_metrics",
    "eligible_for_locked_test",
)
EPOCH_METRIC_NAMES = (
    "train_bce_loss",
    "train_topology_loss",
    "train_topo_graph_sim",
    "train_topo_relative_density",
    "train_topo_degree_mmd",
    "sampled_edge_targets",
    "train_fp_targets",
    "train_fn_targets",
)
HELDOUT_COLUMNS = (
    "heldout_refined_precision",
    "heldout_refined_recall",
    "heldout_refined_f1",
    "heldout_refined_auprc",
    "heldout_refined_auroc",
    "heldout_raw_auprc",
    "heldout_raw_auroc",
    "heldout_topology_relative_density",
    "heldout_topology_graph_sim",
    "heldout_topology_deg_dist_mmd",
    "heldout_topology_cc_mmd",
    "heldout_edges_added",
    "heldout_edges_deleted",
    "heldout_protocol_candidate_universe",
    "heldout_protocol_test_labels_visible_to_model",
    "heldout_raw_topology_relative_density",
    "heldout_raw_topology_graph_sim",
    "heldout_raw_topology_deg_dist_mmd",
    "heldout_raw_topology_cc_mmd",
    "heldout_raw_protocol_candidate_universe",
    "heldout_raw_protocol_test_labels_visible_to_model",
)
NON_HELDOUT_COLUMNS = (
    *BASE_COLUMNS,
    *(f"{metric}_epoch_{epoch}" for epoch in EPOCH_COLUMNS for metric in EPOCH_METRIC_NAMES),
)


def collect_run_row(
    *,
    run_id: str,
    run_dir: Path,
    include_heldout: bool,
    raw_baseline_artifact_dir: Path | None = None,
) -> dict[str, object]:
    """Collect one validation-first summary row from a TCCIG run directory."""
    training_summary = _read_mapping(run_dir / "training_summary.json")
    history = _read_history(training_summary)
    best_row = min(history, key=lambda row: _float(row, "monitor_value"))
    selected_edges = _int(best_row, "selected_rule_positive_edges")
    edge_values = [_int(row, "selected_rule_positive_edges") for row in history]
    threshold_grid = _read_optional_mapping(run_dir / "threshold_grid" / "best_epoch.json")
    threshold_rows = _threshold_grid_rows(threshold_grid)
    relative_density = _float(best_row, "internal_val_relative_density")

    row: dict[str, object] = {
        "run_id": run_id,
        "best_epoch": _int(best_row, "epoch"),
        "selected_threshold": _selected_threshold(best_row, training_summary, threshold_grid),
        "validation_selected_edges": selected_edges,
        "validation_selected_edges_min": min(edge_values),
        "validation_selected_edges_max": max(edge_values),
        "validation_selected_edges_range": max(edge_values) - min(edge_values),
        "val_topology_loss": _float(best_row, "val_topology_loss"),
        "val_auprc": _float(best_row, "val_auprc"),
        "internal_val_graph_sim": _float(best_row, "internal_val_graph_sim"),
        "internal_val_relative_density": relative_density,
        "internal_val_deg_dist_mmd": _float(best_row, "internal_val_deg_dist_mmd"),
        "internal_val_cc_mmd": _float(best_row, "internal_val_cc_mmd"),
        "sampled_edge_targets": _int(best_row, "sampled_edge_targets"),
        "threshold_grid_rows": threshold_rows,
        "threshold_grid_best_epoch": _optional_int(threshold_grid, "epoch"),
        "val_auprc_floor": VAL_AUPRC_FLOOR,
        "exp02_reference_val_auprc": EXP02_REFERENCE_VAL_AUPRC,
        "exp02_reference_val_relative_density": EXP02_REFERENCE_VAL_RELATIVE_DENSITY,
    }
    row.update(_epoch_metrics(history))
    _attach_validation_gates(row)

    if include_heldout:
        _attach_heldout_metrics(
            row,
            run_dir=run_dir,
            raw_baseline_artifact_dir=raw_baseline_artifact_dir,
        )
    return row


def write_exp03_report(
    *,
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> dict[str, Path]:
    """Write JSON, CSV, and Markdown exp03 validation-first summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_rows = _attach_reference_fields(rows)

    json_path = output_dir / "exp03_summary.json"
    csv_path = output_dir / "exp03_summary.csv"
    markdown_path = output_dir / "exp03_summary.md"
    write_json(json_path, {"rows": annotated_rows})
    _write_csv(csv_path, annotated_rows)
    markdown_path.write_text(_render_markdown(annotated_rows), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def main(argv: Sequence[str] | None = None) -> None:
    """Run the exp03 analysis CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--exp02-reference-dir", type=Path, default=DEFAULT_EXP02_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--locked-run-id", action="append", default=[])
    parser.add_argument("--raw-baseline-run-id")
    parser.add_argument("--include-phase-b", action="store_true")
    args = parser.parse_args(argv)

    locked_run_ids = set(cast(list[str], args.locked_run_id))
    run_ids = list(PHASE_A_RUN_IDS)
    if args.include_phase_b:
        run_ids.extend(PHASE_B_CONFIG_RUN_IDS)
    active_run_ids = set(run_ids)
    unknown_locked_run_ids = sorted(locked_run_ids - active_run_ids)
    if unknown_locked_run_ids:
        raise ValueError(f"Locked run IDs are not active: {', '.join(unknown_locked_run_ids)}")
    if locked_run_ids and not args.raw_baseline_run_id:
        raise ValueError("--raw-baseline-run-id is required when --locked-run-id is provided")
    rows: list[dict[str, object]] = []
    if args.exp02_reference_dir.exists():
        rows.append(
            collect_run_row(
                run_id=EXP02_REFERENCE_RUN_ID,
                run_dir=args.exp02_reference_dir,
                include_heldout=False,
            )
        )

    for run_id in run_ids:
        run_dir = args.log_root / run_id
        if not run_dir.exists():
            if run_id in PHASE_A_RUN_IDS:
                raise FileNotFoundError(f"Missing required Phase A run directory: {run_dir}")
            continue
        raw_baseline_dir = None
        if run_id in locked_run_ids and args.raw_baseline_run_id:
            raw_baseline_dir = (
                args.log_root / args.raw_baseline_run_id / "raw_pairwise_topology_baseline"
            )
        rows.append(
            collect_run_row(
                run_id=run_id,
                run_dir=run_dir,
                include_heldout=run_id in locked_run_ids,
                raw_baseline_artifact_dir=raw_baseline_dir,
            )
        )

    write_exp03_report(rows=rows, output_dir=args.output_dir)


def _read_mapping(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return cast(Mapping[str, object], payload)


def _read_optional_mapping(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    return _read_mapping(path)


def _read_history(training_summary: Mapping[str, object]) -> list[Mapping[str, object]]:
    history = training_summary.get("history")
    if not isinstance(history, Sequence) or isinstance(history, str):
        raise ValueError("training_summary.json must contain a history list")
    rows: list[Mapping[str, object]] = []
    for item in history:
        if not isinstance(item, Mapping):
            raise ValueError("training_summary.json history entries must be objects")
        rows.append(cast(Mapping[str, object], item))
    if not rows:
        raise ValueError("training_summary.json history must not be empty")
    return rows


def _threshold_grid_rows(threshold_grid: Mapping[str, object] | None) -> int:
    if threshold_grid is None:
        return 0
    rows = threshold_grid.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return 0
    return len(rows)


def _selected_threshold(
    best_row: Mapping[str, object],
    training_summary: Mapping[str, object],
    threshold_grid: Mapping[str, object] | None,
) -> float | None:
    for source in (
        best_row.get("selected_rule"),
        training_summary.get("selected_rule"),
        None if threshold_grid is None else threshold_grid.get("selected_rule"),
    ):
        if isinstance(source, Mapping) and "value" in source:
            return float(cast(float | int | str, source["value"]))
    return None


def _epoch_metrics(history: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_epoch = {_int(row, "epoch"): row for row in history}
    metrics: dict[str, object] = {}
    for epoch in EPOCH_COLUMNS:
        row = by_epoch.get(epoch)
        for metric in EPOCH_METRIC_NAMES:
            metrics[f"{metric}_epoch_{epoch}"] = None if row is None else row.get(metric)
    return metrics


def _attach_validation_gates(row: dict[str, object]) -> None:
    selected_edges = cast(int, row["validation_selected_edges"])
    edge_range = cast(int, row["validation_selected_edges_range"])
    val_auprc = cast(float, row["val_auprc"])
    relative_density = cast(float, row["internal_val_relative_density"])
    gate_val_auprc_ok = val_auprc >= VAL_AUPRC_FLOOR
    gate_selected_edges_stable = selected_edges > 0 and edge_range <= max(
        1000.0, 0.25 * selected_edges
    )
    gate_density_closer_than_exp02 = abs(relative_density - 1.0) < abs(
        EXP02_REFERENCE_VAL_RELATIVE_DENSITY - 1.0
    )
    gate_threshold_grid_present = cast(int, row["threshold_grid_rows"]) > 0
    row["gate_val_auprc_ok"] = gate_val_auprc_ok
    row["gate_selected_edges_stable"] = gate_selected_edges_stable
    row["gate_density_closer_than_exp02"] = gate_density_closer_than_exp02
    row["gate_threshold_grid_present"] = gate_threshold_grid_present
    row["eligible_for_locked_test"] = all(
        (
            gate_val_auprc_ok,
            gate_selected_edges_stable,
            gate_density_closer_than_exp02,
            gate_threshold_grid_present,
        )
    )


def _attach_heldout_metrics(
    row: dict[str, object],
    *,
    run_dir: Path,
    raw_baseline_artifact_dir: Path | None,
) -> None:
    refined = _read_mapping(run_dir / "pairwise_test" / "refined_metrics.json")
    raw = _read_mapping(run_dir / "pairwise_test" / "raw_metrics.json")
    topology = _read_mapping(run_dir / "topology_test" / "topology_metrics.json")
    topology_summary = _mapping(topology, "summary")
    deletion = _mapping(topology, "deletion_diagnostics")
    protocol = _mapping(topology, "protocol")

    for metric in ("precision", "recall", "f1", "auprc", "auroc"):
        row[f"heldout_refined_{metric}"] = refined.get(metric)
    for metric in ("auprc", "auroc"):
        row[f"heldout_raw_{metric}"] = raw.get(metric)
    _attach_topology_summary(row, prefix="heldout_topology", summary=topology_summary)
    row["heldout_edges_added"] = deletion.get("edges_added")
    row["heldout_edges_deleted"] = deletion.get("edges_deleted")
    row["heldout_protocol_candidate_universe"] = protocol.get("candidate_universe")
    row["heldout_protocol_test_labels_visible_to_model"] = protocol.get(
        "test_labels_visible_to_model"
    )

    if raw_baseline_artifact_dir is None:
        return
    raw_topology_path = raw_baseline_artifact_dir / "topology_metrics.json"
    if not raw_topology_path.exists():
        raise FileNotFoundError(f"Missing raw topology baseline metrics: {raw_topology_path}")
    raw_topology = _read_mapping(raw_topology_path)
    _attach_topology_summary(
        row,
        prefix="heldout_raw_topology",
        summary=_mapping(raw_topology, "summary"),
    )
    raw_protocol = _mapping(raw_topology, "protocol")
    row["heldout_raw_protocol_candidate_universe"] = raw_protocol.get("candidate_universe")
    row["heldout_raw_protocol_test_labels_visible_to_model"] = raw_protocol.get(
        "test_labels_visible_to_model"
    )


def _attach_topology_summary(
    row: dict[str, object],
    *,
    prefix: str,
    summary: Mapping[str, object],
) -> None:
    row[f"{prefix}_relative_density"] = summary.get("relative_density")
    row[f"{prefix}_graph_sim"] = summary.get("graph_sim")
    row[f"{prefix}_deg_dist_mmd"] = summary.get("deg_dist_mmd")
    row[f"{prefix}_cc_mmd"] = summary.get("cc_mmd")


def _attach_reference_fields(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    reference = next((row for row in rows if row.get("run_id") == EXP02_REFERENCE_RUN_ID), None)
    reference_val_auprc = _reference_float(reference, "val_auprc", EXP02_REFERENCE_VAL_AUPRC)
    reference_density = _reference_float(
        reference,
        "internal_val_relative_density",
        EXP02_REFERENCE_VAL_RELATIVE_DENSITY,
    )
    reference_topology_loss = _reference_optional_float(reference, "val_topology_loss")

    annotated: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        val_auprc = _maybe_float(row.get("val_auprc"))
        topology_loss = _maybe_float(row.get("val_topology_loss"))
        density = _maybe_float(row.get("internal_val_relative_density"))
        row["exp02_delta_val_auprc"] = (
            None if val_auprc is None else val_auprc - reference_val_auprc
        )
        row["exp02_delta_val_topology_loss"] = _delta(topology_loss, reference_topology_loss)
        row["exp02_delta_relative_density_abs_error"] = (
            None if density is None else abs(density - 1.0) - abs(reference_density - 1.0)
        )
        row["gate_non_worse_reference_metrics"] = _non_worse_reference(
            val_auprc=val_auprc,
            topology_loss=topology_loss,
            density=density,
            reference_val_auprc=reference_val_auprc,
            reference_topology_loss=reference_topology_loss,
            reference_density=reference_density,
        )
        annotated.append(row)
    return annotated


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    has_heldout = any(any(key.startswith("heldout_") for key in row) for row in rows)
    standard_columns = (
        (*NON_HELDOUT_COLUMNS, *HELDOUT_COLUMNS) if has_heldout else NON_HELDOUT_COLUMNS
    )
    extra_columns = sorted({key for row in rows for key in row if key not in standard_columns})
    columns = [*standard_columns, *extra_columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _render_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    has_heldout = any(any(key.startswith("heldout_") for key in row) for row in rows)
    lines = [
        "# Validation-first exp03 summary",
        "",
        f"- Runs summarized: {len(rows)}",
        f"- Validation AUPRC floor: {VAL_AUPRC_FLOOR}",
    ]
    if not has_heldout:
        lines.append("- Held-out metrics were not included.")
    else:
        lines.append("- Held-out metrics were included only for explicitly locked runs.")
    lines.extend(
        (
            "",
            "| run_id | best_epoch | val_auprc | relative_density | eligible_for_locked_test |",
            "| --- | ---: | ---: | ---: | --- |",
        )
    )
    for row in rows:
        lines.append(
            "| {run_id} | {best_epoch} | {val_auprc} | {density} | {eligible} |".format(
                run_id=row.get("run_id", ""),
                best_epoch=_markdown_value(row.get("best_epoch")),
                val_auprc=_markdown_value(row.get("val_auprc")),
                density=_markdown_value(row.get("internal_val_relative_density")),
                eligible=_markdown_value(row.get("eligible_for_locked_test")),
            )
        )
    if has_heldout:
        lines.extend(
            (
                "",
                "## Held-out protocol",
                "",
                "| run_id | heldout_protocol_candidate_universe | "
                "heldout_protocol_test_labels_visible_to_model | "
                "heldout_raw_protocol_candidate_universe | "
                "heldout_raw_protocol_test_labels_visible_to_model |",
                "| --- | --- | --- | --- | --- |",
            )
        )
        for row in rows:
            if not any(key.startswith("heldout_") for key in row):
                continue
            lines.append(
                "| {run_id} | {candidate_universe} | {labels_visible} | "
                "{raw_candidate_universe} | {raw_labels_visible} |".format(
                    run_id=row.get("run_id", ""),
                    candidate_universe=_markdown_value(
                        row.get("heldout_protocol_candidate_universe")
                    ),
                    labels_visible=_markdown_value(
                        row.get("heldout_protocol_test_labels_visible_to_model")
                    ),
                    raw_candidate_universe=_markdown_value(
                        row.get("heldout_raw_protocol_candidate_universe")
                    ),
                    raw_labels_visible=_markdown_value(
                        row.get("heldout_raw_protocol_test_labels_visible_to_model")
                    ),
                )
            )
    return "\n".join(lines) + "\n"


def _float(row: Mapping[str, object], key: str) -> float:
    value = _maybe_float(row.get(key))
    if value is None:
        raise ValueError(f"Missing numeric field: {key}")
    return value


def _int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int | float | str):
        raise ValueError(f"Missing integer field: {key}")
    return int(value)


def _optional_int(row: Mapping[str, object] | None, key: str) -> int | None:
    if row is None or key not in row:
        return None
    return _int(row, key)


def _mapping(row: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        return {}
    return cast(Mapping[str, object], value)


def _reference_float(
    row: Mapping[str, object] | None,
    key: str,
    fallback: float,
) -> float:
    if row is None:
        return fallback
    value = _maybe_float(row.get(key))
    return fallback if value is None else value


def _reference_optional_float(row: Mapping[str, object] | None, key: str) -> float | None:
    if row is None:
        return None
    return _maybe_float(row.get(key))


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if not isinstance(value, int | float | str):
        return None
    return float(value)


def _delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return value - reference


def _non_worse_reference(
    *,
    val_auprc: float | None,
    topology_loss: float | None,
    density: float | None,
    reference_val_auprc: float,
    reference_topology_loss: float | None,
    reference_density: float,
) -> bool:
    if val_auprc is None or density is None:
        return False
    checks = [
        val_auprc >= reference_val_auprc,
        abs(density - 1.0) <= abs(reference_density - 1.0),
    ]
    if topology_loss is not None and reference_topology_loss is not None:
        checks.append(topology_loss <= reference_topology_loss)
    return all(checks)


def _csv_value(value: object) -> object:
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else value


def _markdown_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
