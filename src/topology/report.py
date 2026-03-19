"""Human Table 2 comparison report generation for PRING topology runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

STRATEGY_ORDER = ("BFS", "DFS", "RANDOM_WALK")
STRATEGY_PREFIX = {"BFS": "bfs", "DFS": "dfs", "RANDOM_WALK": "rw"}
METRIC_ORDER = (
    ("graph_sim", "graph_sim"),
    ("relative_density", "relative_density"),
    ("deg_dist_mmd", "deg_dist"),
    ("cc_mmd", "cc"),
    ("laplacian_eigen_mmd", "spectral"),
)


def load_human_table2_baselines(path: Path) -> list[dict[str, Any]]:
    """Load baseline rows from a checked-in JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Baseline payload must contain a 'rows' list")
    return rows


def _competition_ranks(scores: dict[str, float], *, reverse: bool = False) -> dict[str, int]:
    """Assign competition ranks, preserving ties."""
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=reverse)
    ranks: dict[str, int] = {}
    current_rank = 0
    previous_score: float | None = None
    for index, (name, score) in enumerate(ordered, start=1):
        if previous_score is None or score != previous_score:
            current_rank = index
            previous_score = score
        ranks[name] = current_rank
    return ranks


def _row_from_metrics(
    *,
    category: str,
    model: str,
    strategy_metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Flatten one model row into report columns."""
    row: dict[str, Any] = {"category": category, "model": model}
    for strategy in STRATEGY_ORDER:
        prefix = STRATEGY_PREFIX[strategy]
        metrics = strategy_metrics[strategy]
        for metric_name, column_suffix in METRIC_ORDER:
            row[f"{prefix}_{column_suffix}"] = float(metrics[metric_name])
    return row


def _metric_sort_value(column_name: str, value: float) -> float:
    """Return comparable metric score for ranking."""
    if column_name.endswith("graph_sim"):
        return -value
    if column_name.endswith("relative_density"):
        return abs(value - 1.0)
    return value


def build_human_table2_rows(
    *,
    baselines: list[dict[str, Any]],
    model_name: str,
    model_category: str,
    strategy_metrics: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Merge baseline rows with one model row and recompute ranks."""
    rows = [
        _row_from_metrics(
            category=str(baseline["category"]),
            model=str(baseline["model"]),
            strategy_metrics=baseline["metrics"],
        )
        for baseline in baselines
    ]
    rows.append(
        _row_from_metrics(
            category=model_category,
            model=model_name,
            strategy_metrics=strategy_metrics,
        )
    )

    metric_columns = [
        f"{STRATEGY_PREFIX[strategy]}_{column_suffix}"
        for strategy in STRATEGY_ORDER
        for _, column_suffix in METRIC_ORDER
    ]
    per_metric_ranks: dict[str, dict[str, int]] = {}
    for column_name in metric_columns:
        scores = {
            str(row["model"]): _metric_sort_value(column_name, float(row[column_name]))
            for row in rows
        }
        per_metric_ranks[column_name] = _competition_ranks(scores=scores, reverse=False)

    avg_rank_scores: dict[str, float] = {}
    for row in rows:
        model = str(row["model"])
        avg_rank_scores[model] = sum(
            per_metric_ranks[column_name][model] for column_name in metric_columns
        ) / len(metric_columns)
    avg_ranks = _competition_ranks(scores=avg_rank_scores, reverse=False)

    for row in rows:
        model = str(row["model"])
        row["avg_rank_score"] = float(avg_rank_scores[model])
        row["avg_rank"] = int(avg_ranks[model])

    rows.sort(
        key=lambda row: (
            int(row["avg_rank"]),
            float(row["avg_rank_score"]),
            str(row["model"]),
        )
    )
    return rows


def write_human_table2_reports(
    *,
    output_dir: Path,
    baselines: list[dict[str, Any]],
    model_name: str,
    model_category: str,
    strategy_metrics: dict[str, dict[str, float]],
) -> tuple[Path, Path]:
    """Write CSV and Markdown comparison tables."""
    rows = build_human_table2_rows(
        baselines=baselines,
        model_name=model_name,
        model_category=model_category,
        strategy_metrics=strategy_metrics,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"human_table2_{model_name}.csv"
    markdown_path = output_dir / f"human_table2_{model_name}.md"

    fieldnames = [
        "category",
        "model",
        "bfs_graph_sim",
        "bfs_relative_density",
        "bfs_deg_dist",
        "bfs_cc",
        "bfs_spectral",
        "dfs_graph_sim",
        "dfs_relative_density",
        "dfs_deg_dist",
        "dfs_cc",
        "dfs_spectral",
        "rw_graph_sim",
        "rw_relative_density",
        "rw_deg_dist",
        "rw_cc",
        "rw_spectral",
        "avg_rank",
        "avg_rank_score",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    markdown_lines = [
        (
            "| Category | Model | BFS GS | BFS RD | BFS Deg. | BFS Clus. | BFS Spectral | "
            "DFS GS | DFS RD | DFS Deg. | DFS Clus. | DFS Spectral | RW GS | RW RD | "
            "RW Deg. | RW Clus. | RW Spectral | Avg. Rank |"
        ),
        (
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
            "--- | --- | --- | --- | --- | --- |"
        ),
    ]
    for row in rows:
        markdown_lines.append(
            "| {category} | {model} | {bfs_graph_sim:.4f} | {bfs_relative_density:.4f} | "
            "{bfs_deg_dist:.4f} | {bfs_cc:.4f} | {bfs_spectral:.4f} | "
            "{dfs_graph_sim:.4f} | {dfs_relative_density:.4f} | {dfs_deg_dist:.4f} | "
            "{dfs_cc:.4f} | {dfs_spectral:.4f} | {rw_graph_sim:.4f} | "
            "{rw_relative_density:.4f} | {rw_deg_dist:.4f} | {rw_cc:.4f} | "
            "{rw_spectral:.4f} | {avg_rank} |".format(**row)
        )
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    return csv_path, markdown_path
