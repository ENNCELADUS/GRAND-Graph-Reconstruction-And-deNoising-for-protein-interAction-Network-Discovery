"""PRING-style graph-level topology metrics."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh

MetricDict = dict[str, dict[int, list[float] | float]]
KernelFn = Callable[[np.ndarray, np.ndarray], float]
EPSILON = 1.0e-12


def reconstruct_graph(ppis: Iterable[tuple[str, str]]) -> nx.Graph:
    """Build an undirected graph from predicted positive PPI pairs."""
    graph = nx.Graph()
    graph.add_edges_from(ppis)
    return graph


def compute_graph_similarity(pred_graph: nx.Graph, gt_graph: nx.Graph) -> float:
    """Compute official PRING graph similarity."""
    if set(pred_graph.nodes) != set(gt_graph.nodes):
        raise ValueError("Graphs must have the same set of nodes")

    sorted_nodes = sorted(gt_graph.nodes)
    pred_matrix = nx.to_numpy_array(pred_graph, nodelist=sorted_nodes)
    gt_matrix = nx.to_numpy_array(gt_graph, nodelist=sorted_nodes)
    difference = np.abs(pred_matrix - gt_matrix).sum()
    denominator = np.sum(gt_matrix) + np.sum(pred_matrix)
    if denominator <= 0:
        return 1.0
    return float(1.0 - (difference / denominator))


def compute_relative_density(pred_graph: nx.Graph, gt_graph: nx.Graph) -> float:
    """Compute official PRING relative density."""
    pred_density = nx.density(pred_graph)
    gt_density = nx.density(gt_graph)
    if gt_density == 0:
        return float("inf") if pred_density != 0 else 1.0
    return float(pred_density / gt_density)


def gaussian_tv(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian kernel over total variation distance."""
    support_size = max(len(x), len(y))
    x_values = x.astype(np.float64, copy=False)
    y_values = y.astype(np.float64, copy=False)
    if len(x_values) < len(y_values):
        x_values = np.hstack((x_values, [0.0] * (support_size - len(x_values))))
    elif len(y_values) < len(x_values):
        y_values = np.hstack((y_values, [0.0] * (support_size - len(y_values))))
    distance = np.abs(x_values - y_values).sum() / 2.0
    return float(np.exp(-(distance * distance) / (2 * sigma * sigma)))


def _kernel_parallel_unpacked(
    x_value: np.ndarray,
    samples2: list[np.ndarray],
    kernel: KernelFn,
) -> float:
    distance = 0.0
    for sample in samples2:
        distance += float(kernel(x_value, sample))
    return distance


def disc(
    samples1: list[np.ndarray],
    samples2: list[np.ndarray],
    kernel: KernelFn,
    *,
    is_parallel: bool = True,
) -> float:
    """Discrepancy between two samples."""
    distance = 0.0
    if not is_parallel:
        for sample1 in samples1:
            for sample2 in samples2:
                distance += float(kernel(sample1, sample2))
    else:
        tasks = [(sample1, samples2, kernel) for sample1 in samples1]
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            for partial_distance in executor.map(
                lambda args: _kernel_parallel_unpacked(*args),
                tasks,
            ):
                distance += float(partial_distance)
    if len(samples1) * len(samples2) > 0:
        distance /= len(samples1) * len(samples2)
    else:
        distance = 1e6
    return float(distance)


def compute_mmd(
    samples1: list[np.ndarray],
    samples2: list[np.ndarray],
    *,
    kernel: KernelFn = gaussian_tv,
    is_hist: bool = True,
) -> float:
    """Compute MMD between two sample collections."""
    if is_hist:
        samples1 = [sample / (np.sum(sample) + 1e-6) for sample in samples1]
        samples2 = [sample / (np.sum(sample) + 1e-6) for sample in samples2]
    return float(
        disc(samples1, samples1, kernel)
        + disc(samples2, samples2, kernel)
        - 2 * disc(samples1, samples2, kernel)
    )


def _split_reference_samples(
    samples: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return deterministic disjoint reference samples for paper-style normalization."""
    return samples[::2], samples[1::2]


def compute_normalized_mmd_ratio(
    pred_samples: list[np.ndarray],
    gt_samples: list[np.ndarray],
    *,
    kernel: KernelFn = gaussian_tv,
    is_hist: bool = True,
) -> float:
    """Compute PRING paper-style normalized MMD ratio.

    PRING reports ``MMD(pred, test) / MMD(test, test)`` for degree,
    clustering-coefficient, and spectral metrics. With finite sampled graphs,
    the reference denominator is estimated from a deterministic even/odd split
    of the ground-truth sample bucket.
    """
    numerator = compute_mmd(pred_samples, gt_samples, kernel=kernel, is_hist=is_hist)
    if len(gt_samples) < 2:
        return numerator
    reference_left, reference_right = _split_reference_samples(gt_samples)
    if not reference_left or not reference_right:
        return numerator
    denominator = compute_mmd(
        reference_left,
        reference_right,
        kernel=kernel,
        is_hist=is_hist,
    )
    return float(numerator / max(denominator, EPSILON))


def degree_distribution(pred_graph: nx.Graph, gt_graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
    """Return degree histograms for a predicted and ground-truth subgraph."""
    pred_histogram = np.array(nx.degree_histogram(pred_graph))
    gt_histogram = np.array(nx.degree_histogram(gt_graph))
    max_len = max(len(pred_histogram), len(gt_histogram))
    pred_histogram = np.pad(pred_histogram, (0, max_len - len(pred_histogram)))
    gt_histogram = np.pad(gt_histogram, (0, max_len - len(gt_histogram)))
    return pred_histogram, gt_histogram


def clustering_worker(graph: nx.Graph) -> np.ndarray:
    """Build a histogram of node clustering coefficients."""
    clustering_coeffs = list(nx.clustering(graph).values())
    histogram, _ = np.histogram(
        clustering_coeffs,
        bins=100,
        range=(0.0, 1.0),
        density=False,
    )
    return histogram


def clustering_stats(graph_ref_list: list[nx.Graph], graph_pred_list: list[nx.Graph]) -> float:
    """Compute paper-normalized clustering-coefficient MMD."""
    sample_ref: list[np.ndarray] = []
    sample_pred: list[np.ndarray] = []
    pred_graphs_non_empty = [graph for graph in graph_pred_list if graph.number_of_nodes() != 0]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for histogram in executor.map(clustering_worker, graph_ref_list):
            sample_ref.append(histogram)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for histogram in executor.map(clustering_worker, pred_graphs_non_empty):
            sample_pred.append(histogram)
    return compute_normalized_mmd_ratio(sample_pred, sample_ref, kernel=gaussian_tv)


def spectral_worker(graph: nx.Graph, n_eigvals: int = -1) -> np.ndarray:
    """Compute normalized Laplacian eigenvalue histogram for one graph."""
    try:
        eigenvalues = eigvalsh(nx.normalized_laplacian_matrix(graph).todense())
    except Exception:
        eigenvalues = np.zeros(graph.number_of_nodes())
    if n_eigvals > 0:
        eigenvalues = eigenvalues[1 : n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigenvalues, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / max(1.0, spectral_pmf.sum())
    return np.asarray(spectral_pmf)


def spectral_stats(
    graph_ref_list: list[nx.Graph],
    graph_pred_list: list[nx.Graph],
    *,
    n_eigvals: int = -1,
) -> float:
    """Compute paper-normalized Laplacian-eigenvalue MMD."""
    sample_ref: list[np.ndarray] = []
    sample_pred: list[np.ndarray] = []
    kernel = partial(spectral_worker, n_eigvals=n_eigvals)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for histogram in executor.map(kernel, graph_ref_list):
            sample_ref.append(histogram)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for histogram in executor.map(kernel, graph_pred_list):
            sample_pred.append(histogram)
    return compute_normalized_mmd_ratio(sample_pred, sample_ref, kernel=gaussian_tv)


def _summary_metric_value(
    metric_name: str,
    values_by_size: Mapping[int, list[float] | float],
) -> float:
    """Compute the official summary value over node-size buckets."""
    if metric_name in {"graph_sim", "relative_density"}:
        flattened = [
            float(value)
            for per_size_values in values_by_size.values()
            if isinstance(per_size_values, list)
            for value in per_size_values
        ]
        return float(np.mean(flattened)) if flattened else 0.0
    numeric_values = [
        float(value) for value in values_by_size.values() if not isinstance(value, list)
    ]
    return float(np.mean(numeric_values)) if numeric_values else 0.0


def evaluate_graph_samples(
    *,
    pred_graphs_by_size: Mapping[int, list[nx.Graph]],
    gt_graphs_by_size: Mapping[int, list[nx.Graph]],
    include_spectral_stats: bool = True,
    include_clustering_stats: bool = True,
) -> dict[str, Any]:
    """Evaluate predicted and ground-truth graph samples grouped by node size."""
    if set(pred_graphs_by_size) != set(gt_graphs_by_size):
        raise ValueError("pred_graphs_by_size and gt_graphs_by_size must share node-size keys")

    graph_level_results: MetricDict = {
        "graph_sim": {},
        "relative_density": {},
        "deg_dist_mmd": {},
        "cc_mmd": {},
    }
    if include_spectral_stats:
        graph_level_results["laplacian_eigen_mmd"] = {}
    per_node_size: dict[int, dict[str, float | int]] = {}

    for node_size, gt_graphs in gt_graphs_by_size.items():
        pred_graphs = pred_graphs_by_size[node_size]
        if len(pred_graphs) != len(gt_graphs):
            raise ValueError("Predicted and ground-truth graph samples must have matching counts")

        graph_sim_values: list[float] = []
        density_values: list[float] = []
        pred_deg_dist: list[np.ndarray] = []
        gt_deg_dist: list[np.ndarray] = []

        for pred_graph, gt_graph in zip(pred_graphs, gt_graphs, strict=True):
            normalized_pred_graph = pred_graph.copy()
            if len(gt_graph.nodes()) > len(normalized_pred_graph.nodes()):
                missing_nodes = set(gt_graph.nodes()) - set(normalized_pred_graph.nodes())
                normalized_pred_graph.add_nodes_from(missing_nodes)
            graph_sim = compute_graph_similarity(
                pred_graph=normalized_pred_graph,
                gt_graph=gt_graph,
            )
            relative_density = compute_relative_density(
                pred_graph=normalized_pred_graph,
                gt_graph=gt_graph,
            )
            graph_sim_values.append(graph_sim)
            density_values.append(relative_density)

            deg_pred, deg_gt = degree_distribution(
                pred_graph=normalized_pred_graph,
                gt_graph=gt_graph,
            )
            pred_deg_dist.append(deg_pred)
            gt_deg_dist.append(deg_gt)
            pred_graph = normalized_pred_graph

        deg_dist_mmd = compute_normalized_mmd_ratio(pred_deg_dist, gt_deg_dist)
        cc_mmd = clustering_stats(gt_graphs, pred_graphs) if include_clustering_stats else 0.0

        graph_level_results["graph_sim"][node_size] = graph_sim_values
        graph_level_results["relative_density"][node_size] = density_values
        graph_level_results["deg_dist_mmd"][node_size] = deg_dist_mmd
        graph_level_results["cc_mmd"][node_size] = cc_mmd
        per_node_size[node_size] = {
            "graph_count": len(gt_graphs),
            "graph_sim": float(np.mean(graph_sim_values)) if graph_sim_values else 0.0,
            "relative_density": float(np.mean(density_values)) if density_values else 0.0,
            "deg_dist_mmd": deg_dist_mmd,
            "cc_mmd": cc_mmd,
        }
        if include_spectral_stats:
            laplacian_eigen_mmd = spectral_stats(gt_graphs, pred_graphs)
            graph_level_results["laplacian_eigen_mmd"][node_size] = laplacian_eigen_mmd
            per_node_size[node_size]["laplacian_eigen_mmd"] = laplacian_eigen_mmd

    summary = {
        metric_name: _summary_metric_value(metric_name, values_by_size)
        for metric_name, values_by_size in graph_level_results.items()
    }
    return {
        "details": graph_level_results,
        "summary": summary,
        "per_node_size": per_node_size,
    }


def merge_graph_sample_evaluations(
    *,
    shard_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge node-size-disjoint graph-evaluation shard results."""
    merged_details: MetricDict = {}
    merged_per_node_size: dict[int, dict[str, float | int]] = {}

    for shard_result in shard_results:
        details = shard_result.get("details", {})
        per_node_size = shard_result.get("per_node_size", {})
        if not isinstance(details, Mapping):
            raise ValueError("shard_result.details must be a mapping")
        if not isinstance(per_node_size, Mapping):
            raise ValueError("shard_result.per_node_size must be a mapping")

        for metric_name, values_by_size in details.items():
            if not isinstance(values_by_size, Mapping):
                raise ValueError("details values must be mappings keyed by node size")
            metric_bucket = merged_details.setdefault(str(metric_name), {})
            for node_size, values in values_by_size.items():
                node_size_int = int(node_size)
                if node_size_int in metric_bucket:
                    raise ValueError(
                        f"Duplicate node-size shard for metric {metric_name}: {node_size_int}"
                    )
                if isinstance(values, list):
                    metric_bucket[node_size_int] = [float(value) for value in values]
                else:
                    metric_bucket[node_size_int] = float(values)

        for node_size, values in per_node_size.items():
            node_size_int = int(node_size)
            if node_size_int in merged_per_node_size:
                raise ValueError(f"Duplicate per-node-size shard: {node_size_int}")
            if not isinstance(values, Mapping):
                raise ValueError("per_node_size values must be mappings")
            merged_per_node_size[node_size_int] = {
                key: int(value) if key == "graph_count" else float(value)
                for key, value in values.items()
            }

    summary = {
        metric_name: _summary_metric_value(metric_name, values_by_size)
        for metric_name, values_by_size in merged_details.items()
    }
    return {
        "details": merged_details,
        "summary": summary,
        "per_node_size": merged_per_node_size,
    }


def evaluate_predicted_graph(
    *,
    pred_graph: nx.Graph,
    gt_graph: nx.Graph,
    test_graph_nodes: Mapping[int, list[list[str]]],
) -> dict[str, Any]:
    """Evaluate a reconstructed graph against PRING topology metrics."""
    if len(gt_graph.nodes()) > len(pred_graph.nodes()):
        missing_nodes = set(gt_graph.nodes()) - set(pred_graph.nodes())
        pred_graph.add_nodes_from(missing_nodes)

    pred_graphs_by_size: dict[int, list[nx.Graph]] = {}
    gt_graphs_by_size: dict[int, list[nx.Graph]] = {}
    for node_size, node_lists in test_graph_nodes.items():
        gt_graphs: list[nx.Graph] = []
        pred_graphs: list[nx.Graph] = []
        for nodes in node_lists:
            gt_subgraph = gt_graph.subgraph(nodes)
            pred_subgraph = pred_graph.subgraph(nodes)
            gt_graphs.append(gt_subgraph)
            pred_graphs.append(pred_subgraph)
        gt_graphs_by_size[node_size] = gt_graphs
        pred_graphs_by_size[node_size] = pred_graphs

    return evaluate_graph_samples(
        pred_graphs_by_size=pred_graphs_by_size,
        gt_graphs_by_size=gt_graphs_by_size,
    )
