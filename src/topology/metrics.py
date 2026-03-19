"""PRING-style graph-level topology metrics."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from typing import Any

import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh

MetricDict = dict[str, dict[int, list[float] | float]]
KernelFn = Callable[[np.ndarray, np.ndarray], float]


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
    """Compute clustering-coefficient MMD."""
    sample_ref: list[np.ndarray] = []
    sample_pred: list[np.ndarray] = []
    pred_graphs_non_empty = [graph for graph in graph_pred_list if graph.number_of_nodes() != 0]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for histogram in executor.map(clustering_worker, graph_ref_list):
            sample_ref.append(histogram)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for histogram in executor.map(clustering_worker, pred_graphs_non_empty):
            sample_pred.append(histogram)
    return compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)


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
    return spectral_pmf


def spectral_stats(
    graph_ref_list: list[nx.Graph],
    graph_pred_list: list[nx.Graph],
    *,
    n_eigvals: int = -1,
) -> float:
    """Compute Laplacian-eigenvalue MMD."""
    sample_ref: list[np.ndarray] = []
    sample_pred: list[np.ndarray] = []
    kernel = partial(spectral_worker, n_eigvals=n_eigvals)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for histogram in executor.map(kernel, graph_ref_list):
            sample_ref.append(histogram)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for histogram in executor.map(kernel, graph_pred_list):
            sample_pred.append(histogram)
    return compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)


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

    graph_level_results: MetricDict = {
        "graph_sim": {},
        "relative_density": {},
        "deg_dist_mmd": {},
        "cc_mmd": {},
        "laplacian_eigen_mmd": {},
    }
    per_node_size: dict[int, dict[str, float | int]] = {}

    for node_size, node_lists in test_graph_nodes.items():
        gt_deg_dist: list[np.ndarray] = []
        pred_deg_dist: list[np.ndarray] = []
        gt_graphs: list[nx.Graph] = []
        pred_graphs: list[nx.Graph] = []
        graph_sim_values: list[float] = []
        density_values: list[float] = []

        for nodes in node_lists:
            gt_subgraph = gt_graph.subgraph(nodes)
            pred_subgraph = pred_graph.subgraph(nodes)

            graph_sim = compute_graph_similarity(pred_graph=pred_subgraph, gt_graph=gt_subgraph)
            relative_density = compute_relative_density(
                pred_graph=pred_subgraph,
                gt_graph=gt_subgraph,
            )
            graph_sim_values.append(graph_sim)
            density_values.append(relative_density)

            deg_pred, deg_gt = degree_distribution(pred_graph=pred_subgraph, gt_graph=gt_subgraph)
            pred_deg_dist.append(deg_pred)
            gt_deg_dist.append(deg_gt)
            gt_graphs.append(gt_subgraph)
            pred_graphs.append(pred_subgraph)

        deg_dist_mmd = compute_mmd(pred_deg_dist, gt_deg_dist)
        cc_mmd = clustering_stats(gt_graphs, pred_graphs)
        laplacian_eigen_mmd = spectral_stats(gt_graphs, pred_graphs)

        graph_level_results["graph_sim"][node_size] = graph_sim_values
        graph_level_results["relative_density"][node_size] = density_values
        graph_level_results["deg_dist_mmd"][node_size] = deg_dist_mmd
        graph_level_results["cc_mmd"][node_size] = cc_mmd
        graph_level_results["laplacian_eigen_mmd"][node_size] = laplacian_eigen_mmd
        per_node_size[node_size] = {
            "graph_count": len(node_lists),
            "graph_sim": float(np.mean(graph_sim_values)) if graph_sim_values else 0.0,
            "relative_density": float(np.mean(density_values)) if density_values else 0.0,
            "deg_dist_mmd": deg_dist_mmd,
            "cc_mmd": cc_mmd,
            "laplacian_eigen_mmd": laplacian_eigen_mmd,
        }

    summary = {
        metric_name: _summary_metric_value(metric_name, values_by_size)
        for metric_name, values_by_size in graph_level_results.items()
    }
    return {
        "details": graph_level_results,
        "summary": summary,
        "per_node_size": per_node_size,
    }
