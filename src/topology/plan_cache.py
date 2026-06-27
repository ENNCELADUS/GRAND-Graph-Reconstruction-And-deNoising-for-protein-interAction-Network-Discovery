"""Disk cache for the TCCIG topology internal-validation plan.

Stores a JSON-safe payload (node sets + canonical induced edges) instead of the
live ``InternalValidationPlan`` (which holds ``nx.Graph`` objects). ``pair_records``
are regenerated on load by deterministic upper-triangle enumeration.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import networkx as nx
from tccig.prepare import write_json

from src.topology.finetune_data import (
    InternalValidationNodeBucketPlan,
    InternalValidationPairRecord,
    InternalValidationPlan,
    _canonical_edge,
    _normalize_sampling_strategy,
)

LOGGER = logging.getLogger(__name__)

PAYLOAD_VERSION = 1


def _bucket_target_edges(subgraph: nx.Graph) -> list[list[str]]:
    """Return sorted canonical induced edges of one target subgraph."""
    edges = {_canonical_edge(node_a, node_b) for node_a, node_b in subgraph.edges()}
    return [list(edge) for edge in sorted(edges)]


def plan_to_payload(plan: InternalValidationPlan) -> dict[str, object]:
    """Serialize an ``InternalValidationPlan`` to a JSON-safe payload."""
    buckets: list[dict[str, object]] = []
    for bucket in plan.buckets:
        target_edges = [_bucket_target_edges(sub) for sub in bucket.target_subgraphs]
        buckets.append(
            {
                "node_size": bucket.node_size,
                "sampled_subgraphs": [list(nodes) for nodes in bucket.sampled_subgraphs],
                "target_edges": target_edges,
            }
        )
    return {
        "version": PAYLOAD_VERSION,
        "buckets": buckets,
        "total_subgraphs": plan.total_subgraphs,
        "total_pairs": plan.total_pairs,
    }


def _rebuild_target_subgraph(nodes: Sequence[str], edges: Sequence[Sequence[str]]) -> nx.Graph:
    """Reconstruct a node-induced target subgraph from cached data."""
    subgraph = nx.Graph()
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from((edge[0], edge[1]) for edge in edges)
    return subgraph


def _pair_records_for(
    subgraph_index: int, nodes: Sequence[str]
) -> list[InternalValidationPairRecord]:
    """Regenerate the upper-triangle pair records for one sampled subgraph."""
    records: list[InternalValidationPairRecord] = []
    for index_a, protein_a in enumerate(nodes):
        for index_b in range(index_a + 1, len(nodes)):
            records.append(
                InternalValidationPairRecord(
                    subgraph_index=subgraph_index,
                    pair_index_a=index_a,
                    pair_index_b=index_b,
                    protein_a=protein_a,
                    protein_b=nodes[index_b],
                )
            )
    return records


def payload_to_plan(payload: Mapping[str, object], *, graph: nx.Graph) -> InternalValidationPlan:
    """Rehydrate an ``InternalValidationPlan`` from a cached payload.

    ``graph`` is accepted for signature symmetry with the builder; target
    subgraphs are rebuilt from cached induced edges and do not read ``graph``.
    """
    raw_buckets = payload.get("buckets")
    if not isinstance(raw_buckets, list):
        raise ValueError("plan payload missing 'buckets' list")
    buckets: list[InternalValidationNodeBucketPlan] = []
    protein_ids: set[str] = set()
    total_subgraphs = 0
    total_pairs = 0
    for raw_bucket in raw_buckets:
        node_sets = tuple(tuple(nodes) for nodes in raw_bucket["sampled_subgraphs"])
        target_edges = raw_bucket["target_edges"]
        target_subgraphs = tuple(
            _rebuild_target_subgraph(nodes, edges)
            for nodes, edges in zip(node_sets, target_edges, strict=True)
        )
        pair_records: list[InternalValidationPairRecord] = []
        for subgraph_index, nodes in enumerate(node_sets):
            protein_ids.update(nodes)
            pair_records.extend(_pair_records_for(subgraph_index, nodes))
        buckets.append(
            InternalValidationNodeBucketPlan(
                node_size=int(raw_bucket["node_size"]),
                sampled_subgraphs=node_sets,
                target_subgraphs=target_subgraphs,
                pair_records=tuple(pair_records),
            )
        )
        total_subgraphs += len(node_sets)
        total_pairs += len(pair_records)
    return InternalValidationPlan(
        buckets=tuple(buckets),
        protein_ids=frozenset(protein_ids),
        total_subgraphs=total_subgraphs,
        total_pairs=total_pairs,
    )


def _graph_hash(graph: nx.Graph) -> str:
    """Return a stable digest over canonical edges and node ids."""
    digest = hashlib.sha256()
    for node in sorted(graph.nodes()):
        digest.update(b"n")
        digest.update(str(node).encode("utf-8"))
        digest.update(b"\n")
    edges = sorted(_canonical_edge(node_a, node_b) for node_a, node_b in graph.edges())
    for node_a, node_b in edges:
        digest.update(node_a.encode("utf-8"))
        digest.update(b"\0")
        digest.update(node_b.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def plan_payload_metadata(
    *,
    split: str,
    graph: nx.Graph,
    node_sizes: Sequence[int],
    samples_per_size: int,
    seed: int,
    strategy: str,
    coverage_augmentation: bool,
) -> dict[str, object]:
    """Build the strict cache key for a topology plan payload.

    ``strategy`` is normalized so that, for example, ``mixed`` and ``MIXED``
    map to the same cache entry. Any change to the graph or a sampling
    parameter changes the returned metadata, invalidating a stale cache.
    """
    return {
        "version": PAYLOAD_VERSION,
        "split": split,
        "graph_hash": _graph_hash(graph),
        "node_sizes": [int(size) for size in node_sizes],
        "samples_per_size": int(samples_per_size),
        "seed": int(seed),
        "strategy": _normalize_sampling_strategy(strategy),
        "coverage_augmentation": bool(coverage_augmentation),
    }


def _plan_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / "plans" / f"{split}.json"


def _manifest_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / "manifests" / f"{split}_plan.json"


def load_plan_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
) -> dict[str, object] | None:
    """Load a cached plan payload, or ``None`` on miss/mismatch/corruption."""
    path = _plan_path(cache_dir, split)
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("ignoring corrupt topology plan cache at %s", path)
        return None
    if not isinstance(document, Mapping):
        LOGGER.warning("ignoring malformed topology plan cache at %s", path)
        return None
    if document.get("metadata") != dict(metadata):
        return None
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def write_plan_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
    payload: Mapping[str, object],
) -> None:
    """Persist a plan payload plus its cache-key metadata and a manifest."""
    write_json(
        _plan_path(cache_dir, split),
        {"metadata": dict(metadata), "payload": dict(payload)},
    )
    write_json(_manifest_path(cache_dir, split), dict(metadata))
