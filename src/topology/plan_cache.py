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
from tccig.prepare import (
    embedding_index_sha256,
    optional_file_sha256,
    write_json,
)

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
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "graph_hash": _graph_hash(graph),
        "node_sizes": [int(size) for size in node_sizes],
        "samples_per_size": int(samples_per_size),
        "seed": int(seed),
        "strategy": _normalize_sampling_strategy(strategy),
        "coverage_augmentation": bool(coverage_augmentation),
    }


SUBSET_METADATA_VERSION = 1


def _scorer_identity(scorer_config: Mapping[str, object]) -> dict[str, object]:
    """Return the same scorer-identity block score_cache_metadata uses.

    Embedding this in the subset-plan cache key means a changed checkpoint, model
    config, embedding index, or max_sequence_length invalidates the persisted plan —
    so stale scored pairs are never silently reused.
    """
    return {
        "model_config_sha256": optional_file_sha256(scorer_config.get("model_config_path")),
        "checkpoint_sha256": optional_file_sha256(scorer_config.get("checkpoint_path")),
        "embedding_index_sha256": embedding_index_sha256(scorer_config),
        "max_sequence_length": scorer_config.get("max_sequence_length"),
    }


def subset_plan_payload_metadata(
    *,
    split: str,
    graph: nx.Graph,
    node_sizes: Sequence[int],
    samples_per_size: int,
    seed: int,
    strategy: str,
    coverage_augmentation: bool,
    candidate_ratio: int,
    pool_ratio: int,
    epoch_ratio: int,
    hard_fraction: float,
    uniform_fraction: float,
    hard_stratum_fraction: float,
    max_subgraphs_per_size: int,
    max_labeled_pairs_per_size: int,
    pair_scope: str,
    scorer_config: Mapping[str, object],
) -> dict[str, object]:
    """Build the strict cache key for a subset topology plan payload.

    The key covers the sampling parameters (so a sampler change invalidates the plan)
    AND the frozen scorer identity (so a checkpoint/config/embedding change invalidates
    the *scored* pairs baked into the plan). `pair_scope` distinguishes this from the
    full-plan key. `kind="subset"` lets the loader route to the subset validator.
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
    metadata.update(
        {
            "kind": "subset",
            "subset_version": SUBSET_METADATA_VERSION,
            "pair_scope": pair_scope,
            "candidate_ratio": int(candidate_ratio),
            "pool_ratio": int(pool_ratio),
            "epoch_ratio": int(epoch_ratio),
            "hard_fraction": float(hard_fraction),
            "uniform_fraction": float(uniform_fraction),
            "hard_stratum_fraction": float(hard_stratum_fraction),
            "max_subgraphs_per_size": int(max_subgraphs_per_size),
            "max_labeled_pairs_per_size": int(max_labeled_pairs_per_size),
            "scorer": _scorer_identity(scorer_config),
        }
    )
    return metadata


def subset_diagnostic_payload_metadata(
    *,
    split: str,
    graph: nx.Graph,
    node_sizes: Sequence[int],
    samples_per_size: int,
    seed: int,
    strategy: str,
    coverage_augmentation: bool,
    candidate_ratio: int,
    pool_ratio: int,
    epoch_ratio: int,
    hard_fraction: float,
    uniform_fraction: float,
    hard_stratum_fraction: float,
    max_subgraphs_per_size: int,
    max_labeled_pairs_per_size: int,
    bias_diagnostic_max_node_size: int,
    bias_diagnostic_max_subgraphs: int,
    scorer_config: Mapping[str, object],
) -> dict[str, object]:
    """Build the cache key for diagnostic-only full-space scorer payloads."""
    metadata = subset_plan_payload_metadata(
        split=split,
        graph=graph,
        node_sizes=node_sizes,
        samples_per_size=samples_per_size,
        seed=seed,
        strategy=strategy,
        coverage_augmentation=coverage_augmentation,
        candidate_ratio=candidate_ratio,
        pool_ratio=pool_ratio,
        epoch_ratio=epoch_ratio,
        hard_fraction=hard_fraction,
        uniform_fraction=uniform_fraction,
        hard_stratum_fraction=hard_stratum_fraction,
        max_subgraphs_per_size=max_subgraphs_per_size,
        max_labeled_pairs_per_size=max_labeled_pairs_per_size,
        pair_scope="subset_diagnostic",
        scorer_config=scorer_config,
    )
    metadata["bias_diagnostic_max_node_size"] = int(bias_diagnostic_max_node_size)
    metadata["bias_diagnostic_max_subgraphs"] = int(bias_diagnostic_max_subgraphs)
    return metadata


def _plan_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / "plans" / f"{split}.json"


def _manifest_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / "manifests" / f"{split}_plan.json"


def _validate_target_edges(edges: object, node_set: frozenset[str]) -> bool:
    """Return whether ``edges`` is a list of canonical edges inside ``node_set``."""
    if not isinstance(edges, list):
        return False
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            return False
        node_a, node_b = edge
        if not isinstance(node_a, str) or not isinstance(node_b, str):
            return False
        if node_a not in node_set or node_b not in node_set:
            return False
        if (node_a, node_b) != _canonical_edge(node_a, node_b):
            return False
    return True


def _validate_bucket(bucket: object) -> tuple[bool, int, int]:
    """Validate one bucket's shape; return ``(ok, subgraph_count, pair_count)``."""
    if not isinstance(bucket, Mapping):
        return False, 0, 0
    if not isinstance(bucket.get("node_size"), int):
        return False, 0, 0
    node_sets = bucket.get("sampled_subgraphs")
    target_edges = bucket.get("target_edges")
    if not isinstance(node_sets, list) or not isinstance(target_edges, list):
        return False, 0, 0
    if len(node_sets) != len(target_edges):
        return False, 0, 0
    pair_count = 0
    for nodes, edges in zip(node_sets, target_edges, strict=True):
        if not isinstance(nodes, list) or not all(isinstance(node, str) for node in nodes):
            return False, 0, 0
        if not _validate_target_edges(edges, frozenset(nodes)):
            return False, 0, 0
        size = len(nodes)
        pair_count += size * (size - 1) // 2
    return True, len(node_sets), pair_count


def _payload_is_rehydratable(payload: Mapping[str, object]) -> bool:
    """Return whether ``payload`` is structurally valid for rehydration.

    A cheap schema check (no O(n²) pair-record materialization). Guards against a
    payload that matches the cache key but is structurally incomplete or corrupt:
    wrong version, malformed buckets, mismatched list lengths, non-string node
    ids, edges that are not exactly two canonical endpoints inside the sampled
    node set, or recomputed totals that disagree with the stored totals.
    """
    if payload.get("version") != PAYLOAD_VERSION:
        return False
    raw_buckets = payload.get("buckets")
    if not isinstance(raw_buckets, list):
        return False
    total_subgraphs = 0
    total_pairs = 0
    for bucket in raw_buckets:
        ok, subgraph_count, pair_count = _validate_bucket(bucket)
        if not ok:
            return False
        total_subgraphs += subgraph_count
        total_pairs += pair_count
    if payload.get("total_subgraphs") != total_subgraphs:
        return False
    return payload.get("total_pairs") == total_pairs


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
    if not _payload_is_rehydratable(payload):
        LOGGER.warning("ignoring structurally invalid topology plan cache at %s", path)
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


def _subset_payload_is_rehydratable(payload: Mapping[str, object]) -> bool:
    """Cheap schema check for a serialized TopologySubsetPlan payload.

    Distinct from ``_payload_is_rehydratable`` (full-plan only): validates the subset
    payload kind/version and that every subgraph carries the four sample lists, so a
    full-plan payload or a stale subset schema is rejected instead of KeyError-ing in
    ``payload_to_subset_plan``. The accepted version is owned by the module that
    *writes* the payload (``tccig.topology_subset``), imported here so there is a single
    source of truth -- not a second constant that can drift out of sync.
    """
    from tccig.topology_subset import SUBSET_PAYLOAD_VERSION as _WRITER_VERSION

    if payload.get("payload_kind") != "topology_subset":
        return False
    if payload.get("subset_payload_version") != _WRITER_VERSION:
        return False
    raw_subgraphs = payload.get("subgraphs")
    if not isinstance(raw_subgraphs, list):
        return False
    required_lists = ("positives", "candidate_negatives", "hard_pool", "uniform_pool")
    for subgraph in raw_subgraphs:
        if not isinstance(subgraph, Mapping):
            return False
        if not isinstance(subgraph.get("node_size"), int):
            return False
        if not isinstance(subgraph.get("nodes"), list):
            return False
        for key in required_lists:
            if not isinstance(subgraph.get(key), list):
                return False
    for key in (
        "active_sizes",
        "skipped_sizes",
        "total_positive_pairs",
        "total_candidate_negatives",
        "total_pool_negatives",
    ):
        if key not in payload:
            return False
    return True


def load_subset_plan_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
) -> dict[str, object] | None:
    """Load a cached subset-plan payload, or ``None`` on miss/mismatch/corruption.

    Mirrors ``load_plan_cache`` but validates the subset payload shape via
    ``_subset_payload_is_rehydratable`` instead of the full-plan
    ``_payload_is_rehydratable``.
    """
    path = _plan_path(cache_dir, split)
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("ignoring corrupt topology subset plan cache at %s", path)
        return None
    if not isinstance(document, Mapping):
        LOGGER.warning("ignoring malformed topology subset plan cache at %s", path)
        return None
    if document.get("metadata") != dict(metadata):
        return None
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        return None
    if not _subset_payload_is_rehydratable(payload):
        LOGGER.warning("ignoring structurally invalid topology subset plan cache at %s", path)
        return None
    return dict(payload)


def write_subset_plan_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
    payload: Mapping[str, object],
) -> None:
    """Persist a subset-plan payload plus its cache-key metadata and a manifest.

    Identical on-disk layout to ``write_plan_cache``; kept separate so the subset path
    never silently writes under a full-plan validator's assumptions.
    """
    write_json(
        _plan_path(cache_dir, split),
        {"metadata": dict(metadata), "payload": dict(payload)},
    )
    write_json(_manifest_path(cache_dir, split), dict(metadata))


def load_subset_diagnostic_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
) -> dict[str, dict[str, float]] | None:
    """Load diagnostic full-space scorer probabilities, or ``None`` on miss."""
    path = _plan_path(cache_dir, split)
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("ignoring corrupt topology subset diagnostic cache at %s", path)
        return None
    if not isinstance(document, Mapping):
        LOGGER.warning("ignoring malformed topology subset diagnostic cache at %s", path)
        return None
    if document.get("metadata") != dict(metadata):
        return None
    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        return None
    result: dict[str, dict[str, float]] = {}
    for subgraph_id, raw_pairs in payload.items():
        if not isinstance(subgraph_id, str) or not isinstance(raw_pairs, Mapping):
            LOGGER.warning("ignoring malformed topology subset diagnostic cache at %s", path)
            return None
        result[subgraph_id] = {}
        for pair_id, probability in raw_pairs.items():
            if not isinstance(pair_id, str) or isinstance(probability, bool):
                LOGGER.warning("ignoring malformed topology subset diagnostic cache at %s", path)
                return None
            if not isinstance(probability, (int, float)):
                LOGGER.warning("ignoring malformed topology subset diagnostic cache at %s", path)
                return None
            result[subgraph_id][pair_id] = float(probability)
    return result


def write_subset_diagnostic_cache(
    *,
    cache_dir: Path,
    split: str,
    metadata: Mapping[str, object],
    payload: Mapping[str, Mapping[str, float]],
) -> None:
    """Persist diagnostic full-space scorer probabilities."""
    write_json(
        _plan_path(cache_dir, split),
        {
            "metadata": dict(metadata),
            "payload": {
                str(subgraph_id): {
                    str(pair_id): float(probability)
                    for pair_id, probability in pairs.items()
                }
                for subgraph_id, pairs in payload.items()
            },
        },
    )
    write_json(_manifest_path(cache_dir, split), dict(metadata))
