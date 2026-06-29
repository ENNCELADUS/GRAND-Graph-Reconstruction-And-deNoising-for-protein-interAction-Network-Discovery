"""Bounded subset sampling contracts for TCCIG topology training."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isclose
from typing import TypeVar, cast

import networkx as nx

T = TypeVar("T")


class SamplingStratum(StrEnum):
    """Topology subset sampling strata."""

    POSITIVE = "positive"
    HARD_NEGATIVE = "hard_negative"
    UNIFORM_NEGATIVE = "uniform_negative"


@dataclass(frozen=True)
class TopologySubsetSamplerConfig:
    """Configuration for bounded candidate -> pool -> epoch subset sampling."""

    enabled: bool = True
    candidate_ratio: int = 20
    pool_ratio: int = 10
    epoch_ratio: int = 5
    hard_fraction: float = 0.5
    uniform_fraction: float = 0.5
    hard_stratum_fraction: float = 0.2
    seed: int = 0
    # Per-size budget (spec §4). 0 == unbounded. Caps stop the 200-node bucket from
    # dominating memory/objective and bound the labeled-pair count scored per size.
    max_subgraphs_per_size: int = 0
    max_labeled_pairs_per_size: int = 0
    # Bias diagnostic (spec §3.7 / §9). 0 == off; spec §3.7 proposes 5. The max node
    # size caps the subgraph the diagnostic decodes so the full n*(n-1) space is cheap.
    # bias_diagnostic_max_subgraphs bounds how many eligible subgraphs the production
    # diagnostic samples across the size mixture (spec §3.7 asks for "a few capped
    # validation subgraphs"), so bias from larger sizes is also exposed, not just the
    # single smallest one. 0 == every eligible subgraph.
    bias_diagnostic_every_n_epochs: int = 0
    bias_diagnostic_max_node_size: int = 40
    bias_diagnostic_max_subgraphs: int = 4

    def validate(self) -> None:
        """Validate ratio and fraction constraints."""
        for name, value in (
            ("candidate_ratio", self.candidate_ratio),
            ("pool_ratio", self.pool_ratio),
            ("epoch_ratio", self.epoch_ratio),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("max_subgraphs_per_size", self.max_subgraphs_per_size),
            ("max_labeled_pairs_per_size", self.max_labeled_pairs_per_size),
            ("bias_diagnostic_every_n_epochs", self.bias_diagnostic_every_n_epochs),
            ("bias_diagnostic_max_node_size", self.bias_diagnostic_max_node_size),
            ("bias_diagnostic_max_subgraphs", self.bias_diagnostic_max_subgraphs),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0 (0 == unbounded)")
        for name, value in (
            ("hard_fraction", self.hard_fraction),
            ("uniform_fraction", self.uniform_fraction),
            ("hard_stratum_fraction", self.hard_stratum_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not isclose(self.hard_fraction + self.uniform_fraction, 1.0, abs_tol=1.0e-8):
            raise ValueError("hard_fraction + uniform_fraction must equal 1.0")
        if self.pool_ratio > self.candidate_ratio:
            raise ValueError("pool_ratio must be <= candidate_ratio")
        if self.epoch_ratio > self.pool_ratio:
            raise ValueError("epoch_ratio must be <= pool_ratio")


@dataclass(frozen=True)
class TopologyPairSample:
    """One pair selected for a topology subset subgraph in one epoch."""

    pair_id: str
    subgraph_id: str
    node_size: int
    protein_a: str
    protein_b: str
    local_index_a: int
    local_index_b: int
    stratum: SamplingStratum
    pi_cand: float
    pi_pool_given_cand: float
    pi_epoch_given_pool: float
    pi_total: float
    target: float
    scorer_probability: float

    def validate(self) -> None:
        """Validate sampling probabilities and target semantics."""
        for name, value in (
            ("pi_cand", self.pi_cand),
            ("pi_pool_given_cand", self.pi_pool_given_cand),
            ("pi_epoch_given_pool", self.pi_epoch_given_pool),
            ("pi_total", self.pi_total),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        expected = self.pi_cand * self.pi_pool_given_cand * self.pi_epoch_given_pool
        if not isclose(self.pi_total, expected, rel_tol=1.0e-7, abs_tol=1.0e-9):
            raise ValueError("pi_total must equal pi_cand * pi_pool_given_cand * pi_epoch_given_pool")
        if self.stratum is SamplingStratum.POSITIVE:
            if self.target != 1.0:
                raise ValueError("positive samples must have target=1.0")
            if (self.pi_cand, self.pi_pool_given_cand, self.pi_epoch_given_pool, self.pi_total) != (
                1.0,
                1.0,
                1.0,
                1.0,
            ):
                raise ValueError("positive samples must have all inclusion probabilities equal to 1")


def canonical_pair_id(protein_a: str, protein_b: str) -> str:
    """Return a stable undirected pair id."""
    left, right = sorted((protein_a, protein_b))
    return f"{left}||{right}"


def active_node_sizes(
    *,
    node_sizes: tuple[int, ...],
    graph_node_count: int,
    subgraphs_per_size: dict[int, int],
    labeled_pairs_per_size: dict[int, int],
) -> tuple[tuple[int, ...], dict[int, str]]:
    """Return globally active node sizes and skipped-size reasons."""
    active: list[int] = []
    skipped: dict[int, str] = {}
    for size in node_sizes:
        if size > graph_node_count:
            skipped[size] = "larger_than_graph"
            continue
        if subgraphs_per_size.get(size, 0) <= 0:
            skipped[size] = "zero_subgraph_budget"
            continue
        if labeled_pairs_per_size.get(size, 0) <= 0:
            skipped[size] = "zero_labeled_pair_budget"
            continue
        active.append(size)
    if not active:
        raise ValueError("topology subset sampling has no active node sizes")
    return tuple(active), skipped


@dataclass(frozen=True)
class TopologySubgraphPlan:
    """Bounded candidate and pool plan for one sampled subgraph."""

    subgraph_id: str
    node_size: int
    nodes: tuple[str, ...]
    positives: tuple[TopologyPairSample, ...]
    candidate_negatives: tuple[TopologyPairSample, ...]
    hard_pool: tuple[TopologyPairSample, ...]
    uniform_pool: tuple[TopologyPairSample, ...]


@dataclass(frozen=True)
class TopologySubsetPlan:
    """All subgraph subset-sampling metadata for train topology."""

    subgraphs: tuple[TopologySubgraphPlan, ...]
    active_sizes: tuple[int, ...]
    skipped_sizes: Mapping[int, str]
    total_positive_pairs: int
    total_candidate_negatives: int
    total_pool_negatives: int


def _all_local_pairs(nodes: tuple[str, ...]) -> list[tuple[int, int, str, str, str]]:
    pairs: list[tuple[int, int, str, str, str]] = []
    for index_a, protein_a in enumerate(nodes):
        for index_b in range(index_a + 1, len(nodes)):
            protein_b = nodes[index_b]
            pairs.append(
                (index_a, index_b, protein_a, protein_b, canonical_pair_id(protein_a, protein_b))
            )
    return pairs


def _draw_without_replacement(items: Sequence[T], *, count: int, rng: random.Random) -> list[T]:
    if count >= len(items):
        return list(items)
    return rng.sample(list(items), count)


def _replace_sample_probability(
    sample: TopologyPairSample,
    *,
    stratum: SamplingStratum,
    pi_pool_given_cand: float,
    pi_epoch_given_pool: float,
) -> TopologyPairSample:
    pi_total = sample.pi_cand * pi_pool_given_cand * pi_epoch_given_pool
    return TopologyPairSample(
        pair_id=sample.pair_id,
        subgraph_id=sample.subgraph_id,
        node_size=sample.node_size,
        protein_a=sample.protein_a,
        protein_b=sample.protein_b,
        local_index_a=sample.local_index_a,
        local_index_b=sample.local_index_b,
        stratum=stratum,
        pi_cand=sample.pi_cand,
        pi_pool_given_cand=pi_pool_given_cand,
        pi_epoch_given_pool=pi_epoch_given_pool,
        pi_total=pi_total,
        target=sample.target,
        scorer_probability=sample.scorer_probability,
    )


def build_topology_subset_plan(
    *,
    graph: nx.Graph,
    sampled_subgraphs: Mapping[int, Sequence[tuple[str, ...]]],
    config: TopologySubsetSamplerConfig,
    scorer_probabilities: Mapping[str, float],
) -> TopologySubsetPlan:
    """Build a bounded candidate/pool topology subset plan."""
    config.validate()
    rng = random.Random(config.seed)
    sampled_subgraphs = {
        int(size): [tuple(nodes) for nodes in subsets]
        for size, subsets in sampled_subgraphs.items()
    }
    remaining_labeled_budget: dict[int, int] = {
        int(size): config.max_labeled_pairs_per_size for size in sampled_subgraphs
    }
    subgraphs: list[TopologySubgraphPlan] = []
    subgraph_budget_by_size = {
        int(size): len(subsets) for size, subsets in sampled_subgraphs.items()
    }
    labeled_budget_by_size: dict[int, int] = defaultdict(int)
    for size, subsets in sampled_subgraphs.items():
        for nodes in subsets:
            labeled_budget_by_size[int(size)] += len(_all_local_pairs(tuple(nodes)))
    active_sizes, skipped = active_node_sizes(
        node_sizes=tuple(sorted(int(size) for size in sampled_subgraphs)),
        graph_node_count=graph.number_of_nodes(),
        subgraphs_per_size=subgraph_budget_by_size,
        labeled_pairs_per_size=dict(labeled_budget_by_size),
    )
    for node_size in active_sizes:
        for subgraph_index, raw_nodes in enumerate(sampled_subgraphs[node_size]):
            nodes = tuple(sorted(raw_nodes))
            subgraph_id = f"size={node_size}:index={subgraph_index}"
            positives: list[TopologyPairSample] = []
            negatives: list[tuple[int, int, str, str, str]] = []
            for index_a, index_b, protein_a, protein_b, pair_id in _all_local_pairs(nodes):
                if graph.has_edge(protein_a, protein_b):
                    positives.append(
                        TopologyPairSample(
                            pair_id=pair_id,
                            subgraph_id=subgraph_id,
                            node_size=node_size,
                            protein_a=protein_a,
                            protein_b=protein_b,
                            local_index_a=index_a,
                            local_index_b=index_b,
                            stratum=SamplingStratum.POSITIVE,
                            pi_cand=1.0,
                            pi_pool_given_cand=1.0,
                            pi_epoch_given_pool=1.0,
                            pi_total=1.0,
                            target=1.0,
                            scorer_probability=float(scorer_probabilities.get(pair_id, 1.0)),
                        )
                    )
                else:
                    negatives.append((index_a, index_b, protein_a, protein_b, pair_id))
            candidate_count = min(
                len(negatives), max(1, config.candidate_ratio * max(1, len(positives)))
            )
            if config.max_labeled_pairs_per_size > 0:
                remaining_labeled_budget[node_size] -= len(positives)
                size_cap = max(0, remaining_labeled_budget[node_size])
                candidate_count = min(candidate_count, size_cap)
                remaining_labeled_budget[node_size] = size_cap - candidate_count
            candidate_rows = _draw_without_replacement(negatives, count=candidate_count, rng=rng)
            pi_cand = (
                1.0
                if (not negatives or candidate_count == 0)
                else candidate_count / float(len(negatives))
            )
            candidate_samples = [
                TopologyPairSample(
                    pair_id=pair_id,
                    subgraph_id=subgraph_id,
                    node_size=node_size,
                    protein_a=protein_a,
                    protein_b=protein_b,
                    local_index_a=index_a,
                    local_index_b=index_b,
                    stratum=SamplingStratum.UNIFORM_NEGATIVE,
                    pi_cand=pi_cand,
                    pi_pool_given_cand=1.0,
                    pi_epoch_given_pool=1.0,
                    pi_total=pi_cand,
                    target=0.0,
                    scorer_probability=float(scorer_probabilities.get(pair_id, 0.0)),
                )
                for index_a, index_b, protein_a, protein_b, pair_id in candidate_rows
            ]
            sorted_candidates = tuple(
                sorted(candidate_samples, key=lambda sample: sample.scorer_probability, reverse=True)
            )
            hard_count = min(
                len(sorted_candidates),
                max(1, int(round(len(sorted_candidates) * config.hard_stratum_fraction))),
            )
            hard_frame = sorted_candidates[:hard_count]
            uniform_frame = sorted_candidates[hard_count:]
            pool_per_pos = config.pool_ratio * max(1, len(positives))
            hard_pool_count = min(
                len(hard_frame), max(0, int(round(pool_per_pos * config.hard_fraction)))
            )
            uniform_pool_count = min(len(uniform_frame), max(0, pool_per_pos - hard_pool_count))
            hard_pool_base = _draw_without_replacement(hard_frame, count=hard_pool_count, rng=rng)
            uniform_pool_base = _draw_without_replacement(
                uniform_frame, count=uniform_pool_count, rng=rng
            )
            hard_pi_pool = 1.0 if not hard_frame else hard_pool_count / float(len(hard_frame))
            uniform_pi_pool = (
                1.0 if not uniform_frame else uniform_pool_count / float(len(uniform_frame))
            )
            hard_pool = tuple(
                _replace_sample_probability(
                    sample,
                    stratum=SamplingStratum.HARD_NEGATIVE,
                    pi_pool_given_cand=hard_pi_pool,
                    pi_epoch_given_pool=1.0,
                )
                for sample in hard_pool_base
            )
            uniform_pool = tuple(
                _replace_sample_probability(
                    sample,
                    stratum=SamplingStratum.UNIFORM_NEGATIVE,
                    pi_pool_given_cand=uniform_pi_pool,
                    pi_epoch_given_pool=1.0,
                )
                for sample in uniform_pool_base
            )
            subgraphs.append(
                TopologySubgraphPlan(
                    subgraph_id=subgraph_id,
                    node_size=node_size,
                    nodes=nodes,
                    positives=tuple(positives),
                    candidate_negatives=tuple(candidate_samples),
                    hard_pool=hard_pool,
                    uniform_pool=uniform_pool,
                )
            )
    return TopologySubsetPlan(
        subgraphs=tuple(subgraphs),
        active_sizes=active_sizes,
        skipped_sizes=skipped,
        total_positive_pairs=sum(len(subgraph.positives) for subgraph in subgraphs),
        total_candidate_negatives=sum(len(subgraph.candidate_negatives) for subgraph in subgraphs),
        total_pool_negatives=sum(
            len(subgraph.hard_pool) + len(subgraph.uniform_pool) for subgraph in subgraphs
        ),
    )


def sample_epoch_topology_subset(
    *,
    plan: TopologySubsetPlan,
    epoch: int,
    config: TopologySubsetSamplerConfig | None = None,
) -> tuple[TopologyPairSample, ...]:
    """Draw one epoch's topology pair subset from cached pools."""
    cfg = config or TopologySubsetSamplerConfig()
    cfg.validate()
    rng = random.Random(cfg.seed + epoch)
    selected: list[TopologyPairSample] = []
    for subgraph in plan.subgraphs:
        selected.extend(subgraph.positives)
        positives = max(1, len(subgraph.positives))
        epoch_negative_count = cfg.epoch_ratio * positives
        hard_count = min(
            len(subgraph.hard_pool), int(round(epoch_negative_count * cfg.hard_fraction))
        )
        uniform_count = min(len(subgraph.uniform_pool), max(0, epoch_negative_count - hard_count))
        hard_draws = _draw_without_replacement(subgraph.hard_pool, count=hard_count, rng=rng)
        uniform_draws = _draw_without_replacement(
            subgraph.uniform_pool, count=uniform_count, rng=rng
        )
        hard_pi_epoch = 1.0 if not subgraph.hard_pool else hard_count / float(len(subgraph.hard_pool))
        uniform_pi_epoch = (
            1.0 if not subgraph.uniform_pool else uniform_count / float(len(subgraph.uniform_pool))
        )
        selected.extend(
            _replace_sample_probability(
                sample,
                stratum=SamplingStratum.HARD_NEGATIVE,
                pi_pool_given_cand=sample.pi_pool_given_cand,
                pi_epoch_given_pool=hard_pi_epoch,
            )
            for sample in hard_draws
        )
        selected.extend(
            _replace_sample_probability(
                sample,
                stratum=SamplingStratum.UNIFORM_NEGATIVE,
                pi_pool_given_cand=sample.pi_pool_given_cand,
                pi_epoch_given_pool=uniform_pi_epoch,
            )
            for sample in uniform_draws
        )
    for sample in selected:
        sample.validate()
    return tuple(selected)


def _sample_to_dict(sample: TopologyPairSample) -> dict[str, object]:
    return {
        "pair_id": sample.pair_id,
        "subgraph_id": sample.subgraph_id,
        "node_size": sample.node_size,
        "protein_a": sample.protein_a,
        "protein_b": sample.protein_b,
        "local_index_a": sample.local_index_a,
        "local_index_b": sample.local_index_b,
        "stratum": sample.stratum.value,
        "pi_cand": sample.pi_cand,
        "pi_pool_given_cand": sample.pi_pool_given_cand,
        "pi_epoch_given_pool": sample.pi_epoch_given_pool,
        "pi_total": sample.pi_total,
        "target": sample.target,
        "scorer_probability": sample.scorer_probability,
    }


def _sample_from_dict(raw: Mapping[str, object]) -> TopologyPairSample:
    return TopologyPairSample(
        pair_id=str(raw["pair_id"]),
        subgraph_id=str(raw["subgraph_id"]),
        node_size=int(raw["node_size"]),  # type: ignore[arg-type]
        protein_a=str(raw["protein_a"]),
        protein_b=str(raw["protein_b"]),
        local_index_a=int(raw["local_index_a"]),  # type: ignore[arg-type]
        local_index_b=int(raw["local_index_b"]),  # type: ignore[arg-type]
        stratum=SamplingStratum(str(raw["stratum"])),
        pi_cand=float(raw["pi_cand"]),  # type: ignore[arg-type]
        pi_pool_given_cand=float(raw["pi_pool_given_cand"]),  # type: ignore[arg-type]
        pi_epoch_given_pool=float(raw["pi_epoch_given_pool"]),  # type: ignore[arg-type]
        pi_total=float(raw["pi_total"]),  # type: ignore[arg-type]
        target=float(raw["target"]),  # type: ignore[arg-type]
        scorer_probability=float(raw["scorer_probability"]),  # type: ignore[arg-type]
    )


SUBSET_PAYLOAD_VERSION = 1


def subset_plan_to_payload(plan: TopologySubsetPlan) -> dict[str, object]:
    """Serialize a TopologySubsetPlan to a JSON-friendly payload.

    The ``payload_kind``/``subset_payload_version`` stamp lets the subset cache loader
    reject a full-plan payload (different shape) or a stale subset schema instead of
    crashing in ``payload_to_subset_plan`` on a missing key.
    """
    return {
        "payload_kind": "topology_subset",
        "subset_payload_version": SUBSET_PAYLOAD_VERSION,
        "active_sizes": list(plan.active_sizes),
        "skipped_sizes": {str(size): reason for size, reason in plan.skipped_sizes.items()},
        "total_positive_pairs": plan.total_positive_pairs,
        "total_candidate_negatives": plan.total_candidate_negatives,
        "total_pool_negatives": plan.total_pool_negatives,
        "subgraphs": [
            {
                "subgraph_id": subgraph.subgraph_id,
                "node_size": subgraph.node_size,
                "nodes": list(subgraph.nodes),
                "positives": [_sample_to_dict(sample) for sample in subgraph.positives],
                "candidate_negatives": [
                    _sample_to_dict(sample) for sample in subgraph.candidate_negatives
                ],
                "hard_pool": [_sample_to_dict(sample) for sample in subgraph.hard_pool],
                "uniform_pool": [_sample_to_dict(sample) for sample in subgraph.uniform_pool],
            }
            for subgraph in plan.subgraphs
        ],
    }


def payload_to_subset_plan(payload: Mapping[str, object]) -> TopologySubsetPlan:
    """Rebuild a TopologySubsetPlan from its serialized payload."""
    raw_subgraphs = cast("Sequence[Mapping[str, object]]", payload["subgraphs"])
    subgraphs = tuple(
        TopologySubgraphPlan(
            subgraph_id=str(raw["subgraph_id"]),
            node_size=int(raw["node_size"]),  # type: ignore[arg-type]
            nodes=tuple(str(node) for node in cast("Sequence[object]", raw["nodes"])),
            positives=tuple(
                _sample_from_dict(item)
                for item in cast("Sequence[Mapping[str, object]]", raw["positives"])
            ),
            candidate_negatives=tuple(
                _sample_from_dict(item)
                for item in cast("Sequence[Mapping[str, object]]", raw["candidate_negatives"])
            ),
            hard_pool=tuple(
                _sample_from_dict(item)
                for item in cast("Sequence[Mapping[str, object]]", raw["hard_pool"])
            ),
            uniform_pool=tuple(
                _sample_from_dict(item)
                for item in cast("Sequence[Mapping[str, object]]", raw["uniform_pool"])
            ),
        )
        for raw in raw_subgraphs
    )
    skipped_raw = cast("Mapping[str, object]", payload["skipped_sizes"])
    return TopologySubsetPlan(
        subgraphs=subgraphs,
        active_sizes=tuple(int(size) for size in cast("Sequence[object]", payload["active_sizes"])),
        skipped_sizes={int(size): str(reason) for size, reason in skipped_raw.items()},
        total_positive_pairs=int(payload["total_positive_pairs"]),  # type: ignore[arg-type]
        total_candidate_negatives=int(payload["total_candidate_negatives"]),  # type: ignore[arg-type]
        total_pool_negatives=int(payload["total_pool_negatives"]),  # type: ignore[arg-type]
    )


def candidate_pairs_for_scoring(plan: TopologySubsetPlan) -> tuple[tuple[str, str, str], ...]:
    """Return unique candidate pair ids and endpoints that need frozen-scorer scores."""
    by_id: dict[str, tuple[str, str, str]] = {}
    for subgraph in plan.subgraphs:
        for sample in (*subgraph.positives, *subgraph.candidate_negatives):
            by_id.setdefault(sample.pair_id, (sample.pair_id, sample.protein_a, sample.protein_b))
    return tuple(by_id[pair_id] for pair_id in sorted(by_id))


def scored_pairs_from_subset_plan(
    plan: TopologySubsetPlan,
) -> tuple[list[tuple[str, str]], list[float]]:
    """Return (endpoints, scorer_probability) for every unique scored pair, id-ordered."""
    by_id: dict[str, tuple[str, str, float]] = {}
    for subgraph in plan.subgraphs:
        for sample in (*subgraph.positives, *subgraph.candidate_negatives):
            by_id.setdefault(
                sample.pair_id,
                (sample.protein_a, sample.protein_b, sample.scorer_probability),
            )
    endpoints = [(by_id[pid][0], by_id[pid][1]) for pid in sorted(by_id)]
    probabilities = [by_id[pid][2] for pid in sorted(by_id)]
    return endpoints, probabilities


def _induced_covered_edges(
    sampled: Mapping[int, Sequence[tuple[str, ...]]], graph: nx.Graph
) -> set[frozenset[str]]:
    covered: set[frozenset[str]] = set()
    for rows in sampled.values():
        for nodes in rows:
            for node_a, node_b in graph.subgraph(set(nodes)).edges():
                covered.add(frozenset((node_a, node_b)))
    return covered


def apply_per_size_subgraph_budget(
    *,
    graph: nx.Graph,
    base_sampled: Mapping[int, Sequence[tuple[str, ...]]],
    node_sizes: tuple[int, ...],
    strategy: str,
    seed: int,
    max_subgraphs_per_size: int,
    coverage_augmentation: bool = True,
) -> tuple[dict[int, list[tuple[str, ...]]], dict[str, float | int]]:
    """Cap base subgraphs per size, then add budget-aware coverage subgraphs.

    This is the SINGLE entry point for the subset path's budget + coverage logic. Do
    NOT call ``augment_plan_for_positive_edge_coverage`` separately around it — that
    helper appends coverage buckets to ``max(node_sizes)`` and asserts full coverage,
    which both bypasses the per-size budget and would double-augment. This helper
    instead interleaves capping and coverage so the cap can never silently delete a
    coverage subgraph that augmentation just added (review: coverage-redistribution
    finding).

    Order of operations:
      1. Cap each size's base subgraphs to ``max_subgraphs_per_size`` (0 == unbounded).
      2. If ``coverage_augmentation``, walk the still-uncovered positive edges and place
         a coverage subgraph for each into the smallest eligible size that still has
         remaining budget — so the largest bucket is not the sole coverage dump.
      3. Recompute coverage from the realized (post-budget) plan and return it, so the
         logged number is the coverage that the built plan actually has.

    Unlike ``augment_plan_for_positive_edge_coverage``, this does NOT raise when coverage
    stays below 1.0 under a tight budget: budgeting is an explicit memory/coverage
    tradeoff and the realized coverage is reported in ``stats`` for the operator to see.
    """
    from src.topology.finetune_data import _expand_chunk_nodes  # local import: heavy module

    rng = random.Random(seed)
    normalized = strategy.upper()
    if normalized not in {"BFS", "DFS", "RANDOM_WALK"}:
        normalized = "BFS"
    # (1) Cap the base sampled subgraphs per size.
    budgeted: dict[int, list[tuple[str, ...]]] = {}
    for size in sorted(int(s) for s in base_sampled):
        rows = [tuple(sorted(nodes)) for nodes in base_sampled[size]]
        if max_subgraphs_per_size > 0:
            rows = rows[:max_subgraphs_per_size]
        budgeted[size] = rows
    base_bucket_count = sum(len(rows) for rows in budgeted.values())

    def _remaining(size: int) -> int:
        if max_subgraphs_per_size <= 0:
            return 2**31  # effectively unbounded
        return max(0, max_subgraphs_per_size - len(budgeted.get(size, [])))

    eligible_sizes = tuple(
        size for size in sorted(node_sizes) if size <= graph.number_of_nodes()
    )
    all_positive = {frozenset((node_a, node_b)) for node_a, node_b in graph.edges()}
    covered = _induced_covered_edges(budgeted, graph)
    coverage_bucket_count = 0
    # (2) Distribute coverage subgraphs across eligible sizes under remaining budget.
    if coverage_augmentation:
        for edge in sorted(tuple(sorted(e)) for e in (all_positive - covered)):
            if frozenset(edge) in covered:
                continue  # drained by a previously added coverage bucket
            target_size = next(
                (size for size in eligible_sizes if _remaining(size) > 0), None
            )
            if target_size is None:
                break  # every eligible size is at budget; report honest partial coverage
            nodes = _expand_chunk_nodes(
                graph=graph,
                edge_chunk=[(edge[0], edge[1])],
                target_size=target_size,
                strategy=normalized,
                rng=rng,
            )
            budgeted.setdefault(target_size, []).append(tuple(sorted(nodes)))
            for node_a, node_b in graph.subgraph(set(nodes)).edges():
                covered.add(frozenset((node_a, node_b)))
            coverage_bucket_count += 1

    # (3) Recompute coverage from the realized plan so the logged number is honest.
    matched = len(covered & all_positive)
    coverage = 1.0 if not all_positive else matched / len(all_positive)
    stats: dict[str, float | int] = {
        "base_bucket_count": base_bucket_count,
        "coverage_bucket_count": coverage_bucket_count,
        "positive_edge_coverage": coverage,
    }
    return budgeted, stats


@dataclass(frozen=True)
class TopologySubgraphEpochChunk:
    """One differentiable topology chunk for a sampled subgraph in one epoch."""

    subgraph_id: str
    node_size: int
    samples: tuple[TopologyPairSample, ...]


def group_epoch_samples_by_subgraph(
    samples: Sequence[TopologyPairSample],
) -> dict[str, TopologySubgraphEpochChunk]:
    """Group epoch samples into subgraph chunks."""
    grouped: dict[str, list[TopologyPairSample]] = defaultdict(list)
    size_by_id: dict[str, int] = {}
    for sample in samples:
        grouped[sample.subgraph_id].append(sample)
        size_by_id[sample.subgraph_id] = sample.node_size
    return {
        subgraph_id: TopologySubgraphEpochChunk(
            subgraph_id=subgraph_id,
            node_size=size_by_id[subgraph_id],
            samples=tuple(rows),
        )
        for subgraph_id, rows in sorted(grouped.items())
    }


def relative_error(*, estimate: float, reference: float) -> float:
    """Return bounded relative error for diagnostic logging."""
    if reference == 0.0:
        return 0.0 if estimate == 0.0 else 1.0
    return abs(estimate - reference) / abs(reference)


def compute_subset_bias_diagnostic(
    *,
    node_size: int,
    full_space_probabilities: Mapping[str, float],
    subset_samples: Sequence[tuple[str, float, float]],
) -> dict[str, float]:
    """Compare IPW-reweighted subset statistics against exact full-space statistics.

    Shared primitive for the §9 smoke sanity check and the §3.7 production diagnostic.

    - `full_space_probabilities`: every upper-triangle pair id -> predicted probability
      over the FULL `n·(n-1)/2` pair space of one (capped) subgraph.
    - `subset_samples`: `(pair_id, predicted_probability, weight)` for the pairs the
      sampler actually selected, where `weight == 1 / pi_total`.

    Returns per-metric relative error of the IPW subset estimate vs the exact full-space
    value for density and mean soft-degree (both linear accumulators, so HT-unbiased).
    """
    if node_size < 2:
        raise ValueError("node_size must be >= 2")
    normalizer = float(node_size * (node_size - 1))
    # Exact full-space statistics.
    full_sum = sum(full_space_probabilities.values())
    full_density = (2.0 * full_sum) / normalizer
    # IPW subset estimates (Horvitz-Thompson) of the same linear accumulators.
    ipw_sum = sum(prob * weight for _pair_id, prob, weight in subset_samples)
    ipw_density = (2.0 * ipw_sum) / normalizer
    # Mean soft degree = (2 * sum of pair probs) / num_nodes for both views.
    full_mean_degree = (2.0 * full_sum) / float(node_size)
    ipw_mean_degree = (2.0 * ipw_sum) / float(node_size)
    return {
        "density_estimate": ipw_density,
        "density_reference": full_density,
        "density_relative_error": relative_error(
            estimate=ipw_density, reference=full_density
        ),
        "mean_degree_estimate": ipw_mean_degree,
        "mean_degree_reference": full_mean_degree,
        "mean_degree_relative_error": relative_error(
            estimate=ipw_mean_degree, reference=full_mean_degree
        ),
        "full_space_pairs": float(len(full_space_probabilities)),
        "subset_pairs": float(len(subset_samples)),
    }
