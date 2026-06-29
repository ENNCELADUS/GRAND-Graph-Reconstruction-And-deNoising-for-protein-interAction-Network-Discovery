"""Bounded subset sampling contracts for TCCIG topology training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isclose


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
