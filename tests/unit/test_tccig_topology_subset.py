"""Tests for bounded TCCIG topology subset sampling."""

from __future__ import annotations

import networkx as nx
import pytest
from tccig.topology_subset import (
    SamplingStratum,
    TopologyPairSample,
    TopologySubgraphPlan,
    TopologySubsetPlan,
    TopologySubsetSamplerConfig,
    active_node_sizes,
    apply_per_size_subgraph_budget,
    build_topology_subset_plan,
    candidate_pairs_for_scoring,
    canonical_pair_id,
    compute_subset_bias_diagnostic,
    diagnostic_full_space_scoring_pairs,
    group_epoch_samples_by_subgraph,
    payload_to_subset_plan,
    relative_error,
    sample_epoch_topology_subset,
    scored_pairs_from_subset_plan,
    select_diagnostic_subgraphs,
    subset_plan_to_payload,
)


def test_pair_sample_validates_three_stage_probability_product() -> None:
    sample = TopologyPairSample(
        pair_id="a||b",
        subgraph_id="size=4:index=0",
        node_size=4,
        protein_a="a",
        protein_b="b",
        local_index_a=0,
        local_index_b=1,
        stratum=SamplingStratum.UNIFORM_NEGATIVE,
        pi_cand=0.5,
        pi_pool_given_cand=0.4,
        pi_epoch_given_pool=0.25,
        pi_total=0.05,
        target=0.0,
        scorer_probability=0.2,
    )
    sample.validate()


def test_pair_sample_rejects_bad_probability_product() -> None:
    sample = TopologyPairSample(
        pair_id="a||b",
        subgraph_id="size=4:index=0",
        node_size=4,
        protein_a="a",
        protein_b="b",
        local_index_a=0,
        local_index_b=1,
        stratum=SamplingStratum.HARD_NEGATIVE,
        pi_cand=0.5,
        pi_pool_given_cand=0.5,
        pi_epoch_given_pool=0.5,
        pi_total=0.5,
        target=0.0,
        scorer_probability=0.9,
    )
    with pytest.raises(ValueError, match="pi_total must equal"):
        sample.validate()


def test_positive_sample_must_have_probability_one() -> None:
    sample = TopologyPairSample(
        pair_id="a||b",
        subgraph_id="size=4:index=0",
        node_size=4,
        protein_a="a",
        protein_b="b",
        local_index_a=0,
        local_index_b=1,
        stratum=SamplingStratum.POSITIVE,
        pi_cand=1.0,
        pi_pool_given_cand=1.0,
        pi_epoch_given_pool=0.5,
        pi_total=0.5,
        target=1.0,
        scorer_probability=0.8,
    )
    with pytest.raises(ValueError, match="positive samples must have"):
        sample.validate()


def test_active_node_sizes_drop_zero_budget_sizes() -> None:
    active, skipped = active_node_sizes(
        node_sizes=(20, 40, 80),
        graph_node_count=60,
        subgraphs_per_size={20: 3, 40: 1, 80: 2},
        labeled_pairs_per_size={20: 100, 40: 0, 80: 200},
    )
    assert active == (20,)
    assert skipped == {40: "zero_labeled_pair_budget", 80: "larger_than_graph"}


def test_sampler_config_defaults_match_rerun_decision() -> None:
    cfg = TopologySubsetSamplerConfig()
    assert cfg.enabled is True
    assert cfg.candidate_ratio == 20
    assert cfg.pool_ratio == 10
    assert cfg.epoch_ratio == 5
    assert cfg.hard_fraction == pytest.approx(0.5)
    assert cfg.uniform_fraction == pytest.approx(0.5)
    assert cfg.hard_stratum_fraction == pytest.approx(0.2)
    # Per-size budget (spec §4): default unbounded (0).
    assert cfg.max_subgraphs_per_size == 0
    assert cfg.max_labeled_pairs_per_size == 0


def test_sampler_config_rejects_negative_budget() -> None:
    with pytest.raises(ValueError, match="max_subgraphs_per_size must be >= 0"):
        TopologySubsetSamplerConfig(max_subgraphs_per_size=-1).validate()


def _toy_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d", "e"])
    graph.add_edges_from([("a", "b"), ("b", "c")])
    return graph


def test_build_subset_plan_scores_only_candidate_frame() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d")]}
    cfg = TopologySubsetSamplerConfig(
        candidate_ratio=2,
        pool_ratio=1,
        epoch_ratio=1,
        hard_fraction=0.5,
        uniform_fraction=0.5,
        hard_stratum_fraction=0.5,
        seed=3,
    )
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities={},
    )
    assert plan.total_positive_pairs == 2
    assert plan.total_candidate_negatives <= 4
    assert plan.total_candidate_negatives < 4 * 3 // 2


def test_epoch_subset_keeps_all_positives_and_samples_negatives_with_pi() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d")]}
    cfg = TopologySubsetSamplerConfig(
        candidate_ratio=3,
        pool_ratio=2,
        epoch_ratio=1,
        hard_fraction=0.5,
        uniform_fraction=0.5,
        hard_stratum_fraction=0.5,
        seed=4,
    )
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities={"a||c": 0.9, "a||d": 0.2, "b||d": 0.7, "c||d": 0.1},
    )
    samples = sample_epoch_topology_subset(plan=plan, epoch=1)
    positives = [sample for sample in samples if sample.stratum is SamplingStratum.POSITIVE]
    negatives = [sample for sample in samples if sample.stratum is not SamplingStratum.POSITIVE]
    assert {sample.pair_id for sample in positives} == {"a||b", "b||c"}
    assert negatives
    assert all(0.0 < sample.pi_total <= 1.0 for sample in negatives)
    assert all(sample.target == 0.0 for sample in negatives)


def test_builder_trusts_pre_budgeted_subgraph_counts() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d"), ("a", "b", "c", "e"), ("a", "b", "d", "e")]}
    cfg = TopologySubsetSamplerConfig(
        candidate_ratio=2,
        pool_ratio=1,
        epoch_ratio=1,
        hard_stratum_fraction=0.5,
        seed=3,
        max_subgraphs_per_size=1,  # honored upstream, NOT by the builder
    )
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities={},
    )
    assert len([sg for sg in plan.subgraphs if sg.node_size == 4]) == 3


def test_max_labeled_pairs_per_size_caps_scored_candidates() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d"), ("a", "b", "c", "e")]}
    capped = TopologySubsetSamplerConfig(
        candidate_ratio=10,
        pool_ratio=1,
        epoch_ratio=1,
        hard_stratum_fraction=0.5,
        seed=3,
        max_labeled_pairs_per_size=3,
    )
    uncapped = TopologySubsetSamplerConfig(
        candidate_ratio=10,
        pool_ratio=1,
        epoch_ratio=1,
        hard_stratum_fraction=0.5,
        seed=3,
    )
    capped_plan = build_topology_subset_plan(
        graph=graph, sampled_subgraphs=sampled, config=capped, scorer_probabilities={}
    )
    uncapped_plan = build_topology_subset_plan(
        graph=graph, sampled_subgraphs=sampled, config=uncapped, scorer_probabilities={}
    )
    size_four = sum(
        len(sg.candidate_negatives) for sg in capped_plan.subgraphs if sg.node_size == 4
    )
    assert size_four <= 3
    assert capped_plan.total_candidate_negatives < uncapped_plan.total_candidate_negatives
    for subgraph in capped_plan.subgraphs:
        for sample in subgraph.candidate_negatives:
            assert 0.0 < sample.pi_cand <= 1.0


def test_subset_plan_payload_round_trips() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d")]}
    cfg = TopologySubsetSamplerConfig(candidate_ratio=3, pool_ratio=2, epoch_ratio=1, seed=4)
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities={"a||c": 0.9, "a||d": 0.2, "b||d": 0.7, "c||d": 0.1},
    )
    restored = payload_to_subset_plan(subset_plan_to_payload(plan))
    assert restored == plan


def test_candidate_pairs_for_scoring_are_unique_and_ordered() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d"), ("a", "b", "c", "e")]}
    cfg = TopologySubsetSamplerConfig(candidate_ratio=2, pool_ratio=1, epoch_ratio=1, seed=7)
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities={},
    )
    pairs = candidate_pairs_for_scoring(plan)
    pair_ids = [pair_id for pair_id, _, _ in pairs]
    assert pair_ids == sorted(set(pair_ids))


def test_budget_distributes_coverage_and_reports_realized_coverage() -> None:
    # size-4 base is capped to 1, but a coverage subgraph for the uncovered edge d-e
    # must still be placed (in an eligible size with remaining budget), and the
    # returned stats must reflect the realized (post-cap) coverage, not the pre-cap claim.
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d", "e", "f"])
    graph.add_edges_from([("a", "b"), ("b", "c"), ("d", "e")])
    base_sampled = {4: [("a", "b", "c", "f"), ("a", "b", "c", "d"), ("a", "c", "d", "f")]}
    # node_sizes includes a smaller eligible size (3) so the d-e coverage subgraph has a
    # bucket with remaining budget after size-4 is capped to 2 (review: with only size 4
    # available the cap would exhaust the sole eligible size and coverage could not be
    # placed at all — the helper distributes coverage into the smallest free size).
    budgeted, stats = apply_per_size_subgraph_budget(
        graph=graph,
        base_sampled=base_sampled,
        node_sizes=(3, 4),
        strategy="BFS",
        seed=0,
        max_subgraphs_per_size=2,
    )
    # No size exceeds its cap.
    assert all(len(rows) <= 2 for rows in budgeted.values())
    # The d-e positive edge is covered by some retained subgraph.
    covered_edges = {
        frozenset((u, v))
        for rows in budgeted.values()
        for nodes in rows
        for u, v in graph.subgraph(set(nodes)).edges()
    }
    assert frozenset(("d", "e")) in covered_edges
    # Realized coverage is reported honestly in [0, 1].
    assert 0.0 <= float(stats["positive_edge_coverage"]) <= 1.0
    assert stats["positive_edge_coverage"] == 1.0


def test_budget_unbounded_is_passthrough_with_coverage() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(["a", "b", "c", "d"])
    graph.add_edges_from([("a", "b"), ("c", "d")])
    base_sampled = {4: [("a", "b", "c", "d")]}
    budgeted, stats = apply_per_size_subgraph_budget(
        graph=graph,
        base_sampled=base_sampled,
        node_sizes=(4,),
        strategy="BFS",
        seed=0,
        max_subgraphs_per_size=0,  # unbounded
    )
    assert budgeted[4] == [("a", "b", "c", "d")]
    assert stats["positive_edge_coverage"] == 1.0


def test_scored_pairs_from_subset_plan_track_scorer_probability() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d")]}
    cfg = TopologySubsetSamplerConfig(candidate_ratio=3, pool_ratio=2, epoch_ratio=1, seed=4)
    scores = {"a||c": 0.9, "a||d": 0.2, "b||d": 0.7, "c||d": 0.1}
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities=scores,
    )
    endpoints, probabilities = scored_pairs_from_subset_plan(plan)
    # id-ordered uniqueness: one entry per unique pair id, sorted by id.
    pair_ids = [canonical_pair_id(a, b) for a, b in endpoints]
    assert pair_ids == sorted(set(pair_ids))
    # Probabilities track the stored scorer_probability for each pair.
    by_id = {
        sample.pair_id: sample.scorer_probability
        for subgraph in plan.subgraphs
        for sample in (*subgraph.positives, *subgraph.candidate_negatives)
    }
    for (a, b), prob in zip(endpoints, probabilities, strict=True):
        assert prob == by_id[canonical_pair_id(a, b)]


def test_group_epoch_samples_by_subgraph_preserves_size_and_weights() -> None:
    graph = _toy_graph()
    sampled = {4: [("a", "b", "c", "d")]}
    cfg = TopologySubsetSamplerConfig(candidate_ratio=3, pool_ratio=2, epoch_ratio=1, seed=4)
    plan = build_topology_subset_plan(
        graph=graph,
        sampled_subgraphs=sampled,
        config=cfg,
        scorer_probabilities={"a||c": 0.9, "a||d": 0.2, "b||d": 0.7, "c||d": 0.1},
    )
    grouped = group_epoch_samples_by_subgraph(sample_epoch_topology_subset(plan=plan, epoch=1))
    assert list(grouped) == ["size=4:index=0"]
    chunk = grouped["size=4:index=0"]
    assert chunk.node_size == 4
    assert len(chunk.samples) >= 2
    assert all(sample.pi_total > 0.0 for sample in chunk.samples)


def test_relative_error_handles_zero_reference() -> None:
    assert relative_error(estimate=0.0, reference=0.0) == 0.0
    assert relative_error(estimate=1.0, reference=0.0) == 1.0
    assert relative_error(estimate=9.0, reference=10.0) == pytest.approx(0.1)


def test_bias_diagnostic_recovers_full_space_density_under_ipw() -> None:
    # Full space: a 4-node subgraph, all 6 upper-triangle pairs with known probs.
    # The IPW estimate from a pi-sampled subset must approximately recover the exact
    # full-space density (Horvitz-Thompson unbiasedness of the linear numerator).
    full_probs = {
        "a||b": 0.9,
        "a||c": 0.1,
        "a||d": 0.7,
        "b||c": 0.3,
        "b||d": 0.5,
        "c||d": 0.2,
    }
    # A subset that kept every pair (pi=1) must give EXACTLY the full-space stats.
    subset = [(pair_id, prob, 1.0) for pair_id, prob in full_probs.items()]
    diagnostic = compute_subset_bias_diagnostic(
        node_size=4,
        full_space_probabilities=full_probs,
        subset_samples=subset,
    )
    assert diagnostic["density_relative_error"] == pytest.approx(0.0, abs=1e-9)
    assert diagnostic["mean_degree_relative_error"] == pytest.approx(0.0, abs=1e-9)


def test_bias_diagnostic_flags_missing_weight() -> None:
    # If a down-sampled pair (pi=0.5) is NOT reweighted (weight forced to 1.0), the
    # density estimate is biased low and the diagnostic must report nonzero error.
    full_probs = {"a||b": 1.0, "a||c": 1.0, "b||c": 1.0}
    # Keep only 2 of 3 pairs, each with pi=0.5 but WRONG weight 1.0 (bug simulation).
    subset = [("a||b", 1.0, 1.0), ("a||c", 1.0, 1.0)]
    diagnostic = compute_subset_bias_diagnostic(
        node_size=3,
        full_space_probabilities=full_probs,
        subset_samples=subset,
    )
    assert diagnostic["density_relative_error"] > 0.1


def test_select_diagnostic_subgraphs_spreads_across_size_mixture() -> None:
    # _select_diagnostic_subgraphs lives in s2gae.py (it needs only plan/chunk metadata,
    # no model), so a tight max_subgraphs budget must still sample the SIZE MIXTURE
    # (round-robin one per active size first), not just the smallest size (Finding 3).
    from tccig.s2gae import _select_diagnostic_subgraphs
    from tccig.topology_subset import (
        TopologySubgraphEpochChunk,
        TopologySubgraphPlan,
        TopologySubsetPlan,
    )

    def _plan_subgraph(size: int, index: int) -> TopologySubgraphPlan:
        return TopologySubgraphPlan(
            subgraph_id=f"size={size}:index={index}",
            node_size=size,
            nodes=tuple(f"s{size}_n{index}_{j}" for j in range(size)),
            positives=(),
            candidate_negatives=(),
            hard_pool=(),
            uniform_pool=(),
        )

    subgraphs = tuple(
        _plan_subgraph(size, index) for size in (4, 8) for index in range(3)
    )
    plan = TopologySubsetPlan(
        subgraphs=subgraphs,
        active_sizes=(4, 8),
        skipped_sizes={},
        total_positive_pairs=0,
        total_candidate_negatives=0,
        total_pool_negatives=0,
    )
    # Every subgraph produced epoch samples this round.
    chunk_by_id = {
        sg.subgraph_id: TopologySubgraphEpochChunk(
            subgraph_id=sg.subgraph_id, node_size=sg.node_size, samples=()
        )
        for sg in subgraphs
    }
    selected = _select_diagnostic_subgraphs(
        plan=plan, chunk_by_id=chunk_by_id, max_node_size=40, max_subgraphs=2
    )
    # Budget of 2 must take one subgraph from EACH active size, not two from size 4.
    assert {sg.node_size for sg in selected} == {4, 8}
    # max_node_size filters out sizes above the cap.
    capped = _select_diagnostic_subgraphs(
        plan=plan, chunk_by_id=chunk_by_id, max_node_size=4, max_subgraphs=0
    )
    assert {sg.node_size for sg in capped} == {4}
    assert len(capped) == 3  # 0 == every eligible subgraph


def test_select_diagnostic_subgraphs_is_deterministic_and_limited() -> None:
    def _subgraph(size: int, index: int) -> TopologySubgraphPlan:
        return TopologySubgraphPlan(
            subgraph_id=f"size={size}:index={index}",
            node_size=size,
            nodes=tuple(f"n{size}_{index}_{j}" for j in range(size)),
            positives=(),
            candidate_negatives=(),
            hard_pool=(),
            uniform_pool=(),
        )

    plan = TopologySubsetPlan(
        subgraphs=tuple(_subgraph(size, index) for size in (4, 8) for index in range(3)),
        active_sizes=(4, 8),
        skipped_sizes={},
        total_positive_pairs=0,
        total_candidate_negatives=0,
        total_pool_negatives=0,
    )

    selected = select_diagnostic_subgraphs(plan, max_node_size=40, max_subgraphs=2)
    assert [subgraph.subgraph_id for subgraph in selected] == [
        "size=4:index=0",
        "size=8:index=0",
    ]
    assert selected == select_diagnostic_subgraphs(
        plan, max_node_size=40, max_subgraphs=2
    )

    capped = select_diagnostic_subgraphs(plan, max_node_size=4, max_subgraphs=0)
    assert {subgraph.node_size for subgraph in capped} == {4}
    assert len(capped) == 3


def test_diagnostic_full_space_scoring_pairs_are_unique_and_ordered() -> None:
    shared = TopologySubgraphPlan(
        subgraph_id="size=4:index=0",
        node_size=4,
        nodes=("a", "b", "c", "d"),
        positives=(),
        candidate_negatives=(),
        hard_pool=(),
        uniform_pool=(),
    )
    overlapping = TopologySubgraphPlan(
        subgraph_id="size=4:index=1",
        node_size=4,
        nodes=("a", "b", "c", "e"),
        positives=(),
        candidate_negatives=(),
        hard_pool=(),
        uniform_pool=(),
    )
    plan = TopologySubsetPlan(
        subgraphs=(overlapping, shared),
        active_sizes=(4,),
        skipped_sizes={},
        total_positive_pairs=0,
        total_candidate_negatives=0,
        total_pool_negatives=0,
    )

    rows = diagnostic_full_space_scoring_pairs(
        plan,
        max_node_size=40,
        max_subgraphs=0,
    )

    pair_ids = [row[0] for row in rows]
    assert pair_ids == sorted(pair_ids)
    assert pair_ids == sorted(set(pair_ids))
    assert len(rows) == 9
    assert rows[0] == ("a||b", "a", "b")
