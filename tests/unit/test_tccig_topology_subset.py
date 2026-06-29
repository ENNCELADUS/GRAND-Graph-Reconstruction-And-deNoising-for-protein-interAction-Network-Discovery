"""Tests for bounded TCCIG topology subset sampling."""

from __future__ import annotations

import networkx as nx
import pytest

from tccig.topology_subset import (
    SamplingStratum,
    TopologyPairSample,
    TopologySubsetSamplerConfig,
    active_node_sizes,
    build_topology_subset_plan,
    sample_epoch_topology_subset,
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


from tccig.topology_subset import subset_plan_to_payload, payload_to_subset_plan


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
