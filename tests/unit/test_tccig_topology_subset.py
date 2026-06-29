"""Tests for bounded TCCIG topology subset sampling."""

from __future__ import annotations

import pytest

from tccig.topology_subset import (
    SamplingStratum,
    TopologyPairSample,
    TopologySubsetSamplerConfig,
    active_node_sizes,
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
