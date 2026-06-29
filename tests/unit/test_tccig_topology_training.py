"""Tests for TCCIG Run 02 topology-conditioned training loss."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
import tccig.train as tccig_train
import torch
from src.topology.finetune_data import build_internal_validation_plan
from tccig.s2gae import asymmetric_residual_anchor
from tccig.train import (
    _coverage_stats_from_payload,
    _load_or_build_topology_plan,
    augment_plan_for_positive_edge_coverage,
)


def test_asymmetric_anchor_leaves_deletion_free() -> None:
    negative_delta = torch.tensor([-3.0, -1.0, -0.5])
    assert float(asymmetric_residual_anchor(negative_delta)) == 0.0


def test_asymmetric_anchor_penalizes_upward_push() -> None:
    positive_delta = torch.tensor([2.0, 0.0, -4.0])
    # only +2.0 contributes: (2^2 + 0 + 0) / 3
    assert float(asymmetric_residual_anchor(positive_delta)) == pytest.approx(4.0 / 3.0)


def _base_refiner_config() -> dict:
    return {
        "encoder": "graphconv",
        "input_dim": 8,
        "embedding_cache_dir": "data/embeddings/esm3_1024",
        "monitor_metric": "val_topology_loss",
        "topology_validation": {
            "enabled": True,
            "losses": {"alpha": 1.0, "beta": 8.0, "gamma": 0.5, "delta": 0.1},
        },
        "optimizer": {"type": "adamw", "lr": 1e-4},
        "scheduler": {"type": "none"},
        "optimization": {"gradient_clip_norm": 1.0},
        "residual_anchor": {"form": "asymmetric_relu", "weight": 1.0e-4},
        "topology_training": {
            "enabled": True,
            "node_sizes": [20, 40],
            "samples_per_size": 5,
            "strategy": "mixed",
            "seed": 0,
            "coverage_augmentation": True,
            "topology_weight": 1.0,
            "weights": {"alpha": 1.0, "beta": 8.0, "gamma": 0.5, "delta": 0.1},
            "schedule": {"warmup_epochs": 5, "ramp_epochs": 10, "schedule": "linear"},
        },
    }


def test_parse_config_reads_residual_anchor_and_topology_training() -> None:
    from tccig.s2gae import _parse_config

    cfg = _parse_config(_base_refiner_config())
    assert cfg.residual_anchor.form == "asymmetric_relu"
    assert cfg.residual_anchor.weight == pytest.approx(1.0e-4)
    assert cfg.topology_training.enabled is True
    assert cfg.topology_training.node_sizes == (20, 40)
    assert cfg.topology_training.topology_weight == pytest.approx(1.0)
    assert cfg.topology_training.weights.beta == pytest.approx(8.0)
    assert cfg.topology_training.warmup_epochs == 5
    assert cfg.topology_training.ramp_epochs == 10


def test_parse_config_defaults_topology_training_disabled() -> None:
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    del config["topology_training"]
    del config["residual_anchor"]
    cfg = _parse_config(config)
    assert cfg.topology_training.enabled is False
    assert cfg.residual_anchor.form == "symmetric"


def test_parse_config_reads_topology_subset_sampler() -> None:
    from tccig.s2gae import _parse_config

    config = _base_refiner_config()
    config["topology_training"]["subset"] = {
        "enabled": True,
        "candidate_ratio": 20,
        "pool_ratio": 10,
        "epoch_ratio": 5,
        "hard_fraction": 0.5,
        "uniform_fraction": 0.5,
        "hard_stratum_fraction": 0.2,
        "seed": 11,
    }
    cfg = _parse_config(config)
    assert cfg.topology_training.subset.enabled is True
    assert cfg.topology_training.subset.candidate_ratio == 20
    assert cfg.topology_training.subset.seed == 11
    assert cfg.topology_validation.compute_clustering_mmd is True


def test_coverage_augmentation_covers_isolated_positive_edge() -> None:
    import networkx as nx
    from tccig.train import augment_plan_for_positive_edge_coverage

    # A dense core plus one far-apart positive edge unlikely to be sampled at size 4.
    graph = nx.Graph()
    core = [f"C{i}" for i in range(6)]
    graph.add_edges_from((core[i], core[j]) for i in range(6) for j in range(i + 1, 6))
    graph.add_edge("FARLEFT", "FARRIGHT")  # connected via no core node
    # Force a base sample that misses the far edge.
    base_sampled = {4: [tuple(sorted(core[:4]))]}

    augmented, stats = augment_plan_for_positive_edge_coverage(
        graph=graph,
        base_sampled=base_sampled,
        node_sizes=[4],
        strategy="BFS",
        seed=0,
    )

    assert stats["positive_edge_coverage"] == 1.0
    assert stats["coverage_bucket_count"] >= 1
    # the previously-missing edge endpoints now appear together in some bucket
    covered = {
        frozenset((a, b))
        for nodes in [n for buckets in augmented.values() for n in buckets]
        for a in nodes
        for b in nodes
        if graph.has_edge(a, b)
    }
    assert frozenset(("FARLEFT", "FARRIGHT")) in covered


def test_coverage_augmentation_handles_positive_self_pair_in_pair_file(tmp_path: Path) -> None:
    import networkx as nx
    from src.topology.finetune_data import build_pair_supervision_graph
    from tccig.train import augment_plan_for_positive_edge_coverage

    # A positive self-pair (HPC job 928974 root cause) must be filtered by the
    # graph builder so coverage augmentation never sees a self-loop frozenset.
    pair_path = tmp_path / "human_train_ppi.txt"
    pair_path.write_text(
        "P1\tP2\t1\nP3\tP3\t1\nP3\tP4\t1\n",
        encoding="utf-8",
    )

    graph = build_pair_supervision_graph(
        pair_path=pair_path,
        node_ids={"P1", "P2", "P3", "P4"},
    )
    assert nx.number_of_selfloops(graph) == 0

    augmented, stats = augment_plan_for_positive_edge_coverage(
        graph=graph,
        base_sampled={2: [("P1", "P2")]},
        node_sizes=[2],
        strategy="BFS",
        seed=0,
    )

    assert stats["positive_edge_coverage"] == 1.0
    assert isinstance(augmented, dict)


def test_train_refiner_request_accepts_train_topology_fields() -> None:
    from tccig.s2gae import TrainRefinerRequest

    request = TrainRefinerRequest(
        train=None,  # type: ignore[arg-type]
        validation=None,  # type: ignore[arg-type]
        runtime=None,  # type: ignore[arg-type]
        config={},
        graph_rule=None,  # type: ignore[arg-type]
        train_topology=None,
        train_topology_plan=None,
    )
    assert request.train_topology is None
    assert request.train_topology_plan is None


@pytest.fixture
def make_tiny_refiner_and_plan():
    import networkx as nx
    from src.topology.finetune_data import build_internal_validation_plan
    from tccig.s2gae import S2GAERefiner, _SplitGraph

    def _build(*, overdense: bool) -> tuple[object, object, object, dict[str, int]]:
        torch.manual_seed(0)
        node_ids = ["P0", "P1", "P2", "P3"]
        node_index = {protein: idx for idx, protein in enumerate(node_ids)}
        num_nodes = len(node_ids)

        refiner = S2GAERefiner(
            encoder="graphconv",
            input_dim=8,
            hidden_dim=4,
            num_layers=2,
            decoder_hidden_dim=8,
            decoder_layers=2,
            dropout=0.0,
            encoder_aggr="mean",
            layer_norm=True,
            residual_scale=4.0,
        )
        # Force a non-zero residual decoder so deletion is trainable from the start.
        with torch.no_grad():
            output_layer = refiner.decoder.layers[-1]
            output_layer.weight.normal_(0.0, 0.1)
            output_layer.bias.fill_(0.5)

        node_features = torch.randn(num_nodes, 8)
        # All undirected pairs as candidate pairs.
        pair_columns = [
            (node_index[a], node_index[b])
            for ia, a in enumerate(node_ids)
            for b in node_ids[ia + 1 :]
        ]
        pair_index = torch.tensor(pair_columns, dtype=torch.long).t().contiguous()
        prob_value = 0.9 if overdense else 0.5
        pairwise_probabilities = torch.full((pair_index.size(1),), prob_value)
        # Encoder edges mirror the over-dense candidate set.
        edge_index = torch.cat([pair_index, pair_index.flip(0)], dim=1)
        edge_weight = torch.ones(edge_index.size(1))

        graph = _SplitGraph(
            node_features=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            pair_index=pair_index,
            pairwise_probabilities=pairwise_probabilities,
        )

        # True graph for this bucket has a single edge P0-P1.
        true_graph = nx.Graph()
        true_graph.add_nodes_from(node_ids)
        true_graph.add_edge("P0", "P1")
        plan = build_internal_validation_plan(
            graph=true_graph,
            sampled_subgraphs={4: [tuple(node_ids)]},
        )
        return refiner, graph, plan, node_index

    return _build


def test_topology_plan_loss_backprops_and_pressures_density_down(
    make_tiny_refiner_and_plan: object,
) -> None:
    from src.topology.finetune_losses import TopologyLossWeights
    from tccig.s2gae import topology_plan_loss

    refiner, graph, plan, node_index = make_tiny_refiner_and_plan(overdense=True)
    weights = TopologyLossWeights(alpha=1.0, beta=8.0, gamma=0.5, delta=0.0)

    loss, components = topology_plan_loss(
        refiner=refiner,
        graph=graph,
        plan=plan,
        node_index=node_index,
        weights=weights,
        include_clustering_mmd=False,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert components["relative_density"] > 0.0
    grads = [p.grad for p in refiner.parameters() if p.grad is not None]
    assert grads, "topology loss did not propagate to refiner parameters"


def test_topology_loss_scale_zero_during_warmup_then_ramps() -> None:
    from src.topology.finetune_losses import TopologyLossWeightSchedule, topology_loss_scale

    schedule = TopologyLossWeightSchedule(warmup_epochs=5, ramp_epochs=10, schedule="linear")
    assert topology_loss_scale(epoch=0, schedule=schedule) == 0.0
    assert topology_loss_scale(epoch=4, schedule=schedule) == 0.0
    assert 0.0 < topology_loss_scale(epoch=9, schedule=schedule) < 1.0
    assert topology_loss_scale(epoch=15, schedule=schedule) == 1.0


def _coverage_graph() -> nx.Graph:
    graph = nx.Graph()
    # two clusters joined by a bridge so some positives need coverage buckets
    graph.add_edges_from(
        [("a", "b"), ("b", "c"), ("a", "c"), ("c", "d"), ("d", "e"), ("e", "f"), ("d", "f")]
    )
    return graph


def test_coverage_augmentation_reaches_full_coverage() -> None:
    graph = _coverage_graph()
    augmented, stats = augment_plan_for_positive_edge_coverage(
        graph=graph,
        base_sampled={4: [("a", "b", "c", "d")]},
        node_sizes=[4],
        strategy="bfs",
        seed=0,
    )
    assert stats["positive_edge_coverage"] == 1.0
    # every positive edge appears in some bucket
    covered: set[frozenset[str]] = set()
    for buckets in augmented.values():
        for nodes in buckets:
            for edge in graph.subgraph(set(nodes)).edges():
                covered.add(frozenset(edge))
    all_positive = {frozenset(edge) for edge in graph.edges()}
    assert covered >= all_positive


def test_coverage_augmentation_is_deterministic() -> None:
    graph = _coverage_graph()
    base = {4: [("a", "b", "c", "d")]}
    first, first_stats = augment_plan_for_positive_edge_coverage(
        graph=graph, base_sampled=base, node_sizes=[4], strategy="bfs", seed=0
    )
    second, second_stats = augment_plan_for_positive_edge_coverage(
        graph=graph, base_sampled=base, node_sizes=[4], strategy="bfs", seed=0
    )
    assert first == second
    assert first_stats == second_stats


def test_coverage_augmentation_no_extra_buckets_when_already_covered() -> None:
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("a", "c")])
    augmented, stats = augment_plan_for_positive_edge_coverage(
        graph=graph, base_sampled={3: [("a", "b", "c")]}, node_sizes=[3], strategy="bfs", seed=0
    )
    assert stats["coverage_bucket_count"] == 0
    assert stats["positive_edge_coverage"] == 1.0


def _fake_runtime(*, is_main_process: bool) -> object:
    accelerator = SimpleNamespace(wait_for_everyone=lambda: None)
    return SimpleNamespace(accelerator=accelerator, is_main_process=is_main_process)


def _plan_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "d"), ("a", "c")])
    return graph


def test_load_or_build_writes_then_reuses_cache(tmp_path: Path) -> None:
    graph = _plan_graph()
    calls = {"n": 0}

    def build_fn() -> tuple[object, dict[str, float | int]]:
        calls["n"] += 1
        plan = build_internal_validation_plan(graph=graph, sampled_subgraphs={3: [("a", "b", "c")]})
        return plan, {
            "base_bucket_count": 1,
            "coverage_bucket_count": 0,
            "positive_edge_coverage": 1.0,
        }

    common = {
        "split": "train_topology",
        "graph": graph,
        "node_sizes": [3],
        "samples_per_size": 1,
        "seed": 0,
        "strategy": "bfs",
        "coverage_augmentation": True,
        "runtime": _fake_runtime(is_main_process=True),
        "cache_dir": tmp_path,
        "build_fn": build_fn,
    }
    plan_first, stats_first = _load_or_build_topology_plan(**common)
    plan_second, stats_second = _load_or_build_topology_plan(**common)

    assert calls["n"] == 1  # second call served from cache, build_fn not re-run
    assert stats_first == stats_second
    assert plan_second.total_pairs == plan_first.total_pairs


def test_load_or_build_non_main_rank_raises_when_cache_missing(tmp_path: Path) -> None:
    graph = _plan_graph()

    def build_fn() -> tuple[object, dict[str, float | int]]:
        raise AssertionError("build_fn must not run on a non-main rank")

    with pytest.raises(RuntimeError, match="topology plan cache was not written"):
        _load_or_build_topology_plan(
            split="train_topology",
            graph=graph,
            node_sizes=[3],
            samples_per_size=1,
            seed=0,
            strategy="bfs",
            coverage_augmentation=True,
            runtime=_fake_runtime(is_main_process=False),
            cache_dir=tmp_path,
            build_fn=build_fn,
        )


def test_build_train_topology_bundle_uses_plan_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _coverage_graph()

    monkeypatch.setattr(tccig_train, "load_split_node_ids", lambda **_: set(graph.nodes()))
    monkeypatch.setattr(tccig_train, "build_pair_supervision_graph", lambda **_: graph)

    sample_calls = {"n": 0}
    real_sample = tccig_train.sample_topology_evaluation_subgraphs

    def counting_sample(**kwargs: object) -> object:
        sample_calls["n"] += 1
        return real_sample(**kwargs)

    monkeypatch.setattr(tccig_train, "sample_topology_evaluation_subgraphs", counting_sample)

    captured: dict[str, object] = {}

    def fake_score_split(
        *,
        split: str,
        pairs: Sequence[object],
        scorer_cfg: object,
        runtime: object,
        cache_dir: object,
    ) -> list[float]:
        captured["pairs"] = pairs
        return [0.5] * len(pairs)

    monkeypatch.setattr(tccig_train, "_score_split", fake_score_split)

    config = {
        "refiner": {
            "topology_training": {
                "enabled": True,
                "node_sizes": [4],
                "samples_per_size": 1,
                "strategy": "bfs",
                "seed": 0,
                "coverage_augmentation": True,
            }
        }
    }
    runtime = _fake_runtime(is_main_process=True)
    common = {
        "config": config,
        "processed_dir": tmp_path,
        "scorer_cfg": {},
        "runtime": runtime,
        "cache_dir": tmp_path,
        "pairwise_input_rule": tccig_train._resolve_refined_output_rule({}),
    }

    bundle_first, plan_first, stats_first = tccig_train._build_train_topology_bundle(**common)
    bundle_second, plan_second, stats_second = tccig_train._build_train_topology_bundle(**common)

    assert sample_calls["n"] == 1  # second run served from cache
    assert plan_first.total_pairs == plan_second.total_pairs
    assert stats_first == stats_second
    assert set(stats_first) == {
        "base_bucket_count",
        "coverage_bucket_count",
        "positive_edge_coverage",
    }


def test_coverage_stats_from_payload_extracts_numeric_keys() -> None:
    payload = {
        "coverage_stats": {
            "base_bucket_count": 3,
            "coverage_bucket_count": 2,
            "positive_edge_coverage": 1.0,
        }
    }
    stats = _coverage_stats_from_payload(payload)
    assert stats == {
        "base_bucket_count": 3,
        "coverage_bucket_count": 2,
        "positive_edge_coverage": 1.0,
    }


def test_coverage_stats_from_payload_missing_key_returns_empty() -> None:
    assert _coverage_stats_from_payload({}) == {}


def test_coverage_stats_from_payload_non_mapping_returns_empty() -> None:
    assert _coverage_stats_from_payload({"coverage_stats": "bad"}) == {}


def test_coverage_stats_from_payload_drops_non_numeric_values() -> None:
    payload = {
        "coverage_stats": {
            "base_bucket_count": 3,
            "positive_edge_coverage": "bad",
            "coverage_bucket_count": None,
        }
    }
    # Non-numeric values must not survive: the log path calls float() on them.
    assert _coverage_stats_from_payload(payload) == {"base_bucket_count": 3}


def test_topology_subset_chunk_loss_uses_inclusion_weights() -> None:
    from src.topology.finetune_losses import TopologyLossWeights
    from tccig.s2gae import topology_subset_chunk_loss
    from tccig.topology_subset import (
        SamplingStratum,
        TopologyPairSample,
        TopologySubgraphEpochChunk,
    )

    chunk = TopologySubgraphEpochChunk(
        subgraph_id="size=3:index=0",
        node_size=3,
        samples=(
            TopologyPairSample(
                pair_id="a||b",
                subgraph_id="size=3:index=0",
                node_size=3,
                protein_a="a",
                protein_b="b",
                local_index_a=0,
                local_index_b=1,
                stratum=SamplingStratum.POSITIVE,
                pi_cand=1.0,
                pi_pool_given_cand=1.0,
                pi_epoch_given_pool=1.0,
                pi_total=1.0,
                target=1.0,
                scorer_probability=0.9,
            ),
            TopologyPairSample(
                pair_id="b||c",
                subgraph_id="size=3:index=0",
                node_size=3,
                protein_a="b",
                protein_b="c",
                local_index_a=1,
                local_index_b=2,
                stratum=SamplingStratum.UNIFORM_NEGATIVE,
                pi_cand=0.5,
                pi_pool_given_cand=1.0,
                pi_epoch_given_pool=1.0,
                pi_total=0.5,
                target=0.0,
                scorer_probability=0.2,
            ),
        ),
    )
    refined_logits = torch.tensor([2.0, -1.0], requires_grad=True)
    loss, components = topology_subset_chunk_loss(
        refined_logits=refined_logits,
        chunk=chunk,
        weights=TopologyLossWeights(alpha=1.0, beta=1.0, gamma=1.0, delta=0.0),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert refined_logits.grad is not None
    assert components["sample_count"] == 2.0


def test_size_balanced_topology_normalizers_ignore_subgraph_imbalance() -> None:
    from tccig.s2gae import _size_balanced_chunk_scales

    scales = _size_balanced_chunk_scales([20, 20, 20, 200])
    assert scales == pytest.approx([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 2.0])


def test_shard_chunks_partition_is_disjoint_and_complete() -> None:
    from tccig.s2gae import _shard_chunks_for_rank

    node_sizes = [20, 20, 20, 200, 200]
    world_size = 2
    seen: list[int] = []
    for rank in range(world_size):
        shard = _shard_chunks_for_rank(
            node_sizes=node_sizes, rank=rank, world_size=world_size
        )
        # Global scale must use GLOBAL counts (S=2 sizes, N_20=3, N_200=2),
        # not the per-rank counts, regardless of which chunks land on this rank.
        for global_index, scale in shard:
            expected_n = 3 if node_sizes[global_index] == 20 else 2
            assert scale == pytest.approx(1.0 / (2 * expected_n))
            seen.append(global_index)
    # Disjoint and complete cover of every chunk index exactly once.
    assert sorted(seen) == list(range(len(node_sizes)))


def test_train_refiner_accepts_subset_plan_object() -> None:
    from tccig.s2gae import TrainRefinerRequest
    from tccig.topology_subset import TopologySubsetPlan

    request = TrainRefinerRequest(
        train=None,  # type: ignore[arg-type]
        validation=None,  # type: ignore[arg-type]
        runtime=None,  # type: ignore[arg-type]
        config={},
        graph_rule=None,  # type: ignore[arg-type]
        train_topology_plan=TopologySubsetPlan(
            subgraphs=(),
            active_sizes=(),
            skipped_sizes={},
            total_positive_pairs=0,
            total_candidate_negatives=0,
            total_pool_negatives=0,
        ),
    )
    assert isinstance(request.train_topology_plan, TopologySubsetPlan)
