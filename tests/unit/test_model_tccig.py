"""Unit tests for the TCCIG graph-generator model."""

from __future__ import annotations

import inspect
import math

import pytest
import torch
from src.model.tccig import TCCIG
from src.pipeline.stages.train import build_model


def _tccig_config() -> dict[str, object]:
    return {
        "model": "tccig",
        "input_dim": 8,
        "d_model": 16,
        "dropout": 0.0,
        "lowrank_dim": 4,
        "num_modules": 3,
        "self_refinement_rounds": 0,
        "candidate_proposer": {"type": "all_pairs"},
        "pair_mlp": {"hidden_dims": [16]},
        "retrieval": {
            "dim": 8,
            "rff_features": 16,
            "rff_input_dim": 16,
            "rff_backend": "sorf",
            "rff_sigma": 0.5,
            "top_k": 2,
            "logit_gate_init": 0.0,
        },
    }


def test_tccig_encodes_query_key_struct_degree_and_residue_factors() -> None:
    torch.manual_seed(41)
    model = TCCIG(**_tccig_config())
    embeddings = torch.randn(4, 5, 8)
    lengths = torch.tensor([5, 4, 3, 2], dtype=torch.long)

    encoded = model.encode_proteins(
        protein_embeddings=embeddings,
        protein_lengths=lengths,
    )

    assert encoded["node"].shape == (4, 16)
    assert encoded["query"].shape == (4, 8)
    assert encoded["key"].shape == (4, 8)
    assert encoded["struct"].shape == (4, 8)
    assert encoded["residue"].shape == (4, 16)
    assert encoded["module"].shape == (4, 3)
    assert encoded["degree"].shape == (4,)
    assert torch.all(encoded["degree"] >= 0.0)


def test_tccig_sorf_residue_factorization_is_deterministic() -> None:
    torch.manual_seed(43)
    model = TCCIG(**_tccig_config())
    embeddings = torch.randn(3, 4, 8)
    lengths = torch.tensor([4, 3, 2], dtype=torch.long)

    first = model.encode_proteins(
        protein_embeddings=embeddings,
        protein_lengths=lengths,
    )["residue"]
    second = model.encode_proteins(
        protein_embeddings=embeddings,
        protein_lengths=lengths,
    )["residue"]

    assert torch.allclose(first, second)
    assert torch.isfinite(first).all()


def test_tccig_exact_retrieval_excludes_self_edges_and_symmetrizes_pairs() -> None:
    torch.manual_seed(47)
    model = TCCIG(**_tccig_config())
    embeddings = torch.randn(5, 4, 8)

    output = model.forward_graph(protein_embeddings=embeddings)
    retrieved_pairs = model.retrieve_candidate_pairs(
        encoded=output["encoded"],
        top_k=2,
    )

    assert retrieved_pairs.shape[0] == 2
    assert retrieved_pairs.size(1) > 0
    assert not torch.any(retrieved_pairs[0] == retrieved_pairs[1])
    assert torch.all(retrieved_pairs[0] < retrieved_pairs[1])
    assert retrieved_pairs.unique(dim=1).size(1) == retrieved_pairs.size(1)


def test_tccig_decode_graph_candidates_accepts_encoded_retrieval_state() -> None:
    torch.manual_seed(53)
    model = TCCIG(**_tccig_config())
    embeddings = torch.randn(4, 5, 8)
    encoded = model.encode_proteins(protein_embeddings=embeddings)
    node_embeddings = encoded["node"]
    candidate_pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    output = model.decode_graph_candidates(
        node_embeddings=node_embeddings,
        candidate_pairs=candidate_pairs,
        encoded=encoded,
    )

    assert "encoded" in inspect.signature(model.decode_graph_candidates).parameters
    assert torch.allclose(
        output["retrieval_score"],
        model.retrieval_score_matrix(encoded)[candidate_pairs[0], candidate_pairs[1]],
    )


def test_tccig_encode_graph_nodes_does_not_compute_residue_retrieval_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(57)
    model = TCCIG(**_tccig_config())

    def _raise_if_residue_factor_is_used(**_: object) -> torch.Tensor:
        raise AssertionError("encode_graph_nodes should not compute residue retrieval factors")

    monkeypatch.setattr(model, "_encode_residue_factor", _raise_if_residue_factor_is_used)

    node_embeddings = model.encode_graph_nodes(
        protein_embeddings=torch.randn(3, 5, 8),
        protein_lengths=torch.tensor([5, 4, 3], dtype=torch.long),
    )

    assert node_embeddings.shape == (3, 16)


def test_tccig_pooled_input_retrieval_only_matches_esm_cosine_baseline() -> None:
    config = _tccig_config() | {
        "decoder_mode": "retrieval_only",
        "decoder_structural_gate_init": 0.0,
        "retrieval": {
            "dim": 8,
            "rff_features": 8,
            "rff_input_dim": 8,
            "rff_backend": "dense",
            "rff_sigma": 0.5,
            "feature_source": "pooled_input",
            "normalize": True,
            "top_k": 2,
            "logit_gate_init": 1.0,
        },
    }
    model = TCCIG(**config)
    embeddings = torch.randn(3, 4, 8)
    pooled = embeddings.mean(dim=1)
    normalized = torch.nn.functional.normalize(pooled, dim=-1)

    encoded = model.encode_proteins(protein_embeddings=embeddings)
    scores = model.retrieval_score_matrix(encoded)

    expected = 2.0 * (normalized @ normalized.t()) / math.sqrt(8.0)
    assert torch.allclose(scores, expected, atol=1.0e-6)


def test_tccig_lowrank_score_is_scaled_by_dimension() -> None:
    config = _tccig_config() | {"decoder_structural_gate_init": 1.0}
    model = TCCIG(**config)
    with torch.no_grad():
        for module in (model.pair_mlp, model.hub_head, model.module_head, model.density_bias_head):
            for parameter in module.parameters():
                parameter.zero_()
        model.lowrank_head.weight.zero_()
        model.lowrank_head.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    output = model.decode_graph_candidates(
        node_embeddings=torch.zeros(2, 16),
        candidate_pairs=torch.tensor([[0], [1]], dtype=torch.long),
    )

    assert output["lowrank_score"].item() == pytest.approx(15.0)
    assert output["logits"].item() == pytest.approx(15.0)


def test_tccig_structural_scores_are_gated_at_initialization() -> None:
    config = _tccig_config() | {
        "lowrank_dim": 1,
        "num_modules": 2,
        "decoder_structural_gate_init": 0.25,
    }
    model = TCCIG(**config)
    with torch.no_grad():
        for module in (model.pair_mlp, model.density_bias_head):
            for parameter in module.parameters():
                parameter.zero_()
        model.hub_head.weight.zero_()
        model.hub_head.bias.fill_(2.0)
        model.lowrank_head.weight.zero_()
        model.lowrank_head.bias.fill_(3.0)
        model.module_head.weight.zero_()
        model.module_head.bias.zero_()
        model.module_interactions.copy_(torch.eye(2) * 5.0)

    output = model.decode_graph_candidates(
        node_embeddings=torch.zeros(2, 16),
        candidate_pairs=torch.tensor([[0], [1]], dtype=torch.long),
    )

    assert output["hub_score"].item() == pytest.approx(1.0)
    assert output["lowrank_score"].item() == pytest.approx(2.25)
    assert output["module_score"].item() == pytest.approx(0.5)
    assert output["logits"].item() == pytest.approx(3.75)


def test_tccig_forward_graph_is_feature_only_and_returns_graph_outputs() -> None:
    model = TCCIG(**_tccig_config())
    embeddings = torch.randn(4, 5, 8)
    lengths = torch.tensor([5, 4, 3, 2], dtype=torch.long)

    signature = inspect.signature(model.forward_graph)
    forbidden = {"target_adjacency", "adjacency", "degrees", "communities", "laplacian"}
    assert forbidden.isdisjoint(signature.parameters)

    output = model.forward_graph(protein_embeddings=embeddings, protein_lengths=lengths)

    assert output["logits"].shape == (6,)
    assert output["candidate_pairs"].shape == (2, 6)
    assert output["edge_probabilities"].shape == (6,)
    assert output["m_hat"].shape == ()
    assert float(output["m_hat"].detach()) >= 0.0
    assert output["module_memberships"].shape == (4, 3)
    assert output["node_embeddings"].shape == (4, 16)


def test_tccig_edge_budget_stays_finite_for_large_candidate_universe_in_half_precision() -> None:
    model = TCCIG(**_tccig_config()).half()
    node_embeddings = torch.randn(16, 16, dtype=torch.float16)
    candidate_count = 2_033_136

    edge_budget = model.edge_budget_from_node_embeddings(
        node_embeddings=node_embeddings,
        candidate_count=candidate_count,
    )

    assert torch.isfinite(edge_budget)
    assert 0.0 <= float(edge_budget.detach()) <= float(candidate_count)


def test_tccig_decode_graph_candidates_treats_pairs_as_undirected_edges() -> None:
    torch.manual_seed(17)
    model = TCCIG(**_tccig_config())
    model.eval()
    node_embeddings = torch.randn(4, 16)

    with torch.no_grad():
        forward_output = model.decode_graph_candidates(
            node_embeddings=node_embeddings,
            candidate_pairs=torch.tensor([[0], [2]], dtype=torch.long),
        )
        reversed_output = model.decode_graph_candidates(
            node_embeddings=node_embeddings,
            candidate_pairs=torch.tensor([[2], [0]], dtype=torch.long),
        )

    assert torch.equal(
        forward_output["candidate_pairs"],
        reversed_output["candidate_pairs"],
    )
    assert torch.allclose(forward_output["logits"], reversed_output["logits"])


def test_tccig_forward_graph_is_equivariant_to_protein_set_order() -> None:
    torch.manual_seed(23)
    model = TCCIG(**_tccig_config())
    model.eval()
    embeddings = torch.randn(3, 5, 8)
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)

    with torch.no_grad():
        output = model.forward_graph(protein_embeddings=embeddings)
        permuted_output = model.forward_graph(protein_embeddings=embeddings[permutation])

    def probabilities_by_edge(
        *,
        candidate_pairs: torch.Tensor,
        probabilities: torch.Tensor,
        node_ids: torch.Tensor,
    ) -> dict[frozenset[int], torch.Tensor]:
        return {
            frozenset(
                (
                    int(node_ids[int(source_index)]),
                    int(node_ids[int(target_index)]),
                )
            ): probability
            for (source_index, target_index), probability in zip(
                candidate_pairs.t().tolist(),
                probabilities,
                strict=True,
            )
        }

    original = probabilities_by_edge(
        candidate_pairs=output["candidate_pairs"],
        probabilities=output["edge_probabilities"],
        node_ids=torch.arange(3),
    )
    permuted = probabilities_by_edge(
        candidate_pairs=permuted_output["candidate_pairs"],
        probabilities=permuted_output["edge_probabilities"],
        node_ids=permutation,
    )

    assert original.keys() == permuted.keys()
    for edge, probability in original.items():
        assert torch.allclose(probability, permuted[edge])


def test_tccig_pairwise_forward_is_compatible_with_existing_pipeline() -> None:
    model = build_model({"model_config": _tccig_config()})
    batch_size = 3

    output = model(
        emb_a=torch.randn(batch_size, 5, 8),
        emb_b=torch.randn(batch_size, 4, 8),
        len_a=torch.tensor([5, 4, 3], dtype=torch.long),
        len_b=torch.tensor([4, 3, 2], dtype=torch.long),
        label=torch.tensor([1.0, 0.0, 1.0]),
    )

    assert output["logits"].shape == (batch_size,)
    assert output["loss"].shape == ()


def test_tccig_pairwise_forward_is_invariant_to_batch_neighbors() -> None:
    torch.manual_seed(13)
    model = TCCIG(**_tccig_config())
    model.eval()
    emb_a = torch.randn(2, 5, 8)
    emb_b = torch.randn(2, 4, 8)
    len_a = torch.tensor([5, 4], dtype=torch.long)
    len_b = torch.tensor([4, 3], dtype=torch.long)

    with torch.no_grad():
        batched_logit = model(
            emb_a=emb_a,
            emb_b=emb_b,
            len_a=len_a,
            len_b=len_b,
        )["logits"][0]
        single_logit = model(
            emb_a=emb_a[:1],
            emb_b=emb_b[:1],
            len_a=len_a[:1],
            len_b=len_b[:1],
        )["logits"][0]

    assert torch.allclose(batched_logit, single_logit)


def test_tccig_module_score_is_centered_at_neutral_membership() -> None:
    model = TCCIG(**_tccig_config())
    with torch.no_grad():
        model.module_head.weight.zero_()
        model.module_head.bias.zero_()
    node_embeddings = torch.randn(4, 16)
    candidate_pairs = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    output = model.decode_graph_candidates(
        node_embeddings=node_embeddings,
        candidate_pairs=candidate_pairs,
    )

    assert torch.allclose(
        output["module_memberships"].sum(dim=-1),
        torch.ones(4),
    )
    assert torch.allclose(
        output["module_score"],
        torch.zeros(3),
        atol=1.0e-6,
    )


def test_tccig_density_bias_can_initialize_sparse_probability_prior() -> None:
    model = TCCIG(**_tccig_config())
    prior_probability = 0.04
    bias = model.initialize_density_bias_with_prior(prior_probability)
    with torch.no_grad():
        for module in (model.pair_mlp, model.hub_head, model.lowrank_head, model.module_head):
            for parameter in module.parameters():
                parameter.zero_()

    output = model.forward_graph(protein_embeddings=torch.randn(4, 5, 8))

    assert bias == math.log(prior_probability / (1.0 - prior_probability))
    assert output["edge_probabilities"].mean().item() == pytest.approx(
        prior_probability,
        abs=1.0e-6,
    )


def test_tccig_graph_and_pairwise_paths_share_centered_module_term() -> None:
    config = _tccig_config() | {"decoder_structural_gate_init": 0.25}
    model = TCCIG(**config)
    with torch.no_grad():
        for module in (model.pair_mlp, model.density_bias_head):
            for parameter in module.parameters():
                parameter.zero_()
        model.hub_head.weight.fill_(0.1)
        model.hub_head.bias.fill_(0.2)
        model.lowrank_head.weight.fill_(0.1)
        model.lowrank_head.bias.fill_(0.3)
    node_embeddings = torch.randn(3, 16)
    candidate_pairs = torch.tensor([[0], [2]], dtype=torch.long)

    graph_output = model.decode_graph_candidates(
        node_embeddings=node_embeddings,
        candidate_pairs=candidate_pairs,
    )
    pairwise_logit = model._decode_pairwise_edges(
        node_a_embeddings=node_embeddings[:1],
        node_b_embeddings=node_embeddings[2:3],
    )

    assert torch.allclose(graph_output["logits"], pairwise_logit)
