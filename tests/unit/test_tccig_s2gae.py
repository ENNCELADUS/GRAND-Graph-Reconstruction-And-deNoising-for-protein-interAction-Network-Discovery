"""Unit tests for the S2GAE TCCIG refiner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tccig.s2gae import (
    CrossLayerDecoder,
    S2GAERefiner,
    load_mean_pooled_node_features,
    residual_refined_logits,
)


def test_cross_layer_decoder_returns_one_finite_delta_per_pair() -> None:
    decoder = CrossLayerDecoder(
        hidden_dim=4,
        num_layers=2,
        decoder_hidden_dim=8,
        decoder_layers=2,
        dropout=0.0,
    )
    hidden_states = [
        torch.ones((3, 4), dtype=torch.float32),
        torch.full((3, 4), 2.0, dtype=torch.float32),
    ]
    pair_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    deltas = decoder(hidden_states=hidden_states, pair_index=pair_index)

    assert deltas.shape == (2,)
    assert torch.isfinite(deltas).all()


def test_residual_refined_logits_adds_delta_to_pairwise_logits() -> None:
    pairwise = torch.tensor([0.8, 0.2], dtype=torch.float32)
    delta = torch.tensor([0.0, 0.5], dtype=torch.float32)

    refined = residual_refined_logits(pairwise, delta)

    assert torch.sigmoid(refined[0]).item() == pytest.approx(0.8)
    assert refined[1].item() == pytest.approx(torch.logit(pairwise[1]).item() + 0.5)


def test_refiner_preserves_pairwise_probability_when_delta_is_zero() -> None:
    model = S2GAERefiner(
        encoder="sage",
        input_dim=4,
        hidden_dim=4,
        num_layers=1,
        decoder_hidden_dim=4,
        decoder_layers=1,
        dropout=0.0,
    )
    for parameter in model.decoder.parameters():
        torch.nn.init.zeros_(parameter)
    node_features = torch.ones((2, 4), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    pair_index = torch.tensor([[0], [1]], dtype=torch.long)
    pairwise = torch.tensor([0.7], dtype=torch.float32)

    refined_logits, delta = model(
        node_features=node_features,
        edge_index=edge_index,
        pair_index=pair_index,
        pairwise_probabilities=pairwise,
    )

    assert delta.item() == pytest.approx(0.0)
    assert torch.sigmoid(refined_logits).item() == pytest.approx(0.7)


def test_mean_pooled_features_require_embedding_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="embedding_index_path"):
        load_mean_pooled_node_features(
            protein_ids=["P1"],
            cache_dir=tmp_path / "cache",
            index_path=tmp_path / "cache" / "index.json",
            input_dim=4,
            max_sequence_length=8,
            device=torch.device("cpu"),
        )


def test_mean_pooled_features_reject_missing_protein_id(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "index.json").write_text(json.dumps({}), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing from embedding index"):
        load_mean_pooled_node_features(
            protein_ids=["P1"],
            cache_dir=cache_dir,
            index_path=cache_dir / "index.json",
            input_dim=4,
            max_sequence_length=8,
            device=torch.device("cpu"),
        )
