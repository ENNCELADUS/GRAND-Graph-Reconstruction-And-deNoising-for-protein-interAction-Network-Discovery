"""Unit tests for the S2GAE TCCIG refiner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from src.train.config import LossConfig
from src.utils.losses import binary_classification_loss
from tccig.io import CandidatePair
from tccig.s2gae import (
    CrossLayerDecoder,
    S2GAERefiner,
    _build_graph,
    _parse_config,
    apply_gradient_clipping,
    load_mean_pooled_node_features,
    residual_refined_logits,
    s2gae_loss_terms,
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


def test_cross_layer_decoder_includes_final_layer_absolute_difference() -> None:
    decoder = CrossLayerDecoder(
        hidden_dim=2,
        num_layers=1,
        decoder_hidden_dim=4,
        decoder_layers=1,
        dropout=0.0,
    )
    linear = decoder.layers[0]
    assert linear.in_features == 4
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
        linear.weight[0, 3] = 1.0
    hidden_states = [torch.tensor([[1.0, 2.0], [1.0, 5.0]], dtype=torch.float32)]
    pair_index = torch.tensor([[0], [1]], dtype=torch.long)

    delta = decoder(hidden_states=hidden_states, pair_index=pair_index)

    assert delta.item() == pytest.approx(3.0)


def test_residual_refined_logits_adds_delta_to_pairwise_logits() -> None:
    pairwise = torch.tensor([0.8, 0.2], dtype=torch.float32)
    delta = torch.tensor([0.0, 0.5], dtype=torch.float32)

    refined = residual_refined_logits(pairwise, delta)

    assert torch.sigmoid(refined[0]).item() == pytest.approx(0.8)
    assert refined[1].item() == pytest.approx(torch.logit(pairwise[1]).item() + 0.5)


def test_refiner_preserves_pairwise_probability_when_delta_is_zero() -> None:
    model = S2GAERefiner(
        encoder="graphconv",
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
    edge_weight = torch.tensor([0.8, 0.8], dtype=torch.float32)
    pair_index = torch.tensor([[0], [1]], dtype=torch.long)
    pairwise = torch.tensor([0.7], dtype=torch.float32)

    refined_logits, delta = model(
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        pair_index=pair_index,
        pairwise_probabilities=pairwise,
    )

    assert delta.item() == pytest.approx(0.0)
    assert torch.sigmoid(refined_logits).item() == pytest.approx(0.7)


def test_refiner_passes_edge_weight_to_graphconv() -> None:
    class RecordingConv(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.observed_edge_weight: torch.Tensor | None = None

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_weight: torch.Tensor,
        ) -> torch.Tensor:
            del edge_index
            self.observed_edge_weight = edge_weight.detach().clone()
            return x

    model = S2GAERefiner(
        encoder="graphconv",
        input_dim=2,
        hidden_dim=2,
        num_layers=1,
        decoder_hidden_dim=2,
        decoder_layers=1,
        dropout=0.0,
    )
    recorder = RecordingConv()
    model.convs[0] = recorder
    node_features = torch.ones((2, 2), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([0.3, 0.7], dtype=torch.float32)
    pair_index = torch.tensor([[0], [1]], dtype=torch.long)
    pairwise = torch.tensor([0.6], dtype=torch.float32)

    model(
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        pair_index=pair_index,
        pairwise_probabilities=pairwise,
    )

    assert recorder.observed_edge_weight is not None
    assert recorder.observed_edge_weight.tolist() == pytest.approx([0.3, 0.7])


def test_refiner_rejects_non_graphconv_encoder() -> None:
    with pytest.raises(ValueError, match="refiner.encoder must be 'graphconv'"):
        S2GAERefiner(
            encoder="sage",
            input_dim=4,
            hidden_dim=4,
            num_layers=1,
            decoder_hidden_dim=4,
            decoder_layers=1,
            dropout=0.0,
        )


def _base_refiner_config(tmp_path: Path) -> dict[str, object]:
    return {
        "embedding_cache_dir": str(tmp_path / "cache"),
        "optimizer": {
            "type": "adamw",
            "lr": 0.001,
            "weight_decay": 0.01,
            "beta1": 0.8,
            "beta2": 0.9,
            "eps": 1.0e-7,
        },
        "scheduler": {"type": "none"},
        "optimization": {"gradient_clip_norm": 1.0},
    }


def test_parse_config_reads_nested_loss_config(tmp_path: Path) -> None:
    config = _base_refiner_config(tmp_path)
    config.update(
        {
            "loss": {
                "type": "bce_with_logits",
                "pos_weight": 3.0,
                "label_smoothing": 0.1,
            },
            "residual_weight": 0.25,
        }
    )

    cfg = _parse_config(config)

    assert cfg.optimizer.optimizer_type == "adamw"
    assert cfg.optimizer.lr == pytest.approx(0.001)
    assert cfg.optimizer.weight_decay == pytest.approx(0.01)
    assert cfg.optimizer.beta1 == pytest.approx(0.8)
    assert cfg.optimizer.beta2 == pytest.approx(0.9)
    assert cfg.optimizer.eps == pytest.approx(1.0e-7)
    assert cfg.scheduler.scheduler_type == "none"
    assert cfg.optimization.gradient_clip_norm == pytest.approx(1.0)
    assert cfg.loss_config == LossConfig(
        loss_type="bce_with_logits",
        pos_weight=3.0,
        label_smoothing=0.1,
    )
    assert cfg.residual_weight == pytest.approx(0.25)


def test_parse_config_rejects_unsupported_encoder(tmp_path: Path) -> None:
    config = _base_refiner_config(tmp_path)
    config["encoder"] = "sage"

    with pytest.raises(ValueError, match="refiner.encoder must be 'graphconv'"):
        _parse_config(config)


def test_parse_config_rejects_removed_learning_rate(tmp_path: Path) -> None:
    config = _base_refiner_config(tmp_path)
    config["learning_rate"] = 0.001

    with pytest.raises(ValueError, match="refiner.learning_rate is no longer supported"):
        _parse_config(config)


def test_parse_config_rejects_unsupported_optimizer(tmp_path: Path) -> None:
    config = _base_refiner_config(tmp_path)
    optimizer = dict(config["optimizer"])
    optimizer["type"] = "adam"
    config["optimizer"] = optimizer

    with pytest.raises(ValueError, match="refiner.optimizer.type must be 'adamw'"):
        _parse_config(config)


def test_parse_config_rejects_unsupported_scheduler(tmp_path: Path) -> None:
    config = _base_refiner_config(tmp_path)
    config["scheduler"] = {"type": "onecycle"}

    with pytest.raises(ValueError, match="refiner.scheduler.type must be 'none'"):
        _parse_config(config)


def test_parse_config_allows_disabled_gradient_clipping(tmp_path: Path) -> None:
    cfg = _parse_config(
        {
            **_base_refiner_config(tmp_path),
            "optimization": {"gradient_clip_norm": 0.0},
        }
    )

    assert cfg.optimization.gradient_clip_norm is None


def test_apply_gradient_clipping_returns_observed_norm_when_disabled() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]], dtype=torch.float32)

    observed_norm = apply_gradient_clipping(model=model, gradient_clip_norm=None)

    assert observed_norm == pytest.approx(5.0)
    assert model.weight.grad.tolist() == [[3.0, 4.0]]


def test_apply_gradient_clipping_clips_to_configured_norm() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.grad = torch.tensor([[3.0, 4.0]], dtype=torch.float32)

    observed_norm = apply_gradient_clipping(model=model, gradient_clip_norm=1.0)

    assert observed_norm == pytest.approx(5.0)
    assert model.weight.grad.norm().item() == pytest.approx(1.0)


def test_parse_config_requires_optimizer_block(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="refiner.optimizer must be a mapping"):
        _parse_config(
            {
                "embedding_cache_dir": str(tmp_path / "cache"),
                "loss": {
                    "type": "bce_with_logits",
                    "pos_weight": 3.0,
                    "label_smoothing": 0.1,
                },
                "residual_weight": 0.25,
            }
        )


def test_s2gae_loss_terms_use_weighted_bce_and_all_pair_residual_anchor() -> None:
    refined_logits = torch.tensor([0.4, -0.2], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0], dtype=torch.float32)
    delta = torch.tensor([1.0, 3.0], dtype=torch.float32)
    loss_config = LossConfig(
        loss_type="bce_with_logits",
        pos_weight=2.0,
        label_smoothing=0.2,
    )

    terms = s2gae_loss_terms(
        refined_logits=refined_logits,
        labels=labels,
        delta_logits=delta,
        loss_config=loss_config,
        residual_weight=0.25,
    )

    expected_bce = binary_classification_loss(
        logits=refined_logits,
        labels=labels,
        loss_config=loss_config,
    )
    assert terms.bce.item() == pytest.approx(expected_bce.item())
    assert terms.residual_anchor.item() == pytest.approx(5.0)
    assert terms.weighted_residual_anchor.item() == pytest.approx(1.25)
    assert terms.total.item() == pytest.approx(expected_bce.item() + 1.25)


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


def test_build_graph_uses_weighted_bidirectional_edges_without_self_loops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_load_mean_pooled_node_features(**kwargs: object) -> torch.Tensor:
        protein_ids = kwargs["protein_ids"]
        assert isinstance(protein_ids, list)
        return torch.zeros((len(protein_ids), 4), dtype=torch.float32)

    monkeypatch.setattr(
        "tccig.s2gae.load_mean_pooled_node_features",
        fake_load_mean_pooled_node_features,
    )
    cfg = type(
        "Cfg",
        (),
        {
            "embedding_cache_dir": tmp_path / "cache",
            "embedding_index_path": tmp_path / "cache" / "index.json",
            "input_dim": 4,
            "max_sequence_length": 8,
        },
    )()
    graph = _build_graph(
        pairs=[
            CandidatePair("A", "B"),
            CandidatePair("B", "A"),
            CandidatePair("A", "C"),
        ],
        pairwise_probabilities=[0.2, 0.8, 0.4],
        pairwise_graph_edges=[("A", "B"), ("A", "C")],
        cfg=cfg,
        device=torch.device("cpu"),
    )

    assert graph.edge_index.t().tolist() == [[0, 1], [1, 0], [0, 2], [2, 0]]
    assert graph.edge_weight.tolist() == pytest.approx([0.8, 0.8, 0.4, 0.4])
    assert all(src != dst for src, dst in graph.edge_index.t().tolist())
