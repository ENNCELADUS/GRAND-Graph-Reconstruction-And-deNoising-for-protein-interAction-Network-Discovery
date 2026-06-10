"""Unit tests for the checkpoint-backed TCCIG v3.1 pairwise scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from src.pipeline.stages.train import build_model
from tccig.prepare import CandidatePair, TCCIGRuntime
from tccig.train import score_pairs_with_v3_1


class _FakeAccelerator:
    device = torch.device("cpu")
    is_main_process = True
    use_distributed = False
    process_index = 0
    local_process_index = 0
    num_processes = 1

    def prepare(self, *components: object) -> object:
        return components

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def gather_for_metrics(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def pad_across_processes(
        self,
        value: torch.Tensor,
        dim: int = 0,
        pad_index: int = 0,
        pad_first: bool = False,
    ) -> torch.Tensor:
        del dim, pad_index, pad_first
        return value

    def reduce(self, value: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        del reduction
        return value

    def wait_for_everyone(self) -> None:
        return None

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def save(self, obj: object, f: str | Path, safe_serialization: bool = False) -> None:
        del safe_serialization
        torch.save(obj, f)


def _runtime() -> TCCIGRuntime:
    return TCCIGRuntime(
        accelerator=_FakeAccelerator(),
        device="cpu",
        backend="ddp",
        mixed_precision="no",
        is_distributed=False,
        rank=0,
        local_rank=0,
        world_size=1,
        is_main_process=True,
    )


def _model_config(*, interaction_mode: str = "none") -> dict[str, object]:
    return {
        "model_config": {
            "model": "v3.1",
            "input_dim": 8,
            "d_model": 8,
            "encoder_layers": 1,
            "cross_attn_layers": 1,
            "n_heads": 2,
            "mlp_head": {
                "hidden_dims": [8, 4],
                "dropout": 0.0,
                "activation": "gelu",
                "norm": "layernorm",
            },
            "regularization": {
                "dropout": 0.0,
                "token_dropout": 0.0,
                "cross_attention_dropout": 0.0,
                "stochastic_depth": 0.0,
            },
            "rich_pooling": {"components": ["mean", "attn", "max", "gated"]},
            "pair_readout": {
                "mode": "pair_context_gated",
                "order_aggregation": "abba_max",
            },
            "interaction": {"mode": interaction_mode},
        }
    }


def _write_model_config(path: Path, *, interaction_mode: str = "none") -> None:
    path.write_text(
        yaml.safe_dump(_model_config(interaction_mode=interaction_mode)),
        encoding="utf-8",
    )


def _write_checkpoint(path: Path) -> None:
    model = build_model(_model_config())
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _write_embedding_cache(cache_dir: Path) -> None:
    embeddings = {
        "P1": torch.ones((3, 8), dtype=torch.float32),
        "P2": torch.full((4, 8), 2.0, dtype=torch.float32),
        "P3": torch.full((5, 8), 3.0, dtype=torch.float32),
    }
    index: dict[str, str] = {}
    for protein_id, tensor in embeddings.items():
        relative_path = f"embeddings/{protein_id}.pt"
        output_path = cache_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, output_path)
        index[protein_id] = relative_path
    (cache_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")


def _request_config(tmp_path: Path, checkpoint_path: Path | None = None) -> dict[str, object]:
    model_config_path = tmp_path / "v3_1_abba_no_cross.yaml"
    _write_model_config(model_config_path)
    cache_dir = tmp_path / "cache"
    _write_embedding_cache(cache_dir)
    return {
        "model_config_path": str(model_config_path),
        "checkpoint_path": str(checkpoint_path or tmp_path / "best_model.pth"),
        "embedding_cache_dir": str(cache_dir),
        "batch_size": 2,
        "max_sequence_length": 8,
    }


def test_score_pairs_with_v3_1_returns_one_probability_per_candidate(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best_model.pth"
    _write_checkpoint(checkpoint_path)

    probabilities = score_pairs_with_v3_1(
        pairs=[CandidatePair("P1", "P2"), CandidatePair("P2", "P3")],
        runtime=_runtime(),
        config=_request_config(tmp_path, checkpoint_path),
    )

    assert len(probabilities) == 2
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)


def test_score_pairs_with_v3_1_reports_batch_progress(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best_model.pth"
    _write_checkpoint(checkpoint_path)
    progress_events: list[dict[str, object]] = []
    config = _request_config(tmp_path, checkpoint_path)
    config["batch_size"] = 2

    probabilities = score_pairs_with_v3_1(
        pairs=[
            CandidatePair("P1", "P2"),
            CandidatePair("P2", "P3"),
            CandidatePair("P1", "P3"),
        ],
        runtime=_runtime(),
        config=config,
        progress_callback=lambda event: progress_events.append(dict(event)),
    )

    assert len(probabilities) == 3
    assert progress_events == [
        {"batch_index": 1, "processed_pairs": 2, "local_pair_count": 3},
        {"batch_index": 2, "processed_pairs": 3, "local_pair_count": 3},
    ]


def test_score_pairs_with_v3_1_does_not_filter_self_pairs(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best_model.pth"
    _write_checkpoint(checkpoint_path)

    probabilities = score_pairs_with_v3_1(
        pairs=[CandidatePair("P1", "P1"), CandidatePair("P1", "P2")],
        runtime=_runtime(),
        config=_request_config(tmp_path, checkpoint_path),
    )

    assert len(probabilities) == 2


def test_score_pairs_with_v3_1_requires_existing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint_path does not exist"):
        score_pairs_with_v3_1(
            pairs=[CandidatePair("P1", "P2")],
            runtime=_runtime(),
            config=_request_config(tmp_path),
        )


def test_score_pairs_with_v3_1_rejects_non_abba_no_cross_config(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best_model.pth"
    _write_checkpoint(checkpoint_path)
    config = _request_config(tmp_path, checkpoint_path)
    _write_model_config(Path(str(config["model_config_path"])), interaction_mode="block_self")

    with pytest.raises(ValueError, match="model_config.interaction.mode"):
        score_pairs_with_v3_1(
            pairs=[CandidatePair("P1", "P2")],
            runtime=_runtime(),
            config=config,
        )
