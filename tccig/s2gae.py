"""S2GAE-style residual denoiser for the standalone TCCIG pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score
from src.embed import load_cached_embedding
from src.train.config import LossConfig
from src.utils.losses import binary_classification_loss
from torch import nn
from torch.optim import Optimizer

from tccig.io import CandidatePair, canonical_edge, write_json

if TYPE_CHECKING:
    from tccig.train import RefineRequest, SplitBundle, TrainRefinerRequest


@dataclass(frozen=True)
class S2GAEConfig:
    """Parsed S2GAE refiner configuration."""

    encoder: str
    input_dim: int
    hidden_dim: int
    num_layers: int
    decoder_hidden_dim: int
    decoder_layers: int
    dropout: float
    epochs: int
    batch_size: int
    loss_config: LossConfig
    residual_weight: float
    optimizer: S2GAEOptimizerConfig
    scheduler: S2GAESchedulerConfig
    optimization: S2GAEOptimizationConfig
    embedding_cache_dir: Path
    embedding_index_path: Path
    max_sequence_length: int | None
    checkpoint_path: Path
    log_dir: Path


@dataclass
class S2GAERefinerState:
    """In-memory and on-disk state for a trained S2GAE refiner."""

    model: S2GAERefiner
    config: S2GAEConfig
    best_validation_auprc: float
    epochs_trained: int


@dataclass(frozen=True)
class S2GAEOptimizerConfig:
    """Optimizer hyperparameters for the S2GAE refiner."""

    optimizer_type: str
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float


@dataclass(frozen=True)
class S2GAESchedulerConfig:
    """Scheduler configuration for the S2GAE refiner."""

    scheduler_type: str


@dataclass(frozen=True)
class S2GAEOptimizationConfig:
    """Backward and optimization-loop controls."""

    gradient_clip_norm: float | None


@dataclass(frozen=True)
class S2GAELossTerms:
    """S2GAE training loss components."""

    bce: torch.Tensor
    residual_anchor: torch.Tensor
    weighted_residual_anchor: torch.Tensor
    total: torch.Tensor


class S2GAERefiner(nn.Module):
    """PyG GNN encoder plus official S2GAE-style cross-layer decoder."""

    def __init__(
        self,
        *,
        encoder: str,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        decoder_hidden_dim: int,
        decoder_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.dropout = dropout
        self.encoder_name = encoder.lower()
        self.convs = nn.ModuleList()
        sage_conv, gcn_conv = _load_pyg_convs()
        for layer_index in range(num_layers):
            in_channels = input_dim if layer_index == 0 else hidden_dim
            if self.encoder_name == "sage":
                self.convs.append(sage_conv(in_channels, hidden_dim))
            elif self.encoder_name == "gcn":
                self.convs.append(
                    gcn_conv(
                        in_channels,
                        hidden_dim,
                        cached=False,
                        add_self_loops=False,
                    )
                )
            else:
                raise ValueError("refiner.encoder must be 'sage' or 'gcn'")
        self.decoder = CrossLayerDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_layers=decoder_layers,
            dropout=dropout,
        )

    def encode(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> list[torch.Tensor]:
        """Return one hidden representation per GNN layer."""
        hidden_states: list[torch.Tensor] = []
        x = node_features
        for layer_index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if layer_index < len(self.convs) - 1:
                x = functional.relu(x)
                x = functional.dropout(x, p=self.dropout, training=self.training)
            hidden_states.append(x)
        return hidden_states

    def forward(
        self,
        *,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        pair_index: torch.Tensor,
        pairwise_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return refined logits and residual deltas for candidate pairs."""
        hidden_states = self.encode(node_features=node_features, edge_index=edge_index)
        delta = self.decoder(hidden_states=hidden_states, pair_index=pair_index)
        return residual_refined_logits(pairwise_probabilities, delta), delta


class CrossLayerDecoder(nn.Module):
    """S2GAE link decoder using all source/destination layer products."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        decoder_hidden_dim: int,
        decoder_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if decoder_layers <= 0:
            raise ValueError("decoder_layers must be positive")
        self.dropout = dropout
        input_dim = hidden_dim * num_layers * num_layers
        layers: list[nn.Linear] = []
        if decoder_layers == 1:
            layers.append(nn.Linear(input_dim, 1))
        else:
            layers.append(nn.Linear(input_dim, decoder_hidden_dim))
            for _ in range(decoder_layers - 2):
                layers.append(nn.Linear(decoder_hidden_dim, decoder_hidden_dim))
            layers.append(nn.Linear(decoder_hidden_dim, 1))
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        *,
        hidden_states: Sequence[torch.Tensor],
        pair_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode candidate-pair residual logits from cross-layer products."""
        src_index = pair_index[0]
        dst_index = pair_index[1]
        products: list[torch.Tensor] = []
        for src_hidden in hidden_states:
            src_values = src_hidden[src_index]
            for dst_hidden in hidden_states:
                products.append(src_values * dst_hidden[dst_index])
        x = torch.cat(products, dim=1)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = functional.relu(x)
            x = functional.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x).squeeze(-1)


def residual_refined_logits(
    pairwise_probabilities: torch.Tensor,
    delta_logits: torch.Tensor,
) -> torch.Tensor:
    """Add S2GAE residual logits to clamped pairwise logits."""
    clamped = pairwise_probabilities.clamp(min=1.0e-6, max=1.0 - 1.0e-6)
    return torch.logit(clamped) + delta_logits


def s2gae_loss_terms(
    *,
    refined_logits: torch.Tensor,
    labels: torch.Tensor,
    delta_logits: torch.Tensor,
    loss_config: LossConfig,
    residual_weight: float,
) -> S2GAELossTerms:
    """Compute supervised denoising BCE plus all-pair residual anchor."""
    bce = binary_classification_loss(
        logits=refined_logits,
        labels=labels,
        loss_config=loss_config,
    )
    residual_anchor = delta_logits.pow(2).mean()
    weighted_residual_anchor = residual_weight * residual_anchor
    return S2GAELossTerms(
        bce=bce,
        residual_anchor=residual_anchor,
        weighted_residual_anchor=weighted_residual_anchor,
        total=bce + weighted_residual_anchor,
    )


def build_s2gae_optimizer(
    *,
    model: nn.Module,
    config: S2GAEOptimizerConfig,
) -> Optimizer:
    """Build the configured optimizer over refiner parameters only."""
    if config.optimizer_type != "adamw":
        raise ValueError("refiner.optimizer.type must be 'adamw'")
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )


def apply_gradient_clipping(
    *,
    model: nn.Module,
    gradient_clip_norm: float | None,
) -> float:
    """Clip gradients when configured and return the observed norm."""
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        return 0.0
    if gradient_clip_norm is None:
        return _gradient_norm(parameters)
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
    return float(total_norm.detach().cpu().item())


def load_mean_pooled_node_features(
    *,
    protein_ids: Sequence[str],
    cache_dir: Path,
    index_path: Path,
    input_dim: int,
    max_sequence_length: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Load cached ESM3 tensors and mean-pool them into node features."""
    embedding_index = _load_embedding_index(index_path)
    features = [
        load_cached_embedding(
            cache_dir=cache_dir,
            index=embedding_index,
            protein_id=protein_id,
            expected_input_dim=input_dim,
            max_sequence_length=max_sequence_length,
        ).mean(dim=0)
        for protein_id in protein_ids
    ]
    return torch.stack(features, dim=0).to(device)


def train_refiner(request: TrainRefinerRequest) -> S2GAERefinerState:
    """Train an S2GAE residual denoiser on pairwise-generated train graphs."""
    cfg = _parse_config(request.config)
    device = torch.device(request.runtime.device)
    model = S2GAERefiner(
        encoder=cfg.encoder,
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        decoder_layers=cfg.decoder_layers,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = build_s2gae_optimizer(model=model, config=cfg.optimizer)
    prepared = request.runtime.accelerator.prepare(model, optimizer)
    model, optimizer = _prepared_model_and_optimizer(prepared)

    train_graph = _build_split_graph(request.train, cfg=cfg, device=device)
    validation_graph = _build_split_graph(request.validation, cfg=cfg, device=device)
    train_labels = _required_float_tensor(
        request.train.loss_targets,
        "request.train.loss_targets",
        device=device,
    )
    validation_labels = _required_float_tensor(
        request.validation.candidate_labels,
        "request.validation.candidate_labels",
        device=device,
    )

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_auprc = -math.inf
    history: list[dict[str, float | int]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_bce_loss = 0.0
        total_residual_anchor_loss = 0.0
        total_weighted_residual_anchor_loss = 0.0
        total_gradient_norm = 0.0
        total_examples = 0
        for batch_indices in _batch_indices(len(request.train.pairs), cfg.batch_size, device):
            optimizer.zero_grad(set_to_none=True)
            refined_logits, delta = model(
                node_features=train_graph.node_features,
                edge_index=train_graph.edge_index,
                pair_index=train_graph.pair_index[:, batch_indices],
                pairwise_probabilities=train_graph.pairwise_probabilities[batch_indices],
            )
            labels = train_labels[batch_indices]
            loss_terms = s2gae_loss_terms(
                refined_logits=refined_logits,
                labels=labels,
                delta_logits=delta,
                loss_config=cfg.loss_config,
                residual_weight=cfg.residual_weight,
            )
            request.runtime.accelerator.backward(loss_terms.total)
            gradient_norm = apply_gradient_clipping(
                model=model,
                gradient_clip_norm=cfg.optimization.gradient_clip_norm,
            )
            optimizer.step()
            batch_count = int(batch_indices.numel())
            total_loss += float(loss_terms.total.detach().item()) * batch_count
            total_bce_loss += float(loss_terms.bce.detach().item()) * batch_count
            total_residual_anchor_loss += (
                float(loss_terms.residual_anchor.detach().item()) * batch_count
            )
            total_weighted_residual_anchor_loss += (
                float(loss_terms.weighted_residual_anchor.detach().item()) * batch_count
            )
            total_gradient_norm += gradient_norm * batch_count
            total_examples += batch_count

        validation_auprc = _validation_auprc(
            model=model,
            graph=validation_graph,
            labels=validation_labels,
            batch_size=cfg.batch_size,
        )
        epoch_denominator = max(1, total_examples)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / epoch_denominator,
                "train_bce_loss": total_bce_loss / epoch_denominator,
                "train_residual_anchor_loss": total_residual_anchor_loss / epoch_denominator,
                "train_weighted_residual_anchor_loss": (
                    total_weighted_residual_anchor_loss / epoch_denominator
                ),
                "train_gradient_norm": total_gradient_norm / epoch_denominator,
                "learning_rate": _current_learning_rate(optimizer),
                "val_auprc": validation_auprc,
            }
        )
        if validation_auprc > best_validation_auprc:
            best_validation_auprc = validation_auprc
            checkpoint_model = _unwrap_model(model, request.runtime.accelerator)
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in checkpoint_model.state_dict().items()
            }

    if best_state_dict is None:
        raise RuntimeError("S2GAE training did not produce a checkpoint")
    checkpoint_model = _unwrap_model(model, request.runtime.accelerator)
    checkpoint_model.load_state_dict(best_state_dict)
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "config": _config_to_json(cfg),
            "best_validation_auprc": best_validation_auprc,
        },
        cfg.checkpoint_path,
    )
    write_json(
        cfg.log_dir / "training_summary.json",
        {
            "monitor_metric": "val_auprc",
            "best_validation_auprc": best_validation_auprc,
            "epochs_trained": cfg.epochs,
            "checkpoint_path": str(cfg.checkpoint_path),
            "optimizer": _optimizer_config_to_json(cfg.optimizer),
            "scheduler": {"type": cfg.scheduler.scheduler_type},
            "optimization": _optimization_config_to_json(cfg.optimization),
            "current_learning_rate": _current_learning_rate(optimizer),
            "history": history,
        },
    )
    return S2GAERefinerState(
        model=cast(S2GAERefiner, checkpoint_model),
        config=cfg,
        best_validation_auprc=best_validation_auprc,
        epochs_trained=cfg.epochs,
    )


@torch.no_grad()
def predict_refined(request: RefineRequest) -> list[float]:
    """Predict refined probabilities for one split with a trained S2GAE state."""
    if not isinstance(request.refiner_state, S2GAERefinerState):
        raise TypeError("request.refiner_state must be an S2GAERefinerState")
    state = request.refiner_state
    device = torch.device(request.runtime.device)
    state.model.to(device)
    state.model.eval()
    graph = _build_prediction_graph(request, cfg=state.config, device=device)
    probabilities: list[float] = []
    for batch_indices in _batch_indices(len(request.pairs), state.config.batch_size, device):
        refined_logits, _ = state.model(
            node_features=graph.node_features,
            edge_index=graph.edge_index,
            pair_index=graph.pair_index[:, batch_indices],
            pairwise_probabilities=graph.pairwise_probabilities[batch_indices],
        )
        probabilities.extend(torch.sigmoid(refined_logits).detach().cpu().tolist())
    return [float(probability) for probability in probabilities]


@dataclass(frozen=True)
class _SplitGraph:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    pair_index: torch.Tensor
    pairwise_probabilities: torch.Tensor


def _build_split_graph(
    bundle: SplitBundle,
    *,
    cfg: S2GAEConfig,
    device: torch.device,
) -> _SplitGraph:
    return _build_graph(
        pairs=bundle.pairs,
        pairwise_probabilities=bundle.pairwise_probabilities,
        pairwise_graph_edges=bundle.pairwise_graph_edges,
        cfg=cfg,
        device=device,
    )


def _build_prediction_graph(
    request: RefineRequest,
    *,
    cfg: S2GAEConfig,
    device: torch.device,
) -> _SplitGraph:
    return _build_graph(
        pairs=request.pairs,
        pairwise_probabilities=request.pairwise_probabilities,
        pairwise_graph_edges=request.pairwise_graph_edges,
        cfg=cfg,
        device=device,
    )


def _build_graph(
    *,
    pairs: Sequence[CandidatePair],
    pairwise_probabilities: Sequence[float],
    pairwise_graph_edges: Sequence[tuple[str, str]],
    cfg: S2GAEConfig,
    device: torch.device,
) -> _SplitGraph:
    if len(pairs) != len(pairwise_probabilities):
        raise ValueError("pairs and pairwise_probabilities must have matching lengths")
    node_ids = _collect_node_ids(pairs=pairs, graph_edges=pairwise_graph_edges)
    node_to_index = {protein_id: index for index, protein_id in enumerate(node_ids)}
    node_features = load_mean_pooled_node_features(
        protein_ids=node_ids,
        cache_dir=cfg.embedding_cache_dir,
        index_path=cfg.embedding_index_path,
        input_dim=cfg.input_dim,
        max_sequence_length=cfg.max_sequence_length,
        device=device,
    )
    return _SplitGraph(
        node_features=node_features,
        edge_index=_edge_index_from_edges(
            edges=pairwise_graph_edges,
            node_to_index=node_to_index,
            num_nodes=len(node_ids),
            device=device,
        ),
        pair_index=_pair_index_from_pairs(pairs=pairs, node_to_index=node_to_index, device=device),
        pairwise_probabilities=torch.tensor(
            [float(value) for value in pairwise_probabilities],
            dtype=torch.float32,
            device=device,
        ),
    )


def _collect_node_ids(
    *,
    pairs: Sequence[CandidatePair],
    graph_edges: Sequence[tuple[str, str]],
) -> list[str]:
    protein_ids: set[str] = set()
    for pair in pairs:
        protein_ids.add(pair.protein_a)
        protein_ids.add(pair.protein_b)
    for protein_a, protein_b in graph_edges:
        protein_ids.add(protein_a)
        protein_ids.add(protein_b)
    if not protein_ids:
        raise ValueError("S2GAE split graph requires at least one protein")
    return sorted(protein_ids)


def _edge_index_from_edges(
    *,
    edges: Sequence[tuple[str, str]],
    node_to_index: Mapping[str, int],
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    edge_columns: list[tuple[int, int]] = []
    seen: set[tuple[str, str]] = set()
    for protein_a, protein_b in edges:
        edge = canonical_edge(protein_a, protein_b)
        if edge in seen:
            continue
        seen.add(edge)
        if protein_a not in node_to_index or protein_b not in node_to_index:
            raise ValueError("pairwise_graph_edges contain proteins outside candidate pairs")
        src = node_to_index[protein_a]
        dst = node_to_index[protein_b]
        if src == dst:
            continue
        edge_columns.append((src, dst))
        edge_columns.append((dst, src))
    if edge_columns:
        edge_index = torch.tensor(edge_columns, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    add_self_loops = _load_add_self_loops()
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    return cast(torch.Tensor, edge_index)


def _pair_index_from_pairs(
    *,
    pairs: Sequence[CandidatePair],
    node_to_index: Mapping[str, int],
    device: torch.device,
) -> torch.Tensor:
    indices = [
        (node_to_index[pair.protein_a], node_to_index[pair.protein_b])
        for pair in pairs
    ]
    return torch.tensor(indices, dtype=torch.long, device=device).t().contiguous()


def _validation_auprc(
    *,
    model: S2GAERefiner,
    graph: _SplitGraph,
    labels: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    predictions: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_indices in _batch_indices(labels.numel(), batch_size, labels.device):
            refined_logits, _ = model(
                node_features=graph.node_features,
                edge_index=graph.edge_index,
                pair_index=graph.pair_index[:, batch_indices],
                pairwise_probabilities=graph.pairwise_probabilities[batch_indices],
            )
            predictions.append(torch.sigmoid(refined_logits).detach().cpu())
    all_predictions = torch.cat(predictions, dim=0).numpy()
    all_labels = labels.detach().cpu().numpy()
    if len({float(label) for label in all_labels.tolist()}) < 2:
        return 0.0
    return float(average_precision_score(all_labels, all_predictions))


def _batch_indices(total: int, batch_size: int, device: torch.device) -> Iterator[torch.Tensor]:
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        yield torch.arange(start, stop, dtype=torch.long, device=device)


def _required_float_tensor(
    values: Sequence[int] | None,
    field_name: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    if values is None:
        raise ValueError(f"{field_name} is required for S2GAE training")
    return torch.tensor([float(value) for value in values], dtype=torch.float32, device=device)


def _prepared_model_and_optimizer(prepared: object) -> tuple[nn.Module, Optimizer]:
    if not isinstance(prepared, tuple) or len(prepared) != 2:
        raise TypeError("accelerator.prepare(model, optimizer) must return a two-item tuple")
    model, optimizer = prepared
    if not isinstance(model, nn.Module):
        raise TypeError("accelerator.prepare returned a non-module model")
    if not all(hasattr(optimizer, name) for name in ("zero_grad", "step", "param_groups")):
        raise TypeError("accelerator.prepare returned a non-optimizer optimizer")
    return model, cast(Optimizer, optimizer)


def _unwrap_model(model: nn.Module, accelerator: object) -> nn.Module:
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    if callable(unwrap_model):
        unwrapped = unwrap_model(model)
        if isinstance(unwrapped, nn.Module):
            return unwrapped
    return model


def _parse_config(config: Mapping[str, object]) -> S2GAEConfig:
    if "learning_rate" in config:
        raise ValueError("refiner.learning_rate is no longer supported; use refiner.optimizer.lr")
    run_id = str(config.get("_run_id", "tccig_run"))
    log_root = Path(str(config.get("_log_root", "logs")))
    log_dir = _path(config.get("log_dir"), log_root / "tccig" / "refiner" / run_id)
    checkpoint_path = _path(
        config.get("checkpoint_path"),
        Path("models") / "tccig" / "s2gae" / run_id / "best_model.pt",
    )
    embedding_cache_dir = _required_path(config, "embedding_cache_dir")
    embedding_index_path = _path(
        config.get("embedding_index_path"),
        embedding_cache_dir / "index.json",
    )
    max_sequence_length_raw = config.get("max_sequence_length")
    max_sequence_length = (
        None
        if max_sequence_length_raw is None
        else _positive_int(max_sequence_length_raw, "refiner.max_sequence_length")
    )
    return S2GAEConfig(
        encoder=str(config.get("encoder", "sage")).lower(),
        input_dim=_positive_int(config.get("input_dim", 1024), "refiner.input_dim"),
        hidden_dim=_positive_int(config.get("hidden_dim", 128), "refiner.hidden_dim"),
        num_layers=_positive_int(config.get("num_layers", 2), "refiner.num_layers"),
        decoder_hidden_dim=_positive_int(
            config.get("decoder_hidden_dim", 256),
            "refiner.decoder_hidden_dim",
        ),
        decoder_layers=_positive_int(
            config.get("decoder_layers", 2),
            "refiner.decoder_layers",
        ),
        dropout=_probability(config.get("dropout", 0.5), "refiner.dropout"),
        epochs=_positive_int(config.get("epochs", 20), "refiner.epochs"),
        batch_size=_positive_int(config.get("batch_size", 4096), "refiner.batch_size"),
        loss_config=_parse_loss_config(config.get("loss", {})),
        residual_weight=_non_negative_float(
            config.get("residual_weight", 0.001),
            "refiner.residual_weight",
        ),
        optimizer=_parse_optimizer_config(config.get("optimizer")),
        scheduler=_parse_scheduler_config(config.get("scheduler")),
        optimization=_parse_optimization_config(config.get("optimization")),
        embedding_cache_dir=embedding_cache_dir,
        embedding_index_path=embedding_index_path,
        max_sequence_length=max_sequence_length,
        checkpoint_path=checkpoint_path,
        log_dir=log_dir,
    )


def _config_to_json(cfg: S2GAEConfig) -> dict[str, object]:
    return {
        "encoder": cfg.encoder,
        "input_dim": cfg.input_dim,
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "decoder_hidden_dim": cfg.decoder_hidden_dim,
        "decoder_layers": cfg.decoder_layers,
        "dropout": cfg.dropout,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "loss": {
            "type": cfg.loss_config.loss_type,
            "pos_weight": cfg.loss_config.pos_weight,
            "label_smoothing": cfg.loss_config.label_smoothing,
        },
        "residual_weight": cfg.residual_weight,
        "optimizer": _optimizer_config_to_json(cfg.optimizer),
        "scheduler": {"type": cfg.scheduler.scheduler_type},
        "optimization": _optimization_config_to_json(cfg.optimization),
        "embedding_cache_dir": str(cfg.embedding_cache_dir),
        "embedding_index_path": str(cfg.embedding_index_path),
        "max_sequence_length": cfg.max_sequence_length,
        "checkpoint_path": str(cfg.checkpoint_path),
        "log_dir": str(cfg.log_dir),
    }


def _load_embedding_index(index_path: Path) -> dict[str, str]:
    if not index_path.exists():
        raise FileNotFoundError(f"refiner.embedding_index_path does not exist: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("refiner.embedding_index_path must contain a JSON object")
    index: dict[str, str] = {}
    for protein_id, relative_path in payload.items():
        if not isinstance(protein_id, str) or not isinstance(relative_path, str):
            raise ValueError("refiner.embedding_index_path must map protein IDs to paths")
        index[protein_id] = relative_path
    return index


def _load_pyg_convs() -> tuple[type[nn.Module], type[nn.Module]]:
    try:
        from torch_geometric.nn import GCNConv, SAGEConv
    except ImportError as error:
        raise ImportError(
            "S2GAE refiner requires PyG. Run `uv sync --group dev --find-links "
            "https://data.pyg.org/whl/torch-2.10.0+cpu.html` locally, or use the "
            "matching CUDA wheel page on HPC."
        ) from error
    return cast(tuple[type[nn.Module], type[nn.Module]], (SAGEConv, GCNConv))


def _load_add_self_loops() -> object:
    try:
        from torch_geometric.utils import add_self_loops
    except ImportError as error:
        raise ImportError(
            "S2GAE refiner requires torch_geometric.utils.add_self_loops"
        ) from error
    return add_self_loops


def _required_path(config: Mapping[str, object], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"refiner.{key} is required")
    return Path(value)


def _path(value: object, default: Path) -> Path:
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Path config values must be non-empty strings")
    return Path(value)


def _parse_loss_config(raw_loss: object) -> LossConfig:
    if not isinstance(raw_loss, Mapping):
        raise ValueError("refiner.loss must be a mapping")
    return LossConfig(
        loss_type=str(raw_loss.get("type", "bce_with_logits")),
        pos_weight=_positive_float(raw_loss.get("pos_weight", 1.0), "refiner.loss.pos_weight"),
        label_smoothing=_probability(
            raw_loss.get("label_smoothing", 0.0),
            "refiner.loss.label_smoothing",
        ),
    )


def _parse_optimizer_config(raw_optimizer: object) -> S2GAEOptimizerConfig:
    if not isinstance(raw_optimizer, Mapping):
        raise ValueError("refiner.optimizer must be a mapping")
    optimizer_type = str(raw_optimizer.get("type", "")).lower()
    if optimizer_type != "adamw":
        raise ValueError("refiner.optimizer.type must be 'adamw'")
    return S2GAEOptimizerConfig(
        optimizer_type=optimizer_type,
        lr=_positive_float(raw_optimizer.get("lr"), "refiner.optimizer.lr"),
        weight_decay=_non_negative_float(
            raw_optimizer.get("weight_decay", 0.0),
            "refiner.optimizer.weight_decay",
        ),
        beta1=_exclusive_probability(
            raw_optimizer.get("beta1", 0.9),
            "refiner.optimizer.beta1",
        ),
        beta2=_exclusive_probability(
            raw_optimizer.get("beta2", 0.999),
            "refiner.optimizer.beta2",
        ),
        eps=_positive_float(raw_optimizer.get("eps", 1e-8), "refiner.optimizer.eps"),
    )


def _parse_scheduler_config(raw_scheduler: object) -> S2GAESchedulerConfig:
    if not isinstance(raw_scheduler, Mapping):
        raise ValueError("refiner.scheduler must be a mapping")
    scheduler_type = str(raw_scheduler.get("type", "")).lower()
    if scheduler_type != "none":
        raise ValueError("refiner.scheduler.type must be 'none'")
    return S2GAESchedulerConfig(scheduler_type=scheduler_type)


def _parse_optimization_config(raw_optimization: object) -> S2GAEOptimizationConfig:
    if not isinstance(raw_optimization, Mapping):
        raise ValueError("refiner.optimization must be a mapping")
    return S2GAEOptimizationConfig(
        gradient_clip_norm=_optional_positive_float(
            raw_optimization.get("gradient_clip_norm", 1.0),
            "refiner.optimization.gradient_clip_norm",
        )
    )


def _optimizer_config_to_json(config: S2GAEOptimizerConfig) -> dict[str, object]:
    return {
        "type": config.optimizer_type,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "eps": config.eps,
    }


def _optimization_config_to_json(config: S2GAEOptimizationConfig) -> dict[str, object]:
    return {"gradient_clip_norm": config.gradient_clip_norm}


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        parsed = int(cast(int | str, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a positive integer") from error
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _positive_float(value: object, field_name: str) -> float:
    try:
        parsed = float(cast(float | str, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a positive float") from error
    if parsed <= 0.0 or math.isnan(parsed) or math.isinf(parsed):
        raise ValueError(f"{field_name} must be a positive float")
    return parsed


def _optional_positive_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    parsed = _non_negative_float(value, field_name)
    return None if parsed == 0.0 else parsed


def _non_negative_float(value: object, field_name: str) -> float:
    try:
        parsed = float(cast(float | str, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a non-negative float") from error
    if parsed < 0.0 or math.isnan(parsed) or math.isinf(parsed):
        raise ValueError(f"{field_name} must be a non-negative float")
    return parsed


def _probability(value: object, field_name: str) -> float:
    parsed = _non_negative_float(value, field_name)
    if parsed >= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1)")
    return parsed


def _exclusive_probability(value: object, field_name: str) -> float:
    parsed = _positive_float(value, field_name)
    if parsed >= 1.0:
        raise ValueError(f"{field_name} must be in (0, 1)")
    return parsed


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    device = parameters[0].grad.device
    norms = [parameter.grad.detach().norm(2).to(device) for parameter in parameters]
    return float(torch.norm(torch.stack(norms), 2).detach().cpu().item())


def _current_learning_rate(optimizer: Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])
