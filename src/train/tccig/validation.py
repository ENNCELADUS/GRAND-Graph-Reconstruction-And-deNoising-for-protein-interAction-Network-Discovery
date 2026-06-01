"""Validation runner for TCCIG training epochs."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import cast

import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.topology.losses import TopologyLossWeights
from src.train.tccig.config import TCCIGTrainConfig, tccig_train_config
from src.train.tccig.data import TCCIGDataContext
from src.train.tccig.retrieval import build_full_reconstruction_universe
from src.train.topology import shared as topology_train
from src.utils.config import ConfigDict


class TCCIGValidationRunner:
    """Run pairwise and internal topology validation for TCCIG epochs."""

    def __init__(
        self,
        *,
        raw_config: ConfigDict,
        train_cfg: TCCIGTrainConfig,
        data_context: TCCIGDataContext,
        dataloaders: dict[str, DataLoader[dict[str, object]]],
        device: torch.device,
    ) -> None:
        self.raw_config = raw_config
        self.train_cfg = train_cfg
        self.data_context = data_context
        self.dataloaders = dataloaders
        self.device = device

    def evaluate(
        self,
        *,
        model: nn.Module,
        epoch_index: int,
        topology_loss_scale_value: float,
        previous_topology_loss_scale: float | None,
    ) -> topology_train.ValidationEpochResult:
        """Evaluate one TCCIG epoch with the existing topology metrics."""
        model.eval()
        with torch.no_grad():
            val_pair_start = time.perf_counter()
            labels, probabilities, average_loss = (
                self.data_context.evaluator.collect_probabilities_and_labels(
                    model=model,
                    data_loader=self.dataloaders["valid"],
                    device=self.device,
                )
            )
            val_pair_stats = self.data_context.evaluator.metrics_from_outputs(
                labels=labels,
                probabilities=probabilities,
                average_loss=average_loss,
                prefix="val",
            )
            val_pair_stats = dict(val_pair_stats)
            val_pair_stats.update(self._validation_reconstruction_stats(model=model))
            val_pair_pass_seconds = time.perf_counter() - val_pair_start
        threshold_start = time.perf_counter()
        decision_threshold, _ = topology_train._resolve_internal_validation_threshold(
            config=self.raw_config,
            stage_cfg=tccig_train_config(self.raw_config),
            stage_name="tccig_train",
        )
        threshold_resolution_seconds = time.perf_counter() - threshold_start
        should_run_internal_validation = topology_train._should_run_internal_validation(
            finetune_cfg=tccig_train_config(self.raw_config),
            stage_name="tccig_train",
            epoch_index=epoch_index,
            topology_loss_scale_value=topology_loss_scale_value,
            previous_topology_loss_scale=previous_topology_loss_scale,
        )
        if should_run_internal_validation:
            internal_validation_start = time.perf_counter()
            internal_val_topology_stats = topology_train._evaluate_internal_validation_subgraphs(
                model=model,
                validation_plan=self.data_context.internal_validation_plan,
                embedding_repository=self.data_context.embedding_repository,
                inference_batch_size=self.train_cfg.internal_validation_inference_batch_size,
                threshold=decision_threshold,
                device=self.device,
                accelerator=self.data_context.evaluator.accelerator,
                compute_spectral_stats=self.train_cfg.internal_validation_compute_spectral_stats,
                compute_clustering_mmd=self.train_cfg.internal_validation_compute_clustering_mmd,
            )
            internal_validation_seconds = time.perf_counter() - internal_validation_start
        else:
            internal_val_topology_stats = topology_train._empty_internal_validation_summary()
            internal_validation_seconds = 0.0

        validation_loss_weights = TopologyLossWeights(
            alpha=0.0,
            beta=self.train_cfg.loss_weights.density * topology_loss_scale_value,
            gamma=self.train_cfg.loss_weights.degree * topology_loss_scale_value,
            delta=self.train_cfg.loss_weights.clustering * topology_loss_scale_value,
        )
        val_pair_loss = float(val_pair_stats.get("val_loss", 0.0))
        val_topology_loss = topology_train._validation_topology_loss(
            loss_weights=validation_loss_weights,
            internal_val_topology_stats=internal_val_topology_stats,
            include_clustering_mmd=self.train_cfg.internal_validation_compute_clustering_mmd,
        )
        val_pair_stats.update(
            _composite_monitor_stats(
                train_cfg=self.train_cfg,
                val_pair_stats=val_pair_stats,
                internal_val_topology_stats=internal_val_topology_stats,
            )
        )
        return topology_train.ValidationEpochResult(
            decision_threshold=decision_threshold,
            val_pair_stats=val_pair_stats,
            internal_val_topology_stats=internal_val_topology_stats,
            val_pair_pass_seconds=val_pair_pass_seconds,
            threshold_resolution_seconds=threshold_resolution_seconds,
            internal_validation_seconds=internal_validation_seconds,
            val_topology_loss=val_topology_loss,
            val_total_loss=val_pair_loss + val_topology_loss,
        )

    def _validation_reconstruction_stats(self, *, model: nn.Module) -> dict[str, float]:
        """Evaluate retrieval over the full validation reconstruction universe."""
        if not self.train_cfg.validation_reconstruction.enabled:
            return {}
        graph = self.data_context.internal_val_graph
        protein_ids = tuple(sorted(str(node) for node in graph.nodes))
        if len(protein_ids) < 2:
            return {
                "val_reconstruction_candidate_count": 0.0,
                "val_reconstruction_positive_count": 0.0,
                "val_candidate_auprc": 0.0,
                "val_retrieval_recall_at_20": 0.0,
            }
        universe = build_full_reconstruction_universe(
            protein_ids=protein_ids,
            positive_edges=((str(node_a), str(node_b)) for node_a, node_b in graph.edges),
        )
        node_to_index = {protein_id: index for index, protein_id in enumerate(protein_ids)}
        candidate_pairs = torch.tensor(
            [
                [node_to_index[node_a], node_to_index[node_b]]
                for node_a, node_b in universe.records
            ],
            dtype=torch.long,
            device=self.device,
        ).t()
        labels = universe.labels
        max_pairs = self.train_cfg.validation_reconstruction.max_pairs
        if max_pairs is not None and candidate_pairs.size(1) > max_pairs:
            generator = torch.Generator().manual_seed(self.train_cfg.run_seed)
            selected = torch.randperm(candidate_pairs.size(1), generator=generator)[:max_pairs]
            candidate_pairs = candidate_pairs[:, selected.to(device=self.device)]
            labels = labels[selected]
        graph_model = model
        unwrap_model = getattr(self.data_context.evaluator.accelerator, "unwrap_model", None)
        if callable(unwrap_model):
            graph_model = cast(nn.Module, unwrap_model(model))
        encode_proteins = getattr(graph_model, "encode_proteins", None)
        retrieval_score_matrix = getattr(graph_model, "retrieval_score_matrix", None)
        if not callable(encode_proteins) or not callable(retrieval_score_matrix):
            return {}
        protein_embeddings, protein_lengths = _load_validation_node_inputs(
            data_context=self.data_context,
            protein_ids=protein_ids,
            device=self.device,
        )
        encoded = cast(
            Mapping[str, torch.Tensor],
            encode_proteins(
                protein_embeddings=protein_embeddings,
                protein_lengths=protein_lengths,
            ),
        )
        scores = cast(torch.Tensor, retrieval_score_matrix(encoded))
        pair_scores = scores[candidate_pairs[0], candidate_pairs[1]].detach().float().cpu()
        labels = labels.detach().float().cpu()
        stats = {
            "val_reconstruction_candidate_count": float(labels.numel()),
            "val_reconstruction_positive_count": float(labels.sum().item()),
            "val_candidate_auprc": _safe_average_precision(labels, pair_scores),
        }
        for fraction in self.train_cfg.validation_reconstruction.recall_k_percent:
            key = f"val_retrieval_recall_at_{int(fraction)}"
            stats[key] = _recall_at_fraction(
                labels=labels,
                scores=pair_scores,
                fraction_percent=fraction,
            )
        stats.setdefault("val_retrieval_recall_at_20", 0.0)
        return stats


def _composite_monitor_stats(
    *,
    train_cfg: TCCIGTrainConfig,
    val_pair_stats: dict[str, float],
    internal_val_topology_stats: Mapping[str, float],
) -> dict[str, float]:
    """Return graph-prior retrieval monitor metrics."""
    monitor_cfg = train_cfg.monitor
    retrieval_recall = float(
        val_pair_stats.get(
            "val_retrieval_recall_at_20",
            val_pair_stats.get("val_recall", val_pair_stats.get("val_sensitivity", 0.0)),
        )
    )
    candidate_auprc = float(
        val_pair_stats.get("val_candidate_auprc", val_pair_stats.get("val_auprc", 0.0))
    )
    graph_sim = float(internal_val_topology_stats.get("graph_sim", 0.0))
    relative_density = float(internal_val_topology_stats.get("relative_density", 0.0))
    degree_mmd = float(internal_val_topology_stats.get("deg_dist_mmd", 0.0))
    clustering_mmd = float(internal_val_topology_stats.get("cc_mmd", 0.0))
    composite = (
        monitor_cfg.recall_weight * retrieval_recall
        + monitor_cfg.auprc_weight * candidate_auprc
        + monitor_cfg.graph_sim_weight * graph_sim
        - monitor_cfg.relative_density_penalty * abs(relative_density - 1.0)
        - monitor_cfg.degree_mmd_penalty * degree_mmd
        - monitor_cfg.clustering_mmd_penalty * clustering_mmd
    )
    return {
        "val_retrieval_recall_at_20": retrieval_recall,
        "val_candidate_auprc": candidate_auprc,
        "val_composite_score": float(composite),
    }


def _safe_average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Return AUPRC when both classes are present."""
    if labels.numel() == 0 or torch.unique(labels).numel() < 2:
        return 0.0
    return float(average_precision_score(labels.cpu().numpy(), scores.cpu().numpy()))


def _recall_at_fraction(
    *,
    labels: torch.Tensor,
    scores: torch.Tensor,
    fraction_percent: float,
) -> float:
    """Return positive recall inside the top scored fraction of candidates."""
    positive_count = int(labels.sum().item())
    if positive_count <= 0 or labels.numel() == 0:
        return 0.0
    k = max(1, int(round(labels.numel() * float(fraction_percent) / 100.0)))
    k = min(k, labels.numel())
    top_indices = torch.topk(scores.float(), k=k).indices
    return float(labels[top_indices].sum().item() / float(positive_count))


def _load_validation_node_inputs(
    *,
    data_context: TCCIGDataContext,
    protein_ids: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load validation protein embeddings for graph reconstruction metrics."""
    embeddings = data_context.embedding_repository.get_many(tuple(protein_ids))
    embedding_tensors = [embeddings[protein_id] for protein_id in protein_ids]
    protein_embeddings = pad_sequence(embedding_tensors, batch_first=True).to(device)
    protein_lengths = torch.tensor(
        [embedding.size(0) for embedding in embedding_tensors],
        dtype=torch.long,
        device=device,
    )
    return protein_embeddings, protein_lengths
