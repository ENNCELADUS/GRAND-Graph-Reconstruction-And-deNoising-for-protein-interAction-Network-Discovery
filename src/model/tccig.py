"""TCCIG feature-only graph generator for inductive interactome reconstruction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import cast

import torch
import torch.nn.functional as functional
from torch import nn


def _to_int(value: object, field_name: str) -> int:
    """Parse an integer model configuration value."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
    raise ValueError(f"{field_name} must be an integer")


def _to_float(value: object, field_name: str) -> float:
    """Parse a floating-point model configuration value."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a float")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a float") from exc
    raise ValueError(f"{field_name} must be a float")


def _to_mapping(value: object, field_name: str) -> Mapping[str, object]:
    """Return a configuration mapping or raise a clear error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return cast(Mapping[str, object], value)


def _to_string(value: object, field_name: str) -> str:
    """Parse a string model configuration value."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _to_bool(value: object, field_name: str) -> bool:
    """Parse a boolean model configuration value."""
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""
    return value > 0 and (value & (value - 1)) == 0


def _build_mlp(
    *,
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Build a compact GELU MLP."""
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


def _last_linear(module: nn.Module) -> nn.Linear | None:
    """Return the final linear layer contained in a module."""
    final_linear: nn.Linear | None = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            final_linear = child
    return final_linear


def _hadamard_transform(values: torch.Tensor) -> torch.Tensor:
    """Apply an in-place-free Walsh-Hadamard transform on the last dimension."""
    width = values.size(-1)
    if not _is_power_of_two(width):
        raise ValueError("SORF random features require power-of-two rff_input_dim")
    output = values
    step = 1
    while step < width:
        reshaped = output.reshape(*output.shape[:-1], -1, step * 2)
        left = reshaped[..., :step]
        right = reshaped[..., step : step * 2]
        output = torch.cat([left + right, left - right], dim=-1).reshape_as(output)
        step *= 2
    return output / math.sqrt(float(width))


class TCCIG(nn.Module):
    """Topology-constrained conditional interactome generator.

    The graph forward path accepts only protein-intrinsic embeddings and optional
    feature-derived candidate pairs. Training labels and topology targets are
    consumed by the pipeline loss code, not by this model hook.
    """

    name: str = "tccig"

    def __init__(self, **model_config: object) -> None:
        super().__init__()
        self.input_dim: int = _to_int(model_config["input_dim"], "model_config.input_dim")
        self.d_model: int = _to_int(model_config["d_model"], "model_config.d_model")
        self.dropout: float = _to_float(model_config.get("dropout", 0.1), "model_config.dropout")
        self.lowrank_dim: int = _to_int(
            model_config.get("lowrank_dim", max(1, self.d_model // 4)),
            "model_config.lowrank_dim",
        )
        retrieval_cfg = _to_mapping(
            model_config.get("retrieval", {}),
            "model_config.retrieval",
        )
        self.retrieval_dim: int = _to_int(
            retrieval_cfg.get("dim", max(1, self.d_model // 2)),
            "model_config.retrieval.dim",
        )
        self.retrieval_top_k: int = _to_int(
            retrieval_cfg.get("top_k", 128),
            "model_config.retrieval.top_k",
        )
        self.rff_features: int = _to_int(
            retrieval_cfg.get("rff_features", max(64, self.retrieval_dim * 2)),
            "model_config.retrieval.rff_features",
        )
        self.rff_input_dim: int = _to_int(
            retrieval_cfg.get("rff_input_dim", self.d_model),
            "model_config.retrieval.rff_input_dim",
        )
        self.rff_sigma: float = _to_float(
            retrieval_cfg.get("rff_sigma", 0.5),
            "model_config.retrieval.rff_sigma",
        )
        if self.rff_sigma <= 0.0:
            raise ValueError("model_config.retrieval.rff_sigma must be > 0")
        self.rff_backend: str = _to_string(
            retrieval_cfg.get("rff_backend", "sorf"),
            "model_config.retrieval.rff_backend",
        ).lower()
        if self.rff_backend not in {"sorf", "dense"}:
            raise ValueError("model_config.retrieval.rff_backend must be 'sorf' or 'dense'")
        if self.rff_backend == "sorf" and not _is_power_of_two(self.rff_input_dim):
            raise ValueError("model_config.retrieval.rff_input_dim must be a power of two")
        self.normalize_retrieval_embeddings: bool = _to_bool(
            retrieval_cfg.get("normalize", True),
            "model_config.retrieval.normalize",
        )
        self.retrieval_feature_source: str = _to_string(
            retrieval_cfg.get("feature_source", "learned"),
            "model_config.retrieval.feature_source",
        ).lower()
        if self.retrieval_feature_source not in {"learned", "pooled_input"}:
            raise ValueError(
                "model_config.retrieval.feature_source must be 'learned' or 'pooled_input'"
            )
        self.retrieval_logit_gate_init: float = _to_float(
            retrieval_cfg.get("logit_gate_init", 1.0),
            "model_config.retrieval.logit_gate_init",
        )
        self.decoder_mode: str = _to_string(
            model_config.get("decoder_mode", "rerank"),
            "model_config.decoder_mode",
        ).lower()
        if self.decoder_mode not in {"rerank", "retrieval_only"}:
            raise ValueError("model_config.decoder_mode must be 'rerank' or 'retrieval_only'")
        self.decoder_structural_gate_init: float = _to_float(
            model_config.get("decoder_structural_gate_init", 0.1),
            "model_config.decoder_structural_gate_init",
        )
        self.num_modules: int = _to_int(
            model_config.get("num_modules", 64),
            "model_config.num_modules",
        )
        self.self_refinement_rounds: int = _to_int(
            model_config.get("self_refinement_rounds", 0),
            "model_config.self_refinement_rounds",
        )
        if self.self_refinement_rounds not in {0, 1}:
            raise ValueError("model_config.self_refinement_rounds must be 0 or 1")

        candidate_cfg = _to_mapping(
            model_config.get("candidate_proposer", {"type": "all_pairs"}),
            "model_config.candidate_proposer",
        )
        self.candidate_proposer_type: str = _to_string(
            candidate_cfg.get("type", "all_pairs"),
            "model_config.candidate_proposer.type",
        ).lower()
        if self.candidate_proposer_type != "all_pairs":
            raise ValueError("TCCIG v1 supports only candidate_proposer.type='all_pairs'")

        pair_mlp_cfg = _to_mapping(
            model_config.get("pair_mlp", {"hidden_dims": [self.d_model]}),
            "model_config.pair_mlp",
        )
        raw_hidden_dims = pair_mlp_cfg.get("hidden_dims", [self.d_model])
        if not isinstance(raw_hidden_dims, Sequence) or isinstance(raw_hidden_dims, (str, bytes)):
            raise ValueError("model_config.pair_mlp.hidden_dims must be a sequence")
        pair_hidden_dims = [
            _to_int(value, "model_config.pair_mlp.hidden_dims") for value in raw_hidden_dims
        ]

        self.protein_projection: nn.Sequential = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.set_context: nn.Sequential = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.pair_mlp: nn.Sequential = _build_mlp(
            input_dim=(3 * self.d_model) + 1,
            hidden_dims=pair_hidden_dims,
            output_dim=1,
            dropout=self.dropout,
        )
        self.hub_head: nn.Linear = nn.Linear(self.d_model, 1)
        self.lowrank_head: nn.Linear = nn.Linear(self.d_model, self.lowrank_dim)
        self.module_head: nn.Linear = nn.Linear(self.d_model, self.num_modules)
        self.module_interactions: nn.Parameter = nn.Parameter(torch.eye(self.num_modules))
        self.query_head: nn.Linear = nn.Linear(self.d_model, self.retrieval_dim)
        self.key_head: nn.Linear = nn.Linear(self.d_model, self.retrieval_dim)
        self.struct_head: nn.Linear = nn.Linear(self.d_model, self.retrieval_dim)
        self.degree_head: nn.Linear = nn.Linear(self.d_model, 1)
        self.residue_projection: nn.Sequential = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.rff_input_dim),
        )
        self.residue_attention_head: nn.Linear = nn.Linear(self.rff_input_dim, 1)
        self.retrieval_logit_gate: nn.Parameter = nn.Parameter(
            torch.tensor(self.retrieval_logit_gate_init)
        )
        self.residue_score_gate: nn.Parameter = nn.Parameter(torch.tensor(1.0))
        self.struct_score_gate: nn.Parameter = nn.Parameter(
            torch.tensor(self.decoder_structural_gate_init)
        )
        self.degree_score_gate: nn.Parameter = nn.Parameter(
            torch.tensor(self.decoder_structural_gate_init)
        )
        self.hub_score_gate: nn.Parameter = nn.Parameter(
            torch.tensor(self.decoder_structural_gate_init)
        )
        self.lowrank_score_gate: nn.Parameter = nn.Parameter(
            torch.tensor(self.decoder_structural_gate_init)
        )
        self.module_score_gate: nn.Parameter = nn.Parameter(
            torch.tensor(self.decoder_structural_gate_init)
        )
        self.density_bias_head: nn.Sequential = _build_mlp(
            input_dim=self.d_model,
            hidden_dims=[self.d_model],
            output_dim=1,
            dropout=self.dropout,
        )
        self.edge_budget_head: nn.Sequential = _build_mlp(
            input_dim=self.d_model,
            hidden_dims=[self.d_model],
            output_dim=1,
            dropout=self.dropout,
        )
        self.message_projection: nn.Linear | None = None
        self.refinement_cell: nn.GRUCell | None = None
        if self.self_refinement_rounds == 1:
            self.message_projection = nn.Linear(self.d_model, self.d_model)
            self.refinement_cell = nn.GRUCell(self.d_model, self.d_model)
        phase = torch.rand(self.rff_features) * (2.0 * math.pi)
        self.rff_phase: torch.Tensor
        self.rff_weight: torch.Tensor
        self.sorf_signs: torch.Tensor
        self.register_buffer("rff_phase", phase)
        if self.rff_backend == "dense":
            rff_weight = torch.randn(self.rff_features, self.rff_input_dim)
            self.register_buffer("rff_weight", rff_weight)
            self.register_buffer("sorf_signs", torch.empty(0))
        else:
            block_count = math.ceil(self.rff_features / float(self.rff_input_dim))
            signs = torch.randint(
                low=0,
                high=2,
                size=(block_count, 3, self.rff_input_dim),
                dtype=torch.float32,
            )
            signs = signs.mul(2.0).sub(1.0)
            self.register_buffer("sorf_signs", signs)
            self.register_buffer("rff_weight", torch.empty(0))

    @staticmethod
    def _masked_mean_pool(embeddings: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """Mean-pool token embeddings using sequence lengths."""
        if embeddings.dim() != 3:
            raise ValueError("protein embeddings must have shape (n, seq_len, input_dim)")
        if lengths is None:
            return embeddings.mean(dim=1)
        lengths = lengths.to(device=embeddings.device, dtype=torch.long)
        if lengths.dim() != 1 or lengths.numel() != embeddings.size(0):
            raise ValueError("protein lengths must have shape (n,)")
        clipped_lengths = lengths.clamp(min=1, max=embeddings.size(1))
        token_ids = torch.arange(embeddings.size(1), device=embeddings.device).unsqueeze(0)
        mask = token_ids < clipped_lengths.unsqueeze(1)
        weighted = embeddings * mask.unsqueeze(-1).to(dtype=embeddings.dtype)
        return weighted.sum(dim=1) / clipped_lengths.unsqueeze(-1).to(dtype=embeddings.dtype)

    def _encode_nodes(
        self,
        protein_embeddings: torch.Tensor,
        protein_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode a protein set without using graph topology."""
        pooled = self._masked_mean_pool(protein_embeddings, protein_lengths)
        h0 = cast(torch.Tensor, self.protein_projection(pooled))
        set_summary = h0.mean(dim=0, keepdim=True).expand_as(h0)
        return cast(torch.Tensor, self.set_context(torch.cat([h0, set_summary], dim=-1)))

    @staticmethod
    def _token_mask(
        *,
        embeddings: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return a boolean token mask for residue-level pooling."""
        if lengths is None:
            return torch.ones(
                embeddings.shape[:2],
                device=embeddings.device,
                dtype=torch.bool,
            )
        clipped_lengths = lengths.to(device=embeddings.device, dtype=torch.long).clamp(
            min=1,
            max=embeddings.size(1),
        )
        token_ids = torch.arange(embeddings.size(1), device=embeddings.device).unsqueeze(0)
        return token_ids < clipped_lengths.unsqueeze(1)

    def _random_fourier_features(self, projected_tokens: torch.Tensor) -> torch.Tensor:
        """Return residue-level random Fourier features."""
        if self.rff_backend == "dense":
            projection = torch.matmul(projected_tokens, self.rff_weight.t()) / self.rff_sigma
        else:
            block_count = self.sorf_signs.size(0)
            expanded = projected_tokens.unsqueeze(2).expand(
                *projected_tokens.shape[:-1],
                block_count,
                self.rff_input_dim,
            )
            transformed = expanded
            for sign_index in range(3):
                signs = self.sorf_signs[:, sign_index, :].to(
                    device=projected_tokens.device,
                    dtype=projected_tokens.dtype,
                )
                transformed = _hadamard_transform(
                    transformed * signs.view(1, 1, block_count, self.rff_input_dim)
                )
            projection = transformed.reshape(
                *projected_tokens.shape[:-1],
                block_count * self.rff_input_dim,
            )[..., : self.rff_features]
            projection = projection / self.rff_sigma
        phase = self.rff_phase.to(device=projected_tokens.device, dtype=projected_tokens.dtype)
        return math.sqrt(2.0 / float(self.rff_features)) * torch.cos(projection + phase)

    def _encode_residue_factor(
        self,
        *,
        protein_embeddings: torch.Tensor,
        protein_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode token embeddings into one residue-aware retrieval vector per protein."""
        projected_tokens = cast(torch.Tensor, self.residue_projection(protein_embeddings))
        residue_features = self._random_fourier_features(projected_tokens)
        attention_logits = self.residue_attention_head(projected_tokens).squeeze(-1)
        mask = self._token_mask(embeddings=protein_embeddings, lengths=protein_lengths)
        attention_logits = attention_logits.masked_fill(
            ~mask,
            torch.finfo(attention_logits.dtype).min,
        )
        attention = torch.softmax(attention_logits, dim=-1)
        return (attention.unsqueeze(-1) * residue_features).sum(dim=1)

    def _maybe_normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize retrieval embeddings when configured."""
        if not self.normalize_retrieval_embeddings:
            return embeddings
        return functional.normalize(embeddings, dim=-1)

    @staticmethod
    def _resize_feature_vector(features: torch.Tensor, output_dim: int) -> torch.Tensor:
        """Deterministically truncate or pad feature vectors to ``output_dim``."""
        input_dim = features.size(-1)
        if input_dim == output_dim:
            return features
        if input_dim > output_dim:
            return features[..., :output_dim]
        return functional.pad(features, (0, output_dim - input_dim))

    def encode_proteins(
        self,
        *,
        protein_embeddings: torch.Tensor,
        protein_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode proteins into retrieval, structural-prior, and reranking states."""
        if protein_embeddings.size(-1) != self.input_dim:
            raise ValueError("protein embedding dimension must match model_config.input_dim")
        pooled_input = self._masked_mean_pool(protein_embeddings, protein_lengths)
        node_embeddings = self._encode_nodes(protein_embeddings, protein_lengths)
        module_memberships = functional.softmax(self.module_head(node_embeddings), dim=-1)
        degree_prediction = functional.softplus(self.degree_head(node_embeddings).squeeze(-1))
        if self.retrieval_feature_source == "pooled_input":
            retrieval_features = self._maybe_normalize(
                self._resize_feature_vector(pooled_input, self.retrieval_dim)
            )
            residue_features = self._maybe_normalize(
                self._resize_feature_vector(pooled_input, self.rff_features)
            )
            return {
                "node": node_embeddings,
                "query": retrieval_features,
                "key": retrieval_features,
                "struct": retrieval_features,
                "residue": residue_features,
                "module": module_memberships,
                "degree": degree_prediction,
            }
        return {
            "node": node_embeddings,
            "query": self._maybe_normalize(self.query_head(node_embeddings)),
            "key": self._maybe_normalize(self.key_head(node_embeddings)),
            "struct": self._maybe_normalize(self.struct_head(node_embeddings)),
            "residue": self._maybe_normalize(
                self._encode_residue_factor(
                    protein_embeddings=protein_embeddings,
                    protein_lengths=protein_lengths,
                )
            ),
            "module": module_memberships,
            "degree": degree_prediction,
        }

    def encode_proteins_batched(
        self,
        *,
        protein_embeddings: Sequence[torch.Tensor],
        device: torch.device | None = None,
        batch_size: int = 64,
    ) -> dict[str, torch.Tensor]:
        """Encode variable-length proteins without one global padded token tensor."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        embedding_tensors = tuple(protein_embeddings)
        if not embedding_tensors:
            raise ValueError("protein_embeddings must not be empty")
        for embedding in embedding_tensors:
            if embedding.dim() != 2 or embedding.size(-1) != self.input_dim:
                raise ValueError("each protein embedding must have shape (seq_len, input_dim)")
        resolved_device = device or next(self.parameters()).device
        pooled_chunks: list[torch.Tensor] = []
        for start in range(0, len(embedding_tensors), batch_size):
            batch_embeddings, batch_lengths = self._pad_embedding_batch(
                embedding_tensors[start : start + batch_size],
                device=resolved_device,
            )
            pooled_chunks.append(self._masked_mean_pool(batch_embeddings, batch_lengths))
        pooled_input = torch.cat(pooled_chunks, dim=0)
        h0 = cast(torch.Tensor, self.protein_projection(pooled_input))
        set_summary = h0.mean(dim=0, keepdim=True).expand_as(h0)
        node_embeddings = cast(
            torch.Tensor,
            self.set_context(torch.cat([h0, set_summary], dim=-1)),
        )
        module_memberships = functional.softmax(self.module_head(node_embeddings), dim=-1)
        degree_prediction = functional.softplus(self.degree_head(node_embeddings).squeeze(-1))
        if self.retrieval_feature_source == "pooled_input":
            retrieval_features = self._maybe_normalize(
                self._resize_feature_vector(pooled_input, self.retrieval_dim)
            )
            residue_features = self._maybe_normalize(
                self._resize_feature_vector(pooled_input, self.rff_features)
            )
            return {
                "node": node_embeddings,
                "query": retrieval_features,
                "key": retrieval_features,
                "struct": retrieval_features,
                "residue": residue_features,
                "module": module_memberships,
                "degree": degree_prediction,
            }
        residue = self._encode_residue_factor_batched(
            embedding_tensors=embedding_tensors,
            device=resolved_device,
            batch_size=batch_size,
        )
        return {
            "node": node_embeddings,
            "query": self._maybe_normalize(self.query_head(node_embeddings)),
            "key": self._maybe_normalize(self.key_head(node_embeddings)),
            "struct": self._maybe_normalize(self.struct_head(node_embeddings)),
            "residue": self._maybe_normalize(residue),
            "module": module_memberships,
            "degree": degree_prediction,
        }

    @staticmethod
    def _pad_embedding_batch(
        embedding_tensors: Sequence[torch.Tensor],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad one bounded protein batch."""
        batch_embeddings = nn.utils.rnn.pad_sequence(
            [embedding.to(device) for embedding in embedding_tensors],
            batch_first=True,
        )
        batch_lengths = torch.tensor(
            [embedding.size(0) for embedding in embedding_tensors],
            dtype=torch.long,
            device=device,
        )
        return batch_embeddings, batch_lengths

    def _encode_residue_factor_batched(
        self,
        *,
        embedding_tensors: Sequence[torch.Tensor],
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """Encode residue factors in length-sorted bounded batches."""
        ordered_indices = sorted(
            range(len(embedding_tensors)),
            key=lambda index: int(embedding_tensors[index].size(0)),
            reverse=True,
        )
        residue_chunks: list[torch.Tensor] = []
        chunk_indices: list[int] = []
        for start in range(0, len(ordered_indices), batch_size):
            batch_indices = ordered_indices[start : start + batch_size]
            batch_embeddings, batch_lengths = self._pad_embedding_batch(
                [embedding_tensors[index] for index in batch_indices],
                device=device,
            )
            residue_chunks.append(
                self._encode_residue_factor(
                    protein_embeddings=batch_embeddings,
                    protein_lengths=batch_lengths,
                )
            )
            chunk_indices.extend(batch_indices)
        encoded = torch.cat(residue_chunks, dim=0)
        restore_indices = torch.empty(len(chunk_indices), dtype=torch.long, device=device)
        restore_indices[
            torch.tensor(chunk_indices, dtype=torch.long, device=device)
        ] = torch.arange(len(chunk_indices), dtype=torch.long, device=device)
        return encoded.index_select(0, restore_indices)

    def encode_graph_nodes(
        self,
        *,
        protein_embeddings: torch.Tensor,
        protein_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a protein set for graph-level candidate scoring."""
        if protein_embeddings.size(-1) != self.input_dim:
            raise ValueError("protein embedding dimension must match model_config.input_dim")
        return self._encode_nodes(protein_embeddings, protein_lengths)

    @staticmethod
    def _all_pairs(num_nodes: int, device: torch.device) -> torch.Tensor:
        """Return upper-triangle candidate pairs shaped ``(2, num_pairs)``."""
        if num_nodes < 2:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)

    @staticmethod
    def _validate_candidate_pairs(
        candidate_pairs: torch.Tensor,
        *,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Validate and normalize candidate-pair tensors."""
        pairs = candidate_pairs.to(device=device, dtype=torch.long)
        if pairs.dim() != 2:
            raise ValueError("candidate_pairs must have shape (2, e) or (e, 2)")
        if pairs.size(0) == 2:
            normalized = pairs
        elif pairs.size(1) == 2:
            normalized = pairs.t().contiguous()
        else:
            raise ValueError("candidate_pairs must have shape (2, e) or (e, 2)")
        if normalized.numel() and (
            int(normalized.min().item()) < 0 or int(normalized.max().item()) >= num_nodes
        ):
            raise ValueError("candidate_pairs contains node indices outside the protein set")
        if normalized.numel() and torch.any(normalized[0] == normalized[1]):
            raise ValueError("candidate_pairs must not contain self edges")
        return torch.stack(
            (
                torch.minimum(normalized[0], normalized[1]),
                torch.maximum(normalized[0], normalized[1]),
            ),
            dim=0,
        )

    def edge_budget_from_node_embeddings(
        self,
        *,
        node_embeddings: torch.Tensor,
        candidate_count: int,
    ) -> torch.Tensor:
        """Estimate the edge budget for a fixed candidate universe size."""
        if candidate_count < 0:
            raise ValueError("candidate_count must be non-negative")
        set_state = node_embeddings.mean(dim=0, keepdim=True)
        budget_fraction = torch.sigmoid(self.edge_budget_head(set_state).squeeze()).float()
        return budget_fraction * float(candidate_count)

    def retrieval_score_matrix(self, encoded: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return symmetric retrieval scores for every encoded protein pair."""
        query = encoded["query"]
        key = encoded["key"]
        residue = encoded["residue"]
        struct = encoded["struct"]
        degree = encoded["degree"]
        directed = torch.matmul(query, key.t()) / math.sqrt(float(self.retrieval_dim))
        residue_score = torch.matmul(residue, residue.t()) / math.sqrt(float(self.rff_features))
        struct_score = torch.matmul(struct, struct.t()) / math.sqrt(float(self.retrieval_dim))
        degree_score = torch.log1p(degree).unsqueeze(0) + torch.log1p(degree).unsqueeze(1)
        score = 0.5 * (directed + directed.t())
        score = score + self.residue_score_gate * residue_score
        score = score + self.struct_score_gate * struct_score
        score = score + self.degree_score_gate * degree_score
        return score

    def retrieve_candidate_pairs(
        self,
        *,
        encoded: Mapping[str, torch.Tensor],
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Return exact top-k undirected retrieval candidates without self edges."""
        query = encoded["query"]
        num_nodes = query.size(0)
        if num_nodes < 2:
            return torch.empty((2, 0), dtype=torch.long, device=query.device)
        resolved_top_k = min(max(1, top_k or self.retrieval_top_k), num_nodes - 1)
        scores = self.retrieval_score_matrix(encoded).clone()
        scores.fill_diagonal_(torch.finfo(scores.dtype).min)
        partners = torch.topk(scores, k=resolved_top_k, dim=1).indices
        sources = torch.arange(num_nodes, device=query.device).unsqueeze(1).expand_as(partners)
        raw_pairs = torch.stack([sources.reshape(-1), partners.reshape(-1)], dim=0)
        raw_pairs = raw_pairs[:, raw_pairs[0] != raw_pairs[1]]
        canonical = torch.stack(
            [
                torch.minimum(raw_pairs[0], raw_pairs[1]),
                torch.maximum(raw_pairs[0], raw_pairs[1]),
            ],
            dim=0,
        )
        if canonical.numel() == 0:
            return canonical
        return cast(torch.Tensor, torch.unique(canonical.t(), dim=0).t().contiguous())

    def initialize_density_bias_with_prior(self, positive_edge_probability: float) -> float:
        """Initialize set-density bias to a sparse positive-edge prior."""
        final_linear = _last_linear(self.density_bias_head)
        if final_linear is None or final_linear.bias is None:
            raise ValueError("density_bias_head must expose a final linear layer with bias")
        clipped_probability = min(max(float(positive_edge_probability), 1.0e-8), 1.0 - 1.0e-8)
        bias_value = math.log(clipped_probability / (1.0 - clipped_probability))
        with torch.no_grad():
            final_linear.weight.zero_()
            final_linear.bias.fill_(bias_value)
        return bias_value

    @staticmethod
    def _symmetric_pair_features(
        node_a_embeddings: torch.Tensor,
        node_b_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Build unordered edge features for a protein-pair decoder."""
        cosine = functional.cosine_similarity(
            node_a_embeddings,
            node_b_embeddings,
            dim=-1,
        ).unsqueeze(-1)
        return torch.cat(
            [
                node_a_embeddings + node_b_embeddings,
                node_a_embeddings * node_b_embeddings,
                torch.abs(node_a_embeddings - node_b_embeddings),
                cosine,
            ],
            dim=-1,
        )

    def _decode_candidates(
        self,
        node_embeddings: torch.Tensor,
        candidate_pairs: torch.Tensor,
        encoded: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode sparse candidate edge logits from feature-only node embeddings."""
        candidate_count = candidate_pairs.size(1)
        set_state = node_embeddings.mean(dim=0, keepdim=True)
        density_bias = self.density_bias_head(set_state).squeeze()
        m_hat = self.edge_budget_from_node_embeddings(
            node_embeddings=node_embeddings,
            candidate_count=candidate_count,
        )
        module_memberships = (
            encoded["module"]
            if encoded is not None
            else functional.softmax(self.module_head(node_embeddings), dim=-1)
        )
        degree_prediction = (
            encoded["degree"]
            if encoded is not None
            else functional.softplus(self.degree_head(node_embeddings).squeeze(-1))
        )
        if candidate_count == 0:
            return {
                "logits": node_embeddings.new_zeros((0,)),
                "retrieval_logits": node_embeddings.new_zeros((0,)),
                "edge_probabilities": node_embeddings.new_zeros((0,)),
                "m_hat": m_hat,
                "module_memberships": module_memberships,
                "degree_predictions": degree_prediction,
                "density_bias": density_bias.reshape(()),
            }

        src = candidate_pairs[0]
        dst = candidate_pairs[1]
        h_src = node_embeddings[src]
        h_dst = node_embeddings[dst]
        pair_features = self._symmetric_pair_features(h_src, h_dst)
        pair_score = self.pair_mlp(pair_features).squeeze(-1)
        hub_score = self.hub_score_gate * (
            self.hub_head(h_src).squeeze(-1) + self.hub_head(h_dst).squeeze(-1)
        )
        lowrank = self.lowrank_head(node_embeddings)
        lowrank_score = self.lowrank_score_gate * (
            (lowrank[src] * lowrank[dst]).sum(dim=-1) / math.sqrt(float(self.lowrank_dim))
        )
        module_src = module_memberships[src]
        module_dst = module_memberships[dst]
        module_score = self.module_score_gate * self._centered_module_score(
            module_src,
            module_dst,
        )
        retrieval_matrix = (
            self.retrieval_score_matrix(encoded)
            if encoded is not None
            else self._node_only_retrieval_score_matrix(
                node_embeddings=node_embeddings,
                degree_prediction=degree_prediction,
            )
        )
        retrieval_logits = retrieval_matrix[src, dst]
        zero_component = retrieval_logits * 0.0
        if self.decoder_mode == "retrieval_only":
            logits = self.retrieval_logit_gate * retrieval_logits + density_bias
            return {
                "logits": logits,
                "retrieval_logits": retrieval_logits,
                "edge_probabilities": torch.sigmoid(logits),
                "m_hat": m_hat.reshape(()),
                "module_memberships": module_memberships,
                "degree_predictions": degree_prediction,
                "density_bias": density_bias.reshape(()),
                "pair_score": zero_component,
                "retrieval_score": retrieval_logits,
                "hub_score": zero_component,
                "lowrank_score": zero_component,
                "module_score": zero_component,
            }
        logits = (
            self.retrieval_logit_gate * retrieval_logits
            + pair_score
            + hub_score
            + lowrank_score
            + module_score
            + density_bias
        )
        return {
            "logits": logits,
            "retrieval_logits": retrieval_logits,
            "edge_probabilities": torch.sigmoid(logits),
            "m_hat": m_hat.reshape(()),
            "module_memberships": module_memberships,
            "degree_predictions": degree_prediction,
            "density_bias": density_bias.reshape(()),
            "pair_score": pair_score,
            "retrieval_score": retrieval_logits,
            "hub_score": hub_score,
            "lowrank_score": lowrank_score,
            "module_score": module_score,
        }

    def _node_only_retrieval_score_matrix(
        self,
        *,
        node_embeddings: torch.Tensor,
        degree_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Return retrieval scores when only graph node embeddings are available."""
        query = self._maybe_normalize(self.query_head(node_embeddings))
        key = self._maybe_normalize(self.key_head(node_embeddings))
        struct = self._maybe_normalize(self.struct_head(node_embeddings))
        directed = torch.matmul(query, key.t()) / math.sqrt(float(self.retrieval_dim))
        struct_score = torch.matmul(struct, struct.t()) / math.sqrt(float(self.retrieval_dim))
        degree_score = torch.log1p(degree_prediction).unsqueeze(0) + torch.log1p(
            degree_prediction
        ).unsqueeze(1)
        return (
            0.5 * (directed + directed.t())
            + self.struct_score_gate * struct_score
            + self.degree_score_gate * degree_score
        )

    def decode_graph_candidates(
        self,
        *,
        node_embeddings: torch.Tensor,
        candidate_pairs: torch.Tensor,
        encoded: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode graph candidate edges from precomputed node embeddings."""
        pairs = self._validate_candidate_pairs(
            candidate_pairs,
            num_nodes=node_embeddings.size(0),
            device=node_embeddings.device,
        )
        decoded = self._decode_candidates(node_embeddings, pairs, encoded=encoded)
        decoded["candidate_pairs"] = pairs
        return decoded

    def _decode_pairwise_edges(
        self,
        *,
        node_a_embeddings: torch.Tensor,
        node_b_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Decode independent pairwise logits without batch-level graph context."""
        pair_state = 0.5 * (node_a_embeddings + node_b_embeddings)
        density_bias = self.density_bias_head(pair_state).squeeze(-1)
        pair_features = self._symmetric_pair_features(node_a_embeddings, node_b_embeddings)
        pair_score = self.pair_mlp(pair_features).squeeze(-1)
        hub_score = self.hub_score_gate * (
            self.hub_head(node_a_embeddings).squeeze(-1)
            + self.hub_head(node_b_embeddings).squeeze(-1)
        )
        lowrank_a = self.lowrank_head(node_a_embeddings)
        lowrank_b = self.lowrank_head(node_b_embeddings)
        lowrank_score = self.lowrank_score_gate * (
            (lowrank_a * lowrank_b).sum(dim=-1) / math.sqrt(float(self.lowrank_dim))
        )
        module_a = functional.softmax(self.module_head(node_a_embeddings), dim=-1)
        module_b = functional.softmax(self.module_head(node_b_embeddings), dim=-1)
        module_score = self.module_score_gate * self._centered_module_score(module_a, module_b)
        degree_a = functional.softplus(self.degree_head(node_a_embeddings).squeeze(-1))
        degree_b = functional.softplus(self.degree_head(node_b_embeddings).squeeze(-1))
        query_a = self._maybe_normalize(self.query_head(node_a_embeddings))
        key_b = self._maybe_normalize(self.key_head(node_b_embeddings))
        query_b = self._maybe_normalize(self.query_head(node_b_embeddings))
        key_a = self._maybe_normalize(self.key_head(node_a_embeddings))
        retrieval_score = 0.5 * (
            (query_a * key_b).sum(dim=-1) + (query_b * key_a).sum(dim=-1)
        ) / math.sqrt(float(self.retrieval_dim))
        struct_a = self._maybe_normalize(self.struct_head(node_a_embeddings))
        struct_b = self._maybe_normalize(self.struct_head(node_b_embeddings))
        retrieval_score = retrieval_score + self.struct_score_gate * (
            (struct_a * struct_b).sum(dim=-1) / math.sqrt(float(self.retrieval_dim))
        )
        retrieval_score = retrieval_score + self.degree_score_gate * (
            torch.log1p(degree_a) + torch.log1p(degree_b)
        )
        return cast(
            torch.Tensor,
            self.retrieval_logit_gate * retrieval_score
            + pair_score
            + hub_score
            + lowrank_score
            + module_score
            + density_bias,
        )

    def _centered_module_score(
        self,
        module_a: torch.Tensor,
        module_b: torch.Tensor,
    ) -> torch.Tensor:
        """Return module compatibility centered at neutral uniform membership."""
        raw_score = (module_a @ self.module_interactions * module_b).sum(dim=-1)
        neutral_score = 1.0 / float(self.num_modules)
        return raw_score - raw_score.new_tensor(neutral_score)

    def estimate_edge_budget(
        self,
        *,
        protein_embeddings: torch.Tensor,
        protein_lengths: torch.Tensor | None = None,
        candidate_count: int,
    ) -> torch.Tensor:
        """Estimate the graph edge budget without decoding candidate logits."""
        node_embeddings = self.encode_graph_nodes(
            protein_embeddings=protein_embeddings,
            protein_lengths=protein_lengths,
        )
        return self.edge_budget_from_node_embeddings(
            node_embeddings=node_embeddings,
            candidate_count=candidate_count,
        )

    def _refine_once(
        self,
        *,
        node_embeddings: torch.Tensor,
        candidate_pairs: torch.Tensor,
        edge_probabilities: torch.Tensor,
    ) -> torch.Tensor:
        """Run one predicted-soft-graph refinement pass."""
        if candidate_pairs.size(1) == 0:
            return node_embeddings
        if self.message_projection is None or self.refinement_cell is None:
            raise RuntimeError("self refinement modules are not initialized")
        src = candidate_pairs[0]
        dst = candidate_pairs[1]
        messages = node_embeddings.new_zeros(node_embeddings.shape)
        weighted_src = edge_probabilities.unsqueeze(-1) * self.message_projection(
            node_embeddings[src]
        )
        weighted_dst = edge_probabilities.unsqueeze(-1) * self.message_projection(
            node_embeddings[dst]
        )
        messages.index_add_(0, dst, weighted_src)
        messages.index_add_(0, src, weighted_dst)
        return cast(torch.Tensor, self.refinement_cell(messages, node_embeddings))

    def forward_graph(
        self,
        *,
        protein_embeddings: torch.Tensor,
        protein_lengths: torch.Tensor | None = None,
        candidate_pairs: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Generate candidate edge probabilities for a protein set from features only."""
        if protein_embeddings.size(-1) != self.input_dim:
            raise ValueError("protein embedding dimension must match model_config.input_dim")
        encoded = self.encode_proteins(
            protein_embeddings=protein_embeddings,
            protein_lengths=protein_lengths,
        )
        node_embeddings = encoded["node"]
        pairs = (
            self._all_pairs(node_embeddings.size(0), node_embeddings.device)
            if candidate_pairs is None
            else self._validate_candidate_pairs(
                candidate_pairs,
                num_nodes=node_embeddings.size(0),
                device=node_embeddings.device,
            )
        )
        decoded: dict[str, object] = dict(
            self._decode_candidates(node_embeddings, pairs, encoded=encoded)
        )
        if self.self_refinement_rounds == 1:
            refined_embeddings = self._refine_once(
                node_embeddings=node_embeddings,
                candidate_pairs=pairs,
                edge_probabilities=cast(torch.Tensor, decoded["edge_probabilities"]),
            )
            decoded = dict(self._decode_candidates(refined_embeddings, pairs))
            node_embeddings = refined_embeddings
        decoded["candidate_pairs"] = pairs
        decoded["node_embeddings"] = node_embeddings
        decoded["retrieval_score_matrix"] = self.retrieval_score_matrix(encoded)
        decoded["encoded"] = encoded
        return decoded

    def forward(
        self,
        *,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        len_a: torch.Tensor | None = None,
        len_b: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Run pairwise scoring through the repository model contract."""
        if emb_a.dim() != 3 or emb_b.dim() != 3:
            raise ValueError("Input embeddings must be shaped (batch, seq_len, embedding_dim)")
        if emb_a.size(0) != emb_b.size(0):
            raise ValueError("Protein pair batches must have matching batch dimension")
        pooled_a = self._masked_mean_pool(emb_a, len_a)
        pooled_b = self._masked_mean_pool(emb_b, len_b)
        h_a = cast(torch.Tensor, self.protein_projection(pooled_a))
        h_b = cast(torch.Tensor, self.protein_projection(pooled_b))
        set_state = 0.5 * (h_a + h_b)
        context_a = cast(torch.Tensor, self.set_context(torch.cat([h_a, set_state], dim=-1)))
        context_b = cast(torch.Tensor, self.set_context(torch.cat([h_b, set_state], dim=-1)))
        logits = self._decode_pairwise_edges(
            node_a_embeddings=context_a,
            node_b_embeddings=context_b,
        )
        result = {"logits": logits}
        if label is not None:
            labels = label.to(device=logits.device, dtype=logits.dtype).reshape_as(logits)
            result["loss"] = functional.binary_cross_entropy_with_logits(logits, labels)
        return result
