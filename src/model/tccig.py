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
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc


def _to_float(value: object, field_name: str) -> float:
    """Parse a floating-point model configuration value."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a float")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a float") from exc


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


class TCCIG(nn.Module):
    """Topology-constrained conditional interactome generator.

    The graph forward path accepts only protein-intrinsic embeddings and optional
    feature-derived candidate pairs. Training labels and topology targets are
    consumed by the pipeline loss code, not by this model hook.
    """

    name: str = "tccig"

    def __init__(self, **model_config: object) -> None:
        super().__init__()
        self.input_dim = _to_int(model_config["input_dim"], "model_config.input_dim")
        self.d_model = _to_int(model_config["d_model"], "model_config.d_model")
        self.dropout = _to_float(model_config.get("dropout", 0.1), "model_config.dropout")
        self.lowrank_dim = _to_int(
            model_config.get("lowrank_dim", max(1, self.d_model // 4)),
            "model_config.lowrank_dim",
        )
        self.num_modules = _to_int(
            model_config.get("num_modules", 64),
            "model_config.num_modules",
        )
        self.self_refinement_rounds = _to_int(
            model_config.get("self_refinement_rounds", 0),
            "model_config.self_refinement_rounds",
        )
        if self.self_refinement_rounds not in {0, 1}:
            raise ValueError("model_config.self_refinement_rounds must be 0 or 1")

        candidate_cfg = _to_mapping(
            model_config.get("candidate_proposer", {"type": "all_pairs"}),
            "model_config.candidate_proposer",
        )
        self.candidate_proposer_type = _to_string(
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

        self.protein_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.set_context = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.pair_mlp = _build_mlp(
            input_dim=(3 * self.d_model) + 1,
            hidden_dims=pair_hidden_dims,
            output_dim=1,
            dropout=self.dropout,
        )
        self.hub_head = nn.Linear(self.d_model, 1)
        self.lowrank_head = nn.Linear(self.d_model, self.lowrank_dim)
        self.module_head = nn.Linear(self.d_model, self.num_modules)
        self.module_interactions = nn.Parameter(torch.eye(self.num_modules))
        self.density_bias_head = _build_mlp(
            input_dim=self.d_model,
            hidden_dims=[self.d_model],
            output_dim=1,
            dropout=self.dropout,
        )
        self.edge_budget_head = _build_mlp(
            input_dim=self.d_model,
            hidden_dims=[self.d_model],
            output_dim=1,
            dropout=self.dropout,
        )
        if self.self_refinement_rounds == 1:
            self.message_projection = nn.Linear(self.d_model, self.d_model)
            self.refinement_cell = nn.GRUCell(self.d_model, self.d_model)

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
    ) -> dict[str, torch.Tensor]:
        """Decode sparse candidate edge logits from feature-only node embeddings."""
        candidate_count = candidate_pairs.size(1)
        set_state = node_embeddings.mean(dim=0, keepdim=True)
        density_bias = self.density_bias_head(set_state).squeeze()
        m_hat = self.edge_budget_from_node_embeddings(
            node_embeddings=node_embeddings,
            candidate_count=candidate_count,
        )
        module_memberships = functional.softmax(self.module_head(node_embeddings), dim=-1)
        if candidate_count == 0:
            return {
                "logits": node_embeddings.new_zeros((0,)),
                "edge_probabilities": node_embeddings.new_zeros((0,)),
                "m_hat": m_hat,
                "module_memberships": module_memberships,
                "density_bias": density_bias.reshape(()),
            }

        src = candidate_pairs[0]
        dst = candidate_pairs[1]
        h_src = node_embeddings[src]
        h_dst = node_embeddings[dst]
        pair_features = self._symmetric_pair_features(h_src, h_dst)
        pair_score = self.pair_mlp(pair_features).squeeze(-1)
        hub_score = self.hub_head(h_src).squeeze(-1) + self.hub_head(h_dst).squeeze(-1)
        lowrank = self.lowrank_head(node_embeddings)
        lowrank_score = (lowrank[src] * lowrank[dst]).sum(dim=-1)
        module_src = module_memberships[src]
        module_dst = module_memberships[dst]
        module_score = self._centered_module_score(module_src, module_dst)
        logits = pair_score + hub_score + lowrank_score + module_score + density_bias
        return {
            "logits": logits,
            "edge_probabilities": torch.sigmoid(logits),
            "m_hat": m_hat.reshape(()),
            "module_memberships": module_memberships,
            "density_bias": density_bias.reshape(()),
            "pair_score": pair_score,
            "hub_score": hub_score,
            "lowrank_score": lowrank_score,
            "module_score": module_score,
        }

    def decode_graph_candidates(
        self,
        *,
        node_embeddings: torch.Tensor,
        candidate_pairs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode graph candidate edges from precomputed node embeddings."""
        pairs = self._validate_candidate_pairs(
            candidate_pairs,
            num_nodes=node_embeddings.size(0),
            device=node_embeddings.device,
        )
        decoded = self._decode_candidates(node_embeddings, pairs)
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
        hub_score = self.hub_head(node_a_embeddings).squeeze(-1) + self.hub_head(
            node_b_embeddings
        ).squeeze(-1)
        lowrank_a = self.lowrank_head(node_a_embeddings)
        lowrank_b = self.lowrank_head(node_b_embeddings)
        lowrank_score = (lowrank_a * lowrank_b).sum(dim=-1)
        module_a = functional.softmax(self.module_head(node_a_embeddings), dim=-1)
        module_b = functional.softmax(self.module_head(node_b_embeddings), dim=-1)
        module_score = self._centered_module_score(module_a, module_b)
        return pair_score + hub_score + lowrank_score + module_score + density_bias

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
        return self.refinement_cell(messages, node_embeddings)

    def forward_graph(
        self,
        *,
        protein_embeddings: torch.Tensor,
        protein_lengths: torch.Tensor | None = None,
        candidate_pairs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate candidate edge probabilities for a protein set from features only."""
        if protein_embeddings.size(-1) != self.input_dim:
            raise ValueError("protein embedding dimension must match model_config.input_dim")
        node_embeddings = self._encode_nodes(protein_embeddings, protein_lengths)
        pairs = (
            self._all_pairs(node_embeddings.size(0), node_embeddings.device)
            if candidate_pairs is None
            else self._validate_candidate_pairs(
                candidate_pairs,
                num_nodes=node_embeddings.size(0),
                device=node_embeddings.device,
            )
        )
        decoded = self._decode_candidates(node_embeddings, pairs)
        if self.self_refinement_rounds == 1:
            refined_embeddings = self._refine_once(
                node_embeddings=node_embeddings,
                candidate_pairs=pairs,
                edge_probabilities=decoded["edge_probabilities"],
            )
            decoded = self._decode_candidates(refined_embeddings, pairs)
            node_embeddings = refined_embeddings
        decoded["candidate_pairs"] = pairs
        decoded["node_embeddings"] = node_embeddings
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
