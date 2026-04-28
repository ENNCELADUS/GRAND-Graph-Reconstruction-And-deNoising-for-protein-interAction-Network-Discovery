"""V3.1 model — V3 with richer per-protein pooling before interaction head.

Architecture changes vs V3:
- ``RichPooling`` module: CLS + mean + attention + max pooling fused via a
  learned soft gate, then projected back to ``d_model``.
- ``InteractionCrossAttention`` applies ``RichPooling`` independently to the
  final ``h_a`` and ``h_b`` hidden states, then fuses them with the CLS token
  via a learned projection.  Output dimensionality is unchanged (``d_model``),
  so the ``MLPHead`` and all downstream config keys are identical to V3.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

# Re-use shared building blocks from v3 directly to avoid duplication.
from src.model.v3 import (
    MLPHead,
    SiameseEncoder,
    _build_padding_mask,
    _to_float,
    _to_int,
    _to_mapping,
)

POOLING_BASE_COMPONENTS = ("esm_cls", "mean", "attn", "max")
POOLING_GATED_COMPONENT = "gated"
DEFAULT_RICH_POOLING_COMPONENTS = (*POOLING_BASE_COMPONENTS, POOLING_GATED_COMPONENT)


def _parse_rich_pooling_components(value: object) -> tuple[str, ...]:
    """Parse enabled rich-pooling components in canonical order."""
    if value is None:
        return DEFAULT_RICH_POOLING_COMPONENTS
    if not isinstance(value, list) or not value:
        raise ValueError("model_config.rich_pooling.components must be a non-empty list")

    raw_components: set[str] = set()
    valid_components = set(DEFAULT_RICH_POOLING_COMPONENTS)
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("model_config.rich_pooling.components must contain strings")
        component = item.strip().lower()
        if component not in valid_components:
            raise ValueError(
                "model_config.rich_pooling.components contains unsupported component: "
                f"{component}"
            )
        if component in raw_components:
            raise ValueError(
                f"model_config.rich_pooling.components contains duplicate: {component}"
            )
        raw_components.add(component)

    base_components = tuple(
        component for component in POOLING_BASE_COMPONENTS if component in raw_components
    )
    if not base_components:
        raise ValueError("model_config.rich_pooling.components must enable a base component")
    if POOLING_GATED_COMPONENT in raw_components and not base_components:
        raise ValueError("model_config.rich_pooling.components cannot enable gated alone")

    components = list(base_components)
    if POOLING_GATED_COMPONENT in raw_components:
        components.append(POOLING_GATED_COMPONENT)
    return tuple(components)


# ---------------------------------------------------------------------------
# Cross-attention layer (identical to V3 — shared weights, bidirectional)
# ---------------------------------------------------------------------------


class CrossAttentionLayer(nn.Module):
    """Shared-weight bidirectional cross-attention with FFN and CLS pooling."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.norm_cls_attn = nn.LayerNorm(d_model)
        self.norm_cls_ffn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.attn_cls = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.ff_cls = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop_attn = nn.Dropout(dropout)
        self.drop_ffn = nn.Dropout(dropout)
        self.drop_cls_attn = nn.Dropout(dropout)
        self.drop_cls_ffn = nn.Dropout(dropout)

    def _attend(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        query_norm = self.norm_attn(query)
        attn_out, _ = self.attn(query_norm, key_value, key_value, key_padding_mask=key_padding_mask)
        return query + cast(torch.Tensor, self.drop_attn(attn_out))

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward sub-layer with residual connection."""
        return x + cast(torch.Tensor, self.drop_ffn(self.ffn(self.norm_ffn(x))))

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        cls_token: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one bidirectional cross-attention block.

        Args:
            h_a: Protein A hidden states.
            h_b: Protein B hidden states.
            cls_token: CLS token state.
            mask_a: Padding mask for sequence A.
            mask_b: Padding mask for sequence B.

        Returns:
            Updated ``(h_a, h_b, cls_token)`` tuple.
        """
        h_a = self._attend(h_a, h_b, mask_b)
        h_a = self._ffn(h_a)
        h_b = self._attend(h_b, h_a, mask_a)
        h_b = self._ffn(h_b)

        combined = torch.cat([h_a, h_b], dim=1)
        combined_mask = (
            torch.cat([mask_a, mask_b], dim=1)
            if mask_a is not None and mask_b is not None
            else None
        )

        cls_norm = self.norm_cls_attn(cls_token)
        attn_cls, _ = self.attn_cls(cls_norm, combined, combined, key_padding_mask=combined_mask)
        cls_token = cls_token + self.drop_cls_attn(attn_cls)
        cls_token = cls_token + self.drop_cls_ffn(self.ff_cls(self.norm_cls_ffn(cls_token)))

        return h_a, h_b, cls_token


# ---------------------------------------------------------------------------
# Rich pooling
# ---------------------------------------------------------------------------


class RichPooling(nn.Module):
    """Configurable ESM-CLS and residue pooling with optional gated fusion.

    ``esm_cls`` uses the first ESM3 special-token embedding. Residue pooling
    components use positions between the first BOS token and final EOS token.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float,
        components: tuple[str, ...] = DEFAULT_RICH_POOLING_COMPONENTS,
    ) -> None:
        super().__init__()
        self.components = _parse_rich_pooling_components(list(components))
        self.base_components = tuple(
            component for component in POOLING_BASE_COMPONENTS if component in self.components
        )
        self.use_gated = POOLING_GATED_COMPONENT in self.components
        self.attn_scorer = nn.Linear(d_model, 1) if "attn" in self.base_components else None
        self.pool_gate = (
            nn.Linear(d_model * len(self.base_components), len(self.base_components))
            if self.use_gated
            else None
        )
        projection_components = len(self.base_components) + int(self.use_gated)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model * projection_components),
            nn.Linear(d_model * projection_components, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _residue_mask(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Return a mask for non-special residue tokens."""
        batch_size, sequence_length, _ = x.shape
        if padding_mask is not None:
            valid_mask = ~padding_mask
        else:
            valid_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            )

        positions = torch.arange(sequence_length, device=x.device).unsqueeze(0)
        valid_lengths = valid_mask.sum(dim=1, keepdim=True)
        residue_mask = valid_mask & (positions > 0) & (positions < valid_lengths - 1)
        if bool((residue_mask.sum(dim=1) == 0).any()):
            raise ValueError("RichPooling requires at least one residue token between BOS and EOS")
        return residue_mask

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Pool token sequence to a single ``d_model`` vector.

        Args:
            x: Token hidden states ``(batch, seq_len, d_model)``.
            padding_mask: Boolean mask ``(batch, seq_len)`` — ``True`` = pad.

        Returns:
            Pooled vector ``(batch, d_model)``.
        """
        pooled_by_component: dict[str, torch.Tensor] = {}
        if "esm_cls" in self.base_components:
            pooled_by_component["esm_cls"] = x[:, 0]

        if any(component in self.base_components for component in ("mean", "attn", "max")):
            residue_mask = self._residue_mask(x=x, padding_mask=padding_mask)
            residue_float_mask = residue_mask.float()
            residue_3d = residue_float_mask.unsqueeze(-1)

            if "mean" in self.base_components:
                pooled_by_component["mean"] = (x * residue_3d).sum(dim=1) / residue_3d.sum(
                    dim=1
                ).clamp_min(1.0)

            if "attn" in self.base_components:
                if self.attn_scorer is None:
                    raise RuntimeError(
                        "attn_scorer must be initialized when attention pooling is enabled"
                    )
                scores = self.attn_scorer(x).squeeze(-1)
                scores = scores.masked_fill(~residue_mask, torch.finfo(scores.dtype).min)
                attn_weights = torch.softmax(scores, dim=1).unsqueeze(-1)
                pooled_by_component["attn"] = (x * attn_weights).sum(dim=1)

            if "max" in self.base_components:
                masked_x = x.masked_fill(~residue_mask.unsqueeze(-1), torch.finfo(x.dtype).min)
                pooled_by_component["max"] = masked_x.max(dim=1).values

        base_vectors = [pooled_by_component[component] for component in self.base_components]
        output_vectors = list(base_vectors)
        if self.use_gated:
            if self.pool_gate is None:
                raise RuntimeError("pool_gate must be initialized when gated pooling is enabled")
            gate_input = torch.cat(base_vectors, dim=1)
            gate_weights = torch.softmax(self.pool_gate(gate_input), dim=1).unsqueeze(-1)
            pooled_stack = torch.stack(base_vectors, dim=1)
            output_vectors.append((pooled_stack * gate_weights).sum(dim=1))

        combined = torch.cat(output_vectors, dim=1)
        return cast(torch.Tensor, self.proj(combined))


# ---------------------------------------------------------------------------
# Interaction cross-attention with rich pooling
# ---------------------------------------------------------------------------


class InteractionCrossAttention(nn.Module):
    """Stacked cross-attention encoder with rich CLS + gated pooling."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        pooling_components: tuple[str, ...] = DEFAULT_RICH_POOLING_COMPONENTS,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            CrossAttentionLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        )
        shared_pool = RichPooling(d_model=d_model, dropout=dropout, components=pooling_components)
        self.pool_a = shared_pool
        self.pool_b = shared_pool
        # Fuse CLS + pooled_a + pooled_b → d_model
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        """Pool pair representation from encoded proteins.

        Args:
            h_a: Protein A hidden states ``(batch, seq_len_a, d_model)``.
            h_b: Protein B hidden states ``(batch, seq_len_b, d_model)``.
            lengths_a: Sequence lengths for A ``(batch,)``.
            lengths_b: Sequence lengths for B ``(batch,)``.

        Returns:
            Fused pair representation ``(batch, d_model)``.
        """
        if h_a.dim() != 3 or h_b.dim() != 3:
            raise ValueError(
                "Cross-attention inputs must have shape (batch_size, seq_len, d_model)"
            )
        if h_a.size(0) != h_b.size(0):
            raise ValueError("Protein pair batches must have matching batch dimension")

        batch_size = h_a.size(0)
        mask_a = _build_padding_mask(lengths_a, h_a.size(1))
        mask_b = _build_padding_mask(lengths_b, h_b.size(1))

        cls_token = self.cls_token.repeat(batch_size, 1, 1)
        for layer in self.layers:
            h_a, h_b, cls_token = layer(h_a, h_b, cls_token, mask_a, mask_b)

        pooled_a = self.pool_a(h_a, mask_a)
        pooled_b = self.pool_b(h_b, mask_b)
        cls_vec = cls_token.squeeze(1)

        fused = torch.cat([cls_vec, pooled_a, pooled_b], dim=1)
        return cast(torch.Tensor, self.fusion(fused))


# ---------------------------------------------------------------------------
# V3_1 top-level model
# ---------------------------------------------------------------------------


class V3_1(nn.Module):
    """V3.1 model — V3 with rich per-protein pooling (CLS+mean+attn+max+gate).

    Config keys are identical to V3; no new required fields.
    """

    name: str = "v3.1"

    def __init__(self, **model_config: object) -> None:
        super().__init__()
        required_fields = [
            "input_dim",
            "d_model",
            "encoder_layers",
            "cross_attn_layers",
            "n_heads",
        ]
        missing = [f for f in required_fields if f not in model_config]
        if missing:
            raise ValueError(f"Missing required model configuration fields: {missing}")

        self.input_dim = _to_int(model_config["input_dim"], "model_config.input_dim")
        self.d_model = _to_int(model_config["d_model"], "model_config.d_model")
        self.encoder_layers = _to_int(model_config["encoder_layers"], "model_config.encoder_layers")
        self.cross_attn_layers = _to_int(
            model_config["cross_attn_layers"], "model_config.cross_attn_layers"
        )
        self.n_heads = _to_int(model_config["n_heads"], "model_config.n_heads")

        mlp_cfg_raw = model_config.get("mlp_head")
        if not isinstance(mlp_cfg_raw, dict) or not mlp_cfg_raw:
            raise ValueError("mlp_head configuration is required for V3_1")
        mlp_cfg = _to_mapping(mlp_cfg_raw, "model_config.mlp_head")
        if "hidden_dims" not in mlp_cfg or "dropout" not in mlp_cfg:
            raise ValueError("mlp_head.hidden_dims and mlp_head.dropout must be provided")
        hidden_dims_raw = mlp_cfg["hidden_dims"]
        if not isinstance(hidden_dims_raw, list) or not hidden_dims_raw:
            raise ValueError("mlp_head.hidden_dims must be a non-empty list")
        self.mlp_hidden_dims = [
            _to_int(v, "model_config.mlp_head.hidden_dims") for v in hidden_dims_raw
        ]
        self.mlp_dropout = _to_float(mlp_cfg["dropout"], "model_config.mlp_head.dropout")
        self.mlp_activation = str(mlp_cfg.get("activation", "gelu"))
        self.mlp_norm = str(mlp_cfg.get("norm", "layernorm"))

        reg_cfg_raw = model_config.get("regularization")
        if not isinstance(reg_cfg_raw, dict) or "dropout" not in reg_cfg_raw:
            raise ValueError("regularization.dropout must be provided for V3_1")
        reg_cfg = _to_mapping(reg_cfg_raw, "model_config.regularization")
        self.encoder_dropout = _to_float(reg_cfg["dropout"], "model_config.regularization.dropout")
        self.cross_attention_dropout = _to_float(
            reg_cfg.get("cross_attention_dropout", self.encoder_dropout),
            "model_config.regularization.cross_attention_dropout",
        )
        self.token_dropout = _to_float(
            reg_cfg.get("token_dropout", 0.0), "model_config.regularization.token_dropout"
        )
        self.stochastic_depth = _to_float(
            reg_cfg.get("stochastic_depth", 0.0), "model_config.regularization.stochastic_depth"
        )
        rich_pooling_cfg_raw = model_config.get("rich_pooling", {})
        if rich_pooling_cfg_raw is None:
            rich_pooling_cfg_raw = {}
        if not isinstance(rich_pooling_cfg_raw, dict):
            raise ValueError("model_config.rich_pooling must be a mapping")
        rich_pooling_cfg = _to_mapping(rich_pooling_cfg_raw, "model_config.rich_pooling")
        self.rich_pooling_components = _parse_rich_pooling_components(
            rich_pooling_cfg.get("components")
        )

        self.encoder = SiameseEncoder(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_layers=self.encoder_layers,
            n_heads=self.n_heads,
            dropout=self.encoder_dropout,
            token_dropout=self.token_dropout,
            stochastic_depth=self.stochastic_depth,
        )
        self.cross_attention = InteractionCrossAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.cross_attn_layers,
            dropout=self.cross_attention_dropout,
            pooling_components=self.rich_pooling_components,
        )
        self.output_head = MLPHead(
            input_dim=self.d_model,
            hidden_dims=self.mlp_hidden_dims,
            output_dim=1,
            dropout=self.mlp_dropout,
            activation=self.mlp_activation,
            norm=self.mlp_norm,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor] | None = None,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run a model forward pass.

        Args:
            batch: Optional batch dictionary.
            **kwargs: Additional batch tensors merged into ``batch``.

        Returns:
            Output dictionary containing ``logits`` and optional ``loss``.
        """
        merged: dict[str, torch.Tensor] = {}
        if batch is not None:
            merged.update(batch)
        merged.update(kwargs)

        if "emb_a" not in merged or "emb_b" not in merged:
            raise KeyError("Batch must contain 'emb_a' and 'emb_b' tensors")

        emb_a = merged["emb_a"]
        emb_b = merged["emb_b"]
        if emb_a.dim() != 3 or emb_b.dim() != 3:
            raise ValueError("Input embeddings must be shaped (batch, seq_len, embedding_dim)")
        if emb_a.size(2) != self.input_dim or emb_b.size(2) != self.input_dim:
            raise ValueError("Input embedding dimension must match model input_dim")
        if emb_a.size(0) != emb_b.size(0):
            raise ValueError("Protein pair batches must have matching batch dimension")

        device = emb_a.device
        lengths_a = merged.get("len_a")
        lengths_b = merged.get("len_b")
        if lengths_a is None:
            lengths_a = torch.full((emb_a.size(0),), emb_a.size(1), device=device, dtype=torch.long)
        else:
            lengths_a = lengths_a.to(device=device, dtype=torch.long)
        if lengths_b is None:
            lengths_b = torch.full((emb_b.size(0),), emb_b.size(1), device=device, dtype=torch.long)
        else:
            lengths_b = lengths_b.to(device=device, dtype=torch.long)

        encoded_a = self.encoder(emb_a, lengths_a)
        encoded_b = self.encoder(emb_b, lengths_b)
        cls_repr = self.cross_attention(encoded_a, encoded_b, lengths_a, lengths_b)
        logits = self.output_head(cls_repr)

        output: dict[str, torch.Tensor] = {"logits": logits}
        if "label" in merged:
            labels = merged["label"].float()
            logits_for_loss = (
                logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            )
            labels_for_loss = (
                labels.squeeze(-1) if labels.dim() > 1 and labels.size(-1) == 1 else labels
            )
            output["loss"] = nn.functional.binary_cross_entropy_with_logits(
                logits_for_loss, labels_for_loss
            )

        return output
