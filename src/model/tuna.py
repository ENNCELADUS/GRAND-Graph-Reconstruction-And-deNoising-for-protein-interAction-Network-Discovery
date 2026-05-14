"""TUnA-style small transformer head for frozen protein embeddings.

This module ports the core TUnA architecture into the repository model
contract: cached ESM embeddings in, binary-interaction logits out.  The
downstream encoder follows the official TUnA design closely: spectral-normalized
self-attention, a small intra-protein encoder, an inter-protein encoder, and
AB/BA order aggregation by feature-wise max.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.model.v3 import _to_float, _to_int, _to_mapping

INTER_MASK_MODES = ("official_block", "cross_chain")
OUTPUT_HEAD_TYPES = ("linear", "sngp")


def _activation(name: str) -> nn.Module:
    """Return the activation module used by TUnA feed-forward blocks."""
    activation_name = name.lower()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "gelu":
        return nn.GELU()
    if activation_name == "elu":
        return nn.ELU()
    if activation_name == "swish":
        return nn.SiLU()
    if activation_name == "leaky_relu":
        return nn.LeakyReLU()
    if activation_name == "mish":
        return nn.Mish()
    raise ValueError(f"Unsupported activation_function: {name}")


def _to_bool(value: object, field_name: str) -> bool:
    """Convert a config value to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be bool-compatible")


def _xavier_initialize(module: nn.Module) -> None:
    """Initialize linear weights, including spectral-norm ``weight_orig``."""
    for parameter in module.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)


class SelfAttention(nn.Module):
    """Official TUnA-style spectral-normalized multi-head self-attention."""

    def __init__(self, hid_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if hid_dim % n_heads != 0:
            raise ValueError("hid_dim must be divisible by n_heads")
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.w_q = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.w_k = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.w_v = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.fc = spectral_norm(nn.Linear(hid_dim, hid_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply multi-head attention with a keep mask shaped ``(B, 1, Q, K)``."""
        batch_size = query.size(0)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(float(self.head_dim))
        if mask is not None:
            energy = energy.masked_fill(~mask, -1.0e4)
        attention = self.dropout(torch.softmax(energy, dim=-1))
        attended = torch.matmul(attention, v)
        attended = attended.permute(0, 2, 1, 3).contiguous()
        attended = attended.view(batch_size, -1, self.hid_dim)
        return cast(torch.Tensor, self.fc(attended))


class Feedforward(nn.Module):
    """Official TUnA feed-forward block with spectral-normalized linear layers."""

    def __init__(self, hid_dim: int, ff_dim: int, dropout: float, activation_function: str) -> None:
        super().__init__()
        self.fc_1 = spectral_norm(nn.Linear(hid_dim, ff_dim))
        self.fc_2 = spectral_norm(nn.Linear(ff_dim, hid_dim))
        self.dropout = nn.Dropout(dropout)
        self.activation = _activation(activation_function)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation."""
        return cast(torch.Tensor, self.fc_2(self.dropout(self.activation(self.fc_1(x)))))


class EncoderLayer(nn.Module):
    """TUnA encoder layer: self-attention followed by feed-forward residuals."""

    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        activation_function: str,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hid_dim)
        self.ln2 = nn.LayerNorm(hid_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.self_attention = SelfAttention(hid_dim=hid_dim, n_heads=n_heads, dropout=dropout)
        self.feedforward = Feedforward(
            hid_dim=hid_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            activation_function=activation_function,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run one TUnA encoder layer."""
        x = self.ln1(x + self.dropout1(self.self_attention(x, x, x, mask)))
        return cast(torch.Tensor, self.ln2(x + self.dropout2(self.feedforward(x))))


class IntraEncoder(nn.Module):
    """Shared intra-protein TUnA encoder."""

    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        activation_function: str,
    ) -> None:
        super().__init__()
        self.input_projection = spectral_norm(nn.Linear(input_dim, hid_dim))
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hid_dim=hid_dim,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    activation_function=activation_function,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Project embeddings and run intra-protein encoder layers."""
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class InterEncoder(nn.Module):
    """TUnA inter-protein encoder over concatenated A/B token states."""

    def __init__(
        self,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        activation_function: str,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    hid_dim=hid_dim,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    activation_function=activation_function,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        attention_mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the concatenated pair and mean-pool valid tokens."""
        combined = torch.cat([encoded_a, encoded_b], dim=1)
        for layer in self.layers:
            combined = layer(combined, attention_mask)
        token_weights = token_mask.to(dtype=combined.dtype).unsqueeze(-1)
        return cast(
            torch.Tensor,
            (combined * token_weights).sum(dim=1) / token_weights.sum(dim=1).clamp_min(1.0),
        )


class SpectralLinearHead(nn.Module):
    """Single spectral-normalized logit layer."""

    def __init__(self, hid_dim: int) -> None:
        super().__init__()
        self.projection = spectral_norm(nn.Linear(hid_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary logits."""
        return cast(torch.Tensor, self.projection(x))


class DiagonalRFFGaussianProcessHead(nn.Module):
    """RFF-GP classifier head with diagonal precision for cheap SNGP ablations."""

    def __init__(
        self,
        hid_dim: int,
        rffs: int,
        ridge_penalty: float,
        cov_momentum: float,
        random_seed: int,
        mean_field_factor: float = math.pi / 8.0,
    ) -> None:
        super().__init__()
        if rffs <= 0:
            raise ValueError("output_head.rffs must be >= 1")
        if ridge_penalty <= 0:
            raise ValueError("output_head.gp_ridge_penalty must be > 0")
        self.rffs = rffs
        self.ridge_penalty = ridge_penalty
        self.cov_momentum = cov_momentum
        self.mean_field_factor = mean_field_factor
        generator = torch.Generator()
        generator.manual_seed(random_seed)
        random_weight = torch.randn(hid_dim, rffs, generator=generator)
        random_bias = 2.0 * math.pi * torch.rand(rffs, generator=generator)
        self.register_buffer("random_weight", random_weight)
        self.register_buffer("random_bias", random_bias)
        self.beta = nn.Parameter(torch.empty(rffs, 1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("precision_diag", torch.full((rffs,), ridge_penalty))
        self.register_buffer("precision_seen", torch.zeros((), dtype=torch.long))
        nn.init.xavier_uniform_(self.beta)

    def reset_precision(self) -> None:
        """Reset the diagonal precision accumulator before a new epoch."""
        self.precision_diag.fill_(self.ridge_penalty)
        self.precision_seen.zero_()

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states to random Fourier features."""
        projected = x.float() @ self.random_weight.float() + self.random_bias.float()
        return cast(torch.Tensor, math.sqrt(2.0 / float(self.rffs)) * torch.cos(projected))

    def _update_precision(self, features: torch.Tensor) -> None:
        """Update diagonal precision from a training batch."""
        batch_update = features.detach().pow(2).sum(dim=0)
        if self.cov_momentum < 0.0 or int(self.precision_seen.item()) == 0:
            self.precision_diag.add_(batch_update)
        else:
            self.precision_diag.mul_(self.cov_momentum).add_(
                batch_update,
                alpha=1.0 - self.cov_momentum,
            )
        self.precision_seen.add_(1)

    def forward(self, x: torch.Tensor, *, update_precision: bool) -> torch.Tensor:
        """Return logits, mean-field-adjusted at eval once precision is available."""
        features = self._features(x)
        if update_precision and self.training and torch.is_grad_enabled():
            self._update_precision(features)
        logits = features @ self.beta.float() + self.bias.float()
        if not self.training and int(self.precision_seen.item()) > 0:
            variance = features.pow(2) @ (1.0 / self.precision_diag.float()).unsqueeze(-1)
            logits = logits / torch.sqrt(1.0 + self.mean_field_factor * variance)
        return logits.to(dtype=x.dtype)


class TUNA(nn.Module):
    """TUnA architecture adapted to cached ESM embeddings."""

    name: str = "tuna"

    def __init__(self, **model_config: object) -> None:
        super().__init__()
        self.input_dim = _to_int(model_config["input_dim"], "model_config.input_dim")
        self.hid_dim = _to_int(model_config.get("hid_dim", 64), "model_config.hid_dim")
        self.n_layers = _to_int(model_config.get("n_layers", 1), "model_config.n_layers")
        self.n_heads = _to_int(model_config.get("n_heads", 8), "model_config.n_heads")
        self.ff_dim = _to_int(model_config.get("ff_dim", 256), "model_config.ff_dim")
        self.dropout = _to_float(model_config.get("dropout", 0.2), "model_config.dropout")
        self.max_sequence_length = _to_int(
            model_config.get("max_sequence_length", 512),
            "model_config.max_sequence_length",
        )
        if self.max_sequence_length <= 0:
            raise ValueError("model_config.max_sequence_length must be >= 1")
        self.activation_function = str(model_config.get("activation_function", "swish")).lower()
        self.exclude_special_tokens = _to_bool(
            model_config.get("exclude_special_tokens", True),
            "model_config.exclude_special_tokens",
        )
        self.inter_mask_mode = str(model_config.get("inter_mask_mode", "official_block")).lower()
        if self.inter_mask_mode not in INTER_MASK_MODES:
            raise ValueError(
                "model_config.inter_mask_mode must be one of: "
                f"{', '.join(INTER_MASK_MODES)}"
            )

        output_head_raw = model_config.get("output_head", {"type": "linear"})
        output_head_cfg = _to_mapping(output_head_raw, "model_config.output_head")
        self.output_head_type = str(output_head_cfg.get("type", "linear")).lower()
        if self.output_head_type not in OUTPUT_HEAD_TYPES:
            raise ValueError(
                "model_config.output_head.type must be one of: "
                f"{', '.join(OUTPUT_HEAD_TYPES)}"
            )
        random_seed = _to_int(model_config.get("random_seed", 47), "model_config.random_seed")

        self.intra_encoder = IntraEncoder(
            input_dim=self.input_dim,
            hid_dim=self.hid_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            activation_function=self.activation_function,
        )
        self.inter_encoder = InterEncoder(
            hid_dim=self.hid_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            activation_function=self.activation_function,
        )
        if self.output_head_type == "sngp":
            self.output_head: nn.Module = DiagonalRFFGaussianProcessHead(
                hid_dim=self.hid_dim,
                rffs=_to_int(output_head_cfg.get("rffs", 4096), "model_config.output_head.rffs"),
                ridge_penalty=_to_float(
                    output_head_cfg.get("gp_ridge_penalty", 1.0),
                    "model_config.output_head.gp_ridge_penalty",
                ),
                cov_momentum=_to_float(
                    output_head_cfg.get("gp_cov_momentum", -1.0),
                    "model_config.output_head.gp_cov_momentum",
                ),
                random_seed=random_seed,
            )
        else:
            self.output_head = SpectralLinearHead(hid_dim=self.hid_dim)
        self.update_precision = False
        _xavier_initialize(self)

    def on_train_epoch_start(self, *, epoch_index: int, total_epochs: int) -> None:
        """Prepare optional SNGP precision tracking for the next epoch."""
        del epoch_index, total_epochs
        self.update_precision = self.output_head_type == "sngp"
        if isinstance(self.output_head, DiagonalRFFGaussianProcessHead):
            self.output_head.reset_precision()

    def _residue_windows(
        self,
        embeddings: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cropped residue embeddings and their effective lengths."""
        residues_by_sample: list[torch.Tensor] = []
        residue_lengths: list[int] = []
        for sample, raw_length in zip(embeddings, lengths, strict=True):
            valid_length = int(raw_length.item())
            if valid_length <= 0 or valid_length > sample.size(0):
                raise ValueError("Sequence length must be within the padded embedding length")
            start = 1 if self.exclude_special_tokens else 0
            end = valid_length - 1 if self.exclude_special_tokens else valid_length
            if end <= start:
                raise ValueError("TUnA requires at least one residue token per protein")
            residue = sample[start:end]
            if residue.size(0) > self.max_sequence_length:
                max_start = residue.size(0) - self.max_sequence_length
                crop_start = (
                    int(torch.randint(max_start + 1, (1,), device=sample.device).item())
                    if self.training
                    else 0
                )
                residue = residue[crop_start : crop_start + self.max_sequence_length]
            residues_by_sample.append(residue)
            residue_lengths.append(int(residue.size(0)))

        batch_max_len = max(residue_lengths)
        padded = embeddings.new_zeros((embeddings.size(0), batch_max_len, embeddings.size(2)))
        for row_index, residue in enumerate(residues_by_sample):
            padded[row_index, : residue.size(0)] = residue
        return padded, torch.tensor(residue_lengths, dtype=torch.long, device=embeddings.device)

    @staticmethod
    def _intra_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Build TUnA square keep mask for one protein sequence."""
        positions = torch.arange(max_len, device=lengths.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        return valid.unsqueeze(1).unsqueeze(2) & valid.unsqueeze(1).unsqueeze(3)

    def _inter_masks(
        self,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
        max_len_a: int,
        max_len_b: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build inter attention and token pooling masks."""
        positions_a = torch.arange(max_len_a, device=lengths_a.device)
        positions_b = torch.arange(max_len_b, device=lengths_b.device)
        valid_a = positions_a.unsqueeze(0) < lengths_a.unsqueeze(1)
        valid_b = positions_b.unsqueeze(0) < lengths_b.unsqueeze(1)
        token_mask = torch.cat([valid_a, valid_b], dim=1)
        if self.inter_mask_mode == "cross_chain":
            attention_mask = token_mask.unsqueeze(1).unsqueeze(2) & token_mask.unsqueeze(
                1
            ).unsqueeze(3)
            return attention_mask, token_mask

        total_len = max_len_a + max_len_b
        # Match the official TUnA code path for block-diagonal inter masks:
        # pooling uses ``combined_mask[:, 0, :, 0]``, which keeps only the first
        # protein in the ordered AB or BA concatenation.  AB/BA max aggregation
        # then supplies the order-invariant pair feature.
        ordered_first_token_mask = torch.cat([valid_a, torch.zeros_like(valid_b)], dim=1)
        attention_mask = torch.zeros(
            lengths_a.size(0),
            1,
            total_len,
            total_len,
            dtype=torch.bool,
            device=lengths_a.device,
        )
        attention_mask[:, :, :max_len_a, :max_len_a] = (
            valid_a.unsqueeze(1).unsqueeze(2) & valid_a.unsqueeze(1).unsqueeze(3)
        )
        attention_mask[:, :, max_len_a:, max_len_a:] = (
            valid_b.unsqueeze(1).unsqueeze(2) & valid_b.unsqueeze(1).unsqueeze(3)
        )
        return attention_mask, ordered_first_token_mask

    def _pair_feature(
        self,
        encoded_a: torch.Tensor,
        encoded_b: torch.Tensor,
        lengths_a: torch.Tensor,
        lengths_b: torch.Tensor,
    ) -> torch.Tensor:
        """Run inter encoder for one ordered protein pair."""
        attention_mask, token_mask = self._inter_masks(
            lengths_a=lengths_a,
            lengths_b=lengths_b,
            max_len_a=encoded_a.size(1),
            max_len_b=encoded_b.size(1),
        )
        return self.inter_encoder(encoded_a, encoded_b, attention_mask, token_mask)

    def _logits_from_feature(self, feature: torch.Tensor) -> torch.Tensor:
        """Apply the configured output head."""
        if isinstance(self.output_head, DiagonalRFFGaussianProcessHead):
            return self.output_head(feature, update_precision=self.update_precision)
        return cast(torch.Tensor, self.output_head(feature))

    def forward(
        self,
        batch: Mapping[str, torch.Tensor] | None = None,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run TUnA forward pass using repository batch keys."""
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
            lengths_a = torch.full((emb_a.size(0),), emb_a.size(1), dtype=torch.long, device=device)
        else:
            lengths_a = lengths_a.to(device=device, dtype=torch.long)
        if lengths_b is None:
            lengths_b = torch.full((emb_b.size(0),), emb_b.size(1), dtype=torch.long, device=device)
        else:
            lengths_b = lengths_b.to(device=device, dtype=torch.long)

        residue_a, residue_lengths_a = self._residue_windows(emb_a, lengths_a)
        residue_b, residue_lengths_b = self._residue_windows(emb_b, lengths_b)
        mask_a = self._intra_mask(residue_lengths_a, residue_a.size(1))
        mask_b = self._intra_mask(residue_lengths_b, residue_b.size(1))
        encoded_a = self.intra_encoder(residue_a, mask_a)
        encoded_b = self.intra_encoder(residue_b, mask_b)
        feature_ab = self._pair_feature(
            encoded_a=encoded_a,
            encoded_b=encoded_b,
            lengths_a=residue_lengths_a,
            lengths_b=residue_lengths_b,
        )
        feature_ba = self._pair_feature(
            encoded_a=encoded_b,
            encoded_b=encoded_a,
            lengths_a=residue_lengths_b,
            lengths_b=residue_lengths_a,
        )
        pair_feature = torch.max(torch.stack([feature_ab, feature_ba], dim=-1), dim=-1).values
        logits = self._logits_from_feature(pair_feature)
        output: dict[str, torch.Tensor] = {"logits": logits}
        if "label" in merged:
            labels = merged["label"].float()
            logits_for_loss = (
                logits.squeeze(-1) if logits.dim() > 1 and logits.size(-1) == 1 else logits
            )
            labels_for_loss = (
                labels.squeeze(-1) if labels.dim() > 1 and labels.size(-1) == 1 else labels
            )
            output["loss"] = F.binary_cross_entropy_with_logits(logits_for_loss, labels_for_loss)
        return output
