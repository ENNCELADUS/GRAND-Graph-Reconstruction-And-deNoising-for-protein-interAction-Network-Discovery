"""Reusable pair-readout modules for V3.1 ablations."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


def residue_mask(x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    """Return mask for non-special residue tokens, excluding BOS/EOS/padding."""
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
    mask = valid_mask & (positions > 0) & (positions < valid_lengths - 1)
    if bool((mask.sum(dim=1) == 0).any()):
        raise ValueError("Pair readout requires at least one residue token between BOS and EOS")
    return mask


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool a token sequence with a boolean keep mask."""
    mask_3d = mask.float().unsqueeze(-1)
    return cast(torch.Tensor, (x * mask_3d).sum(dim=1) / mask_3d.sum(dim=1).clamp_min(1.0))


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max-pool a token sequence with a boolean keep mask."""
    masked_x = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
    return cast(torch.Tensor, masked_x.max(dim=1).values)


class ContactTokenCompressor(nn.Module):
    """Mask-aware residue compressor for fixed-size contact sketches."""

    def __init__(self, num_tokens: int) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("contact_tokens must be >= 1")
        self.num_tokens = int(num_tokens)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Compress residue states to ``num_tokens`` by adaptive average pooling."""
        mask = residue_mask(x=x, padding_mask=padding_mask)
        pooled_rows: list[torch.Tensor] = []
        for sample, sample_mask in zip(x, mask, strict=True):
            residues = sample[sample_mask].transpose(0, 1).unsqueeze(0)
            pooled = F.adaptive_avg_pool1d(residues, self.num_tokens).squeeze(0).transpose(0, 1)
            pooled_rows.append(pooled)
        return torch.stack(pooled_rows, dim=0)


class PairContextGatedReadout(nn.Module):
    """Residue-only mean/max/pair-conditioned-attention readout."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_scorer = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.branch_gate = nn.Linear(d_model * 12, 3)
        self.branch_proj = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.final_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _attention_pool(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        other_context: torch.Tensor,
    ) -> torch.Tensor:
        context = other_context.unsqueeze(1).expand(-1, x.size(1), -1)
        scores = self.attn_scorer(torch.cat([x, context], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return cast(torch.Tensor, (x * weights).sum(dim=1))

    @staticmethod
    def _pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        cls_vec: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build a pair representation from residue-only branch features."""
        residue_a = residue_mask(x=h_a, padding_mask=mask_a)
        residue_b = residue_mask(x=h_b, padding_mask=mask_b)

        mean_a = masked_mean(h_a, residue_a)
        mean_b = masked_mean(h_b, residue_b)
        max_a = masked_max(h_a, residue_a)
        max_b = masked_max(h_b, residue_b)
        context_a = self.context_proj(torch.cat([mean_a, max_a], dim=-1))
        context_b = self.context_proj(torch.cat([mean_b, max_b], dim=-1))
        attn_a = self._attention_pool(h_a, residue_a, context_b)
        attn_b = self._attention_pool(h_b, residue_b, context_a)

        branches = torch.stack(
            [
                self._pair_features(mean_a, mean_b),
                self._pair_features(max_a, max_b),
                self._pair_features(attn_a, attn_b),
            ],
            dim=1,
        )
        gate_input = branches.flatten(start_dim=1)
        gate_weights = torch.softmax(self.branch_gate(gate_input), dim=1).unsqueeze(-1)
        branch_repr = (branches * gate_weights).sum(dim=1)
        projected = self.branch_proj(branch_repr)
        return cast(torch.Tensor, self.final_proj(torch.cat([cls_vec, projected], dim=-1)))


class ContactSketchFusionReadout(nn.Module):
    """Fuse no-CLS rich-pooling representation with a latent contact sketch."""

    def __init__(
        self,
        d_model: int,
        contact_tokens: int,
        pair_dim: int,
        cnn_dim: int,
        cnn_blocks: int,
        cnn_dropout: float,
        dropout: float,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if pair_dim <= 0:
            raise ValueError("pair_dim must be >= 1")
        if cnn_dim <= 0:
            raise ValueError("cnn_dim must be >= 1")
        if cnn_blocks <= 0:
            raise ValueError("cnn_blocks must be >= 1")
        self.eps = float(eps)
        self.compressor = ContactTokenCompressor(num_tokens=contact_tokens)
        self.proj_a = nn.Sequential(nn.Linear(d_model, pair_dim), nn.GELU())
        self.proj_b = nn.Sequential(nn.Linear(d_model, pair_dim), nn.GELU())
        in_channels = 4 * pair_dim + 1
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, cnn_dim),
            nn.GELU(),
            *[
                _ContactResidualBlock(channels=cnn_dim, dropout=cnn_dropout)
                for _ in range(cnn_blocks)
            ],
        )
        self.contact_proj = nn.Sequential(
            nn.LayerNorm(cnn_dim * 2),
            nn.Linear(cnn_dim * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _build_grid(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        z_a = self.proj_a(h_a)
        z_b = self.proj_b(h_b)
        len_a = z_a.size(1)
        len_b = z_b.size(1)
        z_a_exp = z_a.unsqueeze(2).expand(-1, -1, len_b, -1)
        z_b_exp = z_b.unsqueeze(1).expand(-1, len_a, -1, -1)
        cosine = F.cosine_similarity(z_a_exp, z_b_exp, dim=-1, eps=self.eps).unsqueeze(-1)
        grid = torch.cat(
            [
                z_a_exp,
                z_b_exp,
                torch.abs(z_a_exp - z_b_exp),
                z_a_exp * z_b_exp,
                cosine,
            ],
            dim=-1,
        )
        return grid.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        base_repr: torch.Tensor,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return fused rich-pooling plus contact-sketch pair representation."""
        sketch_a = self.compressor(h_a, mask_a)
        sketch_b = self.compressor(h_b, mask_b)
        features = self.cnn(self._build_grid(sketch_a, sketch_b))
        pooled_max = F.adaptive_max_pool2d(features, (1, 1)).flatten(1)
        pooled_mean = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        contact_repr = self.contact_proj(torch.cat([pooled_max, pooled_mean], dim=1))
        return cast(torch.Tensor, self.fusion(torch.cat([base_repr, contact_repr], dim=1)))


class _ContactResidualBlock(nn.Module):
    """Small residual CNN block for contact sketches."""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.activation(x + self.block(x)))
