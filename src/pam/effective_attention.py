"""Effective-attention computation and head reduction.

Raw cross-attention tells you *where* a prompt token looked in the image.
Effective attention refines that by weighting each attended position by the
L2 norm of its value vector — high attention to a position whose value
vector has near-zero norm is effectively a no-op.

Formula
-------
Given:
    A  — raw attention weights          [B, H, T_prompt, T_image]
    V  — value vectors                  [B, H, T_image, D_head]

    value_norm = ||V||₂  over D_head    [B, H, T_image]
    effective  = A * value_norm[:, :, None, :]
    effective  = effective / (effective.sum(dim=-1, keepdim=True) + ε)

Then reduce heads with mean or max.
"""

from __future__ import annotations

import torch


def compute_effective_attention(
    attn: torch.Tensor,
    values: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute effective attention from raw attention and value vectors.

    Parameters
    ----------
    attn : Tensor
        Raw attention weights — shape ``[B, H, T_prompt, T_image]``.
    values : Tensor
        Value vectors — shape ``[B, H, T_image, D_head]``.
    eps : float
        Small constant to avoid division by zero during re-normalisation.

    Returns
    -------
    Tensor
        Effective attention — shape ``[B, H, T_prompt, T_image]``.
        Each row (over the T_image dimension) sums to ≈ 1.
    """
    # attn:   [B, H, T_prompt, T_image]
    # values: [B, H, T_image, D_head]

    # L2 norm of each value vector → [B, H, T_image]
    value_norm = values.norm(p=2, dim=-1)  # [B, H, T_image]

    # Weight raw attention by value norms.
    # value_norm[:, :, None, :] → [B, H, 1, T_image]
    effective = attn * value_norm[:, :, None, :]  # [B, H, T_prompt, T_image]

    # Re-normalise so each prompt token's distribution sums to 1.
    effective = effective / (effective.sum(dim=-1, keepdim=True) + eps)

    return effective


def reduce_heads(
    tensor: torch.Tensor,
    token_index: int,
    mode: str = "mean",
) -> torch.Tensor:
    """Reduce a multi-head attention tensor to a single 1-D heatmap.

    Parameters
    ----------
    tensor : Tensor
        Shape ``[B, H, T_prompt, T_image]``.
    token_index : int
        Which prompt-token slice to extract.
    mode : str
        ``"mean"`` or ``"max"`` reduction over the head dimension.

    Returns
    -------
    Tensor
        1-D heatmap of shape ``[T_image]``.
    """
    # Select batch 0, all heads, one token → [H, T_image]
    per_head = tensor[0, :, token_index, :]

    if mode == "mean":
        return per_head.mean(dim=0)
    elif mode == "max":
        return per_head.max(dim=0).values
    else:
        raise ValueError(f"Unknown head_reduction mode: {mode!r}")
