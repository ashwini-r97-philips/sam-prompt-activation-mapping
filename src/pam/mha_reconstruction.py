"""MultiheadAttention reconstruction for attribution flow.

Reconstructs the internal Q, K, V projections, attention probabilities,
per-head outputs, and full output of a ``torch.nn.MultiheadAttention``
module from captured forward-hook data.  The reconstruction is performed
*outside* the autograd graph — only the actual module output tensor
participates in the backward pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class MHAReconstruction:
    """Result of reconstructing a ``nn.MultiheadAttention`` forward pass."""

    module_name: str
    group: str
    layer_index: int | None
    batch_first: bool

    # All shapes normalised to batch-first:
    query: torch.Tensor  # [B, Tq, E]
    key: torch.Tensor  # [B, Ts, E]
    value: torch.Tensor  # [B, Ts, E]
    attn_probs: torch.Tensor  # [B, H, Tq, Ts]
    value_heads: torch.Tensor  # [B, H, Ts, Dh]
    pre_out_heads: torch.Tensor  # [B, H, Tq, Dh]
    pre_out_concat: torch.Tensor  # [B, Tq, E]
    reconstructed_output: torch.Tensor  # [B, Tq, E]
    actual_output: torch.Tensor  # [B, Tq, E]

    max_abs_reconstruction_error: float
    mean_abs_reconstruction_error: float

    input_shapes: dict = field(default_factory=dict)
    output_shape: tuple = ()
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core reconstruction
# ---------------------------------------------------------------------------


def reconstruct_mha(
    module: nn.MultiheadAttention,
    inputs: tuple,
    kwargs: dict,
    output: Any,
    *,
    module_name: str = "",
    group: str = "",
    layer_index: int | None = None,
    device: str | torch.device = "cpu",
) -> MHAReconstruction:
    """Reconstruct attention internals from a captured forward pass.

    Parameters
    ----------
    module : nn.MultiheadAttention
        The hooked module.
    inputs : tuple
        Positional args captured by the forward hook ``(query, key, value, …)``.
    kwargs : dict
        Keyword args captured by the forward hook.
    output : Any
        Output captured by the forward hook — typically ``(attn_output, attn_weights)``.
    module_name, group, layer_index
        Metadata for the result.
    device : str | torch.device
        Device to move reconstruction tensors to (default ``"cpu"``).

    Returns
    -------
    MHAReconstruction
    """
    warns: list[str] = []
    num_heads: int = module.num_heads
    embed_dim: int = module.embed_dim
    head_dim: int = embed_dim // num_heads
    batch_first: bool = getattr(module, "batch_first", False)

    # -- resolve Q, K, V from inputs / kwargs --------------------------------
    query = inputs[0] if len(inputs) > 0 else kwargs.get("query")
    key = inputs[1] if len(inputs) > 1 else kwargs.get("key")
    value = inputs[2] if len(inputs) > 2 else kwargs.get("value")

    if query is None or key is None or value is None:
        raise ValueError(
            f"[{module_name}] Could not resolve Q/K/V from hook inputs."
        )

    # -- resolve actual output -----------------------------------------------
    if isinstance(output, (tuple, list)):
        actual_out = output[0]
    else:
        actual_out = output

    # Record shapes before transposing.
    input_shapes = {
        "query": list(query.shape),
        "key": list(key.shape),
        "value": list(value.shape),
    }
    output_shape = tuple(actual_out.shape)

    # -- normalise to batch-first [B, T, E] ----------------------------------
    query_bf = query.clone().detach()
    key_bf = key.clone().detach()
    value_bf = value.clone().detach()
    actual_bf = actual_out.clone().detach()

    if not batch_first and query_bf.dim() == 3:
        query_bf = query_bf.transpose(0, 1)
        key_bf = key_bf.transpose(0, 1)
        value_bf = value_bf.transpose(0, 1)
        actual_bf = actual_bf.transpose(0, 1)

    # Cast to weight dtype for projection accuracy.
    w_dtype = _weight_dtype(module)
    query_bf = query_bf.to(w_dtype)
    key_bf = key_bf.to(w_dtype)
    value_bf = value_bf.to(w_dtype)

    bsz = query_bf.shape[0]

    # -- project Q, K, V using module weights --------------------------------
    with torch.no_grad():
        q, k, v = _project_qkv(module, query_bf, key_bf, value_bf, embed_dim)

    # Reshape to [B, H, T, Dh]
    q_heads = q.view(bsz, -1, num_heads, head_dim).transpose(1, 2)
    k_heads = k.view(bsz, -1, num_heads, head_dim).transpose(1, 2)
    v_heads = v.view(bsz, -1, num_heads, head_dim).transpose(1, 2)

    # -- compute attention probabilities -------------------------------------
    scale = math.sqrt(head_dim)
    scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / scale

    # Apply attn_mask if captured.
    attn_mask = kwargs.get("attn_mask")
    if attn_mask is None and len(inputs) > 3:
        attn_mask = inputs[3]
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    # Apply key_padding_mask.
    key_padding_mask = kwargs.get("key_padding_mask")
    if key_padding_mask is not None:
        # key_padding_mask: [B, Ts] — True means ignore.
        mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, Ts]
        scores = scores.masked_fill(mask, float("-inf"))

    attn_probs = F.softmax(scores, dim=-1)

    # -- per-head output before out_proj -------------------------------------
    pre_out_heads = torch.matmul(attn_probs, v_heads)  # [B, H, Tq, Dh]

    # Concat heads → [B, Tq, E]
    pre_out_concat = (
        pre_out_heads.transpose(1, 2).contiguous().view(bsz, -1, embed_dim)
    )

    # Apply output projection.
    with torch.no_grad():
        reconstructed = F.linear(
            pre_out_concat, module.out_proj.weight, module.out_proj.bias
        )

    # -- validation ----------------------------------------------------------
    actual_bf = actual_bf.to(reconstructed.dtype)
    diff = (reconstructed - actual_bf).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    # -- move to target device -----------------------------------------------
    dev = torch.device(device)

    return MHAReconstruction(
        module_name=module_name,
        group=group,
        layer_index=layer_index,
        batch_first=batch_first,
        query=query_bf.to(dev),
        key=key_bf.to(dev),
        value=value_bf.to(dev),
        attn_probs=attn_probs.to(dev),
        value_heads=v_heads.to(dev),
        pre_out_heads=pre_out_heads.to(dev),
        pre_out_concat=pre_out_concat.to(dev),
        reconstructed_output=reconstructed.to(dev),
        actual_output=actual_bf.to(dev),
        max_abs_reconstruction_error=max_err,
        mean_abs_reconstruction_error=mean_err,
        input_shapes=input_shapes,
        output_shape=output_shape,
        warnings=warns,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weight_dtype(module: nn.MultiheadAttention) -> torch.dtype:
    """Return the dtype of the module's projection weights."""
    if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
        return module.in_proj_weight.dtype
    if hasattr(module, "q_proj_weight") and module.q_proj_weight is not None:
        return module.q_proj_weight.dtype
    return torch.float32


def _project_qkv(
    module: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Q, K, V using the module's weights.

    Supports packed ``in_proj_weight`` and separate
    ``q_proj_weight`` / ``k_proj_weight`` / ``v_proj_weight``.

    All inputs/outputs are ``[B, T, E]``.
    """
    if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
        w = module.in_proj_weight
        b = module.in_proj_bias if module.in_proj_bias is not None else None
        q = F.linear(query, w[:embed_dim], b[:embed_dim] if b is not None else None)
        k = F.linear(key, w[embed_dim : 2 * embed_dim], b[embed_dim : 2 * embed_dim] if b is not None else None)
        v = F.linear(value, w[2 * embed_dim :], b[2 * embed_dim :] if b is not None else None)
        return q, k, v

    if hasattr(module, "q_proj_weight") and module.q_proj_weight is not None:
        b = module.in_proj_bias if module.in_proj_bias is not None else None
        q = F.linear(query, module.q_proj_weight, b[:embed_dim] if b is not None else None)
        k = F.linear(key, module.k_proj_weight, b[embed_dim : 2 * embed_dim] if b is not None else None)
        v = F.linear(value, module.v_proj_weight, b[2 * embed_dim :] if b is not None else None)
        return q, k, v

    raise ValueError(
        "Module has neither in_proj_weight nor q_proj_weight. "
        "Cannot project Q/K/V."
    )
