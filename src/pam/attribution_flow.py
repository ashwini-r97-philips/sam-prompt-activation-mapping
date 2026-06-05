"""Gradient-weighted edge attribution computation.

Core attribution primitive:

    R_ij^h = A_ij^h * dot(V_j^h, dS/dO_i^h)

where:
    i = receiving/query token
    j = source/key-value token
    h = attention head
    A = attention probability
    V = value vector
    dS/dO = gradient of target scalar w.r.t. pre-output-projection head output

This module computes edge attributions, aggregates them into spatial
heatmaps per group, and produces contributor rankings.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .attribution_capture import CapturedMHA


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class MHAAttribution:
    """Edge attribution for one MultiheadAttention module."""

    module_name: str
    group: str
    layer_index: int | None

    attn_probs: torch.Tensor  # [B, H, Tq, Ts]
    value_heads: torch.Tensor  # [B, H, Ts, Dh]
    grad_pre_heads: torch.Tensor  # [B, H, Tq, Dh]
    edge_attr: torch.Tensor  # [B, H, Tq, Ts]
    positive_edge_attr: torch.Tensor  # [B, H, Tq, Ts]
    negative_edge_attr: torch.Tensor  # [B, H, Tq, Ts]

    local_conservation_error: float = 0.0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gradient through output projection
# ---------------------------------------------------------------------------


def compute_grad_pre_heads(record: CapturedMHA) -> torch.Tensor:
    """Compute gradient w.r.t. pre-output-projection per-head outputs.

    The MHA output projection is::

        out = pre_out_concat @ W_out.T + b

    Therefore::

        grad_pre_concat = grad_out @ W_out

    Parameters
    ----------
    record : CapturedMHA
        Must have ``actual_output.grad`` available (after backward).

    Returns
    -------
    Tensor [B, H, Tq, Dh]
    """
    grad_out = record.actual_output.grad
    if grad_out is None:
        raise RuntimeError(
            f"[{record.module_name}] actual_output.grad is None. "
            "Was retain_grad() called before backward?"
        )

    # Normalise to batch-first [B, Tq, E].
    if not record.batch_first and grad_out.dim() == 3:
        grad_out = grad_out.transpose(0, 1)

    W_out = record.module.out_proj.weight  # [E, E]
    # grad_pre_concat = grad_out @ W_out  (because out = concat @ W.T + b)
    grad_pre_concat = torch.matmul(
        grad_out.to(W_out.dtype), W_out
    )  # [B, Tq, E]

    num_heads = record.module.num_heads
    head_dim = record.module.embed_dim // num_heads
    bsz = grad_pre_concat.shape[0]
    tq = grad_pre_concat.shape[1]

    grad_pre_heads = grad_pre_concat.view(bsz, tq, num_heads, head_dim).transpose(1, 2)
    return grad_pre_heads  # [B, H, Tq, Dh]


# ---------------------------------------------------------------------------
# Edge attribution
# ---------------------------------------------------------------------------


def compute_edge_attribution(
    record: CapturedMHA,
    device: str | torch.device = "cpu",
) -> MHAAttribution:
    """Compute gradient-weighted edge attribution for one module.

    Parameters
    ----------
    record : CapturedMHA
        With valid reconstruction and grad available.
    device : str | torch.device
        Target device for result tensors.

    Returns
    -------
    MHAAttribution
    """
    warns: list[str] = []
    recon = record.reconstruction
    if recon is None:
        raise ValueError(f"[{record.module_name}] No reconstruction available.")

    # Compute grad w.r.t. pre-head outputs.
    G = compute_grad_pre_heads(record).to(device)  # [B, H, Tq, Dh]
    A = recon.attn_probs.to(device)  # [B, H, Tq, Ts]
    V = recon.value_heads.to(device)  # [B, H, Ts, Dh]

    # Cast to same dtype.
    compute_dtype = torch.float32
    G = G.to(compute_dtype)
    A = A.to(compute_dtype)
    V = V.to(compute_dtype)

    # value_grad_dot[b,h,q,s] = sum_d V[b,h,s,d] * G[b,h,q,d]
    value_grad_dot = torch.einsum("bhsd,bhqd->bhqs", V, G)

    # Edge attribution.
    edge_attr = A * value_grad_dot  # [B, H, Tq, Ts]

    positive = F.relu(edge_attr)
    negative = F.relu(-edge_attr)

    # -- local conservation check -------------------------------------------
    # edge_attr.sum(source_dim) should ≈ (pre_out_heads * G).sum(d_dim)
    pre_out = recon.pre_out_heads.to(device).to(compute_dtype)  # [B, H, Tq, Dh]
    head_output_attr = (pre_out * G).sum(dim=-1)  # [B, H, Tq]
    edge_sum = edge_attr.sum(dim=-1)  # [B, H, Tq]
    conservation_diff = (edge_sum - head_output_attr).abs()
    conservation_err = conservation_diff.max().item()

    if conservation_err > 0.1:
        warns.append(
            f"Local conservation error is large: {conservation_err:.4e}. "
            "Attribution decomposition may be inaccurate."
        )

    return MHAAttribution(
        module_name=record.module_name,
        group=record.group,
        layer_index=record.layer_index,
        attn_probs=A.to(device),
        value_heads=V.to(device),
        grad_pre_heads=G.to(device),
        edge_attr=edge_attr.to(device),
        positive_edge_attr=positive.to(device),
        negative_edge_attr=negative.to(device),
        local_conservation_error=conservation_err,
        warnings=warns,
    )


# ---------------------------------------------------------------------------
# Group-specific spatial products
# ---------------------------------------------------------------------------


def compute_group_a_encoder_maps(
    attributions: list[MHAAttribution],
    spatial_grid: tuple[int, int],
    num_prompt_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute prompt-write spatial maps from Group A encoder attributions.

    Each module has edge_attr ``[B, H, 5184, 33]`` (image queries × prompt sources).

    Returns
    -------
    dict with:
        ``per_layer_token_maps_pos`` : ``[L, H, T, Gh, Gw]``
        ``per_layer_token_maps_neg`` : ``[L, H, T, Gh, Gw]``
        ``prompt_write_positive``    : ``[Gh, Gw]``
        ``prompt_write_negative``    : ``[Gh, Gw]``
    """
    Gh, Gw = spatial_grid
    L = len(attributions)
    if L == 0:
        raise ValueError("No Group A encoder attributions provided.")

    H = attributions[0].positive_edge_attr.shape[1]
    Ts = attributions[0].positive_edge_attr.shape[3]
    T = num_prompt_tokens if num_prompt_tokens is not None else Ts

    pos_maps = torch.zeros(L, H, T, Gh, Gw)
    neg_maps = torch.zeros(L, H, T, Gh, Gw)

    for layer_idx, attr in enumerate(attributions):
        # edge_attr: [B, H, N_image, T_prompt]
        # We want: for each prompt token t, the spatial map over image tokens.
        pos = attr.positive_edge_attr[0, :, :, :T]  # [H, N_image, T]
        neg = attr.negative_edge_attr[0, :, :, :T]

        # Transpose to [H, T, N_image] then reshape spatial.
        pos = pos.transpose(1, 2).reshape(H, T, Gh, Gw)
        neg = neg.transpose(1, 2).reshape(H, T, Gh, Gw)
        pos_maps[layer_idx] = pos.cpu()
        neg_maps[layer_idx] = neg.cpu()

    # Aggregate: sum over layers and heads, for selected prompt tokens.
    prompt_write_pos = pos_maps.sum(dim=(0, 1, 2))  # [Gh, Gw]
    prompt_write_neg = neg_maps.sum(dim=(0, 1, 2))  # [Gh, Gw]

    return {
        "per_layer_token_maps_pos": pos_maps,
        "per_layer_token_maps_neg": neg_maps,
        "prompt_write_positive": prompt_write_pos,
        "prompt_write_negative": prompt_write_neg,
    }


def compute_group_a_geometry_maps(
    attributions: list[MHAAttribution],
    spatial_grid: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Compute geometry-read spatial maps from Group A geometry attributions.

    Each module has edge_attr ``[B, H, 1, 5184]`` (1 geo query × image sources).

    Returns
    -------
    dict with:
        ``per_layer_maps_pos`` : ``[L, H, Gh, Gw]``
        ``geometry_positive``  : ``[Gh, Gw]``
        ``geometry_negative``  : ``[Gh, Gw]``
    """
    Gh, Gw = spatial_grid
    L = len(attributions)
    if L == 0:
        return {
            "per_layer_maps_pos": torch.zeros(0, 8, Gh, Gw),
            "geometry_positive": torch.zeros(Gh, Gw),
            "geometry_negative": torch.zeros(Gh, Gw),
        }

    H = attributions[0].positive_edge_attr.shape[1]
    pos_maps = torch.zeros(L, H, Gh, Gw)
    neg_maps = torch.zeros(L, H, Gh, Gw)

    for i, attr in enumerate(attributions):
        # [B, H, 1, N_image] → [H, N_image] → [H, Gh, Gw]
        pos_maps[i] = attr.positive_edge_attr[0, :, 0, :].reshape(H, Gh, Gw).cpu()
        neg_maps[i] = attr.negative_edge_attr[0, :, 0, :].reshape(H, Gh, Gw).cpu()

    return {
        "per_layer_maps_pos": pos_maps,
        "per_layer_maps_neg": neg_maps,
        "geometry_positive": pos_maps.sum(dim=(0, 1)),
        "geometry_negative": neg_maps.sum(dim=(0, 1)),
    }


def compute_group_b_decoder_maps(
    attributions: list[MHAAttribution],
    query_index: int,
    spatial_grid: tuple[int, int],
    presence_query_index: int = 0,
) -> dict[str, torch.Tensor]:
    """Compute selected-query-read spatial maps from Group B decoder attributions.

    Each module has edge_attr ``[B, H, 201, 5184]`` (queries × image sources).

    The decoder has 201 query tokens (1 presence + 200 object queries).
    The ``query_index`` is the *raw object query index* (0-199).  The
    presence token is prepended, so the attention-tensor index is
    ``query_index + offset`` where ``offset = N_attn_queries - N_obj_queries``.

    Parameters
    ----------
    attributions : list[MHAAttribution]
    query_index : int
        Raw object query index (0-based, from pred_logits).
    spatial_grid : (Gh, Gw)
    presence_query_index : int
        Index of presence token in attention tensor (default 0 — prepended).

    Returns
    -------
    dict with:
        ``per_layer_head_maps_pos``  : ``[L, H, Gh, Gw]``
        ``per_layer_head_maps_neg``  : ``[L, H, Gh, Gw]``
        ``query_read_positive``      : ``[Gh, Gw]``
        ``query_read_negative``      : ``[Gh, Gw]``
        ``contribution_per_query``   : ``[Nq]``
    """
    Gh, Gw = spatial_grid
    L = len(attributions)
    if L == 0:
        raise ValueError("No Group B decoder attributions provided.")

    H = attributions[0].positive_edge_attr.shape[1]
    N_attn_queries = attributions[0].edge_attr.shape[2]

    # Determine offset: presence token is prepended at index 0 in SAM3.
    # N_attn = 201, N_obj = 200, offset = 1.
    # Object query q in pred_logits corresponds to attn index q + 1.
    n_obj_queries = N_attn_queries - 1  # 200
    attn_query_idx = query_index + 1  # presence token is at index 0

    pos_maps = torch.zeros(L, H, Gh, Gw)
    neg_maps = torch.zeros(L, H, Gh, Gw)

    for i, attr in enumerate(attributions):
        # [B, H, Nq, Ni] → select query → [H, Ni] → [H, Gh, Gw]
        pos_maps[i] = attr.positive_edge_attr[0, :, attn_query_idx, :].reshape(H, Gh, Gw).cpu()
        neg_maps[i] = attr.negative_edge_attr[0, :, attn_query_idx, :].reshape(H, Gh, Gw).cpu()

    # Per-query contribution table (sum over layers, heads, image tokens).
    all_pos = torch.stack(
        [a.positive_edge_attr[0] for a in attributions], dim=0
    )  # [L, H, Nq, Ni]
    contrib = all_pos.sum(dim=(0, 1, 3))  # [Nq]

    return {
        "per_layer_head_maps_pos": pos_maps,
        "per_layer_head_maps_neg": neg_maps,
        "query_read_positive": pos_maps.sum(dim=(0, 1)),
        "query_read_negative": neg_maps.sum(dim=(0, 1)),
        "contribution_per_query": contrib.cpu(),
        "attn_query_index": attn_query_idx,
    }


def compute_joint_map(
    prompt_write_positive: torch.Tensor,
    query_read_positive: torch.Tensor,
) -> torch.Tensor:
    """Compute joint product heatmap: ``sqrt(normalize(P) * normalize(Q))``."""
    P = _minmax_normalize(prompt_write_positive)
    Q = _minmax_normalize(query_read_positive)
    return torch.sqrt(P * Q)


def compute_joint_sum_map(
    prompt_write_positive: torch.Tensor,
    query_read_positive: torch.Tensor,
) -> torch.Tensor:
    """Compute joint sum heatmap: ``0.5 * normalize(P) + 0.5 * normalize(Q)``."""
    P = _minmax_normalize(prompt_write_positive)
    Q = _minmax_normalize(query_read_positive)
    return 0.5 * P + 0.5 * Q


def _minmax_normalize(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-10:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Contributor rankings / CSVs
# ---------------------------------------------------------------------------


def save_group_a_token_contributions(
    maps: dict[str, torch.Tensor],
    token_labels: list[str],
    path: str | Path,
) -> Path:
    """Save per-prompt-token contribution CSV."""
    path = Path(path)
    pos = maps["per_layer_token_maps_pos"]  # [L, H, T, Gh, Gw]
    neg = maps["per_layer_token_maps_neg"]

    T = pos.shape[2]
    pos_per_token = pos.sum(dim=(0, 1, 3, 4))  # [T]
    neg_per_token = neg.sum(dim=(0, 1, 3, 4))
    signed = pos_per_token - neg_per_token
    total_pos = pos_per_token.sum().item() or 1.0

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "token_index", "token_label",
            "positive_contribution", "negative_contribution",
            "signed_contribution", "normalized_positive_fraction",
        ])
        for t in range(T):
            label = token_labels[t] if t < len(token_labels) else f"prompt_{t:02d}"
            w.writerow([
                t, label,
                f"{pos_per_token[t].item():.6f}",
                f"{neg_per_token[t].item():.6f}",
                f"{signed[t].item():.6f}",
                f"{pos_per_token[t].item() / total_pos:.4f}",
            ])
    return path


def save_group_a_layer_head_token_contributions(
    maps: dict[str, torch.Tensor],
    token_labels: list[str],
    path: str | Path,
) -> Path:
    """Save per-layer/head/token contribution CSV."""
    path = Path(path)
    pos = maps["per_layer_token_maps_pos"]  # [L, H, T, Gh, Gw]
    neg = maps["per_layer_token_maps_neg"]
    L, H, T, Gh, Gw = pos.shape

    pos_lht = pos.sum(dim=(3, 4))  # [L, H, T]
    neg_lht = neg.sum(dim=(3, 4))
    signed_lht = pos_lht - neg_lht

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "layer", "head", "token_index", "token_label",
            "positive_contribution", "negative_contribution", "signed_contribution",
        ])
        for l in range(L):
            for h in range(H):
                for t in range(T):
                    label = token_labels[t] if t < len(token_labels) else f"prompt_{t:02d}"
                    w.writerow([
                        l, h, t, label,
                        f"{pos_lht[l, h, t].item():.6f}",
                        f"{neg_lht[l, h, t].item():.6f}",
                        f"{signed_lht[l, h, t].item():.6f}",
                    ])
    return path


def save_group_b_query_layer_head_contributions(
    maps: dict[str, torch.Tensor],
    object_query_index: int,
    decoder_token_index: int,
    path: str | Path,
) -> Path:
    """Save per-layer/head contribution for the selected query."""
    path = Path(path)
    pos = maps["per_layer_head_maps_pos"]  # [L, H, Gh, Gw]
    neg = maps["per_layer_head_maps_neg"]
    L, H, Gh, Gw = pos.shape

    pos_lh = pos.sum(dim=(2, 3))  # [L, H]
    neg_lh = neg.sum(dim=(2, 3))
    signed_lh = pos_lh - neg_lh

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "layer", "head", "object_query_index", "decoder_token_index",
            "positive_contribution", "negative_contribution", "signed_contribution",
        ])
        for l in range(L):
            for h in range(H):
                w.writerow([
                    l, h, object_query_index, decoder_token_index,
                    f"{pos_lh[l, h].item():.6f}",
                    f"{neg_lh[l, h].item():.6f}",
                    f"{signed_lh[l, h].item():.6f}",
                ])
    return path


def save_group_b_per_query_contributions(
    contrib: torch.Tensor,
    selected_object_query_index: int,
    selected_decoder_token_index: int,
    presence_token_index: int,
    path: str | Path,
) -> Path:
    """Save contribution-per-query CSV.

    Rows are indexed by ``decoder_token_index`` (0–200).  The corresponding
    ``object_query_index`` is ``decoder_token_index - 1`` for non-presence
    tokens.
    """
    path = Path(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "decoder_token_index", "object_query_index",
            "positive_contribution",
            "is_presence_token", "selected",
        ])
        for dti in range(len(contrib)):
            is_presence = (dti == presence_token_index)
            oqi = "" if is_presence else dti - 1
            selected = (dti == selected_decoder_token_index)
            w.writerow([
                dti,
                oqi,
                f"{contrib[dti].item():.6f}",
                is_presence,
                selected,
            ])
    return path


def save_top_contributors(
    group_a_maps: dict[str, torch.Tensor] | None,
    group_b_maps: dict[str, torch.Tensor] | None,
    token_labels: list[str],
    object_query_index: int,
    path: str | Path,
    top_k: int = 24,
) -> Path:
    """Save top-K contributors across both groups."""
    path = Path(path)
    entries: list[dict] = []

    if group_a_maps is not None:
        pos = group_a_maps["per_layer_token_maps_pos"]  # [L, H, T, Gh, Gw]
        neg = group_a_maps["per_layer_token_maps_neg"]
        L, H, T, Gh, Gw = pos.shape
        pos_lht = pos.sum(dim=(3, 4))  # [L, H, T]
        neg_lht = neg.sum(dim=(3, 4))
        for l in range(L):
            for h in range(H):
                for t in range(T):
                    label = token_labels[t] if t < len(token_labels) else f"prompt_{t:02d}"
                    entries.append({
                        "group": "A_encoder",
                        "module_name": f"transformer.encoder.layers.{l}.cross_attn_image",
                        "layer": l, "head": h,
                        "query_index": "",
                        "token_index": t, "token_label": label,
                        "positive": pos_lht[l, h, t].item(),
                        "negative": neg_lht[l, h, t].item(),
                        "signed": (pos_lht[l, h, t] - neg_lht[l, h, t]).item(),
                    })

    if group_b_maps is not None:
        pos = group_b_maps["per_layer_head_maps_pos"]  # [L, H, Gh, Gw]
        neg = group_b_maps["per_layer_head_maps_neg"]
        L, H, Gh, Gw = pos.shape
        pos_lh = pos.sum(dim=(2, 3))
        neg_lh = neg.sum(dim=(2, 3))
        for l in range(L):
            for h in range(H):
                entries.append({
                    "group": "B_decoder",
                    "module_name": f"transformer.decoder.layers.{l}.cross_attn",
                    "layer": l, "head": h,
                    "query_index": object_query_index,
                    "token_index": "", "token_label": "",
                    "positive": pos_lh[l, h].item(),
                    "negative": neg_lh[l, h].item(),
                    "signed": (pos_lh[l, h] - neg_lh[l, h]).item(),
                })

    # Sort by positive contribution descending.
    entries.sort(key=lambda e: e["positive"], reverse=True)
    total_pos = sum(e["positive"] for e in entries) or 1.0

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "group", "module_name", "layer", "head",
            "query_index", "token_index", "token_label",
            "positive_contribution", "negative_contribution",
            "signed_contribution", "normalized_weight",
        ])
        for rank, e in enumerate(entries[:top_k]):
            w.writerow([
                rank + 1, e["group"], e["module_name"], e["layer"], e["head"],
                e["query_index"], e["token_index"], e["token_label"],
                f"{e['positive']:.6f}", f"{e['negative']:.6f}",
                f"{e['signed']:.6f}", f"{e['positive'] / total_pos:.4f}",
            ])
    return path


def save_query_validation_table(
    contrib: torch.Tensor,
    pred_logits: torch.Tensor,
    presence_token_index: int,
    selected_object_query_index: int,
    path: str | Path,
    n_show: int = 10,
) -> Path:
    """Save a validation table comparing object query indices to decoder token indices.

    Shows the top-N decoder tokens by positive contribution, plus the
    selected token and presence token (if not already in top-N).

    Columns:
        object_query_index, decoder_token_index, pred_logit_value,
        decoder_query_positive_contribution, is_presence_token
    """
    path = Path(path)
    n_tokens = len(contrib)
    selected_dti = selected_object_query_index + 1

    # Determine which rows to show.
    top_indices = torch.argsort(contrib, descending=True)[:n_show].tolist()
    must_show = {presence_token_index, selected_dti}
    show_set = list(dict.fromkeys(top_indices + [i for i in must_show if i not in top_indices]))

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "object_query_index", "decoder_token_index",
            "pred_logit_value", "decoder_query_positive_contribution",
            "is_presence_token",
        ])
        for dti in show_set:
            is_presence = (dti == presence_token_index)
            if is_presence:
                oqi_str = ""
                logit_str = ""
            else:
                oqi = dti - 1
                oqi_str = str(oqi)
                if 0 <= oqi < pred_logits.shape[1]:
                    logit_str = f"{pred_logits[0, oqi, 0].item():.4f}"
                else:
                    logit_str = ""
            w.writerow([
                oqi_str,
                dti,
                logit_str,
                f"{contrib[dti].item():.6f}",
                is_presence,
            ])
    return path
