"""Query-index resolution for attribution flow.

Maps a user-visible *mask index* (from the filtered detections list) to the
internal *decoder query index* used in the attention tensors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class QueryIndexMapping:
    """Maps between object-query space and decoder-token space.

    object_query_index : int
        Index into pred_logits / pred_boxes / pred_masks.  Range 0–199.
    decoder_token_index : int
        Index into decoder cross-attention query dim.  Range 0–200.
        ``decoder_token_index = object_query_index + 1`` when the presence
        token is prepended at index 0.
    presence_token_index : int
        Position of the presence token in the decoder attention tensor.
    method : str
        How the mapping was resolved.
    confidence : float
        Confidence of the resolution (1.0 = deterministic).
    warnings : list[str]
        Any diagnostic messages.
    """

    object_query_index: int
    decoder_token_index: int
    presence_token_index: int = 0
    method: str = "kept_indices"
    confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class QueryResolution:
    """Result of resolving the decoder query index."""

    query_index: int
    method: str  # "manual_override", "kept_indices", "gradient_contribution_argmax", "assumed"
    confidence: float | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_query_index(
    target_mask_index: int,
    query_index_override: int | None = None,
    outputs: Dict[str, Any] | None = None,
    decoder_edge_attrs: list[torch.Tensor] | None = None,
    presence_query_index: int = 200,
    n_object_queries: int | None = None,
) -> QueryResolution:
    """Resolve the decoder query index for the selected detection.

    Parameters
    ----------
    target_mask_index : int
        Index into the filtered detections.
    query_index_override : int | None
        If provided, use this directly (``--query-index``).
    outputs : dict | None
        Model outputs containing ``_kept_indices``.
    decoder_edge_attrs : list[Tensor] | None
        Per-layer edge attributions ``[B, H, Nq, Ni]`` for gradient-based
        resolution.  Only used as fallback.
    presence_query_index : int
        Index of the presence token to exclude from argmax.
    n_object_queries : int | None
        Number of object queries (for offset computation).

    Returns
    -------
    QueryResolution
    """
    warns: list[str] = []

    # 1. Manual override.
    if query_index_override is not None:
        return QueryResolution(
            query_index=query_index_override,
            method="manual_override",
            confidence=1.0,
        )

    # 2. Explicit mapping from run_inference_with_grad.
    if outputs is not None:
        kept = outputs.get("_kept_indices")
        if kept is not None and target_mask_index < len(kept):
            qi = int(kept[target_mask_index].item())
            return QueryResolution(
                query_index=qi,
                method="kept_indices",
                confidence=1.0,
            )

    # 3. Gradient contribution argmax (post-backward fallback).
    if decoder_edge_attrs is not None and len(decoder_edge_attrs) > 0:
        # Stack layers → sum over layers, heads, image tokens.
        stacked = torch.stack(decoder_edge_attrs, dim=0)  # [L, B, H, Nq, Ni]
        contrib = stacked.clamp(min=0).sum(dim=(0, 1, 2, 4))  # [Nq]
        # Zero out presence token.
        if presence_query_index < len(contrib):
            contrib[presence_query_index] = 0.0
        qi = int(contrib.argmax().item())
        # Compute confidence as fraction of total.
        total = contrib.sum().item()
        conf = float(contrib[qi].item() / total) if total > 0 else 0.0
        return QueryResolution(
            query_index=qi,
            method="gradient_contribution_argmax",
            confidence=conf,
            warnings=warns,
        )

    # 4. Fallback: assume mask index = query index.
    warns.append(
        "Could not resolve query index from outputs or gradients. "
        f"Assuming target_mask_index={target_mask_index} equals query_index. "
        "This may be wrong if postprocessing reorders detections."
    )
    return QueryResolution(
        query_index=target_mask_index,
        method="assumed",
        confidence=0.0,
        warnings=warns,
    )


def save_query_resolution(resolution: QueryResolution, path: str | Path) -> Path:
    """Save query resolution info as JSON."""
    path = Path(path)
    data = {
        "query_index": resolution.query_index,
        "method": resolution.method,
        "confidence": resolution.confidence,
        "warnings": resolution.warnings,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def build_query_index_mapping(
    resolution: QueryResolution,
    presence_token_index: int = 0,
) -> QueryIndexMapping:
    """Build a ``QueryIndexMapping`` from a ``QueryResolution``.

    ``resolution.query_index`` is always in *object_query* space (0–199).
    The decoder token index is ``object_query_index + 1`` because the
    presence token is prepended at position 0.
    """
    oqi = resolution.query_index
    dti = oqi + 1  # presence token occupies index 0
    return QueryIndexMapping(
        object_query_index=oqi,
        decoder_token_index=dti,
        presence_token_index=presence_token_index,
        method=resolution.method,
        confidence=resolution.confidence if resolution.confidence is not None else 1.0,
        warnings=list(resolution.warnings),
    )


def save_query_index_mapping(mapping: QueryIndexMapping, path: str | Path) -> Path:
    """Save query index mapping as JSON."""
    path = Path(path)
    data = {
        "object_query_index": mapping.object_query_index,
        "decoder_token_index": mapping.decoder_token_index,
        "presence_token_index": mapping.presence_token_index,
        "method": mapping.method,
        "confidence": mapping.confidence,
        "warnings": mapping.warnings,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
