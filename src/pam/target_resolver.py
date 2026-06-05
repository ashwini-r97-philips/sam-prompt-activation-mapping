"""Target scalar resolution for attribution flow.

Selects a differentiable scalar from the model's outputs that can be
backpropagated through to compute attributions.
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
class TargetResolution:
    """Result of resolving a target scalar for backward."""

    target_scalar: torch.Tensor
    target_description: str
    target_mask_index: int | None
    query_index: int | None
    source: str
    requires_grad: bool
    grad_fn: str | None
    warnings: list[str] = field(default_factory=list)
    _foreground_info: Any = None  # ForegroundMaskInfo, if applicable


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_target_scalar(
    outputs: Dict[str, Any],
    target_mask_index: int,
    target_objective: str = "score",
    query_index: int | None = None,
    *,
    background_penalty: float = 0.25,
    lambda_mask: float = 1.0,
) -> TargetResolution:
    """Resolve a differentiable target scalar from model outputs.

    Parameters
    ----------
    outputs : dict
        Output of :func:`~pam.sam3_loader.run_inference_with_grad`.
    target_mask_index : int
        Index into the *filtered* detections (0-based).
    target_objective : str
        Target objective name.
    query_index : int | None
        Required for ``"raw_query_score"``; ignored otherwise.
    background_penalty : float
        Weight for background term in ``mask_contrastive_logit``.
    lambda_mask : float
        Weight for mask term in ``combined_query_and_mask``.

    Returns
    -------
    TargetResolution
    """
    warns: list[str] = []

    if target_objective == "score":
        return _resolve_score(outputs, target_mask_index, warns)
    elif target_objective == "object_query_score":
        return _resolve_object_query_score(outputs, target_mask_index, warns)
    elif target_objective == "raw_query_score":
        return _resolve_raw_query_score(outputs, target_mask_index, query_index, warns)
    elif target_objective == "mask_logit_mean":
        return _resolve_mask_logit_mean(outputs, target_mask_index, warns)
    elif target_objective == "mask_foreground_logit_mean":
        return _resolve_mask_foreground_logit_mean(outputs, target_mask_index, warns)
    elif target_objective == "mask_probability_foreground_mean":
        return _resolve_mask_probability_foreground_mean(outputs, target_mask_index, warns)
    elif target_objective == "mask_contrastive_logit":
        return _resolve_mask_contrastive_logit(
            outputs, target_mask_index, warns, background_penalty=background_penalty,
        )
    elif target_objective == "combined_query_and_mask":
        return _resolve_combined_query_and_mask(
            outputs, target_mask_index, warns, lambda_mask=lambda_mask,
        )
    elif target_objective == "semantic_logit_mean":
        return _resolve_semantic_logit_mean(outputs, target_mask_index, warns)
    elif target_objective == "semantic_foreground_logit_mean":
        return _resolve_semantic_foreground_logit_mean(
            outputs, target_mask_index, warns,
        )
    elif target_objective == "semantic_contrastive_logit":
        return _resolve_semantic_contrastive_logit(
            outputs, target_mask_index, warns,
            background_penalty=background_penalty,
        )
    else:
        raise ValueError(f"Unknown target_objective: {target_objective!r}")


# ---------------------------------------------------------------------------
# Objective implementations
# ---------------------------------------------------------------------------


def _resolve_score(
    outputs: Dict[str, Any],
    target_mask_index: int,
    warns: list[str],
) -> TargetResolution:
    """Combined score = sigmoid(logit) * sigmoid(presence)."""
    kept = outputs["_kept_indices"]
    if target_mask_index >= len(kept):
        raise ValueError(
            f"target_mask_index={target_mask_index} but only "
            f"{len(kept)} detection(s) passed confidence threshold."
        )
    query_idx = int(kept[target_mask_index].item())

    pred_logits = outputs["pred_logits"]  # [B, N, 1]
    presence = outputs["presence_logit_dec"]  # [B, 1] or [B]

    logit = pred_logits[0, query_idx, 0]
    pres = presence.view(-1)[0] if presence.dim() <= 2 else presence[0]

    target = logit.sigmoid() * pres.sigmoid()

    return TargetResolution(
        target_scalar=target,
        target_description=(
            f"sigmoid(pred_logits[0,{query_idx},0]) * sigmoid(presence) "
            f"= {target.item():.4f}"
        ),
        target_mask_index=target_mask_index,
        query_index=query_idx,
        source="score",
        requires_grad=target.requires_grad,
        grad_fn=str(target.grad_fn) if target.grad_fn else None,
        warnings=warns,
    )


def _resolve_raw_query_score(
    outputs: Dict[str, Any],
    target_mask_index: int,
    query_index: int | None,
    warns: list[str],
) -> TargetResolution:
    """Raw logit for a specific query."""
    if query_index is None:
        # Try to resolve from kept_indices.
        kept = outputs.get("_kept_indices")
        if kept is not None and target_mask_index < len(kept):
            query_index = int(kept[target_mask_index].item())
        else:
            raise ValueError(
                "--query-index is required for target_objective='raw_query_score' "
                "when query index cannot be inferred."
            )

    pred_logits = outputs["pred_logits"]  # [B, N, 1]
    target = pred_logits[0, query_index, 0]

    return TargetResolution(
        target_scalar=target,
        target_description=f"pred_logits[0, {query_index}, 0] = {target.item():.4f}",
        target_mask_index=target_mask_index,
        query_index=query_index,
        source="raw_query_score",
        requires_grad=target.requires_grad,
        grad_fn=str(target.grad_fn) if target.grad_fn else None,
        warnings=warns,
    )


def _resolve_object_query_score(
    outputs: Dict[str, Any],
    target_mask_index: int,
    warns: list[str],
) -> TargetResolution:
    """Object-query score = sigmoid(pred_logits[0, qi, 0]) without presence."""
    kept = outputs["_kept_indices"]
    if target_mask_index >= len(kept):
        raise ValueError(
            f"target_mask_index={target_mask_index} but only "
            f"{len(kept)} detection(s) passed confidence threshold."
        )
    query_idx = int(kept[target_mask_index].item())

    pred_logits = outputs["pred_logits"]  # [B, N, 1]
    logit = pred_logits[0, query_idx, 0]
    target = logit.sigmoid()

    return TargetResolution(
        target_scalar=target,
        target_description=(
            f"sigmoid(pred_logits[0,{query_idx},0]) = {target.item():.4f}"
        ),
        target_mask_index=target_mask_index,
        query_index=query_idx,
        source="object_query_score",
        requires_grad=target.requires_grad,
        grad_fn=str(target.grad_fn) if target.grad_fn else None,
        warnings=warns,
    )


def _resolve_mask_logit_mean(
    outputs: Dict[str, Any],
    target_mask_index: int,
    warns: list[str],
) -> TargetResolution:
    """Mean mask logit for the selected detection."""
    warns.append(
        "mask_logit_mean uses output mask pixels as the target scalar. "
        "This is allowed but be aware the heatmap target is mask-derived."
    )
    kept = outputs["_kept_indices"]
    if target_mask_index >= len(kept):
        raise ValueError(
            f"target_mask_index={target_mask_index} but only "
            f"{len(kept)} detection(s)."
        )
    query_idx = int(kept[target_mask_index].item())

    pred_masks = outputs.get("pred_masks")
    if pred_masks is None:
        raise ValueError("No pred_masks in outputs for mask_logit_mean.")

    mask_logits = pred_masks[0, query_idx]  # [H, W]
    target = mask_logits.mean()

    return TargetResolution(
        target_scalar=target,
        target_description=f"mean(pred_masks[0, {query_idx}]) = {target.item():.4f}",
        target_mask_index=target_mask_index,
        query_index=query_idx,
        source="mask_logit_mean",
        requires_grad=target.requires_grad,
        grad_fn=str(target.grad_fn) if target.grad_fn else None,
        warnings=warns,
    )


# ---------------------------------------------------------------------------
# Foreground mask builder
# ---------------------------------------------------------------------------


@dataclass
class ForegroundMaskInfo:
    """Information about the foreground mask used for mask objectives."""

    foreground: torch.Tensor  # bool [H, W]
    background: torch.Tensor  # bool [H, W]
    foreground_pixels: int
    background_pixels: int
    foreground_area_fraction: float
    background_area_fraction: float
    target_hw: tuple[int, int]


def build_foreground_mask(
    outputs: Dict[str, Any],
    target_mask_index: int,
    target_hw: tuple[int, int],
) -> ForegroundMaskInfo:
    """Build a foreground boolean mask at the mask-logit resolution.

    Uses the postprocessed selected mask only to determine which output
    mask-logit pixels are "foreground".  Does NOT use the mask to rank
    or construct attention heatmaps.

    Parameters
    ----------
    outputs : dict
        Must contain ``"masks"`` (filtered, thresholded, at original resolution).
    target_mask_index : int
        Index into the filtered detections.
    target_hw : (int, int)
        Resolution of ``pred_masks`` logits, e.g. ``(288, 288)``.
    """
    filtered_masks = outputs.get("masks")
    if filtered_masks is None:
        raise ValueError("No postprocessed masks in outputs.  Cannot build foreground mask.")

    if target_mask_index >= filtered_masks.shape[0]:
        raise ValueError(
            f"target_mask_index={target_mask_index} but only "
            f"{filtered_masks.shape[0]} filtered mask(s)."
        )

    # [1, orig_H, orig_W] bool → float → resize to target_hw
    sel_mask = filtered_masks[target_mask_index, 0].float()  # [orig_H, orig_W]
    sel_mask = sel_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, oH, oW]
    resized = torch.nn.functional.interpolate(
        sel_mask, size=target_hw, mode="nearest",
    )  # [1, 1, H, W]
    foreground = (resized[0, 0] > 0.5)  # bool [H, W]
    background = ~foreground

    n_fg = int(foreground.sum().item())
    n_bg = int(background.sum().item())
    total = n_fg + n_bg

    if n_fg == 0:
        raise ValueError(
            "Foreground mask is empty after resizing to mask-logit resolution "
            f"{target_hw}.  Cannot compute mask-foreground objectives."
        )

    fg_frac = n_fg / total
    if fg_frac > 0.95:
        import warnings as _warnings
        _warnings.warn(
            f"Foreground covers {fg_frac:.1%} of the mask-logit plane.  "
            "The foreground mask objective may behave like mask_logit_mean.",
            stacklevel=2,
        )

    return ForegroundMaskInfo(
        foreground=foreground,
        background=background,
        foreground_pixels=n_fg,
        background_pixels=n_bg,
        foreground_area_fraction=fg_frac,
        background_area_fraction=1.0 - fg_frac,
        target_hw=target_hw,
    )


# ---------------------------------------------------------------------------
# New mask-foreground objectives
# ---------------------------------------------------------------------------


def _get_mask_logits_and_qi(
    outputs: Dict[str, Any],
    target_mask_index: int,
) -> tuple[torch.Tensor, int]:
    """Extract differentiable mask logits for the selected query."""
    kept = outputs["_kept_indices"]
    if target_mask_index >= len(kept):
        raise ValueError(
            f"target_mask_index={target_mask_index} but only "
            f"{len(kept)} detection(s) passed confidence threshold."
        )
    qi = int(kept[target_mask_index].item())

    pred_masks = outputs.get("pred_masks")
    if pred_masks is None:
        raise ValueError("No pred_masks in outputs.")
    if not pred_masks.requires_grad:
        raise ValueError(
            "pred_masks does not require grad.  The forward path may be "
            "under no_grad or inference_mode."
        )

    mask_logits = pred_masks[0, qi]  # [H, W]
    return mask_logits, qi


def _resolve_mask_foreground_logit_mean(
    outputs: Dict[str, Any],
    target_mask_index: int,
    warns: list[str],
) -> TargetResolution:
    """mean(mask_logits[foreground])."""
    mask_logits, qi = _get_mask_logits_and_qi(outputs, target_mask_index)
    target_hw = (mask_logits.shape[0], mask_logits.shape[1])
    fg_info = build_foreground_mask(outputs, target_mask_index, target_hw)

    target = mask_logits[fg_info.foreground].mean()

    return TargetResolution(
        target_scalar=target,
        target_description=(
            f"mean(pred_masks[0,{qi}][foreground]) = {target.item():.4f}  "
            f"(fg_frac={fg_info.foreground_area_fraction:.3f})"
        ),
        target_mask_index=target_mask_index,
        query_index=qi,
        source="mask_foreground_logit_mean",
        requires_grad=target.requires_grad,
        grad_fn=str(target.grad_fn) if target.grad_fn else None,
        warnings=warns,
        _foreground_info=fg_info,
    )


def _resolve_mask_probability_foreground_mean(
    outputs: Dict[str, Any],
    target_mask_index: int,
    warns: list[str],
) -> TargetResolution:
    """mean(sigmoid(mask_logits[foreground]))."""
    mask_logits, qi = _get_mask_logits_and_qi(outputs, target_mask_index)
    target_hw = (mask_logits.shape[0], mask_logits.shape[1])
    fg_info = build_foreground_mask(outputs, target_mask_index, target_hw)

    target = mask_logits[fg_info.foreground].sigmoid().mean()

    return TargetResolution(
        target_scalar=target,
        target_description=(
            f"mean(sigmoid(pred_masks[0,{qi}][foreground])) = {target.item():.4f}  "
            f"(fg_frac={fg_info.foreground_area_fraction:.3f})"
        ),
        target_mask_index=target_mask_index,
        query_index=qi,
        source="mask_probability_foreground_mean",
        requires_grad=target.requires_grad,
        grad_fn=str(target.grad_fn) if target.grad_fn else None,
        warnings=warns,
        _foreground_info=fg_info,
    )


def _resolve_mask_contrastive_logit(
    outputs: Dict[str, Any],
    target_mask_index: int,
    warns: list[str],
    *,
    background_penalty: float = 0.25,
) -> TargetResolution:
    """fg_mean - alpha * bg_mean."""
    mask_logits, qi = _get_mask_logits_and_qi(outputs, target_mask_index)
    target_hw = (mask_logits.shape[0], mask_logits.shape[1])
    fg_info = build_foreground_mask(outputs, target_mask_index, target_hw)

    fg_mean = mask_logits[fg_info.foreground].mean()
    bg_mean = mask_logits[fg_info.background].mean()
    target = fg_mean - background_penalty * bg_mean

    return TargetResolution(
        target_scalar=target,
        target_description=(
            f"mean(fg) - {background_penalty}*mean(bg) = {target.item():.4f}  "
            f"(fg={fg_mean.item():.4f}, bg={bg_mean.item():.4f}, "
            f"fg_frac={fg_info.foreground_area_fraction:.3f})"
        ),
        target_mask_index=target_mask_index,
        query_index=qi,
        source="mask_contrastive_logit",
        requires_grad=target.requires_grad,
        grad_fn=str(target.grad_fn) if target.grad_fn else None,
        warnings=warns,
        _foreground_info=fg_info,
    )


def _resolve_combined_query_and_mask(
    outputs: Dict[str, Any],
    target_mask_index: int,
    warns: list[str],
    *,
    lambda_mask: float = 1.0,
) -> TargetResolution:
    """sigmoid(logit) + lambda * mean(mask_logits[foreground])."""
    mask_logits, qi = _get_mask_logits_and_qi(outputs, target_mask_index)
    target_hw = (mask_logits.shape[0], mask_logits.shape[1])
    fg_info = build_foreground_mask(outputs, target_mask_index, target_hw)

    pred_logits = outputs["pred_logits"]
    query_score = pred_logits[0, qi, 0].sigmoid()
    mask_target = mask_logits[fg_info.foreground].mean()
    target = query_score + lambda_mask * mask_target

    return TargetResolution(
        target_scalar=target,
        target_description=(
            f"sigmoid(logit[{qi}]) + {lambda_mask}*mean(fg_logits) = {target.item():.4f}  "
            f"(query={query_score.item():.4f}, mask={mask_target.item():.4f}, "
            f"fg_frac={fg_info.foreground_area_fraction:.3f})"
        ),
        target_mask_index=target_mask_index,
        query_index=qi,
        source="combined_query_and_mask",
        requires_grad=target.requires_grad,
        grad_fn=str(target.grad_fn) if target.grad_fn else None,
        warnings=warns,
        _foreground_info=fg_info,
    )


# ---------------------------------------------------------------------------
# Semantic segmentation objectives
# ---------------------------------------------------------------------------


def _get_semantic_logits(
    outputs: Dict[str, Any],
) -> torch.Tensor:
    """Extract differentiable semantic segmentation logits.

    Returns
    -------
    Tensor [Hs, Ws]  (typically 288×288)
    """
    raw = outputs.get("_raw_outputs")
    if raw is None:
        raise ValueError(
            "No _raw_outputs in outputs dict.  Cannot access semantic_seg."
        )

    semantic = raw.get("semantic_seg")
    if semantic is None:
        raise ValueError(
            "No semantic_seg in _raw_outputs.  "
            "The model may not have a semantic segmentation head."
        )

    if not semantic.requires_grad:
        raise ValueError(
            "semantic_seg does not require grad.  The forward path may be "
            "under no_grad or inference_mode."
        )

    # [B, 1, H, W] → [H, W]
    return semantic[0, 0]


def _resolve_qi_for_semantic(
    outputs: Dict[str, Any],
    target_mask_index: int,
) -> int | None:
    """Resolve query index from kept_indices for Group B even under semantic targets."""
    kept = outputs.get("_kept_indices")
    if kept is not None and target_mask_index < len(kept):
        return int(kept[target_mask_index].item())
    return None


def _resolve_semantic_logit_mean(
    outputs: Dict[str, Any],
    target_mask_index: int,
    warns: list[str],
) -> TargetResolution:
    """mean(semantic_seg[0, 0])."""
    semantic = _get_semantic_logits(outputs)
    qi = _resolve_qi_for_semantic(outputs, target_mask_index)

    target = semantic.mean()

    return TargetResolution(
        target_scalar=target,
        target_description=(
            f"mean(semantic_seg[0,0]) = {target.item():.4f}"
        ),
        target_mask_index=target_mask_index,
        query_index=qi,
        source="semantic_seg",
        requires_grad=target.requires_grad,
        grad_fn=str(target.grad_fn) if target.grad_fn else None,
        warnings=warns,
    )


def _resolve_semantic_foreground_logit_mean(
    outputs: Dict[str, Any],
    target_mask_index: int,
    warns: list[str],
) -> TargetResolution:
    """mean(semantic_seg[0,0][foreground])."""
    semantic = _get_semantic_logits(outputs)
    qi = _resolve_qi_for_semantic(outputs, target_mask_index)
    target_hw = (semantic.shape[0], semantic.shape[1])
    fg_info = build_foreground_mask(outputs, target_mask_index, target_hw)

    target = semantic[fg_info.foreground].mean()

    return TargetResolution(
        target_scalar=target,
        target_description=(
            f"mean(semantic_seg[0,0][foreground]) = {target.item():.4f}  "
            f"(fg_frac={fg_info.foreground_area_fraction:.3f})"
        ),
        target_mask_index=target_mask_index,
        query_index=qi,
        source="semantic_seg",
        requires_grad=target.requires_grad,
        grad_fn=str(target.grad_fn) if target.grad_fn else None,
        warnings=warns,
        _foreground_info=fg_info,
    )


def _resolve_semantic_contrastive_logit(
    outputs: Dict[str, Any],
    target_mask_index: int,
    warns: list[str],
    *,
    background_penalty: float = 0.25,
) -> TargetResolution:
    """fg_mean - alpha * bg_mean on semantic_seg."""
    semantic = _get_semantic_logits(outputs)
    qi = _resolve_qi_for_semantic(outputs, target_mask_index)
    target_hw = (semantic.shape[0], semantic.shape[1])
    fg_info = build_foreground_mask(outputs, target_mask_index, target_hw)

    if fg_info.background_pixels == 0:
        raise ValueError(
            "Background mask is empty.  Cannot compute semantic_contrastive_logit."
        )

    fg_mean = semantic[fg_info.foreground].mean()
    bg_mean = semantic[fg_info.background].mean()
    target = fg_mean - background_penalty * bg_mean

    return TargetResolution(
        target_scalar=target,
        target_description=(
            f"mean(sem_fg) - {background_penalty}*mean(sem_bg) = {target.item():.4f}  "
            f"(fg={fg_mean.item():.4f}, bg={bg_mean.item():.4f}, "
            f"fg_frac={fg_info.foreground_area_fraction:.3f})"
        ),
        target_mask_index=target_mask_index,
        query_index=qi,
        source="semantic_seg",
        requires_grad=target.requires_grad,
        grad_fn=str(target.grad_fn) if target.grad_fn else None,
        warnings=warns,
        _foreground_info=fg_info,
    )


# ---------------------------------------------------------------------------
# Debug: output tensor inventory
# ---------------------------------------------------------------------------


def inventory_output_tensors(
    outputs: Dict[str, Any],
    prefix: str = "",
) -> list[dict]:
    """Recursively catalogue every tensor in *outputs*.

    Returns a list of dicts with ``path``, ``shape``, ``dtype``,
    ``requires_grad``, ``grad_fn``.
    """
    inventory: list[dict] = []
    for key, val in outputs.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, torch.Tensor):
            inventory.append({
                "path": path,
                "shape": list(val.shape),
                "dtype": str(val.dtype),
                "requires_grad": val.requires_grad,
                "grad_fn": str(val.grad_fn) if val.grad_fn else None,
            })
        elif isinstance(val, dict):
            inventory.extend(inventory_output_tensors(val, prefix=path))
        elif isinstance(val, (list, tuple)):
            for i, item in enumerate(val):
                if isinstance(item, torch.Tensor):
                    inventory.append({
                        "path": f"{path}[{i}]",
                        "shape": list(item.shape),
                        "dtype": str(item.dtype),
                        "requires_grad": item.requires_grad,
                        "grad_fn": str(item.grad_fn) if item.grad_fn else None,
                    })
    return inventory


def save_tensor_inventory(
    outputs: Dict[str, Any],
    path: str | Path,
) -> Path:
    """Save the tensor inventory as JSON."""
    path = Path(path)
    inv = inventory_output_tensors(outputs)
    with open(path, "w") as f:
        json.dump(inv, f, indent=2)
    return path
