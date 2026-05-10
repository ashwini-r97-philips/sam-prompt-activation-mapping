"""Heatmap visualisation and overlay utilities.

All rendering uses PIL, matplotlib (for colour-maps), and numpy — no OpenCV
dependency.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from .tokenization import safe_filename


# ---------------------------------------------------------------------------
# Spatial grid inference
# ---------------------------------------------------------------------------


def infer_spatial_grid(
    num_image_tokens: int,
    metadata: dict | None = None,
) -> tuple[int, int] | None:
    """Infer the (H, W) spatial grid from the number of image tokens.

    Parameters
    ----------
    num_image_tokens : int
        Total number of image-token positions from the attention tensor.
    metadata : dict | None
        Optional dict that may contain ``"feat_h"`` and ``"feat_w"`` keys
        from the model's feature map.

    Returns
    -------
    (H, W) or None
        ``None`` if the grid cannot be inferred (non-square token count and
        no metadata).
    """
    if metadata is not None:
        h = metadata.get("feat_h")
        w = metadata.get("feat_w")
        if h is not None and w is not None and h * w == num_image_tokens:
            return (int(h), int(w))

    sqrt = math.isqrt(num_image_tokens)
    if sqrt * sqrt == num_image_tokens:
        return (sqrt, sqrt)

    warnings.warn(
        f"Cannot infer spatial grid: {num_image_tokens} image tokens is not "
        "a perfect square and no feature-map metadata was provided. "
        "Heatmaps will be saved as 1-D bar plots.",
        stacklevel=2,
    )
    return None


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------


def make_attention_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Blend a normalised heatmap onto *image*.

    Parameters
    ----------
    image : PIL.Image
        Original RGB image.
    heatmap : ndarray
        2-D array normalised to [0, 1], spatial dimensions need not match
        image — it will be resized.
    alpha : float
        Transparency of the heatmap overlay.
    colormap : str
        Matplotlib colormap name.

    Returns
    -------
    PIL.Image
        The blended overlay image.
    """
    # Resize heatmap to image size (W, H for PIL).
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(image.size, resample=Image.BILINEAR)

    # Apply colormap via matplotlib.
    cmap = cm.get_cmap(colormap)
    heatmap_rgba = cmap(np.array(heatmap_img) / 255.0)  # [H, W, 4] float
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    # Blend.
    overlay = Image.blend(image.convert("RGB"), heatmap_pil, alpha)
    return overlay


# ---------------------------------------------------------------------------
# 1-D fallback plot
# ---------------------------------------------------------------------------


def _save_1d_heatmap(
    heatmap_1d: np.ndarray,
    title: str,
    path: Path,
) -> None:
    """Save a simple bar-plot for non-square token counts."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.bar(range(len(heatmap_1d)), heatmap_1d)
    ax.set_title(title)
    ax.set_xlabel("Image token index")
    ax.set_ylabel("Attention")
    fig.tight_layout()
    fig.savefig(str(path), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------


def _make_comparison(
    original: Image.Image,
    raw_overlay: Image.Image,
    eff_overlay: Image.Image,
) -> Image.Image:
    """Place original, raw-overlay, and effective-overlay side by side."""
    w, h = original.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(original.convert("RGB"), (0, 0))
    canvas.paste(raw_overlay, (w, 0))
    canvas.paste(eff_overlay, (2 * w, 0))
    return canvas


# ---------------------------------------------------------------------------
# Main save routine
# ---------------------------------------------------------------------------


def save_token_heatmaps(
    image: Image.Image,
    raw_attention: torch.Tensor,
    effective_attention: torch.Tensor,
    token_labels: list[str],
    output_dir: str | Path,
    head_reduction: str = "mean",
    alpha: float = 0.5,
    no_overlay: bool = False,
    max_tokens: int | None = None,
    resize_long_side: int | None = None,
    spatial_grid: tuple[int, int] | None = None,
) -> list[dict]:
    """Save per-token heatmap images to *output_dir*.

    Parameters
    ----------
    image : PIL.Image
        Original input image.
    raw_attention : Tensor  [B, H, T_prompt, T_image]
    effective_attention : Tensor  [B, H, T_prompt, T_image]
    token_labels : list[str]
    output_dir : str | Path
    head_reduction : str  ``"mean"`` or ``"max"``
    alpha : float
    no_overlay : bool  If True, save heatmaps only (no blending).
    max_tokens : int | None  Cap on number of tokens visualised.
    resize_long_side : int | None  Resize image for visualisation only.
    spatial_grid : (H, W) | None  From :func:`infer_spatial_grid`.

    Returns
    -------
    list of dicts with ``token_index``, ``label``, ``raw_path``,
    ``effective_path``, ``comparison_path``.
    """
    from .effective_attention import reduce_heads

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional visualisation resize.
    vis_image = image.copy()
    if resize_long_side is not None:
        w, h = vis_image.size
        scale = resize_long_side / max(w, h)
        vis_image = vis_image.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )

    num_tokens = raw_attention.shape[2]
    if max_tokens is not None:
        num_tokens = min(num_tokens, max_tokens)

    saved: list[dict] = []

    for t_idx in range(num_tokens):
        label = token_labels[t_idx] if t_idx < len(token_labels) else f"token_{t_idx:02d}"
        safe_label = safe_filename(label)

        # Reduce heads → 1-D heatmaps [T_image].
        raw_1d = reduce_heads(raw_attention, t_idx, mode=head_reduction)
        eff_1d = reduce_heads(effective_attention, t_idx, mode=head_reduction)

        raw_np = raw_1d.cpu().float().numpy()
        eff_np = eff_1d.cpu().float().numpy()

        entry: dict = {
            "token_index": t_idx,
            "label": label,
        }

        if spatial_grid is not None:
            H, W = spatial_grid
            raw_2d = raw_np.reshape(H, W)
            eff_2d = eff_np.reshape(H, W)

            # Normalise to [0, 1].
            raw_2d = _normalise(raw_2d)
            eff_2d = _normalise(eff_2d)

            if no_overlay:
                raw_path = output_dir / f"raw_token_{t_idx:02d}_{safe_label}.png"
                eff_path = output_dir / f"effective_token_{t_idx:02d}_{safe_label}.png"
                Image.fromarray((raw_2d * 255).astype(np.uint8)).save(raw_path)
                Image.fromarray((eff_2d * 255).astype(np.uint8)).save(eff_path)
            else:
                raw_overlay = make_attention_overlay(vis_image, raw_2d, alpha)
                eff_overlay = make_attention_overlay(vis_image, eff_2d, alpha)

                raw_path = output_dir / f"raw_token_{t_idx:02d}_{safe_label}.png"
                eff_path = output_dir / f"effective_token_{t_idx:02d}_{safe_label}.png"
                raw_overlay.save(raw_path)
                eff_overlay.save(eff_path)

                # Side-by-side comparison.
                comp = _make_comparison(vis_image, raw_overlay, eff_overlay)
                comp_path = output_dir / f"comparison_token_{t_idx:02d}_{safe_label}.png"
                comp.save(comp_path)
                entry["comparison_path"] = str(comp_path)

            entry["raw_path"] = str(raw_path)
            entry["effective_path"] = str(eff_path)

        else:
            # Fallback: 1-D bar plot.
            raw_path = output_dir / f"raw_token_{t_idx:02d}_{safe_label}_1d.png"
            eff_path = output_dir / f"effective_token_{t_idx:02d}_{safe_label}_1d.png"
            _save_1d_heatmap(raw_np, f"Raw attn — {label}", raw_path)
            _save_1d_heatmap(eff_np, f"Effective attn — {label}", eff_path)
            entry["raw_path"] = str(raw_path)
            entry["effective_path"] = str(eff_path)

        saved.append(entry)

    return saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
