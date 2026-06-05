"""Attribution-flow visualisation utilities.

Produces overlay heatmaps, top-K contributor grids, and summary panels
for the Gradient-Weighted Attribution Flow PAM pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .visualization import make_attention_overlay


# ---------------------------------------------------------------------------
# Percentile normalisation
# ---------------------------------------------------------------------------


def _percentile_normalize(
    arr: np.ndarray,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> np.ndarray:
    """Clip to [lo, hi] percentiles and scale to [0, 1]."""
    if arr.size == 0:
        return arr
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    if hi - lo < 1e-10:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Overlay saving
# ---------------------------------------------------------------------------


def save_attribution_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    path: str | Path,
    *,
    alpha: float = 0.5,
    colormap: str = "inferno",
) -> Path:
    """Save a single heatmap overlay with percentile normalisation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normed = _percentile_normalize(heatmap)
    overlay = make_attention_overlay(image, normed, alpha=alpha, colormap=colormap)
    overlay.save(path)
    return path


def save_signed_overlays(
    image: Image.Image,
    positive: np.ndarray,
    negative: np.ndarray,
    output_dir: str | Path,
    prefix: str,
    *,
    alpha: float = 0.5,
) -> dict[str, Path]:
    """Save positive and negative overlay pair."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    pos_path = output_dir / f"{prefix}_positive_overlay.png"
    save_attribution_overlay(image, positive, pos_path, alpha=alpha, colormap="inferno")
    paths["positive"] = pos_path

    neg_path = output_dir / f"{prefix}_negative_overlay.png"
    save_attribution_overlay(image, negative, neg_path, alpha=alpha, colormap="coolwarm")
    paths["negative"] = neg_path

    return paths


# ---------------------------------------------------------------------------
# Heatmap-only save
# ---------------------------------------------------------------------------


def save_heatmap_png(
    heatmap: np.ndarray,
    path: str | Path,
    colormap: str = "inferno",
) -> Path:
    """Save a raw heatmap (no image blend) as PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normed = _percentile_normalize(heatmap)
    cmap = cm.get_cmap(colormap)
    rgba = cmap(normed)[:, :, :3]
    img = Image.fromarray((rgba * 255).astype(np.uint8))
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Top-K grids
# ---------------------------------------------------------------------------


def save_top_k_grid(
    image: Image.Image,
    maps: list[np.ndarray],
    titles: list[str],
    path: str | Path,
    *,
    cols: int = 6,
    alpha: float = 0.5,
    colormap: str = "inferno",
) -> Path:
    """Save a grid of top-K attribution overlay panels."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    K = len(maps)
    if K == 0:
        return path

    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("off")
        if idx < K:
            normed = _percentile_normalize(maps[idx])
            overlay = make_attention_overlay(image, normed, alpha=alpha, colormap=colormap)
            ax.imshow(np.array(overlay))
            ax.set_title(titles[idx], fontsize=7, pad=2)

    fig.tight_layout(pad=0.5)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def make_top_group_a_grid(
    image: Image.Image,
    per_layer_token_maps: np.ndarray,
    token_labels: list[str],
    path: str | Path,
    top_k: int = 24,
    **kwargs: Any,
) -> Path:
    """Save top-K Group A prompt token contributor grid.

    Parameters
    ----------
    per_layer_token_maps : ndarray  [L, H, T, Gh, Gw]
    """
    L, H, T, Gh, Gw = per_layer_token_maps.shape
    # Sum over spatial dims to rank.
    scores = per_layer_token_maps.sum(axis=(3, 4))  # [L, H, T]
    flat_idx = np.argsort(scores.ravel())[::-1][:top_k]

    maps_list: list[np.ndarray] = []
    titles: list[str] = []
    for fi in flat_idx:
        l = fi // (H * T)
        h = (fi % (H * T)) // T
        t = fi % T
        label = token_labels[t] if t < len(token_labels) else f"p{t:02d}"
        maps_list.append(per_layer_token_maps[l, h, t])
        titles.append(f"A L{l} H{h} {label}\npos={scores[l, h, t]:.2e}")

    return save_top_k_grid(image, maps_list, titles, path, **kwargs)


def make_top_group_b_grid(
    image: Image.Image,
    per_layer_head_maps: np.ndarray,
    object_query_index: int,
    decoder_token_index: int,
    path: str | Path,
    top_k: int = 24,
    **kwargs: Any,
) -> Path:
    """Save top-K Group B decoder head contributor grid.

    Parameters
    ----------
    per_layer_head_maps : ndarray  [L, H, Gh, Gw]
    """
    L, H, Gh, Gw = per_layer_head_maps.shape
    scores = per_layer_head_maps.sum(axis=(2, 3))  # [L, H]
    flat_idx = np.argsort(scores.ravel())[::-1][:top_k]

    maps_list: list[np.ndarray] = []
    titles: list[str] = []
    for fi in flat_idx:
        l = fi // H
        h = fi % H
        maps_list.append(per_layer_head_maps[l, h])
        titles.append(
            f"B L{l} H{h} objQ{object_query_index} / dec{decoder_token_index}"
            f"\npos={scores[l, h]:.2e}"
        )

    return save_top_k_grid(image, maps_list, titles, path, **kwargs)


# ---------------------------------------------------------------------------
# Summary panel
# ---------------------------------------------------------------------------


def save_summary_panel(
    image: Image.Image,
    group_a_overlay: np.ndarray | None,
    group_b_overlay: np.ndarray | None,
    joint_overlay: np.ndarray | None,
    path: str | Path,
    *,
    mask_preview: np.ndarray | None = None,
    alpha: float = 0.5,
) -> Path:
    """Save a compact summary panel."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, Image.Image | None]] = [
        ("Original", image),
    ]
    if mask_preview is not None:
        mask_img = Image.fromarray((mask_preview * 255).astype(np.uint8)).convert("RGB")
        panels.append(("Selected mask", mask_img))
    if group_a_overlay is not None:
        normed = _percentile_normalize(group_a_overlay)
        panels.append(("Group A prompt-write", make_attention_overlay(image, normed, alpha)))
    if group_b_overlay is not None:
        normed = _percentile_normalize(group_b_overlay)
        panels.append(("Group B query-read", make_attention_overlay(image, normed, alpha)))
    if joint_overlay is not None:
        normed = _percentile_normalize(joint_overlay)
        panels.append(("Joint", make_attention_overlay(image, normed, alpha)))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(np.array(img))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_decoder_token_comparison(
    image: Image.Image,
    dec_attributions: list,
    decoder_token_a: int,
    decoder_token_b: int,
    spatial_grid: tuple[int, int],
    path: str | Path,
    alpha: float = 0.5,
) -> Path:
    """Side-by-side comparison of Group B spatial maps for two decoder tokens.

    Shows ``decoder_token_a`` (left) vs ``decoder_token_b`` (right) to
    visually confirm the correct token is selected.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Gh, Gw = spatial_grid

    def _spatial_map(dti: int) -> np.ndarray:
        total = np.zeros((Gh, Gw), dtype=np.float32)
        for attr in dec_attributions:
            pos = attr.positive_edge_attr[0, :, dti, :]  # [H, Ni]
            total += pos.sum(dim=0).reshape(Gh, Gw).cpu().float().detach().numpy()
        return total

    map_a = _spatial_map(decoder_token_a)
    map_b = _spatial_map(decoder_token_b)

    # Normalise both on the same scale.
    vmax = max(map_a.max(), map_b.max(), 1e-10)
    norm_a = map_a / vmax
    norm_b = map_b / vmax

    ov_a = make_attention_overlay(image, norm_a, alpha)
    ov_b = make_attention_overlay(image, norm_b, alpha)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.array(ov_a))
    axes[0].set_title(
        f"decoder token {decoder_token_a} (objQ{decoder_token_a - 1})"
        f"\nmass={map_a.sum():.4e}", fontsize=10,
    )
    axes[0].axis("off")
    axes[1].imshow(np.array(ov_b))
    axes[1].set_title(
        f"decoder token {decoder_token_b} (objQ{decoder_token_b - 1})"
        f"\nmass={map_b.sum():.4e}", fontsize=10,
    )
    axes[1].axis("off")
    fig.suptitle("Decoder Token Comparison (Group B positive attribution)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_full_mask_summary_panel(
    image: Image.Image,
    mask_preview: np.ndarray | None,
    group_a_overlay: np.ndarray | None,
    group_b_overlay: np.ndarray | None,
    joint_product_overlay: np.ndarray | None,
    joint_sum_overlay: np.ndarray | None,
    path: str | Path,
    *,
    alpha: float = 0.5,
) -> Path:
    """Save a 6-panel summary: Original, Mask, Group A, Group B, Joint product, Joint sum."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    panels: list[tuple[str, Image.Image | None]] = [("Original", image)]
    if mask_preview is not None:
        mask_img = Image.fromarray((mask_preview * 255).astype(np.uint8)).convert("RGB")
        panels.append(("Selected mask", mask_img))
    if group_a_overlay is not None:
        normed = _percentile_normalize(group_a_overlay)
        panels.append(("Group A prompt-write", make_attention_overlay(image, normed, alpha)))
    if group_b_overlay is not None:
        normed = _percentile_normalize(group_b_overlay)
        panels.append(("Group B query-read", make_attention_overlay(image, normed, alpha)))
    if joint_product_overlay is not None:
        normed = _percentile_normalize(joint_product_overlay)
        panels.append(("Joint product", make_attention_overlay(image, normed, alpha)))
    if joint_sum_overlay is not None:
        normed = _percentile_normalize(joint_sum_overlay)
        panels.append(("Joint sum", make_attention_overlay(image, normed, alpha)))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(np.array(img))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_objective_comparison(
    image: Image.Image,
    results: dict[str, dict[str, np.ndarray | None]],
    path: str | Path,
    *,
    alpha: float = 0.5,
) -> Path:
    """Save a comparison grid across multiple target objectives.

    Parameters
    ----------
    results : dict
        ``{objective_name: {"group_b": ndarray, "joint_product": ndarray, "joint_sum": ndarray}}``
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    objectives = list(results.keys())
    row_labels = ["Group B query-read", "Joint product", "Joint sum"]
    row_keys = ["group_b", "joint_product", "joint_sum"]

    n_cols = len(objectives) + 1  # +1 for row labels
    n_rows = len(row_labels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r in range(n_rows):
        # Row label in first column.
        axes[r, 0].text(
            0.5, 0.5, row_labels[r], fontsize=11, ha="center", va="center",
            transform=axes[r, 0].transAxes,
        )
        axes[r, 0].axis("off")

        for c, obj_name in enumerate(objectives, 1):
            ax = axes[r, c]
            arr = results[obj_name].get(row_keys[r])
            if arr is not None:
                normed = _percentile_normalize(arr)
                overlay = make_attention_overlay(image, normed, alpha)
                ax.imshow(np.array(overlay))
            ax.set_title(obj_name if r == 0 else "", fontsize=9)
            ax.axis("off")

    fig.suptitle("Target Objective Comparison", fontsize=13)
    fig.tight_layout(pad=0.5)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
