"""Compare instance-mask and semantic-segmentation attribution outputs.

Usage:
    python -m pam.compare_outputs \
        --instance-dir outputs/bus_mask_contrastive_attribution \
        --semantic-dir outputs/bus_semantic_contrastive_attribution \
        --output comparison_panel.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _load_or_none(path: Path) -> Image.Image | None:
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


def build_comparison_panel(
    instance_dir: Path,
    semantic_dir: Path,
    output_path: Path,
) -> Path:
    """Build a 6-panel comparison from two attribution run directories.

    Layout (2 rows × 3 cols):
        Row 1: Original | Instance joint_sum | Instance joint_product
        Row 2: Selected mask | Semantic joint_sum | Semantic joint_product
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to load original image from either dir.
    original = _load_or_none(instance_dir / "original.png")
    if original is None:
        original = _load_or_none(semantic_dir / "original.png")

    mask_preview = _load_or_none(instance_dir / "mask_preview.png")
    if mask_preview is None:
        mask_preview = _load_or_none(semantic_dir / "mask_preview.png")

    inst_joint_sum = _load_or_none(instance_dir / "joint_sum_positive_overlay.png")
    inst_joint_prod = _load_or_none(instance_dir / "joint_product_positive_overlay.png")
    sem_joint_sum = _load_or_none(semantic_dir / "joint_sum_positive_overlay.png")
    sem_joint_prod = _load_or_none(semantic_dir / "joint_product_positive_overlay.png")

    panels = [
        [("Original", original),
         ("Instance joint_sum", inst_joint_sum),
         ("Instance joint_product", inst_joint_prod)],
        [("Selected mask", mask_preview),
         ("Semantic joint_sum", sem_joint_sum),
         ("Semantic joint_product", sem_joint_prod)],
    ]

    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            title, img = panels[r][c]
            if img is not None:
                ax.imshow(np.array(img))
            else:
                ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                        fontsize=12, transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    fig.suptitle("Instance mask vs Semantic segmentation attribution", fontsize=12)
    fig.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison panel saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare instance-mask and semantic-segmentation attribution outputs.",
    )
    parser.add_argument(
        "--instance-dir", type=Path, required=True,
        help="Directory with instance-mask attribution outputs.",
    )
    parser.add_argument(
        "--semantic-dir", type=Path, required=True,
        help="Directory with semantic-segmentation attribution outputs.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("comparison_panel.png"),
        help="Output path for the comparison panel image.",
    )
    args = parser.parse_args()

    build_comparison_panel(args.instance_dir, args.semantic_dir, args.output)


if __name__ == "__main__":
    main()
