#!/usr/bin/env python3
"""Aggregate per-layer / per-head / per-token PAM output images into
cohesive stitched grids for easy visual inspection.

Produces:
  1. Per-token grids  — rows=layers(0-5), cols=heads(0-7)
     → One image per token showing how each head behaves across layers.
  2. Per-layer grids  — rows=tokens, cols=heads(0-7)
     → One image per layer showing all tokens side-by-side.
  3. Summary grid     — rows=tokens, cols=layers; cells = head-mean overlay
     → Compact single-page overview of all mean reductions.
  4. Layer-max summary — single row per token of the layer_max aggregations.

All outputs are saved into <output_dir>/aggregated/.

Usage:
    python aggregate_outputs.py outputs/bus
    python aggregate_outputs.py outputs/bus --cell-width 320 --quality 90
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Default thumbnail cell width — keeps collages manageable.
DEFAULT_CELL_WIDTH = 280


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def discover_structure(output_dir: Path) -> dict:
    """Scan the output directory and return a structured description."""
    layers = sorted(
        [d.name for d in output_dir.iterdir()
         if d.is_dir() and d.name.startswith("layer_") and d.name != "layer_max"],
        key=lambda x: int(x.split("_")[1]),
    )
    has_layer_max = (output_dir / "layer_max").is_dir()

    # Discover tokens from the first layer directory.
    tokens = []
    if layers:
        first_layer = output_dir / layers[0]
        pattern = re.compile(r"(token_\d+_txt_\w+)_head_mean\.png")
        for f in sorted(first_layer.iterdir()):
            m = pattern.match(f.name)
            if m:
                tokens.append(m.group(1))

    # Discover heads from the first token's heads/ subdirectory.
    num_heads = 0
    if layers and tokens:
        heads_dir = output_dir / layers[0] / f"{tokens[0]}_heads"
        if heads_dir.is_dir():
            num_heads = len(list(heads_dir.glob("head_*.png")))

    return {
        "layers": layers,           # ["layer_0", ..., "layer_5"]
        "tokens": tokens,           # ["token_00_txt_yellow", ...]
        "num_heads": num_heads,      # 8
        "has_layer_max": has_layer_max,
    }


def token_display_name(token_key: str) -> str:
    """token_00_txt_yellow → 'yellow (t0)'."""
    m = re.match(r"token_(\d+)_txt_(\w+)", token_key)
    if m:
        return f"{m.group(2)} (t{int(m.group(1))})"
    return token_key


def _try_load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a sans font; fall back to default."""
    if bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _load_and_resize(path: Path, cell_w: int) -> Image.Image | None:
    """Load an image and resize to cell_w keeping aspect ratio."""
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    w, h = img.size
    cell_h = int(h * cell_w / w)
    return img.resize((cell_w, cell_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Grid stitching
# ---------------------------------------------------------------------------

def stitch_grid(
    images: list[list[Image.Image | None]],
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    padding: int = 3,
    bg_color: tuple = (255, 255, 255),
    text_color: tuple = (33, 33, 33),
    border_color: tuple = (200, 200, 200),
) -> Image.Image:
    """Stitch a 2-D grid of PIL images — matplotlib-style white bg with labels."""
    nrows = len(images)
    ncols = max(len(row) for row in images) if nrows else 0

    # Determine cell size from the first non-None image.
    cell_w, cell_h = 0, 0
    for row in images:
        for img in row:
            if img is not None:
                cell_w, cell_h = img.size
                break
        if cell_w:
            break
    if cell_w == 0:
        return Image.new("RGB", (100, 100), bg_color)

    # Font sizes scaled to cell width.
    title_font = _try_load_font(max(14, cell_w // 12), bold=True)
    subtitle_font = _try_load_font(max(11, cell_w // 16))
    label_font = _try_load_font(max(10, cell_w // 18), bold=True)

    # Measure label areas.
    row_label_w = 0
    if row_labels:
        row_label_w = max(label_font.getlength(l) for l in row_labels) + 8
        row_label_w = int(row_label_w)

    col_label_h = int(label_font.size * 1.6) if col_labels else 0
    title_h = 0
    if title:
        title_h += int(title_font.size * 1.5)
    if subtitle:
        title_h += int(subtitle_font.size * 1.4)
    if title or subtitle:
        title_h += 6  # bottom margin

    canvas_w = row_label_w + ncols * cell_w + (ncols - 1) * padding
    canvas_h = title_h + col_label_h + nrows * cell_h + (nrows - 1) * padding

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Title block (top-left, matplotlib-style).
    ty = 4
    if title:
        draw.text((row_label_w, ty), title, fill=text_color, font=title_font)
        ty += int(title_font.size * 1.5)
    if subtitle:
        draw.text((row_label_w, ty), subtitle, fill=(100, 100, 100), font=subtitle_font)

    # Column labels (centered above each column).
    if col_labels:
        for c, lbl in enumerate(col_labels):
            x = row_label_w + c * (cell_w + padding)
            tw = label_font.getlength(lbl)
            draw.text((x + (cell_w - tw) / 2, title_h + 2), lbl,
                      fill=text_color, font=label_font)

    # Row labels & images.
    for r, row in enumerate(images):
        y = title_h + col_label_h + r * (cell_h + padding)

        if row_labels and r < len(row_labels):
            lbl = row_labels[r]
            # Vertically center the row label.
            draw.text((4, y + cell_h // 2 - label_font.size // 2), lbl,
                      fill=text_color, font=label_font)

        for c, img in enumerate(row):
            x = row_label_w + c * (cell_w + padding)
            if img is not None:
                if img.size != (cell_w, cell_h):
                    img = img.resize((cell_w, cell_h), Image.LANCZOS)
                canvas.paste(img, (x, y))
                # Thin border around each cell.
                draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1],
                               outline=border_color, width=1)
            else:
                # Empty cell — light gray fill.
                draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1],
                               fill=(240, 240, 240), outline=border_color)

    return canvas


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

def build_per_token_grid(output_dir: Path, info: dict, token_key: str,
                         cell_w: int, padding: int) -> Image.Image:
    """Rows = layers, Cols = heads.  One grid per token."""
    rows = []
    for layer in info["layers"]:
        heads_dir = output_dir / layer / f"{token_key}_heads"
        row = []
        for h in range(info["num_heads"]):
            row.append(_load_and_resize(heads_dir / f"head_{h}.png", cell_w))
        rows.append(row)

    return stitch_grid(
        rows,
        row_labels=[l.replace("layer_", "L") for l in info["layers"]],
        col_labels=[f"Head {h}" for h in range(info["num_heads"])],
        title=f"Token: {token_display_name(token_key)}",
        subtitle="Rows = transformer layers  |  Cols = attention heads",
        padding=padding,
    )


def build_per_layer_grid(output_dir: Path, info: dict, layer: str,
                         cell_w: int, padding: int) -> Image.Image:
    """Rows = tokens, Cols = heads.  One grid per layer."""
    rows = []
    for token_key in info["tokens"]:
        heads_dir = output_dir / layer / f"{token_key}_heads"
        row = []
        for h in range(info["num_heads"]):
            row.append(_load_and_resize(heads_dir / f"head_{h}.png", cell_w))
        rows.append(row)

    return stitch_grid(
        rows,
        row_labels=[token_display_name(t) for t in info["tokens"]],
        col_labels=[f"Head {h}" for h in range(info["num_heads"])],
        title=layer.replace("_", " ").title(),
        subtitle="Rows = tokens  |  Cols = attention heads",
        padding=padding,
    )


def build_summary_grid(output_dir: Path, info: dict,
                       reduction: str, cell_w: int, padding: int) -> Image.Image:
    """Rows = tokens, Cols = layers.  Cells = head-mean or head-max overlay."""
    rows = []
    for token_key in info["tokens"]:
        row = []
        for layer in info["layers"]:
            row.append(_load_and_resize(
                output_dir / layer / f"{token_key}_head_{reduction}.png", cell_w))
        if info["has_layer_max"]:
            row.append(_load_and_resize(
                output_dir / "layer_max" / f"{token_key}_head_{reduction}.png", cell_w))
        rows.append(row)

    col_labels = [l.replace("layer_", "L") for l in info["layers"]]
    if info["has_layer_max"]:
        col_labels.append("L-max")

    return stitch_grid(
        rows,
        row_labels=[token_display_name(t) for t in info["tokens"]],
        col_labels=col_labels,
        title=f"Summary — head {reduction}",
        subtitle="Rows = tokens  |  Cols = layers (head-reduced)",
        padding=padding,
    )


def build_detection_strip(output_dir: Path, info: dict,
                          cell_w: int, padding: int) -> Image.Image | None:
    """Optional strip: detection_boxes + mask(s) side by side."""
    imgs: list[Image.Image | None] = []
    labels: list[str] = []
    det = output_dir / "detection_boxes.png"
    if det.exists():
        imgs.append(_load_and_resize(det, cell_w))
        labels.append("Detections")
    for mask_path in sorted(output_dir.glob("mask_*.png")):
        imgs.append(_load_and_resize(mask_path, cell_w))
        labels.append(mask_path.stem)
    if not imgs:
        return None
    return stitch_grid(
        [imgs],
        col_labels=labels,
        title="Detection & Masks",
        padding=padding,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save(img: Image.Image, path: Path, quality: int = 85) -> None:
    """Save as JPEG for small file sizes."""
    img.save(path, "JPEG", quality=quality, optimize=True)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate PAM per-layer/head/token images into stitched grids."
    )
    parser.add_argument("output_dir", type=str,
                        help="Path to the PAM output directory (e.g. outputs/bus).")
    parser.add_argument("--cell-width", type=int, default=DEFAULT_CELL_WIDTH,
                        help=f"Thumbnail width per cell in px (default {DEFAULT_CELL_WIDTH}).")
    parser.add_argument("--padding", type=int, default=3,
                        help="Pixels of padding between grid cells.")
    parser.add_argument("--quality", type=int, default=85,
                        help="JPEG quality (1-100, default 85).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    info = discover_structure(output_dir)
    print(f"Discovered: {len(info['layers'])} layers, "
          f"{len(info['tokens'])} tokens, "
          f"{info['num_heads']} heads, "
          f"layer_max={'yes' if info['has_layer_max'] else 'no'}")
    print(f"Cell width: {args.cell_width}px  |  JPEG quality: {args.quality}")

    agg_dir = output_dir / "aggregated"
    agg_dir.mkdir(exist_ok=True)

    cw, pad, q = args.cell_width, args.padding, args.quality

    # 1. Per-token grids (the most useful view for your use-case).
    for token_key in info["tokens"]:
        grid = build_per_token_grid(output_dir, info, token_key, cw, pad)
        name = token_display_name(token_key).replace(" ", "_").replace("(", "").replace(")", "")
        path = agg_dir / f"per_token_{name}.jpg"
        _save(grid, path, q)
        print(f"  Saved {path}  ({path.stat().st_size // 1024} KB)")

    # 2. Per-layer grids.
    for layer in info["layers"]:
        grid = build_per_layer_grid(output_dir, info, layer, cw, pad)
        path = agg_dir / f"per_{layer}.jpg"
        _save(grid, path, q)
        print(f"  Saved {path}  ({path.stat().st_size // 1024} KB)")

    # 3. Summary grids (mean + max reductions).
    for reduction in ("mean", "max"):
        grid = build_summary_grid(output_dir, info, reduction, cw, pad)
        path = agg_dir / f"summary_{reduction}.jpg"
        _save(grid, path, q)
        print(f"  Saved {path}  ({path.stat().st_size // 1024} KB)")

    # 4. Detection & mask strip.
    det_strip = build_detection_strip(output_dir, info, cw, pad)
    if det_strip is not None:
        path = agg_dir / f"detection_masks.jpg"
        _save(det_strip, path, q)
        print(f"  Saved {path}  ({path.stat().st_size // 1024} KB)")

    total_kb = sum(f.stat().st_size for f in agg_dir.glob("*.jpg")) // 1024
    print(f"\nAll aggregated images saved to {agg_dir}/")
    print(f"Total: {len(list(agg_dir.glob('*.jpg')))} images  ({total_kb} KB)")


if __name__ == "__main__":
    main()
