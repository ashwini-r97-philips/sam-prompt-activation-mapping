"""Prompt Activation Mapping CLI for SAM 3.

Two modes
---------
1. **List modules** — discover and print candidate attention modules::

       python -m pam.cli --list-attention-modules --device cuda

2. **Map** — run one forward pass and save attention heatmaps::

       python -m pam.cli \\
           --image path/to/image.jpg \\
           --prompt "yellow school bus" \\
           --output-dir outputs/bus_pam \\
           --device cuda \\
           --save-debug
"""

from __future__ import annotations

import argparse
import math as _math
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pam",
        description="Prompt Activation Mapping for SAM 3 — "
        "cross-attention & effective-attention heatmaps.",
    )

    # -- mode selector -------------------------------------------------------
    p.add_argument(
        "--list-attention-modules",
        action="store_true",
        help="Print candidate attention modules and exit.",
    )

    # -- inputs --------------------------------------------------------------
    p.add_argument("--image", type=str, help="Path to input image.")
    p.add_argument("--prompt", type=str, help='Text prompt, e.g. "yellow school bus".')

    # -- outputs -------------------------------------------------------------
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pam_run",
        help="Directory for output files.",
    )

    # -- device --------------------------------------------------------------
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu.",
    )

    # -- module selection ----------------------------------------------------
    p.add_argument(
        "--module-filter",
        type=str,
        default=None,
        help="Substring filter for candidate attention modules "
        "(default: auto-detect cross-attention).",
    )
    p.add_argument(
        "--module-name",
        type=str,
        default=None,
        help="Exact or substring module name override.",
    )
    p.add_argument(
        "--layer-index",
        type=str,
        default="first",
        help='"first", "last", or integer index among matching modules.',
    )

    # -- head reduction & visualisation --------------------------------------
    p.add_argument(
        "--head-reduction",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="How to reduce across attention heads.",
    )
    p.add_argument("--alpha", type=float, default=0.5, help="Overlay transparency.")
    p.add_argument(
        "--no-overlay",
        action="store_true",
        help="Save heatmaps only (no blending with original image).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to visualise.",
    )
    p.add_argument(
        "--resize-long-side",
        type=int,
        default=None,
        help="Resize image (long side) for visualisation only.",
    )

    # -- debug ---------------------------------------------------------------
    p.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug.json and debug_tensors.npz.",
    )

    return p


# ---------------------------------------------------------------------------
# Default cross-attention filter heuristic
# ---------------------------------------------------------------------------

_DEFAULT_CROSS_ATTN_SUBSTRINGS = [
    "cross_attn",
    "ca_text",
    "cross_attend",
    "cross_attention",
]

# Preferred module path substrings, in priority order.  The DETR
# transformer encoder is where the full prompt (text + geometry)
# cross-attends to image features — this is the most useful layer for
# prompt activation mapping.
_PREFERRED_MODULE_HINTS = [
    "transformer.encoder",   # DETR encoder: prompt → image cross-attn
    "transformer.decoder",   # DETR decoder: queries → text
]


def _auto_filter_candidates(
    candidates: list[tuple[str, "torch.nn.Module"]],
) -> list[tuple[str, "torch.nn.Module"]]:
    """Keep only candidates whose names contain a cross-attention hint,
    prioritising the DETR transformer encoder over geometry/other encoders."""
    filtered = []
    for name, mod in candidates:
        lower = name.lower()
        if any(hint in lower for hint in _DEFAULT_CROSS_ATTN_SUBSTRINGS):
            filtered.append((name, mod))

    if not filtered:
        return candidates  # fall back to all if none match

    # Among cross-attn modules, prefer the DETR transformer encoder.
    for hint in _PREFERRED_MODULE_HINTS:
        preferred = [(n, m) for n, m in filtered if hint in n]
        if preferred:
            return preferred

    return filtered


# ---------------------------------------------------------------------------
# Mode: list attention modules
# ---------------------------------------------------------------------------


def cmd_list_modules(args: argparse.Namespace) -> None:
    from .sam3_loader import load_sam3_image_model
    from .attention_capture import discover_attention_modules, _module_summary

    print(f"Loading SAM 3 image model on {args.device} …")
    model, _processor = load_sam3_image_model(device=args.device)

    candidates = discover_attention_modules(model, filter_text=args.module_filter)
    if not candidates:
        print("No attention modules found.")
        return

    print(f"\nFound {len(candidates)} attention module(s):\n")
    for name, mod in candidates:
        print(f"  {_module_summary(name, mod)}")
    print()


# ---------------------------------------------------------------------------
# Save detection / segmentation output
# ---------------------------------------------------------------------------


def _save_detection_output(state: dict, image_path: str, output_dir: str) -> None:
    """Save SAM 3 detection output: bounding boxes, masks, and scores."""
    import json
    import numpy as np
    from PIL import Image as PILImage

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boxes = state.get("boxes")
    scores = state.get("scores")
    masks = state.get("masks")

    if boxes is None or len(boxes) == 0:
        print("  detection: no objects detected.")
        return

    boxes_np = boxes.cpu().float().numpy()
    scores_np = scores.cpu().float().numpy()

    print(f"  detection: {len(boxes_np)} object(s) found")

    # Save detection JSON.
    det_info = {
        "image": str(image_path),
        "num_detections": len(boxes_np),
        "detections": [],
    }
    for i in range(len(boxes_np)):
        det_info["detections"].append({
            "box_xyxy": boxes_np[i].tolist(),
            "score": float(scores_np[i]),
        })
    det_json_path = out_dir / "detections.json"
    with open(det_json_path, "w") as f:
        json.dump(det_info, f, indent=2)
    print(f"  detections JSON     : {det_json_path}")

    # Draw boxes on the original image.
    image = PILImage.open(image_path).convert("RGB")
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes_np)):
        x0, y0, x1, y1 = boxes_np[i]
        score = float(scores_np[i])
        draw.rectangle([x0, y0, x1, y1], outline="lime", width=3)
        draw.text((x0 + 4, y0 + 4), f"{score:.2f}", fill="lime")
    det_img_path = out_dir / "detection_boxes.png"
    image.save(det_img_path)
    print(f"  detection image     : {det_img_path}")

    # Save binary masks.
    if masks is not None and len(masks) > 0:
        masks_np = masks.cpu().numpy()  # [N, 1, H, W] bool
        for i in range(len(masks_np)):
            mask_2d = masks_np[i, 0]  # [H, W]
            mask_img = PILImage.fromarray((mask_2d * 255).astype(np.uint8))
            mask_path = out_dir / f"mask_{i:02d}.png"
            mask_img.save(mask_path)
        print(f"  masks saved         : {len(masks_np)} mask(s)")


# ---------------------------------------------------------------------------
# Mode: main mapping pipeline
# ---------------------------------------------------------------------------


def cmd_map(args: argparse.Namespace) -> None:
    from PIL import Image as PILImage

    from .sam3_loader import load_sam3_image_model, run_inference
    from .attention_capture import (
        AttentionCapture,
        discover_attention_modules,
        _module_summary,
    )
    from .effective_attention import compute_effective_attention
    from .tokenization import get_token_labels
    from .visualization import infer_spatial_grid, save_token_heatmaps
    from .debug_utils import save_debug_json, save_debug_tensors

    # -- validate args -------------------------------------------------------
    if not args.image:
        print("ERROR: --image is required.", file=sys.stderr)
        raise SystemExit(1)
    if not args.prompt:
        print("ERROR: --prompt is required.", file=sys.stderr)
        raise SystemExit(1)
    if not Path(args.image).is_file():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        raise SystemExit(1)

    # -- load model ----------------------------------------------------------
    print(f"Loading SAM 3 image model on {args.device} …")
    model, processor = load_sam3_image_model(device=args.device)

    # -- discover cross-attention modules ------------------------------------
    all_candidates = discover_attention_modules(model, filter_text=args.module_filter)
    candidates = all_candidates

    # Apply auto-filter when no explicit filter was given.
    if args.module_filter is None and args.module_name is None:
        candidates = _auto_filter_candidates(all_candidates)

    # Select ALL matching encoder cross-attn modules (multi-layer).
    if args.module_name is not None:
        # User explicitly selected one module — honour that.
        from .attention_capture import select_attention_module
        sel_name, sel_mod = select_attention_module(
            candidates, module_name=args.module_name, layer_index=args.layer_index,
        )
        selected_modules = [(sel_name, sel_mod)]
    else:
        selected_modules = candidates

    print(f"Hooking {len(selected_modules)} cross-attention module(s):")
    for name, mod in selected_modules:
        print(f"  {_module_summary(name, mod)}")

    # -- register hooks on ALL selected modules & run inference --------------
    captures = []
    for name, mod in selected_modules:
        cap = AttentionCapture(name, mod)
        cap.register()
        captures.append(cap)

    print(f"Running inference on {args.image} with prompt: {args.prompt!r} …")
    try:
        state = run_inference(processor, args.image, args.prompt, args.device)
    finally:
        for cap in captures:
            cap.remove()

    # -- save detection output -----------------------------------------------
    _save_detection_output(state, args.image, args.output_dir)

    # -- aggregate attention across layers -----------------------------------
    all_warnings: list[str] = []
    layer_attns = []  # each entry: [B, H, N_prompt, N_image]
    layer_names = []
    first_captured = None

    for cap in captures:
        captured = cap.result()
        if captured.raw_attention is None:
            all_warnings.append(f"Layer {cap.module_name}: no attention captured.")
            continue
        if first_captured is None:
            first_captured = captured

        raw = captured.raw_attention  # [B, H, T_q, T_kv]
        all_warnings.extend(captured.warnings)

        # Detect and transpose image→prompt orientation.
        dim_q, dim_kv = raw.shape[2], raw.shape[3]
        sq_q = _math.isqrt(dim_q)
        if sq_q * sq_q == dim_q and dim_q > dim_kv:
            raw = raw.transpose(2, 3)  # → [B, H, N_prompt, N_image]
            raw = raw / (raw.sum(dim=-1, keepdim=True) + 1e-8)
        layer_attns.append(raw)
        layer_names.append(cap.module_name)

    if not layer_attns:
        print("ERROR: No attention captured from any layer.", file=sys.stderr)
        for w in all_warnings:
            print(f"  WARNING: {w}", file=sys.stderr)
        raise SystemExit(1)

    num_layers = len(layer_attns)
    num_heads = layer_attns[0].shape[1]
    num_prompt_tokens = layer_attns[0].shape[2]
    num_image_tokens = layer_attns[0].shape[3]

    # Layer-max aggregate: sharpest signal at each position across layers.
    stacked = torch.stack(layer_attns, dim=0)  # [L, B, H, T_prompt, T_image]
    layer_max_attn = stacked.max(dim=0).values  # [B, H, T_prompt, T_image]

    print(f"  layers captured     : {num_layers}")
    print(f"  heads per layer     : {num_heads}")
    print(f"  per-layer shape     : {list(layer_attns[0].shape)}")

    # -- token labels --------------------------------------------------------
    token_labels = get_token_labels(
        model, processor, args.prompt, num_tokens=num_prompt_tokens
    )
    print(f"  token labels        : {token_labels}")

    # -- spatial grid --------------------------------------------------------
    spatial_grid = infer_spatial_grid(num_image_tokens)
    if spatial_grid is not None:
        print(f"  spatial grid        : {spatial_grid[0]}×{spatial_grid[1]}")

    # -- load image for visualisation ----------------------------------------
    image = PILImage.open(args.image).convert("RGB")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_tokens_to_save = num_prompt_tokens
    if args.max_tokens is not None:
        num_tokens_to_save = min(num_tokens_to_save, args.max_tokens)

    # -- save per-layer, per-head heatmaps -----------------------------------
    print(f"Saving outputs to {args.output_dir} …")
    from .visualization import (
        make_attention_overlay,
        _normalise,
    )
    import numpy as np

    saved_files: list[dict] = []

    for layer_idx, (layer_name, layer_attn) in enumerate(zip(layer_names, layer_attns)):
        layer_dir = output_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        for t_idx in range(num_tokens_to_save):
            label = token_labels[t_idx] if t_idx < len(token_labels) else f"token_{t_idx:02d}"
            safe_label = label.replace(" ", "_").replace("/", "_")

            # --- per-head heatmaps ---
            head_dir = layer_dir / f"token_{t_idx:02d}_{safe_label}_heads"
            head_dir.mkdir(parents=True, exist_ok=True)

            for h_idx in range(num_heads):
                h_1d = layer_attn[0, h_idx, t_idx, :].cpu().float().numpy()
                if spatial_grid is not None:
                    h_2d = _normalise(h_1d.reshape(spatial_grid))
                    overlay = make_attention_overlay(image, h_2d, args.alpha)
                    overlay.save(head_dir / f"head_{h_idx}.png")

            # --- head-mean heatmap ---
            mean_1d = layer_attn[0, :, t_idx, :].mean(dim=0).cpu().float().numpy()
            if spatial_grid is not None:
                mean_2d = _normalise(mean_1d.reshape(spatial_grid))
                overlay = make_attention_overlay(image, mean_2d, args.alpha)
                mean_path = layer_dir / f"token_{t_idx:02d}_{safe_label}_head_mean.png"
                overlay.save(mean_path)

            # --- head-max heatmap ---
            max_1d = layer_attn[0, :, t_idx, :].max(dim=0).values.cpu().float().numpy()
            if spatial_grid is not None:
                max_2d = _normalise(max_1d.reshape(spatial_grid))
                overlay = make_attention_overlay(image, max_2d, args.alpha)
                max_path = layer_dir / f"token_{t_idx:02d}_{safe_label}_head_max.png"
                overlay.save(max_path)

            saved_files.append({
                "token_index": t_idx,
                "label": label,
                "layer": layer_idx,
                "layer_name": layer_name,
                "head_mean_path": str(mean_path) if spatial_grid else None,
                "head_max_path": str(max_path) if spatial_grid else None,
                "per_head_dir": str(head_dir),
            })

        print(f"  layer {layer_idx} ({layer_name.split('.')[-2]}): "
              f"{num_tokens_to_save} tokens × {num_heads} heads")

    # --- layer-max aggregate heatmaps ---
    agg_dir = output_dir / "layer_max"
    agg_dir.mkdir(parents=True, exist_ok=True)
    for t_idx in range(num_tokens_to_save):
        label = token_labels[t_idx] if t_idx < len(token_labels) else f"token_{t_idx:02d}"
        safe_label = label.replace(" ", "_").replace("/", "_")

        # head-mean of layer-max
        agg_1d = layer_max_attn[0, :, t_idx, :].mean(dim=0).cpu().float().numpy()
        if spatial_grid is not None:
            agg_2d = _normalise(agg_1d.reshape(spatial_grid))
            overlay = make_attention_overlay(image, agg_2d, args.alpha)
            overlay.save(agg_dir / f"token_{t_idx:02d}_{safe_label}_head_mean.png")

        # head-max of layer-max
        agg_max_1d = layer_max_attn[0, :, t_idx, :].max(dim=0).values.cpu().float().numpy()
        if spatial_grid is not None:
            agg_max_2d = _normalise(agg_max_1d.reshape(spatial_grid))
            overlay = make_attention_overlay(image, agg_max_2d, args.alpha)
            overlay.save(agg_dir / f"token_{t_idx:02d}_{safe_label}_head_max.png")

    print(f"  layer_max aggregate : {num_tokens_to_save} tokens")

    # -- debug artefacts -----------------------------------------------------
    if args.save_debug:
        all_candidate_names = [n for n, _ in all_candidates]
        dbg_json = save_debug_json(
            output_dir=args.output_dir,
            prompt=args.prompt,
            image_path=args.image,
            selected_module_name=layer_names,
            selected_module_repr=[repr(m) for _, m in selected_modules],
            all_candidate_names=all_candidate_names,
            raw_attention_shape=list(layer_attns[0].shape),
            value_tensor_shape=[],
            normalised_attention_shape=list(layer_max_attn.shape),
            normalised_value_shape=[],
            num_prompt_tokens=num_prompt_tokens,
            num_image_tokens=num_image_tokens,
            inferred_spatial_grid=spatial_grid,
            token_labels=token_labels,
            output_file_paths=saved_files,
            warnings=all_warnings,
        )
        print(f"  debug JSON          : {dbg_json}")

        dbg_npz = save_debug_tensors(
            output_dir=args.output_dir,
            raw_attention=layer_max_attn,
            effective_attention=layer_max_attn,
            values=None,
            query=first_captured.query if first_captured else None,
            key=first_captured.key if first_captured else None,
        )
        # Also save per-layer tensors.
        per_layer_dict = {}
        for i, (lname, lattn) in enumerate(zip(layer_names, layer_attns)):
            per_layer_dict[f"layer_{i}_attn"] = lattn.cpu().float().numpy()
        np.savez_compressed(
            str(output_dir / "per_layer_tensors.npz"),
            **per_layer_dict,
        )
        print(f"  debug tensors       : {dbg_npz}")
        print(f"  per-layer tensors   : {output_dir / 'per_layer_tensors.npz'}")

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_attention_modules:
        cmd_list_modules(args)
    else:
        cmd_map(args)


if __name__ == "__main__":
    main()
