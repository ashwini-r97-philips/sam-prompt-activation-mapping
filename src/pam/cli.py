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
    p.add_argument(
        "--target-mask-index",
        type=int,
        default=None,
        help="Index of the detected mask to target (0-based).",
    )
    p.add_argument(
        "--skip-per-head",
        action="store_true",
        help="Skip saving per-head heatmap PNGs (much faster).",
    )
    p.add_argument(
        "--skip-per-layer",
        action="store_true",
        help="Skip all per-layer/per-token heatmap PNGs and layer-max aggregates.",
    )

    # -- attribution-flow mode -----------------------------------------------
    p.add_argument(
        "--aggregation-mode",
        choices=["attention", "attribution-flow"],
        default="attention",
        help="Method: 'attention' (raw/effective heatmaps) or "
        "'attribution-flow' (gradient-based attribution).",
    )
    p.add_argument(
        "--query-index",
        type=int,
        default=None,
        help="Override decoder query index (0-based object query).",
    )
    p.add_argument(
        "--target-objective",
        type=str,
        default="score",
        choices=[
            "score", "object_query_score", "raw_query_score",
            "mask_logit_mean", "mask_foreground_logit_mean",
            "mask_probability_foreground_mean", "mask_contrastive_logit",
            "combined_query_and_mask",
            "semantic_logit_mean", "semantic_foreground_logit_mean",
            "semantic_contrastive_logit",
        ],
        help="Target scalar for attribution-flow backward pass.",
    )
    p.add_argument(
        "--include-all-prompt-tokens",
        action="store_true",
        help="Include all prompt tokens (text + geometry) in maps.",
    )
    p.add_argument("--text-token-start", type=int, default=None)
    p.add_argument("--text-token-end", type=int, default=None)
    p.add_argument(
        "--presence-query-index",
        type=int,
        default=0,
        help="Index of presence token in decoder attention (default: 0, prepended).",
    )
    p.add_argument(
        "--top-k-contributors",
        type=int,
        default=24,
        help="Number of top contributors to save in grid/CSV.",
    )
    p.add_argument(
        "--positive-only",
        action="store_true",
        default=True,
        help="Show only positive attribution in overlays (default).",
    )
    p.add_argument("--save-negative", action="store_true")
    p.add_argument("--save-signed", action="store_true")
    p.add_argument("--validate-mha-reconstruction", action="store_true")
    p.add_argument(
        "--attribution-capture-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for attribution tensors (cpu saves GPU memory).",
    )
    p.add_argument("--allow-nongrad-target", action="store_true")
    p.add_argument(
        "--background-penalty",
        type=float,
        default=0.25,
        help="Background penalty weight for mask_contrastive_logit.",
    )
    p.add_argument(
        "--lambda-mask",
        type=float,
        default=1.0,
        help="Mask weight for combined_query_and_mask.",
    )
    p.add_argument(
        "--compare-target-objectives",
        nargs="+",
        default=None,
        metavar="OBJ",
        help="Run attribution for multiple objectives and save comparison panel.",
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

    boxes_np = boxes.detach().cpu().float().numpy()
    scores_np = scores.detach().cpu().float().numpy()

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
        masks_np = masks.detach().cpu().numpy()  # [N, 1, H, W] bool
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

    _skip_layer_saves = getattr(args, "skip_per_layer", False)

    if not _skip_layer_saves:
        for layer_idx, (layer_name, layer_attn) in enumerate(zip(layer_names, layer_attns)):
            layer_dir = output_dir / f"layer_{layer_idx}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            for t_idx in range(num_tokens_to_save):
                label = token_labels[t_idx] if t_idx < len(token_labels) else f"token_{t_idx:02d}"
                safe_label = label.replace(" ", "_").replace("/", "_")

                # --- per-head heatmaps ---
                if not getattr(args, "skip_per_head", False):
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
                    "per_head_dir": str(head_dir) if not getattr(args, "skip_per_head", False) else None,
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
    else:
        print("  (skipping per-layer / layer-max heatmap saves)")

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
# Mode: attribution-flow
# ---------------------------------------------------------------------------


def cmd_attribution_flow(args: argparse.Namespace) -> None:
    """Run Gradient-Weighted Attribution Flow PAM."""
    import json

    import numpy as np
    from PIL import Image as PILImage

    from .sam3_loader import load_sam3_image_model, run_inference_with_grad
    from .attribution_capture import AttributionCapture
    from .target_resolver import (
        resolve_target_scalar,
        inventory_output_tensors,
        save_tensor_inventory,
    )
    from .query_resolver import (
        resolve_query_index,
        save_query_resolution,
        build_query_index_mapping,
        save_query_index_mapping,
    )
    from .attribution_flow import (
        compute_edge_attribution,
        compute_group_a_encoder_maps,
        compute_group_a_geometry_maps,
        compute_group_b_decoder_maps,
        compute_joint_map,
        compute_joint_sum_map,
        save_group_a_token_contributions,
        save_group_a_layer_head_token_contributions,
        save_group_b_query_layer_head_contributions,
        save_group_b_per_query_contributions,
        save_top_contributors,
        save_query_validation_table,
    )
    from .attribution_visualization import (
        save_attribution_overlay,
        save_signed_overlays,
        save_heatmap_png,
        make_top_group_a_grid,
        make_top_group_b_grid,
        save_summary_panel,
        save_decoder_token_comparison,
        save_full_mask_summary_panel,
    )
    from .tokenization import get_token_labels
    from .visualization import infer_spatial_grid

    # -- validate ------------------------------------------------------------
    if not args.image:
        print("ERROR: --image is required.", file=sys.stderr)
        raise SystemExit(1)
    if not args.prompt:
        print("ERROR: --prompt is required.", file=sys.stderr)
        raise SystemExit(1)
    if args.target_mask_index is None:
        print("ERROR: --target-mask-index is required for attribution-flow.",
              file=sys.stderr)
        raise SystemExit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cap_device = args.attribution_capture_device

    # -- load model ----------------------------------------------------------
    print("--- Attribution Flow PAM ---")
    print(f"Loading SAM 3 image model on {args.device} …")
    model, processor = load_sam3_image_model(device=args.device)

    # -- register attribution hooks ------------------------------------------
    capture = AttributionCapture(
        reconstruction_device=cap_device,
        validate_reconstruction=args.validate_mha_reconstruction,
        reconstruction_error_threshold=1e-2,
    )
    n_hooked = capture.register(model)
    print(f"Hooked {n_hooked} cross-attention module(s)")

    # -- run inference with grad ---------------------------------------------
    print(f"Running inference (with grad) on {args.image} …")
    outputs = run_inference_with_grad(
        model=model,
        processor=processor,
        image_path=args.image,
        prompt=args.prompt,
        device=args.device,
    )

    n_detections = len(outputs["_kept_indices"])
    print(f"  detections: {n_detections} object(s) found")

    if n_detections == 0:
        print("ERROR: No detections found.", file=sys.stderr)
        capture.remove()
        raise SystemExit(1)

    if args.target_mask_index >= n_detections:
        print(f"ERROR: --target-mask-index={args.target_mask_index} but only "
              f"{n_detections} detection(s).", file=sys.stderr)
        capture.remove()
        raise SystemExit(1)

    # -- retain grads --------------------------------------------------------
    capture.retain_grads()

    # -- resolve target scalar -----------------------------------------------
    target_res = resolve_target_scalar(
        outputs,
        target_mask_index=args.target_mask_index,
        target_objective=args.target_objective,
        query_index=args.query_index,
        background_penalty=args.background_penalty,
        lambda_mask=args.lambda_mask,
    )
    target_scalar = target_res.target_scalar

    print(f"target mask index          : {args.target_mask_index}")
    print(f"target objective           : {args.target_objective}")
    print(f"target scalar requires grad: {target_res.requires_grad}")
    print(f"target description         : {target_res.target_description}")

    if not target_res.requires_grad and not args.allow_nongrad_target:
        # Save tensor inventory for debugging.
        inv_path = save_tensor_inventory(outputs, output_dir / "output_tensor_inventory.json")
        print(f"  tensor inventory saved: {inv_path}", file=sys.stderr)
        print(
            "ERROR: target_scalar.requires_grad is False. "
            "The forward path may be under no_grad or inference_mode. "
            "Use --allow-nongrad-target to override.",
            file=sys.stderr,
        )
        capture.remove()
        raise SystemExit(1)

    # -- backward ------------------------------------------------------------
    model.zero_grad(set_to_none=True)
    target_scalar.backward(retain_graph=False)

    # -- build records & compute attributions --------------------------------
    records = capture.build_records()
    grouped = capture.records_by_group(records)
    print(capture.summarize(records))

    # Compute edge attribution for each module.
    group_attributions: dict[str, list] = {}
    for group_name, recs in grouped.items():
        attrs = []
        for rec in recs:
            if rec.reconstruction is None:
                print(f"  SKIP {rec.module_name}: no reconstruction")
                continue
            try:
                attr = compute_edge_attribution(rec, device=cap_device)
                attrs.append(attr)
            except RuntimeError as e:
                print(f"  SKIP {rec.module_name}: {e}")
        group_attributions[group_name] = attrs

    capture.remove()

    # -- check attribution mass ----------------------------------------------
    enc_attrs = group_attributions.get("group_a_encoder", [])
    geo_attrs = group_attributions.get("group_a_geometry", [])
    dec_attrs = group_attributions.get("group_b_decoder", [])

    dec_mass = sum(
        a.positive_edge_attr.sum().item() for a in dec_attrs
    )
    print(f"Group B positive attribution mass: {dec_mass:.4e}")
    if abs(dec_mass) < 1e-12:
        if args.target_objective.startswith("semantic_"):
            print("NOTE: Group B attribution mass is near-zero. "
                  "This is expected for semantic_seg targets because "
                  "semantic_seg depends on encoder (Group A) not decoder (Group B).",
                  file=sys.stderr)
        else:
            print("WARNING: Group B attribution mass is zero. "
                  "Gradients may not flow through decoder cross-attention.",
                  file=sys.stderr)

    # Max reconstruction error.
    max_recon_err = max(
        (r.reconstruction.max_abs_reconstruction_error
         for r in records if r.reconstruction),
        default=0.0,
    )
    print(f"max reconstruction error   : {max_recon_err:.2e}")

    # Max conservation error.
    all_attrs = [a for attrs in group_attributions.values() for a in attrs]
    max_cons_err = max(
        (a.local_conservation_error for a in all_attrs),
        default=0.0,
    )
    print(f"max conservation error     : {max_cons_err:.2e}")

    # -- resolve query index -------------------------------------------------
    query_res = resolve_query_index(
        target_mask_index=args.target_mask_index,
        query_index_override=args.query_index,
        outputs=outputs,
        presence_query_index=args.presence_query_index,
    )
    qmap = build_query_index_mapping(query_res, args.presence_query_index)
    print(f"object_query_index         : {qmap.object_query_index}")
    print(f"decoder_token_index        : {qmap.decoder_token_index}")
    print(f"query resolution method    : {qmap.method}")
    if qmap.object_query_index == args.presence_query_index:
        print("WARNING: resolved query is the presence token!", file=sys.stderr)

    # -- infer spatial grid --------------------------------------------------
    # Group A encoder: edge_attr [B, H, 5184_image_queries, 33_prompt_sources]
    #   -> image tokens are dim 2 (queries)
    # Group B decoder: edge_attr [B, H, 201_queries, 5184_image_sources]
    #   -> image tokens are dim 3 (sources)
    N_image = None
    for a in enc_attrs:
        if a.group == "group_a_encoder":
            N_image = a.edge_attr.shape[2]  # image tokens are queries
            break
    if N_image is None:
        for a in dec_attrs:
            N_image = a.edge_attr.shape[3]  # image tokens are sources
            break
    if N_image is None:
        print("ERROR: Could not determine image token count.", file=sys.stderr)
        raise SystemExit(1)

    spatial_grid = infer_spatial_grid(N_image)
    if spatial_grid is None:
        print(f"ERROR: {N_image} is not a perfect square.", file=sys.stderr)
        raise SystemExit(1)
    Gh, Gw = spatial_grid
    print(f"spatial grid               : {Gh}×{Gw}")

    # -- token labels --------------------------------------------------------
    token_labels = get_token_labels(
        model, processor, args.prompt, num_tokens=args.max_tokens,
    )
    # Determine prompt token count from encoder attribution.
    T_prompt = enc_attrs[0].edge_attr.shape[3] if enc_attrs else 33
    num_prompt_tokens = T_prompt
    if args.max_tokens:
        num_prompt_tokens = min(args.max_tokens, T_prompt)
    token_labels = token_labels[:num_prompt_tokens]
    # Pad if needed.
    while len(token_labels) < num_prompt_tokens:
        token_labels.append(f"prompt_{len(token_labels):02d}")

    # -- compute group-specific maps -----------------------------------------
    image = PILImage.open(args.image).convert("RGB")

    # Group A encoder.
    group_a_maps = None
    if enc_attrs:
        group_a_maps = compute_group_a_encoder_maps(
            enc_attrs, spatial_grid, num_prompt_tokens=num_prompt_tokens,
        )

    # Group A geometry.
    geo_maps = None
    if geo_attrs:
        geo_maps = compute_group_a_geometry_maps(geo_attrs, spatial_grid)

    # Group B decoder.
    group_b_maps = None
    if dec_attrs:
        group_b_maps = compute_group_b_decoder_maps(
            dec_attrs, qmap.object_query_index, spatial_grid,
            presence_query_index=args.presence_query_index,
        )

    # Joint maps.
    joint_map = None
    joint_sum_map = None
    if group_a_maps is not None and group_b_maps is not None:
        joint_map = compute_joint_map(
            group_a_maps["prompt_write_positive"],
            group_b_maps["query_read_positive"],
        )
        joint_sum_map = compute_joint_sum_map(
            group_a_maps["prompt_write_positive"],
            group_b_maps["query_read_positive"],
        )

    # -- save outputs --------------------------------------------------------
    print(f"\nSaving outputs to {output_dir} …")
    saved_paths: dict[str, str] = {}

    # Group A overlays.
    if group_a_maps is not None:
        pa = group_a_maps["prompt_write_positive"].detach().numpy()
        na = group_a_maps["prompt_write_negative"].detach().numpy()
        p = save_attribution_overlay(
            image, pa,
            output_dir / "group_a_prompt_write_positive_overlay.png",
            alpha=args.alpha,
        )
        saved_paths["group_a_pos"] = str(p)
        print(f"  group_a positive overlay : {p}")

        p = save_attribution_overlay(
            image, na,
            output_dir / "group_a_prompt_write_negative_overlay.png",
            alpha=args.alpha, colormap="coolwarm",
        )
        saved_paths["group_a_neg"] = str(p)
        print(f"  group_a negative overlay : {p}")

    # Group B overlays.
    if group_b_maps is not None:
        pb = group_b_maps["query_read_positive"].detach().numpy()
        nb = group_b_maps["query_read_negative"].detach().numpy()
        p = save_attribution_overlay(
            image, pb,
            output_dir / "group_b_query_read_positive_overlay.png",
            alpha=args.alpha,
        )
        saved_paths["group_b_pos"] = str(p)
        print(f"  group_b positive overlay : {p}")

        p = save_attribution_overlay(
            image, nb,
            output_dir / "group_b_query_read_negative_overlay.png",
            alpha=args.alpha, colormap="coolwarm",
        )
        saved_paths["group_b_neg"] = str(p)
        print(f"  group_b negative overlay : {p}")

    # Joint overlay.
    if joint_map is not None:
        jnp = joint_map.detach().numpy()
        p = save_attribution_overlay(
            image, jnp,
            output_dir / "joint_product_positive_overlay.png",
            alpha=args.alpha,
        )
        saved_paths["joint_product"] = str(p)
        print(f"  joint product overlay    : {p}")

        save_heatmap_png(
            jnp, output_dir / "joint_product_positive_heatmap.png",
        )

    # Joint sum overlay.
    if joint_sum_map is not None:
        jsnp = joint_sum_map.detach().numpy()
        p = save_attribution_overlay(
            image, jsnp,
            output_dir / "joint_sum_positive_overlay.png",
            alpha=args.alpha,
        )
        saved_paths["joint_sum"] = str(p)
        print(f"  joint sum overlay        : {p}")

        save_heatmap_png(
            jsnp, output_dir / "joint_sum_positive_heatmap.png",
        )

    # Top-K grids.
    if group_a_maps is not None:
        p = make_top_group_a_grid(
            image,
            group_a_maps["per_layer_token_maps_pos"].detach().numpy(),
            token_labels,
            output_dir / "top_group_a_prompt_token_grid.png",
            top_k=args.top_k_contributors,
            alpha=args.alpha,
        )
        saved_paths["top_a_grid"] = str(p)
        print(f"  top group_a grid         : {p}")

    if group_b_maps is not None:
        p = make_top_group_b_grid(
            image,
            group_b_maps["per_layer_head_maps_pos"].detach().numpy(),
            qmap.object_query_index,
            qmap.decoder_token_index,
            output_dir / "top_group_b_query_head_grid.png",
            top_k=args.top_k_contributors,
            alpha=args.alpha,
        )
        saved_paths["top_b_grid"] = str(p)
        print(f"  top group_b grid         : {p}")

    # Diagnostic: compare adjacent decoder tokens.
    if dec_attrs and qmap.decoder_token_index >= 2:
        dti = qmap.decoder_token_index
        p = save_decoder_token_comparison(
            image,
            dec_attrs,
            decoder_token_a=dti - 1,
            decoder_token_b=dti,
            spatial_grid=spatial_grid,
            path=output_dir / "decoder_token_comparison.png",
            alpha=args.alpha,
        )
        saved_paths["decoder_token_comparison"] = str(p)
        print(f"  decoder token comparison : {p}")

    # Summary panel.
    mask_preview = None
    if outputs.get("masks") is not None and len(outputs["masks"]) > args.target_mask_index:
        mask_preview = outputs["masks"][args.target_mask_index, 0].cpu().detach().numpy().astype(np.float32)

    p = save_full_mask_summary_panel(
        image,
        mask_preview=mask_preview,
        group_a_overlay=group_a_maps["prompt_write_positive"].detach().numpy() if group_a_maps else None,
        group_b_overlay=group_b_maps["query_read_positive"].detach().numpy() if group_b_maps else None,
        joint_product_overlay=joint_map.detach().numpy() if joint_map is not None else None,
        joint_sum_overlay=joint_sum_map.detach().numpy() if joint_sum_map is not None else None,
        path=output_dir / "full_mask_attribution_summary.png",
        alpha=args.alpha,
    )
    saved_paths["summary"] = str(p)
    print(f"  summary panel            : {p}")

    # -- CSVs ----------------------------------------------------------------
    if group_a_maps is not None:
        p = save_group_a_token_contributions(
            group_a_maps, token_labels,
            output_dir / "group_a_prompt_token_contributions.csv",
        )
        print(f"  group_a token CSV        : {p}")

        p = save_group_a_layer_head_token_contributions(
            group_a_maps, token_labels,
            output_dir / "group_a_layer_head_token_contributions.csv",
        )
        print(f"  group_a L/H/T CSV        : {p}")

    if group_b_maps is not None:
        p = save_group_b_query_layer_head_contributions(
            group_b_maps,
            qmap.object_query_index,
            qmap.decoder_token_index,
            output_dir / "group_b_query_layer_head_contributions.csv",
        )
        print(f"  group_b query L/H CSV    : {p}")

        p = save_group_b_per_query_contributions(
            group_b_maps["contribution_per_query"],
            qmap.object_query_index,
            qmap.decoder_token_index,
            qmap.presence_token_index,
            output_dir / "group_b_contribution_per_query.csv",
        )
        print(f"  group_b per-query CSV    : {p}")

        # Query validation table.
        p = save_query_validation_table(
            group_b_maps["contribution_per_query"],
            outputs["pred_logits"],
            qmap.presence_token_index,
            qmap.object_query_index,
            output_dir / "query_index_validation.csv",
        )
        print(f"  query validation CSV     : {p}")

    p = save_top_contributors(
        group_a_maps, group_b_maps, token_labels, qmap.object_query_index,
        output_dir / "attribution_flow_top_contributors.csv",
        top_k=args.top_k_contributors,
    )
    print(f"  top contributors CSV     : {p}")

    # -- raw numpy arrays ----------------------------------------------------
    if group_a_maps is not None:
        np.save(output_dir / "group_a_prompt_image_positive.npy",
                group_a_maps["per_layer_token_maps_pos"].detach().numpy())
        np.save(output_dir / "group_a_prompt_image_negative.npy",
                group_a_maps["per_layer_token_maps_neg"].detach().numpy())

    if group_b_maps is not None:
        np.save(output_dir / "group_b_query_image_positive.npy",
                group_b_maps["per_layer_head_maps_pos"].detach().numpy())
        np.save(output_dir / "group_b_query_image_negative.npy",
                group_b_maps["per_layer_head_maps_neg"].detach().numpy())

    # -- debug artifacts -----------------------------------------------------
    if args.save_debug:
        save_tensor_inventory(
            outputs, output_dir / "output_tensor_inventory.json",
        )
        save_query_resolution(query_res, output_dir / "query_resolution.json")
        save_query_index_mapping(qmap, output_dir / "query_index_mapping.json")

        # Foreground mask info (for mask objectives).
        fg_info = target_res._foreground_info
        fg_dict = {}
        if fg_info is not None:
            fg_dict = {
                "foreground_pixels": fg_info.foreground_pixels,
                "background_pixels": fg_info.background_pixels,
                "foreground_area_fraction": fg_info.foreground_area_fraction,
                "background_area_fraction": fg_info.background_area_fraction,
                "foreground_mask_shape": list(fg_info.target_hw),
            }

        pred_masks_shape = None
        if outputs.get("pred_masks") is not None:
            pred_masks_shape = list(outputs["pred_masks"].shape)

        semantic_seg_shape = None
        raw_out = outputs.get("_raw_outputs", {})
        sem_tensor = raw_out.get("semantic_seg") if isinstance(raw_out, dict) else None
        if sem_tensor is not None:
            semantic_seg_shape = list(sem_tensor.shape)

        is_semantic_obj = args.target_objective.startswith("semantic_")
        target_source = "semantic_seg" if is_semantic_obj else (
            "pred_masks" if "mask" in args.target_objective else "pred_logits"
        )

        debug_data = {
            "image": str(args.image),
            "prompt": args.prompt,
            "target_mask_index": args.target_mask_index,
            "target_objective": args.target_objective,
            "target_source": target_source,
            "target_description": target_res.target_description,
            "target_requires_grad": target_res.requires_grad,
            "target_scalar_value": target_scalar.item(),
            "target_grad_fn": target_res.grad_fn,
            "object_query_index": qmap.object_query_index,
            "decoder_token_index": qmap.decoder_token_index,
            "presence_token_index": qmap.presence_token_index,
            "pred_masks_shape": pred_masks_shape,
            "semantic_seg_shape": semantic_seg_shape,
            "background_penalty": args.background_penalty,
            "lambda_mask": args.lambda_mask,
            **fg_dict,
            "note": (
                "Semantic segmentation attribution explains _raw_outputs.semantic_seg, "
                "not _raw_outputs.pred_masks."
                if is_semantic_obj else
                "Selected mask pixels used only to select output mask logits "
                "for the target; not used to rank or construct attention heatmaps."
            ),
            "query_resolution": {
                "object_query_index": qmap.object_query_index,
                "decoder_token_index": qmap.decoder_token_index,
                "method": qmap.method,
                "confidence": qmap.confidence,
            },
            "spatial_grid": [Gh, Gw],
            "num_prompt_tokens": num_prompt_tokens,
            "token_labels": token_labels,
            "module_groups": {
                g: len(recs) for g, recs in grouped.items()
            },
            "reconstruction_errors": {
                r.module_name: r.reconstruction.max_abs_reconstruction_error
                for r in records if r.reconstruction
            },
            "conservation_errors": {
                a.module_name: a.local_conservation_error
                for a in all_attrs
            },
            "max_reconstruction_error": max_recon_err,
            "max_conservation_error": max_cons_err,
            "group_b_positive_mass": dec_mass,
            "group_b_near_zero_expected": is_semantic_obj,
            "n_detections": n_detections,
            "selected_object_query_index": qmap.object_query_index,
            "selected_decoder_token_index": qmap.decoder_token_index,
            "output_paths": saved_paths,
            "warnings": [w for a in all_attrs for w in a.warnings]
                + [w for r in records for w in r.warnings]
                + target_res.warnings
                + query_res.warnings,
        }
        dbg_path = output_dir / "debug_attribution_flow.json"
        with open(dbg_path, "w") as f:
            json.dump(debug_data, f, indent=2)
        print(f"  debug JSON               : {dbg_path}")

        # Target objective details.
        _formulas = {
            "score": "sigmoid(pred_logits[0,q,0]) * sigmoid(presence)",
            "object_query_score": "sigmoid(pred_logits[0,q,0])",
            "raw_query_score": "pred_logits[0,q,0]",
            "mask_logit_mean": "mean(pred_masks[0,q])",
            "mask_foreground_logit_mean": "mean(pred_masks[0,q][foreground])",
            "mask_probability_foreground_mean": "mean(sigmoid(pred_masks[0,q][foreground]))",
            "mask_contrastive_logit": "mean(fg) - alpha*mean(bg)",
            "combined_query_and_mask": "sigmoid(logit) + lambda*mean(fg_logits)",
            "semantic_logit_mean": "mean(semantic_seg[0,0])",
            "semantic_foreground_logit_mean": "mean(semantic_seg[0,0][foreground])",
            "semantic_contrastive_logit": "mean(sem_fg) - alpha*mean(sem_bg)",
        }

        is_semantic = args.target_objective.startswith("semantic_")
        target_source = "semantic_seg" if is_semantic else (
            "pred_masks" if "mask" in args.target_objective else "pred_logits"
        )

        obj_details = {
            "objective_name": args.target_objective,
            "formula": _formulas.get(args.target_objective, ""),
            "target_source": target_source,
            "selected_mask_index": args.target_mask_index,
            "object_query_index": qmap.object_query_index,
            "decoder_token_index": qmap.decoder_token_index,
            "target_value": target_scalar.item(),
        }
        if is_semantic:
            raw_out = outputs.get("_raw_outputs", {})
            sem_t = raw_out.get("semantic_seg")
            if sem_t is not None:
                obj_details["semantic_seg_shape"] = list(sem_t.shape)
                sem_logits = sem_t[0, 0]
                if fg_info is not None:
                    obj_details.update({
                        "foreground_pixels": fg_info.foreground_pixels,
                        "background_pixels": fg_info.background_pixels,
                        "foreground_area_fraction": fg_info.foreground_area_fraction,
                        "raw_semantic_logit_mean_foreground": (
                            sem_logits[fg_info.foreground].mean().item()
                        ),
                        "raw_semantic_logit_mean_background": (
                            sem_logits[fg_info.background].mean().item()
                            if fg_info.background_pixels > 0 else None
                        ),
                    })
        elif fg_info is not None:
            pred_masks_t = outputs.get("pred_masks")
            ml = pred_masks_t[0, qmap.object_query_index] if pred_masks_t is not None else None
            obj_details.update({
                "foreground_pixels": fg_info.foreground_pixels,
                "background_pixels": fg_info.background_pixels,
                "foreground_area_fraction": fg_info.foreground_area_fraction,
                "raw_mask_logit_mean_foreground": (
                    ml[fg_info.foreground].mean().item() if ml is not None else None
                ),
                "raw_mask_logit_mean_background": (
                    ml[fg_info.background].mean().item() if ml is not None else None
                ),
            })
        obj_path = output_dir / "target_objective_details.json"
        with open(obj_path, "w") as f:
            json.dump(obj_details, f, indent=2)
        print(f"  objective details JSON   : {obj_path}")

        # Reconstruction report CSV.
        import csv as _csv
        recon_csv = output_dir / "mha_reconstruction_report.csv"
        with open(recon_csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["module_name", "group", "layer", "max_error", "mean_error", "valid"])
            for r in records:
                if r.reconstruction:
                    w.writerow([
                        r.module_name, r.group, r.layer_index,
                        f"{r.reconstruction.max_abs_reconstruction_error:.4e}",
                        f"{r.reconstruction.mean_abs_reconstruction_error:.4e}",
                        r.valid_reconstruction,
                    ])
        print(f"  reconstruction report    : {recon_csv}")

        # Conservation report CSV.
        cons_csv = output_dir / "local_conservation_report.csv"
        with open(cons_csv, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["module_name", "group", "layer", "max_conservation_error"])
            for a in all_attrs:
                w.writerow([
                    a.module_name, a.group, a.layer_index,
                    f"{a.local_conservation_error:.4e}",
                ])
        print(f"  conservation report      : {cons_csv}")

    # -- save detection output -----------------------------------------------
    _save_detection_output(outputs, args.image, args.output_dir)

    print("\nsaved outputs              :", output_dir)
    print("Done.")

    # Return maps for comparison mode.
    return {
        "group_a_maps": group_a_maps,
        "group_b_maps": group_b_maps,
        "joint_map": joint_map,
        "joint_sum_map": joint_sum_map,
    }


# ---------------------------------------------------------------------------
# Mode: compare target objectives
# ---------------------------------------------------------------------------


def cmd_compare_objectives(args: argparse.Namespace) -> None:
    """Run attribution for multiple objectives and produce a comparison panel."""
    import copy

    import numpy as np
    from PIL import Image as PILImage

    from .attribution_visualization import save_objective_comparison

    objectives = args.compare_target_objectives
    image = PILImage.open(args.image).convert("RGB")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_results: dict[str, dict[str, np.ndarray | None]] = {}

    for obj_name in objectives:
        print(f"\n{'=' * 60}")
        print(f"  Objective: {obj_name}")
        print(f"{'=' * 60}")
        obj_args = copy.copy(args)
        obj_args.target_objective = obj_name
        obj_args.output_dir = str(output_dir / obj_name)
        obj_args.compare_target_objectives = None  # prevent recursion

        result = cmd_attribution_flow(obj_args)

        group_b = None
        joint_product = None
        joint_sum = None
        if result["group_b_maps"] is not None:
            group_b = result["group_b_maps"]["query_read_positive"].detach().numpy()
        if result["joint_map"] is not None:
            joint_product = result["joint_map"].detach().numpy()
        if result["joint_sum_map"] is not None:
            joint_sum = result["joint_sum_map"].detach().numpy()

        comparison_results[obj_name] = {
            "group_b": group_b,
            "joint_product": joint_product,
            "joint_sum": joint_sum,
        }

    p = save_objective_comparison(
        image,
        comparison_results,
        output_dir / "objective_comparison_summary.png",
        alpha=args.alpha,
    )
    print(f"\nComparison panel saved      : {p}")
    print("Comparison done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_attention_modules:
        cmd_list_modules(args)
    elif getattr(args, "compare_target_objectives", None):
        cmd_compare_objectives(args)
    elif getattr(args, "aggregation_mode", "attention") == "attribution-flow":
        cmd_attribution_flow(args)
    else:
        cmd_map(args)


if __name__ == "__main__":
    main()
