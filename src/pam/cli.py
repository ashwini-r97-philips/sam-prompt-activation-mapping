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
# Mode: main mapping pipeline
# ---------------------------------------------------------------------------


def cmd_map(args: argparse.Namespace) -> None:
    from PIL import Image as PILImage

    from .sam3_loader import load_sam3_image_model, run_inference
    from .attention_capture import (
        AttentionCapture,
        discover_attention_modules,
        select_attention_module,
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

    # -- discover & select attention module ----------------------------------
    all_candidates = discover_attention_modules(model, filter_text=args.module_filter)
    candidates = all_candidates

    # Apply auto-filter when no explicit filter was given.
    if args.module_filter is None and args.module_name is None:
        candidates = _auto_filter_candidates(all_candidates)

    selected_name, selected_mod = select_attention_module(
        candidates,
        module_name=args.module_name,
        layer_index=args.layer_index,
    )
    print(f"Selected module: {_module_summary(selected_name, selected_mod)}")

    # -- register hooks & run inference -------------------------------------
    capture = AttentionCapture(selected_name, selected_mod)
    capture.register()

    print(f"Running inference on {args.image} with prompt: {args.prompt!r} …")
    try:
        _output = run_inference(processor, args.image, args.prompt, args.device)
    finally:
        capture.remove()  # always clean up hooks

    captured = capture.result()

    # -- validate capture ----------------------------------------------------
    all_warnings: list[str] = list(captured.warnings)

    if captured.raw_attention is None:
        print("ERROR: Could not capture attention weights.", file=sys.stderr)
        for w in all_warnings:
            print(f"  WARNING: {w}", file=sys.stderr)
        print(
            "\nRun with --list-attention-modules and try a different "
            "--module-name or --module-filter.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if captured.values is None:
        print(
            "WARNING: Value vectors not captured. "
            "Effective attention will be skipped.",
            file=sys.stderr,
        )

    raw_attn = captured.raw_attention  # [B, H, T_q, T_kv]
    values = captured.values  # [B, H, T_kv, D_head] or None

    print(f"  raw attention shape : {list(raw_attn.shape)}")
    if values is not None:
        print(f"  value tensor shape  : {list(values.shape)}")

    # -- detect attention orientation ----------------------------------------
    # In SAM 3's DETR encoder the image tokens are the *query* and the
    # prompt tokens are the *key/value*, giving shape
    #   [B, H, N_image, N_prompt].
    # We need the transpose: [B, H, N_prompt, N_image] so that each row
    # is a prompt token's attention distribution over image positions.
    #
    # Heuristic: if dim-2 >> dim-3 AND dim-2 is a perfect square (spatial
    # grid), the attention is image→prompt and we should transpose.
    dim_q, dim_kv = raw_attn.shape[2], raw_attn.shape[3]
    import math as _math
    sq_q = _math.isqrt(dim_q)
    sq_kv = _math.isqrt(dim_kv)
    transposed = False

    if sq_q * sq_q == dim_q and dim_q > dim_kv:
        # dim_q looks like a spatial grid (e.g. 72²=5184) and dim_kv is
        # much smaller (prompt tokens).  Transpose.
        print(f"  NOTE: detected image→prompt attention ({dim_q} queries, "
              f"{dim_kv} keys). Transposing to prompt→image.")
        raw_attn = raw_attn.transpose(2, 3)  # [B, H, N_prompt, N_image]
        # The transposed rows are NOT valid probability distributions (they
        # are columns of the original softmax).  Re-normalise so each
        # prompt token's attention over image positions sums to 1.
        raw_attn = raw_attn / (raw_attn.sum(dim=-1, keepdim=True) + 1e-8)
        transposed = True
        all_warnings.append(
            "Attention was transposed from image→prompt to prompt→image "
            "and re-normalised.  Values for effective-attention are from "
            "the original (untransposed) direction."
        )

    num_prompt_tokens = raw_attn.shape[2]
    num_image_tokens = raw_attn.shape[3]

    # -- effective attention -------------------------------------------------
    if values is not None and not transposed:
        eff_attn = compute_effective_attention(raw_attn, values)
    elif values is not None and transposed:
        # With transposed attention the value vectors were for the prompt
        # (original KV side), not the image.  Effective attention in the
        # traditional sense doesn't apply.  Fall back to raw.
        eff_attn = raw_attn
        all_warnings.append(
            "Effective attention skipped: value vectors correspond to the "
            "prompt (original KV), not image positions."
        )
    else:
        eff_attn = raw_attn  # fallback: just use raw
        all_warnings.append("Using raw attention as effective (no values captured).")

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

    # -- save heatmaps -------------------------------------------------------
    print(f"Saving outputs to {args.output_dir} …")
    saved_files = save_token_heatmaps(
        image=image,
        raw_attention=raw_attn,
        effective_attention=eff_attn,
        token_labels=token_labels,
        output_dir=args.output_dir,
        head_reduction=args.head_reduction,
        alpha=args.alpha,
        no_overlay=args.no_overlay,
        max_tokens=args.max_tokens,
        resize_long_side=args.resize_long_side,
        spatial_grid=spatial_grid,
    )

    for entry in saved_files:
        print(f"  [{entry['label']}] raw={entry.get('raw_path', '?')}  "
              f"eff={entry.get('effective_path', '?')}")

    # -- debug artefacts -----------------------------------------------------
    if args.save_debug:
        all_candidate_names = [n for n, _ in all_candidates]
        dbg_json = save_debug_json(
            output_dir=args.output_dir,
            prompt=args.prompt,
            image_path=args.image,
            selected_module_name=selected_name,
            selected_module_repr=repr(selected_mod),
            all_candidate_names=all_candidate_names,
            raw_attention_shape=list(raw_attn.shape),
            value_tensor_shape=list(values.shape) if values is not None else [],
            normalised_attention_shape=list(raw_attn.shape),
            normalised_value_shape=list(values.shape) if values is not None else [],
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
            raw_attention=raw_attn,
            effective_attention=eff_attn,
            values=values,
            query=captured.query,
            key=captured.key,
        )
        print(f"  debug tensors       : {dbg_npz}")

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
