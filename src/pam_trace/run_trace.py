"""Runnable example: SAM 3 image + text inference with optional PAM tracing.

Usage::

    python -m pam_trace.run_trace \\
        --image inputs/2.jpg \\
        --prompt "yellow school bus" \\
        --pam_trace \\
        --out_dir pam_trace

When ``--pam_trace`` is omitted the script just runs inference and prints
the number of detections.  When the flag is supplied, four files are
written into *out_dir* describing every major tensor that flowed through
the model during the call.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import PIL.Image
import torch

from pam.sam3_loader import load_sam3_image_model
from pam_trace import (
    AttentionTracer,
    PamTracer,
    SelectedQueryCollector,
    write_attention_outputs,
    write_outputs,
    write_selected_query_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SAM 3 image + text inference with optional PAM tracing.",
    )
    p.add_argument("--image", type=str, required=True, help="Path to input image (jpg/png).")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt, e.g. 'yellow school bus'.")
    p.add_argument(
        "--pam_trace",
        action="store_true",
        help="If set, install non-invasive forward hooks and dump 4 trace files to --out_dir.",
    )
    p.add_argument("--out_dir", type=str, default="pam_trace", help="Output directory for trace files.")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (auto-detect if omitted).")
    p.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold.")
    p.add_argument(
        "--force_attention_weights",
        action="store_true",
        help=(
            "Analysis-only: monkey-patch attention modules so they return "
            "per-head attention weights even when the caller passed "
            "need_weights=False. The attn_output tensor is unchanged so "
            "model outputs are preserved. Required to populate "
            "selected_query_image_attention_layers.npy."
        ),
    )
    p.add_argument("--verbose", action="store_true", help="Print each hook event.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"[error] image not found: {image_path}", file=sys.stderr)
        return 2

    print(f"[pam_trace] device       : {device}")
    print(f"[pam_trace] image        : {image_path}")
    print(f"[pam_trace] prompt       : {args.prompt!r}")
    print(f"[pam_trace] pam_trace    : {args.pam_trace}")
    print(f"[pam_trace] out_dir      : {args.out_dir}")

    t0 = time.perf_counter()
    print("[pam_trace] loading SAM 3 model ...")
    model, processor = load_sam3_image_model(device=device)
    processor.confidence_threshold = args.confidence
    print(f"[pam_trace] model loaded in {time.perf_counter() - t0:.1f}s "
          f"({type(model).__name__})")

    image = PIL.Image.open(image_path).convert("RGB")

    if args.pam_trace:
        tracer = PamTracer(model, verbose=args.verbose)
        n_hooks = tracer.register()
        print(f"[pam_trace] registered hooks on {n_hooks} modules")
        attn = AttentionTracer(model, force_weights=args.force_attention_weights)
        n_attn = attn.register()
        print(
            f"[pam_trace] registered attention hooks on {n_attn} modules "
            f"(force_weights={args.force_attention_weights})"
        )
        sel = SelectedQueryCollector(
            model, force_weights=args.force_attention_weights
        )
        n_sel = sel.register()
        print(f"[pam_trace] registered selected-query hooks on {n_sel} modules")
    else:
        tracer = None
        attn = None
        sel = None

    try:
        t1 = time.perf_counter()
        state = processor.set_image(image)
        state = processor.set_text_prompt(args.prompt, state)
        infer_ms = (time.perf_counter() - t1) * 1000.0
        n_dets = int(state["boxes"].shape[0]) if "boxes" in state else 0
        print(f"[pam_trace] inference completed in {infer_ms:.1f} ms  "
              f"({n_dets} detections)")
    finally:
        if tracer is not None:
            tracer.remove()
        if attn is not None:
            attn.remove()
        if sel is not None:
            sel.remove()

    if tracer is not None:
        paths = write_outputs(tracer, args.out_dir)
        print(f"[pam_trace] captured {len(tracer.events)} hook events")
        for k, p in paths.items():
            size = p.stat().st_size if p.is_file() else 0
            print(f"           {k:14s} -> {p}  ({size:,} bytes)")

    if attn is not None:
        attn_paths = write_attention_outputs(attn, args.out_dir)
        print(f"[pam_trace] captured {len(attn.events)} attention events")
        for k, p in attn_paths.items():
            size = p.stat().st_size if p.is_file() else 0
            print(f"           {k:30s} -> {p}  ({size:,} bytes)")

    if sel is not None:
        sq_paths = write_selected_query_outputs(
            sel,
            args.out_dir,
            image_path=str(image_path),
            prompt=args.prompt,
            image_hw=(image.height, image.width),
            confidence_threshold=args.confidence,
        )
        print("[pam_trace] selected-query files:")
        for k, p in sq_paths.items():
            size = p.stat().st_size if p.is_file() else 0
            print(f"           {k:30s} -> {p}  ({size:,} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
