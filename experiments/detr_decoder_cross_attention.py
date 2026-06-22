from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


TARGET_DECODER_CROSS_ATTN = [
    f"transformer.decoder.layers.{layer}.cross_attn"
    for layer in range(6)
]


@dataclass
class CapturedDecoderAttention:
    module_name: str
    call_index: int
    query: torch.Tensor
    key: torch.Tensor
    attn_mask: torch.Tensor | None
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]


def run_model_inference_with_grad(processor, image_path, prompt, device):
    from pam.sam3_loader import run_inference_with_grad

    return run_inference_with_grad(processor, image_path, prompt, device)


def patch_run_decoder_capture_hs(model, storage):
    """
    Captures DETR decoder output hs.

    hs shape:
        [num_decoder_layers, batch, num_queries, channels]
        e.g. [6, 1, 200, 256]
    """
    if not hasattr(model, "_run_decoder"):
        raise RuntimeError("model does not have _run_decoder")

    original_run_decoder = model._run_decoder

    def wrapped_run_decoder(*args, **kwargs):
        out, hs = original_run_decoder(*args, **kwargs)

        if not isinstance(hs, torch.Tensor):
            raise RuntimeError(f"Expected hs tensor, got {type(hs)}")

        if hs.ndim != 4:
            raise RuntimeError(f"Expected hs [L,B,Q,C], got {tuple(hs.shape)}")

        if not hs.requires_grad:
            raise RuntimeError("Captured hs does not require grad")

        hs.retain_grad()

        storage["hs"] = hs
        storage["hs_shape"] = tuple(hs.shape)

        return out, hs

    model._run_decoder = wrapped_run_decoder

    def restore():
        model._run_decoder = original_run_decoder

    return restore


@contextmanager
def capture_decoder_cross_attention_inputs(model):
    """
    Captures Q/K/attn_mask inside DETR decoder cross-attention modules.

    Target modules:
        transformer.decoder.layers.0.cross_attn
        ...
        transformer.decoder.layers.5.cross_attn

    We do NOT edit SAM3 source. We monkey-patch
    torch.nn.functional.scaled_dot_product_attention during this context.

    Captured Q/K shape should be:
        Q = [B, H, Q_tokens, D]
        K = [B, H, memory_tokens, D]
    """
    modules = dict(model.named_modules())

    missing = [name for name in TARGET_DECODER_CROSS_ATTN if name not in modules]
    if missing:
        print("Missing decoder cross-attention modules:")
        for name in missing:
            print("  ", name)
        raise RuntimeError("Could not find all decoder cross-attention modules")

    captured: dict[str, list[CapturedDecoderAttention]] = {
        name: [] for name in TARGET_DECODER_CROSS_ATTN
    }

    module_stack: list[str] = []
    hooks = []

    def make_pre_hook(module_name):
        def pre_hook(module, inputs):
            module_stack.append(module_name)

        return pre_hook

    def make_post_hook(module_name):
        def post_hook(module, inputs, output):
            if module_stack and module_stack[-1] == module_name:
                module_stack.pop()

        return post_hook

    for name in TARGET_DECODER_CROSS_ATTN:
        module = modules[name]
        hooks.append(module.register_forward_pre_hook(make_pre_hook(name)))
        hooks.append(module.register_forward_hook(make_post_hook(name)))

    original_sdpa = F.scaled_dot_product_attention

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        out = original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

        if module_stack:
            module_name = module_stack[-1]

            if module_name in captured:
                if query.ndim != 4 or key.ndim != 4:
                    raise RuntimeError(
                        f"Expected Q/K [B,H,T,D], got Q={tuple(query.shape)}, K={tuple(key.shape)}"
                    )

                call_index = len(captured[module_name])

                captured[module_name].append(
                    CapturedDecoderAttention(
                        module_name=module_name,
                        call_index=call_index,
                        query=query.detach().float().cpu(),
                        key=key.detach().float().cpu(),
                        attn_mask=(
                            attn_mask.detach().float().cpu()
                            if isinstance(attn_mask, torch.Tensor)
                            else None
                        ),
                        query_shape=tuple(query.shape),
                        key_shape=tuple(key.shape),
                        value_shape=tuple(value.shape),
                    )
                )

        return out

    F.scaled_dot_product_attention = patched_sdpa

    try:
        yield captured
    finally:
        F.scaled_dot_product_attention = original_sdpa
        for hook in hooks:
            hook.remove()


def compute_responsible_query(hs, target_mask_index):
    grad = hs.grad

    if grad is None:
        raise RuntimeError("hs.grad is None")

    if grad.ndim != 4:
        raise RuntimeError(f"Expected hs.grad [L,B,Q,C], got {tuple(grad.shape)}")

    grad_b = grad[:, 0]  # [L,Q,C]

    per_query_l2 = torch.linalg.norm(grad_b, dim=(0, 2))
    per_query_mean_abs = grad_b.abs().mean(dim=(0, 2))
    per_query_max_abs = grad_b.abs().amax(dim=(0, 2))

    q_star = int(torch.argmax(per_query_l2).detach().cpu())
    max_l2 = float(per_query_l2.max().detach().cpu()) + 1e-12

    rows = []
    for q in range(per_query_l2.numel()):
        rows.append(
            {
                "target_mask_index": target_mask_index,
                "query_index": q,
                "l2_grad": float(per_query_l2[q].detach().cpu()),
                "mean_abs_grad": float(per_query_mean_abs[q].detach().cpu()),
                "max_abs_grad": float(per_query_max_abs[q].detach().cpu()),
                "normalized_l2_grad": float((per_query_l2[q] / max_l2).detach().cpu()),
                "is_responsible_query": int(q == q_star),
            }
        )

    return q_star, rows


def apply_attention_mask(scores, attn_mask):
    """
    scores shape:
        [B,H,Q,K]

    attn_mask can be:
        [B*H,Q,K]
        [B,H,Q,K]
        [Q,K]
        or None
    """
    if attn_mask is None:
        return scores

    mask = attn_mask

    if mask.ndim == 2:
        mask = mask[None, None, :, :]
    elif mask.ndim == 3:
        b, h, q, k = scores.shape
        if mask.shape[0] == b * h:
            mask = mask.reshape(b, h, mask.shape[1], mask.shape[2])
        else:
            mask = mask[:, None, :, :]
    elif mask.ndim == 4:
        pass
    else:
        raise RuntimeError(f"Unsupported attn_mask shape: {tuple(mask.shape)}")

    mask = mask.to(scores.device)

    if mask.dtype == torch.bool:
        scores = scores.masked_fill(~mask, float("-inf"))
    else:
        scores = scores + mask

    return scores


def recompute_attention_weights(query, key, attn_mask):
    """
    Recompute attention weights from captured Q and K.

    query shape:
        [B,H,Q,D]
    key shape:
        [B,H,K,D]

    output:
        [B,H,Q,K]
    """
    d = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d)
    scores = apply_attention_mask(scores, attn_mask)
    weights = torch.softmax(scores, dim=-1)
    return weights


def normalize_map(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    denom = arr.max() + 1e-8
    return arr / denom


def save_heatmap(arr, path, title):
    arr_norm = normalize_map(arr)

    plt.figure(figsize=(6, 6))
    plt.imshow(arr_norm, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title(title)
    plt.colorbar()
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()


def save_decoder_attention_maps(captured, q_star, out_dir, top_k):
    out_dir = Path(out_dir)
    trace_dir = out_dir / "detr_decoder_cross_attention"
    trace_dir.mkdir(parents=True, exist_ok=True)

    debug_rows = []
    top_token_rows = []

    for module_name in TARGET_DECODER_CROSS_ATTN:
        calls = captured.get(module_name, [])

        if not calls:
            print(f"WARNING: no captured calls for {module_name}")
            continue

        # Usually one SDPA call per decoder cross-attention module.
        cap = calls[0]

        weights = recompute_attention_weights(
            query=cap.query,
            key=cap.key,
            attn_mask=cap.attn_mask,
        )

        if weights.ndim != 4:
            raise RuntimeError(f"Expected weights [B,H,Q,K], got {tuple(weights.shape)}")

        b, h, q, k = weights.shape

        if q_star >= q:
            raise RuntimeError(
                f"q_star={q_star} out of range for {module_name}; Q={q}"
            )

        side = int(k ** 0.5)
        if side * side != k:
            raise RuntimeError(
                f"Memory token count {k} is not square, cannot make 2D map"
            )

        layer = module_name.split(".")[3]

        # Average over heads for selected query.
        query_attn = weights[0, :, q_star, :]  # [H,K]
        query_attn_mean = query_attn.mean(dim=0)  # [K]
        query_attn_map = query_attn_mean.reshape(side, side).detach().cpu().numpy()

        npy_path = trace_dir / f"decoder_layer_{layer}_query{q_star}_attn.npy"
        png_path = trace_dir / f"decoder_layer_{layer}_query{q_star}_attn.png"

        np.save(npy_path, query_attn_map)
        save_heatmap(
            query_attn_map,
            png_path,
            f"DETR decoder layer {layer} cross-attn, query {q_star}",
        )

        flat = query_attn_mean.detach().cpu().numpy()
        top_indices = np.argsort(flat)[::-1][:top_k]

        for rank, token_idx in enumerate(top_indices, start=1):
            row = int(token_idx // side)
            col = int(token_idx % side)

            top_token_rows.append(
                {
                    "layer": layer,
                    "query_index": q_star,
                    "rank": rank,
                    "token_index": int(token_idx),
                    "row": row,
                    "col": col,
                    "attention_mass": float(flat[token_idx]),
                    "map_path": str(png_path),
                }
            )

        debug_rows.append(
            {
                "layer": layer,
                "module_name": module_name,
                "query_shape": str(cap.query_shape),
                "key_shape": str(cap.key_shape),
                "value_shape": str(cap.value_shape),
                "weights_shape": str(tuple(weights.shape)),
                "memory_tokens": k,
                "spatial_shape": f"({side}, {side})",
                "query_index": q_star,
                "npy_path": str(npy_path),
                "png_path": str(png_path),
            }
        )

    debug_csv = trace_dir / "decoder_attention_debug.csv"
    with open(debug_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(debug_rows[0].keys()))
        writer.writeheader()
        writer.writerows(debug_rows)

    top_csv = trace_dir / "decoder_top_tokens.csv"
    with open(top_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(top_token_rows[0].keys()))
        writer.writeheader()
        writer.writerows(top_token_rows)

    print(f"Saved decoder attention maps to: {trace_dir}")
    print(f"Saved decoder attention debug CSV: {debug_csv}")
    print(f"Saved decoder top tokens CSV: {top_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", default="outputs/detr_decoder_cross_attention")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mask-index", type=int, default=0)
    parser.add_argument(
        "--query-index",
        type=int,
        default=None,
        help="Optional fixed decoder query. If omitted, q* is found by gradient.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    from pam.sam3_loader import load_sam3_image_model

    out_dir = Path(args.out_dir)
    trace_dir = out_dir / "detr_decoder_cross_attention"
    trace_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading SAM3 on {args.device}")
    model, processor = load_sam3_image_model(device=args.device)
    model.eval()

    hs_storage = {}
    restore_run_decoder = patch_run_decoder_capture_hs(model, hs_storage)

    try:
        with capture_decoder_cross_attention_inputs(model) as captured:
            print(f"Running prompt pass with gradients: {args.prompt!r}")
            state = run_model_inference_with_grad(
                processor,
                args.image,
                args.prompt,
                args.device,
            )
    finally:
        restore_run_decoder()

    if "hs" not in hs_storage:
        raise RuntimeError("Did not capture DETR decoder hs")

    hs = hs_storage["hs"]

    print("Captured hs:", tuple(hs.shape), "requires_grad=", hs.requires_grad)

    masks_logits = state["masks_logits"]

    if masks_logits.ndim != 4:
        raise RuntimeError(
            f"Expected masks_logits [B,N,H,W], got {tuple(masks_logits.shape)}"
        )

    if args.target_mask_index >= masks_logits.shape[1]:
        raise RuntimeError(
            f"target_mask_index={args.target_mask_index} out of range"
        )

    target = masks_logits[0, args.target_mask_index].mean()

    print(
        "masks_logits:",
        tuple(masks_logits.shape),
        "requires_grad=",
        masks_logits.requires_grad,
    )
    print("Backward target:", float(target.detach().cpu()))

    target.backward()

    print("Backward complete")

    auto_q_star, query_rows = compute_responsible_query(
        hs=hs,
        target_mask_index=args.target_mask_index,
    )

    q_star = args.query_index if args.query_index is not None else auto_q_star

    query_csv = trace_dir / "query_gradient_scores.csv"
    with open(query_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(query_rows[0].keys()))
        writer.writeheader()
        writer.writerows(query_rows)

    metadata = {
        "image": args.image,
        "prompt": args.prompt,
        "target_mask_index": args.target_mask_index,
        "auto_responsible_query": auto_q_star,
        "used_query_index": q_star,
        "hs_shape": list(hs_storage["hs_shape"]),
        "masks_logits_shape": list(masks_logits.shape),
        "target_modules": TARGET_DECODER_CROSS_ATTN,
    }

    metadata_json = trace_dir / "query_mapping.json"
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved query mapping: {metadata_json}")
    print(f"Saved query scores: {query_csv}")
    print(f"Using decoder query q* = {q_star}")

    print("\nCaptured DETR decoder cross-attention calls:")
    for name in TARGET_DECODER_CROSS_ATTN:
        calls = captured.get(name, [])
        if not calls:
            print(f"  {name}: NO CALLS")
        for cap in calls:
            print(
                f"  {name} call {cap.call_index}: "
                f"Q={cap.query_shape} K={cap.key_shape} V={cap.value_shape}"
            )

    save_decoder_attention_maps(
        captured=captured,
        q_star=q_star,
        out_dir=out_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()