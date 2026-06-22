from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def run_model_inference_with_grad(processor, image_path, prompt, device):
    from pam.sam3_loader import run_inference_with_grad

    return run_inference_with_grad(processor, image_path, prompt, device)


def patch_run_decoder_capture_hs(model, storage):
    """
    Patch model._run_decoder so we can retain gradients on hs.

    In SAM3Image._run_decoder:
        out, hs = self._run_decoder(...)

    hs after transpose has shape:
        [num_layers, batch, num_queries, channels]

    pred_masks are generated from hs, so gradients from final masks_logits
    should flow back to hs.
    """
    if not hasattr(model, "_run_decoder"):
        raise RuntimeError("model does not have _run_decoder")

    original_run_decoder = model._run_decoder

    def wrapped_run_decoder(*args, **kwargs):
        out, hs = original_run_decoder(*args, **kwargs)

        if not isinstance(hs, torch.Tensor):
            raise RuntimeError(f"Expected hs to be tensor, got {type(hs)}")

        if hs.ndim != 4:
            raise RuntimeError(f"Expected hs shape [L,B,Q,C], got {tuple(hs.shape)}")

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


def compute_query_gradient_scores(hs, target_mask_index):
    """
    hs shape:
        [L, B, Q, C]

    hs.grad shape:
        [L, B, Q, C]

    We aggregate gradient norm over layers and channels for each query q.
    """
    grad = hs.grad

    if grad is None:
        raise RuntimeError("hs.grad is None. Backward did not reach decoder hs.")

    if grad.ndim != 4:
        raise RuntimeError(f"Expected hs.grad [L,B,Q,C], got {tuple(grad.shape)}")

    # Use batch 0.
    grad_b = grad[:, 0]  # [L,Q,C]

    # Per query, aggregate across layers and channels.
    per_query_l2 = torch.linalg.norm(grad_b, dim=(0, 2))  # [Q]
    per_query_mean_abs = grad_b.abs().mean(dim=(0, 2))  # [Q]
    per_query_max_abs = grad_b.abs().amax(dim=(0, 2))  # [Q]

    q_star = int(torch.argmax(per_query_l2).detach().cpu())

    rows = []
    max_l2 = float(per_query_l2.max().detach().cpu()) + 1e-12

    for q in range(per_query_l2.numel()):
        rows.append(
            {
                "target_mask_index": target_mask_index,
                "query_index": q,
                "l2_grad": float(per_query_l2[q].detach().cpu()),
                "mean_abs_grad": float(per_query_mean_abs[q].detach().cpu()),
                "max_abs_grad": float(per_query_max_abs[q].detach().cpu()),
                "normalized_l2_grad": float(
                    (per_query_l2[q] / max_l2).detach().cpu()
                ),
                "is_responsible_query": int(q == q_star),
            }
        )

    return q_star, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", default="outputs/decoder_query_trace")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mask-index", type=int, default=0)
    args = parser.parse_args()

    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    from pam.sam3_loader import load_sam3_image_model

    out_dir = Path(args.out_dir)
    trace_dir = out_dir / "decoder_query_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading SAM3 on {args.device}")
    model, processor = load_sam3_image_model(device=args.device)
    model.eval()

    storage = {}
    restore_run_decoder = patch_run_decoder_capture_hs(model, storage)

    try:
        print(f"Running prompt pass with gradients: {args.prompt!r}")
        state = run_model_inference_with_grad(
            processor,
            args.image,
            args.prompt,
            args.device,
        )
    finally:
        restore_run_decoder()

    if "hs" not in storage:
        raise RuntimeError("Did not capture decoder hs")

    hs = storage["hs"]
    print("Captured hs:", tuple(hs.shape), "requires_grad=", hs.requires_grad)

    masks_logits = state["masks_logits"]
    if masks_logits.ndim != 4:
        raise RuntimeError(
            f"Expected masks_logits [B,N,H,W], got {tuple(masks_logits.shape)}"
        )

    num_masks = masks_logits.shape[1]
    if args.target_mask_index >= num_masks:
        raise RuntimeError(
            f"target_mask_index={args.target_mask_index} out of range; "
            f"num_masks={num_masks}"
        )

    print(
        "masks_logits:",
        tuple(masks_logits.shape),
        "requires_grad=",
        masks_logits.requires_grad,
    )

    target_mask_logits = masks_logits[0, args.target_mask_index]
    target = target_mask_logits.mean()

    print("Backward target:", float(target.detach().cpu()))
    target.backward()
    print("Backward complete")

    q_star, query_rows = compute_query_gradient_scores(
        hs=hs,
        target_mask_index=args.target_mask_index,
    )

    query_csv = trace_dir / "query_gradient_scores.csv"
    with open(query_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(query_rows[0].keys()))
        writer.writeheader()
        writer.writerows(query_rows)

    metadata = {
        "image": args.image,
        "prompt": args.prompt,
        "target_mask_index": args.target_mask_index,
        "responsible_query": q_star,
        "hs_shape": list(storage["hs_shape"]),
        "masks_logits_shape": list(masks_logits.shape),
        "mapping": {
            "final_target_mask_index": args.target_mask_index,
            "decoder_query_q_star": q_star,
        },
    }

    metadata_json = trace_dir / "query_mapping.json"
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved query scores: {query_csv}")
    print(f"Saved query mapping: {metadata_json}")
    print()
    print("Responsible decoder query:")
    print(f"  target_mask_index {args.target_mask_index} -> q* = {q_star}")

    top_rows = sorted(query_rows, key=lambda r: r["l2_grad"], reverse=True)[:10]
    print("\nTop 10 queries by gradient norm:")
    for row in top_rows:
        print(
            f"query {row['query_index']} | "
            f"l2_grad={row['l2_grad']:.6e} | "
            f"normalized={row['normalized_l2_grad']:.3f}"
        )


if __name__ == "__main__":
    main()