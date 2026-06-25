from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    a = mask_a.detach().bool().cpu()
    b = mask_b.detach().bool().cpu()
    inter = torch.logical_and(a, b).sum().float()
    union = torch.logical_or(a, b).sum().float()
    if union.item() == 0:
        return 0.0
    return float((inter / union).item())


def patch_run_decoder_memory_ablation(model, token_indices, mode="zero"):
    original = model._run_decoder

    def wrapped(*args, **kwargs):
        memory = kwargs.get("memory", None)
        if memory is None:
            raise RuntimeError("Expected memory in _run_decoder kwargs")

        memory = memory.clone()

        for idx in token_indices:
            if idx >= memory.shape[0]:
                continue
            if mode == "zero":
                memory[idx] = 0.0
            elif mode == "mean":
                memory[idx] = memory.mean(dim=0, keepdim=True)
            else:
                raise ValueError(mode)

        kwargs["memory"] = memory
        return original(*args, **kwargs)

    model._run_decoder = wrapped

    def restore():
        model._run_decoder = original

    return restore


def run_inference(model, processor, image, prompt, device, patcher=None):
    from pam.sam3_loader import run_inference

    restore = None
    if patcher is not None:
        restore = patcher()

    try:
        state = run_inference(processor, image, prompt, device)
    finally:
        if restore is not None:
            restore()

    return state


def extract_score_mask(state, target_mask_index):
    logits = state["masks_logits"]
    masks = state["masks"]

    if logits.numel() == 0 or logits.shape[0] == 0 or logits.shape[1] == 0:
        return None, None

    score = float(logits[0, target_mask_index].mean().detach().cpu())
    mask = masks[0, target_mask_index].detach().cpu()
    return score, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--fused-attn", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mask-index", type=int, default=0)
    parser.add_argument("--k", nargs="+", type=int, default=[50, 100, 200, 500])
    parser.add_argument("--random-trials", type=int, default=10)
    parser.add_argument("--mode", choices=["zero", "mean"], default="zero")
    args = parser.parse_args()

    from pam.sam3_loader import load_sam3_image_model

    out_dir = Path(args.out_dir) / "fused_decoder_attention_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM3 on {args.device}")
    model, processor = load_sam3_image_model(device=args.device)
    model.eval()

    print("Running original")
    original = run_inference(model, processor, args.image, args.prompt, args.device)
    original_score, original_mask = extract_score_mask(original, args.target_mask_index)

    if original_score is None:
        raise RuntimeError("Original produced no mask")

    attn = np.load(args.fused_attn).reshape(-1)
    ranked = np.argsort(attn)[::-1]

    rng = np.random.default_rng(0)
    all_indices = np.arange(attn.shape[0])

    rows = []

    for k in args.k:
        top_tokens = [int(i) for i in ranked[:k]]

        print(f"k={k}: top fused attention ablation")

        state = run_inference(
            model,
            processor,
            args.image,
            args.prompt,
            args.device,
            patcher=lambda toks=top_tokens: patch_run_decoder_memory_ablation(
                model,
                toks,
                mode=args.mode,
            ),
        )

        score, mask = extract_score_mask(state, args.target_mask_index)

        if score is None:
            top_score_drop = "NO_MASK"
            top_iou = "NO_MASK"
        else:
            top_score_drop = original_score - score
            top_iou = mask_iou(original_mask, mask)

        rows.append(
            {
                "condition": "top_fused_attention",
                "k": k,
                "trial": "",
                "original_score": original_score,
                "ablated_score": "NO_MASK" if score is None else score,
                "score_drop": top_score_drop,
                "iou_with_original": top_iou,
            }
        )

        for trial in range(args.random_trials):
            random_tokens = [int(i) for i in rng.choice(all_indices, size=k, replace=False)]

            state = run_inference(
                model,
                processor,
                args.image,
                args.prompt,
                args.device,
                patcher=lambda toks=random_tokens: patch_run_decoder_memory_ablation(
                    model,
                    toks,
                    mode=args.mode,
                ),
            )

            score, mask = extract_score_mask(state, args.target_mask_index)

            rows.append(
                {
                    "condition": "random",
                    "k": k,
                    "trial": trial,
                    "original_score": original_score,
                    "ablated_score": "NO_MASK" if score is None else score,
                    "score_drop": "NO_MASK" if score is None else original_score - score,
                    "iou_with_original": "NO_MASK" if score is None else mask_iou(original_mask, mask),
                }
            )

    csv_path = out_dir / "fused_decoder_attention_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()