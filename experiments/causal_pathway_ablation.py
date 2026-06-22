from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def load_top_tokens(delta_grad_path: Path, k: int) -> list[int]:
    arr = np.load(delta_grad_path)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D map, got {arr.shape}: {delta_grad_path}")

    flat = arr.reshape(-1)
    idx = np.argsort(flat)[::-1][:k]
    return [int(i) for i in idx]


def load_random_tokens(num_tokens: int, k: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(i) for i in rng.choice(num_tokens, size=k, replace=False)]


def mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    a = mask_a.detach().bool().cpu()
    b = mask_b.detach().bool().cpu()

    inter = torch.logical_and(a, b).sum().float()
    union = torch.logical_or(a, b).sum().float()

    if union.item() == 0:
        return 0.0

    return float((inter / union).item())


def patch_run_decoder_ablate_memory(model, token_indices, mode="zero"):
    """
    Ablate fusion-memory tokens immediately before DETR decoder.

    memory shape entering _run_decoder:
        [5184, batch, 256]

    token_indices are spatial memory token indices in [0, 5184).
    """
    original_run_decoder = model._run_decoder

    def wrapped_run_decoder(*args, **kwargs):
        memory = kwargs.get("memory", None)

        if memory is None:
            raise RuntimeError("Expected memory in _run_decoder kwargs")

        if memory.ndim != 3:
            raise RuntimeError(f"Expected memory [T,B,C], got {tuple(memory.shape)}")

        memory = memory.clone()

        if mode == "zero":
            memory[token_indices, :, :] = 0.0
        elif mode == "mean":
            mean_token = memory.mean(dim=0, keepdim=True)
            memory[token_indices, :, :] = mean_token
        else:
            raise ValueError(f"Unknown ablation mode: {mode}")

        kwargs["memory"] = memory
        return original_run_decoder(*args, **kwargs)

    model._run_decoder = wrapped_run_decoder

    def restore():
        model._run_decoder = original_run_decoder

    return restore


def run_inference_with_optional_ablation(
    model,
    processor,
    image_path,
    prompt,
    device,
    token_indices=None,
    ablation_mode="zero",
):
    from pam.sam3_loader import run_inference

    restore = None

    if token_indices is not None:
        restore = patch_run_decoder_ablate_memory(
            model=model,
            token_indices=token_indices,
            mode=ablation_mode,
        )

    try:
        state = run_inference(processor, image_path, prompt, device)
    finally:
        if restore is not None:
            restore()

    return state


def extract_score_and_mask(state, target_mask_index: int):
    logits = state["masks_logits"]
    masks = state["masks"]

    if logits.ndim != 4:
        raise RuntimeError(f"Expected masks_logits [B,N,H,W], got {tuple(logits.shape)}")

    if target_mask_index >= logits.shape[1]:
        raise RuntimeError(
            f"target_mask_index={target_mask_index} out of range for {tuple(logits.shape)}"
        )

    score = float(logits[0, target_mask_index].mean().detach().cpu())
    mask = masks[0, target_mask_index].detach().cpu()

    return score, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--delta-grad-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mask-index", type=int, default=0)
    parser.add_argument("--k", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--random-trials", type=int, default=3)
    parser.add_argument("--ablation-mode", choices=["zero", "mean"], default="zero")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    from pam.sam3_loader import load_sam3_image_model

    delta_grad_dir = Path(args.delta_grad_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM3 on {args.device}")
    model, processor = load_sam3_image_model(device=args.device)
    model.eval()

    print("Running original inference")
    original_state = run_inference_with_optional_ablation(
        model=model,
        processor=processor,
        image_path=args.image,
        prompt=args.prompt,
        device=args.device,
        token_indices=None,
    )

    original_score, original_mask = extract_score_and_mask(
        original_state, args.target_mask_index
    )

    print(f"Original target score: {original_score:.6f}")

    rows = []

    for layer in range(6):
        delta_grad_path = delta_grad_dir / f"layer_{layer}_delta_grad.npy"

        if not delta_grad_path.is_file():
            raise FileNotFoundError(f"Missing DeltaF x Grad map: {delta_grad_path}")

        delta_map = np.load(delta_grad_path)
        num_tokens = int(delta_map.size)

        for k in args.k:
            print(f"Layer {layer}, k={k}: top-token ablation")

            top_tokens = load_top_tokens(delta_grad_path, k)

            top_state = run_inference_with_optional_ablation(
                model=model,
                processor=processor,
                image_path=args.image,
                prompt=args.prompt,
                device=args.device,
                token_indices=top_tokens,
                ablation_mode=args.ablation_mode,
            )

            top_score, top_mask = extract_score_and_mask(
                top_state, args.target_mask_index
            )

            top_drop = original_score - top_score
            top_iou = mask_iou(original_mask, top_mask)
            top_iou_drop = 1.0 - top_iou

            rows.append(
                {
                    "layer": layer,
                    "k": k,
                    "condition": "top_delta_grad",
                    "trial": 0,
                    "original_score": original_score,
                    "ablated_score": top_score,
                    "score_drop": top_drop,
                    "iou_with_original": top_iou,
                    "iou_drop": top_iou_drop,
                    "ablation_mode": args.ablation_mode,
                    "delta_grad_path": str(delta_grad_path),
                    "token_indices_preview": str(top_tokens[:20]),
                }
            )

            print(
                f"  top: score={top_score:.6f}, "
                f"drop={top_drop:.6f}, iou={top_iou:.4f}"
            )

            random_drops = []
            random_ious = []

            for trial in range(args.random_trials):
                seed = args.seed + layer * 1000 + k * 10 + trial
                random_tokens = load_random_tokens(num_tokens, k, seed)

                random_state = run_inference_with_optional_ablation(
                    model=model,
                    processor=processor,
                    image_path=args.image,
                    prompt=args.prompt,
                    device=args.device,
                    token_indices=random_tokens,
                    ablation_mode=args.ablation_mode,
                )

                random_score, random_mask = extract_score_and_mask(
                    random_state, args.target_mask_index
                )

                random_drop = original_score - random_score
                random_iou = mask_iou(original_mask, random_mask)
                random_iou_drop = 1.0 - random_iou

                random_drops.append(random_drop)
                random_ious.append(random_iou)

                rows.append(
                    {
                        "layer": layer,
                        "k": k,
                        "condition": "random",
                        "trial": trial,
                        "original_score": original_score,
                        "ablated_score": random_score,
                        "score_drop": random_drop,
                        "iou_with_original": random_iou,
                        "iou_drop": random_iou_drop,
                        "ablation_mode": args.ablation_mode,
                        "delta_grad_path": str(delta_grad_path),
                        "token_indices_preview": str(random_tokens[:20]),
                    }
                )

            print(
                f"  random mean drop={np.mean(random_drops):.6f}, "
                f"random mean iou={np.mean(random_ious):.4f}"
            )

    csv_path = out_dir / "causal_ablation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for layer in range(6):
        for k in args.k:
            top = [
                r for r in rows
                if r["layer"] == layer and r["k"] == k and r["condition"] == "top_delta_grad"
            ][0]
            randoms = [
                r for r in rows
                if r["layer"] == layer and r["k"] == k and r["condition"] == "random"
            ]

            random_score_drop_mean = float(np.mean([r["score_drop"] for r in randoms]))
            random_iou_drop_mean = float(np.mean([r["iou_drop"] for r in randoms]))

            summary_rows.append(
                {
                    "layer": layer,
                    "k": k,
                    "top_score_drop": top["score_drop"],
                    "random_score_drop_mean": random_score_drop_mean,
                    "score_drop_advantage": top["score_drop"] - random_score_drop_mean,
                    "top_iou_drop": top["iou_drop"],
                    "random_iou_drop_mean": random_iou_drop_mean,
                    "iou_drop_advantage": top["iou_drop"] - random_iou_drop_mean,
                }
            )

    summary_csv = out_dir / "causal_ablation_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    config_path = out_dir / "causal_ablation_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print()
    print(f"Saved causal ablation CSV: {csv_path}")
    print(f"Saved causal ablation summary: {summary_csv}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()