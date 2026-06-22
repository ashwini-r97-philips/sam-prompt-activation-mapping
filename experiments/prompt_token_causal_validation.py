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


def patch_encode_prompt_token_ablation(model, token_indices, mode="zero"):
    original = model._encode_prompt

    def wrapped(*args, **kwargs):
        prompt, prompt_mask, backbone_out = original(*args, **kwargs)
        prompt = prompt.clone()

        for idx in token_indices:
            if idx >= prompt.shape[0]:
                continue

            if mode == "zero":
                prompt[idx] = 0.0
            elif mode == "mean":
                prompt[idx] = prompt.mean(dim=0, keepdim=False)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        return prompt, prompt_mask, backbone_out

    model._encode_prompt = wrapped

    def restore():
        model._encode_prompt = original

    return restore


def patch_run_decoder_fusion_ablation(model, token_indices, mode="zero"):
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
                raise ValueError(f"Unknown mode: {mode}")

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

    if target_mask_index >= logits.shape[1]:
        return None, None

    score = float(logits[0, target_mask_index].mean().detach().cpu())
    mask = masks[0, target_mask_index].detach().cpu()
    return score, mask


def top_fusion_tokens_from_pathway(
    pathway_csv: Path,
    prompt_token_index: int,
    layer: int,
    k: int,
):
    import pandas as pd

    df = pd.read_csv(pathway_csv)
    row = df[
        (df["prompt_token_index"] == prompt_token_index)
        & (df["layer"] == layer)
    ]

    if row.empty:
        raise RuntimeError(
            f"No pathway row for layer={layer}, token={prompt_token_index}"
        )

    npy_path = Path(row.iloc[0]["npy_path"])
    arr = np.load(npy_path)
    flat = arr.reshape(-1)
    idx = np.argsort(flat)[::-1][:k]
    return [int(i) for i in idx]


def add_no_mask_row(
    rows,
    condition,
    token_idx,
    original_score,
    mode,
    fusion_layer="",
    fusion_k="",
):
    rows.append(
        {
            "condition": condition,
            "prompt_token_index": token_idx,
            "fusion_layer": fusion_layer,
            "fusion_k": fusion_k,
            "original_score": original_score,
            "ablated_score": "NO_MASK",
            "score_drop": "NO_MASK",
            "iou_with_original": "NO_MASK",
            "mode": mode,
        }
    )


def add_result_row(
    rows,
    condition,
    token_idx,
    original_score,
    original_mask,
    score,
    mask,
    mode,
    fusion_layer="",
    fusion_k="",
):
    rows.append(
        {
            "condition": condition,
            "prompt_token_index": token_idx,
            "fusion_layer": fusion_layer,
            "fusion_k": fusion_k,
            "original_score": original_score,
            "ablated_score": score,
            "score_drop": original_score - score,
            "iou_with_original": mask_iou(original_mask, mask),
            "mode": mode,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mask-index", type=int, default=0)
    parser.add_argument("--prompt-token-indices", type=int, nargs="+", required=True)
    parser.add_argument("--pathway-csv", default=None)
    parser.add_argument("--fusion-layer", type=int, default=5)
    parser.add_argument("--fusion-k", type=int, default=500)
    parser.add_argument("--mode", choices=["zero", "mean"], default="zero")
    args = parser.parse_args()

    from pam.sam3_loader import load_sam3_image_model

    out_dir = Path(args.out_dir) / "prompt_token_causal_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM3 on {args.device}")
    model, processor = load_sam3_image_model(device=args.device)
    model.eval()

    print("Running original")
    original_state = run_inference(
        model,
        processor,
        args.image,
        args.prompt,
        args.device,
    )
    original_score, original_mask = extract_score_mask(
        original_state,
        args.target_mask_index,
    )

    if original_score is None:
        raise RuntimeError("Original inference produced no valid mask")

    rows = []

    for token_idx in args.prompt_token_indices:
        print(f"Prompt token ablation: token {token_idx}")

        state = run_inference(
            model,
            processor,
            args.image,
            args.prompt,
            args.device,
            patcher=lambda idx=token_idx: patch_encode_prompt_token_ablation(
                model,
                [idx],
                mode=args.mode,
            ),
        )

        score, mask = extract_score_mask(state, args.target_mask_index)

        if score is None:
            add_no_mask_row(
                rows=rows,
                condition="prompt_token_ablation",
                token_idx=token_idx,
                original_score=original_score,
                mode=args.mode,
            )
        else:
            add_result_row(
                rows=rows,
                condition="prompt_token_ablation",
                token_idx=token_idx,
                original_score=original_score,
                original_mask=original_mask,
                score=score,
                mask=mask,
                mode=args.mode,
            )

        if args.pathway_csv is not None:
            top_tokens = top_fusion_tokens_from_pathway(
                Path(args.pathway_csv),
                prompt_token_index=token_idx,
                layer=args.fusion_layer,
                k=args.fusion_k,
            )

            print(
                f"Token-specific fusion ablation: token {token_idx}, "
                f"layer {args.fusion_layer}, k={args.fusion_k}"
            )

            state = run_inference(
                model,
                processor,
                args.image,
                args.prompt,
                args.device,
                patcher=lambda toks=top_tokens: patch_run_decoder_fusion_ablation(
                    model,
                    toks,
                    mode=args.mode,
                ),
            )

            score, mask = extract_score_mask(state, args.target_mask_index)

            if score is None:
                add_no_mask_row(
                    rows=rows,
                    condition="token_specific_fusion_ablation",
                    token_idx=token_idx,
                    original_score=original_score,
                    mode=args.mode,
                    fusion_layer=args.fusion_layer,
                    fusion_k=args.fusion_k,
                )
            else:
                add_result_row(
                    rows=rows,
                    condition="token_specific_fusion_ablation",
                    token_idx=token_idx,
                    original_score=original_score,
                    original_mask=original_mask,
                    score=score,
                    mask=mask,
                    mode=args.mode,
                    fusion_layer=args.fusion_layer,
                    fusion_k=args.fusion_k,
                )

    csv_path = out_dir / "prompt_token_causal_validation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved: {csv_path}")
    print()

    for r in rows:
        print(
            f"{r['condition']} token={r['prompt_token_index']} "
            f"drop={r['score_drop']} "
            f"iou={r['iou_with_original']}"
        )


if __name__ == "__main__":
    main()