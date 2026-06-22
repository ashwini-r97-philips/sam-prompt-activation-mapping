from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def get_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for x in output:
            if isinstance(x, torch.Tensor):
                return x
    if isinstance(output, dict):
        for x in output.values():
            if isinstance(x, torch.Tensor):
                return x
    raise TypeError(f"Could not find tensor in output type: {type(output)}")


def patch_encode_prompt(model, storage):
    original = model._encode_prompt

    def wrapped(*args, **kwargs):
        prompt, prompt_mask, backbone_out = original(*args, **kwargs)

        prompt.retain_grad()
        storage["prompt"] = prompt
        storage["prompt_mask"] = prompt_mask

        return prompt, prompt_mask, backbone_out

    model._encode_prompt = wrapped

    def restore():
        model._encode_prompt = original

    return restore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mask-index", type=int, default=0)
    args = parser.parse_args()

    from pam.sam3_loader import load_sam3_image_model, run_inference_with_grad

    out_dir = Path(args.out_dir) / "prompt_token_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM3 on {args.device}")
    model, processor = load_sam3_image_model(device=args.device)
    model.eval()

    storage = {}
    restore = patch_encode_prompt(model, storage)

    try:
        print(f"Running prompt pass with gradients: {args.prompt!r}")
        state = run_inference_with_grad(processor, args.image, args.prompt, args.device)
    finally:
        restore()

    logits = state["masks_logits"]
    target = logits[0, args.target_mask_index].mean()

    print("masks_logits:", tuple(logits.shape), "requires_grad=", logits.requires_grad)
    print("Backward target:", float(target.detach().cpu()))

    model.zero_grad(set_to_none=True)
    target.backward()

    prompt_tensor = storage["prompt"]
    prompt_mask = storage.get("prompt_mask", None)
    prompt_grad = prompt_tensor.grad

    if prompt_grad is None:
        raise RuntimeError("Prompt gradient was not captured")

    # prompt shape is usually [T, B, C]
    grad_norm = torch.linalg.norm(prompt_grad.detach().float(), dim=-1)
    mean_abs_grad = prompt_grad.detach().float().abs().mean(dim=-1)
    max_abs_grad = prompt_grad.detach().float().abs().amax(dim=-1)

    if grad_norm.ndim == 2:
        grad_norm = grad_norm[:, 0]
        mean_abs_grad = mean_abs_grad[:, 0]
        max_abs_grad = max_abs_grad[:, 0]

    rows = []
    for i in range(grad_norm.numel()):
        active = ""
        if prompt_mask is not None:
            # prompt_mask usually has shape [B, T_without_extra]
            if prompt_mask.ndim == 2 and i < prompt_mask.shape[1]:
                active = int(not bool(prompt_mask[0, i].detach().cpu()))

        rows.append(
            {
                "prompt_token_index": i,
                "active_token": active,
                "l2_grad": float(grad_norm[i].detach().cpu()),
                "mean_abs_grad": float(mean_abs_grad[i].detach().cpu()),
                "max_abs_grad": float(max_abs_grad[i].detach().cpu()),
            }
        )

    max_l2 = max(r["l2_grad"] for r in rows) + 1e-12
    for r in rows:
        r["normalized_l2_grad"] = r["l2_grad"] / max_l2

    rows_sorted = sorted(rows, key=lambda r: r["l2_grad"], reverse=True)

    csv_path = out_dir / "prompt_token_importance.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    json_path = out_dir / "prompt_token_importance_top.json"
    with open(json_path, "w") as f:
        json.dump(rows_sorted[:20], f, indent=2)

    config_path = out_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print()
    print("Top prompt tokens:")
    for r in rows_sorted[:10]:
        print(
            f"token {r['prompt_token_index']} | "
            f"l2_grad={r['l2_grad']:.6e} | "
            f"normalized={r['normalized_l2_grad']:.3f} | "
            f"active={r['active_token']}"
        )


if __name__ == "__main__":
    main()