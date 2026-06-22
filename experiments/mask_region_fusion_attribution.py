from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


TARGET_LAYERS = [
    "transformer.encoder.layers.0",
    "transformer.encoder.layers.1",
    "transformer.encoder.layers.2",
    "transformer.encoder.layers.3",
    "transformer.encoder.layers.4",
    "transformer.encoder.layers.5",
]


def normalize(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    return arr / (arr.max() + 1e-8)


def save_map(arr, path, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(normalize(arr), cmap="viridis", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(title)
    plt.colorbar()
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()


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


def register_fusion_hooks(model, storage):
    hooks = []
    modules = dict(model.named_modules())

    for name in TARGET_LAYERS:
        if name not in modules:
            raise RuntimeError(f"Missing layer: {name}")

        def make_hook(layer_name):
            def hook(mod, inputs, output):
                tensor = get_tensor(output)
                tensor.retain_grad()
                storage[layer_name] = tensor
            return hook

        hooks.append(modules[name].register_forward_hook(make_hook(name)))

    return hooks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mask-index", type=int, default=0)
    parser.add_argument("--target-mode", choices=["mean", "pixel", "region"], default="mean")
    parser.add_argument("--pixel-y", type=int, default=None)
    parser.add_argument("--pixel-x", type=int, default=None)
    parser.add_argument("--y1", type=int, default=None)
    parser.add_argument("--y2", type=int, default=None)
    parser.add_argument("--x1", type=int, default=None)
    parser.add_argument("--x2", type=int, default=None)
    args = parser.parse_args()

    from pam.sam3_loader import load_sam3_image_model, run_inference_with_grad

    out_dir = Path(args.out_dir) / "mask_region_fusion_attribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM3 on {args.device}")
    model, processor = load_sam3_image_model(device=args.device)
    model.eval()

    fusion_storage = {}
    hooks = register_fusion_hooks(model, fusion_storage)

    try:
        print(f"Running prompt pass with gradients: {args.prompt!r}")
        state = run_inference_with_grad(processor, args.image, args.prompt, args.device)
    finally:
        for h in hooks:
            h.remove()

    logits = state["masks_logits"]

    if args.target_mode == "mean":
        target = logits[0, args.target_mask_index].mean()
        target_desc = "mean_mask_logit"
    elif args.target_mode == "pixel":
        if args.pixel_y is None or args.pixel_x is None:
            raise RuntimeError("--pixel-y and --pixel-x are required for pixel mode")
        target = logits[0, args.target_mask_index, args.pixel_y, args.pixel_x]
        target_desc = f"pixel_y{args.pixel_y}_x{args.pixel_x}"
    else:
        needed = [args.y1, args.y2, args.x1, args.x2]
        if any(v is None for v in needed):
            raise RuntimeError("--y1 --y2 --x1 --x2 are required for region mode")
        target = logits[0, args.target_mask_index, args.y1:args.y2, args.x1:args.x2].mean()
        target_desc = f"region_y{args.y1}_{args.y2}_x{args.x1}_{args.x2}"

    print("masks_logits:", tuple(logits.shape), "requires_grad=", logits.requires_grad)
    print("Backward target:", float(target.detach().cpu()), target_desc)

    model.zero_grad(set_to_none=True)
    target.backward()

    rows = []

    for layer in range(6):
        layer_name = f"transformer.encoder.layers.{layer}"
        fusion = fusion_storage[layer_name]
        grad = fusion.grad

        if grad is None:
            raise RuntimeError(f"No grad for {layer_name}")

        grad = grad.detach().float().cpu().squeeze()
        grad_norm = torch.linalg.norm(grad, dim=-1)

        n = grad_norm.numel()
        side = int(n ** 0.5)
        if side * side != n:
            raise RuntimeError(f"Token count is not square: {n}")

        spatial = grad_norm.reshape(side, side).numpy()

        npy_path = out_dir / f"layer_{layer}_{target_desc}_fusion_grad.npy"
        png_path = out_dir / f"layer_{layer}_{target_desc}_fusion_grad.png"

        np.save(npy_path, spatial)
        save_map(spatial, png_path, f"Layer {layer} Fusion Grad: {target_desc}")

        rows.append(
            {
                "layer": layer,
                "target_desc": target_desc,
                "mean_grad": float(spatial.mean()),
                "max_grad": float(spatial.max()),
                "spatial_shape": str(spatial.shape),
                "npy_path": str(npy_path),
                "png_path": str(png_path),
            }
        )

    csv_path = out_dir / "mask_region_fusion_attribution.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved mask-region fusion attribution: {csv_path}")


if __name__ == "__main__":
    main()