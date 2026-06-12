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


def get_tensor(output):
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item

    if isinstance(output, dict):
        for item in output.values():
            if isinstance(item, torch.Tensor):
                return item

    raise TypeError(f"Could not find tensor in output type: {type(output)}")


def register_layer_hooks(model, storage, retain_grad=False):
    hooks = []
    modules = dict(model.named_modules())

    missing = [name for name in TARGET_LAYERS if name not in modules]
    if missing:
        print("Could not find these target layers:")
        for name in missing:
            print("  ", name)

        print("\nAvailable modules containing transformer.encoder.layers:")
        for name in modules:
            if "transformer.encoder.layers" in name:
                print("  ", name)

        raise RuntimeError("Target layers not found")

    for name in TARGET_LAYERS:
        module = modules[name]

        def make_hook(layer_name):
            def hook(mod, inputs, output):
                tensor = get_tensor(output)

                if retain_grad:
                    tensor.retain_grad()
                    storage[layer_name] = tensor
                else:
                    storage[layer_name] = tensor.detach().float().cpu()

            return hook

        hooks.append(module.register_forward_hook(make_hook(name)))

    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def run_model_inference_with_grad(processor, image_path, prompt, device):
    from pam.sam3_loader import run_inference_with_grad

    return run_inference_with_grad(processor, image_path, prompt, device)


def run_model_inference_no_grad(processor, image_path, prompt, device):
    from pam.sam3_loader import run_inference

    return run_inference(processor, image_path, prompt, device)


def normalize_map(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    denom = arr.max() + 1e-8
    return arr / denom


def tensor_to_spatial_map_from_feature_tensor(tensor):
    x = tensor.squeeze()

    if x.ndim != 2:
        raise RuntimeError(
            f"Expected 2D token-feature tensor after squeeze, got shape {tuple(x.shape)}"
        )

    if x.shape[0] == 256 and x.shape[1] != 256:
        x = x.T

    norm = torch.linalg.norm(x, dim=-1)

    n = norm.numel()
    side = int(n ** 0.5)

    if side * side != n:
        raise RuntimeError(
            f"Token count {n} is not a square, tensor shape was {tuple(tensor.shape)}"
        )

    return norm.reshape(side, side).detach().float().cpu().numpy()


def save_heatmap(arr, path, title):
    arr_norm = normalize_map(arr)

    plt.figure(figsize=(6, 6))
    plt.imshow(arr_norm, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title(title)
    plt.colorbar()
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()


def save_deltaf_grad(prompt_storage, baseline_storage, out_dir, target_mask_index):
    out_dir = Path(out_dir)
    grad_dir = out_dir / "mm_grad"
    grad_dir.mkdir(parents=True, exist_ok=True)

    gradient_rows = []
    delta_grad_rows = []

    for layer_name in TARGET_LAYERS:
        z_prompt = prompt_storage[layer_name]
        z_base = baseline_storage[layer_name]

        if z_prompt.grad is None:
            raise RuntimeError(f"No gradient found for {layer_name}")

        if tuple(z_prompt.shape) != tuple(z_base.shape):
            raise RuntimeError(
                f"Shape mismatch for {layer_name}: "
                f"prompt {tuple(z_prompt.shape)} vs baseline {tuple(z_base.shape)}"
            )

        delta = z_prompt.detach().float().cpu() - z_base
        grad = z_prompt.grad.detach().float().cpu()

        delta_map = tensor_to_spatial_map_from_feature_tensor(delta)
        grad_map = tensor_to_spatial_map_from_feature_tensor(grad)
        delta_grad_map = delta_map * grad_map

        layer_id = layer_name.split(".")[-1]

        np.save(grad_dir / f"layer_{layer_id}_delta.npy", delta_map)
        np.save(grad_dir / f"layer_{layer_id}_grad.npy", grad_map)
        np.save(grad_dir / f"layer_{layer_id}_delta_grad.npy", delta_grad_map)

        save_heatmap(
            delta_map,
            grad_dir / f"layer_{layer_id}_delta.png",
            f"DeltaF Layer {layer_id}",
        )

        save_heatmap(
            grad_map,
            grad_dir / f"layer_{layer_id}_grad.png",
            f"Gradient Layer {layer_id}",
        )

        save_heatmap(
            delta_grad_map,
            grad_dir / f"layer_{layer_id}_delta_grad.png",
            f"DeltaF x Grad Layer {layer_id}",
        )

        gradient_rows.append(
            {
                "layer": layer_id,
                "layer_name": layer_name,
                "shape": str(tuple(z_prompt.shape)),
                "mean_abs_grad": float(grad_map.mean()),
                "max_abs_grad": float(grad_map.max()),
                "l2_grad": float(np.linalg.norm(grad_map)),
                "target_mask_index": target_mask_index,
            }
        )

        delta_grad_rows.append(
            {
                "layer": layer_id,
                "layer_name": layer_name,
                "mean_delta": float(delta_map.mean()),
                "max_delta": float(delta_map.max()),
                "mean_grad": float(grad_map.mean()),
                "max_grad": float(grad_map.max()),
                "mean_delta_grad": float(delta_grad_map.mean()),
                "max_delta_grad": float(delta_grad_map.max()),
                "spatial_h": delta_grad_map.shape[0],
                "spatial_w": delta_grad_map.shape[1],
                "target_mask_index": target_mask_index,
            }
        )

    with open(grad_dir / "gradient_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(gradient_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gradient_rows)

    with open(grad_dir / "delta_grad_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(delta_grad_rows[0].keys()))
        writer.writeheader()
        writer.writerows(delta_grad_rows)

    print(f"Saved gradient outputs to: {grad_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--baseline-prompt", default="")
    parser.add_argument("--out-dir", default="outputs/multimodal_deltaf_grad")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mask-index", type=int, default=0)
    args = parser.parse_args()

    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    from pam.sam3_loader import load_sam3_image_model

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading SAM3 on {args.device}")
    model, processor = load_sam3_image_model(device=args.device)
    model.eval()

    print(f"Running baseline pass without gradients: {args.baseline_prompt!r}")
    baseline_storage = {}
    hooks = register_layer_hooks(model, baseline_storage, retain_grad=False)
    try:
        run_model_inference_no_grad(
            processor,
            args.image,
            args.baseline_prompt,
            args.device,
        )
    finally:
        remove_hooks(hooks)

    print(f"Running prompt pass with gradients: {args.prompt!r}")
    prompt_storage = {}
    hooks = register_layer_hooks(model, prompt_storage, retain_grad=True)
    try:
        state = run_model_inference_with_grad(
            processor,
            args.image,
            args.prompt,
            args.device,
        )
    finally:
        remove_hooks(hooks)

    print("Captured prompt layers:")
    for k, v in prompt_storage.items():
        print(f"  {k}: {tuple(v.shape)}, requires_grad={v.requires_grad}")

    print("Captured baseline layers:")
    for k, v in baseline_storage.items():
        print(f"  {k}: {tuple(v.shape)}")

    masks_logits = state["masks_logits"]

    if masks_logits.ndim != 4:
        raise RuntimeError(
            f"Expected masks_logits shape [B, N, H, W], got {tuple(masks_logits.shape)}"
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

    save_deltaf_grad(
        prompt_storage=prompt_storage,
        baseline_storage=baseline_storage,
        out_dir=out_dir,
        target_mask_index=args.target_mask_index,
    )


if __name__ == "__main__":
    main()code outputs/pam_summary.md