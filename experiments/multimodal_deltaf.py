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


def register_layer_hooks(model, storage):
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
                storage[layer_name] = tensor.detach().float().cpu()

            return hook

        hooks.append(module.register_forward_hook(make_hook(name)))

    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def run_model_inference(processor, image_path, prompt, device):
    from pam.sam3_loader import run_inference

    return run_inference(processor, image_path, prompt, device)


def normalize_map(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    denom = arr.max() + 1e-8
    return arr / denom


def tensor_to_spatial_map(tensor):
    x = tensor.squeeze()

    if x.ndim != 2:
        raise RuntimeError(
            f"Expected 2D token-feature tensor after squeeze, got shape {tuple(x.shape)}"
        )

    if x.shape[0] == 256 and x.shape[1] != 256:
        x = x.T

    delta_norm = torch.linalg.norm(x, dim=-1)

    n = delta_norm.numel()
    side = int(n ** 0.5)

    if side * side != n:
        raise RuntimeError(
            f"Token count {n} is not a square, tensor shape was {tuple(tensor.shape)}"
        )

    return delta_norm.reshape(side, side).numpy()


def save_deltaf(prompt_storage, baseline_storage, out_dir):
    out_dir = Path(out_dir)
    delta_dir = out_dir / "mm_delta"
    delta_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for layer_name in TARGET_LAYERS:
        z_prompt = prompt_storage[layer_name]
        z_base = baseline_storage[layer_name]

        if z_prompt.shape != z_base.shape:
            raise RuntimeError(
                f"Shape mismatch for {layer_name}: "
                f"prompt {tuple(z_prompt.shape)} vs baseline {tuple(z_base.shape)}"
            )

        delta = z_prompt - z_base
        spatial = tensor_to_spatial_map(delta)
        spatial_norm = normalize_map(spatial)

        layer_id = layer_name.split(".")[-1]

        np.save(delta_dir / f"layer_{layer_id}_delta.npy", spatial)

        plt.figure(figsize=(6, 6))
        plt.imshow(spatial_norm, cmap="viridis", vmin=0.0, vmax=1.0)
        plt.axis("off")
        plt.title(f"DeltaF Layer {layer_id}")
        plt.colorbar()
        plt.savefig(
            delta_dir / f"layer_{layer_id}_delta.png",
            bbox_inches="tight",
            dpi=200,
        )
        plt.close()

        rows.append(
            {
                "layer": layer_id,
                "layer_name": layer_name,
                "prompt_shape": str(tuple(z_prompt.shape)),
                "baseline_shape": str(tuple(z_base.shape)),
                "delta_shape": str(tuple(delta.shape)),
                "mean_delta": float(spatial.mean()),
                "max_delta": float(spatial.max()),
                "spatial_shape": str(tuple(spatial.shape)),
                "heatmap_path": str(delta_dir / f"layer_{layer_id}_delta.png"),
            }
        )

    with open(delta_dir / "delta_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved DeltaF outputs to: {delta_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--baseline-prompt", default="")
    parser.add_argument("--out-dir", default="outputs/multimodal_deltaf")
    parser.add_argument("--device", default="cuda")
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

    print(f"Running prompt pass: {args.prompt!r}")
    prompt_storage = {}
    hooks = register_layer_hooks(model, prompt_storage)
    try:
        run_model_inference(processor, args.image, args.prompt, args.device)
    finally:
        remove_hooks(hooks)

    print(f"Running baseline pass: {args.baseline_prompt!r}")
    baseline_storage = {}
    hooks = register_layer_hooks(model, baseline_storage)
    try:
        run_model_inference(processor, args.image, args.baseline_prompt, args.device)
    finally:
        remove_hooks(hooks)

    print("Captured prompt layers:")
    for k, v in prompt_storage.items():
        print(f"  {k}: {tuple(v.shape)}")

    print("Captured baseline layers:")
    for k, v in baseline_storage.items():
        print(f"  {k}: {tuple(v.shape)}")

    save_deltaf(prompt_storage, baseline_storage, out_dir)


if __name__ == "__main__":
    main()