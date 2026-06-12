from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


TARGET_ATTENTION_MODULES = []

for layer in range(6):
    TARGET_ATTENTION_MODULES.append(
        f"transformer.encoder.layers.{layer}.self_attn"
    )
    TARGET_ATTENTION_MODULES.append(
        f"transformer.encoder.layers.{layer}.cross_attn_image"
    )


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


def get_num_heads(module):
    if hasattr(module, "num_heads"):
        return int(module.num_heads)
    if hasattr(module, "n_heads"):
        return int(module.n_heads)
    return 8


def register_attention_output_hooks(model, storage):
    hooks = []
    modules = dict(model.named_modules())

    missing = [name for name in TARGET_ATTENTION_MODULES if name not in modules]
    if missing:
        print("Missing target attention modules:")
        for name in missing:
            print("  ", name)

        print("\nAvailable modules containing transformer.encoder.layers:")
        for name in modules:
            if "transformer.encoder.layers" in name:
                print("  ", name)

        raise RuntimeError("Some target attention modules were not found.")

    for name in TARGET_ATTENTION_MODULES:
        module = modules[name]
        num_heads = get_num_heads(module)

        def make_hook(module_name, heads):
            def hook(mod, inputs, output):
                tensor = get_tensor(output)

                if not tensor.requires_grad:
                    raise RuntimeError(
                        f"Output tensor for {module_name} does not require grad."
                    )

                tensor.retain_grad()

                storage[module_name] = {
                    "tensor": tensor,
                    "num_heads": heads,
                    "shape": tuple(tensor.shape),
                }

            return hook

        hooks.append(module.register_forward_hook(make_hook(name, num_heads)))

    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def run_model_inference_with_grad(processor, image_path, prompt, device):
    from pam.sam3_loader import run_inference_with_grad

    return run_inference_with_grad(processor, image_path, prompt, device)


def compute_head_slot_importance(tensor, grad, num_heads):
    """
    Approximate head importance by splitting the attention module output
    feature dimension into num_heads chunks.

    Important caveat:
    This is a head-slot proxy after the attention module output, not a perfect
    pre-out-projection per-head tensor. It is still useful as a first gradient
    importance diagnostic.
    """
    if tensor.ndim != 3:
        raise RuntimeError(f"Expected tensor shape [B, T, C], got {tuple(tensor.shape)}")

    embed_dim = tensor.shape[-1]
    if embed_dim % num_heads != 0:
        raise RuntimeError(
            f"embed_dim={embed_dim} not divisible by num_heads={num_heads}"
        )

    head_dim = embed_dim // num_heads

    tensor_heads = tensor.reshape(tensor.shape[0], tensor.shape[1], num_heads, head_dim)
    grad_heads = grad.reshape(grad.shape[0], grad.shape[1], num_heads, head_dim)

    rows = []

    for h in range(num_heads):
        activation_h = tensor_heads[:, :, h, :]
        grad_h = grad_heads[:, :, h, :]

        mean_abs_grad = grad_h.abs().mean()
        max_abs_grad = grad_h.abs().max()
        grad_x_activation = (grad_h * activation_h).abs().mean()
        l2_grad = torch.linalg.norm(grad_h)

        rows.append(
            {
                "head_index": h,
                "mean_abs_grad": float(mean_abs_grad.detach().cpu()),
                "max_abs_grad": float(max_abs_grad.detach().cpu()),
                "grad_x_activation": float(grad_x_activation.detach().cpu()),
                "l2_grad": float(l2_grad.detach().cpu()),
            }
        )

    return rows


def save_head_importance(storage, out_dir, target_mask_index):
    out_dir = Path(out_dir)
    head_dir = out_dir / "head_importance"
    head_dir.mkdir(parents=True, exist_ok=True)

    head_rows = []
    layer_rows = []

    for module_name, item in storage.items():
        tensor = item["tensor"]
        grad = tensor.grad
        num_heads = item["num_heads"]

        if grad is None:
            raise RuntimeError(f"No gradient found for {module_name}")

        tensor_detached = tensor.detach().float()
        grad_detached = grad.detach().float()

        layer = module_name.split(".")[3]
        attention_module = module_name.split(".")[-1]

        per_head = compute_head_slot_importance(
            tensor=tensor_detached,
            grad=grad_detached,
            num_heads=num_heads,
        )

        total_mean_abs_grad = 0.0
        total_grad_x_activation = 0.0

        for row in per_head:
            total_mean_abs_grad += row["mean_abs_grad"]
            total_grad_x_activation += row["grad_x_activation"]

            head_rows.append(
                {
                    "layer": layer,
                    "attention_module": attention_module,
                    "module_name": module_name,
                    "head_index": row["head_index"],
                    "mean_abs_grad": row["mean_abs_grad"],
                    "max_abs_grad": row["max_abs_grad"],
                    "grad_x_activation": row["grad_x_activation"],
                    "l2_grad": row["l2_grad"],
                    "target_mask_index": target_mask_index,
                    "notes": "head-slot proxy from output feature chunks",
                }
            )

        layer_rows.append(
            {
                "layer": layer,
                "attention_module": attention_module,
                "module_name": module_name,
                "num_heads": num_heads,
                "importance_sum_mean_abs_grad": total_mean_abs_grad,
                "importance_sum_grad_x_activation": total_grad_x_activation,
                "target_mask_index": target_mask_index,
            }
        )

    # Normalize head importance globally.
    max_importance = max(r["grad_x_activation"] for r in head_rows) + 1e-12
    for r in head_rows:
        r["head_importance_normalized"] = r["grad_x_activation"] / max_importance

    head_csv = head_dir / "mm_head_importance.csv"
    layer_csv = head_dir / "mm_layer_importance.csv"
    top_json = head_dir / "top_heads.json"

    head_fields = [
        "layer",
        "attention_module",
        "module_name",
        "head_index",
        "mean_abs_grad",
        "max_abs_grad",
        "grad_x_activation",
        "l2_grad",
        "head_importance_normalized",
        "target_mask_index",
        "notes",
    ]

    with open(head_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=head_fields)
        writer.writeheader()
        writer.writerows(head_rows)

    layer_fields = [
        "layer",
        "attention_module",
        "module_name",
        "num_heads",
        "importance_sum_mean_abs_grad",
        "importance_sum_grad_x_activation",
        "target_mask_index",
    ]

    with open(layer_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=layer_fields)
        writer.writeheader()
        writer.writerows(layer_rows)

    sorted_heads = sorted(
        head_rows,
        key=lambda x: x["grad_x_activation"],
        reverse=True,
    )

    with open(top_json, "w") as f:
        json.dump(sorted_heads[:20], f, indent=2)

    print(f"Saved head importance CSV: {head_csv}")
    print(f"Saved layer importance CSV: {layer_csv}")
    print(f"Saved top heads JSON: {top_json}")

    print("\nTop 10 head-slot importances:")
    for row in sorted_heads[:10]:
        print(
            f"Layer {row['layer']} {row['attention_module']} "
            f"Head {row['head_index']} | "
            f"grad_x_activation={row['grad_x_activation']:.6e} | "
            f"normalized={row['head_importance_normalized']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", default="outputs/multimodal_head_importance")
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

    storage = {}
    hooks = register_attention_output_hooks(model, storage)

    try:
        print(f"Running prompt pass with gradients: {args.prompt!r}")
        state = run_model_inference_with_grad(
            processor,
            args.image,
            args.prompt,
            args.device,
        )
    finally:
        remove_hooks(hooks)

    print("Captured attention module outputs:")
    for name, item in storage.items():
        tensor = item["tensor"]
        print(
            f"  {name}: shape={tuple(tensor.shape)}, "
            f"requires_grad={tensor.requires_grad}, "
            f"num_heads={item['num_heads']}"
        )

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

    save_head_importance(
        storage=storage,
        out_dir=out_dir,
        target_mask_index=args.target_mask_index,
    )


if __name__ == "__main__":
    main()