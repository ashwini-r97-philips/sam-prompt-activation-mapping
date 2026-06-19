from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


TARGET_ATTENTION_MODULES = []

for layer in range(6):
    TARGET_ATTENTION_MODULES.append(
        f"transformer.encoder.layers.{layer}.self_attn"
    )
    TARGET_ATTENTION_MODULES.append(
        f"transformer.encoder.layers.{layer}.cross_attn_image"
    )


def run_model_inference_with_grad(processor, image_path, prompt, device):
    from pam.sam3_loader import run_inference_with_grad

    return run_inference_with_grad(processor, image_path, prompt, device)


def parse_layer_and_type(module_name: str):
    parts = module_name.split(".")
    layer = parts[3]
    attention_type = parts[-1]
    return layer, attention_type


def compute_true_head_importance(head_output: torch.Tensor, grad: torch.Tensor):
    """
    Compute true per-head importance from real SDPA output.

    Expected shape:

        head_output: [B, H, T, D]
        grad       : [B, H, T, D]

    No reshaping from [B,T,C] is allowed here.
    """
    if head_output.ndim != 4:
        raise RuntimeError(
            f"Expected head_output [B,H,T,D], got {tuple(head_output.shape)}"
        )

    if grad.ndim != 4:
        raise RuntimeError(f"Expected grad [B,H,T,D], got {tuple(grad.shape)}")

    if tuple(head_output.shape) != tuple(grad.shape):
        raise RuntimeError(
            f"Activation/gradient shape mismatch: "
            f"{tuple(head_output.shape)} vs {tuple(grad.shape)}"
        )

    num_heads = head_output.shape[1]
    rows = []

    for h in range(num_heads):
        activation_h = head_output[:, h, :, :]
        grad_h = grad[:, h, :, :]

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


def save_true_head_importance(captured, out_dir, target_mask_index):
    out_dir = Path(out_dir)
    head_dir = out_dir / "true_head_importance"
    head_dir.mkdir(parents=True, exist_ok=True)

    head_rows = []
    layer_rows = []
    debug_rows = []

    for module_name, captures in captured.items():
        if not captures:
            print(f"WARNING: no SDPA calls captured for {module_name}")
            continue

        # Usually each attention module should call SDPA once.
        # If multiple calls happen, keep all and record call_index.
        for cap in captures:
            tensor = cap.output
            grad = tensor.grad

            if grad is None:
                raise RuntimeError(
                    f"No gradient found for true SDPA output: "
                    f"{module_name}, call_index={cap.call_index}"
                )

            if tensor.ndim != 4:
                raise RuntimeError(
                    f"Acceptance condition failed: expected [B,H,T,D], "
                    f"got {tuple(tensor.shape)} for {module_name}"
                )

            layer, attention_type = parse_layer_and_type(module_name)

            tensor_detached = tensor.detach().float()
            grad_detached = grad.detach().float()

            per_head = compute_true_head_importance(
                head_output=tensor_detached,
                grad=grad_detached,
            )

            total_mean_abs_grad = 0.0
            total_grad_x_activation = 0.0

            for row in per_head:
                total_mean_abs_grad += row["mean_abs_grad"]
                total_grad_x_activation += row["grad_x_activation"]

                head_rows.append(
                    {
                        "layer": layer,
                        "attention_module": attention_type,
                        "module_name": module_name,
                        "call_index": cap.call_index,
                        "head_index": row["head_index"],
                        "mean_abs_grad": row["mean_abs_grad"],
                        "max_abs_grad": row["max_abs_grad"],
                        "grad_x_activation": row["grad_x_activation"],
                        "l2_grad": row["l2_grad"],
                        "target_mask_index": target_mask_index,
                        "notes": "true pre-out-projection SDPA head output [B,H,T,D]",
                    }
                )

            layer_rows.append(
                {
                    "layer": layer,
                    "attention_module": attention_type,
                    "module_name": module_name,
                    "call_index": cap.call_index,
                    "num_heads": tensor.shape[1],
                    "tokens": tensor.shape[2],
                    "head_dim": tensor.shape[3],
                    "importance_sum_mean_abs_grad": total_mean_abs_grad,
                    "importance_sum_grad_x_activation": total_grad_x_activation,
                    "target_mask_index": target_mask_index,
                }
            )

            debug_rows.append(
                {
                    "layer": layer,
                    "attention_module": attention_type,
                    "module_name": module_name,
                    "call_index": cap.call_index,
                    "query_shape": str(cap.query_shape),
                    "key_shape": str(cap.key_shape),
                    "value_shape": str(cap.value_shape),
                    "sdpa_output_shape": str(cap.output_shape),
                    "target_mask_index": target_mask_index,
                }
            )

    if not head_rows:
        raise RuntimeError(
            "No true SDPA head outputs were captured. "
            "This likely means the target modules did not call "
            "torch.nn.functional.scaled_dot_product_attention."
        )

    max_importance = max(r["grad_x_activation"] for r in head_rows) + 1e-12
    for row in head_rows:
        row["head_importance_normalized"] = (
            row["grad_x_activation"] / max_importance
        )

    sorted_heads = sorted(
        head_rows,
        key=lambda x: x["grad_x_activation"],
        reverse=True,
    )

    head_csv = head_dir / "mm_true_head_importance.csv"
    layer_csv = head_dir / "mm_true_layer_importance.csv"
    debug_csv = head_dir / "mm_true_head_capture_debug.csv"
    top_json = head_dir / "top_true_heads.json"

    head_fields = [
        "layer",
        "attention_module",
        "module_name",
        "call_index",
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
        "call_index",
        "num_heads",
        "tokens",
        "head_dim",
        "importance_sum_mean_abs_grad",
        "importance_sum_grad_x_activation",
        "target_mask_index",
    ]

    with open(layer_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=layer_fields)
        writer.writeheader()
        writer.writerows(layer_rows)

    debug_fields = [
        "layer",
        "attention_module",
        "module_name",
        "call_index",
        "query_shape",
        "key_shape",
        "value_shape",
        "sdpa_output_shape",
        "target_mask_index",
    ]

    with open(debug_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=debug_fields)
        writer.writeheader()
        writer.writerows(debug_rows)

    with open(top_json, "w") as f:
        json.dump(sorted_heads[:20], f, indent=2)

    print(f"Saved true head importance CSV: {head_csv}")
    print(f"Saved true layer importance CSV: {layer_csv}")
    print(f"Saved true head debug CSV: {debug_csv}")
    print(f"Saved top true heads JSON: {top_json}")

    print("\nTop 10 true head importances:")
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
    parser.add_argument("--out-dir", default="outputs/multimodal_true_head_importance")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mask-index", type=int, default=0)
    args = parser.parse_args()

    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    from pam.sam3_loader import load_sam3_image_model
    from pam.sdpa_head_capture import capture_sdpa_head_outputs

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading SAM3 on {args.device}")
    model, processor = load_sam3_image_model(device=args.device)
    model.eval()

    print("Capturing true SDPA head outputs from:")
    for name in TARGET_ATTENTION_MODULES:
        print(f"  {name}")

    with capture_sdpa_head_outputs(model, TARGET_ATTENTION_MODULES) as captured:
        print(f"Running prompt pass with gradients: {args.prompt!r}")
        state = run_model_inference_with_grad(
            processor,
            args.image,
            args.prompt,
            args.device,
        )

        print("Captured SDPA calls:")
        for name, calls in captured.items():
            if not calls:
                print(f"  {name}: NO CALLS")
            for cap in calls:
                print(
                    f"  {name} call {cap.call_index}: "
                    f"Q={cap.query_shape} K={cap.key_shape} "
                    f"V={cap.value_shape} OUT={cap.output_shape} "
                    f"requires_grad={cap.output.requires_grad}"
                )

        masks_logits = state["masks_logits"]

        if masks_logits.ndim != 4:
            raise RuntimeError(
                f"Expected masks_logits shape [B,N,H,W], "
                f"got {tuple(masks_logits.shape)}"
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

        save_true_head_importance(
            captured=captured,
            out_dir=out_dir,
            target_mask_index=args.target_mask_index,
        )


if __name__ == "__main__":
    main()