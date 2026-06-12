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


def shape_of(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    return None


def first_tensor_from_output(output):
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
    return None


def classify_length(length):
    if length is None:
        return "unknown"
    if length == 5184:
        return "visual_grid_72x72"
    if length < 128:
        return "prompt_or_text_tokens"
    return "unknown"


def get_qkv_from_inputs(inputs, kwargs):
    """
    Works for torch.nn.MultiheadAttention-style calls.

    Usually:
    query = inputs[0]
    key   = inputs[1]
    value = inputs[2]

    Sometimes these may be in kwargs.
    """
    q = None
    k = None
    v = None

    if len(inputs) >= 1:
        q = inputs[0]
    if len(inputs) >= 2:
        k = inputs[1]
    if len(inputs) >= 3:
        v = inputs[2]

    if q is None:
        q = kwargs.get("query")
    if k is None:
        k = kwargs.get("key")
    if v is None:
        v = kwargs.get("value")

    return q, k, v


def infer_sequence_length(tensor):
    """
    Infer token length from common attention tensor layouts.

    For SAM3 MultiheadAttention in your earlier output:
    layer output was [1, 5184, 256], so batch_first likely.
    """
    if not isinstance(tensor, torch.Tensor):
        return None

    shape = tuple(tensor.shape)

    if len(shape) == 3:
        # Common batch-first: [B, T, C]
        if shape[0] == 1:
            return shape[1]
        # Common non-batch-first: [T, B, C]
        if shape[1] == 1:
            return shape[0]
        # Fallback: assume middle dimension is tokens if last is feature dim.
        return shape[1]

    if len(shape) == 2:
        return shape[0]

    return None


def register_qkv_hooks(model, rows):
    hooks = []
    modules = dict(model.named_modules())

    missing = [name for name in TARGET_ATTENTION_MODULES if name not in modules]
    if missing:
        print("Missing attention modules:")
        for name in missing:
            print("  ", name)

        print("\nAvailable modules containing transformer.encoder.layers:")
        for name in modules:
            if "transformer.encoder.layers" in name:
                print("  ", name)

        raise RuntimeError("Some target attention modules were not found.")

    for name in TARGET_ATTENTION_MODULES:
        module = modules[name]

        def make_hook(module_name):
            def hook(mod, inputs, kwargs, output):
                q, k, v = get_qkv_from_inputs(inputs, kwargs)
                out = first_tensor_from_output(output)

                q_len = infer_sequence_length(q)
                k_len = infer_sequence_length(k)
                v_len = infer_sequence_length(v)
                out_len = infer_sequence_length(out)

                updated_stream_guess = classify_length(q_len)
                kv_source_guess = classify_length(k_len)

                layer = module_name.split(".")[3]
                attention_type = module_name.split(".")[-1]

                rows.append(
                    {
                        "layer": layer,
                        "module_name": module_name,
                        "attention_type": attention_type,
                        "q_shape": str(shape_of(q)),
                        "k_shape": str(shape_of(k)),
                        "v_shape": str(shape_of(v)),
                        "output_shape": str(shape_of(out)),
                        "q_length": q_len,
                        "k_length": k_len,
                        "v_length": v_len,
                        "output_length": out_len,
                        "updated_stream_guess": updated_stream_guess,
                        "kv_source_guess": kv_source_guess,
                    }
                )

            return hook

        hooks.append(module.register_forward_hook(make_hook(name), with_kwargs=True))

    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def run_model_inference(processor, image_path, prompt, device):
    from pam.sam3_loader import run_inference

    return run_inference(processor, image_path, prompt, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", default="outputs/qkv_trace")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not Path(args.image).is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    from pam.sam3_loader import load_sam3_image_model

    out_dir = Path(args.out_dir)
    attn_dir = out_dir / "mm_attention"
    attn_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Loading SAM3 on {args.device}")
    model, processor = load_sam3_image_model(device=args.device)
    model.eval()

    rows = []
    hooks = register_qkv_hooks(model, rows)

    try:
        print(f"Running inference: image={args.image}, prompt={args.prompt!r}")
        run_model_inference(processor, args.image, args.prompt, args.device)
    finally:
        remove_hooks(hooks)

    csv_path = attn_dir / "qkv_trace.csv"

    fieldnames = [
        "layer",
        "module_name",
        "attention_type",
        "q_shape",
        "k_shape",
        "v_shape",
        "output_shape",
        "q_length",
        "k_length",
        "v_length",
        "output_length",
        "updated_stream_guess",
        "kv_source_guess",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved Q/K/V trace to: {csv_path}")

    print("\nSummary:")
    for row in rows:
        print(
            f"Layer {row['layer']} {row['attention_type']}: "
            f"Q={row['q_shape']} K={row['k_shape']} V={row['v_shape']} "
            f"OUT={row['output_shape']} | "
            f"updates={row['updated_stream_guess']} reads={row['kv_source_guess']}"
        )


if __name__ == "__main__":
    main()