from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


TARGET_LAYERS = [
    "transformer.encoder.layers.0",
    "transformer.encoder.layers.1",
    "transformer.encoder.layers.2",
    "transformer.encoder.layers.3",
    "transformer.encoder.layers.4",
    "transformer.encoder.layers.5",
]

TARGET_CROSS_ATTN = [
    "transformer.encoder.layers.0.cross_attn_image",
    "transformer.encoder.layers.1.cross_attn_image",
    "transformer.encoder.layers.2.cross_attn_image",
    "transformer.encoder.layers.3.cross_attn_image",
    "transformer.encoder.layers.4.cross_attn_image",
    "transformer.encoder.layers.5.cross_attn_image",
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


class SDPAMonkeyPatch:
    def __init__(self, target_modules):
        self.target_modules = set(target_modules)
        self.current_module = None
        self.records = []
        self.original_sdpa = None
        self.hooks = []

    def install_module_context_hooks(self, model):
        modules = dict(model.named_modules())

        for name in self.target_modules:
            if name not in modules:
                raise RuntimeError(f"Missing target attention module: {name}")

            mod = modules[name]

            def make_pre(n):
                def pre_hook(module, inputs):
                    self.current_module = n
                return pre_hook

            def post_hook(module, inputs, output):
                self.current_module = None

            self.hooks.append(mod.register_forward_pre_hook(make_pre(name)))
            self.hooks.append(mod.register_forward_hook(post_hook))

    def install_sdpa_patch(self):
        self.original_sdpa = F.scaled_dot_product_attention

        def wrapped_sdpa(query, key, value, *args, **kwargs):
            out = self.original_sdpa(query, key, value, *args, **kwargs)

            if self.current_module in self.target_modules:
                self.records.append(
                    {
                        "module_name": self.current_module,
                        "query": query.detach().float().cpu(),
                        "key": key.detach().float().cpu(),
                        "value": value.detach().float().cpu(),
                        "out_shape": tuple(out.shape),
                    }
                )

            return out

        F.scaled_dot_product_attention = wrapped_sdpa

    def remove(self):
        if self.original_sdpa is not None:
            F.scaled_dot_product_attention = self.original_sdpa
        for h in self.hooks:
            h.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-mask-index", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=16)
    args = parser.parse_args()

    from pam.sam3_loader import load_sam3_image_model, run_inference_with_grad

    out_dir = Path(args.out_dir) / "prompt_token_fusion_trace"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading SAM3 on {args.device}")
    model, processor = load_sam3_image_model(device=args.device)
    model.eval()

    prompt_storage = {}
    fusion_storage = {}

    restore_prompt = patch_encode_prompt(model, prompt_storage)
    fusion_hooks = register_fusion_hooks(model, fusion_storage)

    sdpa = SDPAMonkeyPatch(TARGET_CROSS_ATTN)
    sdpa.install_module_context_hooks(model)
    sdpa.install_sdpa_patch()

    try:
        print(f"Running prompt pass with gradients: {args.prompt!r}")
        state = run_inference_with_grad(processor, args.image, args.prompt, args.device)
    finally:
        sdpa.remove()
        restore_prompt()
        for h in fusion_hooks:
            h.remove()

    logits = state["masks_logits"]
    target = logits[0, args.target_mask_index].mean()

    print("masks_logits:", tuple(logits.shape), "requires_grad=", logits.requires_grad)
    print("Backward target:", float(target.detach().cpu()))

    model.zero_grad(set_to_none=True)
    target.backward()

    rows = []

    for layer in range(6):
        module_name = f"transformer.encoder.layers.{layer}.cross_attn_image"
        layer_name = f"transformer.encoder.layers.{layer}"

        matches = [r for r in sdpa.records if r["module_name"] == module_name]
        if not matches:
            raise RuntimeError(f"No SDPA record for {module_name}")

        rec = matches[0]
        q = rec["query"]  # [B,H,T,D]
        k = rec["key"]    # [B,H,S,D]

        if q.ndim != 4 or k.ndim != 4:
            raise RuntimeError(f"Expected q/k [B,H,T,D], got {q.shape}, {k.shape}")

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        attn = torch.softmax(scores, dim=-1)  # [B,H,T,S]
        attn = attn.mean(dim=1)[0]  # [T,S]

        num_image_tokens = attn.shape[0]
        side = int(num_image_tokens ** 0.5)
        if side * side != num_image_tokens:
            raise RuntimeError(f"Image token count is not square: {num_image_tokens}")

        fusion = fusion_storage[layer_name]
        fusion_grad = fusion.grad
        if fusion_grad is None:
            raise RuntimeError(f"No fusion grad for {layer_name}")

        grad_norm = torch.linalg.norm(fusion_grad.detach().float().cpu().squeeze(), dim=-1)
        if grad_norm.ndim != 1:
            raise RuntimeError(f"Expected grad_norm [T], got {grad_norm.shape}")

        prompt_tokens = min(attn.shape[1], args.max_prompt_tokens)

        for token_idx in range(prompt_tokens):
            token_attn = attn[:, token_idx].numpy()
            token_grad_weighted = token_attn * grad_norm.numpy()

            attn_map = token_attn.reshape(side, side)
            weighted_map = token_grad_weighted.reshape(side, side)

            attn_npy = out_dir / f"layer_{layer}_prompt_token_{token_idx}_attn.npy"
            weighted_npy = out_dir / f"layer_{layer}_prompt_token_{token_idx}_attn_x_grad.npy"

            np.save(attn_npy, attn_map)
            np.save(weighted_npy, weighted_map)

            attn_png = out_dir / f"layer_{layer}_prompt_token_{token_idx}_attn.png"
            weighted_png = out_dir / f"layer_{layer}_prompt_token_{token_idx}_attn_x_grad.png"

            save_map(attn_map, attn_png, f"Layer {layer} Prompt Token {token_idx} Attention")
            save_map(weighted_map, weighted_png, f"Layer {layer} Prompt Token {token_idx} Attention x Grad")

            rows.append(
                {
                    "layer": layer,
                    "prompt_token_index": token_idx,
                    "attention_mean": float(attn_map.mean()),
                    "attention_max": float(attn_map.max()),
                    "attn_x_grad_mean": float(weighted_map.mean()),
                    "attn_x_grad_max": float(weighted_map.max()),
                    "attention_npy": str(attn_npy),
                    "attention_png": str(attn_png),
                    "attn_x_grad_npy": str(weighted_npy),
                    "attn_x_grad_png": str(weighted_png),
                }
            )

    csv_path = out_dir / "prompt_token_fusion_trace.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved prompt-token fusion trace: {csv_path}")


if __name__ == "__main__":
    main()