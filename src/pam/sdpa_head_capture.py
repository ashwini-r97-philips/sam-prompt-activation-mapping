from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class CapturedSDPAHeadOutput:
    module_name: str
    call_index: int
    output: torch.Tensor
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@contextmanager
def capture_sdpa_head_outputs(model: torch.nn.Module, target_module_names: list[str]):
    """
    Capture true pre-output-projection SDPA head outputs.

    This monkey-patches torch.nn.functional.scaled_dot_product_attention and
    records the returned SDPA tensor while one of the target attention modules
    is executing.

    Expected captured tensor shape:

        [B, H, T, D]

    where:
        B = batch
        H = real attention heads
        T = query tokens
        D = per-head dimension

    This is intentionally different from the old head-slot proxy, which split
    the already mixed [B, T, C] output into chunks. This captures true heads
    before the attention module flattens heads and applies output projection.
    """
    modules = dict(model.named_modules())
    missing = [name for name in target_module_names if name not in modules]
    if missing:
        print("Missing target modules:")
        for name in missing:
            print("  ", name)

        print("\nAvailable modules containing transformer.encoder.layers:")
        for name in modules:
            if "transformer.encoder.layers" in name:
                print("  ", name)

        raise RuntimeError("Some target attention modules were not found.")

    captured: dict[str, list[CapturedSDPAHeadOutput]] = {
        name: [] for name in target_module_names
    }

    module_stack: list[str] = []
    hooks = []

    def make_pre_hook(module_name: str):
        def pre_hook(module: torch.nn.Module, inputs: tuple[Any, ...]):
            module_stack.append(module_name)

        return pre_hook

    def make_post_hook(module_name: str):
        def post_hook(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any):
            if module_stack and module_stack[-1] == module_name:
                module_stack.pop()

        return post_hook

    for name in target_module_names:
        module = modules[name]
        hooks.append(module.register_forward_pre_hook(make_pre_hook(name)))
        hooks.append(module.register_forward_hook(make_post_hook(name)))

    orig_sdpa = F.scaled_dot_product_attention

    def patched_sdpa(query, key, value, *args, **kwargs):
        out = orig_sdpa(query, key, value, *args, **kwargs)

        if module_stack:
            module_name = module_stack[-1]

            if module_name in captured:
                if not isinstance(out, torch.Tensor):
                    raise RuntimeError(
                        f"SDPA output for {module_name} is not a tensor: {type(out)}"
                    )

                if out.ndim != 4:
                    raise RuntimeError(
                        f"Expected true SDPA head output [B,H,T,D], "
                        f"got {tuple(out.shape)} for {module_name}"
                    )

                if out.requires_grad:
                    out.retain_grad()

                call_index = len(captured[module_name])

                captured[module_name].append(
                    CapturedSDPAHeadOutput(
                        module_name=module_name,
                        call_index=call_index,
                        output=out,
                        query_shape=tuple(query.shape),
                        key_shape=tuple(key.shape),
                        value_shape=tuple(value.shape),
                        output_shape=tuple(out.shape),
                    )
                )

        return out

    F.scaled_dot_product_attention = patched_sdpa

    try:
        yield captured
    finally:
        F.scaled_dot_product_attention = orig_sdpa

        for hook in hooks:
            hook.remove()