"""Attention module discovery and capture for SAM 3.

This module provides tools to:
1. Discover candidate attention modules inside a SAM 3 model.
2. Register forward hooks to capture Q, K, V tensors and (when possible)
   reconstruct raw attention weights.
3. Clean up hooks after inference.

Because SAM 3 uses fused scaled-dot-product attention internally (which does
NOT expose attention weight matrices), the primary strategy is to hook into
the ``multi_head_attention_forward`` call-site, intercept Q / K / V *after*
projection, and manually compute ``softmax(Q K^T / sqrt(d))``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dataclass for captured tensors
# ---------------------------------------------------------------------------


@dataclass
class CapturedAttention:
    """Container for tensors captured from a single attention module."""

    module_name: str
    raw_attention: torch.Tensor | None = None
    """Attention weights — shape ``[B, H, T_q, T_kv]``."""
    values: torch.Tensor | None = None
    """Value vectors — shape ``[B, H, T_kv, D_head]``."""
    query: torch.Tensor | None = None
    key: torch.Tensor | None = None
    input_shapes: list | None = None
    output_shapes: list | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Attention-module heuristics
# ---------------------------------------------------------------------------

# Attribute signatures that indicate "this is an attention module".
_ATTN_ATTRS = {
    "in_proj_weight",
    "q_proj_weight",
    "q_proj",
    "k_proj",
    "v_proj",
    "qkv",
    "num_heads",
    "n_heads",
    "embed_dim",
    "head_dim",
}

# Substrings in module *paths* that suggest cross-attention.
_CROSS_ATTN_HINTS = [
    "cross_attn",
    "ca_text",
    "cross_attend",
    "cross_attention",
]


def _is_attention_module(module: nn.Module) -> bool:
    """Heuristic: does *module* look like an attention layer?"""
    if isinstance(module, nn.MultiheadAttention):
        return True
    # Check for our attribute heuristic.
    attrs = {a for a in _ATTN_ATTRS if hasattr(module, a)}
    return len(attrs) >= 2


def _module_summary(name: str, module: nn.Module) -> str:
    """One-line summary of an attention module for printing."""
    cls = type(module).__qualname__
    extras: list[str] = []
    for attr in ("num_heads", "n_heads", "embed_dim", "head_dim"):
        if hasattr(module, attr):
            extras.append(f"{attr}={getattr(module, attr)}")
    extra_str = ", ".join(extras)
    return f"{name}  ({cls})  [{extra_str}]"


def discover_attention_modules(
    model: nn.Module,
    filter_text: str | None = None,
) -> list[tuple[str, nn.Module]]:
    """Walk *model* and return candidate attention modules.

    Parameters
    ----------
    model : nn.Module
        The full SAM 3 image model.
    filter_text : str | None
        If given, only return modules whose *name* contains this substring
        (case-insensitive).

    Returns
    -------
    list of (name, module) tuples.
    """
    candidates: list[tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if not _is_attention_module(mod):
            continue
        if filter_text is not None and filter_text.lower() not in name.lower():
            continue
        candidates.append((name, mod))
    return candidates


def select_attention_module(
    candidates: list[tuple[str, nn.Module]],
    module_name: str | None = None,
    layer_index: str = "first",
) -> tuple[str, nn.Module]:
    """Choose a single module from *candidates*.

    Parameters
    ----------
    candidates : list of (name, module)
        Output of :func:`discover_attention_modules`.
    module_name : str | None
        If given, select the candidate whose name matches exactly or contains
        this substring.
    layer_index : str
        ``"first"``, ``"last"``, or an integer index.

    Returns
    -------
    (name, module)

    Raises
    ------
    ValueError
        If no matching module is found.
    """
    if not candidates:
        raise ValueError(
            "No attention modules found. Run with --list-attention-modules "
            "to see available modules, or adjust --module-filter."
        )

    if module_name is not None:
        # Exact match first.
        for name, mod in candidates:
            if name == module_name:
                return name, mod
        # Substring match.
        matches = [(n, m) for n, m in candidates if module_name in n]
        if not matches:
            raise ValueError(
                f"No module matching '{module_name}'. "
                f"Candidates: {[n for n, _ in candidates]}"
            )
        candidates = matches

    idx: int
    if layer_index == "first":
        idx = 0
    elif layer_index == "last":
        idx = -1
    else:
        idx = int(layer_index)

    try:
        return candidates[idx]
    except IndexError:
        raise ValueError(
            f"layer_index={layer_index} out of range for {len(candidates)} "
            "candidate module(s)."
        )


# ---------------------------------------------------------------------------
# Hook-based attention capture
# ---------------------------------------------------------------------------


class AttentionCapture:
    """Register forward hooks to capture attention tensors from one module.

    Usage::

        cap = AttentionCapture(name, module)
        cap.register()
        # … run forward pass …
        result = cap.result()
        cap.remove()
    """

    def __init__(self, module_name: str, module: nn.Module) -> None:
        self.module_name = module_name
        self.module = module
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._captured_inputs: list[Any] = []
        self._captured_outputs: list[Any] = []
        self._warnings: list[str] = []

    # ----- hook callbacks ---------------------------------------------------

    def _hook_fn(
        self,
        mod: nn.Module,
        inputs: tuple,
        kwargs: dict,
        output: Any,
    ) -> None:
        """Forward-hook callback: store inputs and outputs.

        When registered with ``with_kwargs=True`` (PyTorch 2.0+) the
        signature is ``hook(module, args, kwargs, output)``.  We store
        both positional and keyword args so Q/K/V can be resolved from
        either.
        """
        self._captured_inputs.append((inputs, kwargs))
        self._captured_outputs.append(output)

    # ----- lifecycle --------------------------------------------------------

    def register(self) -> None:
        """Attach forward hooks to the target module."""
        h = self.module.register_forward_hook(self._hook_fn, with_kwargs=True)
        self._hooks.append(h)

    def remove(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ----- tensor reconstruction -------------------------------------------

    def result(self) -> CapturedAttention:
        """Attempt to reconstruct attention weights + value vectors.

        The strategy depends on what type of module was hooked.

        Returns
        -------
        CapturedAttention
        """
        if not self._captured_inputs:
            cap = CapturedAttention(module_name=self.module_name)
            cap.warnings.append("No inputs captured — was the module called?")
            return cap

        raw = self._captured_inputs[-1]  # last call — (positional, kwargs)
        output = self._captured_outputs[-1]

        # Unpack positional args and keyword args captured by the hook.
        if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
            pos_inputs, kw_inputs = raw
        else:
            # Fallback for hooks registered without with_kwargs
            pos_inputs, kw_inputs = raw, {}

        # Build a unified list of query, key, value from positional or kwargs.
        # SAM 3's custom MHA is called with keyword args only (query=, key=, value=).
        query = pos_inputs[0] if len(pos_inputs) > 0 else kw_inputs.get("query")
        key = pos_inputs[1] if len(pos_inputs) > 1 else kw_inputs.get("key")
        value = pos_inputs[2] if len(pos_inputs) > 2 else kw_inputs.get("value")

        # Synthetic inputs tuple for shape recording and downstream methods.
        inputs = tuple(
            x for x in (query, key, value) if x is not None
        ) + tuple(
            x for x in pos_inputs[3:] if True
        )

        # Record shapes for debugging.
        input_shapes = [
            list(t.shape) if isinstance(t, torch.Tensor) else str(type(t))
            for t in inputs
        ]
        output_shapes: list = []
        if isinstance(output, torch.Tensor):
            output_shapes = [list(output.shape)]
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    output_shapes.append(list(o.shape))
                else:
                    output_shapes.append(str(type(o)))

        # Dispatch to the appropriate reconstruction strategy.
        mod = self.module

        # ── Strategy A: nn.MultiheadAttention ──
        if isinstance(mod, nn.MultiheadAttention):
            return self._reconstruct_nn_mha(
                mod, inputs, output, input_shapes, output_shapes
            )

        # ── Strategy B: custom MultiheadAttention with in_proj_weight ──
        if hasattr(mod, "in_proj_weight") or hasattr(mod, "q_proj_weight"):
            return self._reconstruct_custom_mha(
                mod, inputs, output, input_shapes, output_shapes
            )

        # ── Strategy B′: modules with q_proj / k_proj / v_proj sub-modules ──
        if (
            hasattr(mod, "q_proj")
            and hasattr(mod, "k_proj")
            and hasattr(mod, "v_proj")
            and isinstance(getattr(mod, "q_proj"), nn.Linear)
        ):
            return self._reconstruct_qkv_submodule(
                mod, inputs, output, input_shapes, output_shapes
            )

        # ── Strategy C: fallback ──
        cap = CapturedAttention(
            module_name=self.module_name,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
        )
        cap.warnings.append(
            f"Module type {type(mod).__qualname__} is not recognised. "
            "Could not reconstruct attention weights. "
            "Run with --list-attention-modules and select a different "
            "--module-name."
        )
        return cap

    # ----- Strategy A: nn.MultiheadAttention --------------------------------

    def _reconstruct_nn_mha(
        self,
        mod: nn.MultiheadAttention,
        inputs: tuple,
        output: Any,
        input_shapes: list,
        output_shapes: list,
    ) -> CapturedAttention:
        """Reconstruct attention from ``nn.MultiheadAttention``.

        ``nn.MultiheadAttention.forward(query, key, value, …)``
        We project Q, K, V ourselves and compute softmax(QK^T / √d).
        """
        warns: list[str] = []
        query, key, value = inputs[0], inputs[1], inputs[2]

        num_heads: int = mod.num_heads
        embed_dim: int = mod.embed_dim
        head_dim: int = embed_dim // num_heads

        # Project Q, K, V using the module's weights.
        q, k, v = self._project_qkv_packed(
            mod, query, key, value, num_heads, head_dim, warns
        )
        if q is None:
            cap = CapturedAttention(
                module_name=self.module_name,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                warnings=warns,
            )
            return cap

        # Compute raw attention: softmax(Q K^T / √d_head)
        # q, k, v shapes: [B, H, T, D_head]
        raw_attn = self._compute_raw_attention(q, k, head_dim)
        return CapturedAttention(
            module_name=self.module_name,
            raw_attention=raw_attn,
            values=v,
            query=q,
            key=k,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            warnings=warns,
        )

    # ----- Strategy B: custom MHA with in_proj_weight / separate weights ----

    def _reconstruct_custom_mha(
        self,
        mod: nn.Module,
        inputs: tuple,
        output: Any,
        input_shapes: list,
        output_shapes: list,
    ) -> CapturedAttention:
        """Reconstruct attention from SAM 3's custom ``MultiheadAttention``.

        The custom class in ``sam3.model.model_misc`` mirrors PyTorch's MHA
        but routes through ``F.scaled_dot_product_attention`` which never
        returns weights.  We intercept the *inputs to the module* (which are
        the unprojected query/key/value) and project them using the stored
        weight matrices.
        """
        warns: list[str] = []

        # The custom MHA forward signature:
        #   forward(query, key, value, key_padding_mask=…, need_weights=…, …)
        # The first three positional args are query, key, value.
        query, key, value = inputs[0], inputs[1], inputs[2]

        num_heads: int = getattr(mod, "num_heads", 8)
        embed_dim: int = getattr(mod, "embed_dim", query.shape[-1])
        head_dim: int = getattr(mod, "head_dim", embed_dim // num_heads)

        q, k, v = self._project_qkv_packed(
            mod, query, key, value, num_heads, head_dim, warns
        )
        if q is None:
            return CapturedAttention(
                module_name=self.module_name,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                warnings=warns,
            )

        raw_attn = self._compute_raw_attention(q, k, head_dim)
        return CapturedAttention(
            module_name=self.module_name,
            raw_attention=raw_attn,
            values=v,
            query=q,
            key=k,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            warnings=warns,
        )

    # ----- Strategy B′: q_proj / k_proj / v_proj sub-modules ----------------

    def _reconstruct_qkv_submodule(
        self,
        mod: nn.Module,
        inputs: tuple,
        output: Any,
        input_shapes: list,
        output_shapes: list,
    ) -> CapturedAttention:
        """Reconstruct attention for modules with ``q_proj``, ``k_proj``,
        ``v_proj`` as ``nn.Linear`` children (e.g. SAM's ``Attention``)."""
        warns: list[str] = []

        # These modules typically receive (q, k, v) as positional args.
        if len(inputs) < 3:
            warns.append(
                "Expected ≥3 positional inputs (q, k, v) but got "
                f"{len(inputs)}.  Cannot reconstruct attention."
            )
            return CapturedAttention(
                module_name=self.module_name,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                warnings=warns,
            )

        q_in, k_in, v_in = inputs[0], inputs[1], inputs[2]
        num_heads: int = getattr(mod, "num_heads", 8)
        internal_dim: int = getattr(
            mod, "internal_dim", getattr(mod, "embed_dim", q_in.shape[-1])
        )
        head_dim: int = internal_dim // num_heads

        q = mod.q_proj(q_in)  # type: ignore[attr-defined]
        k = mod.k_proj(k_in)  # type: ignore[attr-defined]
        v = mod.v_proj(v_in)  # type: ignore[attr-defined]

        # Reshape to [B, H, T, D_head].
        # Input can be [T, B, D] (seq-first) or [B, T, D] (batch-first).
        q, k, v = self._reshape_to_heads(q, k, v, num_heads, head_dim, warns)
        if q is None:
            return CapturedAttention(
                module_name=self.module_name,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                warnings=warns,
            )

        raw_attn = self._compute_raw_attention(q, k, head_dim)
        return CapturedAttention(
            module_name=self.module_name,
            raw_attention=raw_attn,
            values=v,
            query=q,
            key=k,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            warnings=warns,
        )

    # ----- helpers ----------------------------------------------------------

    @staticmethod
    def _project_qkv_packed(
        mod: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        num_heads: int,
        head_dim: int,
        warns: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Project Q, K, V from either a packed ``in_proj_weight`` or
        separate ``q_proj_weight`` / ``k_proj_weight`` / ``v_proj_weight``.

        Returns (q, k, v) each with shape ``[B, H, T, D_head]``, or
        ``(None, None, None)`` if reconstruction fails.
        """
        embed_dim = num_heads * head_dim

        # Clone tensors to escape inference-mode restrictions so F.linear
        # can compute projections without autograd complaining.
        # Also cast to the weight dtype to avoid mixed-precision mismatches
        # when the model runs under autocast.
        weight_dtype = None
        if hasattr(mod, "in_proj_weight") and mod.in_proj_weight is not None:
            weight_dtype = mod.in_proj_weight.dtype
        elif hasattr(mod, "q_proj_weight") and mod.q_proj_weight is not None:
            weight_dtype = mod.q_proj_weight.dtype

        query = query.clone()
        key = key.clone()
        value = value.clone()
        if weight_dtype is not None:
            query = query.to(weight_dtype)
            key = key.to(weight_dtype)
            value = value.to(weight_dtype)

        # Determine whether inputs are sequence-first [T, B, D] or
        # batch-first [B, T, D].  SAM 3's custom MHA defaults to
        # sequence-first unless ``batch_first=True``.
        batch_first = getattr(mod, "batch_first", False)
        if not batch_first and query.dim() == 3:
            # Convert from [T, B, D] → [B, T, D] for uniform handling.
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Now all tensors are [B, T, D].
        bsz = query.shape[0]

        with torch.no_grad():
            if hasattr(mod, "in_proj_weight") and mod.in_proj_weight is not None:
                # Packed projection: in_proj_weight has shape [3*embed_dim, embed_dim].
                w = mod.in_proj_weight  # [3E, E]
                b = mod.in_proj_bias if hasattr(mod, "in_proj_bias") else None
                q = F.linear(query, w[:embed_dim], b[:embed_dim] if b is not None else None)
                k = F.linear(key, w[embed_dim : 2 * embed_dim], b[embed_dim : 2 * embed_dim] if b is not None else None)
                v = F.linear(value, w[2 * embed_dim :], b[2 * embed_dim :] if b is not None else None)
            elif hasattr(mod, "q_proj_weight") and mod.q_proj_weight is not None:
                b = mod.in_proj_bias if hasattr(mod, "in_proj_bias") else None
                q = F.linear(query, mod.q_proj_weight, b[:embed_dim] if b is not None else None)
                k = F.linear(key, mod.k_proj_weight, b[embed_dim : 2 * embed_dim] if b is not None else None)
                v = F.linear(value, mod.v_proj_weight, b[2 * embed_dim :] if b is not None else None)
            else:
                warns.append(
                    "Module has neither in_proj_weight nor q_proj_weight. "
                    "Cannot project Q/K/V."
                )
                return None, None, None

        # Reshape to [B, H, T, D_head].
        q = q.view(bsz, -1, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, -1, num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, -1, num_heads, head_dim).transpose(1, 2)
        return q, k, v

    @staticmethod
    def _reshape_to_heads(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        head_dim: int,
        warns: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Reshape projected tensors ``[B, T, D]`` → ``[B, H, T, D_head]``.

        Handles both batch-first ``[B, T, D]`` and sequence-first ``[T, B, D]``
        inputs by checking the last dimension.
        """
        if q.dim() == 3:
            # Heuristic: if last dim == num_heads * head_dim it's the feature
            # dimension; the remaining dim is the sequence length.
            if q.shape[-1] == num_heads * head_dim:
                bsz = q.shape[0]
            elif q.shape[0] * q.shape[1] and q.shape[-1] == num_heads * head_dim:
                bsz = q.shape[0]
            else:
                # Assume sequence-first [T, B, D].
                q = q.transpose(0, 1)
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)
                bsz = q.shape[0]

            q = q.reshape(bsz, -1, num_heads, head_dim).transpose(1, 2)
            k = k.reshape(bsz, -1, num_heads, head_dim).transpose(1, 2)
            v = v.reshape(bsz, -1, num_heads, head_dim).transpose(1, 2)
            return q, k, v

        warns.append(f"Unexpected Q tensor ndim={q.dim()}, expected 3.")
        return None, None, None

    @staticmethod
    def _compute_raw_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        head_dim: int,
    ) -> torch.Tensor:
        """Compute ``softmax(Q K^T / √d_head)`` → ``[B, H, T_q, T_kv]``.

        Parameters
        ----------
        q : Tensor  [B, H, T_q, D_head]
        k : Tensor  [B, H, T_kv, D_head]
        head_dim : int

        Returns
        -------
        Tensor  [B, H, T_q, T_kv]
        """
        scale = math.sqrt(head_dim)
        # scores shape: [B, H, T_q, T_kv]
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(scores, dim=-1)
        return attn
