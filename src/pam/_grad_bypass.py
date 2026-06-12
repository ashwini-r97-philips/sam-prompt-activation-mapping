"""Toggleable bypass for ``torch.inference_mode`` so SAM 3's hard-coded
``@torch.inference_mode()`` decorators can be neutralised for gradient
tracing.

Why this exists
---------------
SAM 3 decorates every public processor / predictor entry point with
``@torch.inference_mode()``. Inference-mode is *stricter* than
``torch.no_grad()``: tensors created inside it become "inference tensors"
that can never participate in autograd, even after the block exits.
Wrapping the call site in ``torch.enable_grad()`` is therefore not enough.

This module installs a drop-in subclass of ``torch.inference_mode`` whose
``__enter__`` becomes a no-op while a module-level flag is set. The
``trace_grads()`` context manager flips that flag and additionally enters
``torch.enable_grad()`` so any nested ``@torch.no_grad()`` regions are
also overridden for the duration of the block.

Import order matters
--------------------
This module must be imported **before** any ``sam3`` module is imported,
because the decorators are evaluated when SAM 3's classes are defined.
``pam/__init__.py`` imports it, so ``import pam`` (or any
``from pam.X import ...``) installs the patch in time.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.autograd
import torch.nn.functional as F

_orig_inference_mode = torch.inference_mode


class _ToggleableInferenceMode(_orig_inference_mode):
    """``torch.inference_mode`` subclass with a global kill-switch."""

    _disabled: bool = False

    def __enter__(self):
        if _ToggleableInferenceMode._disabled:
            self._noop = True
            return self
        self._noop = False
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if getattr(self, "_noop", False):
            return False
        return super().__exit__(exc_type, exc_val, exc_tb)


# Install BEFORE any ``import sam3...``. The reference is captured at
# class-definition time by ``@torch.inference_mode()`` in SAM 3 modules,
# so the patch must already be in place when those modules are imported.
torch.inference_mode = _ToggleableInferenceMode
torch.autograd.inference_mode = _ToggleableInferenceMode


@contextmanager
def trace_grads():
    """Inside this block, SAM 3's ``@torch.inference_mode()`` decorators
    behave as no-ops and ``torch.enable_grad()`` is in force.

    Also routes SAM 3's fused ``addmm_act`` op through a gradient-safe
    fallback (see :func:`ensure_fused_patch`).

    Usage::

        with trace_grads():
            state = processor.set_image(image)
            state = processor.set_text_prompt(state=state, prompt="dog")
            state["scores"].sum().backward()
    """
    ensure_fused_patch()
    prev = _ToggleableInferenceMode._disabled
    _ToggleableInferenceMode._disabled = True
    try:
        with torch.enable_grad():
            yield
    finally:
        _ToggleableInferenceMode._disabled = prev


def is_active() -> bool:
    """True iff a ``trace_grads()`` block is currently active."""
    return _ToggleableInferenceMode._disabled


# ---------------------------------------------------------------------------
# Patch 2: SAM 3's fused ``addmm_act`` (Linear+activation) op.
#
# ``sam3.perflib.fused.addmm_act`` calls ``torch.ops.aten._addmm_activation``
# which has no autograd backward registered, and it ``.detach()``s the
# Linear's weight & bias even in eval. The result: any forward pass with
# autograd enabled raises ``ValueError: Expected grad to be disabled.``,
# and even if you bypass that check no gradients would flow through the
# MLP params anyway.
#
# We wrap the original so that when ``trace_grads()`` is active, the call
# falls back to a mathematically-identical ``act(F.linear(x, W, b))`` path
# that preserves gradients through both inputs and parameters. When
# ``trace_grads()`` is not active, the original fused kernel is used.
#
# Must be applied *after* ``sam3.perflib.fused`` is importable, but the
# rebind happens lazily via ``ensure_fused_patch()`` which is called from
# ``trace_grads()``. This keeps import order flexible.
# ---------------------------------------------------------------------------

_fused_patched: bool = False


def _grad_safe_addmm_act(activation, linear, mat1):
    """Gradient-safe drop-in for ``sam3.perflib.fused.addmm_act``.

    Mirrors the original's bfloat16 cast for numerical parity, but uses
    ``F.linear`` + the activation module/function so autograd works.
    """
    weight = linear.weight
    bias = linear.bias
    # Match the original's bf16 compute dtype so numerics line up with
    # an inference forward pass.
    x = mat1.to(torch.bfloat16)
    w = weight.to(torch.bfloat16)
    b = bias.to(torch.bfloat16) if bias is not None else None
    y = F.linear(x, w, b)
    if activation in (torch.nn.functional.relu, torch.nn.ReLU):
        return F.relu(y)
    if activation in (torch.nn.functional.gelu, torch.nn.GELU):
        return F.gelu(y)
    raise ValueError(f"Unexpected activation {activation}")


def ensure_fused_patch() -> None:
    """Lazily monkey-patch ``sam3.perflib.fused.addmm_act`` so it routes
    through a grad-safe implementation when ``trace_grads()`` is active.

    Safe to call multiple times; only patches on the first call.
    """
    global _fused_patched
    if _fused_patched:
        return
    try:
        from sam3.perflib import fused as _fused_mod  # type: ignore[import-untyped]
    except ImportError:
        return  # sam3 not installed yet; nothing to patch

    _orig_addmm_act = _fused_mod.addmm_act

    def _dispatching_addmm_act(activation, linear, mat1):
        # Route to the grad-safe path whenever autograd is on. This
        # catches both the original forward inside ``trace_grads()`` and
        # the *checkpoint recompute* during backward, which re-enters
        # the block forward outside of our context manager.
        if _ToggleableInferenceMode._disabled or torch.is_grad_enabled():
            return _grad_safe_addmm_act(activation, linear, mat1)
        return _orig_addmm_act(activation, linear, mat1)

    _fused_mod.addmm_act = _dispatching_addmm_act

    # Also patch already-imported call sites that did
    # ``from sam3.perflib.fused import addmm_act``.
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.startswith("sam3.") and getattr(mod, "addmm_act", None) is _orig_addmm_act:
            setattr(mod, "addmm_act", _dispatching_addmm_act)

    _fused_patched = True


# ---------------------------------------------------------------------------
# Patch 3: enable per-block activation checkpointing in the ViT trunk.
#
# SAM 3's ``vitdet.ViT.forward`` gates ``checkpoint.checkpoint(blk, ...)``
# on ``self.use_act_checkpoint and self.training``. The pretrained model
# is loaded in eval mode, so checkpointing is off and every block keeps
# its activations for backward -- a 16 GB GPU OOMs on the first attention
# layer when autograd is enabled.
#
# We toggle ``self.training = True`` on *only* the ViT trunk modules
# (not their parents or children), so other modules whose behavior
# branches on ``self.training`` (decoder DAC, dynamic multimask, aux
# outputs) keep their eval semantics. Dropout / DropPath default to 0.0
# in the ViT-Det backbone so this introduces no stochasticity.
# ---------------------------------------------------------------------------


@contextmanager
def enable_backbone_act_checkpointing(model: torch.nn.Module):
    """Temporarily set ``training=True`` on the SAM 3 ViT trunk(s)
    inside *model* so per-block activation checkpointing fires.

    No-op if SAM 3 is not importable or no ``ViT`` instance is found.
    Restores prior flags on exit.
    """
    try:
        from sam3.model.vitdet import ViT as _ViT  # type: ignore[import-untyped]
    except ImportError:
        yield
        return

    touched: list[torch.nn.Module] = []
    for m in model.modules():
        if isinstance(m, _ViT) and not m.training:
            # Direct attribute write -- avoids ``.train()``'s recursion
            # which would flip every child too.
            m.training = True
            touched.append(m)
    try:
        yield
    finally:
        for m in touched:
            m.training = False
