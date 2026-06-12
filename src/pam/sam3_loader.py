"""SAM 3 model loading and inference utilities.

This module handles importing and initialising the SAM 3 image model and
processor. If SAM 3 is not installed the user gets a clear error with
installation instructions rather than a raw ImportError.
"""

from __future__ import annotations

# Ensure torch.inference_mode is patched before any sam3 symbol is imported.
from . import _grad_bypass  # noqa: F401

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Friendly import gate
# ---------------------------------------------------------------------------

_SAM3_INSTALL_MSG = """\
SAM 3 is not installed or not importable.

To install SAM 3, run:

    git clone https://github.com/facebookresearch/sam3.git external/sam3
    cd external/sam3
    pip install -e ".[notebooks]"

Then make sure the environment where SAM 3 is installed is the same one
used to run this tool.
"""


def _import_sam3():
    """Try to import the two SAM 3 entry-points we need."""
    try:
        from sam3.model_builder import build_sam3_image_model  # type: ignore[import-untyped]
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[import-untyped]
    except ImportError as exc:
        print(f"Import failed: {exc}\n", file=sys.stderr)
        print(_SAM3_INSTALL_MSG, file=sys.stderr)
        raise SystemExit(1)

    return build_sam3_image_model, Sam3Processor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_sam3_image_model(
    device: str = "cuda",
) -> tuple[Any, Any]:
    """Load the SAM 3 image model and return ``(model, processor)``."""
    build_fn, ProcessorCls = _import_sam3()

    model = build_fn()
    model.eval().to(device)

    processor = ProcessorCls(model)
    return model, processor


@torch.no_grad()
def run_inference(
    processor: Any,
    image_path: str | Path,
    prompt: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run a no-gradient forward pass on image_path with prompt.

    This is used for normal inference / visualization where gradients are
    not needed.
    """
    image = Image.open(image_path).convert("RGB")

    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=prompt)

    return state


def run_inference_with_grad(
    processor: Any,
    image_path: str | Path,
    prompt: str,
    device: str = "cuda",
    use_autocast: bool = True,
) -> Dict[str, Any]:
    """Run a gradient-enabled forward pass on image_path with prompt.

    Bypasses SAM 3's hard-coded ``@torch.inference_mode()`` decorators
    via :mod:`pam._grad_bypass`. Output tensors in the returned state
    (``masks_logits``, ``scores``, ``boxes``) will have ``requires_grad``
    propagated from any leaf in the graph (model parameters at minimum;
    pixel-level grads require constructing the input tensor manually --
    see notes below).

    Parameters
    ----------
    use_autocast : bool
        Whether to run under ``torch.autocast(bfloat16)``. Backward
        through autocast works; disable only if you need exact fp32
        gradients.

    Notes
    -----
    The default ``set_image`` path converts the image to ``uint8`` inside
    the processor's transform pipeline, breaking the gradient chain
    w.r.t. raw pixels. For input-attribution work, build the normalised
    ``1x3xRxR`` float tensor yourself, call ``requires_grad_(True)``,
    then call ``model.backbone.forward_image(tensor)`` directly and stash
    the result in ``state["backbone_out"]`` before invoking
    ``processor.set_text_prompt``.
    """
    from ._grad_bypass import enable_backbone_act_checkpointing, trace_grads

    image = Image.open(image_path).convert("RGB")

    if use_autocast:
        autocast_ctx = torch.autocast(
            device_type=device.split(":")[0], dtype=torch.bfloat16
        )
    else:
        autocast_ctx = nullcontext()

    with trace_grads(), enable_backbone_act_checkpointing(processor.model), autocast_ctx:
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=prompt)

    return state