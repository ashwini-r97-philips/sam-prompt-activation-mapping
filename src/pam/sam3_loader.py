"""SAM 3 model loading and inference utilities.

This module handles importing and initialising the SAM 3 image model and
processor.  If SAM 3 is not installed the user gets a clear error with
installation instructions rather than a raw ImportError.
"""

from __future__ import annotations

import sys
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
    """Try to import the two SAM 3 entry-points we need.

    Returns
    -------
    build_fn : callable
        ``sam3.model_builder.build_sam3_image_model``
    processor_cls : type
        ``sam3.model.sam3_image_processor.Sam3Processor``

    Raises
    ------
    SystemExit
        If SAM 3 cannot be imported.
    """
    try:
        from sam3.model_builder import build_sam3_image_model  # type: ignore[import-untyped]
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore[import-untyped]
    except ImportError as exc:
        # Show the real error so missing transitive deps are visible.
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
    """Load the SAM 3 image model and return ``(model, processor)``.

    Parameters
    ----------
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    model : torch.nn.Module
        The SAM 3 image model in eval mode on *device*.
    processor : Sam3Processor
        The SAM 3 image processor wrapping the model.
    """
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
    """Run a single no-gradient forward pass on *image_path* with *prompt*.

    Parameters
    ----------
    processor : Sam3Processor
        The processor returned by :func:`load_sam3_image_model`.
    image_path : str | Path
        Path to the input image.
    prompt : str
        Text prompt, e.g. ``"yellow school bus"``.
    device : str
        Device the model lives on (informational – the processor handles
        placement internally).

    Returns
    -------
    output : dict
        The raw output dictionary from ``processor.set_text_prompt()``.
    """
    image = Image.open(image_path).convert("RGB")
    with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
        state = processor.set_image(image)
        state = processor.set_text_prompt(state=state, prompt=prompt)
    return state


# ---------------------------------------------------------------------------
# Grad-enabled inference (bypasses processor's @torch.inference_mode)
# ---------------------------------------------------------------------------


def run_inference_with_grad(
    model: Any,
    processor: Any,
    image_path: str | Path,
    prompt: str,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run a forward pass *with gradients enabled* for attribution flow.

    The ``Sam3Processor`` wraps every method in ``@torch.inference_mode()``,
    which prevents gradient computation.  This function replicates the
    processor's logic by calling the model's methods directly — none of
    which have ``@torch.inference_mode()``.

    Parameters
    ----------
    model : torch.nn.Module
        The SAM 3 image model (already on *device*).
    processor : Sam3Processor
        Used only for its ``transform`` and ``find_stage``.
    image_path : str | Path
        Path to the input image.
    prompt : str
        Text prompt.
    device : str
        Device string (e.g. ``"cuda"``).
    confidence_threshold : float
        Score threshold for keeping detections.

    Returns
    -------
    dict
        Contains ``pred_logits``, ``pred_boxes``, ``pred_masks``,
        ``presence_logit_dec``, ``boxes``, ``masks``, ``scores``,
        ``_kept_indices`` (maps filtered mask index → raw query index),
        and ``_raw_outputs`` (the full unfiltered model output dict).
    """
    from torchvision.transforms import v2

    pil_image = Image.open(image_path).convert("RGB")
    original_w, original_h = pil_image.size

    # -- image preprocessing (same as processor.set_image) -------------------
    image_tensor = v2.functional.to_image(pil_image).to(device)
    image_tensor = processor.transform(image_tensor).unsqueeze(0)

    # -- run backbone (no inference_mode on these methods) --------------------
    device_type = device.split(":")[0]
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        # Backbone can run without grad — its outputs become regular tensors
        # that *can* receive grad when used downstream.
        with torch.no_grad():
            backbone_out = model.backbone.forward_image(image_tensor)
            text_outputs = model.backbone.forward_text([prompt], device=device)
            backbone_out.update(text_outputs)

        # Clone backbone outputs so they are not inference tensors and can
        # participate in the autograd graph from here on.
        _ensure_regular_tensors(backbone_out)

        geometric_prompt = model._get_dummy_prompt()

        # -- run encoder + decoder with grad enabled -------------------------
        outputs = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=processor.find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

    # -- post-process (replicate processor._forward_grounding) ---------------
    pred_logits = outputs["pred_logits"]  # [B, N, 1]
    pred_boxes = outputs["pred_boxes"]  # [B, N, 4]
    pred_masks = outputs.get("pred_masks")  # [B, N, H, W] or None
    presence_logit = outputs["presence_logit_dec"]  # [B, 1] or [B]

    # Compute combined score (same as processor).
    out_probs = pred_logits.sigmoid()
    presence_score = presence_logit.sigmoid()
    if presence_score.dim() == 1:
        presence_score = presence_score.unsqueeze(1)
    combined_probs = (out_probs * presence_score).squeeze(-1)  # [B, N]

    # Confidence filter — keep mask→query index mapping.
    keep = combined_probs[0] > confidence_threshold  # [N]
    kept_indices = torch.where(keep)[0]  # raw query indices

    from sam3.model import box_ops  # type: ignore[import-untyped]
    from sam3.model.data_misc import interpolate  # type: ignore[import-untyped]

    filtered_probs = combined_probs[0, keep]
    filtered_boxes = pred_boxes[0, keep]
    filtered_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(filtered_boxes)

    scale_fct = torch.tensor(
        [original_w, original_h, original_w, original_h],
        device=device, dtype=filtered_boxes_xyxy.dtype,
    )
    filtered_boxes_xyxy = filtered_boxes_xyxy * scale_fct[None, :]

    filtered_masks = None
    if pred_masks is not None:
        fm = pred_masks[0, keep].unsqueeze(1)  # [K, 1, H, W]
        fm = interpolate(fm, (original_h, original_w), mode="bilinear", align_corners=False)
        filtered_masks = (fm.sigmoid() > 0.5)

    return {
        # Raw model outputs (with grad).
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "pred_masks": pred_masks,
        "presence_logit_dec": presence_logit,
        # Filtered outputs.
        "boxes": filtered_boxes_xyxy,
        "masks": filtered_masks,
        "scores": filtered_probs,
        "_kept_indices": kept_indices,
        "_raw_outputs": outputs,
        "_original_size": (original_h, original_w),
    }


def _ensure_regular_tensors(d: dict) -> None:
    """Recursively clone tensors in *d* so they are not inference-mode tensors.

    Tensors produced under ``torch.no_grad()`` are regular tensors (not
    inference tensors), so this is mainly a safety net.  It does NOT enable
    ``requires_grad`` — that happens naturally when these tensors flow through
    grad-enabled operations downstream.
    """
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            if v.is_inference():
                d[k] = v.clone()
        elif isinstance(v, dict):
            _ensure_regular_tensors(v)
