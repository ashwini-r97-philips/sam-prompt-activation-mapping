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
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)
    return output
