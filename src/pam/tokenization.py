"""Prompt-token labelling utilities.

This module provides best-effort token labels for the text prompt.  It tries
to use SAM 3's actual tokeniser (accessed through the model's text encoder
backbone) and falls back to a simple whitespace split with ``[CLS]`` /
``[EOS]`` markers when the tokeniser is not accessible.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, List


def get_token_labels(
    model: Any,
    processor: Any,
    prompt: str,
    num_tokens: int | None = None,
) -> list[str]:
    """Return a list of human-readable token labels for *prompt*.

    Parameters
    ----------
    model : nn.Module
        The SAM 3 image model.
    processor : object
        The ``Sam3Processor`` instance.
    prompt : str
        The text prompt string.
    num_tokens : int | None
        If known, the number of token slots in the cross-attention tensor.
        Used to pad/trim the label list so it aligns with the heatmap tensor.

    Returns
    -------
    list[str]
        One label per token position.
    """
    labels: list[str] | None = None

    # Strategy 1: try to reach SAM 3's tokeniser.
    labels = _try_sam3_tokenizer(model, prompt)

    # Strategy 2: fallback — simple word split + special-token markers.
    if labels is None:
        labels = _fallback_labels(prompt)
        warnings.warn(
            "Could not access SAM 3 tokeniser. Token labels are approximate "
            "(whitespace-split).",
            stacklevel=2,
        )

    # Align with the actual tensor token dimension.
    if num_tokens is not None:
        labels = _align_labels(labels, num_tokens)

    return labels


# ---------------------------------------------------------------------------
# Internal strategies
# ---------------------------------------------------------------------------


def _try_sam3_tokenizer(model: Any, prompt: str) -> list[str] | None:
    """Attempt to tokenise *prompt* via SAM 3's text backbone.

    SAM 3 uses a CLIP-style ``TextTransformer`` that wraps a BPE tokeniser.
    The tokeniser is typically accessible as:
        model.backbone.language_backbone.tokenizer
    or
        model.backbone.text_encoder.tokenizer

    Returns ``None`` if the tokeniser cannot be found.
    """
    tokenizer = None
    # Walk a few plausible paths.
    for attr_chain in [
        ("backbone", "language_backbone", "tokenizer"),
        ("backbone", "text_encoder", "tokenizer"),
        ("backbone", "language_backbone", "tokenize"),
    ]:
        obj = model
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            tokenizer = obj
            break

    if tokenizer is None:
        return None

    try:
        if callable(tokenizer):
            # Some tokenisers accept a string and return token ids or tokens.
            result = tokenizer(prompt)
            if hasattr(result, "tokens"):
                return list(result.tokens())
            if isinstance(result, (list, tuple)):
                # Likely token-id tensors — not directly useful as labels.
                # Fall through.
                return None
    except Exception:
        pass

    return None


def _fallback_labels(prompt: str) -> list[str]:
    """Whitespace-split fallback.

    SAM 3's DETR encoder receives a concatenated prompt tensor:
        [text_tokens | geometry_tokens | visual_prompt_tokens]
    We label the text portion by whitespace-splitting the prompt string.
    Any remaining positions are labelled generically (``geo_00``, etc.)
    and will be padded/trimmed by ``_align_labels``.
    """
    words = prompt.split()
    return [f"txt_{w}" for w in words]


def _align_labels(labels: list[str], num_tokens: int) -> list[str]:
    """Pad or trim *labels* to exactly *num_tokens* entries.

    Extra positions beyond the text tokens are labelled ``prompt_NN``
    (these are typically geometry or visual-prompt tokens in SAM 3's
    concatenated prompt tensor).
    """
    if len(labels) == num_tokens:
        return labels
    if len(labels) < num_tokens:
        # Pad with generic positional labels for non-text prompt tokens.
        for i in range(len(labels), num_tokens):
            labels.append(f"prompt_{i:02d}")
    else:
        labels = labels[:num_tokens]
    return labels


# ---------------------------------------------------------------------------
# Filename utilities
# ---------------------------------------------------------------------------


def safe_filename(label: str) -> str:
    """Sanitise a token label for use in file paths.

    Replaces spaces, slashes, and other problematic characters with
    underscores, strips leading/trailing underscores, and collapses
    consecutive underscores.

    >>> safe_filename("[CLS]")
    'CLS'
    >>> safe_filename("yellow school bus")
    'yellow_school_bus'
    """
    s = re.sub(r"[^\w\-]", "_", label)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s or "unknown"
