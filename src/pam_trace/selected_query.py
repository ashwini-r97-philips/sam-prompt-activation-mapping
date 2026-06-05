"""Selected-query tracing for SAM 3.

Given a single image + text prompt, this module captures the *raw* per-query
outputs (mask logits before sigmoid/resizing, boxes, scores, presence) plus
per-layer detector-decoder activations for the single most-confident query
``q`` and writes:

    pam_trace/selected_query.json
    pam_trace/selected_query_mask.npy
    pam_trace/selected_query_text_attention.csv

Implementation is **non-invasive**: we install forward hooks on detector
decoder layers and ca_text modules, and we monkey-patch
``model.forward_grounding`` for the duration of a single inference call to
capture its raw output dict. No SAM 3 source files are modified.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


# Detector decoder layer paths look like ``transformer.decoder.layers.<i>``.
_DEC_LAYER_RE = re.compile(r"^transformer\.decoder\.layers\.(\d+)$")
_DEC_CATEXT_RE = re.compile(r"^transformer\.decoder\.layers\.(\d+)\.ca_text$")
_DEC_CROSSATTN_RE = re.compile(r"^transformer\.decoder\.layers\.(\d+)\.cross_attn$")


class SelectedQueryCollector:
    """Capture per-layer hidden states, ca_text attention, image cross-attn,
    and the raw ``forward_grounding`` output dict for a single inference call.

    Parameters
    ----------
    model
        The SAM 3 image model.
    force_weights
        If True, monkey-patch the detector decoder ``cross_attn`` modules so
        they return per-head image attention weights even when the caller
        passed ``need_weights=False``. The ``attn_output`` tensor (first
        element of the returned tuple) is unchanged, so model outputs are
        preserved -- this is an analysis-only switch.
    """

    def __init__(self, model: nn.Module, force_weights: bool = False):
        self.model = model
        self.force_weights = force_weights

        # layer_index -> cloned CPU tensor (last-recorded output)
        self.layer_hidden_states: Dict[int, torch.Tensor] = {}
        self.layer_catext_attn: Dict[int, torch.Tensor] = {}
        self.layer_image_attn: Dict[int, torch.Tensor] = {}

        # raw outputs from the most recent forward_grounding call
        self.forward_grounding_out: Optional[Dict[str, Any]] = None

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._original_forward_grounding = None
        # for force_weights restore
        self._restore_force = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self) -> int:
        if self._handles or self._original_forward_grounding is not None:
            raise RuntimeError("SelectedQueryCollector already registered.")

        # Optional: force decoder cross_attn (image) modules to return weights.
        if self.force_weights:
            from .attention_tracer import force_attention_weights
            cross_attn_mods = [
                (path, mod)
                for path, mod in self.model.named_modules()
                if _DEC_CROSSATTN_RE.match(path) is not None
            ]
            self._restore_force = force_attention_weights(cross_attn_mods)

        n = 0
        for path, mod in self.model.named_modules():
            m_layer = _DEC_LAYER_RE.match(path)
            if m_layer:
                idx = int(m_layer.group(1))
                self._handles.append(
                    mod.register_forward_hook(self._make_layer_hook(idx))
                )
                n += 1
                continue
            m_ca = _DEC_CATEXT_RE.match(path)
            if m_ca:
                idx = int(m_ca.group(1))
                self._handles.append(
                    mod.register_forward_hook(self._make_catext_hook(idx))
                )
                n += 1
                continue
            m_xa = _DEC_CROSSATTN_RE.match(path)
            if m_xa:
                idx = int(m_xa.group(1))
                self._handles.append(
                    mod.register_forward_hook(self._make_image_attn_hook(idx))
                )
                n += 1
                continue

        # Wrap forward_grounding to capture its output dict.
        if hasattr(self.model, "forward_grounding"):
            self._original_forward_grounding = self.model.forward_grounding

            def _wrapper(*args, _self=self, _orig=self._original_forward_grounding, **kwargs):
                out = _orig(*args, **kwargs)
                _self.forward_grounding_out = out
                return out

            # Bind on the instance so we don't touch the class.
            self.model.forward_grounding = _wrapper  # type: ignore[assignment]

        return n

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._original_forward_grounding is not None:
            try:
                self.model.forward_grounding = self._original_forward_grounding  # type: ignore[assignment]
            except Exception:  # noqa: BLE001
                pass
            self._original_forward_grounding = None
        if self._restore_force is not None:
            try:
                self._restore_force()
            except Exception:  # noqa: BLE001
                pass
            self._restore_force = None

    def __enter__(self) -> "SelectedQueryCollector":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_layer_hook(self, idx: int):
        def hook(_module, _inputs, output):
            # ``TransformerDecoderLayer.forward`` returns ``tgt`` of shape
            # ``[Q, B, C]`` (sometimes a tuple ``(tgt, presence_token_out)``).
            t = output
            if isinstance(t, tuple):
                t = t[0]
            if isinstance(t, torch.Tensor):
                self.layer_hidden_states[idx] = t.detach().to("cpu", copy=True)
            return None
        return hook

    def _make_catext_hook(self, idx: int):
        def hook(_module, _inputs, output):
            # ca_text is a MultiheadAttentionWrapper: returns
            # (attn_output [Q+1, B, C], attn_weights [B, Q+1, T])
            if isinstance(output, tuple) and len(output) >= 2:
                attn_w = output[1]
                if isinstance(attn_w, torch.Tensor):
                    self.layer_catext_attn[idx] = attn_w.detach().to("cpu", copy=True)
            return None
        return hook

    def _make_image_attn_hook(self, idx: int):
        def hook(_module, _inputs, output):
            # decoder.layers.{i}.cross_attn (image): returns
            # (attn_output [Q, B, C], attn_weights [B, Q, S_img]) when
            # ``need_weights=True`` -- otherwise the weights tensor is None.
            if isinstance(output, tuple) and len(output) >= 2:
                attn_w = output[1]
                if isinstance(attn_w, torch.Tensor):
                    self.layer_image_attn[idx] = attn_w.detach().to("cpu", copy=True)
            return None
        return hook


# ---------------------------------------------------------------------------
# Selected-query selection and writing
# ---------------------------------------------------------------------------


def _per_query_score(out: Dict[str, Any]) -> torch.Tensor:
    """Replicate Sam3Processor scoring: ``sigmoid(pred_logits) * sigmoid(presence)``.

    Returns a 1-D tensor of length Q (the per-query final score for batch 0).
    """
    pred_logits = out["pred_logits"]            # [B, Q, 1]
    presence = out["presence_logit_dec"]        # [B, 1] or [B, 1, 1] depending on build
    probs = pred_logits.sigmoid()
    presence_score = presence.sigmoid()
    while presence_score.dim() < probs.dim():
        presence_score = presence_score.unsqueeze(1)
    combined = (probs * presence_score).squeeze(-1)  # [B, Q]
    return combined[0].detach().float().cpu()


def select_query(out: Dict[str, Any]) -> int:
    """Return the most-confident query index (used for the displayed result)."""
    scores = _per_query_score(out)
    q = int(torch.argmax(scores).item())
    return q


def write_selected_query_outputs(
    collector: SelectedQueryCollector,
    out_dir: str | Path,
    image_path: Optional[str] = None,
    prompt: Optional[str] = None,
    image_hw: Optional[Tuple[int, int]] = None,
    confidence_threshold: Optional[float] = None,
) -> Dict[str, Path]:
    """Write the 3 selected-query files. Returns mapping of file -> path."""
    if collector.forward_grounding_out is None:
        raise RuntimeError(
            "No forward_grounding output captured. Did inference run while "
            "the SelectedQueryCollector was registered?"
        )
    out = collector.forward_grounding_out

    pred_masks: torch.Tensor = out["pred_masks"]                 # [B, Q, H, W] raw logits
    pred_boxes: torch.Tensor = out["pred_boxes"]                 # [B, Q, 4] cxcywh in [0,1]
    pred_boxes_xyxy: torch.Tensor = out.get("pred_boxes_xyxy", None)
    pred_logits: torch.Tensor = out["pred_logits"]               # [B, Q, 1]
    presence_dec: torch.Tensor = out["presence_logit_dec"]       # [B, 1] or [B, 1, 1]

    scores = _per_query_score(out)                               # [Q]
    q = int(torch.argmax(scores).item())
    n_queries = int(scores.shape[0])

    # ------------------------------------------------------------------
    # Per-query slices for the selected query
    # ------------------------------------------------------------------
    mask_q = pred_masks[0, q].detach().float().cpu()             # [H, W] raw logits
    box_cxcywh = pred_boxes[0, q].detach().float().cpu().tolist()
    box_xyxy = (
        pred_boxes_xyxy[0, q].detach().float().cpu().tolist()
        if pred_boxes_xyxy is not None
        else None
    )
    class_logit_q = float(pred_logits[0, q, 0].detach().float().cpu())
    presence_logit = float(presence_dec.detach().float().cpu().flatten()[0])
    score_q = float(scores[q])

    # ------------------------------------------------------------------
    # Per-layer detector-decoder hidden state for query q
    # ------------------------------------------------------------------
    # layer outputs are shape [Q, B, C] (seq-first). Index by q on dim 0.
    per_layer_hs: Dict[str, List[float]] = {}
    per_layer_hs_norm: Dict[str, float] = {}
    for layer_idx in sorted(collector.layer_hidden_states):
        hs = collector.layer_hidden_states[layer_idx]            # [Q, B, C]
        if hs.dim() != 3 or hs.shape[0] <= q:
            continue
        vec = hs[q, 0].float()                                   # [C]
        per_layer_hs[str(layer_idx)] = vec.tolist()
        per_layer_hs_norm[str(layer_idx)] = float(vec.norm().item())

    # ------------------------------------------------------------------
    # ca_text attention weights for query q at each layer
    # ------------------------------------------------------------------
    # ca_text input has the presence token concatenated at index 0, so the
    # detector queries occupy rows 1..Q+1. Query q is at row (q + 1).
    # Shape: [B, Q+1, T_text]  (or [B, H, Q+1, T_text] when force_weights=True)
    per_layer_text_attn: Dict[int, np.ndarray] = {}
    text_seq_len: Optional[int] = None
    presence_offset = 0  # row offset for query q
    for layer_idx in sorted(collector.layer_catext_attn):
        attn = collector.layer_catext_attn[layer_idx]            # [B, Q+1 or Q, T] or 4D
        if attn.dim() == 4:
            # [B, H, Q+1, T]  ->  average over heads
            attn = attn.mean(dim=1)                              # [B, Q+1, T]
        if attn.dim() != 3:
            continue
        rows = attn.shape[1]
        # Detect whether presence row is prepended.
        if rows == n_queries + 1:
            row = q + 1
            presence_offset = 1
        elif rows == n_queries:
            row = q
            presence_offset = 0
        else:
            # Some other layout -- just clamp.
            row = min(q, rows - 1)
        weights = attn[0, row].float().numpy()                   # [T]
        per_layer_text_attn[layer_idx] = weights
        if text_seq_len is None:
            text_seq_len = int(weights.shape[0])

    # ------------------------------------------------------------------
    # Image cross-attention map for query q at each detector layer
    # ------------------------------------------------------------------
    # decoder.layers.{i}.cross_attn returns weights shaped either
    #   ``[B, Q, S_img]``           (averaged over heads, default)  OR
    #   ``[B, num_heads, Q, S_img]``(per-head, when average_attn_weights=False)
    # We store one row per layer (averaged over heads, if present).
    per_layer_image_attn: List[Tuple[int, np.ndarray]] = []
    img_seq_len: Optional[int] = None
    img_grid_hw: Optional[Tuple[int, int]] = None
    image_q_row_offset = 0  # decoder cross_attn does NOT prepend presence
    for layer_idx in sorted(collector.layer_image_attn):
        attn = collector.layer_image_attn[layer_idx]
        if attn.dim() == 4:
            # [B, H, Q, S]  ->  average over heads
            attn_per_q = attn[0].float().mean(dim=0)             # [Q, S]
        elif attn.dim() == 3:
            attn_per_q = attn[0].float()                          # [Q, S]
        else:
            continue
        rows = attn_per_q.shape[0]
        if rows == n_queries:
            row_q = q
        elif rows == n_queries + 1:
            row_q = q + 1
            image_q_row_offset = 1
        else:
            row_q = min(q, rows - 1)
        weights = attn_per_q[row_q].numpy()                       # [S_img]
        per_layer_image_attn.append((layer_idx, weights))
        if img_seq_len is None:
            img_seq_len = int(weights.shape[0])
            # Image features are flattened from a square grid
            side = int(round(img_seq_len ** 0.5))
            if side * side == img_seq_len:
                img_grid_hw = (side, side)

    # Stack into ``[n_layers, H, W]`` (or ``[n_layers, S]`` if not square).
    if per_layer_image_attn:
        if img_grid_hw is not None:
            image_attn_stack = np.stack(
                [w.reshape(*img_grid_hw) for _, w in per_layer_image_attn], axis=0
            )
        else:
            image_attn_stack = np.stack([w for _, w in per_layer_image_attn], axis=0)
    else:
        image_attn_stack = np.zeros((0,), dtype=np.float32)
    image_attn_layers_recorded = [li for li, _ in per_layer_image_attn]

    # ------------------------------------------------------------------
    # Compute argmax / mask binarisation summary stats (raw, no resizing)
    # ------------------------------------------------------------------
    mask_np = mask_q.numpy()
    mask_summary = {
        "shape": list(mask_np.shape),
        "min": float(mask_np.min()),
        "max": float(mask_np.max()),
        "mean": float(mask_np.mean()),
        "std": float(mask_np.std()),
        "frac_positive_logit": float((mask_np > 0).mean()),
    }

    # ------------------------------------------------------------------
    # Write files
    # ------------------------------------------------------------------
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "selected_query_json": out_path / "selected_query.json",
        "selected_query_mask": out_path / "selected_query_mask.npy",
        "selected_query_text_attention": out_path / "selected_query_text_attention.csv",
        "selected_query_image_attention_layers":
            out_path / "selected_query_image_attention_layers.npy",
    }

    # 1. JSON metadata
    record = {
        "image_path": image_path,
        "prompt": prompt,
        "image_hw": list(image_hw) if image_hw else None,
        "confidence_threshold": confidence_threshold,
        "n_queries": n_queries,
        "selected_query_index": q,
        "score": score_q,
        "class_logit": class_logit_q,
        "presence_logit": presence_logit,
        "presence_score": float(torch.sigmoid(torch.tensor(presence_logit)).item()),
        "box_cxcywh_normalized": box_cxcywh,
        "box_xyxy_normalized": box_xyxy,
        "mask_logits_summary": mask_summary,
        "decoder_layers": {
            "n_layers_captured": len(per_layer_hs),
            "per_layer_hidden_state_norm": per_layer_hs_norm,
            "per_layer_hidden_state": per_layer_hs,  # full vectors
        },
        "ca_text": {
            "n_layers_captured": len(per_layer_text_attn),
            "text_seq_len": text_seq_len,
            "presence_token_offset": presence_offset,
            "row_used_for_query_q": q + presence_offset,
        },
        "ca_image": {
            "n_layers_captured": len(per_layer_image_attn),
            "image_seq_len": img_seq_len,
            "image_grid_hw": list(img_grid_hw) if img_grid_hw else None,
            "presence_token_offset": image_q_row_offset,
            "row_used_for_query_q": q + image_q_row_offset,
            "layers_recorded": image_attn_layers_recorded,
            "stack_shape": list(image_attn_stack.shape),
            "note": (
                "Per-head averaging applied. Empty if `force_weights=True` "
                "was not requested -- decoder cross_attn returns weights=None "
                "by default in SAM 3."
            ),
        },
        "files": {
            "mask_npy": str(paths["selected_query_mask"]),
            "text_attention_csv": str(paths["selected_query_text_attention"]),
            "image_attention_layers_npy": str(
                paths["selected_query_image_attention_layers"]
            ),
        },
        "notes": (
            "mask_logits are the RAW pre-sigmoid, pre-resize, pre-threshold "
            "outputs of segmentation_head.pred_masks at the model's native "
            "mask resolution. Boxes are normalised cxcywh in [0,1]. "
            "score = sigmoid(class_logit) * sigmoid(presence_logit)."
        ),
    }
    paths["selected_query_json"].write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    # 2. NPY -- raw mask logits for the selected query (no sigmoid, no resize)
    np.save(paths["selected_query_mask"], mask_np)

    # 2b. NPY -- per-layer image cross-attention map for query q
    np.save(paths["selected_query_image_attention_layers"], image_attn_stack)

    # 3. CSV -- ca_text attention weights for query q at each detector layer
    with paths["selected_query_text_attention"].open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        w = csv.writer(fh)
        n_text = text_seq_len or 0
        header = ["layer_index", "selected_query_index", "row_used", "text_seq_len"] + [
            f"text_token_{i}" for i in range(n_text)
        ]
        w.writerow(header)
        for layer_idx in sorted(per_layer_text_attn):
            weights = per_layer_text_attn[layer_idx]
            row = [
                layer_idx,
                q,
                q + presence_offset,
                int(weights.shape[0]),
            ] + [float(x) for x in weights.tolist()]
            w.writerow(row)

    return paths
