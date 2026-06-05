"""Attention-level tracing for Prompt Activation Mapping on SAM 3.

Hooks the 10 attention module families specified by the user with rich
per-call metadata: query/key/value shapes, output shape, attention-weight
shape (if returned), self-vs-cross classification, and the inferred
"updated stream" (image / text / geometry / detector_query / fused / mask).

Outputs (written by :func:`write_attention_outputs`):
    pam_trace/attention_trace.jsonl
    pam_trace/attention_summary.csv

Optional analysis-only mode ``force_attention_weights=True`` monkey-patches
the ``forward`` method of compatible attention modules (PyTorch
``nn.MultiheadAttention`` and SAM 3 ``MultiheadAttentionWrapper``) so they
return per-head attention weights even when the caller passed
``need_weights=False``.  The first element of the returned tuple
(the ``attn_output`` tensor) is unchanged, so model outputs are preserved.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Module patterns -- (regex, attn_kind, stream, diagram_name)
# ---------------------------------------------------------------------------
# ``attn_kind`` distinguishes self_attn / cross_attn_image / cross_attn_text /
# vit_attn etc.  ``stream`` is the user-facing "updated stream" label.
ATTN_PATTERNS: List[Tuple[re.Pattern, str, str, str]] = [
    (
        re.compile(r"^backbone\.vision_backbone\.trunk\.blocks\.(\d+)\.attn$"),
        "vit_self_attn",
        "image",
        "Image Encoder Self-Attention (ViT)",
    ),
    (
        re.compile(
            r"^backbone\.language_backbone\.encoder\.transformer\.resblocks\.(\d+)\.attn$"
        ),
        "text_self_attn",
        "text",
        "Text Encoder Self-Attention",
    ),
    (
        re.compile(r"^geometry_encoder\.encode\.(\d+)\.self_attn$"),
        "self_attn",
        "geometry",
        "Geometry Encoder Self-Attention",
    ),
    (
        re.compile(r"^geometry_encoder\.encode\.(\d+)\.cross_attn_image$"),
        "cross_attn_image",
        "geometry",
        "Geometry Encoder Cross-Attention to Image",
    ),
    (
        re.compile(r"^transformer\.encoder\.layers\.(\d+)\.self_attn$"),
        "self_attn",
        "fused",
        "Multimodal Decoder Self-Attention",
    ),
    (
        re.compile(r"^transformer\.encoder\.layers\.(\d+)\.cross_attn_image$"),
        "cross_attn_image",
        "fused",
        "Multimodal Decoder Cross-Attention to Image",
    ),
    (
        re.compile(r"^transformer\.decoder\.layers\.(\d+)\.self_attn$"),
        "self_attn",
        "detector_query",
        "Detector Decoder Self-Attention",
    ),
    (
        re.compile(r"^transformer\.decoder\.layers\.(\d+)\.cross_attn$"),
        "cross_attn_image",
        "detector_query",
        "Detector Decoder Cross-Attention to Image",
    ),
    (
        re.compile(r"^transformer\.decoder\.layers\.(\d+)\.ca_text$"),
        "cross_attn_text",
        "detector_query",
        "Detector Decoder Cross-Attention to Text",
    ),
    (
        re.compile(r"^segmentation_head\.cross_attend_prompt$"),
        "cross_attn_prompt",
        "mask",
        "Mask Head Cross-Attention to Prompt",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shape_of(x: Any) -> Optional[List[int]]:
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    return None


def _first_tensor(obj: Any) -> Optional[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def _extract_qkv(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Best-effort recovery of (query, key, value) tensors from a forward call.

    Handles three signatures:
    - ``forward(x)``              -- ViT-style self-attention; q == k == v == x
    - ``forward(query, key, value, ...)`` -- nn.MultiheadAttention / SAM 3 wrapper
    - kwargs ``query=`` / ``key=`` / ``value=``
    """
    q = k = v = None
    # kwargs win first
    if "query" in kwargs:
        q = kwargs["query"]
    if "key" in kwargs:
        k = kwargs["key"]
    if "value" in kwargs:
        v = kwargs["value"]

    # Fall back to positional
    pos_tensors = [a for a in args if isinstance(a, torch.Tensor)]
    if q is None and len(pos_tensors) >= 1:
        q = pos_tensors[0]
    if k is None and len(pos_tensors) >= 2:
        k = pos_tensors[1]
    if v is None and len(pos_tensors) >= 3:
        v = pos_tensors[2]

    # ViT-style: only one tensor -> q == k == v
    if q is not None and k is None and v is None:
        k = q
        v = q

    return q, k, v


def _output_tensor_and_weights(output: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return ``(attn_output, attn_weights)`` from a module's forward output."""
    if isinstance(output, torch.Tensor):
        return output, None
    if isinstance(output, (tuple, list)):
        if len(output) == 0:
            return None, None
        first = output[0] if isinstance(output[0], torch.Tensor) else _first_tensor(output[0])
        weights = None
        if len(output) >= 2 and isinstance(output[1], torch.Tensor):
            weights = output[1]
        return first, weights
    return None, None


def _detect_attn_type(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    declared_kind: str,
) -> str:
    """Return ``"self"`` or ``"cross"`` -- prefers tensor identity, falls back to kind."""
    if q is not None and k is not None:
        if q.data_ptr() == k.data_ptr() and (v is None or v.data_ptr() == q.data_ptr()):
            return "self"
        if list(q.shape) != list(k.shape):
            return "cross"
    if "self" in declared_kind:
        return "self"
    return "cross"


# ---------------------------------------------------------------------------
# Force-weights monkey-patch (analysis-only)
# ---------------------------------------------------------------------------


def force_attention_weights(modules: Sequence[Tuple[str, nn.Module]]) -> Callable[[], None]:
    """Wrap ``forward`` on each module so it returns per-head attention weights.

    Handles two cases:

    1. **PyTorch** ``nn.MultiheadAttention`` -- wrap forward, inject
       ``need_weights=True`` and ``average_attn_weights=False``.
    2. **SAM 3** ``MultiheadAttentionWrapper`` -- this is an ``nn.MultiheadAttention``
       subclass whose own ``forward`` HARDCODES ``need_weights=False``
       (see ``sam3.model.model_misc.MultiheadAttentionWrapper``).
       We must bypass it by calling ``nn.MultiheadAttention.forward``
       directly so the underlying PyTorch implementation honours
       ``need_weights=True``.

    ViT-style custom ``Attention`` classes have a different signature and
    are skipped (their weights have to be captured a different way).

    Returns a callable that restores the original forward methods.
    """
    import types

    saved: List[Tuple[nn.Module, Any]] = []

    for _path, mod in modules:
        cls_name = type(mod).__name__

        if cls_name == "MultiheadAttentionWrapper":
            # SAM 3 wrapper: bypass it -- go straight to nn.MultiheadAttention.forward.
            original_forward = mod.forward

            def _wrapper_bypass(self, *args, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return nn.MultiheadAttention.forward(self, *args, **kwargs)

            mod.forward = types.MethodType(_wrapper_bypass, mod)  # type: ignore[assignment]
            saved.append((mod, original_forward))

        elif cls_name == "MultiheadAttention" or isinstance(mod, nn.MultiheadAttention):
            # PyTorch nn.MultiheadAttention -- wrap forward to inject kwargs.
            original_forward = mod.forward

            def _make_wrapper(_orig: Callable) -> Callable:
                def wrapper(*args, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False
                    try:
                        return _orig(*args, **kwargs)
                    except TypeError:
                        kwargs.pop("need_weights", None)
                        kwargs.pop("average_attn_weights", None)
                        return _orig(*args, **kwargs)
                return wrapper

            mod.forward = _make_wrapper(original_forward)  # type: ignore[assignment]
            saved.append((mod, original_forward))
        # else: unsupported class -- skip silently

    def restore() -> None:
        for mod, orig in saved:
            try:
                mod.forward = orig  # type: ignore[assignment]
            except Exception:  # noqa: BLE001
                pass

    return restore


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass
class AttentionEvent:
    event_id: int
    module_path: str
    code_class: str
    diagram_name: str
    layer_index: Optional[int]
    attn_kind: str            # vit_self_attn / self_attn / cross_attn_image / ...
    attn_type: str            # "self" | "cross"
    stream: str               # image / text / geometry / detector_query / fused / mask
    query_shape: Optional[List[int]] = None
    key_shape: Optional[List[int]] = None
    value_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    attn_weights_shape: Optional[List[int]] = None
    output_seq_matches_query: Optional[bool] = None
    weights_returned: bool = False
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class AttentionTracer:
    """Hook-based tracer for the 10 attention-module families."""

    def __init__(self, model: nn.Module, force_weights: bool = False):
        self.model = model
        self.force_weights = force_weights
        self.events: List[AttentionEvent] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._restore_force: Optional[Callable[[], None]] = None
        # transient per-call buffers keyed by id(module)
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._next_event_id = 0
        # cached classification per matched module
        self._matched: List[
            Tuple[str, nn.Module, str, str, str, Optional[int]]
        ] = []  # (path, mod, attn_kind, stream, diagram_name, layer_idx)

    # ------------------------------------------------------------------
    # Discovery + registration
    # ------------------------------------------------------------------

    def _match_modules(self) -> None:
        for path, mod in self.model.named_modules():
            if path == "":
                continue
            for rx, attn_kind, stream, diagram_name in ATTN_PATTERNS:
                m = rx.match(path)
                if m:
                    layer_idx = int(m.group(1)) if m.groups() else None
                    self._matched.append(
                        (path, mod, attn_kind, stream, diagram_name, layer_idx)
                    )
                    break

    def register(self) -> int:
        if self._handles:
            raise RuntimeError("AttentionTracer already registered.")
        self._match_modules()

        if self.force_weights:
            self._restore_force = force_attention_weights(
                [(p, m) for (p, m, *_rest) in self._matched]
            )

        for path, mod, attn_kind, stream, diagram_name, layer_idx in self._matched:
            self._handles.append(
                mod.register_forward_pre_hook(
                    self._make_pre_hook(path, attn_kind, stream, diagram_name, layer_idx),
                    with_kwargs=True,
                )
            )
            self._handles.append(
                mod.register_forward_hook(
                    self._make_post_hook(path, attn_kind, stream, diagram_name, layer_idx)
                )
            )
        return len(self._matched)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self._restore_force is not None:
            self._restore_force()
            self._restore_force = None

    def __enter__(self) -> "AttentionTracer":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_pre_hook(self, path: str, attn_kind: str, stream: str,
                       diagram_name: str, layer_idx: Optional[int]):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], kwargs: Dict[str, Any]):
            # ``with_kwargs=True`` -- we get both positional and keyword args.
            q, k, v = _extract_qkv(inputs, kwargs)
            self._pending[id(module)] = {
                "path": path,
                "code_class": type(module).__name__,
                "attn_kind": attn_kind,
                "stream": stream,
                "diagram_name": diagram_name,
                "layer_idx": layer_idx,
                "q_shape": _shape_of(q),
                "k_shape": _shape_of(k),
                "v_shape": _shape_of(v),
                "q_len": None if q is None else (q.shape[0] if q.dim() == 3 and q.shape[0] != q.shape[1] else q.shape[-2]),
            }
            return None
        return hook

    def _make_post_hook(self, path: str, attn_kind: str, stream: str,
                        diagram_name: str, layer_idx: Optional[int]):
        def hook(module: nn.Module, _inputs, output: Any):
            pending = self._pending.pop(id(module), {})
            attn_out, attn_weights = _output_tensor_and_weights(output)
            q_shape = pending.get("q_shape")
            out_shape = _shape_of(attn_out)

            # Determine whether the output sequence length matches the query
            # length.  We try to be layout-agnostic: PyTorch MHA / SAM3 wrapper
            # use seq-first ``[L, B, C]`` for transformer encoders/decoders
            # and batch-first ``[B, L, C]`` for some others.  The "seq" dim is
            # whichever is not 1 and not the embed dim.
            output_seq_matches = None
            if q_shape is not None and out_shape is not None:
                # compare along the longest non-channel dim
                def _seq_len(s: List[int]) -> Optional[int]:
                    if len(s) >= 2:
                        # pick the largest dim that's not the last (assume last == channels)
                        return max(s[:-1])
                    return None
                qs = _seq_len(q_shape)
                os_ = _seq_len(out_shape)
                if qs is not None and os_ is not None:
                    output_seq_matches = (qs == os_)

            # Determine attn type from q/k/v identity if we still have them
            # (we don't keep tensors after the call, so we use shapes + kind).
            attn_type = "self" if "self" in attn_kind else "cross"
            if q_shape is not None and pending.get("k_shape") is not None:
                if q_shape == pending["k_shape"]:
                    # likely self-attention if shapes match
                    if "cross" in attn_kind:
                        # but kind says cross -- keep "cross" (e.g. cross_attn_image
                        # could still have q==k shapes if image features match query length)
                        attn_type = "cross"
                    else:
                        attn_type = "self"
                else:
                    attn_type = "cross"

            event = AttentionEvent(
                event_id=self._next_event_id,
                module_path=path,
                code_class=pending.get("code_class", type(module).__name__),
                diagram_name=diagram_name,
                layer_index=layer_idx,
                attn_kind=attn_kind,
                attn_type=attn_type,
                stream=stream,
                query_shape=q_shape,
                key_shape=pending.get("k_shape"),
                value_shape=pending.get("v_shape"),
                output_shape=out_shape,
                attn_weights_shape=_shape_of(attn_weights),
                output_seq_matches_query=output_seq_matches,
                weights_returned=attn_weights is not None,
            )
            self._next_event_id += 1
            self.events.append(event)
            return None
        return hook


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def write_attention_outputs(tracer: AttentionTracer, out_dir: str | Path) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "attention_trace": out / "attention_trace.jsonl",
        "attention_summary": out / "attention_summary.csv",
    }
    _write_attention_jsonl(tracer, paths["attention_trace"])
    _write_attention_csv(tracer, paths["attention_summary"])
    return paths


def _write_attention_jsonl(tracer: AttentionTracer, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for ev in tracer.events:
            row = {
                "event_id": ev.event_id,
                "module_path": ev.module_path,
                "code_class": ev.code_class,
                "diagram_name": ev.diagram_name,
                "layer_index": ev.layer_index,
                "attn_kind": ev.attn_kind,
                "attn_type": ev.attn_type,
                "stream": ev.stream,
                "query_shape": ev.query_shape,
                "key_shape": ev.key_shape,
                "value_shape": ev.value_shape,
                "output_shape": ev.output_shape,
                "attn_weights_shape": ev.attn_weights_shape,
                "weights_returned": ev.weights_returned,
                "output_seq_matches_query": ev.output_seq_matches_query,
            }
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def _shape_str(s: Optional[List[int]]) -> str:
    if s is None:
        return ""
    return "x".join(str(d) for d in s)


def _write_attention_csv(tracer: AttentionTracer, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "event_id",
                "module_path",
                "code_class",
                "diagram_name",
                "layer_index",
                "attn_kind",
                "attn_type",
                "stream",
                "query_shape",
                "key_shape",
                "value_shape",
                "output_shape",
                "attn_weights_shape",
                "weights_returned",
                "output_seq_matches_query",
            ]
        )
        for ev in tracer.events:
            w.writerow(
                [
                    ev.event_id,
                    ev.module_path,
                    ev.code_class,
                    ev.diagram_name,
                    "" if ev.layer_index is None else ev.layer_index,
                    ev.attn_kind,
                    ev.attn_type,
                    ev.stream,
                    _shape_str(ev.query_shape),
                    _shape_str(ev.key_shape),
                    _shape_str(ev.value_shape),
                    _shape_str(ev.output_shape),
                    _shape_str(ev.attn_weights_shape),
                    ev.weights_returned,
                    "" if ev.output_seq_matches_query is None else ev.output_seq_matches_query,
                ]
            )
