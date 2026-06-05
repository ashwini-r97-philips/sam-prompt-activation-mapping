"""Forward-hook tracer for SAM 3 image + text inference.

The tracer is **non-invasive**: it never edits SAM 3 source. It registers
``forward_pre_hook`` and ``forward_hook`` on a curated list of submodules
and records lightweight metadata (shapes, dtypes, simple stats) for every
input / output tensor it sees.

Outputs (written by :func:`write_outputs`):
    pam_trace/module_map.md       -- all named modules + which were hooked
    pam_trace/forward_trace.jsonl -- one JSON line per hook event
    pam_trace/tensor_shapes.csv   -- flat CSV of every input/output tensor
    pam_trace/arrows.csv          -- inferred data-flow edges (chronological)
"""

from __future__ import annotations

import csv
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Default patterns (dotted module path -- shell-style ``*`` matches one segment)
# ---------------------------------------------------------------------------
# Each pattern is a 4-tuple: (pattern, source_class, group, diagram_name).
#   - ``source_class``: tag for the kind of activation
#   - ``group``: section label used for module_map.md grouping
#   - ``diagram_name``: human-readable block name from the SAM 3 architecture
#     diagram (e.g. "Multimodal Decoder", "Detector Decoder", "Pixel Decoder").
#     This intentionally differs from the ``code_name`` in the repo because the
#     SAM 3 codebase renamed several blocks (e.g. ``TransformerEncoderLayer``
#     used to be ``TransformerDecoderLayer`` -- see the class docstring).
DEFAULT_PATTERNS: List[Tuple[str, str, str, str]] = [
    # 1. Text encoder (a.k.a. "language_backbone" in SAM3 image build)
    ("backbone.text_encoder", "text", "text_encoder", "Text Encoder"),
    ("backbone.text_encoder.*", "text", "text_encoder", "Text Encoder"),
    ("backbone.language_backbone", "text", "text_encoder", "Text Encoder"),
    ("backbone.language_backbone.encoder", "text", "text_encoder", "Text Encoder"),
    ("backbone.language_backbone.resizer", "text", "text_encoder", "Text Encoder (Resizer)"),
    # 2. Image encoder (ViT trunk + neck / multi-scale convs)
    ("backbone.image_encoder", "image", "image_encoder", "Image Encoder"),
    ("backbone.trunk", "image", "image_encoder", "Image Encoder (ViT trunk)"),
    ("backbone.neck", "image", "image_encoder", "Image Encoder (Neck)"),
    ("backbone.vision_encoder", "image", "image_encoder", "Image Encoder"),
    ("backbone.vision_backbone", "image", "image_encoder", "Image Encoder"),
    ("backbone.vision_backbone.trunk", "image", "image_encoder", "Image Encoder (ViT trunk)"),
    ("backbone.vision_backbone.convs", "image", "image_encoder", "Image Encoder (Multi-scale Neck)"),
    ("backbone.vision_backbone.position_encoding", "image", "image_encoder", "Image Encoder (Pos Enc)"),
    # Geometry / prompt encoder (boxes / points)
    ("prompt_encoder", "prompt_geom", "prompt_encoder", "Geometry Prompt Encoder"),
    ("prompt_encoder.*", "prompt_geom", "prompt_encoder", "Geometry Prompt Encoder"),
    ("geometry_encoder", "prompt_geom", "prompt_encoder", "Geometry Prompt Encoder"),
    ("geometry_encoder.*", "prompt_geom", "prompt_encoder", "Geometry Prompt Encoder"),
    # 3+4. Multimodal Decoder (text-image fusion).
    # NOTE: The repo calls this ``transformer.encoder`` and the layer class is
    # ``TransformerEncoderLayer``, but the class docstring states it was
    # "previously called TransformerDecoderLayer". In the SAM 3 architecture
    # diagram this is the central "Multimodal Decoder" block: queries are the
    # prompt sequence (text + geometry + visual_prompt), keys/values are image
    # features (via ``cross_attn_image``).
    ("transformer.encoder", "fused", "multimodal_decoder", "Multimodal Decoder"),
    ("transformer.encoder.layers.*", "fused", "multimodal_decoder", "Multimodal Decoder Layer"),
    ("transformer.encoder.layers.*.self_attn", "fused", "multimodal_decoder", "Multimodal Decoder Self-Attention"),
    ("transformer.encoder.layers.*.cross_attn_image", "fused", "multimodal_decoder", "Multimodal Decoder Cross-Attention to Image"),
    ("transformer.encoder.layers.*.cross_attn_text", "fused", "multimodal_decoder", "Multimodal Decoder Cross-Attention to Text"),
    # 6+7. Detector decoder (object queries + cross-attention to fused features)
    ("transformer.decoder", "detector_queries", "detector_decoder", "Detector Decoder"),
    ("transformer.decoder.layers.*", "detector_queries", "detector_decoder", "Detector Decoder Layer"),
    ("transformer.decoder.layers.*.self_attn", "detector_queries", "detector_decoder", "Detector Decoder Self-Attention"),
    ("transformer.decoder.layers.*.cross_attn", "detector_queries", "detector_decoder", "Detector Decoder Cross-Attention to Image"),
    ("transformer.decoder.layers.*.ca_text", "detector_queries", "detector_decoder", "Detector Decoder Cross-Attention to Text"),
    ("transformer.decoder.layers.*.cross_attn_text", "detector_queries", "detector_decoder", "Detector Decoder Cross-Attention to Text"),
    # 9. Final class / box / presence heads
    ("class_embed", "final_outputs", "heads", "Class Head"),
    ("transformer.decoder.bbox_embed", "final_outputs", "heads", "Box Head"),
    ("transformer.decoder.class_embed", "final_outputs", "heads", "Class Head"),
    ("transformer.decoder.presence_head", "final_outputs", "heads", "Presence Head"),
    ("presence_head", "final_outputs", "heads", "Presence Head"),
    # 5+8+10. Segmentation head (pixel decoder, mask predictor, semantic head)
    ("segmentation_head", "mask_head", "segmentation", "Segmentation Head"),
    ("segmentation_head.pixel_decoder", "pixel_features", "segmentation", "Pixel Decoder"),
    ("segmentation_head.instance_seg_head", "mask_head", "segmentation", "Mask Predictor"),
    ("segmentation_head.instance_seg_head.mask_predictor", "mask_head", "segmentation", "Mask Predictor"),
    ("segmentation_head.semantic_seg_head", "semantic", "segmentation", "Semantic Head"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"\.layers\.(\d+)\b")


def _layer_index(module_path: str) -> Optional[int]:
    m = _LAYER_RE.search(module_path)
    return int(m.group(1)) if m else None


def _glob_to_regex(pattern: str) -> re.Pattern:
    """Translate ``a.b.*.c`` into a regex matching one dotted segment per ``*``."""
    parts = pattern.split(".")
    rx = []
    for p in parts:
        if p == "*":
            rx.append(r"[^.]+")
        else:
            rx.append(re.escape(p))
    return re.compile(r"^" + r"\.".join(rx) + r"$")


def _module_source_file(mod: nn.Module) -> str:
    try:
        return inspect.getsourcefile(mod.__class__) or ""
    except (TypeError, OSError):
        return ""


def _tensor_meta(t: torch.Tensor, name: str) -> Dict[str, Any]:
    """Lightweight metadata for one tensor (no copies, no .item() on huge tensors)."""
    meta: Dict[str, Any] = {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(t.device),
        "requires_grad": bool(t.requires_grad),
    }
    if t.is_floating_point() and t.numel() > 0:
        try:
            with torch.no_grad():
                tf = t.detach().float()
                meta["mean"] = float(tf.mean().cpu())
                meta["std"] = float(tf.std().cpu()) if t.numel() > 1 else 0.0
                meta["min"] = float(tf.min().cpu())
                meta["max"] = float(tf.max().cpu())
                meta["has_nan"] = bool(torch.isnan(tf).any().cpu())
        except Exception:  # noqa: BLE001 - stats are best-effort
            pass
    return meta


def _walk_tensors(obj: Any, prefix: str = "") -> List[Tuple[str, torch.Tensor]]:
    """Recursively collect ``(name, Tensor)`` pairs from arbitrary nested structures."""
    out: List[Tuple[str, torch.Tensor]] = []
    if isinstance(obj, torch.Tensor):
        out.append((prefix or "tensor", obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_walk_tensors(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(_walk_tensors(v, f"{prefix}[{i}]" if prefix else f"[{i}]"))
    # other types (None, int, bool, ...) are ignored
    return out


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass
class HookedModuleInfo:
    path: str               # ``code_name`` -- dotted module path in the model
    class_name: str         # Python class name (e.g. ``TransformerEncoderLayer``)
    source_file: str        # absolute path to the file defining the class
    source_class: str       # activation kind tag (text/image/fused/...)
    group: str              # group label used for module_map.md grouping
    diagram_name: str       # human-readable block name from the SAM 3 diagram
    layer_index: Optional[int]
    matched_pattern: str
    hooked: bool = True


@dataclass
class TraceEvent:
    event_id: int
    t_ms: float
    module_path: str        # ``code_name``
    class_name: str
    source_file: str
    source_class: str
    group: str
    diagram_name: str
    layer_index: Optional[int]
    kind: str  # "input" | "output"
    tensors: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class PamTracer:
    """Attach forward hooks to a SAM 3 model and record activations."""

    def __init__(
        self,
        model: nn.Module,
        patterns: Optional[Sequence[Tuple[str, str, str, str]]] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.patterns: List[Tuple[str, str, str, str]] = list(patterns or DEFAULT_PATTERNS)
        self.verbose = verbose

        # discovery + hook bookkeeping
        self.all_modules: List[Tuple[str, nn.Module]] = list(model.named_modules())
        self.hooked_modules: List[HookedModuleInfo] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        # event log
        self.events: List[TraceEvent] = []
        self._t0: Optional[float] = None
        self._next_event_id: int = 0

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def _resolve_patterns(self) -> List[HookedModuleInfo]:
        """Return one HookedModuleInfo per (module, pattern) match."""
        compiled = [
            (pat, _glob_to_regex(pat), src_cls, group, diagram_name)
            for pat, src_cls, group, diagram_name in self.patterns
        ]
        infos: List[HookedModuleInfo] = []
        seen_paths: set[str] = set()
        for path, mod in self.all_modules:
            if path == "":
                continue  # skip top-level model itself
            for raw_pat, rx, src_cls, group, diagram_name in compiled:
                if rx.match(path):
                    if path in seen_paths:
                        # keep first match (more specific patterns can be put later
                        # if you want them to win -- here first wins for stability)
                        continue
                    seen_paths.add(path)
                    infos.append(
                        HookedModuleInfo(
                            path=path,
                            class_name=type(mod).__name__,
                            source_file=_module_source_file(mod),
                            source_class=src_cls,
                            group=group,
                            diagram_name=diagram_name,
                            layer_index=_layer_index(path),
                            matched_pattern=raw_pat,
                        )
                    )
                    break
        return infos

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def register(self) -> int:
        """Discover matching modules and register forward hooks. Returns count."""
        if self._hook_handles:
            raise RuntimeError("PamTracer already registered. Call remove() first.")

        self.hooked_modules = self._resolve_patterns()

        # Build a quick lookup: module_path -> info
        path_to_info = {info.path: info for info in self.hooked_modules}
        for path, mod in self.all_modules:
            info = path_to_info.get(path)
            if info is None:
                continue
            self._hook_handles.append(
                mod.register_forward_pre_hook(self._make_pre_hook(info))
            )
            self._hook_handles.append(
                mod.register_forward_hook(self._make_post_hook(info))
            )
        if self.verbose:
            print(f"[pam_trace] Registered hooks on {len(self.hooked_modules)} modules")
        return len(self.hooked_modules)

    def remove(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def __enter__(self) -> "PamTracer":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _now_ms(self) -> float:
        if self._t0 is None:
            self._t0 = time.perf_counter()
        return (time.perf_counter() - self._t0) * 1000.0

    def _record(self, info: HookedModuleInfo, kind: str, payload: Any) -> None:
        tensors = _walk_tensors(payload)
        if not tensors:
            return
        event = TraceEvent(
            event_id=self._next_event_id,
            t_ms=self._now_ms(),
            module_path=info.path,
            class_name=info.class_name,
            source_file=info.source_file,
            source_class=info.source_class,
            group=info.group,
            diagram_name=info.diagram_name,
            layer_index=info.layer_index,
            kind=kind,
            tensors=[_tensor_meta(t, name) for name, t in tensors],
        )
        self._next_event_id += 1
        self.events.append(event)
        if self.verbose:
            shapes = [t["shape"] for t in event.tensors]
            print(f"[pam_trace] {kind:6s} {info.path:60s} {info.class_name:30s} {shapes}")

    def _make_pre_hook(self, info: HookedModuleInfo):
        def hook(_module: nn.Module, inputs: Tuple[Any, ...]):
            self._record(info, "input", list(inputs))
            return None
        return hook

    def _make_post_hook(self, info: HookedModuleInfo):
        def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any):
            self._record(info, "output", output)
            return None
        return hook

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------

    def write(self, out_dir: str | Path) -> Dict[str, Path]:
        return write_outputs(self, out_dir)


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def _format_shape(shape: Sequence[int]) -> str:
    return "x".join(str(s) for s in shape)


def write_outputs(tracer: PamTracer, out_dir: str | Path) -> Dict[str, Path]:
    """Write the 4 trace files into *out_dir*. Returns mapping of file -> path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "module_map": out / "module_map.md",
        "forward_trace": out / "forward_trace.jsonl",
        "tensor_shapes": out / "tensor_shapes.csv",
        "arrows": out / "arrows.csv",
    }

    _write_module_map(tracer, paths["module_map"])
    _write_forward_trace(tracer, paths["forward_trace"])
    _write_tensor_shapes(tracer, paths["tensor_shapes"])
    _write_arrows(tracer, paths["arrows"])
    return paths


def _write_module_map(tracer: PamTracer, path: Path) -> None:
    hooked = {info.path: info for info in tracer.hooked_modules}

    # Group hooked modules by their group label, preserving discovery order.
    groups: Dict[str, List[HookedModuleInfo]] = {}
    for info in tracer.hooked_modules:
        groups.setdefault(info.group, []).append(info)

    lines: List[str] = []
    lines.append("# SAM 3 Module Map (PAM Trace)")
    lines.append("")
    lines.append(
        f"Total named modules in model: **{len(tracer.all_modules)}**  "
        f"Hooked: **{len(tracer.hooked_modules)}**"
    )
    lines.append("")
    lines.append(
        "Each row carries two names:\n"
        "- **`code_name`** -- the dotted module path in the SAM 3 source code\n"
        "  (e.g. `transformer.encoder.layers.0`).\n"
        "- **`diagram_name`** -- the architecture-block label from the SAM 3\n"
        "  diagram (e.g. *Multimodal Decoder*). These names sometimes differ:\n"
        "  notably the central fusion block is called `transformer.encoder`\n"
        "  in code but appears as the **Multimodal Decoder** in the diagram.\n"
        "  The class docstring of `TransformerEncoderLayer` explicitly notes\n"
        "  that the layer was \"previously called TransformerDecoderLayer\"."
    )
    lines.append("")
    lines.append("## Hooked modules by group")
    lines.append("")
    for group, items in groups.items():
        lines.append(f"### `{group}`")
        lines.append("")
        lines.append("| Layer | code_name (path) | code_name (class) | diagram_name | Source tag | Source file |")
        lines.append("|---:|---|---|---|---|---|")
        for info in items:
            layer = "" if info.layer_index is None else str(info.layer_index)
            src_file = info.source_file
            # shorten path -- keep last 3 components
            if src_file:
                short = "/".join(Path(src_file).parts[-3:])
            else:
                short = ""
            lines.append(
                f"| {layer} | `{info.path}` | `{info.class_name}` "
                f"| {info.diagram_name} | `{info.source_class}` | `{short}` |"
            )
        lines.append("")

    lines.append("## All named modules (full inventory)")
    lines.append("")
    lines.append("| code_name (path) | code_name (class) | Hooked | diagram_name | Source tag |")
    lines.append("|---|---|:---:|---|---|")
    for path_, mod in tracer.all_modules:
        if path_ == "":
            continue
        info = hooked.get(path_)
        mark = "yes" if info else ""
        sc = info.source_class if info else ""
        diagram = info.diagram_name if info else ""
        lines.append(
            f"| `{path_}` | `{type(mod).__name__}` | {mark} | {diagram} | `{sc}` |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_forward_trace(tracer: PamTracer, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for ev in tracer.events:
            row = {
                "event_id": ev.event_id,
                "t_ms": round(ev.t_ms, 3),
                "code_name": ev.module_path,
                "code_class": ev.class_name,
                "diagram_name": ev.diagram_name,
                "source_file": ev.source_file,
                "source_class": ev.source_class,
                "group": ev.group,
                "layer_index": ev.layer_index,
                "kind": ev.kind,
                "tensors": ev.tensors,
            }
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def _write_tensor_shapes(tracer: PamTracer, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "event_id",
                "t_ms",
                "code_name",
                "code_class",
                "diagram_name",
                "source_class",
                "group",
                "layer_index",
                "kind",
                "tensor_idx",
                "tensor_name",
                "shape",
                "dtype",
                "device",
                "requires_grad",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        for ev in tracer.events:
            for i, t in enumerate(ev.tensors):
                w.writerow(
                    [
                        ev.event_id,
                        f"{ev.t_ms:.3f}",
                        ev.module_path,
                        ev.class_name,
                        ev.diagram_name,
                        ev.source_class,
                        ev.group,
                        "" if ev.layer_index is None else ev.layer_index,
                        ev.kind,
                        i,
                        t.get("name", ""),
                        _format_shape(t.get("shape", [])),
                        t.get("dtype", ""),
                        t.get("device", ""),
                        t.get("requires_grad", ""),
                        t.get("mean", ""),
                        t.get("std", ""),
                        t.get("min", ""),
                        t.get("max", ""),
                    ]
                )


def _write_arrows(tracer: PamTracer, path: Path) -> None:
    """Infer a chronological data-flow graph.

    For each module we use its largest (most-elements) output tensor as the
    "primary" output, and link it to the next module whose inputs contain a
    tensor with the same shape. This is heuristic but useful for visualising
    how activations propagate across the network.
    """
    # Collect (event_id, code_name, diagram_name, kind, primary_tensor_meta).
    primary: List[Tuple[int, str, str, str, Dict[str, Any]]] = []
    for ev in tracer.events:
        if not ev.tensors:
            continue
        # pick largest tensor by num elements
        def _numel(meta: Dict[str, Any]) -> int:
            n = 1
            for d in meta.get("shape", []):
                n *= int(d)
            return n
        biggest = max(ev.tensors, key=_numel)
        primary.append(
            (ev.event_id, ev.module_path, ev.diagram_name, ev.kind, biggest)
        )

    # For each output, find next input event with same shape.
    edges: List[Tuple[str, str, str, str, str, str, str]] = []
    for i, (eid_a, mod_a, dia_a, kind_a, t_a) in enumerate(primary):
        if kind_a != "output":
            continue
        shape_a = tuple(t_a.get("shape", []))
        for j in range(i + 1, len(primary)):
            eid_b, mod_b, dia_b, kind_b, t_b = primary[j]
            if kind_b != "input":
                continue
            if mod_b == mod_a:
                continue
            if tuple(t_b.get("shape", [])) == shape_a:
                edges.append(
                    (
                        mod_a,
                        mod_b,
                        dia_a,
                        dia_b,
                        _format_shape(shape_a),
                        t_a.get("dtype", ""),
                        f"{eid_a}->{eid_b}",
                    )
                )
                break

    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "from_code_name",
                "to_code_name",
                "from_diagram_name",
                "to_diagram_name",
                "shape",
                "dtype",
                "event_pair",
            ]
        )
        for row in edges:
            w.writerow(row)
