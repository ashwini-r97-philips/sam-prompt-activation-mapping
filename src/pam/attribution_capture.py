"""Attribution-flow hook system for SAM 3 MultiheadAttention modules.

Registers forward hooks on the three groups of cross-attention modules,
captures inputs/kwargs/outputs, retains gradients on the output tensor,
and reconstructs MHA internals for edge-attribution computation.

Groups
------
- ``group_a_geometry``   : ``geometry_encoder.encode.{0-2}.cross_attn_image``
- ``group_a_encoder``    : ``transformer.encoder.layers.{0-5}.cross_attn_image``
- ``group_b_decoder``    : ``transformer.decoder.layers.{0-5}.cross_attn``

Excluded
--------
- ``self_attn`` (encoder/decoder self-attention)
- ``ca_text``   (decoder text cross-attention — extension point for later)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .mha_reconstruction import MHAReconstruction, reconstruct_mha


# ---------------------------------------------------------------------------
# Group classification
# ---------------------------------------------------------------------------

_GROUP_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "group_a_geometry",
        re.compile(r"geometry_encoder\.encode\.\d+\.cross_attn_image$"),
    ),
    (
        "group_a_encoder",
        re.compile(r"transformer\.encoder\.layers\.\d+\.cross_attn_image$"),
    ),
    (
        "group_b_decoder",
        re.compile(r"transformer\.decoder\.layers\.\d+\.cross_attn$"),
    ),
]

_LAYER_INDEX_RE = re.compile(r"\.(\d+)\.")


def _classify_module(name: str) -> str | None:
    """Return group name for *name*, or ``None`` if it is not a target."""
    for group, pattern in _GROUP_PATTERNS:
        if pattern.search(name):
            return group
    return None


def _parse_layer_index(name: str) -> int | None:
    """Extract the numeric layer index from a module path."""
    matches = _LAYER_INDEX_RE.findall(name)
    return int(matches[-1]) if matches else None


def _is_mha_module(mod: nn.Module) -> bool:
    """Check if *mod* is an MHA module (nn.MultiheadAttention or SAM3 custom)."""
    if isinstance(mod, nn.MultiheadAttention):
        return True
    # SAM3's custom MultiheadAttention (sam3.model.model_misc.MultiheadAttention)
    # has the same attributes but doesn't inherit nn.MHA.
    return (
        hasattr(mod, "in_proj_weight")
        and hasattr(mod, "num_heads")
        and hasattr(mod, "embed_dim")
        and hasattr(mod, "out_proj")
    )


# ---------------------------------------------------------------------------
# Captured record
# ---------------------------------------------------------------------------


@dataclass
class CapturedMHA:
    """One captured MultiheadAttention forward pass."""

    module_name: str
    group: str
    layer_index: int | None
    module: nn.Module
    batch_first: bool

    # The *actual* output tensor — must be on the autograd graph.
    actual_output: torch.Tensor

    # Reconstructed internals (detached, possibly on CPU).
    reconstruction: MHAReconstruction | None = None
    valid_reconstruction: bool = False

    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AttributionCapture
# ---------------------------------------------------------------------------


class AttributionCapture:
    """Register hooks, capture MHA data, reconstruct internals."""

    def __init__(
        self,
        *,
        reconstruction_device: str | torch.device = "cpu",
        validate_reconstruction: bool = False,
        reconstruction_error_threshold: float = 1e-2,
    ) -> None:
        self._recon_device = reconstruction_device
        self._validate = validate_reconstruction
        self._error_threshold = reconstruction_error_threshold

        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._captured: list[dict[str, Any]] = []  # raw hook captures
        self._target_modules: list[tuple[str, str, int | None, nn.MultiheadAttention]] = []

    # ----- discovery --------------------------------------------------------

    @staticmethod
    def list_target_modules(
        model: nn.Module,
    ) -> list[dict[str, Any]]:
        """Discover all target cross-attention modules in *model*.

        Returns a list of dicts with keys ``name``, ``group``,
        ``layer_index``, ``module``.
        """
        results: list[dict[str, Any]] = []
        for name, mod in model.named_modules():
            group = _classify_module(name)
            if group is None:
                continue
            # Accept nn.MultiheadAttention or SAM3's custom MultiheadAttention
            # (sam3.model.model_misc.MultiheadAttention doesn't inherit nn.MHA)
            if not _is_mha_module(mod):
                continue
            results.append({
                "name": name,
                "group": group,
                "layer_index": _parse_layer_index(name),
                "module": mod,
            })
        return results

    # ----- hook registration ------------------------------------------------

    def register(self, model: nn.Module) -> int:
        """Register forward hooks on all target modules.

        Returns the number of modules hooked.
        """
        targets = self.list_target_modules(model)
        for info in targets:
            name = info["name"]
            group = info["group"]
            layer_idx = info["layer_index"]
            mod = info["module"]
            self._target_modules.append((name, group, layer_idx, mod))

            # We need a closure that captures *name*.
            def _make_hook(mod_name: str):
                def _hook(
                    module: nn.Module,
                    args: tuple,
                    kwargs: dict,
                    output: Any,
                ) -> None:
                    self._captured.append({
                        "module_name": mod_name,
                        "inputs": args,
                        "kwargs": kwargs,
                        "output": output,
                    })
                return _hook

            h = mod.register_forward_hook(_make_hook(name), with_kwargs=True)
            self._hooks.append(h)
        return len(targets)

    # ----- retain grad ------------------------------------------------------

    def retain_grads(self) -> None:
        """Call ``retain_grad()`` on captured output tensors.

        Must be called *after* the forward pass but *before* backward.
        """
        for rec in self._captured:
            out = rec["output"]
            attn_out = out[0] if isinstance(out, (tuple, list)) else out
            if attn_out.requires_grad:
                attn_out.retain_grad()
            else:
                rec.setdefault("warnings", []).append(
                    f"[{rec['module_name']}] output does not require grad. "
                    "The forward path may be under no_grad or inference_mode."
                )

    # ----- build records ----------------------------------------------------

    def build_records(self) -> list[CapturedMHA]:
        """Build ``CapturedMHA`` records from raw hook captures.

        This performs MHA reconstruction and validation.
        Should be called *after* backward.
        """
        records: list[CapturedMHA] = []
        module_lookup = {name: (group, layer_idx, mod) for name, group, layer_idx, mod in self._target_modules}

        for cap in self._captured:
            mod_name = cap["module_name"]
            group, layer_idx, mod = module_lookup[mod_name]
            batch_first = getattr(mod, "batch_first", False)

            out = cap["output"]
            attn_out = out[0] if isinstance(out, (tuple, list)) else out

            warns = list(cap.get("warnings", []))
            recon = None
            valid = False

            try:
                recon = reconstruct_mha(
                    module=mod,
                    inputs=cap["inputs"],
                    kwargs=cap["kwargs"],
                    output=cap["output"],
                    module_name=mod_name,
                    group=group,
                    layer_index=layer_idx,
                    device=self._recon_device,
                )
                valid = recon.max_abs_reconstruction_error < self._error_threshold
                if not valid:
                    warns.append(
                        f"Reconstruction error too large: "
                        f"max={recon.max_abs_reconstruction_error:.4e}, "
                        f"threshold={self._error_threshold:.1e}"
                    )
                warns.extend(recon.warnings)
            except Exception as exc:
                warns.append(f"Reconstruction failed: {exc}")

            records.append(CapturedMHA(
                module_name=mod_name,
                group=group,
                layer_index=layer_idx,
                module=mod,
                batch_first=batch_first,
                actual_output=attn_out,
                reconstruction=recon,
                valid_reconstruction=valid,
                warnings=warns,
            ))

        return records

    # ----- helpers ----------------------------------------------------------

    def records_by_group(self, records: list[CapturedMHA] | None = None) -> dict[str, list[CapturedMHA]]:
        """Return records grouped by group name.

        If *records* is ``None``, calls :meth:`build_records` first.
        """
        if records is None:
            records = self.build_records()
        groups: dict[str, list[CapturedMHA]] = {}
        for r in records:
            groups.setdefault(r.group, []).append(r)
        # Sort each group by layer index.
        for g in groups.values():
            g.sort(key=lambda r: (r.layer_index or 0))
        return groups

    def summarize(self, records: list[CapturedMHA] | None = None) -> str:
        """Return a human-readable summary of captured modules."""
        grouped = self.records_by_group(records)
        lines: list[str] = []
        for group, recs in sorted(grouped.items()):
            valid = sum(1 for r in recs if r.valid_reconstruction)
            lines.append(f"  {group:30s}: {valid}/{len(recs)} valid")
            for r in recs:
                err = r.reconstruction.max_abs_reconstruction_error if r.reconstruction else float("nan")
                status = "✓" if r.valid_reconstruction else "✗"
                lines.append(
                    f"    {status} {r.module_name}  (layer={r.layer_index}, max_err={err:.2e})"
                )
                for w in r.warnings:
                    lines.append(f"      WARN: {w}")
        return "\n".join(lines)

    def remove(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
