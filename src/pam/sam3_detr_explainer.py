"""DETR-style relevancy propagation for SAM3's MaskDecoder.

Adapts the Transformer-MM-Explainability library's Generator (from
external/Transformer-MM-Explainability/DETR/modules/ExplanationGenerator.py)
to SAM3's TwoWayTransformer / MaskDecoder architecture.

The library tracks three relevancy matrices through the encoder-decoder:

    R_i_i  [HW × HW]        image-token self-attention relevancy
    R_q_q  [T × T]           query/token self-attention relevancy
    R_q_i  [T × HW]          token→image cross-attention relevancy  ← OUTPUT

For SAM3's MaskDecoder there is no separate ViT encoder, so R_i_i starts as
identity and is never updated.  The TwoWayTransformer has per-layer:
  (1) token self-attention  → updates R_q_q
  (2) token→image cross-attention → updates R_q_i   (via rule 10)
  (3) image→token cross-attention → ignored (updates image side, not query side)
plus a final token→image attention layer at the top level.

Backward signal: sum of sigmoid(mask_logits) over pixels predicted positive
for the chosen mask slot — exactly the "mask logit sum" strategy.

Architecture note
-----------------
SAM3's Attention.forward uses F.scaled_dot_product_attention which DOES NOT
return attention weight matrices.  To capture gradients w.r.t. attention
weights we temporarily replace each Attention.forward with a version that
computes explicit softmax(QK^T/√d) and registers a gradient hook on the
resulting weight tensor.  This is done via _InstrumentedAttention, a context
manager that patches and restores forward methods in-place.
"""

from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup: make the DETR explainability math functions importable
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DETR_MODULES = _REPO_ROOT / "external" / "Transformer-MM-Explainability" / "DETR" / "modules"
if str(_DETR_MODULES) not in sys.path:
    sys.path.insert(0, str(_DETR_MODULES))

from ExplanationGenerator import (  # type: ignore[import]
    avg_heads,
    apply_mm_attention_rules,
    apply_self_attention_rules,
    compute_rollout_attention,
)


# ---------------------------------------------------------------------------
# _InstrumentedAttention
# ---------------------------------------------------------------------------

class _InstrumentedAttention:
    """Context manager that temporarily replaces SAM3 Attention.forward with
    a version that exposes softmax attention weights and their gradients.

    SAM3's Attention class uses F.scaled_dot_product_attention internally,
    which fuses softmax + matmul and never materialises the weight matrix.
    We need explicit weights to apply avg_heads(cam, grad).

    Usage::

        with _InstrumentedAttention(list_of_attention_modules) as instrumented:
            output = model(...)
            # instrumented[i].attn  → softmax weights [B, H, T_q, T_kv]
            # instrumented[i].grad  → d(loss)/d(attn)  (after backward)
    """

    def __init__(self, modules: List[nn.Module]) -> None:
        self._modules = modules
        self._originals: Dict[int, object] = {}
        # After __enter__, this list holds one _Record per module.
        self.records: List[_InstrumentedAttention._Record] = []

    class _Record:
        """Holds attn weights and gradient for one Attention module."""
        __slots__ = ("attn", "grad")

        def __init__(self) -> None:
            self.attn: Optional[torch.Tensor] = None   # [B, H, T_q, T_kv]
            self.grad: Optional[torch.Tensor] = None   # [B, H, T_q, T_kv]

    def __enter__(self) -> "List[_InstrumentedAttention._Record]":
        self.records = [self._Record() for _ in self._modules]

        for idx, mod in enumerate(self._modules):
            record = self.records[idx]
            # Save the original forward.
            self._originals[idx] = mod.forward

            # Build a replacement forward that uses explicit softmax.
            def _make_forward(m: nn.Module, rec: "_InstrumentedAttention._Record"):
                def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                             num_k_exclude_rope: int = 0) -> torch.Tensor:
                    # Project
                    q = m.q_proj(q)
                    k = m.k_proj(k)
                    v = m.v_proj(v)

                    # Separate heads  →  [B, H, T, D_head]
                    q = m._separate_heads(q, m.num_heads)
                    k = m._separate_heads(k, m.num_heads)
                    v = m._separate_heads(v, m.num_heads)

                    head_dim = q.shape[-1]
                    scale = math.sqrt(head_dim)

                    # Explicit softmax (so autograd can see the weight tensor)
                    scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B,H,T_q,T_k]
                    attn_weights = F.softmax(scores, dim=-1)                # [B,H,T_q,T_k]

                    # Store and register grad hook
                    rec.attn = attn_weights
                    attn_weights.register_hook(lambda g: rec.__setattr__("grad", g))

                    out = torch.matmul(attn_weights, v)         # [B,H,T_q,D_head]
                    out = m._recombine_heads(out)                # [B,T_q,C]
                    out = m.out_proj(out)
                    return out

                return _forward

            mod.forward = _make_forward(mod, record)  # type: ignore[method-assign]

        return self.records

    def __exit__(self, *args):
        for idx, mod in enumerate(self._modules):
            mod.forward = self._originals[idx]  # type: ignore[method-assign]
        self._originals.clear()


# ---------------------------------------------------------------------------
# SAM3MaskExplainer
# ---------------------------------------------------------------------------

class SAM3MaskExplainer:
    """DETR-style relevancy explainer for SAM3's MaskDecoder.

    Instantiate once per MaskDecoder, then call an explain_* method for each
    forward pass you want to explain.

    Parameters
    ----------
    mask_decoder : nn.Module
        A SAM3 MaskDecoder instance (``sam3.sam.mask_decoder.MaskDecoder``).

    Example
    -------
    ::

        explainer = SAM3MaskExplainer(sam_model.sam_mask_decoder)
        heatmap = explainer.explain_ours(
            image_embeddings, image_pe,
            sparse_prompt_embeddings, dense_prompt_embeddings,
            target_mask_token=0,
        )
        # heatmap: [H_feat, W_feat]  (spatial relevancy map)
    """

    def __init__(self, mask_decoder: nn.Module) -> None:
        self.mask_decoder = mask_decoder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_attention_modules(self) -> Tuple[
        List[nn.Module],  # per-layer self_attn
        List[nn.Module],  # per-layer cross_attn_token_to_image
        nn.Module,        # final_attn_token_to_image
    ]:
        """Return all Attention modules in the TwoWayTransformer."""
        transformer = self.mask_decoder.transformer
        self_attn_mods = [layer.self_attn for layer in transformer.layers]
        cross_attn_mods = [layer.cross_attn_token_to_image for layer in transformer.layers]
        final_attn = transformer.final_attn_token_to_image
        return self_attn_mods, cross_attn_mods, final_attn

    def _build_backward_signal(
        self,
        masks: torch.Tensor,
        target_mask_token: int,
    ) -> torch.Tensor:
        """Sum sigmoid(mask_logits) over predicted-positive pixels.

        This gives a scalar whose gradient flows back through the mask
        token responsible for the chosen mask slot.

        Parameters
        ----------
        masks : Tensor  [B, N_masks, H, W]   (raw logits)
        target_mask_token : int
        """
        logits = masks[:, target_mask_token]          # [B, H, W]
        probs = torch.sigmoid(logits)
        # Weight by confidence: sum over pixels predicted positive.
        return (probs * (probs > 0.5).float()).sum()

    def _forward_with_instrumentation(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        target_mask_token: int,
        multimask_output: bool,
        repeat_image: bool,
    ) -> Tuple[
        torch.Tensor,                              # scalar backward signal
        "List[_InstrumentedAttention._Record]",    # self-attn records
        "List[_InstrumentedAttention._Record]",    # cross-attn records
        "_InstrumentedAttention._Record",          # final-attn record
        Tuple[int, int],                           # (H, W) feature map dims
        int,                                       # num_tokens
    ]:
        """Run forward pass under instrumented attention, return records."""
        self_attn_mods, cross_attn_mods, final_attn_mod = self._get_attention_modules()
        all_mods = self_attn_mods + cross_attn_mods + [final_attn_mod]
        n_layers = len(self_attn_mods)
        n_cross  = len(cross_attn_mods)

        with _InstrumentedAttention(all_mods) as records:
            # records[0 .. n_layers-1]          → self-attn per layer
            # records[n_layers .. n_layers+n_cross-1] → cross-attn per layer
            # records[-1]                        → final cross-attn
            self_records  = records[:n_layers]
            cross_records = records[n_layers : n_layers + n_cross]
            final_record  = records[-1]

            masks, _, _, _ = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                multimask_output=multimask_output,
                repeat_image=repeat_image,
            )

            # Resolve H, W from image_embeddings (B, C, H, W)
            H, W = image_embeddings.shape[-2], image_embeddings.shape[-1]

            # Count tokens: output tokens + sparse prompts
            s = 1 if hasattr(self.mask_decoder, "obj_score_token") and self.mask_decoder.pred_obj_scores else 0
            num_output_tokens = s + 1 + self.mask_decoder.num_mask_tokens
            num_tokens = num_output_tokens + sparse_prompt_embeddings.shape[1]

            # Build scalar signal and differentiate
            signal = self._build_backward_signal(masks, target_mask_token)
            self.mask_decoder.zero_grad()
            signal.backward(retain_graph=True)

            # Snapshot records INSIDE context (before forward is restored)
            # The tensors themselves persist after __exit__; we just need
            # to hold references.
            snapshot_self  = [(r.attn, r.grad) for r in self_records]
            snapshot_cross = [(r.attn, r.grad) for r in cross_records]
            snapshot_final = (final_record.attn, final_record.grad)

        return signal, snapshot_self, snapshot_cross, snapshot_final, (H, W), num_tokens

    # ------------------------------------------------------------------
    # Public explain methods
    # ------------------------------------------------------------------

    def explain_ours(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        target_mask_token: int = 0,
        multimask_output: bool = False,
        repeat_image: bool = False,
        normalize_self_attention: bool = True,
        apply_self_in_rule_10: bool = True,
    ) -> torch.Tensor:
        """Full relevancy propagation (Rules 5, 6/7, 10 from the paper).

        This is the primary method — mirrors ``Generator.generate_ours`` with
        ``use_lrp=False`` (gradient × attention, no separate LRP backward).

        Returns
        -------
        Tensor  [H_feat, W_feat]
            Relevancy heatmap for ``target_mask_token`` over the spatial
            feature-map grid.  Reshape / upsample to overlay on the input image.
        """
        _, snapshot_self, snapshot_cross, snapshot_final, (H, W), num_tokens = (
            self._forward_with_instrumentation(
                image_embeddings, image_pe,
                sparse_prompt_embeddings, dense_prompt_embeddings,
                target_mask_token, multimask_output, repeat_image,
            )
        )

        device = image_embeddings.device
        HW = H * W

        # Relevancy matrices (same as Generator.generate_ours)
        R_q_q = torch.eye(num_tokens, num_tokens, device=device)
        R_q_i = torch.zeros(num_tokens, HW,       device=device)
        # No image encoder → R_i_i stays identity throughout
        R_i_i = torch.eye(HW, HW, device=device)

        for i, ((sa_attn, sa_grad), (ca_attn, ca_grad)) in enumerate(
            zip(snapshot_self, snapshot_cross)
        ):
            # ── token self-attention (rules 6 + 7) ──────────────────────────
            if sa_attn is not None and sa_grad is not None:
                # avg over batch dim: take B=0, then reshape heads
                cam_ss = avg_heads(
                    sa_attn[0].unsqueeze(0),   # [1, H, T, T]
                    sa_grad[0].unsqueeze(0),
                )  # [T, T]
                R_q_q_add, R_q_i_add = apply_self_attention_rules(R_q_q, R_q_i, cam_ss)
                R_q_q = R_q_q + R_q_q_add
                R_q_i = R_q_i + R_q_i_add

            # ── token→image cross-attention (rule 10) ────────────────────────
            if ca_attn is not None and ca_grad is not None:
                cam_qi = avg_heads(
                    ca_attn[0].unsqueeze(0),   # [1, H, T, HW]
                    ca_grad[0].unsqueeze(0),
                )  # [T, HW]
                R_q_i = R_q_i + apply_mm_attention_rules(
                    R_q_q, R_i_i, cam_qi,
                    apply_normalization=normalize_self_attention,
                    apply_self_in_rule_10=apply_self_in_rule_10,
                )

        # ── final cross-attention layer ──────────────────────────────────────
        fa_attn, fa_grad = snapshot_final
        if fa_attn is not None and fa_grad is not None:
            cam_qi = avg_heads(
                fa_attn[0].unsqueeze(0),
                fa_grad[0].unsqueeze(0),
            )
            R_q_i = R_q_i + apply_mm_attention_rules(
                R_q_q, R_i_i, cam_qi,
                apply_normalization=normalize_self_attention,
                apply_self_in_rule_10=apply_self_in_rule_10,
            )

        # Extract the heatmap for the target token slot.
        # token layout: [obj_score?] [iou_token] [mask_tokens...] [sparse_prompts...]
        # target_mask_token indexes into mask_tokens (0-based).
        s = 1 if (hasattr(self.mask_decoder, "pred_obj_scores")
                  and self.mask_decoder.pred_obj_scores) else 0
        token_idx = s + 1 + target_mask_token  # offset past obj_score + iou tokens

        heatmap = R_q_i[token_idx].detach()    # [HW]
        return heatmap.reshape(H, W)

    def explain_raw_attn(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        target_mask_token: int = 0,
        multimask_output: bool = False,
        repeat_image: bool = False,
    ) -> torch.Tensor:
        """Raw cross-attention from the final token→image attention layer only.

        No gradient weighting, no aggregation across layers.  Fast, but only
        reflects the last attention step.

        Returns
        -------
        Tensor  [H_feat, W_feat]
        """
        _, _, _, snapshot_final, (H, W), _ = self._forward_with_instrumentation(
            image_embeddings, image_pe,
            sparse_prompt_embeddings, dense_prompt_embeddings,
            target_mask_token, multimask_output, repeat_image,
        )

        fa_attn, _ = snapshot_final
        if fa_attn is None:
            raise RuntimeError("Final attention was not captured.")

        # fa_attn: [B, H_heads, T_tokens, HW]
        # Mean over heads, take batch 0
        cam = fa_attn[0].mean(dim=0)   # [T_tokens, HW]

        s = 1 if (hasattr(self.mask_decoder, "pred_obj_scores")
                  and self.mask_decoder.pred_obj_scores) else 0
        token_idx = s + 1 + target_mask_token

        heatmap = cam[token_idx].detach()
        return heatmap.reshape(H, W)

    def explain_attn_gradcam(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        target_mask_token: int = 0,
        multimask_output: bool = False,
        repeat_image: bool = False,
    ) -> torch.Tensor:
        """GradCAM on the final token→image attention layer.

        Weights each attention head's map by the spatial mean of its gradient,
        then sums (clamped to non-negative).

        Returns
        -------
        Tensor  [H_feat, W_feat]
        """
        _, _, _, snapshot_final, (H, W), _ = self._forward_with_instrumentation(
            image_embeddings, image_pe,
            sparse_prompt_embeddings, dense_prompt_embeddings,
            target_mask_token, multimask_output, repeat_image,
        )

        fa_attn, fa_grad = snapshot_final
        if fa_attn is None or fa_grad is None:
            raise RuntimeError("Final attention or gradients were not captured.")

        # GradCAM: weight each head by mean gradient then clamp
        # Shapes: [B, H_heads, T, HW]
        cam  = fa_attn[0]   # [H_heads, T, HW]
        grad = fa_grad[0]   # [H_heads, T, HW]

        # Global-average the gradient over (T, HW) per head → [H_heads, 1, 1]
        weights = grad.mean(dim=[-2, -1], keepdim=True)
        gcam = (cam * weights).sum(dim=0).clamp(min=0)  # [T, HW]

        s = 1 if (hasattr(self.mask_decoder, "pred_obj_scores")
                  and self.mask_decoder.pred_obj_scores) else 0
        token_idx = s + 1 + target_mask_token

        heatmap = gcam[token_idx].detach()
        return heatmap.reshape(H, W)

    def explain_rollout(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        target_mask_token: int = 0,
        multimask_output: bool = False,
        repeat_image: bool = False,
    ) -> torch.Tensor:
        """Attention rollout (no gradient).

        Composes query self-attention across layers via rollout, then
        multiplies by the final cross-attention map.

        This method does NOT require a backward pass, making it faster and
        usable during inference-only (no_grad) contexts — though it then
        provides a gradient-free approximation.

        Returns
        -------
        Tensor  [H_feat, W_feat]
        """
        self_attn_mods, cross_attn_mods, final_attn_mod = self._get_attention_modules()
        all_mods = self_attn_mods + cross_attn_mods + [final_attn_mod]
        n_layers = len(self_attn_mods)

        with _InstrumentedAttention(all_mods) as records:
            self_records = records[:n_layers]
            final_record = records[-1]

            with torch.no_grad():
                _, _, _, _ = self.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_prompt_embeddings,
                    dense_prompt_embeddings=dense_prompt_embeddings,
                    multimask_output=multimask_output,
                    repeat_image=repeat_image,
                )

            H, W = image_embeddings.shape[-2], image_embeddings.shape[-1]

            # Snapshot inside context
            cams_queries = []
            for r in self_records:
                if r.attn is not None:
                    # mean over heads → [T, T]
                    cams_queries.append(r.attn[0].mean(dim=0).detach())

            fa_attn = final_record.attn

        # Rollout for query self-attention
        if cams_queries:
            R_q_q = compute_rollout_attention(cams_queries)  # [T, T]
        else:
            raise RuntimeError("No self-attention maps were captured.")

        # Final cross-attention map → [T, HW]
        if fa_attn is None:
            raise RuntimeError("Final attention was not captured.")
        cam_qi = fa_attn[0].mean(dim=0).detach()  # [T, HW]

        # R_q_i = R_q_q^T @ cam_qi   (same formula as Generator.generate_rollout)
        R_q_i = torch.matmul(R_q_q.t(), cam_qi)   # [T, HW]

        s = 1 if (hasattr(self.mask_decoder, "pred_obj_scores")
                  and self.mask_decoder.pred_obj_scores) else 0
        token_idx = s + 1 + target_mask_token

        heatmap = R_q_i[token_idx].detach()
        return heatmap.reshape(H, W)
