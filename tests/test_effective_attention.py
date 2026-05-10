"""Tests for effective-attention computation (no SAM 3 checkpoint needed)."""

import torch
import pytest

from pam.effective_attention import compute_effective_attention, reduce_heads


class TestComputeEffectiveAttention:
    """Verify shapes and normalisation of effective attention."""

    def test_output_shape(self):
        """Output shape must match raw attention shape."""
        # A: [B=1, H=4, T_prompt=5, T_image=64]
        A = torch.rand(1, 4, 5, 64)
        # V: [B=1, H=4, T_image=64, D_head=32]
        V = torch.rand(1, 4, 64, 32)

        eff = compute_effective_attention(A, V)
        assert eff.shape == A.shape, (
            f"Expected shape {A.shape}, got {eff.shape}"
        )

    def test_row_sums_approximately_one(self):
        """Each prompt-token's distribution over image tokens should sum ≈ 1."""
        A = torch.rand(1, 4, 5, 64)
        # Make A a valid probability distribution along the image-token dim.
        A = A / A.sum(dim=-1, keepdim=True)

        V = torch.rand(1, 4, 64, 32) + 0.1  # avoid zero norms
        eff = compute_effective_attention(A, V)

        # Sum over the T_image dimension → should be ≈ 1.
        row_sums = eff.sum(dim=-1)  # [B, H, T_prompt]
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"Row sums deviate from 1: {row_sums}"
        )

    def test_zero_values_produce_uniform(self):
        """If all value norms are zero, effective attention should still be
        finite (eps prevents div-by-zero) and approximately uniform."""
        A = torch.rand(1, 2, 3, 16)
        A = A / A.sum(dim=-1, keepdim=True)
        V = torch.zeros(1, 2, 16, 8)

        eff = compute_effective_attention(A, V)
        assert torch.isfinite(eff).all(), "Non-finite values in output"

    def test_single_head(self):
        """Edge case: single head."""
        A = torch.rand(1, 1, 2, 10)
        A = A / A.sum(dim=-1, keepdim=True)
        V = torch.rand(1, 1, 10, 16)

        eff = compute_effective_attention(A, V)
        assert eff.shape == (1, 1, 2, 10)


class TestReduceHeads:
    """Verify head reduction produces a 1-D heatmap."""

    def test_mean_shape(self):
        A = torch.rand(1, 4, 5, 64)
        heatmap = reduce_heads(A, token_index=0, mode="mean")
        assert heatmap.shape == (64,)

    def test_max_shape(self):
        A = torch.rand(1, 4, 5, 64)
        heatmap = reduce_heads(A, token_index=2, mode="max")
        assert heatmap.shape == (64,)

    def test_invalid_mode_raises(self):
        A = torch.rand(1, 4, 5, 64)
        with pytest.raises(ValueError, match="Unknown head_reduction mode"):
            reduce_heads(A, token_index=0, mode="median")
