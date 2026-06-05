"""Tests for MHA reconstruction — no SAM 3 checkpoint needed."""

import torch
import torch.nn as nn
import pytest

from pam.mha_reconstruction import reconstruct_mha


class TestMHAReconstructionBatchFirstTrue:
    """nn.MultiheadAttention with batch_first=True."""

    def test_reconstruction_error_is_small(self):
        torch.manual_seed(42)
        mha = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        mha.eval()

        q = torch.randn(1, 64, 256)
        k = torch.randn(1, 33, 256)
        v = torch.randn(1, 33, 256)

        with torch.no_grad():
            out, _ = mha(q, k, v)

        recon = reconstruct_mha(
            module=mha,
            inputs=(q, k, v),
            kwargs={},
            output=(out, None),
            module_name="test_bf_true",
            group="test",
            device="cpu",
        )

        assert recon.max_abs_reconstruction_error < 1e-4, (
            f"Max error {recon.max_abs_reconstruction_error:.2e} exceeds 1e-4"
        )
        assert recon.mean_abs_reconstruction_error < 1e-5

    def test_shapes(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        mha.eval()

        q = torch.randn(1, 64, 256)
        k = torch.randn(1, 33, 256)
        v = torch.randn(1, 33, 256)

        with torch.no_grad():
            out, _ = mha(q, k, v)

        recon = reconstruct_mha(
            module=mha, inputs=(q, k, v), kwargs={}, output=(out, None),
            module_name="test", group="test", device="cpu",
        )

        assert recon.query.shape == (1, 64, 256)
        assert recon.key.shape == (1, 33, 256)
        assert recon.value.shape == (1, 33, 256)
        assert recon.attn_probs.shape == (1, 8, 64, 33)
        assert recon.value_heads.shape == (1, 8, 33, 32)
        assert recon.pre_out_heads.shape == (1, 8, 64, 32)
        assert recon.pre_out_concat.shape == (1, 64, 256)
        assert recon.actual_output.shape == (1, 64, 256)


class TestMHAReconstructionBatchFirstFalse:
    """nn.MultiheadAttention with batch_first=False (seq-first)."""

    def test_reconstruction_error_is_small(self):
        torch.manual_seed(42)
        mha = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=False)
        mha.eval()

        # Seq-first: [T, B, E]
        q = torch.randn(201, 1, 256)
        k = torch.randn(64, 1, 256)
        v = torch.randn(64, 1, 256)

        with torch.no_grad():
            out, _ = mha(q, k, v)

        recon = reconstruct_mha(
            module=mha,
            inputs=(q, k, v),
            kwargs={},
            output=(out, None),
            module_name="test_bf_false",
            group="test",
            device="cpu",
        )

        assert recon.max_abs_reconstruction_error < 1e-4, (
            f"Max error {recon.max_abs_reconstruction_error:.2e} exceeds 1e-4"
        )

    def test_shapes_normalized_to_batch_first(self):
        torch.manual_seed(0)
        mha = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=False)
        mha.eval()

        q = torch.randn(201, 1, 256)
        k = torch.randn(64, 1, 256)
        v = torch.randn(64, 1, 256)

        with torch.no_grad():
            out, _ = mha(q, k, v)

        recon = reconstruct_mha(
            module=mha, inputs=(q, k, v), kwargs={}, output=(out, None),
            module_name="test", group="test", device="cpu",
        )

        # All shapes should be batch-first.
        assert recon.query.shape == (1, 201, 256)
        assert recon.key.shape == (1, 64, 256)
        assert recon.attn_probs.shape == (1, 8, 201, 64)
        assert recon.pre_out_heads.shape == (1, 8, 201, 32)


class TestMHAReconstructionWithBias:
    """Ensure reconstruction works with bias present."""

    def test_with_bias(self):
        torch.manual_seed(7)
        mha = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True, bias=True)
        mha.eval()

        q = torch.randn(2, 16, 128)
        k = torch.randn(2, 32, 128)
        v = torch.randn(2, 32, 128)

        with torch.no_grad():
            out, _ = mha(q, k, v)

        recon = reconstruct_mha(
            module=mha, inputs=(q, k, v), kwargs={}, output=(out, None),
            module_name="test_bias", group="test", device="cpu",
        )
        assert recon.max_abs_reconstruction_error < 1e-4
