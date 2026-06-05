"""Tests for local conservation of edge attribution — no SAM 3 needed."""

import torch
import pytest


class TestLocalConservation:
    """Verify that sum_j R_ij = dot(O_i, grad_i)."""

    def test_conservation_holds(self):
        """With random A, V, G the conservation identity must hold exactly."""
        torch.manual_seed(42)
        B, H, Tq, Ts, Dh = 1, 8, 201, 64, 32

        A = torch.softmax(torch.randn(B, H, Tq, Ts), dim=-1)
        V = torch.randn(B, H, Ts, Dh)
        G = torch.randn(B, H, Tq, Dh)

        # Pre-head output: O = A @ V
        pre_out = torch.matmul(A, V)  # [B, H, Tq, Dh]

        # Edge attribution: R = A * (V^T G)  where (V^T G)[q,s] = dot(V[s], G[q])
        value_grad_dot = torch.einsum("bhsd,bhqd->bhqs", V, G)  # [B, H, Tq, Ts]
        edge_attr = A * value_grad_dot

        # Conservation: sum_s R[q,s] == dot(O[q], G[q])
        edge_sum = edge_attr.sum(dim=-1)  # [B, H, Tq]
        head_output_attr = (pre_out * G).sum(dim=-1)  # [B, H, Tq]

        assert torch.allclose(edge_sum, head_output_attr, atol=1e-5), (
            f"Conservation error: max diff = {(edge_sum - head_output_attr).abs().max().item():.2e}"
        )

    def test_conservation_different_sizes(self):
        """Test with encoder-like dimensions: Tq=5184, Ts=33."""
        torch.manual_seed(7)
        B, H, Tq, Ts, Dh = 1, 8, 64, 33, 32  # Smaller for speed

        A = torch.softmax(torch.randn(B, H, Tq, Ts), dim=-1)
        V = torch.randn(B, H, Ts, Dh)
        G = torch.randn(B, H, Tq, Dh)

        pre_out = torch.matmul(A, V)
        value_grad_dot = torch.einsum("bhsd,bhqd->bhqs", V, G)
        edge_attr = A * value_grad_dot

        edge_sum = edge_attr.sum(dim=-1)
        head_output_attr = (pre_out * G).sum(dim=-1)

        assert torch.allclose(edge_sum, head_output_attr, atol=1e-5)

    def test_conservation_with_relu_split(self):
        """Positive + negative should sum back to signed."""
        torch.manual_seed(99)
        B, H, Tq, Ts, Dh = 1, 4, 16, 8, 16

        A = torch.softmax(torch.randn(B, H, Tq, Ts), dim=-1)
        V = torch.randn(B, H, Ts, Dh)
        G = torch.randn(B, H, Tq, Dh)

        value_grad_dot = torch.einsum("bhsd,bhqd->bhqs", V, G)
        edge_attr = A * value_grad_dot
        positive = torch.relu(edge_attr)
        negative = torch.relu(-edge_attr)

        reconstructed = positive - negative
        assert torch.allclose(reconstructed, edge_attr, atol=1e-6)
