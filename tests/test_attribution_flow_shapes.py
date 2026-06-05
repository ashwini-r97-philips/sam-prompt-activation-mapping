"""Tests for attribution-flow shapes — no SAM 3 checkpoint needed."""

import torch
import pytest


class TestEdgeAttributionShapes:
    """Verify edge attribution shape from synthetic tensors."""

    def test_basic_shapes(self):
        """A [1,8,201,64], V [1,8,64,32], G [1,8,201,32] → attr [1,8,201,64]."""
        A = torch.rand(1, 8, 201, 64)
        V = torch.rand(1, 8, 64, 32)
        G = torch.rand(1, 8, 201, 32)

        # value_grad_dot = einsum("bhsd,bhqd->bhqs", V, G)
        value_grad_dot = torch.einsum("bhsd,bhqd->bhqs", V, G)
        assert value_grad_dot.shape == (1, 8, 201, 64)

        edge_attr = A * value_grad_dot
        assert edge_attr.shape == (1, 8, 201, 64)


class TestGroupAPromptMapReshape:
    """Verify Group A encoder attribution → spatial token maps."""

    def test_reshape(self):
        """edge_attr [1,8,5184,33] → per-token maps [8,33,72,72]."""
        edge_attr = torch.rand(1, 8, 5184, 33)
        T = 33
        Gh, Gw = 72, 72

        # Extract batch 0, transpose to [H, T, N_image], reshape spatial.
        pos = edge_attr[0, :, :, :T]  # [H, 5184, T]
        maps = pos.transpose(1, 2).reshape(8, T, Gh, Gw)

        assert maps.shape == (8, 33, 72, 72)

    def test_partial_tokens(self):
        """With num_prompt_tokens=3, output is [8,3,72,72]."""
        edge_attr = torch.rand(1, 8, 5184, 33)
        T = 3
        Gh, Gw = 72, 72

        pos = edge_attr[0, :, :, :T]
        maps = pos.transpose(1, 2).reshape(8, T, Gh, Gw)
        assert maps.shape == (8, 3, 72, 72)


class TestGroupBQueryMapReshape:
    """Verify Group B decoder attribution → selected-query spatial maps."""

    def test_reshape(self):
        """edge_attr [1,8,201,5184], select q=7 → [8,72,72]."""
        edge_attr = torch.rand(1, 8, 201, 5184)
        q = 7
        Gh, Gw = 72, 72

        selected = edge_attr[0, :, q, :]  # [H, 5184]
        maps = selected.reshape(8, Gh, Gw)

        assert maps.shape == (8, 72, 72)

    def test_different_query(self):
        """Any valid query index should work."""
        edge_attr = torch.rand(1, 8, 201, 5184)
        for q in [0, 100, 200]:
            selected = edge_attr[0, :, q, :].reshape(8, 72, 72)
            assert selected.shape == (8, 72, 72)


class TestGeometryMapReshape:
    """Verify Group A geometry attribution → spatial maps."""

    def test_reshape(self):
        """edge_attr [1,8,1,5184] → [8,72,72]."""
        edge_attr = torch.rand(1, 8, 1, 5184)
        Gh, Gw = 72, 72

        maps = edge_attr[0, :, 0, :].reshape(8, Gh, Gw)
        assert maps.shape == (8, 72, 72)
