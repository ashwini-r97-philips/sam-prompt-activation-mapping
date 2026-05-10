"""Tests for spatial-grid inference and filename sanitisation."""

import warnings

import pytest

from pam.visualization import infer_spatial_grid
from pam.tokenization import safe_filename


class TestInferSpatialGrid:
    """Verify spatial grid inference from image-token counts."""

    def test_perfect_square_64(self):
        """64 tokens → 8×8."""
        result = infer_spatial_grid(64)
        assert result == (8, 8)

    def test_perfect_square_5184(self):
        """72*72 = 5184 tokens → 72×72."""
        result = infer_spatial_grid(5184)
        assert result == (72, 72)

    def test_perfect_square_1(self):
        """1 token → 1×1."""
        result = infer_spatial_grid(1)
        assert result == (1, 1)

    def test_non_square_warns(self):
        """Non-square token count should warn and return None."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = infer_spatial_grid(65)
            assert result is None
            assert len(w) == 1
            assert "perfect square" in str(w[0].message).lower()

    def test_metadata_override(self):
        """Metadata with feat_h / feat_w should override heuristic."""
        # 70 is not a perfect square, but metadata says 7×10.
        result = infer_spatial_grid(70, metadata={"feat_h": 7, "feat_w": 10})
        assert result == (7, 10)

    def test_metadata_mismatch_ignored(self):
        """If metadata H*W != num_tokens, fall back to sqrt heuristic."""
        result = infer_spatial_grid(64, metadata={"feat_h": 7, "feat_w": 10})
        # 7*10 = 70 ≠ 64, so should fall back to 8×8.
        assert result == (8, 8)


class TestSafeFilename:
    """Token labels → filesystem-safe strings."""

    def test_cls_token(self):
        assert safe_filename("[CLS]") == "CLS"

    def test_eos_token(self):
        assert safe_filename("[EOS]") == "EOS"

    def test_spaces(self):
        assert safe_filename("yellow school bus") == "yellow_school_bus"

    def test_special_characters(self):
        assert safe_filename("hello/world:test") == "hello_world_test"

    def test_empty_string(self):
        assert safe_filename("") == "unknown"

    def test_consecutive_underscores(self):
        # Hyphens are valid filename chars and kept; only underscores collapse.
        assert safe_filename("a___b___c") == "a_b_c"

    def test_already_safe(self):
        assert safe_filename("token_03") == "token_03"
