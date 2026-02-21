"""Tests for pixel manipulation MCP tools."""

import numpy as np
import pytest

from kuavi.context import _decode_frame, _encode_frame
from kuavi.mcp_server import (
    kuavi_blend_frames,
    kuavi_crop_frame,
    kuavi_diff_frames,
    kuavi_frame_info,
    kuavi_threshold_frame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(color, h=100, w=200):
    """Create a solid-color BGR image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _half_bw(h=100, w=200):
    """Left half black, right half white."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, w // 2 :] = 255
    return img


def _gradient(h=100, w=200):
    """Horizontal gradient from 0 to 255 (grayscale)."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.tile(row, (h, 1))
    return np.stack([img, img, img], axis=-1)


# ---------------------------------------------------------------------------
# _decode_frame round-trip
# ---------------------------------------------------------------------------


class TestDecodeFrame:
    def test_round_trip_jpg(self):
        original = _solid((128, 64, 32))
        encoded = _encode_frame(original, format=".jpg")
        decoded = _decode_frame(encoded)
        assert decoded.shape == original.shape
        # JPEG is lossy, so allow some tolerance
        assert np.allclose(decoded, original, atol=10)

    def test_round_trip_png(self):
        original = _solid((128, 64, 32))
        encoded = _encode_frame(original, format=".png")
        decoded = _decode_frame(encoded)
        # PNG is lossless
        np.testing.assert_array_equal(decoded, original)

    def test_invalid_data(self):
        with pytest.raises(ValueError, match="Failed to decode"):
            _decode_frame({"data": "bm90YW5pbWFnZQ=="})  # base64 of "notanimage"


# ---------------------------------------------------------------------------
# kuavi_crop_frame
# ---------------------------------------------------------------------------


class TestCropFrame:
    def test_crop_center(self):
        img = _solid((100, 200, 50), h=100, w=200)
        encoded = _encode_frame(img, format=".png")
        result = kuavi_crop_frame(encoded, 0.25, 0.25, 0.75, 0.75)
        assert "image" in result
        assert "crop" in result
        assert result["crop"]["width"] == 100  # 50% of 200
        assert result["crop"]["height"] == 50  # 50% of 100

    def test_crop_full(self):
        img = _solid((255, 0, 0), h=80, w=120)
        encoded = _encode_frame(img, format=".png")
        result = kuavi_crop_frame(encoded, 0.0, 0.0, 1.0, 1.0)
        assert result["crop"]["width"] == 120
        assert result["crop"]["height"] == 80


# ---------------------------------------------------------------------------
# kuavi_diff_frames
# ---------------------------------------------------------------------------


class TestDiffFrames:
    def test_identical_images(self):
        img = _solid((100, 150, 200))
        enc = _encode_frame(img, format=".png")
        result = kuavi_diff_frames(enc, enc)
        assert result["mean_diff"] == 0.0
        assert result["max_diff"] == 0
        assert result["changed_pct"] == 0.0

    def test_different_images(self):
        black = _encode_frame(_solid((0, 0, 0)), format=".png")
        white = _encode_frame(_solid((255, 255, 255)), format=".png")
        result = kuavi_diff_frames(black, white)
        assert result["mean_diff"] == 255.0
        assert result["max_diff"] == 255
        assert result["changed_pct"] == 100.0


# ---------------------------------------------------------------------------
# kuavi_blend_frames
# ---------------------------------------------------------------------------


class TestBlendFrames:
    def test_single_image(self):
        img = _solid((100, 100, 100))
        enc = _encode_frame(img, format=".png")
        result = kuavi_blend_frames([enc])
        assert result["frame_count"] == 1
        decoded = _decode_frame(result["image"])
        # Single image blend should be ~identity (JPEG compression may shift slightly)
        assert np.allclose(decoded, img, atol=5)

    def test_blend_black_white(self):
        black = _encode_frame(_solid((0, 0, 0)), format=".png")
        white = _encode_frame(_solid((255, 255, 255)), format=".png")
        result = kuavi_blend_frames([black, white])
        assert result["frame_count"] == 2
        decoded = _decode_frame(result["image"])
        # Average of 0 and 255 should be ~127
        assert abs(float(decoded.mean()) - 127.5) < 5

    def test_empty_list(self):
        result = kuavi_blend_frames([])
        assert "error" in result


# ---------------------------------------------------------------------------
# kuavi_threshold_frame
# ---------------------------------------------------------------------------


class TestThresholdFrame:
    def test_half_bw(self):
        img = _half_bw()
        enc = _encode_frame(img, format=".png")
        result = kuavi_threshold_frame(enc, value=128)
        assert "white_pct" in result
        # Right half is white (255 > 128), so ~50% white
        assert abs(result["white_pct"] - 50.0) < 2.0
        assert result["contour_count"] >= 1

    def test_all_black(self):
        img = _solid((0, 0, 0))
        enc = _encode_frame(img, format=".png")
        result = kuavi_threshold_frame(enc, value=128)
        assert result["white_pct"] == 0.0
        assert result["contour_count"] == 0

    def test_all_white(self):
        img = _solid((255, 255, 255))
        enc = _encode_frame(img, format=".png")
        result = kuavi_threshold_frame(enc, value=128)
        assert result["white_pct"] == 100.0

    def test_invert(self):
        img = _solid((255, 255, 255))
        enc = _encode_frame(img, format=".png")
        result = kuavi_threshold_frame(enc, value=128, invert=True)
        # Invert: pixels > 128 become black, so 0% white
        assert result["white_pct"] == 0.0


# ---------------------------------------------------------------------------
# kuavi_frame_info
# ---------------------------------------------------------------------------


class TestFrameInfo:
    def test_solid_color(self):
        img = _solid((50, 100, 200), h=60, w=80)
        enc = _encode_frame(img, format=".png")
        result = kuavi_frame_info(enc)
        assert result["width"] == 80
        assert result["height"] == 60
        assert result["channels"] == 3
        assert result["color_means"]["b"] == 50.0
        assert result["color_means"]["g"] == 100.0
        assert result["color_means"]["r"] == 200.0

    def test_brightness_stats(self):
        img = _solid((0, 0, 0), h=50, w=50)
        enc = _encode_frame(img, format=".png")
        result = kuavi_frame_info(enc)
        assert result["brightness"]["mean"] == 0.0
        assert result["brightness"]["std"] == 0.0
        assert result["brightness"]["min"] == 0
        assert result["brightness"]["max"] == 0

    def test_gradient_stats(self):
        img = _gradient(h=1, w=256)
        enc = _encode_frame(img, format=".png")
        result = kuavi_frame_info(enc)
        assert result["width"] == 256
        assert result["brightness"]["min"] == 0
        assert result["brightness"]["max"] == 255
        assert abs(result["brightness"]["mean"] - 127.5) < 1.0
