"""Tests for tessera_vq.codebook_codec: per-dim uint8 codebook (de)quantization (WS-1)."""

from __future__ import annotations

import numpy as np

from tessera_vq.codebook_codec import (
    dequantize_codebook_uint8,
    quantize_codebook_uint8,
    roundtrip_uint8,
)


def test_roundtrip_within_int8_tolerance() -> None:
    """Dequantized codebook matches the original to within the per-dim quant step."""
    rng = np.random.default_rng(0)
    cb = (rng.standard_normal((5, 64, 128)) * 3.0).astype(np.float32)
    q, lo, hi = quantize_codebook_uint8(cb)
    assert q.dtype == np.uint8 and q.shape == cb.shape
    assert lo.shape == (5, 128) and hi.shape == (5, 128)
    deq = dequantize_codebook_uint8(q, lo, hi)
    assert deq.dtype == np.float32
    # error bounded by half a quant step per (tile, dim): span/255/2
    step = (hi - lo) / 255.0
    assert np.all(np.abs(deq - cb) <= step[:, None, :] / 2 + 1e-5)


def test_constant_dimension_is_exact() -> None:
    """A dimension constant across codes (hi == lo) dequantizes back exactly."""
    cb = np.zeros((2, 4, 8), dtype=np.float32)
    cb[..., 0] = 3.5  # dim 0 constant across all codes
    cb[:, :, 1] = np.arange(4, dtype=np.float32)  # dim 1 varies
    q, lo, hi = quantize_codebook_uint8(cb)
    deq = dequantize_codebook_uint8(q, lo, hi)
    assert np.allclose(deq[..., 0], 3.5)
    assert np.all(q[..., 0] == 0)  # constant dim -> all zero codes


def test_roundtrip_uint8_single_codebook() -> None:
    """The (k, C) round-trip helper matches the original to within the quant step."""
    rng = np.random.default_rng(2)
    cb = (rng.standard_normal((20, 128)) * 5.0).astype(np.float32)
    rt = roundtrip_uint8(cb)
    assert rt.shape == cb.shape and rt.dtype == np.float32
    step = (cb.max(axis=0) - cb.min(axis=0)) / 255.0
    assert np.all(np.abs(rt - cb) <= step / 2 + 1e-5)


def test_empty_stack() -> None:
    """Zero tiles round-trips to empty arrays without error."""
    cb = np.zeros((0, 16, 32), dtype=np.float32)
    q, lo, hi = quantize_codebook_uint8(cb)
    assert q.shape == (0, 16, 32) and lo.shape == (0, 32)
    assert dequantize_codebook_uint8(q, lo, hi).shape == (0, 16, 32)


def test_per_dim_scaling_handles_anisotropy() -> None:
    """Dims with very different scales are each quantized to full uint8 range."""
    rng = np.random.default_rng(1)
    cb = rng.standard_normal((1, 256, 4)).astype(np.float32)
    cb[..., 0] *= 0.01  # tiny-scale dim
    cb[..., 1] *= 100.0  # large-scale dim
    q, lo, hi = quantize_codebook_uint8(cb)
    deq = dequantize_codebook_uint8(q, lo, hi)
    # both dims reconstructed to their own ~0.4% relative resolution, not the global one
    for d in (0, 1):
        rel = np.abs(deq[0, :, d] - cb[0, :, d]).max() / float(np.ptp(cb[0, :, d]))
        assert rel < 0.01
