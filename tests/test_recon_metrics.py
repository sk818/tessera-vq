"""Tests for the anchor-free L2 reconstruction metric + large-tile RVQ (WS-1).

Synthetic tiles only. Metrics are cross-checked against hand-computable limits
(perfect reconstruction, mean-vector baseline) and RVQ is checked to beat its
own stage-1 reconstruction.
"""

from __future__ import annotations

import numpy as np
from blockwise_kmeans import quantize_tile_large

from tessera_vq.metrics import (
    aggregate_reconstruction_metrics,
    reconstruction_metrics,
)
from tessera_vq.rvq_large import rvq_reconstruct_large


def _clustered_tile(h: int, w: int, dim: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, dim)).astype(np.float32) * 10.0
    labels = rng.integers(0, k, size=(h, w))
    return (centres[labels] + 0.3 * rng.standard_normal((h, w, dim))).astype(np.float32)


def test_perfect_reconstruction_is_zero_error_unit_r2() -> None:
    """recon == orig -> zero relative error and R2 == 1."""
    tile = _clustered_tile(32, 32, 16, k=5, seed=0)
    m = reconstruction_metrics(tile, tile)
    assert m["rel_l2_mean"] == 0.0
    assert m["rel_l2_p99"] == 0.0
    assert abs(m["r2"] - 1.0) < 1e-12
    assert m["n_px"] == 32 * 32


def test_mean_vector_baseline_has_r2_near_zero() -> None:
    """Predicting the per-dim mean for every pixel gives R2 ~ 0 (no variance explained)."""
    tile = _clustered_tile(40, 40, 24, k=6, seed=1)
    flat = tile.reshape(-1, 24)
    recon = np.broadcast_to(flat.mean(axis=0), flat.shape).reshape(tile.shape)
    m = reconstruction_metrics(tile, recon.astype(np.float32))
    assert abs(m["r2"]) < 1e-6


def test_relative_error_is_scale_free() -> None:
    """Scaling orig and recon together leaves the relative-L2 metrics unchanged."""
    tile = _clustered_tile(24, 24, 16, k=4, seed=2)
    cb, idx = quantize_tile_large(tile, k=4, seed=42)
    recon = cb[idx]
    m1 = reconstruction_metrics(tile, recon)
    scale = np.float32(1000.0)
    m2 = reconstruction_metrics(tile * scale, recon * scale)
    for key in ("rel_l2_mean", "rel_l2_p50", "rel_l2_p90", "rel_l2_p99", "r2"):
        assert abs(m1[key] - m2[key]) < 1e-6


def test_empty_input_returns_zeroed_metrics() -> None:
    """No pixels -> all-zero metric dict, no crash."""
    empty = np.zeros((0, 16), dtype=np.float32)
    m = reconstruction_metrics(empty, empty)
    assert m["n_px"] == 0.0
    assert m["r2"] == 0.0


def test_rvq_beats_stage1_alone() -> None:
    """Two-stage RVQ has lower residual energy (higher R2) than stage 1 alone."""
    tile = _clustered_tile(64, 64, 32, k=8, seed=3)
    cb1, idx1 = quantize_tile_large(tile, k=8, seed=42)
    stage1_r2 = reconstruction_metrics(tile, cb1[idx1])["r2"]
    res = rvq_reconstruct_large(tile, k1=8, k2=8, seed=42)
    rvq_r2 = reconstruction_metrics(tile, res.recon)["r2"]
    assert rvq_r2 >= stage1_r2
    assert res.recon.shape == tile.shape
    assert res.indices1.shape == (64, 64)
    assert res.indices2.shape == (64, 64)


def test_rvq_shapes_and_codebook_sizes() -> None:
    """Codebooks are (k_eff, C); k_eff caps at the number of distinct support points."""
    tile = _clustered_tile(48, 48, 16, k=10, seed=4)
    res = rvq_reconstruct_large(tile, k1=16, k2=32, seed=7)
    assert res.codebook1.shape == (16, 16)
    assert res.codebook2.shape == (32, 16)
    assert int(res.indices1.max()) < 16
    assert int(res.indices2.max()) < 32


def test_aggregate_reports_mean_sd_and_counts() -> None:
    """Aggregation across tiles yields mean/sd per key and correct counts."""
    tiles = [_clustered_tile(32, 32, 16, k=5, seed=s) for s in (10, 11, 12)]
    per_tile = []
    for t in tiles:
        cb, idx = quantize_tile_large(t, k=5, seed=42)
        per_tile.append(reconstruction_metrics(t, cb[idx]))
    agg = aggregate_reconstruction_metrics(per_tile)
    assert agg["n_tiles"] == 3.0
    assert agg["n_px"] == 3 * 32 * 32
    assert "rel_l2_mean_mean" in agg and "rel_l2_mean_sd" in agg
    assert agg["r2_mean"] <= 1.0


def test_aggregate_empty_is_zeroed() -> None:
    """No tiles -> degenerate aggregate, no crash."""
    agg = aggregate_reconstruction_metrics([])
    assert agg["n_tiles"] == 0.0
    assert agg["n_px"] == 0.0
