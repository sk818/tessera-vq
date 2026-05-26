"""Tests for tessera_vq.sweep: vectorised k-means and the (t, K, m) sweep."""

import numpy as np
import pytest

from tessera_vq.sweep import (
    fast_quantize_tile,
    quantize_window_for_serving,
    quantize_window_residual_norms,
    reconstruction_quantiles,
    rvq_quantize_tile,
    rvq_quantize_window_for_serving,
    rvq_reconstruct_tile,
    sweep_window,
)


def _three_cluster_tile(h: int, w: int, dim: int, seed: int) -> np.ndarray:
    """Synthetic tile with 3 well-separated cluster centres + small noise."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((3, dim)).astype(np.float32) * 5.0
    labels = rng.integers(0, 3, size=(h, w))
    return (centres[labels] + 0.1 * rng.standard_normal((h, w, dim))).astype(np.float32)


def test_fast_quantize_tile_recovers_three_clusters() -> None:
    """k=3 on a 3-cluster synthetic tile should reconstruct near-perfectly."""
    tile = _three_cluster_tile(32, 32, 128, seed=0)
    centers, idx = fast_quantize_tile(tile, k=3, distance="euclidean", seed=42)
    assert centers.shape == (3, 128)
    assert idx.shape == (32, 32)
    err = float(
        np.mean(np.linalg.norm(tile.reshape(-1, 128) - centers[idx].reshape(-1, 128), axis=1))
    )
    # noise has L2 magnitude ~ 0.1 * sqrt(128) ~= 1.13
    assert err < 1.5


def test_fast_quantize_tile_cosine_path() -> None:
    """Cosine distance should also produce a valid k-clustering."""
    tile = _three_cluster_tile(32, 32, 128, seed=1)
    centers, idx = fast_quantize_tile(tile, k=4, distance="cosine", seed=42)
    assert centers.shape == (4, 128)
    assert idx.shape == (32, 32)
    assert int(idx.min()) >= 0
    assert int(idx.max()) < 4


def test_reconstruction_quantiles_zero_when_identical() -> None:
    """Original == reconstruction should give all-zero distance quantiles."""
    tile = _three_cluster_tile(16, 16, 32, seed=2)
    q = reconstruction_quantiles(tile, tile)
    for p in (10, 50, 90, 99):
        assert abs(q[f"cos_p{p}"]) < 1e-10
        assert abs(q[f"l2_p{p}"]) < 1e-10


def test_sweep_window_structure() -> None:
    """sweep_window returns one row per (t, K, m, subtile) with expected keys."""
    window = _three_cluster_tile(64, 64, 128, seed=3)
    rows = sweep_window(window, ts=[32], ks=[4], ms=["euclidean"], seed=42)
    assert len(rows) >= 1
    expected = {"t", "subtile", "k", "m", "n_pixels", "cos_p50", "l2_p50"}
    assert expected.issubset(rows[0].keys())


def test_quantize_window_for_serving_shapes_and_dtypes() -> None:
    """Tiling shapes are (n, k, 128) f32 / (n, t, t) uint8 / (n, 2) i32 for k<=256."""
    window = _three_cluster_tile(64, 64, 128, seed=4)
    cbs, idxs, pos = quantize_window_for_serving(window, t=32, k=4, m="euclidean", seed=42)
    assert cbs.shape == (4, 4, 128)  # 64/32 = 2 -> 2x2 = 4 tiles, k_eff = min(4, 32*32) = 4
    assert cbs.dtype == np.float32
    assert idxs.shape == (4, 32, 32)
    assert idxs.dtype == np.uint8
    assert pos.shape == (4, 2)
    assert pos.dtype == np.int32
    # positions should cover the full (rows, cols) grid {(0,0),(0,1),(1,0),(1,1)}
    assert {tuple(p) for p in pos} == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_quantize_window_residual_norms_shape_and_sanity() -> None:
    """One float per pixel across kept tiles; small for k matching the cluster count."""
    window = _three_cluster_tile(64, 64, 128, seed=6)
    norms = quantize_window_residual_norms(window, t=32, k=4, m="euclidean", seed=42)
    # 4 tiles of 32x32 = 4096 pixels total.
    assert norms.shape == (4096,)
    assert norms.dtype == np.float32
    # noise has L2 magnitude ~ 0.1 * sqrt(128) ~= 1.13; reconstruction with k>=3 should
    # leave each pixel residual at roughly that scale.
    assert float(norms.mean()) < 1.5


def test_rvq_quantize_tile_lowers_error_vs_single_stage() -> None:
    """Two-stage RVQ reconstruction is at least as good as single-stage with the same k1."""
    tile = _three_cluster_tile(32, 32, 128, seed=10)
    cb1, idx1 = fast_quantize_tile(tile, k=4, distance="euclidean", seed=42)
    single_err = float(
        np.mean(np.linalg.norm(tile.reshape(-1, 128) - cb1[idx1].reshape(-1, 128), axis=1))
    )
    cb1, idx1, cb2, idx2 = rvq_quantize_tile(tile, k1=4, k2=4, m="euclidean", seed=42)
    rvq_recon = rvq_reconstruct_tile(cb1, idx1, cb2, idx2)
    rvq_err = float(
        np.mean(np.linalg.norm(tile.reshape(-1, 128) - rvq_recon.reshape(-1, 128), axis=1))
    )
    assert cb1.shape == (4, 128) and cb2.shape == (4, 128)
    assert idx1.shape == (32, 32) and idx2.shape == (32, 32)
    assert rvq_err <= single_err  # RVQ never worse than stage 1 alone


def test_rvq_quantize_tile_rejects_cosine() -> None:
    """Cosine RVQ is not supported (stage 1 discards magnitude)."""
    tile = _three_cluster_tile(16, 16, 32, seed=11)
    with pytest.raises(NotImplementedError, match="euclidean"):
        rvq_quantize_tile(tile, k1=4, k2=4, m="cosine", seed=42)


def test_rvq_quantize_window_for_serving_shapes() -> None:
    """RVQ on a window yields stacked codebooks/indices + positions."""
    window = _three_cluster_tile(64, 64, 128, seed=12)
    cbs1, idxs1, cbs2, idxs2, pos = rvq_quantize_window_for_serving(
        window, t=32, k1=4, k2=4, m="euclidean", seed=42
    )
    assert cbs1.shape == (4, 4, 128) and cbs2.shape == (4, 4, 128)
    assert idxs1.shape == (4, 32, 32) and idxs2.shape == (4, 32, 32)
    assert pos.shape == (4, 2)


def test_quantize_window_residual_norms_skips_nan_tiles() -> None:
    """Tiles containing NaN are dropped from the residual norm pool."""
    window = _three_cluster_tile(64, 64, 128, seed=7).copy()
    window[:32, :32, 0] = np.nan
    norms = quantize_window_residual_norms(window, t=32, k=4, m="euclidean", seed=42)
    assert norms.shape == (3 * 32 * 32,)


def test_quantize_window_for_serving_skips_nan_tiles() -> None:
    """A tile with any NaN is dropped; positions reflect only kept tiles."""
    window = _three_cluster_tile(64, 64, 128, seed=5).copy()
    window[:32, :32, 0] = np.nan  # corrupt the (0, 0) tile
    cbs, idxs, pos = quantize_window_for_serving(window, t=32, k=4, m="euclidean", seed=42)
    assert cbs.shape[0] == 3
    assert idxs.shape[0] == 3
    assert (0, 0) not in {tuple(p) for p in pos}
