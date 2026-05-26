"""Tests for tessera_vq.sweep: vectorised k-means and the (t, K, m) sweep."""

import numpy as np

from tessera_vq.sweep import fast_quantize_tile, reconstruction_quantiles, sweep_window


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
