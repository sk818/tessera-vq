"""Tests for tessera_vq.kmeans_fast: BLAS-GEMM k-means for large tiles.

All fixtures are synthetic; no real Tessera data. Correctness is cross-checked
against a brute-force pairwise-distance reference, and quality against planted
well-separated clusters with known structure.
"""

from __future__ import annotations

import numpy as np

from tessera_vq.kmeans_fast import (
    assign_blocked,
    kmeans_fit,
    kmeans_plusplus_init,
    quantize_tile_large,
)


def _brute_assign(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Reference nearest-centre via full pairwise euclidean distance."""
    d = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
    return np.asarray(d.argmin(axis=1), dtype=np.int32)


def _planted_clusters(n_per: int, k: int, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``k`` well-separated gaussian blobs; return (points, true_labels)."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, dim)).astype(np.float32) * 20.0
    labels = np.repeat(np.arange(k), n_per)
    pts = (centres[labels] + 0.1 * rng.standard_normal((k * n_per, dim))).astype(np.float32)
    return pts, labels


def test_assign_blocked_matches_brute_force() -> None:
    """Blocked GEMM assignment equals the full pairwise-distance argmin."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, 32)).astype(np.float32)
    centers = rng.standard_normal((40, 32)).astype(np.float32)
    assert np.array_equal(assign_blocked(x, centers), _brute_assign(x, centers))


def test_assign_blocked_spans_multiple_blocks() -> None:
    """Correctness holds when n far exceeds one block (block shrinks with k)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200_000, 16)).astype(np.float32)
    centers = rng.standard_normal((512, 16)).astype(np.float32)
    out = assign_blocked(x, centers)
    assert out.shape == (200_000,)
    # spot-check a slice against brute force (full brute force would be huge)
    sl = slice(123_000, 123_500)
    assert np.array_equal(out[sl], _brute_assign(x[sl], centers))


def test_kmeans_recovers_well_separated_clusters() -> None:
    """k centres on k blobs reconstruct each point to ~noise scale."""
    pts, _labels = _planted_clusters(n_per=300, k=8, dim=64, seed=2)
    centers = kmeans_fit(pts, k=8, seed=42)
    assert centers.shape == (8, 64)
    idx = assign_blocked(pts, centers)
    recon_err = float(np.mean(np.linalg.norm(pts - centers[idx], axis=1)))
    # noise L2 ~ 0.1 * sqrt(64) ~= 0.8; good clustering stays near that.
    assert recon_err < 1.5


def test_kmeans_plusplus_init_picks_distinct_blob_centres() -> None:
    """k-means++ on k well-separated blobs seeds one centre near each blob."""
    pts, labels = _planted_clusters(n_per=200, k=6, dim=32, seed=3)
    rng = np.random.default_rng(7)
    centers = kmeans_plusplus_init(pts, 6, rng)
    assert centers.shape == (6, 32)
    # every planted blob should have a seed within its noise radius
    blob_means = np.stack([pts[labels == c].mean(0) for c in range(6)])
    nearest = np.linalg.norm(blob_means[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
    assert np.all(nearest < 5.0)


def test_kmeans_fit_caps_k_at_n() -> None:
    """Requesting more centres than points yields k_eff = n centres."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((30, 8)).astype(np.float32)
    centers = kmeans_fit(x, k=100, seed=42)
    assert centers.shape[0] == 30


def test_kmeans_fit_is_deterministic() -> None:
    """Same seed -> identical centres."""
    pts, _ = _planted_clusters(n_per=150, k=5, dim=16, seed=5)
    a = kmeans_fit(pts, k=5, seed=99)
    b = kmeans_fit(pts, k=5, seed=99)
    assert np.array_equal(a, b)


def test_quantize_tile_large_shapes_and_quality() -> None:
    """Tile wrapper returns (k_eff,C) centres + (H,W) indices and reconstructs well."""
    rng = np.random.default_rng(6)
    centres = rng.standard_normal((10, 48)).astype(np.float32) * 15.0
    labels = rng.integers(0, 10, size=(96, 96))
    tile = (centres[labels] + 0.1 * rng.standard_normal((96, 96, 48))).astype(np.float32)
    cb, idx = quantize_tile_large(tile, k=10, seed=42)
    assert cb.shape == (10, 48)
    assert idx.shape == (96, 96)
    assert int(idx.min()) >= 0 and int(idx.max()) < 10
    recon = cb[idx]
    assert float(np.mean(np.linalg.norm(tile - recon, axis=-1))) < 1.5


def test_empty_clusters_are_reseeded_not_nan() -> None:
    """Degenerate data (few distinct points, many centres) stays finite via reseed."""
    x = np.repeat(np.eye(4, dtype=np.float32), 50, axis=0)  # 4 distinct points x50
    centers = kmeans_fit(x, k=16, seed=42)  # k_eff capped at 16 but only 4 clusters
    assert np.all(np.isfinite(centers))
