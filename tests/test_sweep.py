"""Tests for tessera_vq.sweep: blockwise k-means delegation and the (t, K) sweep."""

import numpy as np

from tessera_vq.phase3_sweep import rvq_errors
from tessera_vq.sweep import (
    fast_quantize_tile,
    n_tiles_along,
    quantize_window_for_serving,
    quantize_window_residual_norms,
    quantize_window_residual_norms_rvq,
    reconstruction_quantiles,
    rvq_per_tile_errors,
    rvq_quantize_tile,
    rvq_quantize_window_for_serving,
    rvq_reconstruct_tile,
    sweep_window,
    tile_pixel_offset,
)


def test_rvq_per_tile_errors_match_flat_per_pixel_means() -> None:
    """Per-tile mean errors must equal the flat per-pixel errors grouped by tile."""
    rng = np.random.default_rng(7)
    window = rng.standard_normal((64, 96, 128)).astype(np.float32)
    t, k1, k2 = 32, 64, 64
    cb1, idx1, cb2, idx2, pos = rvq_quantize_window_for_serving(window, t, k1, k2, "euclidean", 42)
    l2_tile, _cos_tile = rvq_per_tile_errors(window, t, cb1, idx1, cb2, idx2, pos)
    flat_l2 = rvq_errors(window, t=t, k1=k1, k2=k2, seed=42)
    n_tiles = pos.shape[0]
    px = flat_l2.size // n_tiles  # equal-sized tiles -> contiguous per-tile blocks
    assert l2_tile.shape == (n_tiles,)
    assert np.allclose(l2_tile, flat_l2.reshape(n_tiles, px).mean(axis=1), rtol=1e-4, atol=1e-4)


def test_rvq_per_tile_errors_empty_window() -> None:
    """An all-NaN window keeps no tiles and yields empty per-tile error arrays."""
    window = np.full((32, 32, 128), np.nan, dtype=np.float32)
    cb1, idx1, cb2, idx2, pos = rvq_quantize_window_for_serving(window, 32, 64, 64, "euclidean", 42)
    l2, cos = rvq_per_tile_errors(window, 32, cb1, idx1, cb2, idx2, pos)
    assert l2.size == 0
    assert cos.size == 0


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


def test_quantize_window_residual_norms_rvq_smaller_than_single_stage() -> None:
    """RVQ residual norms (mean) should be no larger than single-stage at the same k1."""
    window = _three_cluster_tile(32, 32, 128, seed=13)
    single = quantize_window_residual_norms(window, t=32, k=4, m="euclidean", seed=42)
    rvq = quantize_window_residual_norms_rvq(window, t=32, k1=4, k2=4, m="euclidean", seed=42)
    assert single.shape == (1024,)
    assert rvq.shape == (1024,)
    assert float(rvq.mean()) <= float(single.mean())


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


# --- n_tiles_along / tile_pixel_offset: the tiling fix itself. The bug this
# closes: a window not an exact multiple of t used to silently drop the
# remainder strip (up to t-1 real, valid pixels) instead of covering it with
# a tile pulled back to end exactly at the window's true edge. ---


def test_n_tiles_along_exact_multiple() -> None:
    """No remainder -> plain floor division (== ceil here)."""
    assert n_tiles_along(64, 32) == 2
    assert n_tiles_along(96, 32) == 3


def test_n_tiles_along_with_remainder_rounds_up() -> None:
    """A remainder still gets one more tile -- not silently dropped."""
    assert n_tiles_along(65, 32) == 3  # floor would give 2, dropping the last pixel
    assert n_tiles_along(978, 256) == 4  # the real bbox from the postcard bug report


def test_n_tiles_along_smaller_than_t_is_zero() -> None:
    """Not even one tile fits -> 0, matching the existing 'no coverage' path."""
    assert n_tiles_along(10, 32) == 0


def test_tile_pixel_offset_regular_tiles_are_plain_stride() -> None:
    """Every tile except the last is untouched: idx * t, same as before this fix."""
    n = n_tiles_along(978, 256)
    assert tile_pixel_offset(0, n, 978, 256) == 0
    assert tile_pixel_offset(1, n, 978, 256) == 256
    assert tile_pixel_offset(2, n, 978, 256) == 512


def test_tile_pixel_offset_last_tile_ends_exactly_at_full_dim() -> None:
    """The last tile is pulled back to (full_dim - t), not dropped or left
    hanging past the edge -- covers real ground the old floor-based tiling
    silently discarded (this exact case: 978px at t=256 used to keep only
    768px, dropping a 210px/~2.5km strip -- the root cause of the postcard
    misalignment bug this fix closes)."""
    n = n_tiles_along(978, 256)
    last_offset = tile_pixel_offset(n - 1, n, 978, 256)
    assert last_offset == 978 - 256
    assert last_offset + 256 == 978  # tile's footprint reaches the true edge exactly


def test_tile_pixel_offset_reduces_to_plain_stride_when_no_remainder() -> None:
    """Exact multiple of t -> the 'last tile' rule agrees with idx * t (no
    special-casing needed; this is a strict generalisation, not a branch)."""
    n = n_tiles_along(1024, 256)
    assert n == 4
    assert tile_pixel_offset(3, n, 1024, 256) == 3 * 256 == 1024 - 256


def test_quantize_window_for_serving_covers_the_remainder_tile() -> None:
    """A window with a genuine remainder now yields a tile covering it, at the
    pulled-back offset -- not silently dropped the way floor-based tiling did."""
    window = _three_cluster_tile(65, 32, 128, seed=20)  # 65 rows: one row of remainder at t=32
    cbs, idxs, pos = quantize_window_for_serving(window, t=32, k=4, m="euclidean", seed=42)
    assert pos.shape[0] == 3  # 2 regular row-tiles + 1 pulled-back remainder tile, 1 col
    assert {tuple(p) for p in pos} == {(0, 0), (1, 0), (2, 0)}
    # The remainder (last) row-tile is index 2 of 3; its true pixel offset is 65-32=33.
    n_rows = n_tiles_along(65, 32)
    assert tile_pixel_offset(2, n_rows, 65, 32) == 33
