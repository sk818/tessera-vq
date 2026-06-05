"""Tests for tessera_vq.tiling: jittered, NaN-aware large-tile selection.

Synthetic windows only; no geotessera / Tessera data. Covers in-bounds jitter,
NaN-edge avoidance, non-overlap, the finite-fraction threshold, and determinism.
"""

from __future__ import annotations

import numpy as np

from tessera_vq.tiling import extract_finite_tiles, finite_mask


def _finite_window(h: int, w: int, c: int = 8) -> np.ndarray:
    """All-finite synthetic window."""
    return np.random.default_rng(0).standard_normal((h, w, c)).astype(np.float32)


def test_returns_requested_count_and_shape_on_finite_window() -> None:
    """A fully finite roomy window yields n_tiles tiles of exact (t, t, C) shape."""
    win = _finite_window(2048, 2048)
    out = extract_finite_tiles(win, t=512, n_tiles=2, seed=42)
    assert len(out) == 2
    for s in out:
        assert s.tile.shape == (512, 512, 8)
        assert s.finite_frac == 1.0


def test_origins_stay_in_bounds() -> None:
    """Every selected tile lies fully inside the window."""
    win = _finite_window(1200, 1000)
    for s in extract_finite_tiles(win, t=768, n_tiles=2, seed=7, n_candidates=20):
        assert 0 <= s.row <= 1200 - 768
        assert 0 <= s.col <= 1000 - 768


def test_selected_tiles_do_not_overlap() -> None:
    """Greedy selection never returns two overlapping tiles."""
    win = _finite_window(2048, 2048)
    out = extract_finite_tiles(win, t=512, n_tiles=2, seed=1, n_candidates=32)
    assert len(out) == 2
    a, b = out
    assert abs(a.row - b.row) >= 512 or abs(a.col - b.col) >= 512


def test_avoids_nan_edges() -> None:
    """With NaN borders, selected tiles come from the finite interior."""
    win = _finite_window(1600, 1600)
    win[:400, :, 0] = np.nan  # top no-data band
    win[-400:, :, 0] = np.nan  # bottom no-data band
    out = extract_finite_tiles(win, t=512, n_tiles=2, seed=3, n_candidates=48)
    assert out  # the finite interior [400, 1200) fits fully finite 512-tiles
    for s in out:
        assert s.finite_frac == 1.0
        assert s.row >= 400 and s.row + 512 <= 1200


def test_window_smaller_than_tile_returns_empty() -> None:
    """No tile fits -> empty list, no error."""
    assert extract_finite_tiles(_finite_window(400, 400), t=512) == []


def test_strict_threshold_rejects_all_when_no_fully_finite_tile() -> None:
    """If every candidate has some NaN and threshold=1.0, nothing is returned."""
    win = _finite_window(700, 700)
    win[::50, ::50, 0] = np.nan  # sparse NaN grid hits every 512-tile
    assert extract_finite_tiles(win, t=512, min_finite_frac=1.0, n_candidates=16) == []


def test_relaxed_threshold_accepts_mostly_finite_tile() -> None:
    """Lowering min_finite_frac admits a tile with a little no-data."""
    win = _finite_window(700, 700)
    win[::50, ::50, 0] = np.nan
    out = extract_finite_tiles(win, t=512, min_finite_frac=0.9, n_candidates=16)
    assert out
    assert all(s.finite_frac >= 0.9 for s in out)


def test_deterministic_for_fixed_seed() -> None:
    """Same seed -> identical origins; selection is reproducible."""
    win = _finite_window(1200, 1200)
    a = extract_finite_tiles(win, t=512, seed=99, n_candidates=16)
    b = extract_finite_tiles(win, t=512, seed=99, n_candidates=16)
    assert [(s.row, s.col) for s in a] == [(s.row, s.col) for s in b]


def test_finite_mask_flags_any_nan_in_vector() -> None:
    """A pixel with one NaN component is non-finite in the mask."""
    win = _finite_window(4, 4)
    win[1, 2, 3] = np.nan
    m = finite_mask(win)
    assert m.shape == (4, 4)
    assert not m[1, 2]
    assert m.sum() == 15  # noqa: PLR2004
