"""Tests for the pure grid/sampling logic of the large-tile recon sweep (WS-1).

The streaming ``main`` reads real data and is not unit-tested here; this covers
the deterministic helpers only.
"""

from __future__ import annotations

from scripts.phase3_large_recon_sweep import (
    DEFAULT_CONFIGS,
    DEFAULT_TILE_SIZES,
    _build_cells,
    _sample_bboxes,
)


def test_default_grid_is_eight_cells() -> None:
    """Two tile sizes x four k1 values (k2=256) = eight cells, all k < t*t."""
    cells = _build_cells(list(DEFAULT_TILE_SIZES), list(DEFAULT_CONFIGS))
    assert len(cells) == 8
    assert (512, 20, 256) in cells
    assert (1024, 128, 256) in cells
    assert all(k2 == 256 for _t, _k1, k2 in cells)
    for t, k1, k2 in cells:
        assert k1 < t * t and k2 < t * t


def test_build_cells_drops_degenerate_k_ge_t2() -> None:
    """A tiny tile whose area is below k is dropped (k_eff would equal tile area)."""
    cells = _build_cells([16], ["64:1024"])  # 16*16 = 256 < both k -> dropped
    assert cells == []


def test_build_cells_parses_pairs_in_order() -> None:
    """Configs are specific (k1, k2) pairs, not a cross product."""
    cells = _build_cells([512], ["64:1024", "256:256"])
    assert cells == [(512, 64, 1024), (512, 256, 256)]


def test_sample_bboxes_returns_all_when_n_exceeds_pool() -> None:
    """Requesting more than the pool returns the whole pool unchanged."""
    pool = [f"b{i}" for i in range(5)]
    assert _sample_bboxes(pool, 10, seed=42) == pool  # type: ignore[arg-type]


def test_sample_bboxes_is_seeded_and_sorted() -> None:
    """A seed fixes the (ascending-index) selection; same seed -> same pick."""
    pool = [f"b{i}" for i in range(100)]
    a = _sample_bboxes(pool, 10, seed=7)  # type: ignore[arg-type]
    b = _sample_bboxes(pool, 10, seed=7)  # type: ignore[arg-type]
    assert a == b
    assert len(a) == 10
    assert a == sorted(a, key=lambda s: int(s[1:]))  # original-index order preserved
