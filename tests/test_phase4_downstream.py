"""Tests for the downstream sweep's pure helpers (WS-3).

The geotessera + tessera_eval wiring needs those (cross-repo) deps and real data, so
only the dependency-free grid helper is unit-tested. Importing the module here also
asserts its heavy deps stay deferred (guarded inside functions).
"""

from __future__ import annotations

from scripts.phase4_downstream import DEFAULT_CONFIGS, DEFAULT_TILE_SIZES, _build_cells


def test_default_grid_is_eight_cells() -> None:
    """Two tile sizes x four k1 values (k2=256) = eight cells, all k < t*t."""
    cells = _build_cells(list(DEFAULT_TILE_SIZES), list(DEFAULT_CONFIGS))
    assert len(cells) == 8
    assert all(k2 == 256 for _t, _k1, k2 in cells)
    assert (1024, 20, 256) in cells


def test_build_cells_parses_pairs() -> None:
    """k1:k2 configs are parsed as specific pairs in order."""
    assert _build_cells([512], ["20:256", "128:256"]) == [(512, 20, 256), (512, 128, 256)]
