"""Tests for the index-compression sweep helpers + an RVQ integration check (WS-2).

The streaming ``main`` reads real data and is not unit-tested; this covers the grid,
the per-cell aggregation/derivation, and that a real RVQ stage-1 map compresses.
"""

from __future__ import annotations

import numpy as np

from scripts.phase3_index_compression import (
    DEFAULT_CONFIGS,
    DEFAULT_TILE_SIZES,
    _build_cells,
    _cell_row,
    _tile_index_metrics,
)
from tessera_vq.index_codec import compress_index_map
from tessera_vq.rvq_large import rvq_reconstruct_large


def test_default_grid_is_nine_cells() -> None:
    """Three tile sizes x three index configs = nine cells."""
    assert len(_build_cells(list(DEFAULT_TILE_SIZES), list(DEFAULT_CONFIGS))) == 9


def test_cell_row_derives_bytes_and_ratios() -> None:
    """Aggregation averages per-tile metrics and derives the byte/ratio columns."""
    per_tile = [
        {
            "idx1_row_bpp": 4.0,
            "idx1_morton_bpp": 3.0,
            "idx1_hilbert_bpp": 2.0,
            "idx1_raw_bpp": 6.0,
            "idx2_raw_bpp": 10.0,
            "idx2_hilbert_bpp": 10.0,
        }
    ]
    row = _cell_row((512, 64, 1024), per_tile)
    assert row["n_tiles"] == 1.0
    assert abs(row["codebook_Bpx"] - (64 + 1024) * 128 / (512 * 512)) < 1e-9
    assert row["idx1_best_bpp"] == 2.0  # min over orderings
    # total = codebook + best idx1 + raw idx2, all in bytes/px
    expect = row["codebook_Bpx"] + 2.0 / 8.0 + 10.0 / 8.0
    assert abs(row["total_compressed_Bpx"] - expect) < 1e-9
    assert abs(row["x_fp32_compressed"] - 512.0 / expect) < 1e-6


def test_cell_row_empty_is_minimal() -> None:
    """A cell with no tiles yields just identity + zero count."""
    row = _cell_row((768, 128, 512), [])
    assert row["n_tiles"] == 0.0
    assert "total_compressed_Bpx" not in row


def _autocorrelated_tile(side: int, dim: int, seed: int) -> np.ndarray:
    """Blocky tile: large constant regions + noise (spatially autocorrelated)."""
    rng = np.random.default_rng(seed)
    block = side // 8
    coarse = rng.integers(0, 6, size=(8, 8))
    centres = rng.standard_normal((6, dim)).astype(np.float32) * 10.0
    labels = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)  # (side, side)
    noise = 0.2 * rng.standard_normal((side, side, dim)).astype(np.float32)
    return (centres[labels] + noise).astype(np.float32)


def test_hilbert_reduces_runs_vs_row_major_on_rvq_map() -> None:
    """Locality property (holds for any k1): Hilbert yields <= runs than row-major."""
    tile = _autocorrelated_tile(128, 16, seed=0)
    res = rvq_reconstruct_large(tile, k1=64, k2=128, seed=42)
    row = compress_index_map(res.indices1, 64, "row").n_runs
    hil = compress_index_map(res.indices1, 64, "hilbert").n_runs
    assert hil <= row


def test_rle_beats_raw_when_k1_matches_landscape() -> None:
    """When k1 ~ the number of land-cover types, idx1 is smooth and RLE wins."""
    tile = _autocorrelated_tile(128, 16, seed=0)  # ~6 distinct regions
    res = rvq_reconstruct_large(tile, k1=8, k2=128, seed=42)
    m = _tile_index_metrics(res.indices1, 8, res.indices2, 128)
    hil = compress_index_map(res.indices1, 8, "hilbert").rle_bpp
    assert hil < m["idx1_raw_bpp"]  # space-filling + RLE beats the packed index
    assert m["idx1_raw_bpp"] == 3.0  # ceil(log2 8)
