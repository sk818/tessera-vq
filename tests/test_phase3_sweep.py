"""Tests for tessera_vq.phase3_sweep: per-pixel RVQ error histogram math.

Uses a synthetic 128x128 finite window so the test runs in a few seconds without
touching geotessera / zarr / Tessera data. Covers the rvq_errors -> hist_density
-> aggregate_long -> to_wide pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tessera_vq.phase3_sweep import (
    N_BINS,
    aggregate_long,
    hist_density,
    pick_bin_edges,
    rvq_errors,
    to_wide,
)


def _synthetic_window(h: int = 128, w: int = 128, seed: int = 0) -> np.ndarray:
    """Synthetic finite 128-d embedding window with three clusters + small noise."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((3, 128)).astype(np.float32) * 5.0
    labels = rng.integers(0, 3, size=(h, w))
    noise = 0.1 * rng.standard_normal((h, w, 128)).astype(np.float32)
    return (centres[labels] + noise).astype(np.float32)


def test_rvq_errors_finite_and_nonnegative() -> None:
    """Per-pixel L2 errors are finite and >= 0 on a finite window."""
    window = _synthetic_window(seed=0)
    l2 = rvq_errors(window, t=16, k1=64, k2=64, seed=42)
    assert l2.size > 0
    assert np.all(np.isfinite(l2))
    assert float(l2.min()) >= 0.0


def test_rvq_errors_count_matches_covered_tiles() -> None:
    """All pixels in covered (all-finite) tiles contribute one L2 entry."""
    window = _synthetic_window(h=64, w=64, seed=1)  # 4 tiles at t=32
    l2 = rvq_errors(window, t=32, k1=64, k2=64, seed=42)
    # All 4 tiles are finite -> 4 * 32 * 32 = 4096 pixels
    assert l2.size == 4 * 32 * 32


def test_rvq_errors_empty_when_all_tiles_filtered() -> None:
    """A window of all-NaN tiles -> empty error array (no failure)."""
    window = np.full((32, 32, 128), np.nan, dtype=np.float32)
    l2 = rvq_errors(window, t=16, k1=64, k2=64, seed=42)
    assert l2.size == 0


def test_pick_bin_edges_shape_and_order() -> None:
    """Edges have ``N_BINS + 1`` entries, are strictly increasing, anchored at 0."""
    rng = np.random.default_rng(2)
    l2 = rng.uniform(0.0, 10.0, size=10_000).astype(np.float32)
    edges_l2 = pick_bin_edges(l2)
    assert edges_l2.shape == (N_BINS + 1,)
    assert np.all(np.diff(edges_l2) > 0)
    assert edges_l2[0] == 0.0  # always anchored at 0 (perfect-reconstruction cells)


def test_pick_bin_edges_anchors_at_zero_even_with_positive_p1() -> None:
    """Even if every warm-up sample is large, edges[0] is still 0.

    Regression: pilot run with edges[0] = 4.4e-08 (positive p1 of warm-up errors)
    sent all exact-zero errors from k_eff = tile_area cells into overflow.
    """
    big_only = np.full(1000, 5.0, dtype=np.float32)
    edges_l2 = pick_bin_edges(big_only)
    assert edges_l2[0] == 0.0
    # An exact-zero error should now fall into bin 0 (counts > 0)
    zeros = np.zeros(100, dtype=np.float32)
    counts, _ = np.histogram(zeros, bins=edges_l2)
    assert counts[0] == 100  # noqa: PLR2004


def test_pick_bin_edges_handles_empty_warmup() -> None:
    """Empty warm-up array -> sensible default range, not crash."""
    edges_l2 = pick_bin_edges(np.zeros(0, np.float32))
    assert edges_l2.shape == (N_BINS + 1,)
    assert edges_l2[0] == 0.0 and edges_l2[-1] > 0.0


def test_hist_density_sums_to_1_minus_overflow() -> None:
    """Density + overflow_frac sums to 1 (a partition of all pixels)."""
    rng = np.random.default_rng(3)
    errors = rng.uniform(0.0, 1.0, size=5000).astype(np.float32)
    edges = np.linspace(0.2, 0.8, N_BINS + 1)  # leaves some pixels outside
    density, overflow = hist_density(errors, edges)
    assert density.shape == (N_BINS,)
    assert abs(float(density.sum()) + overflow - 1.0) < 1e-9
    assert 0.0 <= overflow <= 1.0


def test_hist_density_zero_overflow_when_range_covers_all() -> None:
    """If edges span [min..max] of the errors, density sums to 1 and overflow=0."""
    errors = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    density, overflow = hist_density(errors, edges)
    assert abs(float(density.sum()) - 1.0) < 1e-9
    assert overflow == 0.0


def test_aggregate_long_row_schema_and_count() -> None:
    """One row per bin; schema covers all expected columns; n_bboxes correct."""
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    densities = [np.full(N_BINS, 1.0 / N_BINS), np.full(N_BINS, 1.0 / N_BINS)]
    overflows = [0.0, 0.1]
    rows = aggregate_long("l2", edges, densities, overflows, t=16, k1=64, k2=128)
    assert len(rows) == N_BINS
    expected_cols = {
        "t",
        "k1",
        "k2",
        "metric",
        "bin_index",
        "bin_low",
        "bin_high",
        "mean_density",
        "sd_density",
        "overflow_frac_mean",
        "overflow_frac_sd",
        "n_bboxes",
    }
    assert expected_cols.issubset(rows[0].keys())
    assert all(r["n_bboxes"] == 2 for r in rows)  # noqa: PLR2004
    # mean_density of identical densities equals the density value itself
    assert abs(rows[0]["mean_density"] - 1.0 / N_BINS) < 1e-12


def test_to_wide_one_row_per_cell() -> None:
    """to_wide pivots long -> one row per (t, k1, k2) with bin_NN_mean/sd cols."""
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    densities = [np.full(N_BINS, 1.0 / N_BINS)]
    long_rows = aggregate_long("l2", edges, densities, [0.0], t=16, k1=64, k2=64)
    df = pd.DataFrame(long_rows)
    wide = to_wide(df, "l2")
    assert len(wide) == 1
    assert set(wide.columns) >= {
        "t",
        "k1",
        "k2",
        "bin_00_mean",
        "bin_00_sd",
        f"bin_{N_BINS - 1:02d}_mean",
        f"bin_{N_BINS - 1:02d}_sd",
        "overflow_frac_mean",
        "overflow_frac_sd",
    }
    # Single-bbox case: every bin's sd should be exactly 0
    sd_cols = [c for c in wide.columns if c.endswith("_sd") and c.startswith("bin_")]
    for col in sd_cols:
        assert float(wide.iloc[0][col]) == 0.0


def test_end_to_end_smallest_cell() -> None:
    """Synthetic 128x128 window through smallest (t,k1,k2): sums match expectations."""
    window = _synthetic_window(seed=4)
    l2 = rvq_errors(window, t=16, k1=64, k2=64, seed=42)
    edges_l2 = pick_bin_edges(l2)
    d_l2, of_l2 = hist_density(l2, edges_l2)
    # density + overflow partitions pixels
    assert abs(float(d_l2.sum()) + of_l2 - 1.0) < 1e-9
    # edges anchored at 0..p99 -> overflow ~ 1% (only the top-1% tail)
    assert of_l2 < 0.03  # noqa: PLR2004
