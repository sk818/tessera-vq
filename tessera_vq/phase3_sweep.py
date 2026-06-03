"""Phase 3 RVQ sweep math: per-pixel error histograms aggregated across bboxes.

Pure-numpy helpers behind ``scripts/phase3_rvq_sweep.py``. Split out from the
CLI so the math is unit-testable on synthetic windows without geotessera / zarr
in the loop. Depends on ``tessera_vq.sweep`` only (numpy + RVQ k-means), so it
runs under the package's core deps without the ``[server]`` extra.

Conventions:

- "Errors" are per-pixel scalars: L2 distance ``||orig - recon||_2`` and
  cosine distance ``1 - cos(orig, recon)`` between a 128-d Tessera embedding
  and its RVQ reconstruction ``codebook1[idx1] + codebook2[idx2]``.
- Histograms are normalised to *density* (counts / n_pixels), so per-bbox
  histograms sum to ``(1 - overflow_frac)`` and are comparable across bboxes
  of differing native shapes.
- Aggregation across bboxes is mean +- std (ddof=1) of the per-bbox density
  per bin; same for ``overflow_frac``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from tessera_vq.sweep import rvq_quantize_window_for_serving

N_BINS = 50
WARMUP_P_HI = 99.0


def rvq_errors(
    window: npt.NDArray[np.float32],
    t: int,
    k1: int,
    k2: int,
    seed: int,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Run euclidean RVQ on ``window``; return flat per-pixel L2 and cosine errors.

    Pixels in NaN-filtered tiles are excluded (those tiles have no codebook).
    Returns ``(l2, cos)`` arrays of equal length; empty if every candidate
    tile was NaN-filtered.
    """
    cb1, idx1, cb2, idx2, positions = rvq_quantize_window_for_serving(
        window, t, k1, k2, "euclidean", seed
    )
    if positions.shape[0] == 0:
        empty = np.zeros(0, np.float32)
        return empty, empty
    l2_chunks: list[npt.NDArray[np.float32]] = []
    cos_chunks: list[npt.NDArray[np.float32]] = []
    for i in range(int(positions.shape[0])):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        orig = window[r * t : (r + 1) * t, c * t : (c + 1) * t]
        recon = cb1[i][idx1[i]] + cb2[i][idx2[i]]
        diff = orig - recon
        l2 = np.linalg.norm(diff, axis=-1)
        on = np.linalg.norm(orig, axis=-1)
        rn = np.linalg.norm(recon, axis=-1)
        denom = np.where((on > 0) & (rn > 0), on * rn, 1.0)
        # Cosine distance is defined as >= 0; float32 (orig*recon).sum() can round
        # slightly above on*rn when recon == orig, yielding tiny negative values.
        cos = np.maximum(1.0 - (orig * recon).sum(axis=-1) / denom, 0.0)
        l2_chunks.append(l2.ravel().astype(np.float32))
        cos_chunks.append(cos.ravel().astype(np.float32))
    return np.concatenate(l2_chunks), np.concatenate(cos_chunks)


def pick_bin_edges(
    l2: npt.NDArray[np.float32],
    cos: npt.NDArray[np.float32],
    *,
    n_bins: int = N_BINS,
    p_hi: float = WARMUP_P_HI,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Pick frozen bin edges from a warm-up bbox's L2 and cosine errors.

    Always anchors ``edges[0] = 0`` so cells with near-perfect reconstruction
    (``k_eff = tile_area`` -> exact codebook -> per-pixel error ~ 0) fall into
    bin 0 rather than the underflow slot. The upper edge uses the ``p_hi``
    percentile (default 99) so a handful of outliers do not stretch the range.
    Falls back to plausible defaults if a warm-up array is empty.
    """
    if l2.size == 0 or cos.size == 0:
        return (
            np.linspace(0.0, 30.0, n_bins + 1),
            np.linspace(0.0, 1.0, n_bins + 1),
        )
    hi_l2 = float(np.percentile(l2, p_hi))
    hi_cos = float(np.percentile(cos, p_hi))
    if hi_l2 <= 0.0:
        hi_l2 = 1.0
    if hi_cos <= 0.0:
        hi_cos = 1e-3
    return (
        np.linspace(0.0, hi_l2, n_bins + 1),
        np.linspace(0.0, hi_cos, n_bins + 1),
    )


def hist_density(
    errors: npt.NDArray[np.float32], edges: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], float]:
    """Histogram errors -> per-bin density + fraction of pixels outside edges.

    Density is ``counts / n_pixels`` so it sums to ``(1 - overflow_frac)``;
    using counts/n_pixels (rather than the np.histogram default of true PDF)
    keeps the values directly comparable across bboxes regardless of bin width.
    """
    if errors.size == 0:
        return np.zeros(edges.size - 1), 0.0
    counts, _ = np.histogram(errors, bins=edges)
    n_in = int(counts.sum())
    overflow = (errors.size - n_in) / errors.size
    density = counts.astype(np.float64) / errors.size
    return density, float(overflow)


def aggregate_long(
    metric: str,
    edges: npt.NDArray[np.float64],
    densities: list[npt.NDArray[np.float64]],
    overflows: list[float],
    t: int,
    k1: int,
    k2: int,
) -> list[dict[str, Any]]:
    """Stack per-bbox densities, return one long-format row per bin (mean +- sd)."""
    arr = np.stack(densities, axis=0) if densities else np.zeros((0, edges.size - 1))
    if arr.shape[0] == 0:
        return []
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    of = np.asarray(overflows, dtype=np.float64)
    of_mean = float(of.mean())
    of_sd = float(of.std(ddof=1)) if of.size > 1 else 0.0
    rows: list[dict[str, Any]] = []
    for i in range(edges.size - 1):
        rows.append(
            {
                "t": int(t),
                "k1": int(k1),
                "k2": int(k2),
                "metric": metric,
                "bin_index": int(i),
                "bin_low": float(edges[i]),
                "bin_high": float(edges[i + 1]),
                "mean_density": float(mean[i]),
                "sd_density": float(sd[i]),
                "overflow_frac_mean": of_mean,
                "overflow_frac_sd": of_sd,
                "n_bboxes": int(arr.shape[0]),
            }
        )
    return rows


def to_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pivot a long-format sweep CSV to the per-metric wide format.

    Columns: ``t, k1, k2, bin_00_mean, bin_00_sd, ..., bin_<N-1>_sd,
    overflow_frac_mean, overflow_frac_sd``. One row per ``(t, k1, k2)`` cell.
    """
    sub = df[df["metric"] == metric].copy()
    pivot = sub.pivot_table(
        index=["t", "k1", "k2"],
        columns="bin_index",
        values=["mean_density", "sd_density"],
    )
    # pivot.columns is a MultiIndex of (stat, bin_index) tuples; pandas-stubs types
    # it as Index[str] which mypy refuses to unpack, hence the explicit cast.
    flat_cols = cast("list[tuple[str, int]]", list(pivot.columns))
    pivot.columns = pd.Index(
        [
            f"bin_{int(bi):02d}_{('mean' if stat == 'mean_density' else 'sd')}"
            for stat, bi in flat_cols
        ]
    )
    pivot = pivot.reset_index()
    of = sub.drop_duplicates(subset=["t", "k1", "k2"])[
        ["t", "k1", "k2", "overflow_frac_mean", "overflow_frac_sd"]
    ]
    return pivot.merge(of, on=["t", "k1", "k2"])
