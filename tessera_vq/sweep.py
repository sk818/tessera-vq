"""Fast per-tile (t, K) sweep for the interactive bolt-on.

K-means is the per-call hot loop; it is delegated to ``blockwise_kmeans`` (the
project's single source of truth: a k-scaled sampled fit plus an exact, blocked
BLAS-GEMM assignment). Euclidean only — the ``m``/``distance`` arguments are kept
for signature compatibility with the bolt-on API but are ignored (cosine was
dropped).

Designed for use inside ``tessera_vq.server`` (Flask /sweep endpoint).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from blockwise_kmeans import assign_blocked, kmeans_fit

Distance = Literal["euclidean", "cosine"]

_DEFAULT_KMEANS_ITERS = 20  # Lloyd iterations on the fit subsample.


def n_tiles_along(full_dim: int, t: int) -> int:
    """How many ``t``-tiles cover ``full_dim`` under :func:`tile_pixel_offset`'s
    scheme: ``ceil(full_dim / t)``, i.e. one more than a plain floor-tiling
    whenever ``full_dim`` isn't an exact multiple of ``t``. 0 if ``full_dim < t``
    (not even one tile fits)."""
    if full_dim < t:
        return 0
    return -(-full_dim // t)  # ceil division without importing math


def tile_pixel_offset(idx: int, n: int, full_dim: int, t: int) -> int:
    """Pixel offset of tile ``idx`` (0-based) along one axis, ``n = n_tiles_along(...)``.

    Fixed ``t``-stride (``idx * t``) for every tile except the last, which is
    pulled back to end exactly at ``full_dim`` (``full_dim - t``) instead of
    being dropped when ``full_dim`` isn't an exact multiple of ``t``. The two
    formulas agree whenever ``full_dim % t == 0``, so this is a strict
    generalisation of plain non-overlapping tiling, not a special case of it.

    This one function is the single source of truth for where a tile sits,
    shared by the server's tiling loop (this module) and the client's
    reconstruction (``tessera_vq.client``, which duplicates this exact formula
    since the client is deliberately kept free of this [server]-extra module's
    heavier deps -- see that module's docstring). A client that doesn't know
    about the last tile's pull-back (an old client talking to a new server, or
    vice versa) just doesn't request/place it -- see ``reconstruct_from_structure``'s
    docstring for why that degrades safely instead of corrupting anything.
    """
    if idx == n - 1:
        return full_dim - t
    return idx * t


def fast_quantize_tile(
    tile: npt.NDArray[np.float32],
    k: int,
    distance: Distance = "euclidean",
    seed: int = 42,
    *,
    sample_size: int = 2000,
    n_iter: int = _DEFAULT_KMEANS_ITERS,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Sampled k-means quantisation of an ``(H, W, 128)`` tile via ``blockwise_kmeans``.

    Euclidean only; ``distance`` is accepted for API compatibility but ignored.
    ``sample_size`` caps the fit subsample. Returns ``(centers (k_eff, C), indices
    (H, W) int32)``.
    """
    h, w, c = tile.shape
    x = tile.reshape(-1, c).astype(np.float32, copy=False)
    centers = kmeans_fit(x, k, seed=seed, sample_cap=sample_size, n_iter=n_iter)
    indices = assign_blocked(x, centers)
    return centers, indices.reshape(h, w)


def reconstruction_quantiles(
    original: npt.NDArray[np.float32], reconstruction: npt.NDArray[np.float32]
) -> dict[str, float]:
    """Per-pixel cosine distance and L2 quantiles (10/50/90/99) between tile + reconstruction."""
    o = original.reshape(-1, original.shape[-1]).astype(np.float64)
    r = reconstruction.reshape(-1, reconstruction.shape[-1]).astype(np.float64)
    on = np.linalg.norm(o, axis=1)
    rn = np.linalg.norm(r, axis=1)
    denom = np.where((on > 0) & (rn > 0), on * rn, 1.0)
    cos_dist = 1.0 - (o * r).sum(axis=1) / denom
    l2 = np.linalg.norm(o - r, axis=1)
    out: dict[str, float] = {}
    for q in (0.1, 0.5, 0.9, 0.99):
        tag = f"p{int(q * 100)}"
        out[f"cos_{tag}"] = float(np.quantile(cos_dist, q))
        out[f"l2_{tag}"] = float(np.quantile(l2, q))
    return out


def _iterate_subtiles(window: npt.NDArray[np.float32], t: int) -> list[npt.NDArray[np.float32]]:
    """Non-overlapping ``t x t`` sub-tiles of ``window`` that are entirely finite."""
    h, w, _ = window.shape
    out: list[npt.NDArray[np.float32]] = []
    for r in range(0, (h // t) * t, t):
        for c in range(0, (w // t) * t, t):
            tile = window[r : r + t, c : c + t]
            if np.isfinite(tile).all():
                out.append(np.asarray(tile, dtype=np.float32))
    return out


def quantize_window_for_serving(
    window: npt.NDArray[np.float32],
    t: int,
    k: int,
    m: Distance,
    seed: int = 42,
    *,
    sample_size: int = 2000,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint16], npt.NDArray[np.int32]]:
    """Tile ``window`` into t x t blocks (see :func:`tile_pixel_offset` for the
    last row/column's placement); quantise each all-finite block.

    Returns ``(codebooks, indices, positions)`` where:
      ``codebooks``  ``(n_tiles, k_eff, 128)`` float32  (``k_eff = min(k, t * t)``)
      ``indices``    ``(n_tiles, t, t)``  uint8 if ``k_eff <= 256`` else uint16
      ``positions``  ``(n_tiles, 2)`` int32 ``(row, col)`` tile-grid indices --
        NOT pixel offsets; see :func:`tile_pixel_offset` to convert.
    """
    h, w, c = window.shape
    rows, cols = n_tiles_along(h, t), n_tiles_along(w, t)
    k_eff = min(k, t * t)
    # Any: older mypys won't narrow the conditional dtype expression; runtime is correct.
    idx_dtype: Any = np.uint8 if k_eff <= 256 else np.uint16  # noqa: PLR2004
    cbs: list[npt.NDArray[np.float32]] = []
    idxs: list[npt.NDArray[Any]] = []
    pos: list[tuple[int, int]] = []
    for r in range(rows):
        row_off = tile_pixel_offset(r, rows, h, t)
        for col in range(cols):
            col_off = tile_pixel_offset(col, cols, w, t)
            tile = window[row_off : row_off + t, col_off : col_off + t]
            if not np.isfinite(tile).all():
                continue
            cb, idx = fast_quantize_tile(tile, k, m, seed, sample_size=sample_size)
            cbs.append(cb)
            idxs.append(idx.astype(idx_dtype))
            pos.append((r, col))
    if not cbs:
        return (
            np.zeros((0, k_eff, c), dtype=np.float32),
            np.zeros((0, t, t), dtype=idx_dtype),
            np.zeros((0, 2), dtype=np.int32),
        )
    return (
        np.stack(cbs).astype(np.float32, copy=False),
        np.stack(idxs),
        np.asarray(pos, dtype=np.int32),
    )


def rvq_quantize_tile(
    tile: npt.NDArray[np.float32],
    k1: int,
    k2: int,
    m: Distance = "euclidean",
    seed: int = 42,
    *,
    sample_size: int = 2000,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
]:
    """Two-stage Residual VQ on one ``(H, W, 128)`` tile.

    Stage 1: k-means with ``k1`` on the tile  -> ``(codebook1, indices1)``.
    Stage 2: k-means with ``k2`` on the residual ``tile - codebook1[indices1]``
             -> ``(codebook2, indices2)``.

    Reconstruction is ``codebook1[indices1] + codebook2[indices2]``. Stage 2 is just
    ``fast_quantize_tile`` on the residual — if you want to sweep ``k2`` without
    redoing stage 1, compute the residual once and call ``fast_quantize_tile`` on
    it directly.

    Euclidean only; ``m`` is accepted for API compatibility but ignored (cosine
    stage 1 would discard magnitude, leaving stage 2 nothing meaningful to quantise).
    """
    centers1, indices1 = fast_quantize_tile(tile, k1, m, seed, sample_size=sample_size)
    residual = (tile - centers1[indices1]).astype(np.float32, copy=False)
    centers2, indices2 = fast_quantize_tile(residual, k2, m, seed + 1, sample_size=sample_size)
    return centers1, indices1.astype(np.int32), centers2, indices2.astype(np.int32)


def rvq_reconstruct_tile(
    codebook1: npt.NDArray[np.float32],
    indices1: npt.NDArray[np.integer[Any]],
    codebook2: npt.NDArray[np.float32],
    indices2: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.float32]:
    """Reconstruct an RVQ-quantised tile as ``codebook1[idx1] + codebook2[idx2]``."""
    return cast(
        "npt.NDArray[np.float32]",
        (codebook1[indices1] + codebook2[indices2]).astype(np.float32, copy=False),
    )


def rvq_per_tile_errors(
    window: npt.NDArray[np.float32],
    t: int,
    codebooks1: npt.NDArray[np.float32],
    indices1: npt.NDArray[np.integer[Any]],
    codebooks2: npt.NDArray[np.float32],
    indices2: npt.NDArray[np.integer[Any]],
    positions: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Mean per-tile L2 and cosine reconstruction error for already-quantised RVQ codes.

    For each kept tile ``i`` (at ``positions[i]``), reconstructs
    ``codebooks1[i][indices1[i]] + codebooks2[i][indices2[i]]`` and returns the
    *mean over that tile's pixels* of the per-pixel L2 distance and cosine
    distance. Returns two ``(n_tiles,)`` arrays aligned with ``positions``; empty
    if no tiles were kept.

    Works from precomputed codes rather than re-fitting RVQ, so an analysis pass
    can reuse the codebooks from ``rvq_quantize_window_for_serving``. The per-pixel
    formulas match ``phase3_sweep.rvq_errors`` (this just averages within a tile).
    """
    n = int(positions.shape[0])
    if n == 0:
        empty = np.zeros(0, np.float32)
        return empty, empty
    h, w, _dim = window.shape
    rows, cols = n_tiles_along(h, t), n_tiles_along(w, t)
    l2 = np.empty(n, np.float32)
    cos = np.empty(n, np.float32)
    for i in range(n):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        row_off, col_off = tile_pixel_offset(r, rows, h, t), tile_pixel_offset(c, cols, w, t)
        orig = window[row_off : row_off + t, col_off : col_off + t]
        recon = codebooks1[i][indices1[i]] + codebooks2[i][indices2[i]]
        l2[i] = float(np.linalg.norm(orig - recon, axis=-1).mean())
        on = np.linalg.norm(orig, axis=-1)
        rn = np.linalg.norm(recon, axis=-1)
        denom = np.where((on > 0) & (rn > 0), on * rn, 1.0)
        cd = np.maximum(1.0 - (orig * recon).sum(axis=-1) / denom, 0.0)
        cos[i] = float(cd.mean())
    return l2, cos


_TileBatch = list[tuple[tuple[int, int], "npt.NDArray[np.float32]"]]
_TileResult = tuple[
    tuple[int, int],
    "npt.NDArray[np.float32]",
    "npt.NDArray[np.int32]",
    "npt.NDArray[np.float32]",
    "npt.NDArray[np.int32]",
]


def _quantize_tile_batch(
    batch: _TileBatch,
    k1: int,
    k2: int,
    m: Distance,
    seed: int,
    sample_size: int,
) -> list[_TileResult]:
    """Worker-process entry point: RVQ-quantize a batch of tiles sequentially.

    Runs in its own OS process (see ``rvq_quantize_window_for_serving``) so
    each tile's ``blockwise_kmeans`` call is on its own core -- but that means
    capping BLAS's *own* internal threading to 1 here is required, not
    optional. Without it, each of the (many) worker processes independently
    tries to spawn its own OpenBLAS thread pool on top of the process-level
    parallelism already in play, oversubscribing the host's cores. Confirmed
    live: a naive multiprocess attempt without this cap was 10x+ slower than
    doing nothing at all (cores thrashing on context switches, not computing).
    A first attempt at parallelizing this loop with threads instead of
    processes was *also* slower than the sequential baseline: blockwise_kmeans's
    Lloyd loop is many small numpy ops per iteration, not one long
    GIL-releasing BLAS call, so threads mostly just fought over the GIL.
    """
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    return [
        (pos, *rvq_quantize_tile(tile, k1, k2, m, seed, sample_size=sample_size))
        for pos, tile in batch
    ]


def rvq_quantize_window_for_serving(
    window: npt.NDArray[np.float32],
    t: int,
    k1: int,
    k2: int,
    m: Distance = "euclidean",
    seed: int = 42,
    *,
    sample_size: int = 2000,
    n_jobs: int = -1,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[Any],
    npt.NDArray[np.float32],
    npt.NDArray[Any],
    npt.NDArray[np.int32],
]:
    """Tile ``window`` into t x t blocks (see :func:`tile_pixel_offset` for the
    last row/column's placement); run RVQ on each all-finite block.

    Each tile's RVQ is fully independent, so tiles are split into ``n_jobs``
    batches (default: one per available core) and quantized across worker
    *processes* -- see ``_quantize_tile_batch``'s docstring for why processes,
    not threads, and why each worker caps its own BLAS threading to 1.
    Confirmed live (2026-08-14): a 977x1982px postcard mosaic (1922 32x32
    tiles) went from ~60s single-threaded to ~9s with this.

    Output order is restored to the same row-major tile scan order the old
    sequential version produced, even though workers complete out of order:
    the wire format's ``positions`` array is NOT actually consulted by the
    client for tile placement (``public/js/vq_reconstruct.js`` reconstructs
    by array index assuming row-major order), so this order is load-bearing,
    not incidental -- silently changing it would misplace every tile in the
    reconstructed mosaic.

    Returns ``(codebooks1, indices1, codebooks2, indices2, positions)`` where:
      ``codebooks{1,2}``  ``(n_tiles, k{1,2}_eff, 128)`` float32
      ``indices{1,2}``    ``(n_tiles, t, t)`` uint8 if ``k_eff <= 256`` else uint16
      ``positions``       ``(n_tiles, 2)`` int32 ``(row, col)`` tile-grid indices --
        NOT pixel offsets; see :func:`tile_pixel_offset` to convert.
    """
    h, w, c = window.shape
    rows, cols = n_tiles_along(h, t), n_tiles_along(w, t)
    k1_eff = min(k1, t * t)
    k2_eff = min(k2, t * t)
    idx_dtype1: Any = np.uint8 if k1_eff <= 256 else np.uint16  # noqa: PLR2004
    idx_dtype2: Any = np.uint8 if k2_eff <= 256 else np.uint16  # noqa: PLR2004

    # Cheap sequential pass: pick out the all-finite tiles and their grid
    # positions. The expensive part (RVQ quantization) happens below, in
    # parallel, over just this filtered set.
    items: list[tuple[tuple[int, int], npt.NDArray[np.float32]]] = []
    for r in range(rows):
        row_off = tile_pixel_offset(r, rows, h, t)
        for col in range(cols):
            col_off = tile_pixel_offset(col, cols, w, t)
            tile = window[row_off : row_off + t, col_off : col_off + t]
            if not np.isfinite(tile).all():
                continue
            items.append(((r, col), np.asarray(tile, dtype=np.float32)))

    if not items:
        return (
            np.zeros((0, k1_eff, c), dtype=np.float32),
            np.zeros((0, t, t), dtype=idx_dtype1),
            np.zeros((0, k2_eff, c), dtype=np.float32),
            np.zeros((0, t, t), dtype=idx_dtype2),
            np.zeros((0, 2), dtype=np.int32),
        )

    cpu_count = os.cpu_count() or 1
    n_workers = cpu_count if n_jobs is None or n_jobs < 1 else n_jobs
    n_workers = max(1, min(n_workers, len(items)))
    batches: list[_TileBatch] = [items[i::n_workers] for i in range(n_workers)]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        batch_results = ex.map(
            _quantize_tile_batch,
            batches,
            repeat(k1),
            repeat(k2),
            repeat(m),
            repeat(seed),
            repeat(sample_size),
        )
        # ex.map preserves the order batches were submitted in, but batches
        # were built by round-robin striding (items[i::n_workers]) to spread
        # tiles evenly across workers -- so flattening batch-by-batch here
        # would yield tiles in that strided order, not the row-major scan
        # order items[] was built in. The wire format's `positions` array is
        # NOT actually consulted by the client for tile placement
        # (public/js/vq_reconstruct.js reconstructs by array index assuming
        # row-major order) -- so restore that order via a position lookup,
        # rather than changing the (already-shipped, already-relied-upon)
        # wire contract to match.
        by_pos: dict[tuple[int, int], _TileResult] = {
            result[0]: result for batch in batch_results for result in batch
        }
    flat: list[_TileResult] = [by_pos[item_pos] for item_pos, _tile in items]

    pos = [p for p, _cb1, _idx1, _cb2, _idx2 in flat]
    cbs1 = [cb1 for _p, cb1, _idx1, _cb2, _idx2 in flat]
    idxs1 = [idx1.astype(idx_dtype1) for _p, _cb1, idx1, _cb2, _idx2 in flat]
    cbs2 = [cb2 for _p, _cb1, _idx1, cb2, _idx2 in flat]
    idxs2 = [idx2.astype(idx_dtype2) for _p, _cb1, _idx1, _cb2, idx2 in flat]
    return (
        np.stack(cbs1).astype(np.float32, copy=False),
        np.stack(idxs1),
        np.stack(cbs2).astype(np.float32, copy=False),
        np.stack(idxs2),
        np.asarray(pos, dtype=np.int32),
    )


def quantize_window_residual_norms(
    window: npt.NDArray[np.float32],
    t: int,
    k: int,
    m: Distance,
    seed: int = 42,
    *,
    sample_size: int = 2000,
) -> npt.NDArray[np.float32]:
    """Per-pixel L2 residual norms ``||x - c_{idx}||_2`` across all all-finite t x t tiles.

    Returns a flat ``(n_pixels,)`` float32 array where ``n_pixels`` is the total count
    of pixels across kept (all-finite) tiles. Useful for plotting a histogram of "how
    off" each pixel's reconstruction is. Euclidean only (``m`` is ignored).
    """
    h, w, _ = window.shape
    chunks: list[npt.NDArray[np.float32]] = []
    for r in range(0, (h // t) * t, t):
        for col in range(0, (w // t) * t, t):
            tile = window[r : r + t, col : col + t]
            if not np.isfinite(tile).all():
                continue
            tile_f = np.asarray(tile, dtype=np.float32)
            centers, idx = fast_quantize_tile(tile_f, k, m, seed, sample_size=sample_size)
            residual = tile_f - centers[idx]
            chunks.append(np.linalg.norm(residual, axis=-1).astype(np.float32).ravel())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def quantize_window_residual_norms_rvq(
    window: npt.NDArray[np.float32],
    t: int,
    k1: int,
    k2: int,
    m: Distance = "euclidean",
    seed: int = 42,
    *,
    sample_size: int = 2000,
) -> npt.NDArray[np.float32]:
    """Per-pixel L2 residual norms after two-stage RVQ reconstruction.

    For each kept tile, computes ``||x - (c1[idx1] + c2[idx2])||_2`` per pixel and
    concatenates across all all-finite tiles. Euclidean only (matches ``rvq_quantize_tile``).
    """
    h, w, _ = window.shape
    chunks: list[npt.NDArray[np.float32]] = []
    for r in range(0, (h // t) * t, t):
        for col in range(0, (w // t) * t, t):
            tile = window[r : r + t, col : col + t]
            if not np.isfinite(tile).all():
                continue
            tile_f = np.asarray(tile, dtype=np.float32)
            cb1, idx1, cb2, idx2 = rvq_quantize_tile(
                tile_f, k1, k2, m, seed, sample_size=sample_size
            )
            residual = tile_f - (cb1[idx1] + cb2[idx2])
            chunks.append(np.linalg.norm(residual, axis=-1).astype(np.float32).ravel())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def sweep_window(
    window: npt.NDArray[np.float32],
    ts: list[int],
    ks: list[int],
    ms: list[Distance],
    seed: int = 42,
    *,
    sample_size: int = 2000,
) -> list[dict[str, Any]]:
    """Run the (t, K, m) sweep on one window; one row per ``(t, K, m, subtile_idx)``."""
    rows: list[dict[str, Any]] = []
    for t in ts:
        for st_idx, subtile in enumerate(_iterate_subtiles(window, t)):
            for k in ks:
                for m in ms:
                    centers, idx = fast_quantize_tile(subtile, k, m, seed, sample_size=sample_size)
                    errs = reconstruction_quantiles(subtile, centers[idx])
                    rows.append(
                        {"t": t, "subtile": st_idx, "k": k, "m": m, "n_pixels": int(t * t), **errs}
                    )
    return rows
