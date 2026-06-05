"""BLAS-GEMM k-means for large tiles (WS-0a of the downstream-validation plan).

Scales to t up to 1024 (~1.05M pixels) and k up to ~1024 within a bounded RAM
budget. Three deliberate choices (point 7 of the research plan):

- **Sampled fit.** The Lloyd refinement runs on a subsample whose size scales with
  k (``points_per_center * k``, capped), so the fit cost is independent of tile
  area while still giving each centre enough support. The *assignment* is then
  exact over every pixel.
- **k-means++ init** on the subsample, with each candidate's squared distance
  computed by a single GEMV (the ``-2 x.c`` term), so init is O(k * n_sample * d)
  but with no per-point Python work beyond the unavoidable k-step loop.
- **BLAS-GEMM distance.** ``||x-c||^2 = ||x||^2 + ||c||^2 - 2 x.c^T``; the hot path
  is the multithreaded ``x @ c.T`` matmul. The full *assignment* is blocked so the
  ``(block, k)`` score matrix stays cache-resident and its peak RAM is bounded
  (~128 MB) regardless of tile area or k. The *fit* peak is instead set by the
  one-hot update matrix (n_sample x k); measured ~0.2/0.4/0.7/1.7 GB and
  ~0.3/1.1/4.8/11.3 s/tile (fit+assign) at k=256/512/1024/2048 on a 512x512x128
  tile. k=2048 is the cost corner -- drop the (32,2048) cell if it is too slow.

Euclidean only (RVQ stage-1/stage-2 both quantise in raw L2 space). Determinism:
all randomness flows from ``seed`` via ``np.random.default_rng``; argmin ties break
to the lowest index.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

_DEFAULT_ITERS = 25
_CONVERGE_TOL = 1e-4  # frobenius shift of the centre matrix to stop early.
_POINTS_PER_CENTER = 64  # subsample target = this * k (capped by _SAMPLE_CAP).
_SAMPLE_CAP = 100_000
_SCORE_BUDGET_BYTES = 128 * 1024 * 1024  # cap on the (block, k) score matrix.


def _sq_norms(a: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Row-wise squared L2 norms, ``(n,)`` float32."""
    sq = np.einsum("ij,ij->i", a, a).astype(np.float32, copy=False)
    return cast("npt.NDArray[np.float32]", sq)


def _block_rows(k: int) -> int:
    """Pick an assignment block so the ``(block, k)`` score stays under the budget."""
    return int(min(65536, max(4096, _SCORE_BUDGET_BYTES // (k * 4))))


def assign_blocked(
    x: npt.NDArray[np.float32], centers: npt.NDArray[np.float32]
) -> npt.NDArray[np.int32]:
    """Nearest-centre index per row of ``x`` via blocked BLAS-GEMM, bounded RAM.

    ``argmin_c ||x-c||^2 == argmax_c (x.c - 0.5||c||^2)`` (the ``||x||^2`` term is
    constant per row), so each block is one ``xb @ centers.T`` matmul + argmax.
    """
    n = x.shape[0]
    out = np.empty(n, dtype=np.int32)
    half_cc = 0.5 * _sq_norms(centers)
    block = _block_rows(centers.shape[0])
    for i in range(0, n, block):
        xb = x[i : i + block]
        score = xb @ centers.T - half_cc
        out[i : i + block] = score.argmax(axis=1).astype(np.int32)
    return out


def kmeans_plusplus_init(
    x: npt.NDArray[np.float32], k: int, rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """k-means++ seeding; each step's distances come from a single GEMV (``x @ c``)."""
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    xx = _sq_norms(x)
    centers[0] = x[rng.integers(n)]
    closest = xx + _sq_norms(centers[0:1])[0] - 2.0 * (x @ centers[0])
    np.maximum(closest, 0.0, out=closest)
    for i in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:  # all points coincide with a chosen centre
            centers[i] = x[rng.integers(n)]
        else:
            nxt = int(rng.choice(n, p=closest / total))
            centers[i] = x[nxt]
        d = xx + float(_sq_norms(centers[i : i + 1])[0]) - 2.0 * (x @ centers[i])
        np.minimum(closest, np.maximum(d, 0.0), out=closest)
    return centers


def _lloyd(
    x: npt.NDArray[np.float32],
    centers: npt.NDArray[np.float32],
    rng: np.random.Generator,
    n_iter: int,
) -> npt.NDArray[np.float32]:
    """Refine centres on ``x`` (the bounded subsample) with one-hot matmul updates."""
    n, k = x.shape[0], centers.shape[0]
    for _ in range(n_iter):
        labels = assign_blocked(x, centers)
        onehot = np.zeros((n, k), dtype=np.float32)
        onehot[np.arange(n), labels] = 1.0
        counts = onehot.sum(axis=0)
        nonempty = counts > 0
        new = np.zeros_like(centers)
        new[nonempty] = (onehot.T @ x)[nonempty] / counts[nonempty, None]
        empty = np.where(~nonempty)[0]
        if empty.size:
            new[empty] = x[rng.choice(n, size=empty.size, replace=False)]
        shift = float(np.linalg.norm(new - centers))
        centers = new
        if shift < _CONVERGE_TOL:
            break
    return centers


def kmeans_fit(
    x: npt.NDArray[np.float32],
    k: int,
    *,
    seed: int = 42,
    points_per_center: int = _POINTS_PER_CENTER,
    sample_cap: int = _SAMPLE_CAP,
    n_iter: int = _DEFAULT_ITERS,
) -> npt.NDArray[np.float32]:
    """Fit ``k_eff = min(k, n)`` centres on a k-scaled subsample of ``x``."""
    n = x.shape[0]
    k_eff = min(k, n)
    rng = np.random.default_rng(seed)
    target = min(n, sample_cap, max(points_per_center * k_eff, 2000))
    x_fit = x if target >= n else x[rng.choice(n, size=target, replace=False)]
    centers = kmeans_plusplus_init(x_fit, k_eff, rng)
    return _lloyd(x_fit, centers, rng, n_iter)


def quantize_tile_large(
    tile: npt.NDArray[np.float32],
    k: int,
    *,
    seed: int = 42,
    points_per_center: int = _POINTS_PER_CENTER,
    n_iter: int = _DEFAULT_ITERS,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Quantise an ``(H, W, C)`` tile: fit on a subsample, assign every pixel.

    Returns ``(centers (k_eff, C) float32, indices (H, W) int32)``.
    """
    h, w, c = tile.shape
    x = tile.reshape(-1, c).astype(np.float32, copy=False)
    centers = kmeans_fit(x, k, seed=seed, points_per_center=points_per_center, n_iter=n_iter)
    indices = assign_blocked(x, centers)
    return centers, indices.reshape(h, w)
