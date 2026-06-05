"""Hilbert curve ordering for index maps (WS-2).

The Hilbert curve preserves 2-D locality better than Z-order (no long diagonal
jumps), so traversing the stage-1 index map (idx1) along it yields the longest RLE
runs. Vectorised ``xy2d`` (the standard iterative algorithm); for a non-power-of-2
tile the grid is embedded in the next power-of-2 square and only the in-range cells
are enumerated, which keeps locality intact.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _xy2d(p: int, x: npt.NDArray[np.int64], y: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Hilbert distance of ``(x, y)`` on a ``2^p`` square (vectorised, Wikipedia algo)."""
    n = 1 << p
    x = x.astype(np.int64, copy=True)
    y = y.astype(np.int64, copy=True)
    d = np.zeros_like(x)
    s = n >> 1
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        # rotate the quadrant (rot(n, x, y, rx, ry) from the reference)
        flip = (ry == 0) & (rx == 1)
        x[flip] = n - 1 - x[flip]
        y[flip] = n - 1 - y[flip]
        swap = ry == 0
        x[swap], y[swap] = y[swap], x[swap].copy()
        s >>= 1
    return d


def hilbert_order(h: int, w: int) -> npt.NDArray[np.intp]:
    """Permutation of the row-major ``h*w`` indices that traverses the grid by Hilbert curve.

    For non-power-of-2 ``h``/``w`` the curve is computed on the next power-of-2 square
    and the (still locality-preserving) order of the in-range cells is returned.
    """
    p = max(1, (max(h, w) - 1).bit_length())
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d = _xy2d(p, cc.ravel().astype(np.int64), rr.ravel().astype(np.int64))
    return np.argsort(d, kind="stable")
