"""Morton (Z-order) curve encode/decode and grid traversal order (WS-2).

Used to linearise a 2-D index map before run-length encoding: traversing pixels in
Z-order keeps spatially-near pixels near in the 1-D stream, so the spatially
autocorrelated stage-1 index map (idx1) forms long runs.

Vectorised bit-interleaving via the standard "magic number" masks; supports
coordinates up to 16 bits (tile sizes well beyond 1024), codes up to 32 bits held
in uint64.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_M16 = np.uint64(0x0000FFFF)
_M08 = np.uint64(0x00FF00FF)
_M04 = np.uint64(0x0F0F0F0F)
_M02 = np.uint64(0x33333333)
_M01 = np.uint64(0x55555555)


def _part1by1(n: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Spread the low 16 bits of each value so one zero bit sits between them."""
    n = n & _M16
    n = (n | (n << np.uint64(8))) & _M08
    n = (n | (n << np.uint64(4))) & _M04
    n = (n | (n << np.uint64(2))) & _M02
    return (n | (n << np.uint64(1))) & _M01


def _compact1by1(x: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Inverse of :func:`_part1by1`: gather every other bit back down."""
    x = x & _M01
    x = (x | (x >> np.uint64(1))) & _M02
    x = (x | (x >> np.uint64(2))) & _M04
    x = (x | (x >> np.uint64(4))) & _M08
    return (x | (x >> np.uint64(8))) & _M16


def encode_morton2d(
    rows: npt.NDArray[np.integer], cols: npt.NDArray[np.integer]
) -> npt.NDArray[np.uint64]:
    """Interleave ``(row, col)`` into Z-order codes (col -> even bits, row -> odd)."""
    r = np.asarray(rows, dtype=np.uint64)
    c = np.asarray(cols, dtype=np.uint64)
    return _part1by1(c) | (_part1by1(r) << np.uint64(1))


def decode_morton2d(
    codes: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Inverse of :func:`encode_morton2d`; returns ``(rows, cols)``."""
    z = np.asarray(codes, dtype=np.uint64)
    cols = _compact1by1(z)
    rows = _compact1by1(z >> np.uint64(1))
    return rows, cols


def morton_order(h: int, w: int) -> npt.NDArray[np.intp]:
    """Permutation of the row-major ``h*w`` indices that traverses the grid in Z-order."""
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    codes = encode_morton2d(rr.ravel(), cc.ravel())
    return np.argsort(codes, kind="stable")
