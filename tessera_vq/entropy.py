"""Run-length encoding for linearised index maps (WS-2).

After a space-filling-curve traversal, the stage-1 index stream is long constant
runs; RLE collapses each run to ``(value, length)``. Pure numpy and bit-exact
(``rle_decode(rle_encode(a)) == a``). zstd is intentionally not included here to
avoid a new dependency; the byte model in ``index_codec`` is a conservative
fixed-width run cost, so an entropy coder would only do better.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def rle_encode(
    a: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Run-length encode a 1-D integer array; returns ``(values, lengths)``."""
    flat = np.asarray(a).ravel()
    if flat.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    change = np.flatnonzero(flat[1:] != flat[:-1])
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change, [flat.size - 1]))
    values = flat[starts].astype(np.int64)
    lengths = (ends - starts + 1).astype(np.int64)
    return values, lengths


def rle_decode(
    values: npt.NDArray[np.integer], lengths: npt.NDArray[np.integer]
) -> npt.NDArray[np.int64]:
    """Inverse of :func:`rle_encode`."""
    return np.repeat(np.asarray(values, np.int64), np.asarray(lengths, np.int64))
