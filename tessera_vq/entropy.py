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


def rle_encode_stack(
    stack: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Row-major RLE a stack of index maps ``(n, h, w)`` for the wire format.

    Returns ``(values, lengths, runs)``: ``values``/``lengths`` are the concatenated
    runs of every tile (in tile order), ``runs[i]`` is the number of runs in tile ``i``
    (so a decoder can slice them apart). ``values`` keep the input dtype (compact on
    the wire); ``lengths`` and ``runs`` are int64.
    """
    vals: list[npt.NDArray[np.integer]] = []
    lens: list[npt.NDArray[np.int64]] = []
    runs = np.empty(stack.shape[0], dtype=np.int64)
    for i in range(stack.shape[0]):
        v, ln = rle_encode(stack[i].ravel())
        vals.append(v.astype(stack.dtype, copy=False))
        lens.append(ln)
        runs[i] = v.size
    values = np.concatenate(vals) if vals else np.zeros(0, stack.dtype)
    lengths = np.concatenate(lens) if lens else np.zeros(0, np.int64)
    return values, lengths, runs


def rle_decode_stack(
    values: npt.NDArray[np.integer],
    lengths: npt.NDArray[np.integer],
    runs: npt.NDArray[np.integer],
    h: int,
    w: int,
) -> npt.NDArray[np.integer]:
    """Inverse of :func:`rle_encode_stack`; rebuild the ``(n, h, w)`` stack."""
    n = int(runs.size)
    out = np.empty((n, h, w), dtype=values.dtype)
    off = 0
    for i in range(n):
        r = int(runs[i])
        out[i] = rle_decode(values[off : off + r], lengths[off : off + r]).reshape(h, w)
        off += r
    return out
