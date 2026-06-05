"""Index-map compression: traversal order + byte-aligned RLE -> bytes/pixel (WS-2).

A deployable system stores idx1 and idx2 as *separate byte planes* (not bit-packed
interleaved): RLE must be applied to idx1 alone, because the stage-2 residual index
idx2 is spatially white and would destroy runs in a combined stream. Each plane is
byte-addressable (k <= 256 -> 1 byte/symbol), which is also what makes RLE practical.

Byte model for the idx1 plane after a space-filling-curve traversal + RLE:

- each run costs ``sym_bytes + varint(run_length)`` bytes, where ``sym_bytes =
  ceil(ceil(log2 n_symbols) / 8)`` (1 byte for k <= 256) and the run length uses an
  LEB128 varint (ceil(bits/7) bytes, >= 1);
- ``raw_bytes_per_px`` is the uncompressed byte plane (``sym_bytes``);
- ``rle_bytes_per_px`` is the RLE'd plane.

idx2 is treated as an incompressible raw byte plane elsewhere (``raw_bytes_per_px``).
The honest cross-ordering signal remains ``n_runs`` (fewer runs = better locality).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tessera_vq.entropy import rle_encode
from tessera_vq.hilbert import hilbert_order
from tessera_vq.morton import morton_order


def _row_order(h: int, w: int) -> npt.NDArray[np.intp]:
    """Trivial row-major traversal (identity permutation)."""
    return np.arange(h * w, dtype=np.intp)


ORDERINGS: dict[str, Callable[[int, int], npt.NDArray[np.intp]]] = {
    "row": _row_order,
    "morton": morton_order,
    "hilbert": hilbert_order,
}


@dataclass(frozen=True)
class IndexCompression:
    """Compression summary for one (index map, ordering) pair, in bytes/pixel."""

    ordering: str
    n_px: int
    n_symbols: int
    n_runs: int
    runs_per_px: float
    raw_bytes_per_px: float  # uncompressed byte plane
    rle_bytes_per_px: float  # space-filling order + byte-aligned RLE


def _sym_bytes(n_symbols: int) -> int:
    """Bytes to store one symbol of a ``n_symbols``-ary alphabet (byte-aligned)."""
    sym_bits = max(1, (max(int(n_symbols), 1) - 1).bit_length()) if n_symbols > 1 else 1
    return (sym_bits + 7) // 8


def _varint_bytes(lengths: npt.NDArray[np.integer]) -> int:
    """Total LEB128 varint bytes for an array of (>=1) run lengths."""
    if lengths.size == 0:
        return 0
    bits = np.floor(np.log2(lengths.astype(np.float64))).astype(np.int64) + 1
    per_run = np.maximum(1, (bits + 6) // 7)
    return int(per_run.sum())


def compress_index_map(
    idx_map: npt.NDArray[np.integer], n_symbols: int, ordering: str = "hilbert"
) -> IndexCompression:
    """Order ``idx_map`` by ``ordering`` then byte-aligned RLE; report runs and bytes/px."""
    if ordering not in ORDERINGS:
        raise ValueError(f"unknown ordering {ordering!r}; choose from {sorted(ORDERINGS)}")
    h, w = idx_map.shape
    n_px = h * w
    perm = ORDERINGS[ordering](h, w)
    seq = idx_map.ravel()[perm]
    values, lengths = rle_encode(seq)
    n_runs = int(values.size)
    sym_bytes = _sym_bytes(n_symbols)
    rle_bytes = n_runs * sym_bytes + _varint_bytes(lengths)
    return IndexCompression(
        ordering=ordering,
        n_px=n_px,
        n_symbols=n_symbols,
        n_runs=n_runs,
        runs_per_px=n_runs / n_px if n_px else 0.0,
        raw_bytes_per_px=float(sym_bytes),
        rle_bytes_per_px=rle_bytes / n_px if n_px else 0.0,
    )
