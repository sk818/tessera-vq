"""Index-map compression: traversal order + RLE -> bits/pixel (WS-2).

Ties together the orderings (row-major, Z-order, Hilbert) and RLE to measure how
compressible a stage-1 index map (idx1) is. The stage-2 residual index (idx2) is
treated as incompressible elsewhere -- it is spatially white -- so only idx1 is run
through here.

Byte model (deliberately conservative): a run costs ``sym_bits + len_bits`` where
``sym_bits = ceil(log2 n_symbols)`` and ``len_bits = ceil(log2(n_px + 1))`` covers
any run length with a fixed-width field. A real codec (varint / entropy-coded run
lengths) would beat this, so ``rle_bpp`` is an upper bound on what RLE achieves.
The honest cross-ordering signal is ``n_runs`` (fewer runs = better locality).
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
    """Compression summary for one (index map, ordering) pair."""

    ordering: str
    n_px: int
    n_symbols: int
    n_runs: int
    runs_per_px: float
    raw_bpp: float  # bits/px of the packed index, no RLE
    rle_bpp: float  # bits/px after order + RLE (conservative model)


def _bits(n: int) -> int:
    """Minimum bits to represent ``n`` distinct symbols (>= 1)."""
    return max(1, (max(int(n), 1) - 1).bit_length()) if n > 1 else 1


def compress_index_map(
    idx_map: npt.NDArray[np.integer], n_symbols: int, ordering: str = "hilbert"
) -> IndexCompression:
    """Order ``idx_map`` by ``ordering`` then RLE; report runs and bits/pixel."""
    if ordering not in ORDERINGS:
        raise ValueError(f"unknown ordering {ordering!r}; choose from {sorted(ORDERINGS)}")
    h, w = idx_map.shape
    n_px = h * w
    perm = ORDERINGS[ordering](h, w)
    seq = idx_map.ravel()[perm]
    values, _lengths = rle_encode(seq)
    n_runs = int(values.size)
    sym_bits = _bits(n_symbols)
    len_bits = max(1, (n_px).bit_length())  # fixed-width field covering any run length
    rle_bits = n_runs * (sym_bits + len_bits)
    return IndexCompression(
        ordering=ordering,
        n_px=n_px,
        n_symbols=n_symbols,
        n_runs=n_runs,
        runs_per_px=n_runs / n_px if n_px else 0.0,
        raw_bpp=float(sym_bits),
        rle_bpp=rle_bits / n_px if n_px else 0.0,
    )
