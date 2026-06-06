"""Tests for WS-2 index-map compression: Morton, Hilbert, RLE, and the codec.

Space-filling curves are cross-checked against self-contained references (a naive
bit-interleave for Morton; the unit-step adjacency property for Hilbert) rather than
external packages, and RLE is checked bit-exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera_vq.entropy import rle_decode, rle_decode_stack, rle_encode, rle_encode_stack
from tessera_vq.hilbert import hilbert_order
from tessera_vq.index_codec import ORDERINGS, compress_index_map
from tessera_vq.morton import decode_morton2d, encode_morton2d, morton_order


def _naive_morton(r: int, c: int) -> int:
    """Reference Z-order code by interleaving bits (col -> even, row -> odd)."""
    code = 0
    for b in range(16):
        code |= ((c >> b) & 1) << (2 * b)
        code |= ((r >> b) & 1) << (2 * b + 1)
    return code


# ---- Morton ---------------------------------------------------------------


def test_morton_known_small_values() -> None:
    """The 2x2 Z-order codes are 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)."""
    r = np.array([0, 0, 1, 1])
    c = np.array([0, 1, 0, 1])
    assert encode_morton2d(r, c).tolist() == [0, 1, 2, 3]


def test_morton_matches_naive_reference_on_grid() -> None:
    """Vectorised encode equals the bit-by-bit reference across an 8x8 grid."""
    rr, cc = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    got = encode_morton2d(rr.ravel(), cc.ravel())
    want = [_naive_morton(int(r), int(c)) for r, c in zip(rr.ravel(), cc.ravel(), strict=True)]
    assert got.tolist() == want


def test_morton_roundtrip() -> None:
    """decode(encode(r, c)) == (r, c) on random coordinates up to 1024."""
    rng = np.random.default_rng(0)
    r = rng.integers(0, 1024, size=500)
    c = rng.integers(0, 1024, size=500)
    dr, dc = decode_morton2d(encode_morton2d(r, c))
    assert np.array_equal(dr.astype(np.int64), r)
    assert np.array_equal(dc.astype(np.int64), c)


def test_morton_order_is_a_permutation() -> None:
    """morton_order returns each row-major index exactly once."""
    perm = morton_order(6, 10)
    assert np.array_equal(np.sort(perm), np.arange(60))


# ---- Hilbert --------------------------------------------------------------


def test_hilbert_consecutive_cells_are_adjacent() -> None:
    """On a power-of-2 square, successive Hilbert cells differ by Manhattan distance 1."""
    h = w = 8
    perm = hilbert_order(h, w)
    rc = np.column_stack(np.unravel_index(perm, (h, w)))
    steps = np.abs(np.diff(rc, axis=0)).sum(axis=1)
    assert np.all(steps == 1)


def test_hilbert_order_is_a_permutation_even_non_pow2() -> None:
    """Non-power-of-2 grids still yield a full valid permutation."""
    perm = hilbert_order(6, 6)
    assert np.array_equal(np.sort(perm), np.arange(36))


# ---- RLE ------------------------------------------------------------------


def test_rle_roundtrip_random() -> None:
    """rle_decode(rle_encode(a)) == a for a run-heavy random array."""
    rng = np.random.default_rng(1)
    a = np.repeat(rng.integers(0, 5, size=200), rng.integers(1, 6, size=200))
    values, lengths = rle_encode(a)
    assert lengths.sum() == a.size
    assert np.array_equal(rle_decode(values, lengths), a)


def test_rle_empty_and_single_run() -> None:
    """Empty input -> empty output; a constant array -> one run."""
    v, ln = rle_encode(np.zeros(0, np.int32))
    assert v.size == 0 and ln.size == 0
    v, ln = rle_encode(np.full(50, 7, np.int32))
    assert v.tolist() == [7] and ln.tolist() == [50]


def test_rle_stack_roundtrip_preserves_dtype_and_values() -> None:
    """Encoding then decoding a stack of index maps reproduces it exactly (wire format)."""
    rng = np.random.default_rng(5)
    stack = rng.integers(0, 6, size=(4, 8, 8)).astype(np.uint8)
    stack[0] = 3  # a fully constant tile (one run) alongside noisy ones
    values, lengths, runs = rle_encode_stack(stack)
    assert runs.shape == (4,)
    assert runs[0] == 1  # constant tile collapses to a single run
    out = rle_decode_stack(values, lengths, runs, 8, 8)
    assert out.dtype == stack.dtype
    assert np.array_equal(out, stack)


def test_rle_stack_empty() -> None:
    """A zero-tile stack round-trips to an empty (0, h, w) array."""
    values, lengths, runs = rle_encode_stack(np.zeros((0, 8, 8), np.uint8))
    assert runs.size == 0
    assert rle_decode_stack(values, lengths, runs, 8, 8).shape == (0, 8, 8)


# ---- Codec ----------------------------------------------------------------


def _quadrant_map(side: int) -> np.ndarray:
    """side x side map of four constant quadrants (a 2-D-contiguous structure)."""
    m = np.zeros((side, side), dtype=np.int64)
    h = side // 2
    m[:h, h:] = 1
    m[h:, :h] = 2
    m[h:, h:] = 3
    return m


def test_constant_map_is_one_run_everywhere() -> None:
    """A homogeneous tile collapses to a single run under every ordering."""
    m = np.full((32, 32), 9, dtype=np.int64)
    for ordering in ORDERINGS:
        res = compress_index_map(m, n_symbols=64, ordering=ordering)
        assert res.n_runs == 1
        assert res.rle_bytes_per_px < res.raw_bytes_per_px  # RLE wins big on homogeneous data


def test_space_filling_beats_row_major_on_blocky_map() -> None:
    """Z-order and Hilbert give fewer runs than row-major on 2-D-contiguous regions."""
    m = _quadrant_map(8)
    row = compress_index_map(m, n_symbols=4, ordering="row")
    mort = compress_index_map(m, n_symbols=4, ordering="morton")
    hil = compress_index_map(m, n_symbols=4, ordering="hilbert")
    assert mort.n_runs < row.n_runs
    assert hil.n_runs < row.n_runs


def test_raw_bytes_is_byte_aligned_symbol_width() -> None:
    """raw_bytes_per_px is the byte-aligned plane width (1 byte for k <= 256)."""
    m = _quadrant_map(16)
    assert compress_index_map(m, n_symbols=64).raw_bytes_per_px == 1.0
    assert compress_index_map(m, n_symbols=256).raw_bytes_per_px == 1.0
    assert compress_index_map(m, n_symbols=1024).raw_bytes_per_px == 2.0


def test_unknown_ordering_raises() -> None:
    """A bad ordering name is rejected."""
    with pytest.raises(ValueError, match="unknown ordering"):
        compress_index_map(_quadrant_map(8), n_symbols=4, ordering="spiral")
