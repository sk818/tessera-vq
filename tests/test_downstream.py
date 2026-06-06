"""Tests for WS-3 downstream plumbing: block RVQ, labelled extraction, spatial split.

Synthetic arrays only -- no geotessera / shapefiles. The geotessera + tessera_eval
wiring is exercised by the script, not here.
"""

from __future__ import annotations

import numpy as np

from tessera_vq.downstream import (
    extract_labelled,
    reconstruct_tile_blocks,
    spatial_group_kfold,
    spatial_group_split,
)
from tessera_vq.rvq_large import rvq_reconstruct_flat


def _clustered(n: int, c: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((k, c)).astype(np.float32) * 10.0
    labels = rng.integers(0, k, size=n)
    return (centres[labels] + 0.2 * rng.standard_normal((n, c))).astype(np.float32)


def _clustered_tile(h: int, w: int, c: int, k: int, seed: int) -> np.ndarray:
    return _clustered(h * w, c, k, seed).reshape(h, w, c)


def test_rvq_reconstruct_flat_shape_and_quality() -> None:
    """Flat RVQ returns (M, C) and reconstructs clustered data to ~noise scale."""
    x = _clustered(4000, 32, k=8, seed=0)
    rec = rvq_reconstruct_flat(x, k1=8, k2=8, seed=42)
    assert rec.shape == x.shape
    err = float(np.mean(np.linalg.norm(x - rec, axis=1)))
    assert err < 2.0  # noise L2 ~ 0.2*sqrt(32) ~= 1.1


def test_reconstruct_tile_blocks_all_finite() -> None:
    """A finite tile reconstructs everywhere with the right shape and low error."""
    tile = _clustered_tile(64, 64, 16, k=6, seed=1)
    recon = reconstruct_tile_blocks(tile, t=32, k1=8, k2=16, seed=42)
    assert recon.shape == tile.shape
    assert np.all(np.isfinite(recon))
    assert float(np.mean(np.linalg.norm(tile - recon, axis=-1))) < 2.0


def test_int8_codebooks_preserve_reconstruction() -> None:
    """Reconstructing from int8-served codebooks barely changes the error vs float32."""
    tile = _clustered_tile(64, 64, 16, k=6, seed=7)
    r32 = reconstruct_tile_blocks(tile, t=32, k1=8, k2=16, seed=42)
    r8 = reconstruct_tile_blocks(tile, t=32, k1=8, k2=16, seed=42, quantize_codebooks=True)
    assert r8.shape == r32.shape and np.all(np.isfinite(r8))
    e32 = float(np.mean(np.linalg.norm(tile - r32, axis=-1)))
    e8 = float(np.mean(np.linalg.norm(tile - r8, axis=-1)))
    assert e8 <= e32 * 1.5 + 0.1  # int8 adds only sub-quant noise, no blow-up


def test_reconstruct_tile_blocks_leaves_nodata_nan() -> None:
    """A fully-NaN block stays NaN; finite blocks are reconstructed."""
    tile = _clustered_tile(64, 64, 16, k=6, seed=2)
    tile[:32, :32] = np.nan  # one no-data block
    recon = reconstruct_tile_blocks(tile, t=32, k1=8, k2=16, seed=42)
    assert np.all(np.isnan(recon[:32, :32]))
    assert np.all(np.isfinite(recon[32:, 32:]))


def test_reconstruct_tile_blocks_skips_too_sparse_block() -> None:
    """A block with <= k1 finite pixels is left NaN (cannot fit stage-1)."""
    tile = _clustered_tile(64, 64, 16, k=6, seed=3)
    blk = tile[:32, :32].reshape(-1, 16)
    blk[5:] = np.nan  # only 5 finite pixels, <= k1=8
    tile[:32, :32] = blk.reshape(32, 32, 16)
    recon = reconstruct_tile_blocks(tile, t=32, k1=8, k2=16, seed=42)
    assert np.all(np.isnan(recon[:32, :32]))


def test_reconstruct_tile_blocks_handles_partial_edge_blocks() -> None:
    """Non-multiple-of-t tiles reconstruct the partial edge blocks without error."""
    tile = _clustered_tile(70, 70, 8, k=4, seed=4)
    recon = reconstruct_tile_blocks(tile, t=32, k1=4, k2=8, seed=42)
    assert recon.shape == (70, 70, 8)
    assert np.all(np.isfinite(recon))


def test_extract_labelled_keeps_labelled_finite_pixels() -> None:
    """Only labelled (class>0) and finite-in-both pixels survive; labels go 0-based."""
    emb = _clustered_tile(8, 8, 4, k=3, seed=5)
    recon = emb.copy()
    cls = np.zeros((8, 8), dtype=np.int32)
    cls[0, 0] = 1  # class 0 after 0-basing
    cls[1, 1] = 3  # class 2
    cls[2, 2] = 2  # class 1 -- but make it NaN to test dropping
    emb[2, 2] = np.nan
    raw, rec, labels = extract_labelled(emb, recon, cls)
    assert raw.shape == (2, 4) and rec.shape == (2, 4)
    assert sorted(labels.tolist()) == [0, 2]  # the NaN-dropped class-1 pixel is gone


def test_spatial_group_split_holds_out_whole_groups() -> None:
    """No group appears in both train and test; test fraction ~ requested."""
    groups = np.repeat(np.arange(10), 50)  # 10 groups of 50 pixels
    train, test = spatial_group_split(groups, test_frac=0.3, seed=42)
    assert train.sum() + test.sum() == groups.size
    train_groups = set(groups[train].tolist())
    test_groups = set(groups[test].tolist())
    assert train_groups.isdisjoint(test_groups)
    assert len(test_groups) == 3  # 30% of 10 groups


def test_spatial_group_split_single_group_is_all_train() -> None:
    """One group cannot be split spatially -> all train, empty test."""
    groups = np.zeros(100, dtype=np.int64)
    train, test = spatial_group_split(groups, test_frac=0.3, seed=1)
    assert train.all()
    assert not test.any()


def test_spatial_group_split_is_deterministic() -> None:
    """Same seed -> same split."""
    groups = np.repeat(np.arange(8), 10)
    a = spatial_group_split(groups, seed=7)
    b = spatial_group_split(groups, seed=7)
    assert np.array_equal(a[1], b[1])


def test_spatial_kfold_partitions_groups_across_test_folds() -> None:
    """Each group is the test set in exactly one fold; train/test disjoint per fold."""
    groups = np.repeat(np.arange(8), 10)  # 8 groups
    folds = spatial_group_kfold(groups, n_folds=4, seed=1)
    assert len(folds) == 4
    seen_test: set[int] = set()
    for train, test in folds:
        assert not (train & test).any()  # disjoint
        assert (train | test).all()  # cover everything
        tg = set(groups[test].tolist())
        assert tg.isdisjoint(seen_test)  # no group tested twice
        seen_test |= tg
    assert seen_test == set(range(8))  # every group tested once


def test_spatial_kfold_caps_folds_at_group_count() -> None:
    """n_folds larger than the number of groups is capped (Cumbria has few tiles)."""
    groups = np.repeat(np.arange(3), 20)  # only 3 groups
    folds = spatial_group_kfold(groups, n_folds=10, seed=2)
    assert len(folds) == 3


def test_spatial_kfold_single_group_is_empty() -> None:
    """One group cannot be spatially folded -> no folds."""
    assert spatial_group_kfold(np.zeros(50, dtype=np.int64), n_folds=4) == []
