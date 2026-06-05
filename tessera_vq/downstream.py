"""Downstream-task plumbing: apply per-tile RVQ, extract labelled pixels, split (WS-3).

To measure VQ's impact on land-cover/crop classification we reconstruct each Tessera
tile through per-``t``-block RVQ, then pull the labelled pixels (raw and reconstructed)
and feed both to the same classifier. This module holds the pure, testable pieces; the
geotessera + ``tessera_eval`` wiring lives in ``scripts/phase5_downstream.py``.

- ``reconstruct_tile_blocks`` -- tile the (H, W, C) embedding into ``t x t`` blocks and
  RVQ each on its finite pixels (NaN no-data left as NaN); this is the serving-time
  compression applied before any pixel is read.
- ``extract_labelled`` -- gather raw + reconstructed vectors at finite labelled pixels.
- ``spatial_group_split`` -- hold out whole spatial groups (tiles/blocks) as the test
  set, the honest split for autocorrelated pixels (random k-fold leaks neighbours).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tessera_vq.rvq_large import rvq_reconstruct_flat


def reconstruct_tile_blocks(
    emb: npt.NDArray[np.float32],
    t: int,
    k1: int,
    k2: int,
    *,
    seed: int = 42,
    n_iter: int = 25,
) -> npt.NDArray[np.float32]:
    """RVQ-reconstruct an ``(H, W, C)`` tile block-by-block (``t x t``); NaN stays NaN.

    Each block is quantised on its own finite pixels (a block with <= k1 finite pixels
    is left as NaN -- too few to fit a stage-1 codebook). Partial edge blocks are
    handled naturally (``k_eff`` caps at the block's pixel count).
    """
    h, w, c = emb.shape
    recon = np.full((h, w, c), np.nan, dtype=np.float32)
    for r0 in range(0, h, t):
        for c0 in range(0, w, t):
            block = emb[r0 : r0 + t, c0 : c0 + t]
            flat = block.reshape(-1, c)
            finite = ~np.isnan(flat).any(axis=1)
            if int(finite.sum()) <= k1:
                continue
            rec = np.full_like(flat, np.nan)
            rec[finite] = rvq_reconstruct_flat(flat[finite], k1, k2, seed=seed, n_iter=n_iter)
            recon[r0 : r0 + t, c0 : c0 + t] = rec.reshape(block.shape)
    return recon


def extract_labelled(
    emb: npt.NDArray[np.float32],
    recon: npt.NDArray[np.float32],
    class_raster: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """Pull (raw, reconstructed, label) at pixels that are labelled *and* finite in both.

    ``class_raster`` is 1-based (0 = nodata, per ``rasterize_shapefile``); returned
    labels are 0-based. Pixels NaN in either the raw tile or the reconstruction (e.g.
    no-data, or a block too sparse to reconstruct) are dropped.
    """
    labelled = class_raster > 0
    finite = ~np.isnan(emb).any(axis=-1) & ~np.isnan(recon).any(axis=-1)
    mask = labelled & finite
    raw = emb[mask].astype(np.float32, copy=False)
    rec = recon[mask].astype(np.float32, copy=False)
    labels = (class_raster[mask] - 1).astype(np.int64)
    return raw, rec, labels


def spatial_group_split(
    groups: npt.NDArray[np.integer], test_frac: float = 0.3, seed: int = 42
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Hold out whole groups (e.g. tiles) for test; return ``(train_mask, test_mask)``.

    Groups are assigned entirely to train or test, so no pixel's spatial neighbour
    straddles the split. ``test_frac`` is the target fraction of *groups* held out
    (at least one group each side when there are >= 2 groups).
    """
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_test = max(1, min(len(uniq) - 1, round(test_frac * len(uniq)))) if len(uniq) > 1 else 0
    test_groups = set(perm[:n_test].tolist())
    test_mask = np.isin(groups, list(test_groups))
    return ~test_mask, test_mask
