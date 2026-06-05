"""Phase 3 index-map compression sweep (WS-2; spec Phase 4).

Measures how far the stage-1 index map (idx1) compresses under a space-filling-curve
traversal + RLE, and confirms the stage-2 residual index (idx2) is ~incompressible.
For each sampled bbox it reads a ~12 km window, selects the most-finite tiles, runs
two-stage RVQ (``tessera_vq.rvq_large``), and for every ``(t, k1, k2)`` cell records:

- ``idx1_{row,morton,hilbert}_bpp`` -- bits/px of idx1 after that ordering + RLE
  (conservative fixed-width run model; see ``tessera_vq.index_codec``);
- ``idx1_raw_bpp`` = ceil(log2 k1), ``idx2_raw_bpp`` = ceil(log2 k2);
- ``idx2_hilbert_bpp`` -- idx2 after Hilbert+RLE (should ~= raw: residual is white);
- ``codebook_Bpx`` = (k1 + k2) * 128 / t^2 (1 byte/dim per code);
- ``total_packed_Bpx`` (raw 16-bit index) and ``total_compressed_Bpx`` (best idx1
  ordering + incompressible idx2), with compression ratios vs fp32/int8 raw.

Aggregated per cell as the mean across tiles. Streams one window at a time and
checkpoints the provenance-tagged Parquet after every bbox.

Output (``--out-dir``, default ``results/phase3/``): ``{tag}_index_compression.parquet``.

Run::

    uv run python scripts/phase3_index_compression.py --n-bboxes 10 --tag idx_v1
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from tessera_vq.canonical import CanonicalBbox, load_canonical_bboxes, read_canonical_window
from tessera_vq.index_codec import compress_index_map
from tessera_vq.io_utils import write_parquet_with_provenance
from tessera_vq.rvq_large import rvq_reconstruct_large
from tessera_vq.tiling import extract_finite_tiles

logger = logging.getLogger(__name__)

DEFAULT_TILE_SIZES: tuple[int, ...] = (512, 768, 1024)
DEFAULT_CONFIGS: tuple[str, ...] = ("64:1024", "128:512", "256:256")
_ORDERINGS = ("row", "morton", "hilbert")

Cell = tuple[int, int, int]


@dataclass(frozen=True)
class _RunCfg:
    """Per-run knobs threaded through the streaming loop."""

    year: int
    window_px: int
    tiles_per_bbox: int
    min_finite_frac: float
    max_nan_fraction: float
    seed: int


def parse_args() -> argparse.Namespace:
    """Parse CLI flags (mirrors the large-recon sweep)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--config", default="config/canonical_bboxes.yaml")
    p.add_argument("--n-bboxes", type=int, default=10)
    p.add_argument("--tag", default="idx_v1")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/phase3")
    p.add_argument("--window-px", type=int, default=1200)
    p.add_argument("--tiles-per-bbox", type=int, default=2)
    p.add_argument("--min-finite-frac", type=float, default=1.0)
    p.add_argument("--max-nan-fraction", type=float, default=0.5)
    p.add_argument("--tile-sizes", type=int, nargs="+", default=list(DEFAULT_TILE_SIZES))
    p.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    return p.parse_args()


def _build_cells(tile_sizes: list[int], configs: list[str]) -> list[Cell]:
    """``(t, k1, k2)`` grid: tile sizes x specific (k1, k2) pairs, dropping k >= t*t."""
    pairs = [(int(a), int(b)) for a, b in (c.split(":") for c in configs)]
    return [(t, k1, k2) for t in tile_sizes for k1, k2 in pairs if k1 < t * t and k2 < t * t]


def _sample_bboxes(all_bboxes: list[CanonicalBbox], n: int, seed: int) -> list[CanonicalBbox]:
    """Pick ``n`` bboxes uniformly at random (no replacement), in original-index order."""
    if n >= len(all_bboxes):
        return all_bboxes
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(all_bboxes), size=n, replace=False))
    return [all_bboxes[int(i)] for i in idx]


def _tile_index_metrics(
    idx1: npt.NDArray[np.int32], k1: int, idx2: npt.NDArray[np.int32], k2: int
) -> dict[str, float]:
    """idx1 bits/px under each ordering + the idx2 (in)compressibility check."""
    out: dict[str, float] = {}
    for o in _ORDERINGS:
        out[f"idx1_{o}_bpp"] = compress_index_map(idx1, k1, o).rle_bpp
    out["idx1_raw_bpp"] = compress_index_map(idx1, k1, "row").raw_bpp
    out["idx2_raw_bpp"] = compress_index_map(idx2, k2, "row").raw_bpp
    out["idx2_hilbert_bpp"] = compress_index_map(idx2, k2, "hilbert").rle_bpp
    return out


def _accumulate(
    acc: dict[Cell, list[dict[str, float]]],
    cells: list[Cell],
    window: npt.NDArray[np.float32],
    cfg: _RunCfg,
) -> None:
    """Sweep every cell on one window; append per-tile index-compression metrics."""
    tiles_by_t: dict[int, list[npt.NDArray[np.float32]]] = {}
    for t in {c[0] for c in cells}:
        samples = extract_finite_tiles(
            window,
            t,
            n_tiles=cfg.tiles_per_bbox,
            seed=cfg.seed,
            min_finite_frac=cfg.min_finite_frac,
        )
        tiles_by_t[t] = [s.tile for s in samples]
    for t, k1, k2 in cells:
        for tile in tiles_by_t[t]:
            res = rvq_reconstruct_large(tile, k1, k2, seed=cfg.seed)
            acc[(t, k1, k2)].append(_tile_index_metrics(res.indices1, k1, res.indices2, k2))


def _cell_row(cell: Cell, per_tile: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate one cell's per-tile metrics + derive effective bytes/px and ratios."""
    t, k1, k2 = cell
    row: dict[str, float] = {"t": float(t), "k1": float(k1), "k2": float(k2)}
    row["n_tiles"] = float(len(per_tile))
    if not per_tile:
        return row
    for key in per_tile[0]:
        row[key] = float(np.mean([m[key] for m in per_tile]))
    cb_bpx = (k1 + k2) * 128 / (t * t)
    idx1_best = min(row[f"idx1_{o}_bpp"] for o in _ORDERINGS)
    row["codebook_Bpx"] = cb_bpx
    row["idx1_best_bpp"] = idx1_best
    row["total_packed_Bpx"] = cb_bpx + (row["idx1_raw_bpp"] + row["idx2_raw_bpp"]) / 8.0
    row["total_compressed_Bpx"] = cb_bpx + idx1_best / 8.0 + row["idx2_raw_bpp"] / 8.0
    row["x_fp32_compressed"] = 512.0 / row["total_compressed_Bpx"]
    row["x_int8_compressed"] = 128.0 / row["total_compressed_Bpx"]
    return row


def main() -> None:
    """Stream bboxes, measure index compression per cell, checkpoint Parquet per bbox."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    cells = _build_cells(list(args.tile_sizes), list(args.configs))
    if not cells:
        logger.error("empty grid from tile_sizes=%s configs=%s", args.tile_sizes, args.configs)
        sys.exit(1)
    bboxes = _sample_bboxes(load_canonical_bboxes(args.config), args.n_bboxes, args.seed)
    cfg = _RunCfg(
        year=args.year,
        window_px=args.window_px,
        tiles_per_bbox=args.tiles_per_bbox,
        min_finite_frac=args.min_finite_frac,
        max_nan_fraction=args.max_nan_fraction,
        seed=args.seed,
    )
    logger.info(
        "sampled %d bboxes (seed=%d); %d cells; window_px=%d",
        len(bboxes),
        args.seed,
        len(cells),
        args.window_px,
    )
    acc: dict[Cell, list[dict[str, float]]] = {c: [] for c in cells}
    out_path = f"{args.out_dir}/{args.tag}_index_compression.parquet"
    for i, bbox in enumerate(bboxes):
        window, path = read_canonical_window(
            bbox, cfg.year, window_px=cfg.window_px, max_nan_fraction=cfg.max_nan_fraction
        )
        if window is None:
            logger.warning("bbox %d/%d %s unavailable; skipping", i + 1, len(bboxes), bbox.name)
            continue
        _accumulate(acc, cells, window, cfg)
        del window
        rows = [_cell_row(c, acc[c]) for c in cells]
        write_parquet_with_provenance(
            pd.DataFrame(rows), out_path, seed=args.seed, config_path=args.config
        )
        logger.info(
            "bbox %d/%d %s done (%s); checkpoint written", i + 1, len(bboxes), bbox.name, path
        )
    logger.info("sweep complete -> %s", out_path)


if __name__ == "__main__":
    main()
