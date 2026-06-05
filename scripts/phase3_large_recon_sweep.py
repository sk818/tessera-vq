"""Phase 3 (large-tile) reconstruction sweep with an anchor-free L2 metric (WS-1).

Replaces the frozen-histogram sweep, whose bin edges were anchored arbitrarily to
the first bbox and so produced run-unstable "tail" numbers. Here every metric is
scale-free and comparable across runs (see ``tessera_vq.metrics``):

- per-pixel relative L2 error ``||x - x_hat|| / ||x||`` (mean + 50/90/99 pct),
- R^2, the fraction of variance the RVQ reconstruction explains.

Pipeline per bbox: read a ~12 km window at the canonical centre (``--window-px``,
default 1200), pick the most-finite ``--tiles-per-bbox`` tiles per tile size
(jittered +-t/2 about the centre; ``tessera_vq.tiling``), run two-stage RVQ on each
(``tessera_vq.rvq_large`` on the BLAS-GEMM k-means), and score it. Results are
aggregated per ``(t, k1, k2)`` cell as mean +- sd across tiles.

Grid (locked): t in {512, 768, 1024} x (k1, k2) in {(64,1024),(128,512),(256,256)},
all 16-bit-packable index configs. Override with ``--tile-sizes`` / ``--configs``.

Memory: exactly one window is resident at a time (freed before the next read); only
the tiny per-tile metric dicts accumulate. The Parquet output is rewritten after
every bbox as a checkpoint, so an interruption still leaves a valid partial result.

Outputs (``--out-dir``, default ``results/phase3/``): ``{tag}_large_recon.parquet``
with one row per ``(t, k1, k2)`` cell plus provenance columns.

Run::

    uv run python scripts/phase3_large_recon_sweep.py --n-bboxes 10 --tag large_v1
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
from tessera_vq.io_utils import write_parquet_with_provenance
from tessera_vq.metrics import aggregate_reconstruction_metrics, reconstruction_metrics
from tessera_vq.rvq_large import rvq_reconstruct_large
from tessera_vq.tiling import extract_finite_tiles

logger = logging.getLogger(__name__)

DEFAULT_TILE_SIZES: tuple[int, ...] = (512, 768, 1024)
DEFAULT_CONFIGS: tuple[str, ...] = ("64:1024", "128:512", "256:256")

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
    """Parse CLI flags. ``--n-bboxes 10`` is the standard run (1-2 tiles each)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--config", default="config/canonical_bboxes.yaml")
    p.add_argument("--n-bboxes", type=int, default=10, help="bboxes to sample (default 10)")
    p.add_argument("--tag", default="large_v1", help="output filename tag")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/phase3")
    p.add_argument("--window-px", type=int, default=1200, help="~12 km window (default 1200)")
    p.add_argument("--tiles-per-bbox", type=int, default=2, help="most-finite tiles kept per size")
    p.add_argument("--min-finite-frac", type=float, default=1.0, help="reject NaN-heavy tiles")
    p.add_argument("--max-nan-fraction", type=float, default=0.5, help="window-read NaN gate")
    p.add_argument(
        "--tile-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_TILE_SIZES),
        help="tile sizes in pixels (default: %(default)s)",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="k1:k2 index configs (default: %(default)s)",
    )
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


def _accumulate(
    acc: dict[Cell, list[dict[str, float]]],
    cells: list[Cell],
    window: npt.NDArray[np.float32],
    cfg: _RunCfg,
) -> None:
    """Sweep every cell on one window; append this window's per-tile recon metrics."""
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
            acc[(t, k1, k2)].append(reconstruction_metrics(tile, res.recon))


def _rows_from_acc(acc: dict[Cell, list[dict[str, float]]]) -> list[dict[str, float]]:
    """One aggregated row per cell (mean +- sd of each metric across tiles)."""
    rows: list[dict[str, float]] = []
    for (t, k1, k2), per_tile in acc.items():
        row: dict[str, float] = {"t": float(t), "k1": float(k1), "k2": float(k2)}
        row.update(aggregate_reconstruction_metrics(per_tile))
        rows.append(row)
    return rows


def main() -> None:
    """Stream bboxes, sweep cells, checkpoint the aggregate Parquet after each bbox."""
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
        "sampled %d bboxes (seed=%d); %d cells; window_px=%d; streaming one at a time",
        len(bboxes),
        args.seed,
        len(cells),
        args.window_px,
    )
    acc: dict[Cell, list[dict[str, float]]] = {c: [] for c in cells}
    out_path = f"{args.out_dir}/{args.tag}_large_recon.parquet"
    for i, bbox in enumerate(bboxes):
        window, path = read_canonical_window(
            bbox, cfg.year, window_px=cfg.window_px, max_nan_fraction=cfg.max_nan_fraction
        )
        if window is None:
            logger.warning("bbox %d/%d %s unavailable; skipping", i + 1, len(bboxes), bbox.name)
            continue
        _accumulate(acc, cells, window, cfg)
        del window
        write_parquet_with_provenance(
            pd.DataFrame(_rows_from_acc(acc)), out_path, seed=args.seed, config_path=args.config
        )
        logger.info(
            "bbox %d/%d %s done (%s); checkpoint written", i + 1, len(bboxes), bbox.name, path
        )
    logger.info("sweep complete -> %s", out_path)


if __name__ == "__main__":
    main()
