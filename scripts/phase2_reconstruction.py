"""Phase 3 reconstruction sweep — UK pilot.

Streams ``--n-windows`` UK 1024x1024 land patches (parallel zarr reads), carves up to
``--subtiles-per-window`` random sub-tiles per ``grid.tile_sizes``, runs a k-means VQ
sweep over ``grid.k_values`` (both euclidean and cosine at k=16,64), records per-pixel
cosine/L2 reconstruction quantiles, and saves the per-(window, tile_size, k, distance)
table. Wasserstein-1 is deferred (per-tile cost too high for the pilot). No embeddings
are persisted.
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from tessera_vq.data import UK_BBOX, iter_uk_windows_parallel
from tessera_vq.io_utils import load_config, write_parquet_with_provenance
from tessera_vq.quantize import Distance, quantize_tile, reconstruct_tile

logger = logging.getLogger(__name__)

_DUAL_METRIC_KS = {16, 64}


def build_parser() -> argparse.ArgumentParser:
    """CLI: --n-windows, --subtiles-per-window, --n-jobs, --bbox, --seed, --config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--n-windows", type=int, default=20, help="Number of UK 1024^2 windows.")
    parser.add_argument("--subtiles-per-window", type=int, default=4, help="Random sub-tiles cap.")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel read workers.")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON0", "LAT0", "LON1", "LAT1"),
        help="Sampling bbox (lon/lat); default UK.",
    )
    return parser


def carve_subtiles(
    window: npt.NDArray[np.float32], tile_size: int, n: int, seed: int
) -> list[npt.NDArray[np.float32]]:
    """Up to ``n`` random non-overlapping NaN-free sub-tiles of ``tile_size`` from ``window``."""
    h, w, _ = window.shape
    if tile_size >= h or tile_size >= w:
        side = min(h, w)
        side -= side % tile_size
        if side == 0:
            return []
        block = window[:side, :side]
        return [np.asarray(block, dtype=np.float32)] if np.isfinite(block).all() else []
    rng = np.random.default_rng(seed)
    rows, cols = h // tile_size, w // tile_size
    take = min(n, rows * cols)
    idx = rng.choice(rows * cols, size=take, replace=False)
    out: list[npt.NDArray[np.float32]] = []
    for i in idx:
        r, c = divmod(int(i), cols)
        tile = window[r * tile_size : (r + 1) * tile_size, c * tile_size : (c + 1) * tile_size]
        if np.isfinite(tile).all():
            out.append(np.asarray(tile, dtype=np.float32))
    return out


def reconstruction_errors(
    original: npt.NDArray[np.float32], reconstruction: npt.NDArray[np.float32]
) -> dict[str, float]:
    """Per-pixel cosine distance and L2 distance quantiles (10/50/90/99)."""
    o = original.reshape(-1, original.shape[-1]).astype(np.float64)
    r = reconstruction.reshape(-1, reconstruction.shape[-1]).astype(np.float64)
    on = np.linalg.norm(o, axis=1)
    rn = np.linalg.norm(r, axis=1)
    denom = np.where((on > 0) & (rn > 0), on * rn, 1.0)
    cos_dist = 1.0 - (o * r).sum(axis=1) / denom
    l2 = np.linalg.norm(o - r, axis=1)
    out: dict[str, float] = {}
    for q in (0.1, 0.5, 0.9, 0.99):
        tag = f"p{int(q * 100)}"
        out[f"cos_{tag}"] = float(np.quantile(cos_dist, q))
        out[f"l2_{tag}"] = float(np.quantile(l2, q))
    return out


def sweep_subtile(
    tile: npt.NDArray[np.float32], k_values: list[int], seed: int
) -> list[dict[str, Any]]:
    """Run (k, distance) sweep on one sub-tile; dual metric at k in _DUAL_METRIC_KS."""
    rows: list[dict[str, Any]] = []
    for k in k_values:
        dists: tuple[str, ...] = ("euclidean", "cosine") if k in _DUAL_METRIC_KS else ("euclidean",)
        for dist in dists:
            codebook, idx = quantize_tile(tile, k, distance=cast(Distance, dist), seed=seed)
            errs = reconstruction_errors(tile, reconstruct_tile(codebook, idx))
            rows.append(
                {"k": k, "distance": dist, "n_pixels": int(tile.shape[0] * tile.shape[1]), **errs}
            )
    return rows


def main() -> None:
    """Run the UK reconstruction pilot end to end."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    grid = cfg["grid"]
    year = cfg["tessera"]["year"]
    bbox = (args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3]) if args.bbox else UK_BBOX
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for w_idx, window in enumerate(
        iter_uk_windows_parallel(
            args.n_windows, window_px=1024, bbox=bbox, year=year, seed=args.seed, n_jobs=args.n_jobs
        )
    ):
        for ts in grid["tile_sizes"]:
            for st_idx, subtile in enumerate(
                carve_subtiles(window, ts, args.subtiles_per_window, args.seed + ts)
            ):
                for row in sweep_subtile(subtile, grid["k_values"], args.seed):
                    rows.append({"window": w_idx, "tile_size": ts, "subtile": st_idx, **row})
        logger.info("window %d done (elapsed %.0fs)", w_idx + 1, time.time() - t0)

    df = pd.DataFrame(rows)
    write_parquet_with_provenance(
        df, "results/phase2/reconstruction.parquet", seed=args.seed, config_path=args.config
    )
    summary = df.groupby(["tile_size", "k", "distance"])[["cos_p50", "l2_p50"]].median()
    logger.info("=== RECONSTRUCTION PILOT SUMMARY (cosine/L2 medians) ===\n%s", summary.to_string())


if __name__ == "__main__":
    main()
