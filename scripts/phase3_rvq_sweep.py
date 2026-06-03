"""Phase 3 RVQ sweep entry point.

Streams the first ``--n-bboxes`` canonical 10 km bounding boxes (defined in
``config/canonical_bboxes.yaml``), runs a fully-crossed ``(tile_size, k1, k2)``
RVQ sweep on each (via ``tessera_vq.sweep.rvq_quantize_window_for_serving``),
computes per-pixel L2 and cosine reconstruction errors, histograms each
``(bbox, t, k1, k2)`` result against frozen bin edges (auto-picked from a
warm-up bbox), then aggregates across bboxes into mean +- sd density per bin.

Outputs (in ``--out-dir``, default ``results/phase3/``):

- ``{tag}.csv`` (long, source of truth): t, k1, k2, metric, bin_index, bin_low,
  bin_high, mean_density, sd_density, overflow_frac_mean, overflow_frac_sd,
  n_bboxes + provenance columns (git_sha, seed, timestamp_utc, config_hash).
- ``{tag}_l2.csv`` (wide derived): one row per (t, k1, k2), columns per L2 bin.
- ``{tag}_cos.csv`` (wide derived): same shape for cosine.

Run::

    uv run python scripts/phase3_rvq_sweep.py --n-bboxes 5 --tag phase3_pilot

HALT after the pilot: report bbox path usage (zarr vs bbox-fallback), wall-time,
and a preview of the wide CSVs before extending to the full 100 bboxes.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from tessera_vq.canonical import (
    CanonicalBbox,
    PathChoice,
    load_canonical_bboxes,
    read_canonical_window,
)
from tessera_vq.phase3_sweep import (
    N_BINS,
    aggregate_long,
    hist_density,
    pick_bin_edges,
    rvq_errors,
    to_wide,
)

logger = logging.getLogger(__name__)

DEFAULT_TILE_SIZES: tuple[int, ...] = (16, 32)
DEFAULT_K_VALUES: tuple[int, ...] = (64, 128, 256)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags. ``--n-bboxes 5`` for the pilot; ``100`` for full run."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--config", default="config/canonical_bboxes.yaml")
    p.add_argument("--n-bboxes", type=int, default=5, help="number of bboxes (pilot=5)")
    p.add_argument("--tag", default="phase3_pilot", help="output filename tag")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/phase3")
    p.add_argument(
        "--tile-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_TILE_SIZES),
        help="tile sizes in pixels (default: 16 32 -- the sweet-spot range from the pilot)",
    )
    p.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="codebook sizes for both RVQ stages (default: 64 128 256)",
    )
    return p.parse_args()


def _read_windows(
    bboxes: list[CanonicalBbox], year: int
) -> list[tuple[CanonicalBbox, npt.NDArray[np.float32], PathChoice]]:
    """Read each canonical bbox once; skip ones with no Tessera data."""
    out: list[tuple[CanonicalBbox, npt.NDArray[np.float32], PathChoice]] = []
    for b in bboxes:
        t0 = time.monotonic()
        mosaic, path = read_canonical_window(b, year)
        dt = time.monotonic() - t0
        if mosaic is None or path == "unavailable":
            logger.warning("skip %s (%.1fs): no Tessera data available", b.name, dt)
            continue
        logger.info("read %-32s via %-4s -> shape=%s (%.1fs)", b.name, path, mosaic.shape, dt)
        out.append((b, mosaic, path))
    return out


def _run_sweep(
    windows: list[tuple[CanonicalBbox, npt.NDArray[np.float32], PathChoice]],
    edges_l2: npt.NDArray[np.float64],
    edges_cos: npt.NDArray[np.float64],
    seed: int,
    tile_sizes: list[int],
    k_values: list[int],
) -> list[dict[str, object]]:
    """Run the (t, k1, k2) sweep across all windows; return long-format rows."""
    cells = [(t, k1, k2) for t in tile_sizes for k1 in k_values for k2 in k_values]
    logger.info(
        "sweeping %d (t,k1,k2) cells x %d bboxes = %d RVQ runs",
        len(cells),
        len(windows),
        len(cells) * len(windows),
    )
    rows: list[dict[str, object]] = []
    for cell_i, (t, k1, k2) in enumerate(cells, start=1):
        densities_l2: list[npt.NDArray[np.float64]] = []
        densities_cos: list[npt.NDArray[np.float64]] = []
        overflows_l2: list[float] = []
        overflows_cos: list[float] = []
        for _b, window, _path in windows:
            l2, cos = rvq_errors(window, t, k1, k2, seed)
            d_l2, of_l2 = hist_density(l2, edges_l2)
            d_cos, of_cos = hist_density(cos, edges_cos)
            densities_l2.append(d_l2)
            densities_cos.append(d_cos)
            overflows_l2.append(of_l2)
            overflows_cos.append(of_cos)
        rows.extend(aggregate_long("l2", edges_l2, densities_l2, overflows_l2, t, k1, k2))
        rows.extend(aggregate_long("cos", edges_cos, densities_cos, overflows_cos, t, k1, k2))
        if cell_i % 10 == 0 or cell_i == len(cells):
            logger.info("  cell %d/%d done (t=%d, k1=%d, k2=%d)", cell_i, len(cells), t, k1, k2)
    return rows


def _provenance(config_path: Path, seed: int) -> dict[str, str | int]:
    """Standard provenance columns per CLAUDE.md (git_sha, seed, ts, config_hash)."""
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    cfg_text = config_path.read_text()
    return {
        "git_sha": sha,
        "seed": int(seed),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_hash": hashlib.sha256(cfg_text.encode()).hexdigest(),
    }


def main() -> None:
    """Load -> read windows -> warm-up bin edges -> sweep -> write CSVs."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config)
    bboxes = load_canonical_bboxes(config_path)[: args.n_bboxes]
    tile_sizes: list[int] = list(args.tile_sizes)
    k_values: list[int] = list(args.k_values)
    logger.info(
        "loaded %d canonical bboxes; grid: t=%s k=%s -> %d cells",
        len(bboxes),
        tile_sizes,
        k_values,
        len(tile_sizes) * len(k_values) ** 2,
    )

    windows = _read_windows(bboxes, args.year)
    if not windows:
        logger.error("no readable bboxes; aborting")
        sys.exit(1)

    # Warm-up: pick bin edges from the first window at the smallest (t, k1, k2).
    warmup_t = tile_sizes[0]
    warmup_k = k_values[0]
    warmup_l2, warmup_cos = rvq_errors(windows[0][1], warmup_t, warmup_k, warmup_k, args.seed)
    edges_l2, edges_cos = pick_bin_edges(warmup_l2, warmup_cos)
    logger.info(
        "warm-up (t=%d, k1=k2=%d) bin edges: L2=[%.3f..%.3f] (%d bins), "
        "cos=[%.4f..%.4f] (%d bins)",
        warmup_t,
        warmup_k,
        edges_l2[0],
        edges_l2[-1],
        N_BINS,
        edges_cos[0],
        edges_cos[-1],
        N_BINS,
    )

    sweep_t0 = time.monotonic()
    rows = _run_sweep(windows, edges_l2, edges_cos, args.seed, tile_sizes, k_values)
    sweep_dt = time.monotonic() - sweep_t0
    n_cells = len(tile_sizes) * len(k_values) ** 2
    logger.info(
        "sweep took %.1f s (%.2f s per (t,k1,k2) cell)",
        sweep_dt,
        sweep_dt / n_cells,
    )

    df = pd.DataFrame(rows)
    prov = _provenance(config_path, args.seed)
    for col, val in prov.items():
        df[col] = val

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / f"{args.tag}.csv"
    df.to_csv(long_path, index=False)
    logger.info("wrote long CSV: %s (%d rows)", long_path, len(df))

    for metric in ("l2", "cos"):
        wide = to_wide(df, metric)
        wp = out_dir / f"{args.tag}_{metric}.csv"
        wide.to_csv(wp, index=False)
        logger.info("wrote wide %s CSV: %s (%d rows)", metric, wp, len(wide))

    if len(windows) < 100:  # noqa: PLR2004
        proj_min = sweep_dt / len(windows) * 100 / 60
        logger.info(
            "extrapolation to 100 bboxes: ~%.1f min (pilot: %.1f min)", proj_min, sweep_dt / 60
        )


if __name__ == "__main__":
    main()
