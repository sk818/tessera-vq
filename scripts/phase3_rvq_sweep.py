"""Phase 3 RVQ sweep entry point.

Streams ``--n-bboxes`` canonical 10 km bounding boxes sampled **at random**
(seeded by ``--seed``) from ``config/canonical_bboxes.yaml``, **one at a time**,
runs a ``(tile_size, k1, k2)`` RVQ sweep on each (restricted to ``k1 < k2`` -- a
coarse stage-1 base plus a richer stage-2 residual; via
``tessera_vq.sweep.rvq_quantize_window_for_serving``), computes per-pixel L2
reconstruction errors, histograms each ``(bbox, t, k1, k2)`` result against
frozen bin edges (auto-picked from the first readable sampled bbox), then
aggregates across bboxes into mean +- sd density. Only the **first bin**
(``[0, edges[1])`` -- the near-zero-error fraction) is written out.

Memory: exactly one window is resident at a time. A bbox-fallback mosaic is
~1.5 GB; it is freed (``del``) before the next read, and only the tiny per-cell
density histograms (a few KB total) accumulate. The earlier version appended
every window to a list and was OOM-killed ~half-way through the 100-bbox run on
a 48 GB machine -- this version peaks at one window plus the accumulator.

Robustness: the long CSV (and its wide derivative) is rewritten after *every*
bbox as a checkpoint, so an interruption at bbox N still leaves a valid aggregate
over the N-1 bboxes already processed. There is no end-of-run-only write step.

Outputs (in ``--out-dir``, default ``results/phase3/``):

- ``{tag}.csv`` (long, source of truth): t, k1, k2, metric, bin_index, bin_low,
  bin_high, mean_density, sd_density, overflow_frac_mean, overflow_frac_sd,
  n_bboxes + provenance columns (git_sha, seed, timestamp_utc, config_hash).
- ``{tag}_l2.csv`` (wide derived): one row per (t, k1, k2), first-bin L2 columns.

Run::

    uv run python scripts/phase3_rvq_sweep.py --n-bboxes 5 --tag phase3_pilot
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from tessera_vq.canonical import (
    CanonicalBbox,
    load_canonical_bboxes,
    read_canonical_window,
)
from tessera_vq.phase3_sweep import (
    aggregate_long,
    hist_density,
    pick_bin_edges,
    rvq_errors,
    to_wide,
)

logger = logging.getLogger(__name__)

DEFAULT_TILE_SIZES: tuple[int, ...] = (32, 64, 128)
# Stage 1 is a coarse (small-k1) base; stage 2 a richer (large-k2) residual. The
# grid is restricted to k1 < k2 and k2 < t*t (degenerate cells dropped). k1/k2 are sized
# so their indices bit-pack into 16 bits (2 B/px): 64=6b + 1024=10b, 128=7b + 512=9b.
DEFAULT_K1_VALUES: tuple[int, ...] = (64, 128)
DEFAULT_K2_VALUES: tuple[int, ...] = (512, 1024)

Cell = tuple[int, int, int]


@dataclass
class _CellAccum:
    """Per-(t, k1, k2) cell: one density vector + overflow per processed bbox.

    Holds only 50-float histograms, never the windows themselves, so the whole
    accumulator across all cells stays in the low-MB range for 100 bboxes.
    """

    l2_dens: list[npt.NDArray[np.float64]] = field(default_factory=list)
    l2_of: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class _RunCfg:
    """Per-run knobs threaded through the streaming loop (keeps signatures small)."""

    year: int
    seed: int
    prov: dict[str, str | int]
    out_dir: Path
    tag: str


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
        help="tile sizes in pixels (default: %(default)s)",
    )
    p.add_argument(
        "--k1-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K1_VALUES),
        help="stage-1 (coarse base) codebook sizes (default: %(default)s)",
    )
    p.add_argument(
        "--k2-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K2_VALUES),
        help="stage-2 (residual) codebook sizes (default: %(default)s)",
    )
    return p.parse_args()


def _build_cells(tile_sizes: list[int], k1_values: list[int], k2_values: list[int]) -> list[Cell]:
    """``(t, k1, k2)`` grid in deterministic order.

    Restricted to ``k1 < k2`` and to non-degenerate cells: ``k2 < t * t``. A cell
    with ``k2 >= t * t`` would have ``k_eff = t * t`` (one residual code per pixel
    -> near-perfect reconstruction at zero compression), so it is dropped.
    """
    return [
        (t, k1, k2) for t in tile_sizes for k1 in k1_values for k2 in k2_values if k1 < k2 < t * t
    ]


def _accumulate(
    acc: dict[Cell, _CellAccum],
    cells: list[Cell],
    window: npt.NDArray[np.float32],
    edges_l2: npt.NDArray[np.float64],
    seed: int,
) -> None:
    """Sweep every cell on one window; append this bbox's L2 density + overflow to acc."""
    for cell in cells:
        t, k1, k2 = cell
        l2 = rvq_errors(window, t, k1, k2, seed)
        d_l2, o_l2 = hist_density(l2, edges_l2)
        a = acc[cell]
        a.l2_dens.append(d_l2)
        a.l2_of.append(o_l2)


def _accum_to_rows(
    acc: dict[Cell, _CellAccum],
    cells: list[Cell],
    edges_l2: npt.NDArray[np.float64],
) -> list[dict[str, object]]:
    """Aggregate the accumulator into long rows, keeping only the first bin.

    We emit a single row per cell: L2 bin 0, ``[0, edges[1])``. That bin holds the
    fraction of pixels reconstructed to near-zero error, which is the headline
    quality number for the sweep; the remaining bins are dropped.
    """
    rows: list[dict[str, object]] = []
    for cell in cells:
        t, k1, k2 = cell
        a = acc[cell]
        rows.extend(aggregate_long("l2", edges_l2, a.l2_dens, a.l2_of, t, k1, k2))
    return [r for r in rows if r["bin_index"] == 0]


def _write_outputs(
    rows: list[dict[str, object]],
    prov: dict[str, str | int],
    out_dir: Path,
    tag: str,
) -> Path:
    """Write the long CSV (source of truth) + the wide L2 CSV; return long path."""
    df = pd.DataFrame(rows)
    for col, val in prov.items():
        df[col] = val
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / f"{tag}.csv"
    df.to_csv(long_path, index=False)
    to_wide(df, "l2").to_csv(out_dir / f"{tag}_l2.csv", index=False)
    return long_path


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


def _stream_sweep(
    bboxes: list[CanonicalBbox],
    cells: list[Cell],
    warmup_cell: Cell,
    cfg: _RunCfg,
) -> int:
    """Read -> sweep -> free, one bbox at a time; checkpoint the CSVs after each.

    Bin edges are frozen from the first readable bbox at ``warmup_cell`` (a valid
    ``(t, k1, k2)`` with ``k1 < k2``) and reused for every subsequent bbox.
    Returns the number of bboxes actually processed (reads that returned data).
    """
    acc: dict[Cell, _CellAccum] = {c: _CellAccum() for c in cells}
    edges_l2: npt.NDArray[np.float64] | None = None
    n_read = 0
    n_total = len(bboxes)
    wt, wk1, wk2 = warmup_cell
    for i, b in enumerate(bboxes, start=1):
        t0 = time.monotonic()
        mosaic, path = read_canonical_window(b, cfg.year)
        dt = time.monotonic() - t0
        if mosaic is None or path == "unavailable":
            logger.warning("[%d/%d] skip %-28s (%.1fs): no Tessera data", i, n_total, b.name, dt)
            continue
        n_read += 1
        logger.info(
            "[%d/%d] read %-28s via %-4s -> %s (%.1fs)", i, n_total, b.name, path, mosaic.shape, dt
        )
        if edges_l2 is None:
            wl2 = rvq_errors(mosaic, wt, wk1, wk2, cfg.seed)
            edges_l2 = pick_bin_edges(wl2)
            logger.info("  warm-up bin edges: L2=[0..%.3f]", edges_l2[-1])
        st0 = time.monotonic()
        _accumulate(acc, cells, mosaic, edges_l2, cfg.seed)
        del mosaic  # free ~1.5 GB before the next read
        rows = _accum_to_rows(acc, cells, edges_l2)
        long_path = _write_outputs(rows, cfg.prov, cfg.out_dir, cfg.tag)
        logger.info(
            "  swept %d cells (%.1fs); checkpoint %d bbox(es) -> %s",
            len(cells),
            time.monotonic() - st0,
            n_read,
            long_path,
        )
    return n_read


def _sample_bboxes(all_bboxes: list[CanonicalBbox], n: int, seed: int) -> list[CanonicalBbox]:
    """Pick ``n`` bboxes uniformly at random (no replacement), seeded for repeatability.

    Returns them in ascending original-index order so the run log and the frozen
    bin-edge warm-up bbox are deterministic for a given seed. If ``n`` >= the pool
    size, returns all bboxes (still in original order).
    """
    if n >= len(all_bboxes):
        return all_bboxes
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(all_bboxes), size=n, replace=False))
    return [all_bboxes[int(i)] for i in idx]


def main() -> None:
    """Load bboxes -> stream-sweep one at a time (checkpointing each) -> final log."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config)
    bboxes = _sample_bboxes(load_canonical_bboxes(config_path), args.n_bboxes, args.seed)
    tile_sizes: list[int] = list(args.tile_sizes)
    k1_values: list[int] = list(args.k1_values)
    k2_values: list[int] = list(args.k2_values)
    cells = _build_cells(tile_sizes, k1_values, k2_values)
    if not cells:
        logger.error(
            "empty grid: no (t, k1, k2) satisfies k1 < k2 and k2 < t*t (t=%s k1=%s k2=%s)",
            tile_sizes,
            k1_values,
            k2_values,
        )
        sys.exit(1)
    full = len(tile_sizes) * len(k1_values) * len(k2_values)
    logger.info(
        "sampled %d bboxes at random (seed=%d); grid t=%s k1=%s k2=%s -> %d cells "
        "(%d dropped for k1>=k2 or degenerate k2>=t*t); streaming one bbox at a time",
        len(bboxes),
        args.seed,
        tile_sizes,
        k1_values,
        k2_values,
        len(cells),
        full - len(cells),
    )

    cfg = _RunCfg(
        year=args.year,
        seed=args.seed,
        prov=_provenance(config_path, args.seed),
        out_dir=Path(args.out_dir),
        tag=args.tag,
    )
    t0 = time.monotonic()
    n_read = _stream_sweep(bboxes, cells, cells[0], cfg)
    elapsed = time.monotonic() - t0
    if n_read == 0:
        logger.error("no readable bboxes; no output written")
        sys.exit(1)
    logger.info(
        "DONE: %d/%d bboxes processed in %.1f min; outputs in %s (tag=%s)",
        n_read,
        len(bboxes),
        elapsed / 60,
        cfg.out_dir,
        cfg.tag,
    )


if __name__ == "__main__":
    main()
