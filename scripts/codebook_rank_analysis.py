"""Codebook effective-rank / SVD + reconstruction-tail analysis (supervisor-requested).

Streams ``--n-bboxes`` canonical bboxes sampled **at random** (seeded), runs the
``(t, k1, k2)`` RVQ on each window, and measures two things per ``(t, k1, k2)``
cell, one window at a time:

1. **Codebook rank** -- whether ``C (k, 128)`` could be stored as ``U (k, r) @ V
   (r, 128)`` with ``r << 128``. A global Gram ``C^T C`` (the subspace one shared
   basis would have to span, raw + mean-centred) and the per-tile SVD spectrum
   (the per-tile rank the compression premise relies on). See
   ``tessera_vq.effrank``.
2. **Reconstruction tail** -- the distribution of *per-tile* reconstruction error
   (L2 and cosine). This tests the "bad tile -> serve raw" idea: if a few tiles
   own the error tail (thin tail, small ``frac_gt_*x_median``), routing them to
   raw passthrough is cheap; a fat tail means it is not.

Memory: one window resident at a time; only 128x128 Grams and per-tile scalars
accumulate. This is *not* a Phase 3 sweep output -- it is an exploratory study of
codebook compressibility and writes to its own ``--out-dir``.

Outputs (Parquet, with provenance columns):

- ``{tag}_global_effrank.parquet`` -- per (t, k1, k2, stage, mode): participation
  ratio, entropy_eff_dim, dims_90/95/99, n_vectors, ambient_dim.
- ``{tag}_global_spectrum.parquet`` -- scree data: one row per component.
- ``{tag}_per_tile_effrank.parquet`` -- per (t, k1, k2, stage): pr_mean/median/
  p10/p90, dims95_median, n_tiles.
- ``{tag}_per_tile_recon.parquet`` -- per (t, k1, k2, metric): error mean,
  p50/90/95/99/99.9, max, frac_gt_2x_median, frac_gt_5x_median, n_tiles.

Run::

    uv run python scripts/codebook_rank_analysis.py --n-bboxes 5 --tag codebook_rank
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from tessera_vq.canonical import (
    CanonicalBbox,
    load_canonical_bboxes,
    read_canonical_window,
)
from tessera_vq.effrank import (
    GramAccumulator,
    effrank_metrics,
    energy_eigvals,
    per_tile_effrank_batch,
    per_tile_summary,
    recon_tail_summary,
    spectrum_rows,
)
from tessera_vq.io_utils import write_parquet_with_provenance
from tessera_vq.sweep import rvq_per_tile_errors, rvq_quantize_window_for_serving

logger = logging.getLogger(__name__)

DEFAULT_TILE_SIZES: tuple[int, ...] = (32, 64)
DEFAULT_K_VALUES: tuple[int, ...] = (256,)
STAGES: tuple[str, ...] = ("c1", "c2")
METRICS: tuple[str, ...] = ("l2", "cos")
AMBIENT_DIM = 128

Cell = tuple[int, int, int]
StageKey = tuple[int, int, int, str]  # (t, k1, k2, stage)
MetricKey = tuple[int, int, int, str]  # (t, k1, k2, metric)


@dataclass
class _Accum:
    """All streaming state: global Grams, per-tile rank buffers, per-tile error buffers."""

    grams: dict[StageKey, GramAccumulator]
    pr_buf: dict[StageKey, list[npt.NDArray[np.float64]]]
    d95_buf: dict[StageKey, list[npt.NDArray[np.int64]]]
    err: dict[MetricKey, list[npt.NDArray[np.float32]]]

    @classmethod
    def for_cells(cls, cells: list[Cell]) -> _Accum:
        """One Gram + rank buffer per (cell, stage); one error buffer per (cell, metric)."""
        grams = {(*c, s): GramAccumulator(AMBIENT_DIM) for c in cells for s in STAGES}
        err: dict[MetricKey, list[npt.NDArray[np.float32]]] = {
            (*c, m): [] for c in cells for m in METRICS
        }
        return cls(
            grams=grams,
            pr_buf={k: [] for k in grams},
            d95_buf={k: [] for k in grams},
            err=err,
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI flags. Defaults probe the large-k / large-tile corner where codebooks dominate."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--config", default="config/canonical_bboxes.yaml")
    p.add_argument("--n-bboxes", type=int, default=5, help="number of bboxes sampled at random")
    p.add_argument("--tag", default="codebook_rank", help="output filename tag")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/codebook_rank")
    p.add_argument("--tile-sizes", type=int, nargs="+", default=list(DEFAULT_TILE_SIZES))
    p.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
        help="codebook sizes for both RVQ stages (default: 256 -- where codebooks dominate)",
    )
    return p.parse_args()


def _sample_bboxes(all_bboxes: list[CanonicalBbox], n: int, seed: int) -> list[CanonicalBbox]:
    """Pick ``n`` bboxes uniformly at random (no replacement), seeded; original order kept."""
    if n >= len(all_bboxes):
        return all_bboxes
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(all_bboxes), size=n, replace=False))
    return [all_bboxes[int(i)] for i in idx]


def _build_cells(tile_sizes: list[int], k_values: list[int]) -> list[Cell]:
    """Fully-crossed ``(t, k1, k2)`` grid in deterministic order."""
    return [(t, k1, k2) for t in tile_sizes for k1 in k_values for k2 in k_values]


def _fold_codebooks(acc: _Accum, key: StageKey, codebooks: npt.NDArray[np.float32]) -> None:
    """Fold one stage's ``(n_tiles, k_eff, 128)`` codebooks into global + per-tile rank state."""
    if codebooks.shape[0] == 0:
        return
    acc.grams[key].update(codebooks.reshape(-1, AMBIENT_DIM))
    pr, d95 = per_tile_effrank_batch(codebooks)
    acc.pr_buf[key].append(pr)
    acc.d95_buf[key].append(d95)


def _process_window(
    acc: _Accum, window: npt.NDArray[np.float32], cells: list[Cell], seed: int
) -> None:
    """Run RVQ for every cell on one window; fold codebook rank + per-tile recon error."""
    for t, k1, k2 in cells:
        cb1, idx1, cb2, idx2, pos = rvq_quantize_window_for_serving(
            window, t, k1, k2, "euclidean", seed
        )
        _fold_codebooks(acc, (t, k1, k2, "c1"), cb1)
        _fold_codebooks(acc, (t, k1, k2, "c2"), cb2)
        if pos.shape[0] == 0:
            continue
        l2, cos = rvq_per_tile_errors(window, t, cb1, idx1, cb2, idx2, pos)
        acc.err[(t, k1, k2, "l2")].append(l2)
        acc.err[(t, k1, k2, "cos")].append(cos)


def _global_rows(
    grams: dict[StageKey, GramAccumulator],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Build the global eff-rank summary rows and the scree-spectrum rows (raw + centred)."""
    summary: list[dict[str, object]] = []
    spectrum: list[dict[str, object]] = []
    for (t, k1, k2, stage), acc in grams.items():
        for mode, centered in (("raw", False), ("centered", True)):
            eig = energy_eigvals(acc, centered=centered)
            base = {"t": t, "k1": k1, "k2": k2, "stage": stage, "mode": mode}
            summary.append(
                {**base, "n_vectors": acc.count, "ambient_dim": AMBIENT_DIM, **effrank_metrics(eig)}
            )
            spectrum.extend({**base, **row} for row in spectrum_rows(eig))
    return summary, spectrum


def _per_tile_rank_rows(acc: _Accum) -> list[dict[str, object]]:
    """Aggregate per-tile participation ratios into a row per (t, k1, k2, stage)."""
    rows: list[dict[str, object]] = []
    for key, pr_list in acc.pr_buf.items():
        t, k1, k2, stage = key
        prs = np.concatenate(pr_list) if pr_list else np.zeros(0)
        d95 = np.concatenate(acc.d95_buf[key]) if acc.d95_buf[key] else np.zeros(0, np.int64)
        rows.append({"t": t, "k1": k1, "k2": k2, "stage": stage, **per_tile_summary(prs, d95)})
    return rows


def _per_tile_recon_rows(acc: _Accum) -> list[dict[str, object]]:
    """Aggregate per-tile reconstruction errors into a row per (t, k1, k2, metric)."""
    rows: list[dict[str, object]] = []
    for (t, k1, k2, metric), chunks in acc.err.items():
        errs = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        rows.append({"t": t, "k1": k1, "k2": k2, "metric": metric, **recon_tail_summary(errs)})
    return rows


def _log_headline(
    summary: list[dict[str, object]],
    per_tile: list[dict[str, object]],
    recon: list[dict[str, object]],
) -> None:
    """Print the numbers the supervisor reads: centred global rank + L2 bad-tile tail."""
    pt = {(r["t"], r["k1"], r["k2"], r["stage"]): r for r in per_tile}
    rc = {(r["t"], r["k1"], r["k2"], r["metric"]): r for r in recon}
    logger.info("=== CODEBOOK RANK + RECON TAIL (ambient dim = %d) ===", AMBIENT_DIM)
    for r in summary:
        if r["mode"] != "centered" or r["stage"] != "c1":
            continue
        key = (r["t"], r["k1"], r["k2"])
        e = rc[(*key, "l2")]
        logger.info(
            "t=%-2d k1=%-3d k2=%-3d | c1 global(cen) PR=%.1f dims95=%d | per-tile median PR=%.1f "
            "| L2 p50=%.2f p99=%.2f frac>5x med=%.1f%%",
            r["t"],
            r["k1"],
            r["k2"],
            r["participation_ratio"],
            r["dims_95"],
            pt[(*key, "c1")]["pr_median"],
            e["p50"],
            e["p99"],
            100 * cast("float", e["frac_gt_5x_median"]),
        )


def _stream(
    acc: _Accum, bboxes: list[CanonicalBbox], cells: list[Cell], year: int, seed: int
) -> int:
    """Read -> RVQ -> fold -> free, one bbox at a time. Returns bboxes actually read."""
    n_read = 0
    n_total = len(bboxes)
    for i, b in enumerate(bboxes, start=1):
        t0 = time.monotonic()
        window, path = read_canonical_window(b, year)
        if window is None or path == "unavailable":
            logger.warning("[%d/%d] skip %-28s: no Tessera data", i, n_total, b.name)
            continue
        n_read += 1
        _process_window(acc, window, cells, seed)
        del window  # free the ~1.5 GB window before the next read
        logger.info("[%d/%d] folded %-28s (%.1fs)", i, n_total, b.name, time.monotonic() - t0)
    return n_read


def main() -> None:
    """Sample bboxes -> stream codebook rank + recon-tail analysis -> write four Parquet tables."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(args.seed)
    random.seed(args.seed)

    bboxes = _sample_bboxes(load_canonical_bboxes(args.config), args.n_bboxes, args.seed)
    cells = _build_cells(list(args.tile_sizes), list(args.k_values))
    logger.info(
        "sampled %d bboxes at random (seed=%d); %d cells x %d stages",
        len(bboxes),
        args.seed,
        len(cells),
        len(STAGES),
    )
    acc = _Accum.for_cells(cells)
    n_read = _stream(acc, bboxes, cells, args.year, args.seed)
    if n_read == 0:
        logger.error("no readable bboxes; no output written")
        raise SystemExit(1)

    summary, spectrum = _global_rows(acc.grams)
    per_tile = _per_tile_rank_rows(acc)
    recon = _per_tile_recon_rows(acc)
    out = Path(args.out_dir)
    for df, name in (
        (pd.DataFrame(summary), "global_effrank"),
        (pd.DataFrame(spectrum), "global_spectrum"),
        (pd.DataFrame(per_tile), "per_tile_effrank"),
        (pd.DataFrame(recon), "per_tile_recon"),
    ):
        write_parquet_with_provenance(
            df, out / f"{args.tag}_{name}.parquet", seed=args.seed, config_path=args.config
        )
    _log_headline(summary, per_tile, recon)
    logger.info(
        "DONE: %d/%d bboxes folded; tables in %s (tag=%s)", n_read, len(bboxes), out, args.tag
    )


if __name__ == "__main__":
    main()
