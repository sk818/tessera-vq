"""Diagnostic: does gzip/zstd beat byte-aligned RLE on the stage-1 index plane?

Regenerates real idx1 maps from cached GeoTessera windows (offline), then encodes
the *same* row-major idx1 byte plane with several codecs and reports bytes/px:

- ``rle_row`` / ``rle_hilbert``: the current method (byte-aligned RLE, 1 sym byte
  per run + LEB128 run lengths), row-major and Hilbert order.
- ``gzip_row`` / ``gzip_hilbert``: raw DEFLATE (zlib level 9) on the uint8 plane.
- ``zstd_row``: zstandard level 19 on the uint8 plane.
- ``rle_then_gzip``: gzip applied to the RLE symbol+length byte stream.

All divided by n_px. Aggregated as the mean across tiles. Prints a table and, with
``--out-parquet``, writes a provenance-tagged per-cell table (gzip-based
``total_compressed_Bpx`` / ``x_int8_compressed``) for ``scripts/phase4_pareto.py``.

Run::

    uv run python scripts/phase3_gzip_vs_rle.py --n-bboxes 2 --tile-sizes 512 \
        --tiles-per-bbox 2 --k1 20 32 64 128
"""

from __future__ import annotations

import argparse
import logging
import zlib

import numpy as np
import numpy.typing as npt
import pandas as pd
import zstandard as zstd

from tessera_vq.canonical import load_canonical_bboxes, read_canonical_window
from tessera_vq.entropy import rle_encode
from tessera_vq.index_codec import ORDERINGS, compress_index_map
from tessera_vq.io_utils import write_parquet_with_provenance
from tessera_vq.rvq_large import rvq_reconstruct_large
from tessera_vq.tiling import extract_finite_tiles

logger = logging.getLogger(__name__)


def _codec_bytes_per_px(idx1: npt.NDArray[np.integer], k1: int) -> dict[str, float]:
    """bytes/px for idx1 under RLE / gzip / zstd, row-major and (where apt) Hilbert."""
    h, w = idx1.shape
    n_px = h * w
    out: dict[str, float] = {}
    # --- current method: byte-aligned RLE ---
    out["rle_row"] = compress_index_map(idx1, k1, "row").rle_bytes_per_px
    out["rle_hilbert"] = compress_index_map(idx1, k1, "hilbert").rle_bytes_per_px
    # --- generic byte-stream compressors on the raw uint8 plane ---
    plane_row = idx1.astype(np.uint8).ravel().tobytes()
    plane_hil = idx1.astype(np.uint8).ravel()[ORDERINGS["hilbert"](h, w)].tobytes()
    out["gzip_row"] = len(zlib.compress(plane_row, 9)) / n_px
    out["gzip_hilbert"] = len(zlib.compress(plane_hil, 9)) / n_px
    out["zstd_row"] = len(zstd.ZstdCompressor(level=19).compress(plane_row)) / n_px
    # --- RLE first, then gzip the run stream ---
    vals, lens = rle_encode(idx1.ravel()[ORDERINGS["row"](h, w)])
    rle_stream = vals.astype(np.uint8).tobytes() + lens.astype(np.uint32).tobytes()
    out["rle_then_gzip"] = len(zlib.compress(rle_stream, 9)) / n_px
    return out


def _idx2_bytes_per_px(idx2: npt.NDArray[np.integer]) -> dict[str, float]:
    """Can the white residual plane be compressed below its 1-byte raw cost?"""
    n_px = idx2.size
    plane = idx2.astype(np.uint8).ravel().tobytes()
    return {
        "idx2_raw": 1.0,
        "idx2_gzip": len(zlib.compress(plane, 9)) / n_px,
        "idx2_zstd": len(zstd.ZstdCompressor(level=19).compress(plane)) / n_px,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--config", default="config/canonical_bboxes.yaml")
    p.add_argument("--n-bboxes", type=int, default=2)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-px", type=int, default=1200)
    p.add_argument("--tiles-per-bbox", type=int, default=2)
    p.add_argument("--min-finite-frac", type=float, default=1.0)
    p.add_argument("--max-nan-fraction", type=float, default=0.5)
    p.add_argument("--tile-sizes", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--k1", type=int, nargs="+", default=[20, 32, 64, 128])
    p.add_argument("--k2", type=int, default=256)
    p.add_argument("--max-windows", type=int, default=2, help="stop after this many valid windows")
    p.add_argument("--config-path", default="config/canonical_bboxes.yaml")
    p.add_argument(
        "--out-parquet",
        default=None,
        help="if set, write a provenance-tagged per-cell table (gzip-based "
        "total_compressed_Bpx / x_int8_compressed) for scripts/phase4_pareto.py",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    bboxes = load_canonical_bboxes(args.config)
    rows: list[dict[str, float]] = []
    n_valid = 0
    for bbox in bboxes:
        if n_valid >= args.max_windows:
            break
        window, path = read_canonical_window(
            bbox, args.year, window_px=args.window_px, max_nan_fraction=args.max_nan_fraction
        )
        if window is None:
            continue
        n_valid += 1
        logger.info("window %d (%s) from %s", n_valid, bbox.name, path)
        for t in args.tile_sizes:
            samples = extract_finite_tiles(
                window,
                t,
                n_tiles=args.tiles_per_bbox,
                seed=args.seed,
                min_finite_frac=args.min_finite_frac,
            )
            for s in samples:
                for k1 in args.k1:
                    res = rvq_reconstruct_large(s.tile, k1, args.k2, seed=args.seed)
                    m = _codec_bytes_per_px(res.indices1, k1)
                    m.update(_idx2_bytes_per_px(res.indices2))
                    rows.append({"t": t, "k1": k1, **m})
                logger.info("  t=%d tile done (%d k1 values)", t, len(args.k1))
    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("no valid windows found offline; nothing to report")
        return
    agg = df.groupby(["t", "k1"]).mean(numeric_only=True).reset_index()
    agg["n_tiles"] = df.groupby(["t", "k1"]).size().values
    # derived per-cell totals: codebook (analytic) + gzip(idx1) + raw idx2 floor
    agg["k2"] = args.k2
    agg["codebook_Bpx"] = (agg["k1"] + args.k2) * 128 / (agg["t"] ** 2)
    agg["total_compressed_Bpx"] = agg["codebook_Bpx"] + agg["gzip_row"] + agg["idx2_raw"]
    agg["x_int8_compressed"] = 128.0 / agg["total_compressed_Bpx"]
    agg["x_fp32_compressed"] = 512.0 / agg["total_compressed_Bpx"]
    cols = [
        "t",
        "k1",
        "n_tiles",
        "rle_row",
        "rle_hilbert",
        "gzip_row",
        "gzip_hilbert",
        "zstd_row",
        "rle_then_gzip",
    ]
    pd.set_option("display.width", 200)
    print("\n=== idx1 bytes/px by codec (mean over tiles) ===")
    print(agg[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best = agg[["rle_row", "gzip_row", "gzip_hilbert", "zstd_row", "rle_then_gzip"]].idxmin(axis=1)
    print("\nbest codec per (t,k1):", list(best))
    print("\n=== gzip-based totals (for the Pareto figure) ===")
    print(
        agg[["t", "k1", "total_compressed_Bpx", "x_int8_compressed"]].to_string(
            index=False, float_format=lambda x: f"{x:.3f}"
        )
    )
    if args.out_parquet:
        write_parquet_with_provenance(
            agg, args.out_parquet, seed=args.seed, config_path=args.config_path
        )
        print(f"\nwrote {args.out_parquet}")


if __name__ == "__main__":
    main()
