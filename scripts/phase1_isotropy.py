"""Phase 2 isotropy diagnostics (script name keeps ``phase1`` per docs/spec.md).

Samples land embeddings (100 points/window x N windows), standardises per dimension,
projects onto random unit directions, and tests each projection for normality with
Shapiro-Wilk and Epps-Pulley. The rejection fractions inform the cosine-vs-L2 choice.
Raw embeddings are never persisted; only per-dimension stats and per-direction results.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from tessera_vq.data import UK_BBOX, sample_isotropy_uk
from tessera_vq.io_utils import load_config, write_parquet_with_provenance
from tessera_vq.metrics import epps_pulley, shapiro_wilk

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the isotropy diagnostics phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--n-regions", type=int, default=60, help="UK land regions to read.")
    parser.add_argument("--region-px", type=int, default=384, help="Region size (pixels).")
    parser.add_argument("--points-per-region", type=int, default=1000, help="Points per region.")
    parser.add_argument("--n-jobs", type=int, default=12, help="Parallel read workers.")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON0", "LAT0", "LON1", "LAT1"),
        help="Sampling bbox (lon/lat); default UK.",
    )
    return parser


def random_unit_directions(n: int, dim: int, seed: int) -> npt.NDArray[np.float64]:
    """``(n, dim)`` array of independent uniformly-random unit vectors."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, dim))
    unit: npt.NDArray[np.float64] = g / np.linalg.norm(g, axis=1, keepdims=True)
    return unit


def ep_null_threshold(n: int, alpha: float, reps: int, seed: int) -> float:
    """Monte-Carlo Epps-Pulley critical value at level ``alpha`` for sample size ``n``."""
    rng = np.random.default_rng(seed)
    stats = [epps_pulley(rng.standard_normal(n)) for _ in range(reps)]
    return float(np.quantile(stats, 1.0 - alpha))


def per_direction_normality(
    xsub: npt.NDArray[np.float64], directions: npt.NDArray[np.float64], ep_thr: float, alpha: float
) -> pd.DataFrame:
    """Run Shapiro-Wilk and Epps-Pulley on each 1-D projection of ``xsub``."""
    rows: list[dict[str, float | bool | int]] = []
    for j, direction in enumerate(directions):
        proj = xsub @ direction
        sw_stat, sw_p = shapiro_wilk(proj)
        ep = epps_pulley(proj)
        rows.append(
            {
                "direction": j,
                "sw_stat": sw_stat,
                "sw_p": sw_p,
                "ep_stat": ep,
                "sw_reject": bool(sw_p < alpha),
                "ep_reject": bool(ep > ep_thr),
            }
        )
    return pd.DataFrame(rows)


def _per_dim_stats(x: npt.NDArray[np.float32], ratio: float) -> pd.DataFrame:
    """Per-dimension mean/variance with a near-collapsed flag (var < ratio x median)."""
    mean = x.mean(axis=0)
    var = x.var(axis=0)
    threshold = ratio * float(np.median(var))
    return pd.DataFrame(
        {
            "dim": np.arange(x.shape[1]),
            "mean": mean,
            "var": var,
            "near_collapsed": var < threshold,
        }
    )


def main() -> None:
    """Run the isotropy diagnostic end to end."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    p1, seeds = cfg["phase1"], cfg["seeds"]
    year = cfg["tessera"]["year"]

    logger.info(
        "sampling UK isotropy: %d regions, %dpx, %d pts/region",
        args.n_regions,
        args.region_px,
        args.points_per_region,
    )
    bbox = (args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3]) if args.bbox else UK_BBOX
    x, n_ok = sample_isotropy_uk(
        n_regions=args.n_regions,
        points_per_region=args.points_per_region,
        region_px=args.region_px,
        bbox=bbox,
        year=year,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    logger.info("collected %d points from %d/%d UK regions", x.shape[0], n_ok, args.n_regions)

    dim_stats = _per_dim_stats(x, p1["near_collapsed_variance_ratio"])
    write_parquet_with_provenance(
        dim_stats[["dim", "mean", "var"]],
        "results/pool_a_stats.parquet",
        seed=args.seed,
        config_path=args.config,
    )
    write_parquet_with_provenance(
        dim_stats, "results/phase1/per_dim_stats.parquet", seed=args.seed, config_path=args.config
    )

    std = x.std(axis=0)
    x_std = (x - x.mean(axis=0)) / np.where(std > 0, std, 1.0)
    m = min(int(p1["normality_subsample"]), x_std.shape[0])
    idx = np.random.default_rng(args.seed).choice(x_std.shape[0], size=m, replace=False)
    x_sub = x_std[idx].astype(np.float64)

    directions = random_unit_directions(p1["n_directions"], x.shape[1], seeds["random_projections"])
    ep_thr = ep_null_threshold(m, p1["alpha"], p1["ep_mc_reps"], seeds["random_projections"])
    df = per_direction_normality(x_sub, directions, ep_thr, p1["alpha"])
    write_parquet_with_provenance(
        df, "results/phase1/projection_normality.parquet", seed=args.seed, config_path=args.config
    )

    n_collapsed = int(dim_stats["near_collapsed"].sum())
    sw_frac = float(df["sw_reject"].mean())
    ep_frac = float(df["ep_reject"].mean())
    logger.info("=== ISOTROPY SUMMARY (alpha=%.3g, subsample=%d) ===", p1["alpha"], m)
    logger.info("Shapiro-Wilk rejection fraction: %.3f", sw_frac)
    logger.info("Epps-Pulley rejection fraction:  %.3f (MC threshold %.3f)", ep_frac, ep_thr)
    logger.info("near-collapsed dims: %d / %d", n_collapsed, x.shape[1])


if __name__ == "__main__":
    main()
