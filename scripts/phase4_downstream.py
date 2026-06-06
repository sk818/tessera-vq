"""Downstream-task validation: does VQ reconstruction hurt classification? (WS-3).

The determinative test. For each ``(t, k1, k2)`` cell we compress every GeoTessera
tile overlapping a labelled shapefile through per-``t``-block RVQ, then train the same
Random Forest on the **raw** vs the **reconstructed** embeddings at the labelled pixels
and compare macro-F1 under **spatial group k-fold** (each fold holds out whole tiles;
random k-fold would leak spatially-autocorrelated neighbours and flatter VQ). Averaging
over folds is essential when there are few tiles (Cumbria has 4).

Per cell it records ``f1_raw_mean/sd``, ``f1_recon_mean/sd`` and the paired
``delta_f1_mean/sd`` across folds. Tiles are re-fetched per cell to keep memory to one
cell's pixels; GeoTessera disk-caches downloads, so the repeat cost is decode-only.

Requires the cross-repo deps (personal repos): ``geotessera`` and ``tessera_eval``
(``uv pip install -e ../blore/packages/tessera-eval`` + geotessera), plus geopandas /
rasterio. Run once per dataset, e.g. Austria (17 crops) and Cumbria/Naddle (habitats):

    uv run python scripts/phase4_downstream.py \
        --shapefile ../blore/austria.zip --field <COLUMN> --tag austria

Omit ``--field`` to print the shapefile's attribute columns and exit.

Output (``--out-dir``, default ``results/phase4/``): ``{tag}_downstream.parquet``.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from tessera_vq.downstream import (
    extract_labelled,
    reconstruct_tile_blocks,
    spatial_group_kfold,
)
from tessera_vq.io_utils import write_parquet_with_provenance

logger = logging.getLogger(__name__)

DEFAULT_TILE_SIZES: tuple[int, ...] = (512, 1024)
DEFAULT_CONFIGS: tuple[str, ...] = ("20:256", "32:256", "64:256", "128:256")

Cell = tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    """Parse CLI flags. Omit ``--field`` to list the shapefile's columns and exit."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    p.add_argument("--shapefile", required=True, help="path to .shp or zipped shapefile")
    p.add_argument("--field", default=None, help="label column (omit to list columns)")
    p.add_argument("--tag", default="downstream")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/phase4")
    p.add_argument("--spatial-folds", type=int, default=4, help="group k-folds over tiles")
    p.add_argument("--max-train", type=int, default=50000, help="cap on training pixels/fold")
    p.add_argument(
        "--int8-codebooks",
        action="store_true",
        help="reconstruct from int8-served codebooks (WS-1 validation)",
    )
    p.add_argument("--tile-sizes", type=int, nargs="+", default=list(DEFAULT_TILE_SIZES))
    p.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    return p.parse_args()


def _build_cells(tile_sizes: list[int], configs: list[str]) -> list[Cell]:
    """``(t, k1, k2)`` grid: tile sizes x specific (k1, k2) pairs, dropping k >= t*t."""
    pairs = [(int(a), int(b)) for a, b in (c.split(":") for c in configs)]
    return [(t, k1, k2) for t in tile_sizes for k1, k2 in pairs if k1 < t * t and k2 < t * t]


def _load_gdf(path: str) -> Any:
    """Read a shapefile (or zipped shapefile, incl. nested) and reproject to EPSG:4326."""
    import geopandas as gpd  # noqa: PLC0415  (heavy cross-repo dep; deferred)

    uri = path
    if path.endswith(".zip"):
        import zipfile  # noqa: PLC0415

        with zipfile.ZipFile(path) as zf:
            shps = [n for n in zf.namelist() if n.lower().endswith(".shp")]
        # nested .shp (e.g. austria.zip) needs an explicit inner path after '!'
        uri = f"zip://{path}!{shps[0]}" if shps else f"zip://{path}"
    gdf = gpd.read_file(uri)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:  # noqa: PLR2004
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def _collect_cell(
    gt: Any, gdf: Any, field: str, le: Any, cell: Cell, year: int, seed: int, int8: bool
) -> dict[str, Any]:
    """Re-fetch tiles, RVQ each with this cell, return raw/recon/labels/groups arrays."""
    from rasterio.transform import array_bounds  # noqa: PLC0415
    from shapely.geometry import box  # noqa: PLC0415
    from tessera_eval.rasterize import rasterize_shapefile  # noqa: PLC0415

    t, k1, k2 = cell
    raws, recons, labels, groups = [], [], [], []
    tiles = gt.registry.load_blocks_for_region(tuple(gdf.total_bounds), year)
    for gi, (_yr, _lon, _lat, emb, crs, transform) in enumerate(gt.fetch_embeddings(tiles)):
        h, w, _ = emb.shape
        tile_gdf = gdf.to_crs(crs) if gdf.crs != crs else gdf
        tile_gdf = tile_gdf[tile_gdf.intersects(box(*array_bounds(h, w, transform)))]
        if tile_gdf.empty:
            continue
        class_raster = rasterize_shapefile(tile_gdf, field, transform, w, h, label_encoder=le)
        if int((class_raster > 0).sum()) == 0:
            continue
        recon = reconstruct_tile_blocks(
            np.asarray(emb, np.float32), t, k1, k2, seed=seed, quantize_codebooks=int8
        )
        raw_v, rec_v, lab = extract_labelled(np.asarray(emb, np.float32), recon, class_raster)
        if raw_v.shape[0] == 0:
            continue
        raws.append(raw_v)
        recons.append(rec_v)
        labels.append(lab)
        groups.append(np.full(lab.shape[0], gi, dtype=np.int64))
    return {
        "raw": np.concatenate(raws) if raws else np.zeros((0, 128), np.float32),
        "recon": np.concatenate(recons) if recons else np.zeros((0, 128), np.float32),
        "labels": np.concatenate(labels) if labels else np.zeros(0, np.int64),
        "groups": np.concatenate(groups) if groups else np.zeros(0, np.int64),
    }


def _subsample(mask: npt.NDArray[np.bool_], cap: int, seed: int) -> npt.NDArray[np.bool_]:
    """Thin a boolean training mask down to at most ``cap`` True entries (seeded)."""
    idx = np.flatnonzero(mask)
    if idx.size <= cap:
        return mask
    keep = np.random.default_rng(seed).choice(idx, size=cap, replace=False)
    out = np.zeros_like(mask)
    out[keep] = True
    return out


def _f1(
    vecs: npt.NDArray[np.float32],
    labels: npt.NDArray[np.int64],
    train: npt.NDArray[np.bool_],
    test: npt.NDArray[np.bool_],
    seed: int,
) -> float:
    """Macro-F1 of a Random Forest trained on ``train`` pixels, scored on ``test``."""
    from tessera_eval.evaluate import run_learning_curve  # noqa: PLC0415

    events = list(
        run_learning_curve(
            vecs[train],
            labels[train],
            classifier_names=["rf"],
            training_pcts=[100],
            repeats=3,
            test_vectors=vecs[test],
            test_labels=labels[test],
        )
    )
    progs = [e for e in events if e.get("type") == "progress"]
    return float(progs[-1]["classifiers"]["rf"]["mean_f1"]) if progs else float("nan")


def _cell_result(
    data: dict[str, Any], cell: Cell, n_folds: int, max_train: int, seed: int
) -> dict[str, Any]:
    """Spatial group k-fold; return raw/recon F1 mean+-sd and the paired delta."""
    t, k1, k2 = cell
    labels, groups = data["labels"], data["groups"]
    row: dict[str, Any] = {
        "t": t,
        "k1": k1,
        "k2": k2,
        "n_pixels": int(labels.size),
        "n_classes": int(np.unique(labels).size) if labels.size else 0,
    }
    folds = spatial_group_kfold(groups, n_folds=n_folds, seed=seed)
    if not folds:
        return row
    raw_f1, recon_f1 = [], []
    for train, test in folds:
        tr = _subsample(train, max_train, seed)
        raw_f1.append(_f1(data["raw"], labels, tr, test, seed))
        recon_f1.append(_f1(data["recon"], labels, tr, test, seed))
    raw_a, recon_a = np.asarray(raw_f1), np.asarray(recon_f1)
    row["n_folds"] = len(folds)
    row["f1_raw_mean"] = float(raw_a.mean())
    row["f1_raw_sd"] = float(raw_a.std(ddof=1)) if len(folds) > 1 else 0.0
    row["f1_recon_mean"] = float(recon_a.mean())
    row["f1_recon_sd"] = float(recon_a.std(ddof=1)) if len(folds) > 1 else 0.0
    row["delta_f1_mean"] = float((raw_a - recon_a).mean())
    row["delta_f1_sd"] = float((raw_a - recon_a).std(ddof=1)) if len(folds) > 1 else 0.0
    return row


def main() -> None:
    """Per cell: compress tiles, train RF on raw vs recon, log the spatial-holdout F1 gap."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    gdf = _load_gdf(args.shapefile)
    if args.field is None:
        cols = [c for c in gdf.columns if c != gdf.geometry.name]
        logger.info("no --field given; shapefile columns: %s", cols)
        return
    from geotessera import GeoTessera  # noqa: PLC0415
    from sklearn.preprocessing import LabelEncoder  # noqa: PLC0415

    cells = _build_cells(list(args.tile_sizes), list(args.configs))
    le = LabelEncoder().fit(gdf[args.field].dropna().unique())
    gt = GeoTessera()
    rows: list[dict[str, Any]] = []
    out_path = f"{args.out_dir}/{args.tag}_downstream.parquet"
    for cell in cells:
        data = _collect_cell(
            gt, gdf, args.field, le, cell, args.year, args.seed, args.int8_codebooks
        )
        row = _cell_result(data, cell, args.spatial_folds, args.max_train, args.seed)
        row["int8_codebooks"] = bool(args.int8_codebooks)
        rows.append(row)
        write_parquet_with_provenance(
            pd.DataFrame(rows), out_path, seed=args.seed, config_path=args.shapefile
        )
        logger.info("cell t=%d k1=%d k2=%d done: %s", *cell, rows[-1])
    logger.info("downstream sweep complete -> %s", out_path)


if __name__ == "__main__":
    main()
