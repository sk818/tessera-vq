"""Data loaders for Tessera embeddings, Pool A diagnostics, and downstream tasks.

Implemented in Phase 1 (docs/spec.md) over geotessera (zarr via tessera-eval's
``zarr_utils``, with a bounding-box fallback): ``read_region``,
``iter_pool_a_windows``, ``sample_isotropy_points``. Land-only sampling from
geotessera coverage; no embeddings persisted. ``load_downstream`` is deferred
(Phases 5-6).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from tessera_eval import zarr_utils as _zarr_utils

# tessera_eval.zarr_utils is untyped; alias as Any so strict mypy accepts calls into it.
zarr_utils: Any = _zarr_utils

logger = logging.getLogger(__name__)

_TILE_HALF_DEG = 0.05  # geotessera 0.1-degree tiles span +/- 0.05 deg from centre
_M_PER_DEG_LAT = 111_320.0  # metres per degree of latitude (WGS84 approx)
_PIXEL_M = 10.0  # Tessera ground resolution


@lru_cache(maxsize=1)
def get_geotessera() -> Any:
    """Cached GeoTessera client (geotessera is untyped, hence Any)."""
    from geotessera import GeoTessera  # noqa: PLC0415  (lazy: heavy optional import)

    return GeoTessera()


@lru_cache(maxsize=4)
def available_land_centers(year: int) -> npt.NDArray[np.float64]:
    """(M, 2) lon/lat centres of geotessera tiles available for ``year`` (land only)."""
    tiles = get_geotessera().registry.get_available_embeddings()
    centers = np.array([(lon, lat) for (y, lon, lat) in tiles if y == year], dtype=np.float64)
    if centers.size == 0:
        raise RuntimeError(f"no geotessera embeddings available for year {year}")
    return centers


def sample_window_locations(n: int, year: int, seed: int) -> npt.NDArray[np.float64]:
    """Random (n, 2) lon/lat tile centres drawn from land coverage for ``year``."""
    centers = available_land_centers(year)
    rng = np.random.default_rng(seed)
    idx = rng.choice(centers.shape[0], size=n, replace=centers.shape[0] < n)
    return centers[idx]


def read_region(
    bounds: tuple[float, float, float, float],
    year: int,
    *,
    gtz: Any = None,
    gt: Any = None,
) -> tuple[npt.NDArray[np.float32] | None, str]:
    """Read ``(H, W, 128)`` float32 EPSG:4326 for ``bounds``; zarr if covered, else bbox.

    Returns ``(mosaic_or_None, path)`` with ``path`` in ``{"zarr", "bbox", "empty"}``.
    """
    gtz = zarr_utils.get_zarr() if gtz is None else gtz
    if gtz is not None and zarr_utils.probe_zarr_coverage(gtz, bounds, year):
        mosaic, _, _ = zarr_utils.read_region_chunked(gtz, bounds, year)
        if mosaic is not None:
            return np.asarray(mosaic, dtype=np.float32), "zarr"
    gt = get_geotessera() if gt is None else gt
    mosaic, _, _ = gt.fetch_mosaic_for_region(bounds, year=year, target_crs="EPSG:4326")
    if mosaic is None:
        return None, "empty"
    return np.asarray(mosaic, dtype=np.float32), "bbox"


def _finite_pixels(mosaic: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Flatten ``(H, W, C)`` to ``(P, C)`` keeping only all-finite (land) pixels."""
    flat = mosaic.reshape(-1, mosaic.shape[-1])
    return flat[np.isfinite(flat).all(axis=1)]


def _sample_one_window(
    loc: npt.NDArray[np.float64],
    year: int,
    points_per_window: int,
    region_deg: float,
    seed: int,
    gtz: Any,
    gt: Any,
) -> tuple[npt.NDArray[np.float32] | None, str]:
    """Read a small jittered region at ``loc`` and return up to N finite pixels + path."""
    rng = np.random.default_rng(seed)
    half = region_deg / 2.0
    jitter = _TILE_HALF_DEG - half
    clon = float(loc[0]) + rng.uniform(-jitter, jitter)
    clat = float(loc[1]) + rng.uniform(-jitter, jitter)
    bounds = (clon - half, clat - half, clon + half, clat + half)
    try:
        mosaic, path = read_region(bounds, year, gtz=gtz, gt=gt)
    except Exception as exc:  # one bad tile must not abort the whole batch
        logger.debug("window read failed at (%.3f, %.3f): %s", clon, clat, exc)
        return None, "error"
    if mosaic is None:
        return None, path
    finite = _finite_pixels(mosaic)
    if finite.shape[0] == 0:
        return None, "nan_skipped"
    take = min(points_per_window, finite.shape[0])
    return finite[rng.choice(finite.shape[0], size=take, replace=False)], path


def sample_isotropy_points(
    points_per_window: int = 100,
    n_windows: int = 1000,
    year: int = 2024,
    seed: int = 42,
    *,
    region_deg: float = 0.01,
    n_jobs: int = 8,
) -> tuple[npt.NDArray[np.float32], dict[str, int]]:
    """Collect ``points_per_window`` random land pixels from each of ``n_windows`` windows.

    Reads a small region per window (not a whole tile) in parallel and samples finite
    pixels from it; nothing is persisted. Returns ``((N, 128) float32, path_counts)``.
    """
    locs = sample_window_locations(n_windows, year, seed)
    gtz, gt = zarr_utils.get_zarr(), get_geotessera()
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_sample_one_window)(
            locs[i], year, points_per_window, region_deg, seed + 1 + i, gtz, gt
        )
        for i in range(len(locs))
    )
    counts = {"zarr": 0, "bbox": 0, "empty": 0, "nan_skipped": 0}
    chunks: list[npt.NDArray[np.float32]] = []
    for pts, path in results:
        counts[path] = counts.get(path, 0) + 1
        if pts is not None:
            chunks.append(pts)
    if not chunks:
        raise RuntimeError("no isotropy points collected")
    return np.concatenate(chunks, axis=0), counts


def _window_bounds(lon: float, lat: float, window_px: int) -> tuple[float, float, float, float]:
    """Bounding box (EPSG:4326) of a ``window_px`` square centred at ``(lon, lat)``."""
    half_m = window_px * _PIXEL_M / 2.0
    dlat = half_m / _M_PER_DEG_LAT
    dlon = half_m / (_M_PER_DEG_LAT * max(float(np.cos(np.radians(lat))), 1e-6))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def iter_pool_a_windows(
    n_windows: int = 1000,
    window_px: int = 1024,
    year: int = 2024,
    seed: int = 42,
) -> Iterator[npt.NDArray[np.float32]]:
    """Yield ``(H, W, 128)`` float32 land windows of ``window_px`` (zarr, else bbox)."""
    locs = sample_window_locations(n_windows, year, seed)
    gtz, gt = zarr_utils.get_zarr(), get_geotessera()
    for lon, lat in locs:
        bounds = _window_bounds(float(lon), float(lat), window_px)
        mosaic, _ = read_region(bounds, year, gtz=gtz, gt=gt)
        if mosaic is not None:
            yield mosaic


# Approximate UK bounding box (lon_min, lat_min, lon_max, lat_max); zarr-covered.
UK_BBOX = (-8.0, 50.0, 1.9, 58.7)


def iter_region_windows(
    bbox: tuple[float, float, float, float],
    n_windows: int,
    window_px: int = 1024,
    year: int = 2024,
    seed: int = 42,
    *,
    max_nan_fraction: float = 0.5,
    max_tries_factor: int = 30,
) -> Iterator[npt.NDArray[np.float32]]:
    """Yield up to ``n_windows`` land patches (``window_px`` square) random within ``bbox``.

    Zarr-only fast path: probes coverage and re-samples windows that are uncovered or
    mostly NaN (sea). Embeddings are not persisted.
    """
    rng = np.random.default_rng(seed)
    gtz = zarr_utils.get_zarr()
    if gtz is None:
        raise RuntimeError("zarr unavailable; the region fast path requires zarr")
    lon0, lat0, lon1, lat1 = bbox
    kept = 0
    for _ in range(n_windows * max_tries_factor):
        if kept >= n_windows:
            break
        lon, lat = float(rng.uniform(lon0, lon1)), float(rng.uniform(lat0, lat1))
        bounds = _window_bounds(lon, lat, window_px)
        if not zarr_utils.probe_zarr_coverage(gtz, bounds, year):
            continue
        try:
            mosaic, _, _ = zarr_utils.read_region_chunked(gtz, bounds, year)
        except Exception as exc:  # one bad patch must not abort sampling
            logger.debug("region window read failed at (%.3f, %.3f): %s", lon, lat, exc)
            continue
        if mosaic is None:
            continue
        patch = np.asarray(mosaic, dtype=np.float32)
        if float(np.isnan(patch).any(axis=2).mean()) > max_nan_fraction:
            continue
        kept += 1
        yield patch
    if kept < n_windows:
        logger.warning("only %d/%d windows kept within bbox", kept, n_windows)


def _sample_region_points(
    loc: npt.NDArray[np.float64], region_px: int, points: int, year: int, seed: int, gtz: Any
) -> npt.NDArray[np.float32] | None:
    """Native zarr read of a small region at ``loc``; return up to ``points`` land pixels."""
    bounds = _window_bounds(float(loc[0]), float(loc[1]), region_px)
    try:
        mosaic, _, _ = gtz.read_region(bounds, year)  # native CRS: no reprojection
    except Exception as exc:  # one bad region must not abort the batch
        logger.debug("region read failed at (%.3f, %.3f): %s", loc[0], loc[1], exc)
        return None
    if mosaic is None:
        return None
    finite = _finite_pixels(np.asarray(mosaic, dtype=np.float32))
    if finite.shape[0] == 0:
        return None
    rng = np.random.default_rng(seed)
    take = min(points, finite.shape[0])
    sel = rng.choice(finite.shape[0], size=take, replace=False)
    return np.asarray(finite[sel], dtype=np.float32)


def sample_isotropy_uk(
    n_regions: int,
    points_per_region: int = 1000,
    region_px: int = 128,
    bbox: tuple[float, float, float, float] = UK_BBOX,
    year: int = 2024,
    seed: int = 42,
    n_jobs: int = 12,
) -> tuple[npt.NDArray[np.float32], int]:
    """Parallel small native zarr reads across ``bbox``; returns ``((N, 128) float32, n_ok)``.

    Concurrency hides the per-chunk request latency that makes large single reads slow.
    """
    rng = np.random.default_rng(seed)
    centers = available_land_centers(year)
    lon0, lat0, lon1, lat1 = bbox
    mask = (
        (centers[:, 0] >= lon0)
        & (centers[:, 0] <= lon1)
        & (centers[:, 1] >= lat0)
        & (centers[:, 1] <= lat1)
    )
    uk = centers[mask]
    if uk.shape[0] == 0:
        raise RuntimeError("no land tiles in bbox")
    locs = uk[rng.choice(uk.shape[0], size=n_regions, replace=uk.shape[0] < n_regions)]
    gtz = zarr_utils.get_zarr()
    if gtz is None:
        raise RuntimeError("zarr unavailable")
    res = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_sample_region_points)(
            locs[i], region_px, points_per_region, year, seed + 1 + i, gtz
        )
        for i in range(n_regions)
    )
    chunks = [r for r in res if r is not None]
    if not chunks:
        raise RuntimeError("no isotropy points collected")
    return np.concatenate(chunks, axis=0), len(chunks)


def _read_window_native(
    loc: npt.NDArray[np.float64],
    window_px: int,
    year: int,
    gtz: Any,
    max_nan_fraction: float,
) -> npt.NDArray[np.float32] | None:
    """Native zarr read of a ``window_px`` patch at ``loc`` (no reproject)."""
    bounds = _window_bounds(float(loc[0]), float(loc[1]), window_px)
    try:
        mosaic, _, _ = gtz.read_region(bounds, year)
    except Exception as exc:  # one bad window must not abort the batch
        logger.debug("UK window read failed at (%.3f, %.3f): %s", loc[0], loc[1], exc)
        return None
    if mosaic is None:
        return None
    patch = np.asarray(mosaic, dtype=np.float32)
    if float(np.isnan(patch).any(axis=2).mean()) > max_nan_fraction:
        return None
    return patch


def iter_uk_windows_parallel(
    n_windows: int,
    window_px: int = 1024,
    bbox: tuple[float, float, float, float] = UK_BBOX,
    year: int = 2024,
    seed: int = 42,
    *,
    n_jobs: int = 4,
    max_nan_fraction: float = 0.5,
) -> Iterator[npt.NDArray[np.float32]]:
    """Yield UK land patches (``window_px`` square) read in parallel batches via zarr."""
    rng = np.random.default_rng(seed)
    centers = available_land_centers(year)
    lon0, lat0, lon1, lat1 = bbox
    mask = (
        (centers[:, 0] >= lon0)
        & (centers[:, 0] <= lon1)
        & (centers[:, 1] >= lat0)
        & (centers[:, 1] <= lat1)
    )
    uk = centers[mask]
    if uk.shape[0] == 0:
        raise RuntimeError("no land tiles in bbox")
    locs = uk[rng.choice(uk.shape[0], size=n_windows, replace=uk.shape[0] < n_windows)]
    gtz = zarr_utils.get_zarr()
    if gtz is None:
        raise RuntimeError("zarr unavailable")
    kept = 0
    for batch_start in range(0, n_windows, n_jobs):
        batch = locs[batch_start : batch_start + n_jobs]
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_read_window_native)(loc, window_px, year, gtz, max_nan_fraction)
            for loc in batch
        )
        for patch in results:
            if patch is not None:
                kept += 1
                yield patch
    if kept < n_windows:
        logger.warning("kept %d/%d UK windows", kept, n_windows)


def collect_isotropy_points(
    bbox: tuple[float, float, float, float],
    n_windows: int,
    points_per_window: int = 1000,
    window_px: int = 1024,
    year: int = 2024,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float32], int]:
    """Stream land patches in ``bbox`` and sample ``points_per_window`` finite pixels each."""
    rng = np.random.default_rng(seed + 1)
    chunks: list[npt.NDArray[np.float32]] = []
    n_patches = 0
    for patch in iter_region_windows(bbox, n_windows, window_px, year, seed):
        n_patches += 1
        finite = _finite_pixels(patch)
        if finite.shape[0] == 0:
            continue
        take = min(points_per_window, finite.shape[0])
        chunks.append(finite[rng.choice(finite.shape[0], size=take, replace=False)])
    if not chunks:
        raise RuntimeError("no isotropy points collected")
    return np.concatenate(chunks, axis=0), n_patches
