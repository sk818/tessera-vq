"""Zarr region reads for the bolt-on, on geotessera's ``GeoTesseraZarr`` directly.

Folded in from the external ``tessera-zarr-utils`` package once geotessera
0.10.1 fixed the two upstream problems that package existed to work around --
the UTM-zone-boundary bug in zarr region reads, and an incomplete rollout of
years other than 2024. Same public surface (``get_zarr`` /
``probe_zarr_coverage`` / ``read_region_chunked``) so ``data.py`` and
``canonical.py`` only swap the import.

Chunk edges break at 6-degree multiples (UTM zone boundaries): geotessera's
``read_region`` serves a zone-straddling bbox from the centre zone alone and
silently clips the far side to NaN, so every sub-read must stay within one
zone. Mirrors the fix in tessera-eval's compute server.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from affine import Affine

logger = logging.getLogger(__name__)

_Mosaic = npt.NDArray[np.float32]
_Bounds = tuple[float, float, float, float]

CHUNK_THRESHOLD = 0.2  # deg -- regions larger than this go through the chunked merge
CHUNK_SIZE = 0.1  # deg per chunk
_ZARR_CACHE_MAX_BYTES = 20 * 1024**3  # bound the on-disk zarr chunk cache

# Temporarily force every embedding read onto the NPY/bbox path, bypassing
# GeoTesseraZarr entirely. The zarr fast path is currently using a smaller
# chunk size than intended (Keshav, 2026-09-23) -- same symptom reported
# independently for tessera-eval's compute server (Moustafa Eweda,
# forwarded to Anil): the on-disk chunk cache only persists metadata, not
# the actual tile data, so every read still hits the network regardless of
# caching. Mirrors tessera-eval's identical _ZARR_DISABLED flag. get_zarr()
# is the single choke point every caller in this module goes through
# (data.py's own _USE_ZARR gate, canonical.py, iter_region_windows's
# hard-require-zarr path) -- disabling it here covers all of them, rather
# than needing a second flag per call site. Set back to False once
# geotessera's chunk-size/caching behaviour is confirmed fixed upstream.
_ZARR_DISABLED = True


def _cache_dir() -> Path:
    # Same root as the RVQ response cache (TESSERA_VQ_CACHE_DIR, set on the
    # michael deployment); a "zarr" subdir keeps the two apart. The 20 GiB cap
    # below is independent of TESSERA_VQ_CACHE_MAX_GB.
    base = os.environ.get("TESSERA_VQ_CACHE_DIR")
    d = (Path(base) if base else Path.home() / ".cache" / "tessera-vq") / "zarr"
    d.mkdir(parents=True, exist_ok=True)
    return d


@lru_cache(maxsize=1)
def _open_zarr() -> Any:
    """Open ``GeoTesseraZarr`` once (chunk reads cached on disk, 20 GiB cap).

    Returns the handle, or ``None`` if the store can't be opened or has no
    tiles -- cached either way, so a failed open isn't retried every call.
    """
    try:
        from geotessera.store import GeoTesseraZarr  # noqa: PLC0415 (lazy: heavy optional import)

        inst = GeoTesseraZarr(cache_dir=str(_cache_dir()), cache_max_size=_ZARR_CACHE_MAX_BYTES)
        if getattr(inst, "years", None):
            logger.info("GeoTesseraZarr available: %s", inst.url)
            return inst
        logger.info("zarr store has no tiles; using NPY path")
        return None
    except Exception as e:  # noqa: BLE001 -- any failure => NPY fallback
        logger.info("zarr store unavailable (%s); using NPY path", e)
        return None


def get_zarr() -> Any:
    """Cached ``GeoTesseraZarr`` handle, or ``None`` -- callers fall back to NPY."""
    if _ZARR_DISABLED:
        return None
    return _open_zarr()


def probe_zarr_coverage(gtz: Any, bounds: _Bounds, year: int) -> bool:
    """True when zarr has a *valid* embedding at the centre of ``bounds`` for
    ``year``. geotessera's single-pixel probe tells genuine coverage apart from
    sea and from areas not yet produced; anything else sends the caller to NPY.
    """
    try:
        if year not in getattr(gtz, "years", []):
            return False
        cx = (bounds[0] + bounds[2]) / 2.0
        cy = (bounds[1] + bounds[3]) / 2.0
        _vec, status = gtz.probe(cx, cy, year)
        return bool(status == "valid")
    except Exception:  # noqa: BLE001
        return False


def _utm_zone(lon: float) -> int:
    """UTM zone number for a longitude (geotessera routes each read by its centre)."""
    return int((lon + 180.0) // 6.0) + 1


def _zone_split_spans(lo: float, hi: float) -> list[tuple[float, float]]:
    """``[(lo, e0), (e0, e1), ...]`` covering ``[lo, hi)`` -- each span <=
    CHUNK_SIZE and never crossing a 6-degree UTM zone edge."""
    out: list[tuple[float, float]] = []
    x = lo
    while x < hi:
        zone_edge = (np.floor(x / 6.0) + 1) * 6.0
        end = float(min(x + CHUNK_SIZE, zone_edge, hi))
        out.append((x, end))
        x = end
    return out


def _plain_spans(lo: float, hi: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    x = lo
    while x < hi:
        out.append((x, min(x + CHUNK_SIZE, hi)))
        x += CHUNK_SIZE
    return out


def _reproject_to_4326(mosaic: _Mosaic, transform: Any, src_crs: Any) -> tuple[_Mosaic, Any, str]:
    """Reproject a ``(H, W, B)`` mosaic to EPSG:4326.

    geotessera's zarr reads return data in a native metre CRS (a UTM zone),
    but the bolt-on wire format and its TEE consumer assume lon/lat degrees.
    Nearest-neighbour resampling only -- bilinear would blend the 128-d
    embeddings. NaN nodata is carried through. No-op if already EPSG:4326.
    """
    dst_crs = "EPSG:4326"
    if str(src_crs).upper().replace(" ", "") in ("EPSG:4326", "WGS84"):
        return mosaic, transform, dst_crs

    from rasterio.warp import (  # noqa: PLC0415 (lazy: server-only dep)
        Resampling,
        calculate_default_transform,
        reproject,
    )

    h, w, bands = mosaic.shape
    left, top = transform.c, transform.f
    right = left + transform.a * w
    bottom = top + transform.e * h

    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, w, h, left=left, bottom=bottom, right=right, top=top
    )
    src = np.ascontiguousarray(np.transpose(mosaic, (2, 0, 1)))  # (B, H, W)
    dst = np.full((bands, dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        source=src,
        destination=dst,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return np.transpose(dst, (1, 2, 0)), dst_transform, dst_crs


def _reproject_chunk_into(
    mosaic: _Mosaic,
    dst_transform: Any,
    emb: _Mosaic,
    src_transform: Any,
    src_crs: Any,
    chunk_bounds: _Bounds,
) -> None:
    """Reproject one native-CRS chunk into the shared EPSG:4326 ``mosaic`` in place.

    Only the chunk's geographic sub-window is written, and existing non-NaN
    values are kept where the reprojected chunk is NaN. Nearest-neighbour only.
    Temp memory is bounded to one chunk window.
    """
    from rasterio.warp import Resampling, reproject  # noqa: PLC0415 (lazy: server-only dep)

    h_full, w_full, _bands = mosaic.shape
    px_lon, px_lat = dst_transform.a, -dst_transform.e
    west0, north0 = dst_transform.c, dst_transform.f
    lon0, lat0, lon1, lat1 = chunk_bounds

    c0 = max(0, int(np.floor((lon0 - west0) / px_lon)) - 1)
    c1 = min(w_full, int(np.ceil((lon1 - west0) / px_lon)) + 1)
    r0 = max(0, int(np.floor((north0 - lat1) / px_lat)) - 1)
    r1 = min(h_full, int(np.ceil((north0 - lat0) / px_lat)) + 1)
    if c1 <= c0 or r1 <= r0:
        return

    win_transform = Affine(px_lon, 0.0, west0 + c0 * px_lon, 0.0, -px_lat, north0 - r0 * px_lat)
    src = np.ascontiguousarray(np.transpose(emb, (2, 0, 1)))  # (B, h, w)
    tmp = np.full((emb.shape[2], r1 - r0, c1 - c0), np.nan, dtype=np.float32)
    reproject(
        source=src,
        destination=tmp,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=win_transform,
        dst_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    tmp2 = np.transpose(tmp, (1, 2, 0))  # (rows, cols, B)
    covered = ~np.isnan(tmp2).all(axis=2)
    mosaic[r0:r1, c0:c1][covered] = tmp2[covered]


def read_region_chunked(
    gtz: Any, bounds: _Bounds, year: int
) -> tuple[_Mosaic | None, Any, str | None]:
    """Read a region via zarr as one EPSG:4326 ``(H, W, 128)`` float32 mosaic.

    A small in-zone region is one native read + reproject. Anything larger or
    zone-straddling is read as 0.1-degree chunks -- split at 6-degree zone
    edges so no chunk is served clipped -- and reprojected into one shared
    grid. Returns ``(mosaic, transform, "EPSG:4326")`` or ``(None, None, None)``.
    """
    west, south, east, north = bounds
    lon_span, lat_span = east - west, north - south

    if (
        lon_span <= CHUNK_THRESHOLD
        and lat_span <= CHUNK_THRESHOLD
        and _utm_zone(west) == _utm_zone(east)
    ):
        mosaic, transform, crs = gtz.read_region(bounds, year)
        if mosaic is None:
            return None, None, None
        return _reproject_to_4326(np.asarray(mosaic, dtype=np.float32), transform, crs)

    from rasterio.warp import calculate_default_transform  # noqa: PLC0415 (lazy: server-only dep)

    chunk_lons = _zone_split_spans(west, east)
    chunk_lats = _plain_spans(south, north)
    logger.info(
        "zarr region %s: %d chunks (%d x %d) -> shared EPSG:4326 grid",
        bounds,
        len(chunk_lons) * len(chunk_lats),
        len(chunk_lons),
        len(chunk_lats),
    )

    read: list[tuple[Any, Any, Any, _Bounds]] = []
    for lat0, lat1 in chunk_lats:
        for lon0, lon1 in chunk_lons:
            cb: _Bounds = (lon0, lat0, lon1, lat1)
            try:
                emb, tfm, crs = gtz.read_region(cb, year)
            except Exception as e:  # noqa: BLE001 -- one bad chunk must not abort
                logger.warning("zarr chunk %s failed: %s", cb, e)
                continue
            if emb is None or emb.size == 0:
                continue
            read.append((np.asarray(emb, dtype=np.float32), tfm, crs, cb))

    if not read:
        return None, None, None

    emb0, tfm0, crs0, _cb0 = read[0]
    h0, w0, bands = emb0.shape
    dt0, _dw, _dh = calculate_default_transform(
        crs0,
        "EPSG:4326",
        w0,
        h0,
        left=tfm0.c,
        bottom=tfm0.f + tfm0.e * h0,
        right=tfm0.c + tfm0.a * w0,
        top=tfm0.f,
    )
    px_lon, px_lat = dt0.a, -dt0.e
    width = max(1, int(np.ceil(lon_span / px_lon)))
    height = max(1, int(np.ceil(lat_span / px_lat)))
    dst_transform = Affine(px_lon, 0.0, west, 0.0, -px_lat, north)
    mosaic = np.full((height, width, bands), np.nan, dtype=np.float32)

    for emb, tfm, crs, cb in read:
        _reproject_chunk_into(mosaic, dst_transform, emb, tfm, crs, cb)

    return mosaic, dst_transform, "EPSG:4326"
