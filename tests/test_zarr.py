"""tessera_vq._zarr: the seam-safe chunked EPSG:4326 merge.

The fake GeoTesseraZarr mimics geotessera's read_region: it serves a bbox from
its centre's UTM zone only, filling pixels whose true longitude is in that zone
and leaving the rest NaN. So a chunk that stays inside one zone comes back fully
covered; one that straddles a 6-degree edge would come back half-NaN -- which is
exactly what _zone_split_spans must prevent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from affine import Affine

import tessera_vq._zarr as zarr_mod
from tessera_vq._zarr import _zone_split_spans, get_zarr, read_region_chunked

pytest.importorskip("rasterio.warp")
from rasterio.warp import transform as warp_transform  # noqa: E402


def _utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def _zone_of(lon: float) -> int:
    return int((lon + 180) // 6) + 1


class FakeZarr:
    """read_region(bbox) -> data in the bbox centre's UTM zone, NaN outside that zone."""

    def __init__(self, res_m: float = 10.0, bands: int = 4) -> None:
        self.res_m = res_m
        self.bands = bands

    def read_region(self, bounds: tuple[float, float, float, float], year: int) -> Any:
        lon0, lat0, lon1, lat1 = bounds
        clon, clat = (lon0 + lon1) / 2, (lat0 + lat1) / 2
        crs = _utm_epsg(clon, clat)
        centre_zone = _zone_of(clon)
        (x0,), (y1,) = warp_transform("EPSG:4326", crs, [lon0], [lat1])
        (x1,), (y0,) = warp_transform("EPSG:4326", crs, [lon1], [lat0])
        w = max(1, round((x1 - x0) / self.res_m))
        h = max(1, round((y1 - y0) / self.res_m))
        tfm = Affine(self.res_m, 0, x0, 0, -self.res_m, y1)
        cols = x0 + (np.arange(w) + 0.5) * self.res_m
        rows = y1 - (np.arange(h) + 0.5) * self.res_m
        gx, gy = np.meshgrid(cols, rows)
        blons, _ = warp_transform(crs, "EPSG:4326", gx.ravel(), gy.ravel())
        in_zone = np.array([_zone_of(v) for v in blons]).reshape(h, w) == centre_zone
        emb = np.full((h, w, self.bands), np.nan, dtype=np.float32)
        emb[in_zone] = float(centre_zone)  # tag = zone number
        return emb, tfm, crs


def test_zone_split_spans_never_crosses_a_6deg_edge() -> None:
    spans = _zone_split_spans(-0.28, 0.13)  # crosses 0 (zone 30|31)
    assert spans[0][0] == -0.28 and spans[-1][1] == pytest.approx(0.13)
    for lo, hi in spans:
        assert hi - lo <= 0.1 + 1e-9
        assert _zone_of(lo) == _zone_of(hi - 1e-9), f"span {(lo, hi)} straddles a zone edge"
    # 0.0 itself must be a break point
    assert any(abs(hi - 0.0) < 1e-9 for _, hi in spans)


def test_zone_split_spans_plain_when_no_edge_inside() -> None:
    spans = _zone_split_spans(2.30, 2.55)  # no 6-degree edge in range
    assert spans[0][0] == pytest.approx(2.30)
    # contiguous 0.1-degree steps up to the end (each span starts where the last ended)
    assert [round(lo, 6) for lo, _hi in spans] == [2.30, 2.40, 2.50]
    assert [round(hi, 6) for _lo, hi in spans] == [2.40, 2.50, 2.55]


def test_read_region_chunked_covers_both_sides_of_a_zone_seam() -> None:
    # ~5 km box straddling 0degE -> UTM 30N west, 31N east.
    bounds = (-0.04, 52.20, 0.04, 52.26)
    mosaic, transform, crs = read_region_chunked(FakeZarr(), bounds, 2024)

    assert mosaic is not None and transform is not None
    assert crs == "EPSG:4326"
    _h, w, b = mosaic.shape
    assert b == 4
    covered = np.asarray(~np.isnan(mosaic).all(axis=2))
    # >0.9, not ~1.0: the per-chunk nearest-neighbour reproject leaves a thin
    # NaN strip at the merged grid's outer edge (proportionally large for a
    # 5 km box). The seam itself must be seamless -- checked per side below.
    assert covered.mean() > 0.9, f"coverage only {covered.mean():.2f} -- seam half dropped?"

    lons = transform.c + (np.arange(w) + 0.5) * transform.a
    west = np.where(lons < 0)[0]
    east = np.where(lons > 0)[0]
    assert west.size and east.size
    # The bug: read_region served a zone-straddling bbox from one zone only,
    # leaving the other side entirely NaN. Both sides must now carry data...
    assert covered[:, west].any(), "no data west of 0degE"
    assert covered[:, east].any(), "no data east of 0degE (second zone dropped?)"
    # ...and land on the right side: west from zone 30, east from zone 31.
    assert np.nanmedian(mosaic[:, west, 0]) == pytest.approx(30.0)
    assert np.nanmedian(mosaic[:, east, 0]) == pytest.approx(31.0)


def test_read_region_chunked_single_zone_small_region() -> None:
    bounds = (2.32, 48.82, 2.38, 48.88)  # wholly inside zone 31
    mosaic, _transform, crs = read_region_chunked(FakeZarr(), bounds, 2024)
    assert mosaic is not None
    assert crs == "EPSG:4326"
    assert (~np.isnan(mosaic).all(axis=2)).mean() > 0.95
    assert np.nanmedian(mosaic) == pytest.approx(31.0)


def test_read_region_chunked_no_data_returns_none() -> None:
    class Empty:
        def read_region(self, bounds: Any, year: Any) -> Any:  # noqa: ARG002
            return None, None, None

    assert read_region_chunked(Empty(), (2.3, 48.8, 2.5, 49.0), 2024) == (None, None, None)


def test_get_zarr_disabled_returns_none_without_even_trying_to_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The current production default (_ZARR_DISABLED=True, see its own
    module comment -- the zarr fast path's chunk-size/caching bug means
    every read hits the network regardless). get_zarr() must short-circuit
    before ever opening the store, not just happen to return None."""
    monkeypatch.setattr(zarr_mod, "_ZARR_DISABLED", True)
    zarr_mod._open_zarr.cache_clear()
    calls: list[Any] = []

    def _make(**kw: Any) -> FakeZarr:
        calls.append(kw)
        return FakeZarr()

    monkeypatch.setattr("geotessera.store.GeoTesseraZarr", _make)
    assert get_zarr() is None
    assert calls == [], "disabled zarr must not even attempt to open the store"


def test_get_zarr_enabled_still_opens_the_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flip side of the disabled test above -- with _ZARR_DISABLED
    False (re-enabled once the upstream bug is fixed), get_zarr() must
    still reach the real open/cache logic in _open_zarr()."""

    class _StoreWithYears(FakeZarr):
        years = [2024, 2025]  # _open_zarr() only keeps a store with tiles
        url = "fake://store"  # _open_zarr() logs this on success

    monkeypatch.setattr(zarr_mod, "_ZARR_DISABLED", False)
    zarr_mod._open_zarr.cache_clear()
    monkeypatch.setattr("geotessera.store.GeoTesseraZarr", lambda **kw: _StoreWithYears())  # noqa: ARG005
    assert get_zarr() is not None
