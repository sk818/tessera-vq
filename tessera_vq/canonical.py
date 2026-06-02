"""Canonical bbox loader + path-aware read helper for the Phase 3 RVQ sweep.

Loads the 100-entry list in ``config/canonical_bboxes.yaml`` and reads each as
either a native 1000x1000 UTM window (zarr path, when available) or a bbox-
fallback EPSG:4326-reprojected mosaic (numpy fallback). Records which path
was used so the sweep CSV can carry that as provenance.

Per-entry coords end in .X5 so a +/- 0.045-degree (~10 km) box sits inside a
single 0.1-degree Tessera tile and never straddles a tile edge; the loader
validates this invariant on load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import yaml

from tessera_vq import zarr_utils as _zarr_utils
from tessera_vq.data import _read_window_native, _window_bounds, read_region

# Vendored zarr_utils is untyped; alias as Any so strict mypy accepts calls into it.
# Matches the pattern used in tessera_vq/data.py.
zarr_utils: Any = _zarr_utils

PathChoice = Literal["zarr", "bbox", "unavailable"]


@dataclass(frozen=True)
class CanonicalBbox:
    """One canonical bbox: a Tessera tile-centered 10 km square at ``(lon, lat)``.

    ``lon, lat`` are tile centers on the 0.1-degree Tessera grid (coords end in .X5
    so the +/- 0.045-degree window stays inside a single tile). ``biome`` and
    ``continent`` are descriptive only and do not change the read path.
    """

    name: str
    lon: float
    lat: float
    biome: str
    continent: str


def load_canonical_bboxes(path: str | Path) -> list[CanonicalBbox]:
    """Load and validate the canonical bbox list from a YAML config file.

    Raises ``ValueError`` if any entry's ``(lon, lat)`` is not on the .X5 grid.
    """
    with Path(path).open() as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    raw_entries = cast("list[dict[str, Any]]", cfg["bboxes"])
    bboxes = [CanonicalBbox(**b) for b in raw_entries]
    for b in bboxes:
        for axis, val in (("lon", b.lon), ("lat", b.lat)):
            if round((val % 0.1) * 1000) != 50:  # noqa: PLR2004
                raise ValueError(
                    f"{b.name}: {axis}={val} is not on the .X5 Tessera tile-center grid"
                )
    return bboxes


def select_path(bbox: CanonicalBbox, year: int, *, window_px: int = 1000) -> PathChoice:
    """Probe zarr for this bbox; ``"zarr"`` if the window is covered, else ``"bbox"``."""
    gtz = zarr_utils.get_zarr()
    if gtz is None:
        return "bbox"
    bounds = _window_bounds(bbox.lon, bbox.lat, window_px)
    return "zarr" if zarr_utils.probe_zarr_coverage(gtz, bounds, year) else "bbox"


def read_canonical_window(
    bbox: CanonicalBbox,
    year: int = 2024,
    *,
    window_px: int = 1000,
    max_nan_fraction: float = 0.5,
) -> tuple[npt.NDArray[np.float32] | None, PathChoice]:
    """Read the 10 km window for ``bbox`` via zarr-first, bbox-fallback path.

    Returns ``(mosaic, path_used)``. ``mosaic`` is float32 ``(H, W, 128)``; on
    the zarr path it is exactly ``(window_px, window_px, 128)`` in native UTM
    (no reprojection); on the bbox path it is roughly ``(~window_px, variable, 128)``
    in EPSG:4326. ``path_used`` is ``"unavailable"`` iff both reads return None
    (e.g., bbox sits outside Tessera's published coverage entirely).
    """
    path = select_path(bbox, year, window_px=window_px)
    if path == "zarr":
        gtz = zarr_utils.get_zarr()
        loc = np.asarray([bbox.lon, bbox.lat], dtype=np.float64)
        patch = _read_window_native(loc, window_px, year, gtz, max_nan_fraction)
        if patch is not None:
            return patch, "zarr"
    bounds = _window_bounds(bbox.lon, bbox.lat, window_px)
    mosaic, _read_path = read_region(bounds, year)
    if mosaic is None:
        return None, "unavailable"
    return mosaic.astype(np.float32, copy=False), "bbox"
