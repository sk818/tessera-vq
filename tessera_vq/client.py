"""Plug-compatible Python client for the Tessera VQ bolt-on.

Drop-in subset of ``geotessera.GeoTessera`` for downstream code that wants reconstructed
embeddings without holding the raw 128-d floats. Each fetch POSTs ``/quantized`` to the
bolt-on (which runs LAN-close to the embeddings store), receives a small NPZ of
codebooks + per-tile index maps, and rebuilds ``(H, W, 128)`` float32 in EPSG:4326.

Configure ``(t, k, m)`` at construction; only EPSG:4326 is supported as output.

Example::

    from tessera_vq.client import VQTessera

    gt = VQTessera(server_url="http://michael:8000", t=64, k=16, m="cosine")
    mosaic, transform, crs = gt.fetch_mosaic_for_region(
        (0.145, 52.045, 0.155, 52.055), year=2024
    )
"""

from __future__ import annotations

import io
import json
import urllib.request
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from affine import Affine

Distance = Literal["euclidean", "cosine"]


class VQTessera:
    """Plug-compatible subset of ``geotessera.GeoTessera`` over the VQ bolt-on."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        t: int = 64,
        k: int = 16,
        m: Distance = "euclidean",
        timeout: float = 120.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.t = int(t)
        self.k = int(k)
        self.m: Distance = m
        self.timeout = float(timeout)

    def fetch_mosaic_for_region(
        self,
        bbox: tuple[float, float, float, float],
        year: int = 2024,
        target_crs: str = "EPSG:4326",
        auto_download: bool = True,  # noqa: ARG002  (kept for geotessera API compat)
    ) -> tuple[npt.NDArray[np.float32], Affine, str]:
        """Fetch reconstructed embeddings for ``bbox``; returns ``(mosaic, transform, crs)``."""
        if target_crs.upper() not in ("EPSG:4326", "WGS84"):
            raise ValueError(f"only EPSG:4326 is supported; got {target_crs!r}")
        payload: dict[str, Any] = {
            "bbox": list(bbox),
            "year": int(year),
            "t": self.t,
            "k": self.k,
            "m": self.m,
        }
        npz_bytes = self._post("/quantized", payload)
        return _reconstruct(npz_bytes, bbox)

    def fetch_embedding(
        self, lon: float, lat: float, year: int = 2024
    ) -> tuple[npt.NDArray[np.float32], Affine, str]:
        """Fetch the embedding mosaic for the 0.1-degree tile around ``(lon, lat)``."""
        bounds = (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
        return self.fetch_mosaic_for_region(bounds, year=year)

    def _post(self, path: str, payload: dict[str, Any]) -> bytes:
        """POST JSON to the bolt-on; return the raw response body."""
        req = urllib.request.Request(
            self.server_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return bytes(resp.read())


def _reconstruct(
    npz_bytes: bytes, bbox: tuple[float, float, float, float]
) -> tuple[npt.NDArray[np.float32], Affine, str]:
    """Decode a ``/quantized`` NPZ and rebuild ``(H, W, 128)`` float32 in EPSG:4326.

    Uncovered tiles (NaN-filtered out by the server) remain NaN in the output. The
    affine transform maps pixel ``(0, 0)`` to the bbox's top-left corner; pixel size
    is the bbox span divided by the server-reported mosaic shape (may differ from the
    exact zarr-native pixel size by at most one pixel due to reprojection rounding).
    """
    with np.load(io.BytesIO(npz_bytes)) as data:
        codebooks: npt.NDArray[np.float32] = data["codebooks"]
        indices = data["indices"]
        positions = data["positions"]
        meta = data["meta"]
    t = int(meta[0])
    full_h, full_w = int(meta[3]), int(meta[4])
    out_h = (full_h // t) * t
    out_w = (full_w // t) * t
    channels = int(codebooks.shape[-1])
    mosaic = np.full((out_h, out_w, channels), np.nan, dtype=np.float32)
    for i in range(int(positions.shape[0])):
        r, c = int(positions[i, 0]), int(positions[i, 1])
        mosaic[r * t : (r + 1) * t, c * t : (c + 1) * t] = codebooks[i][indices[i]]
    lon0, lat0, lon1, lat1 = bbox
    dx = (lon1 - lon0) / full_w
    dy = (lat1 - lat0) / full_h
    return mosaic, Affine(dx, 0.0, lon0, 0.0, -dy, lat1), "EPSG:4326"
