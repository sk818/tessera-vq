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

For callers that want the per-tile payload (codebooks + index maps + positions) without
the reconstruction step — useful for storage formats or sweep tooling — use
``fetch_quantized_structure``, which returns a :class:`QuantizedStructure`.

If the bolt-on has no quantized tiles for the requested region/params (either the
server explicitly says so via HTTP 422, or the decoded NPZ has ``n_tiles == 0``, or the
reconstructed mosaic is all-NaN), the client raises :class:`NoCoverageError` so callers
can surface "no embeddings for this region" cleanly rather than processing zero-shaped
or all-NaN arrays.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from affine import Affine

Distance = Literal["euclidean", "cosine"]


class NoCoverageError(Exception):
    """Raised when the bolt-on has no quantized tiles for the requested region/params.

    Triggered by any of: server HTTP 422 (no all-finite tile fits the reprojected
    region for the requested ``t``), an NPZ payload with ``positions.shape[0] == 0``,
    or an all-NaN reconstruction. Catch this to surface "no embeddings for this
    region" in your UI without needing the ``np.isnan(mosaic).all()`` heuristic.
    """


@dataclass
class QuantizedStructure:
    """Per-tile codebooks + index maps + positions, without reconstruction.

    ``codebooks2`` / ``indices2`` are present iff this came from the RVQ endpoint
    (two-stage residual VQ). Reconstruction is ``codebooks1[i][indices1[i]]`` for the
    single-stage case and ``codebooks1[i][indices1[i]] + codebooks2[i][indices2[i]]``
    for the RVQ case.

    ``positions`` are ``(row, col)`` indices into the bbox tile-grid (NOT UTM or world
    coordinates): pixel ``(0, 0)`` of tile ``i`` sits at ``(positions[i, 0] * tile_size,
    positions[i, 1] * tile_size)`` inside the EPSG:4326 mosaic of shape ``mosaic_shape``.
    The reprojected mosaic itself is not returned — callers reconstruct it from this
    structure if they need pixels, or pass the structure straight through to a storage
    format if they don't.
    """

    codebooks1: npt.NDArray[np.float32]
    indices1: npt.NDArray[Any]
    codebooks2: npt.NDArray[np.float32] | None
    indices2: npt.NDArray[Any] | None
    positions: npt.NDArray[np.int32]
    tile_size: int
    k1: int
    k2: int | None
    metric: Distance
    mosaic_shape: tuple[int, int]
    bbox: tuple[float, float, float, float]
    year: int

    @property
    def is_rvq(self) -> bool:
        """True if this came from the RVQ endpoint."""
        return self.k2 is not None


class VQTessera:
    """Plug-compatible subset of ``geotessera.GeoTessera`` over the VQ bolt-on."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        t: int = 64,
        k: int = 16,
        m: Distance = "euclidean",
        timeout: float = 120.0,
        *,
        k2: int | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.t = int(t)
        self.k = int(k)
        self.m: Distance = m
        self.k2: int | None = int(k2) if k2 is not None else None
        self.timeout = float(timeout)

    @property
    def is_rvq(self) -> bool:
        """True if the client is configured for two-stage Residual VQ (``k2`` is set)."""
        return self.k2 is not None

    def fetch_mosaic_for_region(
        self,
        bbox: tuple[float, float, float, float],
        year: int = 2024,
        target_crs: str = "EPSG:4326",
        auto_download: bool = True,  # noqa: ARG002  (kept for geotessera API compat)
    ) -> tuple[npt.NDArray[np.float32], Affine, str]:
        """Fetch reconstructed embeddings for ``bbox``; returns ``(mosaic, transform, crs)``.

        Single HTTP round-trip. Raises :class:`NoCoverageError` if the bolt-on has no
        quantized tiles for the requested region/params (see the module docstring).
        """
        struct = self.fetch_quantized_structure(bbox, year=year, target_crs=target_crs)
        return _reconstruct_from_structure(struct)

    def fetch_quantized_structure(
        self,
        bbox: tuple[float, float, float, float],
        year: int = 2024,
        target_crs: str = "EPSG:4326",
    ) -> QuantizedStructure:
        """Fetch the per-tile payload for ``bbox`` without reconstructing the mosaic.

        Hits ``/quantized_rvq`` if ``k2`` was set at construction, otherwise ``/quantized``.
        Raises :class:`NoCoverageError` if the bolt-on returns HTTP 422 or an NPZ with
        zero tiles. ``target_crs`` is validated for API compatibility (server only
        emits EPSG:4326); structure positions are grid indices, not world coordinates.
        """
        if target_crs.upper() not in ("EPSG:4326", "WGS84"):
            raise ValueError(f"only EPSG:4326 is supported; got {target_crs!r}")
        if self.is_rvq:
            path = "/quantized_rvq"
            payload: dict[str, Any] = {
                "bbox": list(bbox),
                "year": int(year),
                "t": self.t,
                "k1": self.k,
                "k2": self.k2,
                "m": self.m,
            }
        else:
            path = "/quantized"
            payload = {
                "bbox": list(bbox),
                "year": int(year),
                "t": self.t,
                "k": self.k,
                "m": self.m,
            }
        npz_bytes = self._post(path, payload)
        struct = _structure_from_npz(npz_bytes, bbox)
        if struct.positions.shape[0] == 0:
            raise NoCoverageError(
                f"bolt-on returned 0 tiles for bbox={bbox} year={year} "
                f"t={self.t} k1={self.k} k2={self.k2}"
            )
        return struct

    def fetch_embedding(
        self, lon: float, lat: float, year: int = 2024
    ) -> tuple[npt.NDArray[np.float32], Affine, str]:
        """Fetch the embedding mosaic for the 0.1-degree tile around ``(lon, lat)``."""
        bounds = (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
        return self.fetch_mosaic_for_region(bounds, year=year)

    def fetch_residual_histogram(
        self,
        bbox: tuple[float, float, float, float],
        year: int = 2024,
        n_bins: int = 50,
    ) -> dict[str, Any]:
        """Per-pixel L2-residual-norm histogram + summary for the bolt-on's chosen (t, k, m).

        Returns ``{n_pixels, bin_edges, counts, stats}``. Useful for plotting a "how off
        is each pixel" histogram in a UI. *Not* in geotessera; additive only.
        """
        payload: dict[str, Any] = {
            "bbox": list(bbox),
            "year": int(year),
            "t": self.t,
            "k": self.k,
            "m": self.m,
            "n_bins": int(n_bins),
        }
        if self.k2 is not None:
            payload["k2"] = self.k2
        body = self._post("/residuals", payload)
        result: dict[str, Any] = json.loads(body)
        return result

    def _post(self, path: str, payload: dict[str, Any]) -> bytes:
        """POST JSON to the bolt-on; return the raw response body.

        Translates a server ``422`` (no coverage for the requested params) into a
        :class:`NoCoverageError` with the server's diagnostic message; other 4xx/5xx
        responses propagate as ``urllib.error.HTTPError``.
        """
        req = urllib.request.Request(
            self.server_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                return bytes(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 422:  # noqa: PLR2004
                raise NoCoverageError(_decode_error_body(exc)) from exc
            raise


def _decode_error_body(exc: urllib.error.HTTPError) -> str:
    """Pull the server's ``{"error": "..."}`` message out of an HTTPError body."""
    try:
        body = json.loads(exc.read().decode())
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return f"server returned HTTP {exc.code} with no decodable body"
    msg = body.get("error") if isinstance(body, dict) else None
    return str(msg) if msg else f"server returned HTTP {exc.code}"


def _structure_from_npz(
    npz_bytes: bytes, bbox: tuple[float, float, float, float]
) -> QuantizedStructure:
    """Decode a ``/quantized`` or ``/quantized_rvq`` NPZ into a :class:`QuantizedStructure`.

    The two endpoints share most of their schema; we detect RVQ by the presence of the
    ``codebooks2`` array, then read ``meta`` accordingly (single-stage ``meta`` is
    ``[t, k, year, H, W]``; RVQ ``meta`` is ``[t, k1, k2, year, H, W]``).
    """
    with np.load(io.BytesIO(npz_bytes)) as data:
        is_rvq = "codebooks2" in data.files
        positions: npt.NDArray[np.int32] = data["positions"].astype(np.int32, copy=False)
        meta = data["meta"]
        metric = cast("Distance", str(data["distance"]))
        if is_rvq:
            cb1: npt.NDArray[np.float32] = data["codebooks1"]
            idx1: npt.NDArray[Any] = data["indices1"]
            cb2: npt.NDArray[np.float32] | None = data["codebooks2"]
            idx2: npt.NDArray[Any] | None = data["indices2"]
            t, k1, k2_val = int(meta[0]), int(meta[1]), int(meta[2])
            year = int(meta[3])
            full_h, full_w = int(meta[4]), int(meta[5])
            k2: int | None = k2_val
        else:
            cb1 = data["codebooks"]
            idx1 = data["indices"]
            cb2 = None
            idx2 = None
            t, k1 = int(meta[0]), int(meta[1])
            year = int(meta[2])
            full_h, full_w = int(meta[3]), int(meta[4])
            k2 = None
    return QuantizedStructure(
        codebooks1=cb1,
        indices1=idx1,
        codebooks2=cb2,
        indices2=idx2,
        positions=positions,
        tile_size=t,
        k1=k1,
        k2=k2,
        metric=metric,
        mosaic_shape=(full_h, full_w),
        bbox=bbox,
        year=year,
    )


def _reconstruct_from_structure(
    struct: QuantizedStructure,
) -> tuple[npt.NDArray[np.float32], Affine, str]:
    """Rebuild ``(H, W, 128)`` float32 from a :class:`QuantizedStructure` in EPSG:4326.

    Uncovered tiles stay NaN. Raises :class:`NoCoverageError` if the structure has zero
    tiles or if the reconstructed mosaic is entirely NaN (i.e. every candidate tile got
    NaN-filtered server-side).
    """
    n = int(struct.positions.shape[0])
    if n == 0:
        raise NoCoverageError(
            f"structure has 0 tiles for bbox={struct.bbox} year={struct.year} "
            f"t={struct.tile_size} k1={struct.k1} k2={struct.k2}"
        )
    full_h, full_w = struct.mosaic_shape
    t = struct.tile_size
    out_h = (full_h // t) * t
    out_w = (full_w // t) * t
    if out_h == 0 or out_w == 0:
        raise NoCoverageError(
            f"reconstructed mosaic would be 0-sized ({out_h}x{out_w}) for "
            f"bbox={struct.bbox} year={struct.year} t={t}; tile_size exceeds "
            f"reprojected region ({full_w}x{full_h} px)"
        )
    channels = int(struct.codebooks1.shape[-1])
    mosaic = np.full((out_h, out_w, channels), np.nan, dtype=np.float32)
    if struct.is_rvq:
        cb2 = cast("npt.NDArray[np.float32]", struct.codebooks2)
        idx2 = cast("npt.NDArray[Any]", struct.indices2)
        for i in range(n):
            r, c = int(struct.positions[i, 0]), int(struct.positions[i, 1])
            mosaic[r * t : (r + 1) * t, c * t : (c + 1) * t] = (
                struct.codebooks1[i][struct.indices1[i]] + cb2[i][idx2[i]]
            )
    else:
        for i in range(n):
            r, c = int(struct.positions[i, 0]), int(struct.positions[i, 1])
            mosaic[r * t : (r + 1) * t, c * t : (c + 1) * t] = struct.codebooks1[i][
                struct.indices1[i]
            ]
    if bool(np.isnan(mosaic).all()):
        raise NoCoverageError(
            f"reconstructed mosaic is entirely NaN for bbox={struct.bbox} "
            f"year={struct.year} t={t} k1={struct.k1} k2={struct.k2}"
        )
    lon0, _lat0, lon1, lat1 = struct.bbox
    dx = (lon1 - lon0) / full_w
    dy = (lat1 - struct.bbox[1]) / full_h
    return mosaic, Affine(dx, 0.0, lon0, 0.0, -dy, lat1), "EPSG:4326"


def _reconstruct(
    npz_bytes: bytes, bbox: tuple[float, float, float, float]
) -> tuple[npt.NDArray[np.float32], Affine, str]:
    """Thin wrapper kept for backwards-compatible imports: decode + reconstruct.

    Equivalent to ``_reconstruct_from_structure(_structure_from_npz(npz_bytes, bbox))``.
    """
    return _reconstruct_from_structure(_structure_from_npz(npz_bytes, bbox))
