"""Plug-compatible Python client for the Tessera VQ bolt-on.

Drop-in subset of ``geotessera.GeoTessera`` for downstream code that wants reconstructed
embeddings without holding the raw 128-d floats. Each fetch POSTs ``/quantized`` to the
bolt-on (which runs LAN-close to the embeddings store), receives a small NPZ of
codebooks + per-tile index maps, and rebuilds ``(H, W, 128)`` float32 in EPSG:4326.

Configure ``(t, k, m)`` at construction; only EPSG:4326 is supported as output.

Example::

    from tessera_vq.client import VQTessera

    gt = VQTessera(server_url="http://michael:8000")  # defaults: t=512, k=20, k2=256, L2
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

Call ``fetch_dataset_version()`` to find out which Tessera dataset release (e.g. "1.0",
"1.1") the bolt-on's embeddings come from — useful for stamping exports with provenance.
"""

from __future__ import annotations

import io
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
from affine import Affine

from tessera_vq.codebook_codec import dequantize_codebook_uint8

Distance = Literal["euclidean", "cosine"]

# Tessera's fixed ground resolution. Mirrors tessera_vq.data._PIXEL_M /
# _M_PER_DEG_LAT -- duplicated here rather than imported so this lightweight
# client module doesn't pull in data.py's heavier deps (tessera_zarr_utils,
# geotessera, joblib), which would defeat the point of a minimal plug-in
# client. See reconstruct_from_structure() for why this is needed.
_PIXEL_M = 10.0
_M_PER_DEG_LAT = 111_320.0


def _n_tiles_along(full_dim: int, t: int) -> int:
    """Mirrors ``tessera_vq.sweep.n_tiles_along`` exactly -- duplicated (not
    imported) for the same reason ``_PIXEL_M`` above is: ``sweep`` sits behind
    the ``[server]``/``[sweep]`` extras this client deliberately doesn't pull
    in. Must stay in sync with that copy -- see ``tile_pixel_offset``'s
    docstring there for why the two sides agreeing matters."""
    if full_dim < t:
        return 0
    return -(-full_dim // t)


def _tile_pixel_offset(idx: int, n: int, full_dim: int, t: int) -> int:
    """Mirrors ``tessera_vq.sweep.tile_pixel_offset`` exactly -- see
    ``_n_tiles_along`` above for why this is duplicated rather than imported."""
    if idx == n - 1:
        return full_dim - t
    return idx * t


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

    ``positions`` are ``(row, col)`` *tile-grid indices* (NOT UTM/world coordinates,
    NOT pixel offsets themselves): convert to pixel ``(0, 0)`` of tile ``i`` via
    ``tessera_vq.sweep.tile_pixel_offset(positions[i, k], n_tiles_along(H_or_W,
    tile_size), H_or_W, tile_size)`` -- fixed ``tile_size``-stride, except the last
    tile along an axis is pulled back to end exactly at that axis's true edge rather
    than a remainder strip being dropped when ``H``/``W`` isn't itself a multiple of
    ``tile_size``. The reprojected mosaic itself is not returned — callers
    reconstruct it via :func:`reconstruct_from_structure` if they need pixels
    (which already applies this offset rule), or pass the structure straight
    through to a storage format if they don't.

    ``mosaic_shape`` is the **full** reprojected EPSG:4326 mosaic shape ``(H, W)`` as
    returned by ``read_region`` -- generally larger than ``bbox`` itself, since
    ``read_region`` rounds fetches out to whole tiles. The reconstructed output is
    exactly ``(H, W, 128)`` -- every real pixel, not truncated to a tile_size
    multiple. Pixel size is Tessera's fixed ~10m ground resolution (NOT
    ``bbox_span / W`` -- that undercounts whenever tile-rounding expanded the fetch
    beyond ``bbox``; see :func:`reconstruct_from_structure`). Use
    :func:`reconstruct_from_structure` if you want pixels + transform without
    re-deriving the math yourself.

    ``origin`` is ``(origin_lon, origin_lat, dx, dy)`` for pixel ``(0, 0)`` of the
    *full* (pre-truncation) mosaic, taken from the real transform ``read_region``
    computed (via geotessera's ``rasterio.merge``, or the zarr path's reprojection) --
    not derived from ``bbox``. ``None`` only when talking to a bolt-on server that
    predates this field (pre-0.5.7), in which case :func:`reconstruct_from_structure`
    falls back to assuming the mosaic starts at ``bbox``'s own corner, which is a
    known-inaccurate approximation (see that function's docstring).
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
    origin: tuple[float, float, float, float] | None = None

    @property
    def is_rvq(self) -> bool:
        """True if this came from the RVQ endpoint."""
        return self.k2 is not None


class VQTessera:
    """Plug-compatible subset of ``geotessera.GeoTessera`` over the VQ bolt-on."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        t: int = 512,
        k: int = 20,
        m: Distance = "euclidean",
        timeout: float = 120.0,
        *,
        k2: int | None = 256,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.t = int(t)
        self.k = int(k)
        self.m: Distance = m
        self.k2: int | None = int(k2) if k2 is not None else None
        self.timeout = float(timeout)
        self._dataset_version: str | None = None

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
        return reconstruct_from_structure(struct)

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

    def fetch_dataset_version(self) -> str:
        """The Tessera dataset version (e.g. "1.0", "1.1") the bolt-on server reads from.

        One GET to ``/version`` per client instance; cached after the first call.
        Servers predating this endpoint (HTTP 404) resolve to ``"unknown"`` rather
        than raising, since this is metadata, not a hard dependency.
        """
        if self._dataset_version is None:
            try:
                body = self._get("/version")
                self._dataset_version = str(json.loads(body)["dataset_version"])
            except urllib.error.HTTPError:
                self._dataset_version = "unknown"
        return self._dataset_version

    def _get(self, path: str) -> bytes:
        """GET from the bolt-on; return the raw response body."""
        req = urllib.request.Request(self.server_url + path, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            return bytes(resp.read())

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
        is_rvq = "codebooks1_q" in data.files
        positions: npt.NDArray[np.int32] = data["positions"].astype(np.int32, copy=False)
        meta = data["meta"]
        metric = cast("Distance", str(data["distance"]))
        if is_rvq:
            t, k1, k2_val = int(meta[0]), int(meta[1]), int(meta[2])
            year = int(meta[3])
            full_h, full_w = int(meta[4]), int(meta[5])
            k2: int | None = k2_val
            # codebooks arrive per-dim uint8 (q + lo/hi); idx1/idx2 are raw uint8 planes
            # inside a DEFLATE-compressed NPZ (np.savez_compressed).
            cb1: npt.NDArray[np.float32] = dequantize_codebook_uint8(
                data["codebooks1_q"], data["codebooks1_lo"], data["codebooks1_hi"]
            )
            idx1: npt.NDArray[Any] = data["indices1"]
            cb2: npt.NDArray[np.float32] | None = dequantize_codebook_uint8(
                data["codebooks2_q"], data["codebooks2_lo"], data["codebooks2_hi"]
            )
            idx2: npt.NDArray[Any] | None = data["indices2"]
        else:
            cb1 = data["codebooks"]
            idx1 = data["indices"]
            cb2 = None
            idx2 = None
            t, k1 = int(meta[0]), int(meta[1])
            year = int(meta[2])
            full_h, full_w = int(meta[3]), int(meta[4])
            k2 = None
        origin = tuple(data["origin"].tolist()) if "origin" in data.files else None
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
        origin=cast("tuple[float, float, float, float] | None", origin),
    )


def reconstruct_from_structure(
    struct: QuantizedStructure,
) -> tuple[npt.NDArray[np.float32], Affine, str]:
    """Rebuild ``(H, W, 128)`` float32 + affine + crs from a :class:`QuantizedStructure`.

    Returned shape is ``struct.mosaic_shape`` (``full_h, full_w``) exactly --
    every real pixel the bolt-on fetched, not ``(full_h // t) * t`` truncated
    down to a whole multiple of the tile size. Tiles are placed via
    :func:`_tile_pixel_offset`: fixed ``t``-stride, except the last tile along
    each axis (if ``full_h``/``full_w`` isn't itself a multiple of ``t``) is
    pulled back to end exactly at the true edge instead of that remainder
    strip being silently dropped -- servers before this scheme (see
    ``tessera_vq.sweep`` for the server-side half) discarded up to ``t - 1``
    real, valid pixels per axis this way, which mattered a lot for callers
    requesting a small crop out of a much-larger-than-t mosaic (a whole tile
    can be a large fraction of a small target). Uncovered pixels (either
    genuinely outside the bolt-on's fetch, or -- talking to an older server --
    the pre-this-fix remainder strip) stay NaN. CRS is ``"EPSG:4326"``.

    The affine transform's anchor comes from ``struct.origin`` when the bolt-on
    provided one (servers >=0.5.7): the *real* transform ``read_region`` computed
    for the returned mosaic, not a guess. Before this existed, the anchor was
    fabricated by assuming pixel ``(0, 0)`` sits at ``bbox``'s own top-left corner
    -- which is frequently wrong, sometimes by several km, because geotessera's
    ``fetch_mosaic_for_region`` returns the union of whichever source tiles
    overlap ``bbox``, not a ``bbox``-exact crop (confirmed empirically: for one
    real bbox the true origin differed from the fabricated one by ~3.8km
    longitude and ~7.8km latitude). That fallback still applies -- with the same
    caveat -- for structures from an older server that predates ``origin``.
    Either way, pixel size uses Tessera's fixed ~10m ground resolution, NOT
    ``bbox_span / full_dim`` (which understates true ground distance per pixel
    whenever tile-rounding expanded the fetch beyond ``bbox`` -- the common
    case, not an edge case).

    Raises :class:`NoCoverageError` if the structure has zero tiles, if not even one
    tile fits (``full_h`` or ``full_w`` < ``t``), or if the reconstructed mosaic is
    entirely NaN. Callers should prefer this helper over re-implementing the math so
    they stay aligned with ``fetch_mosaic_for_region``.
    """
    n = int(struct.positions.shape[0])
    if n == 0:
        raise NoCoverageError(
            f"structure has 0 tiles for bbox={struct.bbox} year={struct.year} "
            f"t={struct.tile_size} k1={struct.k1} k2={struct.k2}"
        )
    full_h, full_w = struct.mosaic_shape
    t = struct.tile_size
    rows, cols = _n_tiles_along(full_h, t), _n_tiles_along(full_w, t)
    if rows == 0 or cols == 0:
        raise NoCoverageError(
            f"not even one {t}x{t} tile fits for bbox={struct.bbox} year={struct.year} "
            f"t={t}; tile_size exceeds reprojected region ({full_w}x{full_h} px)"
        )
    channels = int(struct.codebooks1.shape[-1])
    mosaic = np.full((full_h, full_w, channels), np.nan, dtype=np.float32)
    if struct.is_rvq:
        cb2 = cast("npt.NDArray[np.float32]", struct.codebooks2)
        idx2 = cast("npt.NDArray[Any]", struct.indices2)
        for i in range(n):
            r, c = int(struct.positions[i, 0]), int(struct.positions[i, 1])
            row_off = _tile_pixel_offset(r, rows, full_h, t)
            col_off = _tile_pixel_offset(c, cols, full_w, t)
            mosaic[row_off : row_off + t, col_off : col_off + t] = (
                struct.codebooks1[i][struct.indices1[i]] + cb2[i][idx2[i]]
            )
    else:
        for i in range(n):
            r, c = int(struct.positions[i, 0]), int(struct.positions[i, 1])
            row_off = _tile_pixel_offset(r, rows, full_h, t)
            col_off = _tile_pixel_offset(c, cols, full_w, t)
            mosaic[row_off : row_off + t, col_off : col_off + t] = struct.codebooks1[i][
                struct.indices1[i]
            ]
    if bool(np.isnan(mosaic).all()):
        raise NoCoverageError(
            f"reconstructed mosaic is entirely NaN for bbox={struct.bbox} "
            f"year={struct.year} t={t} k1={struct.k1} k2={struct.k2}"
        )
    if struct.origin is not None:
        # Real transform propagated from read_region() -- accurate anchor, not a guess.
        origin_lon, origin_lat, dx, dy = struct.origin
        return mosaic, Affine(dx, 0.0, origin_lon, 0.0, dy, origin_lat), "EPSG:4326"

    # Fallback for structures from a pre-origin server: fabricate an anchor by
    # assuming pixel (0, 0) sits at bbox's own top-left corner. Known-inaccurate
    # (see this function's docstring) but scale is still correct, which was the
    # reproducible, provable part of the old error -- pixel size can't be derived
    # by dividing the *requested* bbox span by the returned pixel count, since
    # read_region() rounds fetches out to whole tiles larger than bbox (the
    # common case, not an edge case), which understates true ground distance
    # per pixel if you divide the small request by the large pixel count
    # (confirmed empirically: implied ~2.5m/pixel vs the real fixed ~10m/pixel).
    lon0, _lat0, _lon1, lat1 = struct.bbox
    dy = _PIXEL_M / _M_PER_DEG_LAT
    dx = _PIXEL_M / (_M_PER_DEG_LAT * math.cos(math.radians((struct.bbox[1] + struct.bbox[3]) / 2)))
    return mosaic, Affine(dx, 0.0, lon0, 0.0, -dy, lat1), "EPSG:4326"


def _reconstruct(
    npz_bytes: bytes, bbox: tuple[float, float, float, float]
) -> tuple[npt.NDArray[np.float32], Affine, str]:
    """Thin wrapper kept for backwards-compatible imports: decode + reconstruct.

    Equivalent to ``reconstruct_from_structure(_structure_from_npz(npz_bytes, bbox))``.
    """
    return reconstruct_from_structure(_structure_from_npz(npz_bytes, bbox))
