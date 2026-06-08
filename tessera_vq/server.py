"""Flask bolt-on for serving Tessera VQ-quantised embeddings.

The server is intentionally narrow: given a chosen ``(t, k, m)`` and a bbox it runs
one k-means per tile and returns an NPZ. CPU per request is bounded (one k-means at
the sample size + a vectorised assign), and a per-side bbox cap (default 10 km)
prevents over-large requests. Designed to run LAN-close to the geotessera embeddings
store.

The expensive *exploration* sweep over many ``(t, k, m)`` combinations is **not**
served from here — call ``tessera_vq.sweep.sweep_window`` directly as a library on
embeddings you've fetched locally via geotessera. That keeps exploration CPU on the
caller, not on the public server.

Endpoints:

- ``GET  /health``       liveness probe.
- ``POST /quantized``    body ``{bbox, t, k, m?, year?, sample_size?, seed?}`` ->
                          NPZ of codebooks + index maps + tile positions.

Run with::

    uv run python -m tessera_vq.server  # uses waitress, port 8000
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, cast

import numpy as np
from flask import Flask, Response, jsonify, request

from tessera_vq.codebook_codec import quantize_codebook_uint8
from tessera_vq.data import read_region
from tessera_vq.entropy import rle_encode_stack
from tessera_vq.sweep import (
    Distance,
    quantize_window_for_serving,
    quantize_window_residual_norms,
    quantize_window_residual_norms_rvq,
    rvq_quantize_window_for_serving,
)
from tessera_vq.tile_cache import TileCache

logger = logging.getLogger(__name__)

# Max bbox side in km (per side). Default 10 km -> ~1e6 px -> ~500 MB float32 mosaic.
# Override on the server with: export TESSERA_VQ_MAX_BBOX_KM=20
_MAX_BBOX_KM = float(os.environ.get("TESSERA_VQ_MAX_BBOX_KM", "10.0"))
_KM_PER_DEG_LAT = 111.32

# Durable RVQ response cache (WS-2). Off unless TESSERA_VQ_CACHE_DIR is set, so dev/tests
# never write a cache; michael enables it via env. Default cap 500 GB (~287k tiles).
_WIRE_FORMAT = "rvq-int8-rle-1"  # bump if the /quantized_rvq NPZ schema changes
_CACHE_DIR = os.environ.get("TESSERA_VQ_CACHE_DIR")
_CACHE_MAX_GB = float(os.environ.get("TESSERA_VQ_CACHE_MAX_GB", "500"))
_CACHE: TileCache | None = TileCache(_CACHE_DIR, int(_CACHE_MAX_GB * 1e9)) if _CACHE_DIR else None


class _NoDataError(Exception):
    """No embeddings for the requested bbox (-> 404); not cached."""


class _NoTilesError(Exception):
    """No all-finite tiles fit the bbox at this t (-> 422); not cached."""


def _rvq_cache_key(
    bbox: tuple[float, ...], year: int, t: int, k1: int, k2: int, m: str, ssz: int, seed: int
) -> str:
    """Canonical cache key; bbox rounded to ~0.1 m to absorb float jitter."""
    b = ",".join(f"{v:.6f}" for v in bbox)
    return f"{_WIRE_FORMAT}|{b}|{year}|{t}|{k1}|{k2}|{m}|{ssz}|{seed}"


app = Flask("tessera_vq")


@app.get("/health")  # type: ignore
def health() -> Response:
    return jsonify({"ok": True})


@app.post("/quantized")  # type: ignore
def quantized() -> Response:  # noqa: PLR0911
    """Return per-tile codebooks + index maps as NPZ for the chosen (t, k, m).

    Body: ``{bbox, t, k, m?, year?, sample_size?, seed?}``. Response body is an NPZ with
    ``codebooks (n_tiles, k_eff, 128) float32``, ``indices (n_tiles, t, t) uint8/16``,
    ``positions (n_tiles, 2) int32``, ``meta`` and ``distance`` arrays.
    """
    body: dict[str, Any] = cast("dict[str, Any]", request.get_json(force=True))
    bbox = tuple(float(v) for v in body["bbox"])
    if len(bbox) != 4:  # noqa: PLR2004
        return _bad_request("bbox must be [lon0, lat0, lon1, lat1]")
    too_big = _check_bbox_size(bbox)
    if too_big:
        return _bad_request(too_big, code=413)
    if "t" not in body or "k" not in body:
        return _bad_request("missing required 't' and/or 'k'")
    t = int(body["t"])
    k = int(body["k"])
    if t <= 0 or k <= 0:
        return _bad_request("'t' and 'k' must be positive")
    m: Distance = cast("Distance", body.get("m", "euclidean"))
    year = int(body.get("year", 2024))
    sample_size = int(body.get("sample_size", 2000))
    seed = int(body.get("seed", 42))
    mosaic, path = read_region(bbox, year)
    if mosaic is None:
        return _bad_request("no embeddings available for bbox", code=404)
    codebooks, indices, positions = quantize_window_for_serving(
        mosaic, t, k, m, seed, sample_size=sample_size
    )
    logger.info(
        "quantized bbox=%s year=%d t=%d k=%d m=%s path=%s n_tiles=%d",
        bbox,
        year,
        t,
        k,
        m,
        path,
        positions.shape[0],
    )
    if positions.shape[0] == 0:
        return _bad_request(_no_tiles_message(mosaic.shape, t), code=422)
    buf = io.BytesIO()
    np.savez(
        buf,
        codebooks=codebooks,
        indices=indices,
        positions=positions,
        meta=np.asarray([t, k, year, mosaic.shape[0], mosaic.shape[1]], dtype=np.int32),
        distance=np.asarray(m),
    )
    return Response(buf.getvalue(), mimetype="application/octet-stream")


@app.post("/quantized_rvq")  # type: ignore
def quantized_rvq() -> Response:  # noqa: PLR0911
    """Return per-tile RVQ codebooks + index maps as NPZ for ``(t, k1, k2)``.

    Body: ``{bbox, t, k1, k2, m?, year?, sample_size?, seed?}``. Response NPZ:
    ``codebooks1 (n_tiles, k1_eff, 128) float32``,
    ``indices1   (n_tiles, t, t) uint8/16``,
    ``codebooks2 (n_tiles, k2_eff, 128) float32``,
    ``indices2   (n_tiles, t, t) uint8/16``,
    ``positions  (n_tiles, 2) int32``, plus small ``meta``/``distance`` arrays.
    Reconstruct each tile as ``codebooks1[i][indices1[i]] + codebooks2[i][indices2[i]]``.
    """
    body: dict[str, Any] = cast("dict[str, Any]", request.get_json(force=True))
    bbox = tuple(float(v) for v in body["bbox"])
    if len(bbox) != 4:  # noqa: PLR2004
        return _bad_request("bbox must be [lon0, lat0, lon1, lat1]")
    too_big = _check_bbox_size(bbox)
    if too_big:
        return _bad_request(too_big, code=413)
    if "t" not in body or "k1" not in body or "k2" not in body:
        return _bad_request("missing required 't', 'k1', and/or 'k2'")
    t = int(body["t"])
    k1 = int(body["k1"])
    k2 = int(body["k2"])
    if t <= 0 or k1 <= 0 or k2 <= 0:
        return _bad_request("'t', 'k1', and 'k2' must be positive")
    m: Distance = cast("Distance", body.get("m", "euclidean"))
    year = int(body.get("year", 2024))
    sample_size = int(body.get("sample_size", 2000))
    seed = int(body.get("seed", 42))

    def _compute() -> bytes:
        """Read + quantize + pack the NPZ. Raises _NoDataError/_NoTilesError (never cached)."""
        mosaic, path = read_region(bbox, year)
        if mosaic is None:
            raise _NoDataError
        cbs1, idxs1, cbs2, idxs2, positions = rvq_quantize_window_for_serving(
            mosaic, t, k1, k2, m, seed, sample_size=sample_size
        )
        logger.info(
            "quantized_rvq bbox=%s year=%d t=%d k1=%d k2=%d m=%s path=%s n_tiles=%d",
            bbox,
            year,
            t,
            k1,
            k2,
            m,
            path,
            positions.shape[0],
        )
        if positions.shape[0] == 0:
            raise _NoTilesError(_no_tiles_message(mosaic.shape, t))
        # idx1 is spatially smooth -> row-major RLE shrinks the wire payload; idx2 is the
        # white residual and stays raw. Codebooks ship as per-dim uint8 (q + lo/hi).
        idx1_values, idx1_lengths, idx1_runs = rle_encode_stack(idxs1)
        cb1_q, cb1_lo, cb1_hi = quantize_codebook_uint8(cbs1)
        cb2_q, cb2_lo, cb2_hi = quantize_codebook_uint8(cbs2)
        buf = io.BytesIO()
        np.savez(
            buf,
            codebooks1_q=cb1_q,
            codebooks1_lo=cb1_lo,
            codebooks1_hi=cb1_hi,
            idx1_values=idx1_values,
            idx1_lengths=idx1_lengths.astype(np.uint32),
            idx1_runs=idx1_runs.astype(np.int32),
            codebooks2_q=cb2_q,
            codebooks2_lo=cb2_lo,
            codebooks2_hi=cb2_hi,
            indices2=idxs2,
            positions=positions,
            meta=np.asarray([t, k1, k2, year, mosaic.shape[0], mosaic.shape[1]], dtype=np.int32),
            distance=np.asarray(m),
        )
        return buf.getvalue()

    key = _rvq_cache_key(bbox, year, t, k1, k2, m, sample_size, seed)
    try:
        data = _CACHE.get_or_compute(key, _compute) if _CACHE else _compute()
    except _NoDataError:
        return _bad_request("no embeddings available for bbox", code=404)
    except _NoTilesError as exc:
        return _bad_request(str(exc), code=422)
    return Response(data, mimetype="application/octet-stream")


@app.post("/residuals")  # type: ignore
def residuals() -> Response:  # noqa: PLR0911, PLR0912
    """Return a per-pixel L2-residual-norm histogram + summary.

    Body: ``{bbox, t, k, k2?, m?, year?, n_bins?, sample_size?, seed?}``. If ``k2`` is
    omitted the residual is for single-level VQ ``c[idx]``; if ``k2`` is given the
    residual is for two-stage RVQ ``c1[idx1] + c2[idx2]`` (euclidean only). Response:
    ``{n_pixels, bin_edges[n+1], counts[n], stats{mean, p10, p50, p90, p99}}``.
    """
    body: dict[str, Any] = cast("dict[str, Any]", request.get_json(force=True))
    bbox = tuple(float(v) for v in body["bbox"])
    if len(bbox) != 4:  # noqa: PLR2004
        return _bad_request("bbox must be [lon0, lat0, lon1, lat1]")
    too_big = _check_bbox_size(bbox)
    if too_big:
        return _bad_request(too_big, code=413)
    if "t" not in body or "k" not in body:
        return _bad_request("missing required 't' and/or 'k'")
    t = int(body["t"])
    k = int(body["k"])
    k2_raw = body.get("k2")
    k2 = int(k2_raw) if k2_raw is not None else None
    if t <= 0 or k <= 0 or (k2 is not None and k2 <= 0):
        return _bad_request("'t', 'k' (and 'k2' when given) must be positive")
    m: Distance = cast("Distance", body.get("m", "euclidean"))
    year = int(body.get("year", 2024))
    n_bins = max(2, int(body.get("n_bins", 50)))
    sample_size = int(body.get("sample_size", 2000))
    seed = int(body.get("seed", 42))
    mosaic, path = read_region(bbox, year)
    if mosaic is None:
        return _bad_request("no embeddings available for bbox", code=404)
    if k2 is None:
        norms = quantize_window_residual_norms(mosaic, t, k, m, seed, sample_size=sample_size)
    else:
        try:
            norms = quantize_window_residual_norms_rvq(
                mosaic, t, k, k2, m, seed, sample_size=sample_size
            )
        except NotImplementedError as exc:
            return _bad_request(str(exc))
    logger.info(
        "residuals bbox=%s year=%d t=%d k=%d k2=%s m=%s path=%s n_pixels=%d n_bins=%d",
        bbox,
        year,
        t,
        k,
        k2,
        m,
        path,
        norms.size,
        n_bins,
    )
    if norms.size == 0:
        return jsonify({"n_pixels": 0, "bin_edges": [], "counts": [], "stats": {}})
    counts, edges = np.histogram(norms, bins=n_bins)
    return jsonify(
        {
            "n_pixels": int(norms.size),
            "bin_edges": edges.tolist(),
            "counts": counts.tolist(),
            "stats": {
                "mean": float(norms.mean()),
                "p10": float(np.quantile(norms, 0.1)),
                "p50": float(np.quantile(norms, 0.5)),
                "p90": float(np.quantile(norms, 0.9)),
                "p99": float(np.quantile(norms, 0.99)),
            },
        }
    )


def _bad_request(message: str, *, code: int = 400) -> Response:
    """Return a small JSON error response."""
    response = jsonify({"error": message})
    response.status_code = code
    return response


def _no_tiles_message(mosaic_shape: tuple[int, ...], t: int) -> str:
    """Diagnostic for the 422 returned when no all-finite t x t tile fits the region."""
    h, w = int(mosaic_shape[0]), int(mosaic_shape[1])
    return (
        f"no all-finite t={t} tile fits the reprojected region ({w}x{h} px); "
        "either t is larger than the region, or every candidate tile contains NaN "
        "(e.g. from reprojection edges). Try a smaller t or a larger bbox."
    )


def _bbox_size_km(bbox: tuple[float, ...]) -> tuple[float, float]:
    """Approximate ``(width_km, height_km)`` for a lon/lat bbox at its mid-latitude."""
    lon0, lat0, lon1, lat1 = bbox
    mid_lat = (lat0 + lat1) / 2.0
    width_km = abs(lon1 - lon0) * _KM_PER_DEG_LAT * float(np.cos(np.radians(mid_lat)))
    height_km = abs(lat1 - lat0) * _KM_PER_DEG_LAT
    return width_km, height_km


def _check_bbox_size(bbox: tuple[float, ...]) -> str | None:
    """Return an error message if ``bbox`` exceeds ``_MAX_BBOX_KM`` per side, else None."""
    width_km, height_km = _bbox_size_km(bbox)
    if width_km > _MAX_BBOX_KM or height_km > _MAX_BBOX_KM:
        return (
            f"bbox too large ({width_km:.1f} km x {height_km:.1f} km); "
            f"max {_MAX_BBOX_KM:.1f} km per side (set TESSERA_VQ_MAX_BBOX_KM to raise)"
        )
    return None


def main() -> None:
    """Serve the app via waitress."""
    from waitress import serve  # type: ignore  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve(app, host="0.0.0.0", port=8000)  # noqa: S104


if __name__ == "__main__":
    main()
