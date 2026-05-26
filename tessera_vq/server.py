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

from tessera_vq.data import read_region
from tessera_vq.sweep import Distance, quantize_window_for_serving, quantize_window_residual_norms

logger = logging.getLogger(__name__)

# Max bbox side in km (per side). Default 10 km -> ~1e6 px -> ~500 MB float32 mosaic.
# Override on the server with: export TESSERA_VQ_MAX_BBOX_KM=20
_MAX_BBOX_KM = float(os.environ.get("TESSERA_VQ_MAX_BBOX_KM", "10.0"))
_KM_PER_DEG_LAT = 111.32

app = Flask("tessera_vq")


@app.get("/health")  # type: ignore
def health() -> Response:
    return jsonify({"ok": True})


@app.post("/quantized")  # type: ignore
def quantized() -> Response:
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


@app.post("/residuals")  # type: ignore
def residuals() -> Response:  # noqa: PLR0911
    """Return a per-pixel L2-residual-norm histogram + summary for a chosen (t, k, m).

    Body: ``{bbox, t, k, m?, year?, n_bins?, sample_size?, seed?}``. Response JSON:
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
    if t <= 0 or k <= 0:
        return _bad_request("'t' and 'k' must be positive")
    m: Distance = cast("Distance", body.get("m", "euclidean"))
    year = int(body.get("year", 2024))
    n_bins = max(2, int(body.get("n_bins", 50)))
    sample_size = int(body.get("sample_size", 2000))
    seed = int(body.get("seed", 42))
    mosaic, path = read_region(bbox, year)
    if mosaic is None:
        return _bad_request("no embeddings available for bbox", code=404)
    norms = quantize_window_residual_norms(mosaic, t, k, m, seed, sample_size=sample_size)
    logger.info(
        "residuals bbox=%s year=%d t=%d k=%d m=%s path=%s n_pixels=%d n_bins=%d",
        bbox,
        year,
        t,
        k,
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
