"""Flask bolt-on for interactive (t, K, m) exploration of Tessera VQ.

Designed to run LAN-close to the geotessera embeddings store (no client cache).
Endpoints:

- ``GET  /health``       liveness probe.
- ``POST /sweep``        body ``{bbox: [lon0,lat0,lon1,lat1], year?, ts?, ks?, ms?,
                          sample_size?, seed?}``  ->  per-(t, k, m, subtile) reconstruction
                          quantiles for the user's bbox. Fetches embeddings on demand.
- ``POST /quantized``    once (t, k, m) is chosen, returns an NPZ of codebooks + index
                          maps + tile positions per the chosen ``(t, k, m)``.

Run with::

    uv run python -m tessera_vq.server  # uses waitress, port 8000
"""

from __future__ import annotations

import io
import logging
from typing import Any, cast

import numpy as np
from flask import Flask, Response, jsonify, request

from tessera_vq.data import read_region
from tessera_vq.sweep import Distance, quantize_window_for_serving, sweep_window

logger = logging.getLogger(__name__)

_DEFAULT_TS = [16, 64, 256, 1024]
_DEFAULT_KS = [4, 16, 64, 256]
_DEFAULT_MS: list[Distance] = ["euclidean", "cosine"]

app = Flask("tessera_vq")


@app.get("/health")  # type: ignore
def health() -> Response:
    return jsonify({"ok": True})


@app.post("/sweep")  # type: ignore
def sweep() -> Response:
    """Run a (t, K, m) sweep on the embeddings for the requested bbox."""
    body: dict[str, Any] = cast("dict[str, Any]", request.get_json(force=True))
    bbox = tuple(float(v) for v in body["bbox"])
    if len(bbox) != 4:  # noqa: PLR2004
        return _bad_request("bbox must be [lon0, lat0, lon1, lat1]")
    year = int(body.get("year", 2024))
    ts: list[int] = [int(t) for t in body.get("ts", _DEFAULT_TS)]
    ks: list[int] = [int(k) for k in body.get("ks", _DEFAULT_KS)]
    ms: list[Distance] = [cast("Distance", m) for m in body.get("ms", _DEFAULT_MS)]
    sample_size = int(body.get("sample_size", 2000))
    seed = int(body.get("seed", 42))
    mosaic, path = read_region(bbox, year)
    if mosaic is None:
        return _bad_request("no embeddings available for bbox", code=404)
    logger.info(
        "sweep bbox=%s year=%d shape=%s path=%s ts=%s ks=%s ms=%s",
        bbox,
        year,
        mosaic.shape,
        path,
        ts,
        ks,
        ms,
    )
    rows = sweep_window(mosaic, ts, ks, ms, seed=seed, sample_size=sample_size)
    return jsonify({"bbox": list(bbox), "year": year, "shape": list(mosaic.shape), "rows": rows})


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


def _bad_request(message: str, *, code: int = 400) -> Response:
    """Return a small JSON error response."""
    response = jsonify({"error": message})
    response.status_code = code
    return response


def main() -> None:
    """Serve the app via waitress."""
    from waitress import serve  # type: ignore  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve(app, host="0.0.0.0", port=8000)  # noqa: S104


if __name__ == "__main__":
    main()
