# tessera-vq

Per-tile **vector quantisation (VQ)** for [Tessera](https://geotessera.org)
embeddings. Tessera produces 128-d float32 embeddings per ~10 m × 10 m pixel of
Earth's surface; within a tile only a handful of land-cover prototypes typically
appear, so a small per-tile codebook + index map can compress 99%+ of the bytes
with limited downstream accuracy loss.

This repository provides:

1. A **library** for running per-tile k-means VQ on Tessera embeddings you've
   fetched yourself (the primary, public use-case — sweeps run on *your* CPU).
2. An **optional plug-compatible client** (`VQTessera`) that mirrors
   `geotessera.GeoTessera` for code that already speaks that API.
3. An **optional small Flask server** for serving the quantised representation
   (`POST /quantized`) — bbox-capped at 10 km/side, no exploration sweeps.

## Install

```bash
pip install "tessera-vq @ git+https://github.com/sk818/tessera-vq.git@v0.2.0"
# or, for the server too:
pip install "tessera-vq[server] @ git+https://github.com/sk818/tessera-vq.git@v0.2.0"
```

Requires Python ≥ 3.12.

## Quick start — library, on your own embeddings

```python
from geotessera import GeoTessera
from tessera_vq.sweep import sweep_window, quantize_window_for_serving

gt = GeoTessera()
mosaic, transform, crs = gt.fetch_mosaic_for_region(
    (0.145, 52.045, 0.155, 52.055), year=2024
)

# 1) Explore the rate–distortion frontier on your bbox
rows = sweep_window(
    mosaic,
    ts=[16, 64, 256],
    ks=[4, 16, 64, 256],
    ms=["euclidean", "cosine"],
)
# rows: one per (t, k, m, subtile) with cosine/L2 reconstruction quantiles

# 2) Produce the compressed representation for the chosen (t, k, m)
codebooks, indices, positions = quantize_window_for_serving(
    mosaic, t=64, k=16, m="cosine"
)
# codebooks: (n_tiles, k_eff, 128) float32
# indices:   (n_tiles, t, t) uint8/16
# positions: (n_tiles, 2) int32  -- tile (row, col) in the bbox grid

# 3) Or two-stage Residual VQ for lower reconstruction error
#    (euclidean only; trades extra log2(k2) bits/pixel for lower distortion)
from tessera_vq.sweep import rvq_quantize_window_for_serving, rvq_reconstruct_tile

cbs1, idx1, cbs2, idx2, positions = rvq_quantize_window_for_serving(
    mosaic, t=64, k1=256, k2=256, m="euclidean"
)
# reconstruct any tile i as cbs1[i][idx1[i]] + cbs2[i][idx2[i]]
```

K-means is a vectorised NumPy Lloyd implementation (sample-fit then
memory-bounded full-tile assign), no native dependencies. Cosine distance is
implemented as euclidean k-means on L2-normalised inputs.

For RVQ, stage 2 is just `fast_quantize_tile` run on the residual — if you want
to sweep `k2` over a cached residual without redoing stage 1, compute the
residual once and call `fast_quantize_tile(residual, k2, ...)` directly.

## Plug-compatible client (drop-in for `GeoTessera`)

If you host the optional Flask server on a machine LAN-close to your embeddings
store, point `VQTessera` at it:

```python
from tessera_vq.client import VQTessera   # drop-in for geotessera.GeoTessera

# Single-level VQ
gt = VQTessera("http://your-host:8000", t=64, k=16, m="cosine")
mosaic, transform, crs = gt.fetch_mosaic_for_region(bbox, year=2024)

# Two-stage RVQ — pass k2 (euclidean only)
gt = VQTessera("http://your-host:8000", t=64, k=256, m="euclidean", k2=256)
mosaic, transform, crs = gt.fetch_mosaic_for_region(bbox, year=2024)
```

Same `(mosaic, transform, crs)` return shape as `GeoTessera.fetch_mosaic_for_region`,
so existing pipelines that consume `GeoTessera` need a one-line swap. The client
also exposes a histogram diagnostic:

```python
hist = gt.fetch_residual_histogram(bbox, year=2024, n_bins=50)
# {n_pixels, bin_edges, counts, stats{mean, p10, p50, p90, p99}}
# If the client was constructed with k2 set, the histogram reflects the
# two-stage RVQ residual; otherwise the single-level VQ residual.
```

## Optional self-hosted server

```bash
pip install -e ".[server]"
python -m tessera_vq.server   # Flask + waitress on 0.0.0.0:8000
```

Endpoints:

- `GET  /health` — liveness probe.
- `POST /quantized` — body `{bbox, t, k, m?, year?, sample_size?, seed?}`,
  returns an NPZ of `codebooks`, `indices`, `positions`, `meta`, `distance`.
- `POST /residuals` — body `{bbox, t, k, k2?, m?, year?, n_bins?, sample_size?, seed?}`,
  returns JSON `{n_pixels, bin_edges, counts, stats{mean, p10, p50, p90, p99}}` —
  per-pixel L2-residual-norm histogram + summary, for plotting "how off is each
  pixel" in a UI. If `k2` is omitted, the residual is for single-level VQ; if `k2`
  is given, it's the residual after two-stage RVQ reconstruction
  (`x − (c1[idx1] + c2[idx2])`, euclidean only).
- `POST /quantized_rvq` — body `{bbox, t, k1, k2, m?, year?, sample_size?, seed?}`
  for **two-stage Residual VQ** (euclidean only). Returns an NPZ with two codebooks
  + two index maps per tile; reconstruction is
  `codebooks1[i][indices1[i]] + codebooks2[i][indices2[i]]`. Lower reconstruction
  error at the cost of `log₂(k₁·k₂)` bits per pixel. Pass `k2` to `VQTessera` to
  use this from the client.

The expensive exploration sweep is **not** exposed — call `sweep_window` as a
library function on a locally-fetched mosaic instead. Both endpoints reject
bboxes larger than `TESSERA_VQ_MAX_BBOX_KM` per side (default 10 km) with
HTTP 413.

The server ships no authentication of its own — put it behind a reverse proxy
with auth (Apache / nginx) before exposing publicly. See
[`docs/integration.md`](docs/integration.md) for the parent-service integration
pattern.

## Development

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest
uv run ruff check .
uv run mypy tessera_vq scripts
```

## License

MIT — see [`LICENSE`](LICENSE).

## Layout

- `tessera_vq/` — library (`data`, `quantize`, `sweep`, `metrics`, `io_utils`,
  `client`, `server`, vendored `zarr_utils`).
- `scripts/` — analytical-phase entry points (isotropy, reconstruction sweep).
- `tests/` — unit tests on synthetic fixtures.
- `docs/spec.md` — design / phase plan.
- `docs/integration.md` — how to integrate with a parent service.
