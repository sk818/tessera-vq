# tessera-vq

Per-tile **vector quantisation (VQ) for Tessera embedding compression**. Tessera is a
self-supervised foundation model producing 128-dimensional embeddings per ~10 m × 10 m pixel of
Earth's surface. Within a tile only a handful of land-cover prototypes typically appear, so a small
per-tile codebook plus an index map can compress 99%+ of the bytes with limited downstream accuracy
loss. This repository implements and evaluates that idea against the Robinson & Corley compression
frontier, and produces (a) a tech note and (b) an engineering recommendation for GeoTessera. The work
is supervised by S. Keshav as part of the CAC project.

## Where things live

- [`CLAUDE.md`](CLAUDE.md) — stable project conventions (code style, git workflow, determinism, things never to do).
- [`config.yaml`](config.yaml) — all paths, seeds, and the parameter grid. Scripts read from here; nothing is hard-coded.
- [`docs/spec.md`](docs/spec.md) — the phase-by-phase execution plan, with mandatory HALT points between phases.

## Layout

- `tessera_vq/` — library code (loaders, quantisation, Morton/Hilbert ordering, entropy coding, metrics, probes, IO).
- `scripts/` — one entry point per analytical phase.
- `notebooks/` — plotting only.
- `tests/` — unit tests on synthetic fixtures (no real Tessera data required).
- `results/`, `figures/`, `logs/` — outputs (git-ignored).

## Development

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest
uv run ruff check .
uv run mypy tessera_vq scripts
```

## Interactive (t, K, m) bolt-on

The bolt-on lives in `tessera_vq.server` (Flask + waitress; install with `pip install -e .[server]`). Run it LAN-close to the embeddings store: `uv run python -m tessera_vq.server`. Endpoints: `GET /health`, `POST /sweep` (rate-distortion table), `POST /quantized` (NPZ of codebooks + index maps). See `docs/spec.md` §8.

For downstream code, `tessera_vq.client.VQTessera` is a plug-compatible drop-in for `geotessera.GeoTessera`:

```python
from tessera_vq.client import VQTessera

gt = VQTessera(server_url="http://michael:8000", t=64, k=16, m="cosine")
mosaic, transform, crs = gt.fetch_mosaic_for_region(
    (0.145, 52.045, 0.155, 52.055), year=2024
)
```

Same `(mosaic, transform, crs)` return shape as `GeoTessera.fetch_mosaic_for_region`.
