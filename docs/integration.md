# Integrating `tessera-vq` into a parent service

This note describes the boundary between `tessera-vq` and a parent service such
as [TEE](https://github.com/ucam-eo/TEE), so that the two evolve cleanly.

## Three usage modes

1. **Local library** *(primary, public, free)*
   ```python
   from geotessera import GeoTessera
   from tessera_vq.sweep import sweep_window, quantize_window_for_serving

   gt = GeoTessera()
   mosaic, transform, crs = gt.fetch_mosaic_for_region(bbox, year=2024)

   # rate-distortion exploration: returns one row per (t, k, m, subtile)
   rows = sweep_window(
       mosaic,
       ts=[16, 64, 256],
       ks=[4, 16, 64, 256],
       ms=["euclidean", "cosine"],
   )

   # serve-ready quantisation for a chosen (t, k, m)
   codebooks, indices, positions = quantize_window_for_serving(
       mosaic, t=64, k=16, m="cosine",
   )
   ```
   No server. The CPU spent on the sweep is the caller's.

2. **Plug-compatible client** *(when a server is available)*
   ```python
   from tessera_vq.client import VQTessera   # drop-in for geotessera.GeoTessera

   gt = VQTessera("http://your-host:8000", t=64, k=16, m="cosine")
   mosaic, transform, crs = gt.fetch_mosaic_for_region(bbox, year=2024)
   ```
   Same `(mosaic, transform, crs)` return shape as `geotessera.GeoTessera`, so
   downstream consumers swap on one line.

3. **Self-hosted server** *(optional)*
   `python -m tessera_vq.server` exposes **only** `/health` and `/quantized`.
   The exploration `sweep_window` is **not** served — keep that local. CPU per
   `/quantized` request is bounded (one k-means per tile, fast sampled fit,
   vectorised assign) and bbox size is capped at `TESSERA_VQ_MAX_BBOX_KM` (10 km
   per side by default).

## The boundary

`tessera-vq` knows nothing about users, sessions, UIs, or per-user defaults.
The parent service owns all of that:

| Concern                          | Where it lives           |
| -------------------------------- | ------------------------ |
| User accounts, sessions, auth    | Parent service           |
| UI controls (toggle, `t/k/m`)    | Parent service           |
| Per-user `(t, k, m)` defaults    | Parent service           |
| Rate limiting, quotas            | Parent service / proxy   |
| Logging request metadata         | Parent service           |
| HTTP API `/health`, `/quantized` | `tessera-vq`             |
| `sweep_window`, `quantize_*`     | `tessera-vq` (library)   |
| `VQTessera` Python client        | `tessera-vq`             |
| K-means + reconstruction         | `tessera-vq`             |

If you ever feel like reaching into the parent service's session state from
inside the bolt-on, that's the signal you've crossed the line.

## Integration recipe (TEE / Django example)

```python
# parent_service/views.py
from tessera_vq.client import VQTessera
from geotessera import GeoTessera

@login_required
def embeddings_view(request):
    user = request.user
    if user.use_vq_fast_path:
        gt = VQTessera(
            "http://localhost:8000",
            t=user.vq_t,
            k=user.vq_k,
            m=user.vq_m,
        )
    else:
        gt = GeoTessera()
    mosaic, transform, crs = gt.fetch_mosaic_for_region(bbox, year=year)
    # ... existing pipeline unchanged ...
```

The point is the one-liner swap. Everything that consumes the `(mosaic,
transform, crs)` tuple from `GeoTessera` keeps working unchanged.

## Public exposure notes

- `tessera-vq` ships **no authentication** of its own. Put a reverse proxy
  (Apache / nginx) with auth in front before exposing `/quantized` to the
  internet, or proxy it through the parent service's authenticated routes.
- The 10 km bbox guardrail is the per-request CPU bound. Adjust with
  `TESSERA_VQ_MAX_BBOX_KM` if needed.
- `/sweep` is deliberately *not* exposed by the server — sweeps run as a local
  library call on the caller's CPU.
