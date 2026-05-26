# TEE-side wiring brief for `tessera-vq`

**Audience.** A coding agent (Claude Code) extending the TEE codebase at
`~/code/blore` (https://github.com/ucam-eo/TEE) to let TEE users opt into a "fast
path" that fetches **VQ-quantised** Tessera embeddings instead of raw embeddings.

**UI design is OUT OF SCOPE.** This doc covers only the wiring (models, settings,
provider helper, view glue, deployment). All UI decisions — where the toggle sits,
how `(t, k, m)` are exposed in the browser, presets, copy, default values shown to
users — are left to the human supervisor. If you find yourself inventing UI, stop
and ask.

---

## 1. Background

`tessera-vq` (https://github.com/sk818/tessera-vq, MIT, v0.1.0) provides per-tile
vector quantisation of Tessera embeddings. There are three pieces:

| Piece | What | Where it runs |
|---|---|---|
| Library | `tessera_vq.sweep.{sweep_window, quantize_window_for_serving}` | wherever (in TEE process; CPU on TEE) |
| Server | Flask app on `127.0.0.1:8000`: `GET /health`, `POST /quantized` | already deployed on `michael` via systemd unit `tessera-vq.service` |
| Client | `tessera_vq.client.VQTessera`, **plug-compatible** with `geotessera.GeoTessera` | TEE process |

For TEE, only the **client** path matters: instantiate `VQTessera` where TEE
currently instantiates `GeoTessera`, and the returned `(mosaic, transform, crs)`
tuple is identical, so downstream pipelines need no changes.

Read first:
- `README.md` in the VQ repo (top-level usage + install).
- `docs/integration.md` in the VQ repo (the boundary rule).

---

## 2. Boundary (the rule)

| Concern | Owned by |
|---|---|
| User accounts, login, sessions, `@login_required` | TEE |
| UI controls + per-user preferences for `(t, k, m)` | TEE |
| Per-user feature flag (use fast path or not) | TEE |
| Logging which path each call took | TEE |
| HTTP API `/health`, `/quantized`; library `sweep_window` etc. | tessera-vq |
| K-means, reconstruction, NPZ wire format | tessera-vq |
| Plug-compat client `VQTessera` | tessera-vq |

**Things that are not negotiable:**
- The bolt-on stays on `127.0.0.1:8000`. **Do not** add an Apache `ProxyPass /vq/`.
  TEE talks to it from within the Django process via `VQTessera`.
- Do not put VQ-specific computation (k-means, NPZ packing) into TEE; if you need
  something missing, file an issue / change against `tessera-vq`.
- Do not let the user pass arbitrary `sample_size` / `seed` over the wire unless
  the supervisor has approved it (CPU budget).

---

## 3. Install

Add to TEE's `requirements.txt`:

```
tessera-vq @ git+https://github.com/sk818/tessera-vq.git@v0.1.0
```

Then redeploy (or in dev: `pip install -r requirements.txt`).

The VQ package requires Python ≥ 3.12 (TEE already uses ≥ 3.12).

---

## 4. Settings

Add to `tee_project/settings.py` (or wherever TEE keeps configuration):

```python
import os

# --- Tessera VQ bolt-on ---
TESSERA_VQ_URL = os.environ.get("TESSERA_VQ_URL", "http://127.0.0.1:8000")
TESSERA_VQ_DEFAULTS = {
    "t": int(os.environ.get("TESSERA_VQ_DEFAULT_T", "64")),
    "k": int(os.environ.get("TESSERA_VQ_DEFAULT_K", "16")),
    "m": os.environ.get("TESSERA_VQ_DEFAULT_M", "cosine"),
}
TESSERA_VQ_TIMEOUT_SECONDS = float(os.environ.get("TESSERA_VQ_TIMEOUT_SECONDS", "120"))
```

Defaults are sensible starting points; the supervisor will revisit after UX
experiments (see open questions, §10).

---

## 5. User model: per-user fast-path preference

TEE already has Django users (Django auth + `tee_*` management commands). Add a
single `UserProfile` (or extend `User` via a one-to-one) holding the fast-path
preference and optional `(t, k, m)` overrides:

```python
# api/models.py  (or wherever TEE's models live)
from django.conf import settings
from django.db import models


DISTANCE_CHOICES = [("euclidean", "euclidean"), ("cosine", "cosine")]


class UserProfile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="vq_profile",
    )
    use_vq_fast_path = models.BooleanField(default=False)
    # NULL => fall back to settings.TESSERA_VQ_DEFAULTS
    vq_t = models.PositiveIntegerField(null=True, blank=True)
    vq_k = models.PositiveIntegerField(null=True, blank=True)
    vq_m = models.CharField(max_length=16, null=True, blank=True, choices=DISTANCE_CHOICES)
```

Migration:

```bash
python manage.py makemigrations
python manage.py migrate
```

Ensure new users get a profile automatically (signal handler on `post_save` of
`User`, or `get_or_create` in the provider helper below). Either pattern is fine
— pick whichever matches TEE's existing conventions.

---

## 6. Embeddings provider helper

The whole TEE migration hinges on a single helper that decides what client to
instantiate based on the request user. Add this exactly:

```python
# api/embeddings_provider.py
from django.conf import settings
from geotessera import GeoTessera
from tessera_vq.client import VQTessera


def get_embeddings_provider(user, *, embeddings_dir: str | None = None):
    """Return a GeoTessera-compatible client for ``user``.

    If the user has opted into the VQ fast path, returns a VQTessera pointed at
    TESSERA_VQ_URL with their (t, k, m) defaults; otherwise returns a plain
    GeoTessera (optionally with a tile cache dir, matching existing TEE usage).

    Both return values support ``fetch_mosaic_for_region(bbox, year, target_crs)``
    and ``fetch_embedding(lon, lat, year)`` with identical (mosaic, transform, crs)
    signatures, so downstream code does not change.
    """
    profile = getattr(user, "vq_profile", None)
    if profile is None or not profile.use_vq_fast_path:
        if embeddings_dir is not None:
            return GeoTessera(embeddings_dir=embeddings_dir)
        return GeoTessera()

    defaults = settings.TESSERA_VQ_DEFAULTS
    return VQTessera(
        server_url=settings.TESSERA_VQ_URL,
        t=profile.vq_t or defaults["t"],
        k=profile.vq_k or defaults["k"],
        m=profile.vq_m or defaults["m"],
        timeout=settings.TESSERA_VQ_TIMEOUT_SECONDS,
    )
```

Notes:

- `VQTessera` does **not** take `embeddings_dir` — it talks over HTTP to the
  bolt-on. Existing call sites that pass `embeddings_dir=` for `GeoTessera` should
  still pass it; the helper forwards it only when returning a `GeoTessera`.
- Anonymous / unauthenticated users get the standard `GeoTessera` path.
- The helper is cheap to call repeatedly; `VQTessera` is a tiny object (no socket
  is opened until you call a method).

---

## 7. Call-site migration

`tessera-vq` was chosen precisely because **`VQTessera.fetch_mosaic_for_region`
and `fetch_embedding` return the same `(mosaic, transform, crs)` tuple as
`GeoTessera`** — so call sites change to one line.

These are the existing call sites (as of TEE 1.2.x; grep to confirm before
editing):

| File | Line(s) | Current |
|---|---|---|
| `process_viewport.py` | ~64 | `gt.GeoTessera(embeddings_dir=str(EMBEDDINGS_DIR))` (cached singleton — see §7a) |
| `api/views/viewports.py` | ~600 | `gt = GeoTessera()` inside a view |
| `packages/tessera-eval/tessera_eval/server.py` | ~663, ~1273 | `_geotessera_instance = GeoTessera(embeddings_dir=...)` cached singleton |
| `scripts/tee_evaluate.py` | ~87, ~131 | `gt = GeoTessera()` (CLI script — see §7b) |

### 7a. Module-level cached singletons

These are tricky because a cached *singleton* doesn't know about the current
request user. Options:

1. **Keep the singleton as `GeoTessera`** (server-side fallback) and add a
   *per-request* call to `get_embeddings_provider(user)` for code paths that
   actually depend on the user's choice. The singleton stays as a fallback /
   bulk path.
2. **Replace the singleton with a function** returning the provider per call:
   `gt = get_embeddings_provider(request.user, embeddings_dir=EMBEDDINGS_DIR)`.
   Cheap because `VQTessera()` is just an object construction.

Recommendation: **option 2** for any code path that takes a `request` /
`user`; **option 1** for the existing batch path that already has no user (e.g.
the cached singleton in `process_viewport.py` if it's invoked from a worker
without a session).

### 7b. CLI scripts (no user context)

`scripts/tee_evaluate.py` runs without a request. Either:

- Keep using `GeoTessera()` (recommended; CLI is a single user — the operator —
  and they can pass `--vq` to opt in explicitly).
- Or accept `--vq`, `--t`, `--k`, `--m` CLI args and construct `VQTessera`
  directly.

UI/UX of the CLI is out of scope; pick what's idiomatic for TEE.

### 7c. Patch shape

For each view that holds a `request`:

```python
# before
from geotessera import GeoTessera
gt = GeoTessera()
mosaic, transform, crs = gt.fetch_mosaic_for_region(bbox, year=year)

# after
from api.embeddings_provider import get_embeddings_provider
gt = get_embeddings_provider(request.user)
mosaic, transform, crs = gt.fetch_mosaic_for_region(bbox, year=year)
```

That's the whole migration for most call sites. Downstream code that consumes
`(mosaic, transform, crs)` is unchanged.

---

## 8. New views (only if TEE needs ad-hoc per-call control)

If the UI wants to let a user specify `(t, k, m)` for a single call without
mutating their saved profile (e.g. "try k=64 just for this preview"), add one
thin view. **The shape of the view, URL, JSON request/response is wiring;
the rendering on the browser side is UI and is out of scope for this doc.**

```python
# api/views/embeddings.py
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from tessera_vq.client import VQTessera


@login_required
def vq_quantized(request):
    """Server-side proxy: validate, call VQTessera, stream NPZ back to client."""
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    body = json.loads(request.body)
    bbox = tuple(float(v) for v in body["bbox"])  # validate length 4
    year = int(body.get("year", 2024))
    defaults = settings.TESSERA_VQ_DEFAULTS
    t = int(body.get("t", defaults["t"]))
    k = int(body.get("k", defaults["k"]))
    m = body.get("m", defaults["m"])
    client = VQTessera(server_url=settings.TESSERA_VQ_URL, t=t, k=k, m=m,
                       timeout=settings.TESSERA_VQ_TIMEOUT_SECONDS)
    # Use the raw HTTP path so the NPZ bytes pass through unchanged:
    npz_bytes = client._post("/quantized", {
        "bbox": list(bbox), "year": year, "t": t, "k": k, "m": m,
        "sample_size": int(body.get("sample_size", 2000)),
        "seed": int(body.get("seed", 42)),
    })
    return HttpResponse(npz_bytes, content_type="application/octet-stream")
```

(URL routing, REST framework wrapping, CSRF handling etc. — follow TEE's
existing patterns.)

If TEE doesn't need ad-hoc overrides yet, **skip this view entirely**; the
provider helper plus saved user defaults is sufficient.

---

## 9. Deployment / runtime

- The `tessera-vq` bolt-on already runs on `michael` as the `tessera-vq.service`
  systemd unit. No changes there. `curl http://localhost:8000/health` → `{"ok":true}`.
- Apache config on `michael` should **not** add a `/vq/` proxy. The bolt-on
  remains localhost-only.
- Restart TEE after the `requirements.txt` change so Django picks up the new
  package.
- Environment overrides for the helpers (if needed):
  - `TESSERA_VQ_URL` — default `http://127.0.0.1:8000`.
  - `TESSERA_VQ_DEFAULT_T`, `..._K`, `..._M`.
  - `TESSERA_VQ_TIMEOUT_SECONDS`.

---

## 10. Open questions for the human supervisor

These are decisions for the supervisor — do not invent answers:

1. **Default `(t, k, m)`** for new users. Suggested starting point: `t=64, k=16,
   m=cosine`. Confirm or adjust.
2. **Per-user vs per-viewport** preference. Is `use_vq_fast_path` a single bit
   on the user, or per-viewport (so the same user can mix paths across
   viewports)? Current scheme is per-user.
3. **Pre-validation of `t, k, m`** values in TEE, or rely on the bolt-on's own
   guardrails (bbox 10 km cap, k≥1, t≥1)? Both work; TEE pre-validation is
   nicer for UX.
4. **Telemetry**: log which path was used per request? (Useful for evaluating
   fast-path uptake and quality drift.)
5. **A/B logic**: any cohort-based rollout, or pure opt-in? Pure opt-in is
   simplest; the supervisor may want a treatment cohort later.

---

## 11. Testing

- Unit-test `get_embeddings_provider`: with a user who has no profile / profile
  with `use_vq_fast_path=False` / profile with `use_vq_fast_path=True` and
  partial overrides → assert the returned class and key attributes.
- Integration: monkeypatch `tessera_vq.client.VQTessera._post` to return a
  synthetic NPZ (build one with `np.savez` of the shape
  `codebooks (n, k, 128) / indices (n, t, t) / positions (n, 2) / meta (5,) /
  distance ()`); assert the downstream view renders correctly.
- Smoke: against the live bolt-on on michael, hit a small Cambridge bbox:
  `(0.145, 52.045, 0.155, 52.055)` — should return ~18 tiles for `t=16, k=4`.
- Ensure CI doesn't actually hit michael; gate live tests with a marker.

---

## 12. What NOT to do

- Don't expose `/quantized` over Apache — `localhost` only.
- Don't add VQ-specific code (k-means, NPZ packing, sweep math) to TEE.
- Don't bypass the provider helper — every user-aware call site goes through it,
  so the choice is consistent.
- Don't hold a single global cached `VQTessera` keyed to one user's settings —
  the per-call helper is cheap and avoids cross-user bugs.
- Don't invent UI. If your task feels like UI design, stop and ask the
  supervisor.

---

## 13. Pointers

- VQ repo: https://github.com/sk818/tessera-vq (v0.1.0, MIT).
- VQ docs: `README.md`, `docs/integration.md`, `docs/spec.md` (§8 covers the
  bolt-on).
- VQ client API: `tessera_vq/client.py` — read the docstring and the
  `fetch_mosaic_for_region` / `fetch_embedding` signatures.
- VQ server API: `tessera_vq/server.py` — `/health` and `/quantized` only.
- Live bolt-on: `tessera-vq.service` on `michael` (`sudo systemctl status
  tessera-vq`).

---

## 14. Definition of done

For this wiring task:

1. `requirements.txt` updated; package installs cleanly.
2. `UserProfile` (or equivalent) model + migration landed.
3. Settings block added.
4. `get_embeddings_provider` helper exists and is unit-tested.
5. All `request`-aware call sites use the helper.
6. Non-`request` singletons and CLIs are explicitly handled per §7a/§7b.
7. Smoke test against `michael` passes for an opted-in user.
8. No UI work has been done.

UI work — the toggle, the `(t, k, m)` controls, anything users see in the
browser — comes after the wiring is in place, in a separate task, with the
supervisor specifying behaviour.
