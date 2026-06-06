# Bolt-on availability plan — int8 codebooks, write-through cache, rate limiting

**Status:** APPROVED 2026-06-06 (decisions below). Ready to implement.
**Author:** Claude Code, 2026-06-06.
**Scope:** the tessera-vq bolt-on (`tessera_vq.server` on michael) + the `VQTessera`
client. Full static precompute is **explicitly out of scope for now** (deferred).

## Decisions (resolved 2026-06-06)

- **Q1 codebook quant:** per-dim **uint8 (min/max)** — confirmed.
- **Q2 cache unit:** **Tessera-tile granularity** — confirmed (request-path refactor accepted).
- **Q3 store:** **local disk on michael**; ~740 G free → **hard cap 500 GB** → the store is
  durable *up to 500 GB*, then **LRU-evicts** (≈287k tiles fit at ~1.74 MB/tile; a regional
  viewer is unlikely to touch that many, so it is effectively keep-forever with LRU as a
  safety valve). Evicted tiles simply recompute on next request.
- **Q4 rate limit:** **nginx `limit_req`** (supervisor will add it). So *in-repo* WS-3 is
  only the **concurrency cap + 429**; per-IP rate limiting lives in nginx, not the code.
- **Q5 deploy:** **batch all three** into one v0.5.0 deploy.
- **Q6 int8 validation:** a **single Austria fold (or Cumbria one-shot)** is an acceptable
  no-regression gate.

Three workstreams, in priority order: **(1) int8 codebooks in the wire format**,
**(2) durable write-through cache**, **(3) rate limit + concurrency cap**. They share a
small cross-cutting prerequisite (param validation) and one coordinated deploy.

---

## WS-1 — int8 codebooks in the NPZ (do first)

**Why.** The `/quantized_rvq` NPZ currently ships `codebooks1`/`codebooks2` as **float32**
(~0.54 B/px at t=512), so the served/stored artifact is ~60× int8, not the 72× the tech
note assumes. The raw embeddings are themselves int8 (240 TB/yr), so the codebooks — which
are *averages* of int8 values — lose only sub-int8 precision when requantized. Expected
downstream impact ≈ 0.

**Design.**
- Quantize each codebook **per (tile, stage, dimension)** with an affine uint8 map
  (min/max → 0..255), matching TEE's existing `dequantize_uint8(q, dim_min, dim_max)`
  convention. Per-dim (not per-tile) because the space is anisotropic (per-dim mean/scale
  differ). Asymmetric (min/max) because stage-1 prototypes are offset from zero; it also
  handles the near-zero-centred stage-2 residual fine.
- NPZ carries, per stage: `codebooksN_q (n, kN, 128) uint8`, `codebooksN_lo (n, 128)
  float32`, `codebooksN_hi (n, 128) float32`. Scales are ~2 KB/tile → negligible per px.
- Client dequantizes (`lo + q/255 * (hi - lo)`) before reconstruction; handles `hi == lo`
  (constant dim → q=0, value=lo).
- New helper pair in `tessera_vq` (e.g. `quant.py`): `quantize_codebook_uint8` /
  `dequantize_codebook_uint8`, bit-exact round-trip within int8 tolerance, unit-tested.

**Byte effect.** codebook 0.54 → ~0.14 B/px; total ~2.14 → ~1.74 B/px (≈ **73× int8**),
i.e. ~3 TB/yr instead of ~4 if/when stored.

**Validation gate (must pass before deploy).** Re-run the reconstruction metric (R² /
relative-L2) and a downstream F1 spot-check (Cumbria is cheap; or one Austria fold) with
int8 vs float32 codebooks at the recommended config. Require ΔR² and Δf1 within noise.

**Breaking?** Yes — another NPZ schema change. Per the earlier decision (only we use it)
that's fine; fold it into one version bump with WS-2/WS-3 (see Deploy).

---

## WS-2 — durable write-through cache

**Why.** Random global access over 1.6M tiles kills an LRU, but a *durable* store
(compute once, keep forever) pays off whenever a tile is ever requested twice — and for a
viewer, distinct tiles requested grows sublinearly. Storage ∝ actual demand; zero upfront
backfill. Cold tiles pay one live-compute latency on first hit, then are free.

**Key design decision — cache granularity.**
- **Target: Tessera-tile granularity** (the 0.1° grid, the natural unit), so overlapping
  bbox requests reuse tiles. This means restructuring the request path from
  "read bbox window → tile → quantize" into "bbox → set of Tessera tiles → per tile
  {hit store | read+quantize+store} → assemble response." Best reuse; more refactor.
- **Fallback: request-bbox-snap** (cache the whole NPZ keyed by snapped bbox+params).
  Trivial to add, but only catches exact-viewport revisits — far less reuse. (→ Q2.)

**Store.**
- Key = `(tessera_tile_id, year, t, k1, k2, m, seed, format_version)`.
- Path layout: `…/vq-cache/<format_version>/<year>/<t>_<k1>_<k2>/<tile>.npz`.
- Backend: **local disk on michael** (path under michael's free space; S3 later if needed).
- Stored blob = the WS-1 int8 NPZ (so WS-1 lands first).
- **Write atomicity:** temp file + atomic rename (as geotessera does).
- **Thundering herd:** per-key lock (file lock or in-proc dict of locks) so two requests
  for the same cold tile don't both compute — the second waits and then reads.
- **Eviction: LRU at a 500 GB cap.** Track per-tile last-access (mtime/atime or a small
  index); when a write would exceed 500 GB, delete least-recently-used tiles until under.
  Below the cap it is keep-forever; evicted tiles recompute on next request. (Env-tunable
  cap, default 500 GB.)

**Interaction.** Wraps the existing compute; a hit skips geotessera read *and* k-means
entirely — the whole point for availability/cost. Builds on WS-1 (stores int8) and on
param validation (stable keys).

---

## WS-3 — rate limit + concurrency cap (no API keys)

**Why.** Without keys, keep it simple but protect michael's CPU/RAM and bound abuse.

**Design (in-repo = concurrency only; rate limiting is nginx, added by supervisor).**
- **Global concurrency semaphore** around the *live-compute* (cold-miss) path only — a
  `threading.BoundedSemaphore(≈ncores)`; return **429 + Retry-After** when full. Cache
  hits and `/health` don't count (cheap). This is the load-bearing protection (k-means +
  geotessera read are the only expensive work) and the only WS-3 code change.
- **Per-IP rate limit:** handled by **nginx `limit_req`** in front of `:8000` (supervisor
  manages this, outside the repo). The plan will include a sample `limit_req` snippet for
  convenience, but no Flask-Limiter dependency is added.
- Keep the existing **bbox cap** (10 km, env-tunable).

---

## Cross-cutting — param validation (small, shared prerequisite)

Today `t/k1/k2/sample_size/seed` come from the request body unbounded — a DoS vector
(huge k / sample_size) and it muddies cache keys. Rather than hard-lock (which would break
research sweeps that legitimately vary params), **validate/cap**: reject absurd values
(e.g. k1,k2 ≤ 256/1024, sample_size ≤ a ceiling, t within a range) with `400`, and make the
cache key include the actual params. Preserves flexibility, removes the DoS, keys stay
clean.

---

## Sequencing & deploy

1. **WS-1 int8 codebooks** + validation gate.
2. **Param validation** (tiny; unblocks WS-2/WS-3).
3. **WS-2 write-through** (stores the WS-1 int8 NPZ).
4. **WS-3 rate limit + concurrency** (independent; can land alongside WS-2).
5. **One coordinated deploy** as **v0.5.0**: bump version, redeploy the michael bolt-on
   (`git checkout v0.5.0 && uv sync --extra server && restart`), rebuild+push
   `sk818/tee:stable`, redeploy TEE, smoke-test the fast path. Batching all three into one
   breaking deploy avoids three separate fast-path-skew windows.

Each WS is independently testable on synthetic data (unit tests) before the deploy; only
the WS-1 validation gate needs real data (a cheap Cumbria/one-fold run — yours to run).

---

## Decisions I need from you

- **Q1 (WS-1):** per-dim **uint8 (min/max)** codebook quantization (matches TEE's
  `dequantize_uint8`) — OK? Or prefer symmetric int8?
- **Q2 (WS-2):** cache at **Tessera-tile granularity** (more reuse, request-path refactor)
  vs the simpler **bbox-snap** (less reuse)? I recommend tile-level.
- **Q3 (WS-2):** **local disk on michael** for the store now (S3 later)? And any path/size
  budget I should respect?
- **Q4 (WS-3):** is there a reverse proxy in front of `:8000` I should use for rate
  limiting (nginx `limit_req`), or do it **in-app (Flask-Limiter, in-memory)**? (The
  integration doc says no Apache proxy for the bolt-on, which points to in-app.)
- **Q5 (deploy):** batch all three into one **v0.5.0** breaking deploy (recommended), or
  ship WS-1 on its own first?
- **Q6 (validation):** is a **Cumbria one-shot (or single Austria fold)** an acceptable
  gate for "int8 codebooks don't regress downstream," or do you want the full Austria run?
