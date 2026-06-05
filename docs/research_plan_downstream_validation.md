# Research plan — downstream-validated VQ at large tile sizes

**Status:** APPROVED 2026-06-05 with the decisions below. Build proceeds workstream-by-workstream; **no long run starts without supervisor sign-off** (supervisor runs them).
**Author:** Claude Code, 2026-06-05.

## Decisions (resolved 2026-06-05)

- **Q1 `scan`:** reuse `sweep_window` / existing sweep machinery — do **not** build a new `scan`.
- **Q2 split:** **spatial hold-out is mandatory and primary.** Random k-fold is not the headline metric (it leaks spatial autocorrelation).
- **Q3 point 6:** compress the **stage-1 index map idx1** via a space-filling curve (+ RLE); idx2 is the incompressible floor. Codebooks are not the RLE target.
- **Q4 k1=20:** non-power-of-2 k1 is allowed; include the k1≈20 point.
- **Q5 k-means:** use **BLAS-GEMM** (chunked, SIMD via BLAS) — **not FAISS** (slow to precompute / build).
- **Q6 tessera_eval:** OK to **import** `tessera_eval` cross-repo (all personal repos).
- **Q7 sequencing:** workstreams **strictly sequential**, logging all results properly (Parquet + provenance + logs) for a later **technical report**.

**Standing constraints (this work):**
- **Parameter grid is LOCKED** to §3 (t∈{512,768,1024} × the three 2-byte configs {(64,1024),(128,512),(256,256)}, + optional k1≈20 sensitivity). Do **not** change it — ask first if a change seems warranted.
- **(32, 2048) dropped** (2026-06-05): k=2048 k-means ~11 s/tile, supervisor does not think it is needed.
- **Never modify anything under `/Users/skeshav/code/blore`** (read/import only).
- **Local commits only — never push.**
- **Ask before any long run**; supervisor executes it.
- Class imbalance is expected and acceptable — report macro (f1) and weighted (f1w); not treated as a risk.
**Supersedes (in priority, not in record):** the histogram-tail reconstruction metric of the current `phase3_rvq_sweep`. Pulls forward and merges spec Phase 4 (index compression) and Phase 5 (downstream), and expands Phase 3 to large tiles.

---

## 0. The reframing (why this plan exists)

Two problems with the work so far, both correct criticisms:

1. **The reconstruction metric is anchored arbitrarily.** Bin edges (hence the "near0" and "bad-tail" fractions) are frozen from the *first* readable bbox of each run. The threshold therefore moves run-to-run (we saw L2 5.59 / 7.11 / 8.09 across three runs for the *same* configs), so "tail mass" carries no absolute meaning. **The determinative question is the impact of VQ on a downstream task**, and we already have two labelled ground-truth datasets plus a working evaluation engine.

2. **t was too small.** t≤64 was forced by NaN-heavy tile edges on ~5 km bboxes. But the supervisor has worked extensively at **t=512** and finds that visually (first 3 bands) **k1≈20 already captures the landscape**. The apparent "reconstruction collapse" at large t in prior runs is an artifact of problem 1, not a real signal.

### The economics flip at large t (this is the headline)

At small t the per-tile codebook dominated the payload (90–99%). At large t it **inverts** — the 16-bit index map dominates, and the codebook becomes a rounding error:

| t | k1 | k2 | codebook B/px | index B/px | total B/px | ×fp32 | ×int8 | RLE floor* B/px | ×fp32 at floor |
|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 64 | 1024 | 0.53 | 2.00 | 2.53 | 202× | 51× | 1.78 | 287× |
| 512 | 256 | 256 | 0.25 | 2.00 | 2.25 | 228× | 57× | 1.25 | 410× |
| 1024 | 128 | 512 | 0.08 | 2.00 | 2.08 | 246× | 62× | 1.20 | 426× |
| 1024 | 256 | 256 | 0.06 | 2.00 | 2.06 | 248× | 62× | 1.06 | 482× |

\* RLE floor = codebook + idx2 only, i.e. assuming the stage-1 index map (idx1) compresses to ~0 via a space-filling curve + RLE (point 6). idx2 (the residual index, log2(k2) bits) is **incompressible** and sets the floor.

**Consequences that drive the whole plan:**
- Compression at large t is **170–248× over fp32 (42–62× over int8)** before any index compression — far beyond the ~7–26× we saw at t=64.
- The **lever is now the index map, not codebook factorization.** idx2 = log2(k2) bits/px is the hard floor; idx1 = log2(k1) bits/px is the part a Hilbert/Morton curve + RLE can crush (point 6). This is why **smaller k2 is attractive at large t** (lower floor), and why the supervisor's "k1≈20 is enough" matters: tiny k1 → very few base prototypes → long RLE runs → idx1 → ~0.
- **The 2-byte configs are not equivalent.** They trade idx2-floor against fidelity: (256,256) has the lowest floor (1.0–1.25 B/px) but the shallowest residual; (64,1024) has a deeper residual but a higher floor.

This table is *analytic*; the experiments below replace its fidelity guesses with **downstream F1**.

---

## 1. What exists vs. what's missing (from code investigation)

**Exists and reusable:**
- `tessera_eval` (`/Users/skeshav/code/blore/packages/tessera-eval/`) — standalone library, **no server needed**. Key seams:
  - `run_kfold_cv(vectors, labels, model_names, k=5, task="classification")` → per-fold + aggregate `mean_f1`, `mean_f1w`, confusion matrices. (`evaluate.py:429`)
  - `run_learning_curve(vectors, labels, ..., test_vectors=, test_labels=)` → supports a **fixed spatial test set**. (`evaluate.py:20`)
  - `rasterize_shapefile(gdf, field, transform, w, h, label_encoder=)` → polygons → per-pixel class raster. (`rasterize.py:8`)
  - `load_embeddings_for_shapefile(gdf, field, year, gt)` → tiles GeoTessera embeddings, rasterizes, extracts labelled pixels → `(vectors, labels, class_names, stats)`. (`data.py:116`)
  - `detect_field_type`, `make_classifier("rf", ...)` (RF = sklearn RandomForest, n_estimators=100, n_jobs=-1).
  - Classifier seam takes plain `(N,128) float32` + `(N,) int` — **this is exactly where we inject RVQ-reconstructed vectors.**
- Ground truth:
  - **Austria** — `austria.zip` → `austrian_crop_17classes/…shp`, **17 crop classes** (AC01–AC17, imbalanced 150–10 874 polys), CRS EPSG:32633 (UTM 33N), 42 789 polygons.
  - **Cumbria/Naddle** — `cumbria_naddle.zip` → `UKhabs_Naddle_Swindale_Mardale.shp`, UK habitats, CRS EPSG:27700 (British National Grid). The unpacked `Cumbria_naddle/` also has `perfect-validation.zip` and `random-validation.zip` (pre-defined splits — relevant to the split-protocol decision below).
- `tessera_vq.sweep.fast_quantize_tile` — **already** does sampled-fit (`sample_size=2000`) + full-assign, the pattern point 7 asks for. RVQ helpers (`rvq_quantize_window_for_serving`, `rvq_per_tile_errors`) exist.

**Missing / needs work:**
- **No function named `scan`** anywhere in `tessera_vq` (point 5). Closest is `sweep_window` (`sweep.py:406`). → *Open question Q1.*
- `fast_quantize_tile`'s `sample_size=2000` **cannot fit k=2048** (need sample ≫ k) and the full-assign step would materialize a 262 144×2048 distance matrix (~2 GB) per t=512 tile → **needs a fast, memory-blocked k-means** (point 7).
- Canonical bboxes are **10 km / window_px=1000** — too small for t=768/1024 (1024 px > 1000) and NaN-heavy at the edges (point 3). Needs ~12 km windows + jittered placement.
- No Morton/Hilbert/RLE modules yet (spec Phase 4 lists them as to-build; point 6).
- No anchor-free reconstruction metric.

---

## 2. Workstreams

Ordered by dependency. Each ends at a HALT.

### WS-0 — Infrastructure (enables everything large-t)

**0a. Fast CPU k-means for large tiles (point 7).** — **decided: BLAS-GEMM, no FAISS.**
- Refactor the numpy Lloyd to a **BLAS-GEMM distance** (`‖x−c‖² = ‖x‖²+‖c‖²−2x·cᵀ`, the `−2x·cᵀ` term via a single `np.dot` → multithreaded BLAS, SIMD) with **chunked assignment** (block the query rows so the `(chunk×k)` distance matrix stays in cache and peak RAM is bounded) + k-means++ init on a subsample, then one exact full-assign over all pixels.
- Benchmark on one real t=512 tile (262 144×128) at k∈{256,512,1024}: wall-clock fit+assign, peak RAM, reconstruction MSE (must not regress vs current). **k=2048 is optional — if too slow, drop the (32,2048) cell** per supervisor.
- Deliverable: `tessera_vq/kmeans_fast.py` (or extend `sweep.py`) — sampled-fit + blocked full-assign handling k up to ~1024 within a fixed RAM budget. Unit test: matches reference centroids/inertia on a synthetic 3-cluster tile; assignment exact over all pixels.

**0b. Large-tile window sampling (points 3 & 4).**
- Switch the sampler to **~12 km windows (≈1200 px)** so a t∈{512,768,1024} tile fits with margin.
- **Jitter the window centre** uniformly within ±tile_size/2 of the canonical tile centre (seeded) so tile-grid alignment varies across bboxes (decorrelates the sample).
- Per bbox, **select 1–2 tiles** with the highest finite-pixel fraction (drop NaN-heavy edges); **10 bboxes total** (supervisor: enough for stats). → so the reconstruction sweep runs on ~10–20 large tiles, not 100s.
- Deliverable: extend `tessera_vq/canonical.py` / the sampler; unit test on a synthetic mosaic that jitter stays in-bounds and tile selection prefers finite tiles.

HALT 0: report k-means benchmark table (speed/RAM/MSE) + chosen impl; confirm window size, jitter, tiles-per-bbox.

### WS-1 — Anchor-free reconstruction metric (cheap proxy; fixes problem 1)

- Replace the frozen-histogram output with **scale-free, run-stable** per-pixel metrics: relative L2 error `‖e‖/‖x‖`, cosine similarity, and **R² (fraction of variance explained** = 1 − Var(resid)/Var(x)) aggregated per tile and per config. No bin anchoring; directly comparable across runs.
- Keep this as a *fast screen* over the (t,k1,k2) grid; the downstream eval (WS-3) is the arbiter.
- Deliverable: update `phase3_sweep.py` / the sweep script to emit these columns; tests on synthetic windows with known variance.

HALT 1: report the anchor-free reconstruction table over the new grid.

### WS-2 — Index-map compression (point 6; = spec Phase 4)

- Build (spec already specifies these): `tessera_vq/morton.py`, `tessera_vq/hilbert.py`, `tessera_vq/entropy.py` (`rle_encode/decode`, optional `zstd`). Cross-check Morton/Hilbert against `pymorton` / `hilbertcurve` on a 16×16 fixture; bit-exact roundtrips.
- **Measure** (not assume) compressed bytes/px for idx1 under: row-major+RLE, Z-order+RLE, **Hilbert+RLE** (+ optional zstd), on the real stage-1 index maps from large-t tiles. idx2 reported at its incompressible floor (log2(k2) bits) — confirm empirically it does *not* compress (residual indices ≈ spatially white).
- Output the **effective bytes/px per config** (codebook + RLE(idx1) + idx2) to replace the analytic table in §0.
- Interpretation note (point 6 wording): I read "the **k1** part compresses, **k2** doesn't" as **the stage-1 index map idx1** (spatially autocorrelated → long runs) vs the stage-2 residual index idx2 (incompressible). The *codebooks* themselves aren't what RLE shrinks. → confirm Q3.

HALT 2: report effective bytes/px table (per pipeline, per config), and the idx1-compressibility result vs k1.

### WS-3 — Downstream validation (point 1; the determinative test; = spec Phase 5, RF variant)

The core experiment. For each dataset (Austria, Cumbria) and each config in the grid **plus a raw-float32 baseline**:

1. Tile the labelled region at size t; for each tile fetch raw embeddings (GeoTessera), fit per-tile RVQ (k1,k2) with the WS-0a k-means, reconstruct → `(H,W,128)` recon.
2. Rasterize labels (`rasterize_shapefile`), extract labelled pixels from **raw** and **recon** → `(N,128)` each + shared labels.
3. Run `tessera_eval` RF: `run_kfold_cv(vectors, labels, ["rf"], k=5, task="classification")` (and optionally `nn`).
4. Record **mean_f1 / mean_f1w** for raw vs recon; the metric is **ΔF1 = F1(raw) − F1(recon)** per config.

- **Codebooks are fit on all finite pixels of each tile** (the realistic compression scenario), even though only labelled pixels feed the classifier.
- **Validation gate (from spec):** raw-float32 F1 must match the known/Frank baseline within tolerance; if not, the loader is wrong — stop and debug before trusting any Δ.
- Deliverable: an adapter (likely a small module in `tessera_vq` or a script under `scripts/`) that wraps `load_embeddings_for_shapefile` with a per-tile RVQ step, reusing `rasterize_shapefile`. Results → `results/phase4/downstream.parquet` (config, dataset, classifier, f1_raw, f1_recon, delta, bytes_px, provenance).

HALT 3: report ΔF1 vs compression for both datasets.

### WS-4 — Synthesis

- The **Pareto frontier**: ΔF1 (downstream cost) vs effective bytes/px (from WS-2), per dataset, with the raw baseline at the origin. This is the engineering recommendation deliverable.
- Cross-check: does the cheap anchor-free reconstruction metric (WS-1) *rank* configs the same way downstream F1 does? If yes, the reconstruction proxy is vindicated for future tuning; if no, downstream wins and we say so.

HALT 4: Pareto plots + recommended (t, k1, k2) per dataset and overall.

---

## 3. Parameter grid (point 2)

- **Tile sizes:** t ∈ {512, 768, 1024}.
- **2-byte (16-bit-packed) configs** — `idx1_bits + idx2_bits = 16`:
  - (k1=64, k2=1024) — 6+10 bits
  - (k1=128, k2=512) — 7+9 bits
  - (k1=256, k2=256) — 8+8 bits
- → 3 × 3 = **9 cells** + raw baseline. **(32, 2048) was dropped** (supervisor, 2026-06-05: k=2048 too slow / not needed). This **relaxes the old k1<k2 rule** (256,256 has k1=k2); all k2 ≪ t² so the degeneracy guard is moot.
- Optional sensitivity: a **k1≈20** point to match the supervisor's visual finding (non-power-of-2, 5-bit idx1) — to be slotted in when WS-2/WS-3 run; k2 partner TBD.
- **Window:** 12 km (window_px ≈ 1200) read at the **existing canonical bbox centres** (supervisor: option (a), for comparability), not freshly sampled locations.

---

## 4. Decisions I need from you (before any action)

- **Q1 — `scan`:** there is no `scan` function in `tessera_vq`. Did you mean `sweep_window` / the `phase3_rvq_sweep` machinery (reuse it), or is `scan` something to build, or in another repo? 
- **Q2 — Split protocol (research-judgement; CLAUDE.md flags this):** random stratified k-fold *leaks spatial autocorrelation* (adjacent pixels in train and test) and **inflates** F1 — making VQ look harmless. A **spatial hold-out** (the engine supports `test_vectors/test_labels`; Cumbria ships `perfect-validation`/`random-validation`) is the honest test. Recommend **spatial split as primary**, random k-fold as a secondary sanity number. Confirm?
- **Q3 — Point 6 interpretation:** confirm "k1 compresses, k2 doesn't" means the **stage-1 index map (idx1)** vs the residual index (idx2), not the codebook arrays.
- **Q4 — k1≈20:** include a non-power-of-2 k1=20 sensitivity point (matches your visual result; same 5-bit idx1 budget as k1=32)?
- **Q5 — Dependencies:** OK to add `faiss-cpu` (and `zstandard`, `pymorton`/`hilbertcurve` for tests) as a `[fast]` / `[dev]` extra? If FAISS is unwanted I'll use the BLAS-GEMM numpy path (slower but no new core dep).
- **Q6 — Downstream eval seam:** OK for `tessera-vq` to import `tessera_eval` (cross-repo, via the blore venv / an editable install), rather than copying its loaders? This keeps us bit-identical to Frank's protocol but couples the two repos for this experiment.
- **Q7 — Scope/sequencing:** all four workstreams in one go, or stop after WS-0+WS-3 (the determinative result) and defer WS-2/WS-4? Also: budget — large-t k-means × 12 configs × 2 datasets could be hours; I'll confirm wall-time before any long run (CLAUDE.md rule 4) and **you run long jobs yourself** per our standing arrangement.

---

## 5. What I will NOT do without explicit approval

- Run any sweep / downstream job (you run long jobs).
- Change the parameter grid in committed code (this plan proposes it; awaiting sign-off).
- Modify anything under `/Users/skeshav/code/blore` (read/​import only unless you say otherwise).
- Add dependencies, push to remote, or alter split/metric protocols.

---

## 6. Risk register

- **k-means at k=2048, 262k px** is the compute bottleneck; if even FAISS is too slow per tile, fall back to fewer tiles (point 4 already says 1–2/bbox) or MiniBatch.
- **GeoTessera coverage** for Austria (UTM33) / Cumbria (BNG) must be checked early (a 1-tile smoke test) — if embeddings aren't served there, WS-3 stalls.
- **Class imbalance** (Austria AC06=150 vs AC04=10 874) → report macro-F1 (f1) not just weighted, and watch tiny classes.
- **Spatial leakage** (Q2) is the single biggest threat to a *meaningful* number; getting the split right matters more than the grid.
