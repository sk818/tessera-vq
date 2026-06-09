# Per-tile vector quantization for Tessera embedding compression — tech note

**Author:** Tessera-VQ study (S. Keshav, CAC project). **Date:** 2026-06-05
(rev. 2026-06-08: gzip replaces RLE for the stage-1 plane; tile size denoted `p`).
**Status:** results note for the engineering recommendation to GeoTessera.

---

## 1. Summary

Tessera produces a 128-dimensional embedding per ~10 m pixel. Stored as int8 that is
128 bytes/pixel; as fp32, 512 bytes/pixel. We compress it with **per-tile residual
vector quantization (RVQ)** and validate the result not on reconstruction error but on
a **downstream land-cover/crop classifier**.

**Recommendation: tile size p = 512, stage-1 codebook k1 = 20, stage-2 codebook
k2 = 256, stored as two separate byte planes — a gzip-compressed stage-1 index plane
plus a raw stage-2 index plane — giving ≈ 1.38 bytes/pixel = 93× over int8 (372× over
fp32) with no measurable downstream accuracy loss** (reconstructed embeddings classify
as well as raw; if anything VQ mildly denoises them). The headline trade-off is shown in
`figures/phase4_pareto.png`.

The road here included one genuine wrong turn — small tiles (p = 32/64) judged by a
mis-anchored reconstruction metric — corrected once we switched to a downstream task.
A later correction: the stage-1 index plane is better compressed by a general-purpose
entropy coder (gzip/zstd, ≈ 2.3× smaller) than by the bespoke run-length encoding we
first used. This note records the hypothesis, the geometry diagnostics, the wrong turn,
the relative roles of k1 and k2, the index-coding choice (gzip over RLE), and the
justification for the final choice.

---

## 2. Initial hypothesis

Tessera embeddings are strongly **spatially autocorrelated**: within a small ground
patch, only a handful of land-cover "prototypes" (a crop, a road, water, woodland)
actually occur. So a tile of p×p pixels should be representable by a **small per-tile
codebook** of prototype vectors plus a **per-pixel index map** pointing each pixel at
its prototype. If the codebook is small and the index map compresses, the per-pixel cost
collapses far below the raw 128 bytes — the bet was 99%+ byte reduction at limited
downstream cost.

Two stages (RVQ) refine this: stage 1 quantizes the embedding to one of `k1` coarse base
prototypes; stage 2 quantizes the *residual* to one of `k2` prototypes. The
reconstruction is `codebook1[idx1] + codebook2[idx2]`. Each codebook entry is 128 bytes
(one int8 per dimension), so codebook cost is `(k1 + k2) · 128 / p²` bytes/pixel and the
index cost is whatever the two index planes take.

---

## 3. Embedding geometry: isotropy diagnostics and the distance metric

Before quantizing we characterized the embedding distribution, because it determines
whether euclidean k-means is the right tool.

- **Marginals are near-Gaussian in shape but not formally normal.** Over 200 random
  1-D projections, Shapiro–Wilk rejected normality for 99% and Epps–Pulley for 95% of
  directions — but the Shapiro–Wilk statistic sat at ≈ 0.99 (1.0 = exactly Gaussian).
  The rejections are the usual large-sample artifact (millions of pixels make tiny
  departures "significant"); the *shape* is close to Gaussian.
- **The space is anisotropic.** Per-dimension means are offset (spanning ≈ −3.9 to 5.1;
  e.g. dim 0 mean ≈ 4.2) and variances span a wide range (≈ 1.3–15.0), i.e. dimensions
  carry different scales and centres. No dimensions are collapsed.
- **Euclidean beats cosine decisively for reconstruction.** In the Phase-2 sweep, median
  per-pixel reconstruction error under euclidean k-means was ≈ 4.6 (L2) versus ≈ 27 under
  cosine k-means at the same k — cosine discards magnitude, which carries real signal
  here. We therefore fixed **euclidean (L2) k-means** as the quantizer and later dropped
  cosine from all sweeps.

Conclusion: a near-Gaussian, anisotropic space in which L2 quantization is appropriate;
RVQ operates in raw L2 space (stage 1 already discards no magnitude, so cosine is not
meaningful for the residual either).

---

## 4. The wrong turn: small tiles and a mis-anchored metric

The first sweeps used **small tiles (p = 32, 64)**. Two things drove this and both were
mistakes:

1. **NaN-heavy edges on small bounding boxes.** A ~5 km box leaves large-tile grids full
   of no-data edge pixels, so we shrank the tile. The right fix was a larger (~12 km)
   window with jittered tile placement, not a smaller tile.
2. **A reconstruction metric anchored arbitrarily.** Per-pixel L2 errors were histogrammed
   against bin edges frozen from the *first* bounding box of each run. The "tail mass" and
   "near-zero fraction" therefore moved run-to-run (the same configuration's overflow
   threshold drifted across L2 = 5.6 / 7.1 / 8.1 in three runs), so the numbers carried no
   absolute meaning, and the apparent "reconstruction collapse" at larger tiles was an
   artifact of the moving anchor.

At small p the economics were also genuinely bad: the per-tile codebook **dominated the
payload (90–99% of bytes)** because `(k1 + k2) · 128 / p²` is large when p is small, so
compression topped out around 7–26× and degraded quickly. We initially read this as "large
tiles reconstruct poorly," which was backwards.

Two corrections fixed the study:

- **Replace the metric with an anchor-free one** — per-pixel *relative* L2 error
  `‖x − x̂‖ / ‖x‖` and **R²** (fraction of variance explained), both scale-free and
  run-stable.
- **Make the downstream task the arbiter.** Reconstruction error is a proxy; what matters
  is whether a classifier trained on reconstructed embeddings matches one trained on raw.

---

## 5. Large tiles flip the economics

Re-running at large tiles (p ∈ {512, 1024}, ~12 km windows) with the anchor-free metric:

- **R² ≈ 0.79, and essentially flat** — across the k1/k2 split (spread ≤ 0.004, ~25× below
  the between-tile sd) and across tile size (0.798 at p = 512 → 0.791 at p = 1024).
- **The byte economics invert.** At large p the codebook term `(k1+k2)·128/p²` becomes
  negligible (2–34% of the payload at p = 512, ~2% at p = 1024); the **16-bit index map now
  dominates**. Raw compression jumps to ~170–313× over fp32 before any index coding.

So at large tiles the lever is no longer the codebook (the earlier obsession with
codebook factorization) — it is the **index map**.

---

## 6. The relative roles of k1 and k2

Because reconstruction R² is flat across the split, k1 and k2 are free to be chosen for
*compressibility* and *downstream accuracy* rather than fidelity. They play distinct roles:

- **k2 (the residual index) is the incompressible floor.** The stage-2 residual is
  spatially white, so its index map does not run-length-encode (Hilbert+RLE on it is
  *worse* than raw). Stored byte-aligned it costs exactly 1 byte/pixel at k2 ≤ 256.
  Crucially, **k2 = 256 dominates** k2 = 128: both cost one byte, but 256 gives more
  residual codewords for free — and going beyond a byte (k2 = 512/1024) doubles the
  dominant cost for ~zero R² gain. So **k2 = 256, fixed.**
- **k1 (the base index) is the compressible part.** The stage-1 map is spatially smooth
  (it tracks land cover), so it compresses — and the smaller k1 is, the smoother the map
  and the better it compresses. Measured idx1 cost fell monotonically with k1 under every
  coder (gzip: ≈ 0.24 B/px at k1 = 20 up to ≈ 0.46 B/px at k1 = 128, p = 1024).

Two consequences shaped the storage format:

- **The planes must be stored separately and byte-aligned.** A 16-bit *interleaved*
  (idx1,idx2) index defeats compression — the white idx2 destroys the structure a coder
  would exploit. Stored separately and byte-aligned (k ≤ 256 → 1 byte), each plane is
  byte-addressable and independently compressible.
- **A general-purpose entropy coder beats RLE on the stage-1 plane.** RLE was the obvious
  first choice for a smooth map, but it captures only *immediate* repetition and
  entropy-codes nothing, whereas the base indices repeat at longer range *and* are far
  from uniformly distributed. Measured across four contrasting biomes (Amazon, Sahel,
  Welsh pasture, Iowa cropland), **gzip (DEFLATE) is ≈ 2.1–2.6× smaller than byte-aligned
  RLE** in every cell, and zstd a few % smaller still; at p = 512, k1 = 20 the plane falls
  from ≈ 0.63 B/px (RLE) to ≈ 0.24 (gzip) / ≈ 0.22 (zstd). Tellingly, gzip *after* RLE
  (≈ 0.27) is worse than gzip on the raw plane — RLE has already discarded the
  symbol-frequency structure DEFLATE's Huffman stage needs. Scan order barely matters for
  either coder (among RLE orderings row < Hilbert < Morton; for gzip, < 3%), so **the
  space-filling machinery can be dropped** — the plain row-major byte plane fed to
  gzip/zstd is both simplest and best.

This also makes the supervisor's visual observation precise: "k1 ≈ 20 captures the
landscape" is exactly the regime where the stage-1 map is smoothest and most compressible.

---

## 7. Downstream validation decides p

We compressed every GeoTessera tile overlapping two labelled datasets, reconstructed it
through per-tile RVQ, and trained the same Random Forest on **raw vs reconstructed**
embeddings at the labelled pixels, under **spatial group k-fold** (whole tiles held out —
random k-fold leaks autocorrelated neighbours and would flatter VQ). Metric: macro-F1
retention = reconstructed / raw F1.

- **Datasets:** Austria (17 crop classes, 8.2M labelled pixels — tight error bars) and
  Cumbria/Naddle (16 habitat classes, only 4 tiles — noisy). Absolute macro-F1 is low
  (≈ 0.20–0.22) because spatial hold-out over many fine classes is genuinely hard; the
  *relative* raw-vs-recon comparison under an identical protocol is the valid signal.
- **p = 512 is downstream-lossless.** Δf1 ≈ 0 at every k1 (|Δ/σ| < 1), and retention is
  consistently ≥ 1.0 — VQ mildly *denoises* the embeddings.
- **p = 1024 has a small but rock-solid loss.** Δf1 ≈ +0.017 (~8% relative) at Δ/σ = 6–7.5
  on Austria (sd ≈ 0.002 over 8.2M pixels — not noise). One codebook covering ~1M pixels is
  coarser per pixel than four covering 512² each; this is the per-pixel penalty that R²
  (variance-weighted) hid but the classifier feels. Cumbria's 4-tile noise could not
  resolve it — Austria's scale could.
- **k1 is irrelevant to F1 within a tile size**, so k1 = 20 (most compressible) is safe.

---

## 8. The Pareto and the final recommendation

Joining index-compression bytes/pixel with downstream retention gives the frontier in
`figures/phase4_pareto.png`. Only two points are non-dominated, both at k1 = 20:

| configuration | bytes/px | × int8 | × fp32 | downstream |
|---|---:|---:|---:|---|
| **p = 512, k1 = 20, k2 = 256** | **1.38** | **93×** | **372×** | **lossless (Δf1 ≈ 0)** |
| p = 1024, k1 = 20, k2 = 256 | 1.27 | 101× | 403× | −8% relative F1 (significant) |

(Bytes/px with the stage-1 plane gzip-coded; the frontier's membership is
encoder-invariant — switching RLE→gzip rescales the compression axis but does not change
which configs are non-dominated.) p = 1024 buys ~8% more compression for ~8% relative accuracy — a poor trade for a
foundation-model product whose value is the embedding's downstream utility. **We recommend
p = 512, k1 = 20, k2 = 256.** (p = 1024 aligns with the GeoTessera UTM tile, an
integration convenience; it is the right choice only if that alignment is judged worth the
8% accuracy cost — it was not, here.)

### Deployable format (engineering recommendation for GeoTessera)

Per 512×512 tile (one serving unit):

- **codebook1**: 20 × 128 bytes, **codebook2**: 256 × 128 bytes (int8 prototypes) —
  ≈ 0.14 bytes/pixel amortized.
- **idx1 plane**: per-pixel stage-1 index (values 0–19), stored as a raw row-major byte
  plane and **gzip-compressed** (DEFLATE) — ≈ 0.24 bytes/pixel (zstd reaches ≈ 0.22).
- **idx2 plane**: per-pixel stage-2 index (values 0–255), **raw 1 byte/pixel** (full-rank,
  white; does not compress).
- **Total ≈ 1.38 bytes/pixel ≈ 93× smaller than int8 served embeddings** (372× over fp32),
  reconstructable to within downstream-classifier tolerance.

---

## 9. Limitations and future work

- **Absolute downstream F1 is low** (≈ 0.20) under spatial hold-out on fine-grained
  classes; the conclusion rests on *relative* retention, which is the correct quantity for
  a compression study but means we have not characterized absolute task ceilings.
- **Two datasets, one classifier (Random Forest), pixel-only features.** Spatial-context
  classifiers or other tasks could be more sensitive to the lost residual; worth a check
  before broad deployment.
- **k1 was swept on {20, 32, 64, 128}**; k1 < 20 may compress idx1 further at still-zero
  F1 cost and is worth probing.
- **Codebook training cost.** Per-tile k-means at k ≤ 256 on a 512² tile is sub-second on
  CPU (BLAS-GEMM assignment, sampled fit); k = 2048 was dropped as unnecessarily slow.
- The reconstruction proxy (R²) and the downstream metric **agreed on the split but
  disagreed on t** — a reminder that intrinsic reconstruction error is necessary but not
  sufficient, and the downstream task must remain the arbiter.

---

### Provenance

Results: `results/phase3/idx_v2_index_compression.parquet` (RLE/Hilbert/Morton bytes/px),
`results/phase3/idx_gzip_index_compression.parquet` (RLE-vs-gzip-vs-zstd base-plane
comparison + gzip-based totals, via `scripts/phase3_gzip_vs_rle.py`; four-biome
diagnostic, n = 4 tiles, seed 42 — pending promotion to the full bbox set),
`results/phase3/large_v1_large_recon.parquet` (R²),
`results/phase4/austria_downstream.parquet`, `results/phase4/cumbria_downstream.parquet`
(downstream F1). Figure: `figures/phase4_pareto.png`/`.pdf` via
`scripts/phase4_pareto.py` (now plotting gzip-based ratios). All Parquet outputs carry
git SHA, seed, timestamp, and config hash.

---

## Appendix A — Experiments run but not in the paper

The academic write-up (`docs/paper_vq_compression.tex`) reports only the
experiments on the critical path: metric selection (Phase 1–2), large-tile
reconstruction + index compression (Phase 3), and downstream F1 (Phase 4). Several
other runs were executed along the way and live in `results/`. They are recorded
here for completeness; most are either superseded by the headline runs or are
supporting diagnostics whose summary numbers (not full tables) reached the paper.

### A.1 Codebook effective-rank / spectrum study (`results/codebook_rank/`)

The paper quotes two headline numbers from this study — the stage-1 codebook has
effective dimension ≈ 5 and the stage-2 residual codebook ≈ 83 — but the full
analysis is larger and unpublished:

- **Global effective rank** (`codebook_rank_global_effrank.parquet`, 8 rows):
  participation ratio, entropy effective dimension, and the dims-to-90/95/99%
  energy thresholds for each codebook (stage c1 vs c2), in both **raw** and
  **mean-centered** modes, at p ∈ {32, 64}. Centering roughly doubles the stage-1
  effective dimension (entropy eff-dim ≈ 5.4 → 8.7), i.e. a large part of the
  stage-1 codebook's apparent low rank is a shared offset (the dim-0 mean ≈ 4.2
  seen in Phase 1) — worth a sentence if a reviewer asks whether the ~5-D claim is
  an artifact of the offset.
- **Full singular spectra** (`codebook_rank_global_spectrum.parquet`, 1024 rows):
  per-component singular value, energy fraction, and cumulative energy for each
  codebook/mode. The stage-1 spectrum is steep — the top component alone holds
  ≈ 60% of the energy, the top two ≈ 75% — which is the underlying evidence for the
  "few prototypes" claim. Could become a scree-plot figure in a longer version.
- **Per-tile effective-rank distributions** (`codebook_rank_per_tile_effrank.parquet`):
  participation-ratio mean/median/p10/p90 and median dims-to-95% over **5756 tiles
  (p=32)** and **1386 tiles (p=64)**. Stage-1 PR median ≈ 1.2 (dims95 ≈ 3); stage-2
  PR median ≈ 22 (dims95 ≈ 47). The large tile count here is the broadest sample in
  the whole study — but at small p.
- **Per-tile reconstruction percentiles** (`codebook_rank_per_tile_recon.parquet`):
  L2 and cosine error percentiles (p50/p90/p99/max) over the same thousands of
  small tiles. A second, independent confirmation of the L2-over-cosine story at a
  different scale from the Phase-2 sweep that the paper actually cites.

*Caveat:* this entire study is at **small tiles (p = 32, 64), k1 = k2 = 256** —
the pre-pivot regime. It supports the geometry argument but not the large-tile
recommendation.

### A.2 Small-tile pilot reconstruction sweeps (`results/phase3/phase3_pilot_*.csv`)

The original Phase-3 sweep at **p ∈ {16, 32}, k1,k2 ∈ {64,128,256}**, in both
**L2** (`phase3_pilot_l2.csv`) and **cosine** (`phase3_pilot_cos.csv`) variants
(18 configs each), plus a scaled variant (`phase3_pilot_s.csv`). These use the
**anchored-histogram reconstruction metric** (50 frozen bins + an overflow
fraction) that §4 of this note identifies as the wrong turn — the bin edges drift
run-to-run, so the numbers are not comparable across runs. Retained only as the
record of the superseded methodology; **do not** cite these figures.

### A.3 16-bit interleaved index packing (`results/phase3/phase3_16bit*.csv`)

A sweep at **p ∈ {32, 64, 128}, (k1,k2) ∈ {(64,1024),(128,512)}** built around the
idea of a single bit-packed 16-bit (idx1,idx2) word per pixel. This is the
experiment that established (negatively) the result the paper now states as a
design rule: a 16-bit interleaved index **cannot be run-length-encoded**, because
the white idx2 destroys every run. The finding survived into the paper as the
"store separate byte planes" decision; the raw sweep did not.

### A.4 Sub-byte k1 < k2 sweep (`results/phase3/phase3_k1lt_k2*.csv`)

A sweep at **p ∈ {32, 64}, k1 ∈ {16,32,64}, k2 ∈ {128,256,512}** probing
configurations where the base codebook is smaller than the residual codebook (18
configs, L2 + scaled). Superseded by the locked grid (k2 = 256 fixed, k1 swept)
once the byte-plane decision made k2 = 256 the obvious operating point. Contains
the only k1 = 16 data point collected — relevant to the open question of whether
k1 < 20 compresses idx1 further at zero F1 cost (§9), though at small p.

### A.5 Early per-dimension pool statistics (`results/pool_a_stats.parquet`)

A 128-row per-dimension mean/variance table from 2026-05-25, superseded by
`results/phase1/per_dim_stats.parquet` (the version the paper cites). Numbers
agree (dim-0 mean ≈ 4.15); kept only as the earlier provenance.

### A.6 Smoke tests (`*smoke*.parquet`)

`results/phase3/smoke_large_recon.parquet` (2 tiles) and
`results/phase4/cumbria_smoke_downstream.parquet` (1 config, p=1024) are pipeline
smoke tests, not experiments; ignore for any reported result.

*Summary:* the paper omits (i) the full codebook spectrum/rank tables, (ii) the
superseded small-tile pilot sweeps and their anchored-histogram metric, (iii) the
16-bit-packing and (iv) k1<k2 exploratory sweeps, and (v) early/ smoke artifacts.
None changes a headline conclusion; A.1 is the only one with publishable
standalone value (a geometry figure for a longer version).
