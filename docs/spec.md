# Tessera VQ compression study — Claude Code execution spec

**Supervisor:** S. Keshav
**Executor:** Claude Code (claude-opus-4-7)
**Project (GitHub):** CAC
**Repository:** `tessera-vq`, under the supervisor's GitHub org or account, private
**Mode of operation:** phased, with mandatory **HALT** after each phase. Do not begin Phase N+1 until the supervisor has approved Phase N.

This spec is the work plan. Stable conventions (style, git workflow, things never to do) live in `CLAUDE.md` at the repo root, alongside `config.yaml` which holds all paths and parameters. Read `CLAUDE.md` and `config.yaml` first, then this file.

---

## 0. How to read this spec

- Every phase has **Inputs**, **Tasks** (numbered and atomic), **Outputs**, **Validation**, and a **HALT** point.
- Tasks are written so that each one is a single bounded change with a verifiable result. Run validation after each task, not just at the end of a phase.
- When the spec says **HALT**, stop, summarise progress in chat (not in a file), and wait for explicit "proceed" before continuing.
- When the spec says **CHECK WITH SUPERVISOR**, do not guess — pause and ask. This applies especially to: data paths, split protocols, anything requiring research judgement.
- `<FILL IN>` markers in `config.yaml` are values only the supervisor can provide. Read them from `config.yaml`, never hard-code.
- Commit to git at the end of each task with a message like `phase-2/task-3: implement Wasserstein-1 projection`.
- Push to GitHub at every HALT, never between (unless the supervisor asks).

---

## 1. Phase 0 — Repository bootstrap and GitHub setup

**Inputs:** the two reference files in the supervisor's hand-off: `CLAUDE.md` and `config.yaml`. The supervisor will create the CAC GitHub Project and provide the org/user name for the repository.

### Tasks

1. Ask the supervisor: confirm GitHub org/user (to populate `config.yaml::github.org`), confirm local path for the repo (suggest `~/code/tessera-vq`), confirm authentication for `gh` CLI is in place (`gh auth status`). **HALT** until confirmed.
2. Initialise the local repository:
   ```bash
   mkdir -p ~/code/tessera-vq && cd ~/code/tessera-vq
   git init -b main
   ```
3. Copy the supervisor-provided `CLAUDE.md` and `config.yaml` into the repo root. Do not modify them. If they are missing, **HALT** and request them.
4. Create the directory structure:
   ```
   tessera-vq/
   ├── CLAUDE.md                       (provided)
   ├── config.yaml                     (provided)
   ├── README.md
   ├── pyproject.toml
   ├── .gitignore
   ├── .pre-commit-config.yaml
   ├── docs/
   │   └── spec.md                     (this file, copied in)
   ├── .github/
   │   ├── workflows/
   │   │   └── ci.yml
   │   └── ISSUE_TEMPLATE/
   │       └── phase.md
   ├── tessera_vq/
   │   ├── __init__.py
   │   ├── data.py
   │   ├── quantize.py
   │   ├── morton.py
   │   ├── hilbert.py
   │   ├── entropy.py
   │   ├── metrics.py
   │   ├── probes.py
   │   └── io_utils.py
   ├── scripts/
   │   ├── phase1_isotropy.py
   │   ├── phase2_reconstruction.py
   │   ├── phase3_index_compression.py
   │   ├── phase4_downstream.py
   │   └── phase5_pareto.py
   ├── notebooks/
   │   ├── 01_isotropy.ipynb
   │   ├── 02_reconstruction.ipynb
   │   ├── 03_index_compression.ipynb
   │   ├── 04_downstream.ipynb
   │   └── 05_pareto.ipynb
   ├── tests/
   │   ├── test_morton.py
   │   ├── test_hilbert.py
   │   ├── test_quantize.py
   │   ├── test_metrics.py
   │   └── fixtures/
   ├── results/                        (git-ignored)
   ├── figures/                        (git-ignored except headline figures)
   └── logs/                           (git-ignored)
   ```
5. Populate `pyproject.toml` from Appendix A.
6. Populate `.gitignore` from Appendix B.
7. Populate `.pre-commit-config.yaml` from Appendix C and run `pre-commit install`.
8. Populate `.github/workflows/ci.yml` from Appendix D.
9. Populate `.github/ISSUE_TEMPLATE/phase.md` from Appendix E.
10. Write a one-paragraph `README.md` describing the project; link to `CLAUDE.md`, `config.yaml`, and `docs/spec.md`.
11. Initial commit:
    ```bash
    git add .
    git commit -m "phase-0/task-11: bootstrap repository structure"
    ```
12. Create the GitHub repository and push:
    ```bash
    gh repo create <org>/tessera-vq --private --source=. --remote=origin --push
    ```
    Replace `<org>` with `config.yaml::github.org`. If the supervisor prefers to create the repo manually via the web UI, **HALT** and request the URL, then `git remote add origin <url> && git push -u origin main`.
13. Create one GitHub issue per phase (1–6), each titled `[Phase N] <short description>`, using the `phase.md` issue template. Add each to the CAC project board with status "Todo".

### Outputs

- A clean repo on GitHub at `<org>/tessera-vq`, linked to the CAC project, with six phase issues created.
- Green CI on the bootstrap commit.

### Validation

- `uv run pytest` → 0 tests, exit code 0.
- `uv run mypy tessera_vq scripts` → clean (initially trivial — all stubs).
- `uv run ruff check .` → clean.
- The GitHub Actions CI run on the first push is green.
- `gh issue list` shows six phase issues.

### **HALT** — confirm the repo URL and project linkage with the supervisor before starting Phase 1. Move the Phase 1 issue to "In progress" on the CAC project board.

---

## 2. Phase 1 — Data loaders and smoke tests

**Inputs:** populated `config.yaml` with real paths.

### Tasks

1. Implement `tessera_vq/data.py` with three loader functions:
   - `load_tile(tile_id: str) -> np.ndarray` — returns `(H, W, 128)` float32.
   - `iter_pool_a(n_tiles: int = 1000, seed: int = 42, biome_strata: bool = True) -> Iterator[np.ndarray]` — yields tiles from the diagnostic pool.
   - `load_downstream(task: Literal["pastis", "treesatai", "borneo", "austrian_crops", "haweswater"]) -> tuple[X_train, y_train, X_val, y_val, X_test, y_test]` — uses split definitions from Frank's harness at `config.yaml::paths.frank_eval_repo`. **CHECK WITH SUPERVISOR** for each task before implementing.
2. Write a smoke test that loads one tile of each shape (16, 64, 256) and asserts dtype, shape, finite values, reasonable norm.
3. Add `tests/fixtures/` with three tiny synthetic tiles (4×4, 8×8, 16×16) and 50-dim embeddings so unit tests run without real data.
4. Implement biome stratification for Pool A. **CHECK WITH SUPERVISOR** on the biome layer source.
5. Implement `tessera_vq/io_utils.py::write_parquet_with_provenance` per `CLAUDE.md`.

### Outputs

- `tessera_vq/data.py` working for all five downstream tasks.
- Pool A iteration verified end-to-end on 10 tiles.

### Validation

- `uv run pytest tests/test_data.py` — all green.
- `python -c "from tessera_vq.data import iter_pool_a; ts = list(iter_pool_a(n_tiles=10)); print([t.shape for t in ts])"` produces ten tiles of expected shape.

### **HALT** — show the supervisor sample shapes, biome distribution of the 10 test tiles, confirmation that downstream loaders match Frank's expectations. Close Phase 1 issue on supervisor approval; open Phase 2.

---

## 3. Phase 2 — Isotropy diagnostics

**Inputs:** Phase 1 loaders working.

### Tasks

1. Implement `tessera_vq/metrics.py::epps_pulley(samples_1d, mu=0, sigma=1) -> float` and `shapiro_wilk(samples_1d) -> tuple[stat, p]`. Use `scipy.stats.shapiro` for the latter; implement Epps–Pulley from the formula (it's not in scipy). Cross-check Epps–Pulley against known-Gaussian and known-non-Gaussian samples in tests.
2. Write `scripts/phase1_isotropy.py` (note: filename has "phase1" because it's the first analytical phase; Phase 0 is bootstrap):
   - Draws 10,000 embeddings from Pool A.
   - Standardises per-dimension using Pool A statistics (saved to `results/pool_a_stats.parquet`).
   - Samples 200 random unit-norm directions in ℝ¹²⁸.
   - For each direction, computes Shapiro–Wilk p-value and Epps–Pulley statistic.
   - Saves per-direction results to `results/phase1/projection_normality.parquet`.
   - Computes summary: fraction of directions rejecting at α=0.01.
3. Save per-dimension mean and variance to `results/phase1/per_dim_stats.parquet`. Flag any dimensions with variance below 0.01 of the median (near-collapsed).
4. Notebook `01_isotropy.ipynb` produces:
   - Histogram of Shapiro–Wilk p-values.
   - Histogram of Epps–Pulley statistics.
   - Bar chart of per-dimension variance (log scale).
   - Save as `figures/phase1_*.png`.

### Outputs

- `results/phase1/projection_normality.parquet`
- `results/phase1/per_dim_stats.parquet`
- `results/pool_a_stats.parquet`
- `figures/phase1_*.png`

### Validation

- Unit test: Epps–Pulley on `np.random.standard_normal(10000)` returns statistic close to expected null value.
- Unit test: Epps–Pulley on `np.random.exponential(1, 10000)` returns clearly larger statistic.
- Standardised Pool A embeddings: per-dim mean ≈ 0, var ≈ 1.

### **HALT** — report:
- Rejection fraction at α=0.01 for both tests.
- Number of near-collapsed dimensions.
- The figures.
- One-paragraph interpretation: prioritise L2 (≥80% non-rejection) or run both metrics in Phase 3.

Wait for the supervisor's metric decision before Phase 3.

---

## 4. Phase 3 — Reconstruction quality vs k

**Inputs:** Phase 2 complete; metric priority decided.

### Tasks

1. Implement `tessera_vq/quantize.py::quantize_tile(tile, k, distance, seed) -> tuple[codebook, indices]`.
   - `sklearn.cluster.KMeans(n_init=10, random_state=seed)` for k ≤ 64.
   - `sklearn.cluster.MiniBatchKMeans(batch_size=1024, n_init=10, random_state=seed)` for k > 64.
   - For `distance="cosine"`, L2-normalise inputs before clustering.
   - Returns `codebook: (k, 128) float32`, `indices: (H, W) uint8 or uint16`.
2. Implement `tessera_vq/metrics.py::wasserstein1_random_projections(X, Y, n_proj, seed) -> float`.
3. Write `scripts/phase2_reconstruction.py`:
   - Iterate over the grid: `tile_size ∈ {16, 64, 256}`, `k ∈ {2, 4, 8, 16, 32, 64, 128, 256}`.
   - Test both distance metrics at k=16 and k=64; propagate the winner.
   - For each (tile_size, k, distance), process 1000 tiles from Pool A: run k-means, reconstruct, compute cosine, L2 (raw and standardised), per-dim error, Wasserstein-1.
   - Save **quantiles only** (10/50/90/99) per tile plus Wasserstein-1 to `results/phase2/reconstruction.parquet`. Per-pixel errors would balloon the file.
4. Notebook `02_reconstruction.ipynb`:
   - Cosine-vs-k plot (10/50/90 lines) per tile size.
   - Same for L2.
   - Same for Wasserstein-1.
   - Table: smallest k achieving 95th-percentile cosine < 0.1.
5. Sanity-check script `scripts/sanity_phase2.py`: verify errors → 0 as k → tile_area.

### Outputs

- `results/phase2/reconstruction.parquet`
- `figures/phase2_cosine_vs_k.png`, `phase2_l2_vs_k.png`, `phase2_wasserstein_vs_k.png`
- `results/phase2/distance_metric_comparison.parquet`

### Validation

- Sanity script passes (errors collapse to 0 at k=tile_area).
- Cosine and L2 monotonically non-increasing in k.
- Compression ratios at k ∈ {16, 64, 256} match back-of-envelope numbers within 5%.

### **HALT** — report all three plots and the distance-metric winner. Update `config.yaml::grid.clustering_distance_chosen`.

---

## 5. Phase 4 — Index map compression

**Inputs:** Phase 3 codebooks and index maps cached.

### Tasks

1. Implement `tessera_vq/morton.py::encode_morton2d` and `decode_morton2d` using bit-interleaving. Cross-check against `pymorton` on a 16×16 grid.
2. Implement `tessera_vq/hilbert.py` using `hilbertcurve` as reference; vectorise with numpy and cross-check.
3. Implement `tessera_vq/entropy.py`:
   - `rle_encode`, `rle_decode`
   - `zstd_compress(b, level=19)` using `zstandard`
   - Roundtrip tests for all combinations.
4. Write `scripts/phase3_index_compression.py` covering pipelines A–G:

   | # | Pipeline |
   | --- | --- |
   | A | Raw bit-packed indices |
   | B | Row-major + RLE |
   | C | Z-order + RLE |
   | D | Hilbert + RLE |
   | E | Raw bit-packed + zstd(19) |
   | F | Z-order + RLE + zstd |
   | G | Hilbert + RLE + zstd |

   Save bytes/pixel (excluding and including codebook amortisation) to `results/phase3/compression.parquet`. Include per-biome breakdown.
5. Notebook `03_index_compression.ipynb`:
   - Bar chart: bytes/pixel by pipeline, faceted by k.
   - Heatmap: bytes/pixel by biome × pipeline at k=16.

### Outputs

- `results/phase3/compression.parquet`
- `figures/phase3_*.png`

### Validation

- All roundtrips bit-exact.
- Morton and Hilbert agree with reference implementations on the fixture.
- Single-biome forest tiles: Z-order+RLE beats raw bit-packed by ≥5×.

### **HALT** — report bytes/pixel table and per-biome breakdown.

---

## 6. Phase 5 — Downstream linear probes

**Inputs:** Phase 4 complete. Frank's evaluation harness paths confirmed.

### Tasks

1. Implement `tessera_vq/probes.py::train_linear_probe(X_train, y_train, X_val, y_val, task_type)`. `LogisticRegression` (classification) or `Ridge` (regression). Tune `C` (or `alpha`) on validation over a log grid (see `config.yaml::phase4.probe_C_grid`).
2. **CHECK WITH SUPERVISOR** that this probe protocol matches Frank's harness exactly. Do not deviate silently.
3. For each compression config in `config.yaml::phase4.configs` × each downstream task:
   - Compute reconstructed embeddings from cached codebooks.
   - Train probe-raw (train on raw, eval on reconstructed).
   - Train probe-reconstructed (train and eval on reconstructed).
   - Record metric values to `results/phase4/downstream.parquet` with columns `task`, `config`, `probe_type`, `metric_name`, `metric_value`, `per_class_metrics` (JSON).
4. Script `scripts/phase4_downstream.py` orchestrates the 8 × 5 × 2 = 80 probe trainings with joblib parallelism. Cache intermediates so re-runs don't redo finished work.
5. Notebook `04_downstream.ipynb`:
   - Table: probe-raw vs probe-reconstructed accuracy per (task, config).
   - Plot: probe-raw gap vs effective bytes/pixel.

### Outputs

- `results/phase4/downstream.parquet`
- `figures/phase4_*.png`

### Validation

- Raw float32 baseline matches Frank's harness within 0.5 percentage points. If not, the loader is wrong — stop and debug.
- Probe-raw degrades monotonically with compression aggressiveness (within noise).

### **HALT** — first Pareto plot. Expect surprises here; expect to iterate.

---

## 7. Phase 6 — Pareto plots, per-class failure modes, writeup stubs

**Inputs:** Phases 1–5 complete.

### Tasks

1. `scripts/phase5_pareto.py`: join compression with downstream on configuration, compute effective bytes/pixel including codebook amortisation, save to `results/phase5/pareto.parquet`.
2. Notebook `05_pareto.ipynb`:
   - One Pareto plot per task.
   - One aggregate plot with rank-normalised accuracy.
   - Per-class F1 table for k=16 VQ vs raw baseline per task. Flag classes with ≥10 point F1 drops.
3. Draft `writeup/tech_note.qmd` (Quarto). Structure: motivation → diagnostics → method → Pareto plot → per-class failures → takeaways. All figures and tables filled in; prose sections left as stubs.
4. Draft `writeup/engineering_memo.md` (half page). Bullet structure: recommendation (blank — supervisor's call), rationale, recommended k, regimes where VQ is not advised, GeoTessera roadmap dependencies.

### Outputs

- `results/phase5/pareto.parquet`
- `figures/phase5_pareto_{task}.png`, `phase5_pareto_aggregate.png`
- `writeup/tech_note.qmd`
- `writeup/engineering_memo.md`

### Validation

- Every figure referenced in `tech_note.qmd` exists in `figures/`.
- `quarto render writeup/tech_note.qmd` succeeds.

### **HALT** — final supervisor review before any external sharing.

---

## 8. Phase 7 — Stretch goals (only with explicit authorisation)

- arXiv:2405.12497 learned codebook baseline.
- Adaptive k per tile via gap statistic or BIC.
- Hilbert vs Morton performance benchmark.

Do not start without supervisor "go".

---

## Appendix A — `pyproject.toml`

```toml
[project]
name = "tessera-vq"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.13",
    "scikit-learn>=1.5",
    "pandas>=2.2",
    "pyarrow>=17",
    "matplotlib>=3.9",
    "seaborn>=0.13",
    "zstandard>=0.22",
    "hilbertcurve>=2.0",
    "pymorton>=1.0",
    "tqdm>=4.66",
    "pyyaml>=6.0",
    "joblib>=1.4",
    "rasterio>=1.3",
    "xarray>=2024.6",
    "torch>=2.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "mypy>=1.10",
    "ruff>=0.5",
    "pre-commit>=3.7",
    "jupyterlab>=4.2",
    "quarto-cli>=1.5",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "RET", "PL"]

[tool.mypy]
strict = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=tessera_vq --cov-report=term-missing"
```

## Appendix B — `.gitignore`

```
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.venv/
.uv/

# Outputs
results/
logs/
figures/*
!figures/.gitkeep
!figures/headline/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Local config overrides
.env
.env.local
```

## Appendix C — `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies:
          - numpy
          - pandas-stubs
          - types-PyYAML
  - repo: local
    hooks:
      - id: pytest-fast
        name: pytest (fast tests only)
        entry: uv run pytest -m "not slow"
        language: system
        pass_filenames: false
        stages: [pre-push]
```

## Appendix D — `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      - name: Set up Python
        run: uv python install 3.11
      - name: Install dependencies
        run: uv pip install -e ".[dev]"
      - name: Lint
        run: |
          uv run ruff check .
          uv run ruff format --check .
      - name: Type check
        run: uv run mypy tessera_vq scripts
      - name: Test
        run: uv run pytest -m "not slow"
```

## Appendix E — `.github/ISSUE_TEMPLATE/phase.md`

```markdown
---
name: Phase tracking
about: Track a phase of the Tessera VQ study
title: '[Phase N] '
labels: phase
---

## Phase
N

## Goal
<one-line summary from docs/spec.md>

## Tasks
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

## HALT-point outputs required
- <e.g., cosine-vs-k plot>
- <e.g., rejection fraction>

## Supervisor sign-off
- [ ] Reviewed
- [ ] Approved to proceed to Phase N+1
```

---

## Appendix F — Wall-time expectations

| Phase | Wall time | HALT-point output |
| --- | --- | --- |
| 0. Bootstrap + GitHub | 1–2 h | green CI, repo + issues live |
| 1. Data loaders | 1–2 h | 10-tile smoke test report |
| 2. Isotropy | 30 min | rejection fraction + figures |
| 3. Reconstruction | 4–8 h | cosine/L2/Wasserstein plots |
| 4. Index compression | 1–2 h | bytes/pixel table |
| 5. Downstream probes | 4–12 h | first Pareto plot |
| 6. Writeup | 2 h | tech note + memo with stubs |

If a phase exceeds 2× its estimate, **HALT** and report.
