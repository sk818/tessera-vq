# Tessera VQ — project conventions

This file is read by Claude Code at the start of every session. It defines the conventions that hold regardless of which phase of work is active. The phase-by-phase work plan lives in `docs/spec.md`.

---

## Project context

This repository implements and evaluates **per-tile vector quantisation (VQ) for Tessera embedding compression**. The work is supervised by S. Keshav and is part of the CAC project. The output is (a) a tech note evaluating VQ against the Robinson & Corley compression frontier, and (b) an engineering recommendation for GeoTessera.

Tessera is a self-supervised foundation model producing 128-d embeddings per ~10 m × 10 m pixel of Earth's surface. The compression idea exploits *spatial autocorrelation*: within a tile, only a handful of land-cover prototypes typically appear, so a small per-tile codebook plus an index map can compress 99%+ of the bytes with limited downstream accuracy loss.

---

## Mode of operation

This project runs phased, with mandatory **HALT** points between phases. After each phase:

1. Summarise progress to the supervisor in chat — not in a file.
2. Quote the headline numbers / figures the spec asks for.
3. Wait for an explicit "proceed" before starting the next phase.

Never start a new phase autonomously. Never silently change scope.

When the spec or this file says **CHECK WITH SUPERVISOR**, stop and ask. This applies especially to: data paths, split protocols, evaluation metrics, anything requiring research judgement.

---

## Code style

- Python 3.11+. Type hints throughout. No `Any` without justification.
- Functions ≤ 50 lines. Files ≤ 400 lines. Refactor when crossed.
- All scripts callable as `python scripts/phaseN_X.py --help`.
- All scripts accept `--seed` (default 42) and propagate it to numpy, random, torch, sklearn.
- Logging via `logging` module; INFO to stdout, DEBUG to `logs/{phase}/{script}_{timestamp}.log`.

### Tooling

- Environment: `uv` (`uv venv`, `uv pip install -e ".[dev]"`).
- Lint: `ruff check`. Format: `ruff format`. Type check: `mypy --strict`. Test: `pytest`.
- Pre-commit hooks must pass before every commit.

---

## Data flow

- **Raw embeddings:** read-only. Never modified, never copied unless the spec explicitly says so.
- **Intermediate results:** Parquet files under `results/{phaseN}/`. Never edited by hand. Never re-written from notebooks.
- **Notebooks:** plotting only. If a notebook computes anything more than a one-line aggregation, that computation belongs in a script and the result belongs in Parquet.
- **Figures:** PNG and PDF under `figures/`. Named to match the phase that produced them.

### Provenance columns

Every Parquet output must include these columns:

- `git_sha` — short SHA of the commit that produced the file
- `seed` — the seed that was used
- `timestamp_utc` — ISO 8601, UTC
- `config_hash` — SHA-256 of the resolved `config.yaml` contents

A helper `tessera_vq.io_utils.write_parquet_with_provenance` should be used for all writes.

---

## Determinism

- Default seed is 42 throughout. Sub-seeds (sampling, k-means init, random projections) defined in `config.yaml::seeds`.
- Any source of non-determinism (multi-threading, GPU non-determinism, etc.) must be either eliminated or explicitly documented in the script's docstring.

---

## Testing

- `uv run pytest` must pass on every commit. Coverage target: 80% on the `tessera_vq/` package (scripts excluded).
- Unit tests use synthetic fixtures under `tests/fixtures/`. No unit test depends on real Tessera data.
- For functions that interact with external packages (e.g., Morton encoding, Hilbert curve, Epps–Pulley), cross-check against a reference implementation in tests.

---

## Git workflow

- One **task** per commit. Commit messages follow the form: `phase-N/task-M: <imperative summary>`. Example: `phase-3/task-2: implement RLE encoder for index maps`.
- Work on `main` for routine commits; use feature branches `phase-N-experiment-<name>` for exploratory work that may not land.
- Push to GitHub after every **HALT** point (so the supervisor sees the state during review). Do not push between HALT points unless the supervisor asks.
- Never force-push to `main`.

### GitHub Issues

- Each phase has a corresponding tracking issue, prefixed `[Phase N]`. Comment progress on the issue at each task boundary.
- Use the CAC GitHub Project board to track issue status.
- Close the phase issue at the HALT point only after the supervisor has approved.

---

## Things to never do without asking

1. Modify split definitions, evaluation protocols, or metrics that Frank's harness uses.
2. Change the parameter grid (tile sizes, k values) after Phase 3 starts.
3. Add a new downstream task.
4. Run anything that takes more than 2 hours of wall time without confirming budget.
5. Train probes longer than the protocol specifies "to see if it helps".
6. Push to a public remote or change repository visibility.
7. Make architectural changes outside the spec to "improve" things.

---

## Files in this repo

- `docs/spec.md` — the phase-by-phase execution plan. Read this on first session and after every HALT.
- `config.yaml` — paths, seeds, parameter grid. All scripts read from here; never hard-code.
- `CLAUDE.md` — this file. Stable conventions.
- `README.md` — short human-facing description.
- `pyproject.toml` — dependencies and tool config.
- `tessera_vq/` — library code.
- `scripts/` — entry points, one per phase.
- `notebooks/` — plotting only.
- `tests/` — unit tests.
- `results/`, `figures/`, `logs/` — outputs (git-ignored except for headline tables explicitly checked in by the supervisor).

---

## When in doubt

Ask. The supervisor would much rather answer a one-line question than discover at HALT that the phase ran with the wrong assumption.
