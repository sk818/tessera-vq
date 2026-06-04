"""Effective-rank / SVD and reconstruction-tail diagnostics for RVQ codebooks.

Pure-numpy helpers behind ``scripts/codebook_rank_analysis.py``. The question
they answer: how low-rank are the per-tile RVQ codebooks, i.e. could a codebook
``C (k, 128)`` be stored as ``U (k, r) @ V (r, 128)`` with ``r << 128`` at
acceptable loss?

Two complementary views:

- **Global** -- accumulate the Gram matrix ``C^T C`` (128x128) over *all* per-tile
  codebook vectors via :class:`GramAccumulator`, then read its eigenvalues
  (= squared singular values of the stacked codebook). This is the rank of the
  subspace a single *shared* basis ``V`` would have to span. Both raw and
  column-mean-centred variants are reported (centring corresponds to storing one
  global 128-d mean vector, then factoring the residual).
- **Per tile** -- the SVD of each individual ``C (k_eff, 128)``. Low per-tile rank
  is the compression premise (few land-cover prototypes per tile), but each tile
  spans a *different* subspace, so a single shared ``V`` cannot exploit it; this
  view quantifies that per-tile rank distribution directly.

"Effective rank" is read off the energy spectrum ``lambda_i`` (eigenvalues =
``sigma_i^2``) three ways: participation ratio ``(sum l)^2 / sum l^2``, entropy
effective dim ``exp(-sum p_i ln p_i)``, and the count of components needed for
90 / 95 / 99% cumulative energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

ENERGY_THRESHOLDS: tuple[float, ...] = (0.90, 0.95, 0.99)


@dataclass
class GramAccumulator:
    """Streaming accumulator of ``C^T C``, the row sum, and the row count.

    Memory is bounded at ``dim x dim`` regardless of how many codebook vectors
    are folded in, so the whole 100-bbox sweep can stream one window at a time.
    """

    dim: int
    gram: npt.NDArray[np.float64] = field(init=False)
    sum_vec: npt.NDArray[np.float64] = field(init=False)
    count: int = 0

    def __post_init__(self) -> None:
        self.gram = np.zeros((self.dim, self.dim), dtype=np.float64)
        self.sum_vec = np.zeros(self.dim, dtype=np.float64)

    def update(self, mat: npt.NDArray[np.float32]) -> None:
        """Fold an ``(m, dim)`` block of codebook vectors into the running Gram."""
        if mat.size == 0:
            return
        m = mat.astype(np.float64, copy=False)
        self.gram += m.T @ m
        self.sum_vec += m.sum(axis=0)
        self.count += int(m.shape[0])


def energy_eigvals(acc: GramAccumulator, *, centered: bool) -> npt.NDArray[np.float64]:
    """Eigenvalues (descending, clipped >= 0) of the optionally-centred Gram matrix.

    Centred Gram is ``C^T C - n * mu mu^T`` with ``mu`` the column mean -- i.e. the
    spectrum of the data after subtracting a single shared 128-d mean vector.
    """
    g = acc.gram
    if centered and acc.count > 0:
        mu = acc.sum_vec / acc.count
        g = g - acc.count * np.outer(mu, mu)
    eig = np.linalg.eigvalsh(g)  # ascending, symmetric
    return np.clip(eig[::-1], 0.0, None)


def effrank_metrics(eigvals: npt.NDArray[np.float64]) -> dict[str, float]:
    """Participation ratio, entropy eff. dim, and 90/95/99% energy dims from eigenvalues.

    ``eigvals`` are the energy spectrum (``sigma_i^2``), assumed sorted descending
    and non-negative. Returns zeros for a degenerate (all-zero) spectrum.
    """
    total = float(eigvals.sum())
    zero_dims = {f"dims_{int(t * 100)}": 0 for t in ENERGY_THRESHOLDS}
    if total <= 0.0:
        return {"participation_ratio": 0.0, "entropy_eff_dim": 0.0, **zero_dims}
    pr = total**2 / float((eigvals**2).sum())
    p = eigvals / total
    nz = p[p > 0]
    entropy = float(np.exp(-(nz * np.log(nz)).sum()))
    cum = np.cumsum(eigvals) / total
    dims = {f"dims_{int(t * 100)}": int(np.searchsorted(cum, t) + 1) for t in ENERGY_THRESHOLDS}
    return {"participation_ratio": pr, "entropy_eff_dim": entropy, **dims}


def spectrum_rows(eigvals: npt.NDArray[np.float64]) -> list[dict[str, float]]:
    """Per-component scree rows: singular value, energy fraction, cumulative fraction."""
    total = float(eigvals.sum())
    if total <= 0.0:
        return []
    cum = np.cumsum(eigvals) / total
    sv = np.sqrt(eigvals)
    return [
        {
            "component_index": int(i),
            "singular_value": float(sv[i]),
            "energy_frac": float(eigvals[i] / total),
            "cum_energy_frac": float(cum[i]),
        }
        for i in range(eigvals.size)
    ]


def per_tile_effrank_batch(
    codebooks: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Vectorised per-tile participation ratio + 95%-energy dim over an ``(n, k, dim)`` stack.

    Uses the raw (un-centred) SVD of each codebook, since a per-tile mean would
    itself cost ``dim`` floats per tile to store. Returns ``(pr, dims95)`` arrays
    of length ``n``; an all-zero codebook yields ``pr = 0``.
    """
    if codebooks.size == 0:
        return np.zeros(0, np.float64), np.zeros(0, np.int64)
    sv = np.linalg.svd(codebooks.astype(np.float64, copy=False), compute_uv=False)
    eig = sv**2  # (n, min(k, dim))
    total = eig.sum(axis=1)
    safe = np.where(total > 0, total, 1.0)
    pr = np.where(total > 0, total**2 / (eig**2).sum(axis=1), 0.0)
    cum = np.cumsum(eig, axis=1) / safe[:, None]
    # First component index (1-based) at which cumulative energy reaches 95%.
    dims95 = (cum < 0.95).sum(axis=1) + 1  # noqa: PLR2004
    return pr.astype(np.float64), dims95.astype(np.int64)


def per_tile_summary(
    prs: npt.NDArray[np.float64], dims95: npt.NDArray[np.int64]
) -> dict[str, float]:
    """Aggregate per-tile participation ratios + 95%-dims into mean/median/p10/p90."""
    if prs.size == 0:
        return {
            "n_tiles": 0.0,
            "pr_mean": 0.0,
            "pr_median": 0.0,
            "pr_p10": 0.0,
            "pr_p90": 0.0,
            "dims95_median": 0.0,
        }
    return {
        "n_tiles": float(prs.size),
        "pr_mean": float(prs.mean()),
        "pr_median": float(np.median(prs)),
        "pr_p10": float(np.percentile(prs, 10)),
        "pr_p90": float(np.percentile(prs, 90)),
        "dims95_median": float(np.median(dims95)),
    }


_RECON_KEYS: tuple[str, ...] = (
    "n_tiles",
    "mean",
    "p50",
    "p90",
    "p95",
    "p99",
    "p999",
    "max",
    "frac_gt_2x_median",
    "frac_gt_5x_median",
)


def recon_tail_summary(errors: npt.NDArray[np.float32]) -> dict[str, float]:
    """Summarise a per-tile reconstruction-error distribution for the bad-tile question.

    The percentiles invert "serve the worst f% of tiles raw" into the error
    ceiling among the tiles still coded (e.g. ``p99`` is that ceiling if the worst
    1% are served raw). ``frac_gt_{2,5}x_median`` is a direct readout of how rare
    bad tiles are: a fat tail means the hybrid raw-passthrough scheme is costly,
    a thin one means it is cheap.
    """
    if errors.size == 0:
        return dict.fromkeys(_RECON_KEYS, 0.0)
    a = errors.astype(np.float64)
    med = float(np.median(a))
    return {
        "n_tiles": float(a.size),
        "mean": float(a.mean()),
        "p50": med,
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "p999": float(np.percentile(a, 99.9)),
        "max": float(a.max()),
        "frac_gt_2x_median": float((a > 2 * med).mean()) if med > 0 else 0.0,
        "frac_gt_5x_median": float((a > 5 * med).mean()) if med > 0 else 0.0,
    }
