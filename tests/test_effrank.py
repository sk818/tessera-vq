"""Unit tests for tessera_vq.effrank on synthetic codebooks with known rank.

Effective-rank metrics are cross-checked against hand-computable limits: a rank-1
matrix has participation ratio ~1, an orthonormal (white) matrix has it ~= dim,
and the streaming Gram accumulator must reproduce a direct ``C^T C`` SVD.
"""

from __future__ import annotations

import numpy as np

from tessera_vq.effrank import (
    GramAccumulator,
    effrank_metrics,
    energy_eigvals,
    per_tile_effrank_batch,
    per_tile_summary,
    recon_tail_summary,
    spectrum_rows,
)


def test_recon_tail_summary_percentiles_and_bad_tile_fraction() -> None:
    """Percentiles are ordered and the bad-tile fraction tracks a planted heavy tail."""
    errs = np.concatenate(
        [np.full(990, 1.0, np.float32), np.full(10, 100.0, np.float32)]  # 1% pathological tiles
    )
    s = recon_tail_summary(errs)
    assert s["n_tiles"] == 1000.0
    assert s["p50"] <= s["p90"] <= s["p99"] <= s["max"]
    assert s["max"] == 100.0
    # 10 of 1000 tiles are far above the median -> ~1% flagged at both thresholds.
    assert abs(s["frac_gt_5x_median"] - 0.01) < 1e-9
    assert abs(s["frac_gt_2x_median"] - 0.01) < 1e-9


def test_recon_tail_summary_empty_is_zeroed() -> None:
    """Empty error array returns an all-zero summary rather than crashing."""
    s = recon_tail_summary(np.zeros(0, np.float32))
    assert s["n_tiles"] == 0.0
    assert s["frac_gt_5x_median"] == 0.0


def test_rank_one_matrix_has_unit_participation_ratio() -> None:
    """A rank-1 codebook concentrates all energy in one component."""
    rng = np.random.default_rng(0)
    u = rng.standard_normal((64, 1)).astype(np.float32)
    v = rng.standard_normal((1, 128)).astype(np.float32)
    eig = (np.linalg.svd(u @ v, compute_uv=False) ** 2).astype(np.float64)
    m = effrank_metrics(eig)
    assert abs(m["participation_ratio"] - 1.0) < 1e-3
    assert m["dims_90"] == 1
    assert m["dims_99"] == 1


def test_orthonormal_rows_give_full_effective_rank() -> None:
    """An orthonormal set of 128 vectors is maximally isotropic: PR == dim."""
    q, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((128, 128)))
    eig = (np.linalg.svd(q.astype(np.float32), compute_uv=False) ** 2).astype(np.float64)
    m = effrank_metrics(eig)
    assert abs(m["participation_ratio"] - 128.0) < 1e-2
    assert m["entropy_eff_dim"] > 120.0
    assert m["dims_90"] >= 110


def test_gram_accumulator_matches_direct_svd() -> None:
    """Streaming C^T C in two blocks reproduces the full-matrix singular values."""
    rng = np.random.default_rng(2)
    c = rng.standard_normal((200, 128)).astype(np.float32)
    acc = GramAccumulator(128)
    acc.update(c[:80])
    acc.update(c[80:])
    eig_stream = energy_eigvals(acc, centered=False)
    eig_direct = np.sort(np.linalg.svd(c, compute_uv=False) ** 2)[::-1]
    assert np.allclose(eig_stream, eig_direct, rtol=1e-4, atol=1e-3)
    assert acc.count == 200


def test_centering_removes_a_shared_mean_direction() -> None:
    """A large shared offset inflates raw rank but vanishes after centring."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((300, 128)).astype(np.float32)
    shifted = base + 50.0  # big common mean vector
    acc = GramAccumulator(128)
    acc.update(shifted)
    pr_raw = effrank_metrics(energy_eigvals(acc, centered=False))["participation_ratio"]
    pr_cen = effrank_metrics(energy_eigvals(acc, centered=True))["participation_ratio"]
    assert pr_raw < pr_cen  # the offset dominates one direction until removed


def test_per_tile_batch_and_summary_shapes_and_values() -> None:
    """Batched per-tile SVD returns one PR per tile and summary percentiles are ordered."""
    rng = np.random.default_rng(4)
    codebooks = rng.standard_normal((10, 256, 128)).astype(np.float32)
    pr, d95 = per_tile_effrank_batch(codebooks)
    assert pr.shape == (10,)
    assert d95.shape == (10,)
    assert np.all(pr > 1.0)  # random codebooks are not rank-1
    s = per_tile_summary(pr, d95)
    assert s["n_tiles"] == 10.0
    assert s["pr_p10"] <= s["pr_median"] <= s["pr_p90"]


def test_empty_inputs_are_degenerate_not_crashing() -> None:
    """Zero spectra / empty stacks return zeros rather than dividing by zero."""
    assert effrank_metrics(np.zeros(0))["participation_ratio"] == 0.0
    assert spectrum_rows(np.zeros(5)) == []
    pr, d95 = per_tile_effrank_batch(np.zeros((0, 256, 128), np.float32))
    assert pr.size == 0
    assert per_tile_summary(pr, d95)["n_tiles"] == 0.0
