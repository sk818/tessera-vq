"""Two-stage residual VQ for large tiles, built on the BLAS-GEMM k-means (WS-1).

Mirrors ``tessera_vq.sweep.rvq_quantize_tile`` but uses ``kmeans_fast`` so it scales
to t up to 1024. Stage 1 quantises the tile into ``k1`` base prototypes; stage 2
quantises the *residual* into ``k2`` prototypes. The reconstruction is
``codebook1[idx1] + codebook2[idx2]``. Euclidean only (RVQ discards magnitude
direction information at stage 1, so cosine is not meaningful here).

The codebooks and index maps are returned alongside the reconstruction because
WS-2 (index-map compression) consumes ``idx1`` (the spatially autocorrelated
stage-1 map) directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tessera_vq.codebook_codec import roundtrip_uint8
from tessera_vq.kmeans_fast import assign_blocked, kmeans_fit, quantize_tile_large


@dataclass(frozen=True)
class RVQResult:
    """Reconstruction plus the codebooks/index maps that produced it."""

    recon: npt.NDArray[np.float32]  # (H, W, C)
    codebook1: npt.NDArray[np.float32]  # (k1_eff, C)
    indices1: npt.NDArray[np.int32]  # (H, W)
    codebook2: npt.NDArray[np.float32]  # (k2_eff, C)
    indices2: npt.NDArray[np.int32]  # (H, W)


def rvq_reconstruct_large(
    tile: npt.NDArray[np.float32],
    k1: int,
    k2: int,
    *,
    seed: int = 42,
    n_iter: int = 25,
) -> RVQResult:
    """Two-stage residual VQ of an ``(H, W, C)`` tile; returns recon + codebooks/indices.

    Stage 2 uses ``seed + 1`` so its k-means++ draws differ from stage 1's.
    """
    cb1, idx1 = quantize_tile_large(tile, k1, seed=seed, n_iter=n_iter)
    recon1 = cb1[idx1]
    residual = (tile - recon1).astype(np.float32, copy=False)
    cb2, idx2 = quantize_tile_large(residual, k2, seed=seed + 1, n_iter=n_iter)
    recon = (recon1 + cb2[idx2]).astype(np.float32, copy=False)
    return RVQResult(recon=recon, codebook1=cb1, indices1=idx1, codebook2=cb2, indices2=idx2)


def rvq_reconstruct_flat(
    x: npt.NDArray[np.float32],
    k1: int,
    k2: int,
    *,
    seed: int = 42,
    n_iter: int = 25,
    quantize_codebooks: bool = False,
) -> npt.NDArray[np.float32]:
    """Two-stage residual VQ of a flat ``(M, C)`` pixel array; returns recon ``(M, C)``.

    The downstream path only needs reconstructed vectors -- no spatial index maps -- so
    this works on already-flattened, NaN-free pixels (callers mask NaN out per block).

    ``quantize_codebooks=True`` reconstructs from int8-round-tripped codebooks (the
    precision the bolt-on actually serves), for validating WS-1's int8 wire format
    downstream. Indices and the stage-1 residual still use the float32 codebooks, exactly
    as the server computes them before quantizing for the wire.
    """
    cb1 = kmeans_fit(x, k1, seed=seed, n_iter=n_iter)
    idx1 = assign_blocked(x, cb1)
    residual = (x - cb1[idx1]).astype(np.float32, copy=False)
    cb2 = kmeans_fit(residual, k2, seed=seed + 1, n_iter=n_iter)
    idx2 = assign_blocked(residual, cb2)
    if quantize_codebooks:
        cb1 = roundtrip_uint8(cb1)
        cb2 = roundtrip_uint8(cb2)
    return (cb1[idx1] + cb2[idx2]).astype(np.float32, copy=False)
