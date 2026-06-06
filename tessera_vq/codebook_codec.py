"""Per-dimension uint8 (de)quantization of per-tile codebooks (WS-1).

The bolt-on used to ship float32 codebooks, ~4x larger than the int8 the byte model
assumes. The raw Tessera embeddings are themselves int8, so the codebooks (which are
*averages* of int8 values) lose only sub-int8 precision when requantized — downstream
impact is expected to be nil (validated separately).

Scheme: affine per ``(tile, dimension)`` map to uint8 (min/max -> 0..255). Per-dimension
(not per-tile) because the embedding space is anisotropic; asymmetric (min/max, not
symmetric) because stage-1 prototypes are offset from zero. The scales (lo/hi) are
``(n_tiles, C)`` float32 -> ~2 KB/tile, negligible per pixel.

Pure numpy so the lightweight ``VQTessera`` client can dequantize without the server
extra (sklearn etc.) -- same rationale as ``entropy.py``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt


def quantize_codebook_uint8(
    cb: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Quantize codebooks ``(n_tiles, k, C)`` -> ``(q uint8, lo (n,C), hi (n,C))``.

    ``lo``/``hi`` are per-tile, per-dimension min/max over the ``k`` codes. A constant
    dimension (``hi == lo``) maps to ``q = 0`` and dequantizes back to ``lo`` exactly.
    """
    if cb.shape[0] == 0:
        c = cb.shape[-1]
        return (
            np.zeros((0, cb.shape[1], c), np.uint8),
            np.zeros((0, c), np.float32),
            np.zeros((0, c), np.float32),
        )
    lo = cb.min(axis=1).astype(np.float32)  # (n, C)
    hi = cb.max(axis=1).astype(np.float32)
    span = (hi - lo).astype(np.float32)
    safe = np.where(span > 0, span, 1.0).astype(np.float32)
    q = np.round((cb - lo[:, None, :]) / safe[:, None, :] * 255.0).astype(np.uint8)
    return q, lo, hi


def dequantize_codebook_uint8(
    q: npt.NDArray[np.uint8],
    lo: npt.NDArray[np.float32],
    hi: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Inverse of :func:`quantize_codebook_uint8`; returns ``(n_tiles, k, C)`` float32."""
    span = (hi - lo).astype(np.float32)
    out = lo[:, None, :] + q.astype(np.float32) / 255.0 * span[:, None, :]
    return out.astype(np.float32)


def roundtrip_uint8(cb: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Quantize then dequantize a single ``(k, C)`` codebook (the served int8 precision).

    Used to evaluate reconstruction/downstream quality *as the bolt-on serves it* — the
    client reconstructs from int8-precision codebooks, so the validation path must too.
    """
    q, lo, hi = quantize_codebook_uint8(cb[None])
    return cast("npt.NDArray[np.float32]", dequantize_codebook_uint8(q, lo, hi)[0])
