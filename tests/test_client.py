"""Tests for tessera_vq.client.VQTessera and its _reconstruct helper."""

import io

import numpy as np
import pytest
from affine import Affine

from tessera_vq.client import VQTessera, _reconstruct


def _make_npz(
    t: int = 16,
    k: int = 4,
    full_h: int = 32,
    full_w: int = 48,
    positions: list[tuple[int, int]] | None = None,
) -> bytes:
    """Build an NPZ matching the bolt-on's /quantized payload shape."""
    if positions is None:
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    n = len(positions)
    rng = np.random.default_rng(0)
    codebooks = rng.standard_normal((n, k, 128)).astype(np.float32)
    indices = rng.integers(0, k, size=(n, t, t), dtype=np.uint8)
    pos_arr = np.asarray(positions, dtype=np.int32)
    meta = np.array([t, k, 2024, full_h, full_w], dtype=np.int32)
    buf = io.BytesIO()
    np.savez(
        buf,
        codebooks=codebooks,
        indices=indices,
        positions=pos_arr,
        meta=meta,
        distance=np.asarray("euclidean"),
    )
    return buf.getvalue()


def test_reconstruct_shape_and_transform() -> None:
    """A full grid yields the expected mosaic shape, EPSG:4326, and top-left affine."""
    bbox = (0.0, 50.0, 0.1, 50.05)
    mosaic, transform, crs = _reconstruct(_make_npz(), bbox)
    assert mosaic.shape == (32, 48, 128)
    assert mosaic.dtype == np.float32
    assert crs == "EPSG:4326"
    assert isinstance(transform, Affine)
    # Top-left pixel corner sits at (lon0, lat1).
    assert abs(transform.c - bbox[0]) < 1e-12
    assert abs(transform.f - bbox[3]) < 1e-12
    # Pixel size matches bbox span / mosaic shape.
    assert abs(transform.a - (bbox[2] - bbox[0]) / 48) < 1e-12
    assert abs(transform.e - -(bbox[3] - bbox[1]) / 32) < 1e-12


def test_reconstruct_fills_only_covered_tiles_with_nan_elsewhere() -> None:
    """Tiles missing from ``positions`` stay NaN; covered tiles are finite."""
    bbox = (0.0, 50.0, 0.1, 50.05)
    npz = _make_npz(positions=[(0, 0)])  # only top-left tile covered
    mosaic, _, _ = _reconstruct(npz, bbox)
    assert np.isfinite(mosaic[:16, :16]).all()
    assert np.isnan(mosaic[:16, 16:]).all()
    assert np.isnan(mosaic[16:, :]).all()


def test_reconstruct_truncates_to_tile_multiple() -> None:
    """If meta dims aren't multiples of t, output is truncated to the largest fit."""
    bbox = (0.0, 50.0, 0.1, 50.05)
    npz = _make_npz(t=16, full_h=33, full_w=49, positions=[(0, 0)])
    mosaic, _, _ = _reconstruct(npz, bbox)
    assert mosaic.shape == (32, 48, 128)  # 33//16 = 2, 49//16 = 3 -> 32, 48


def test_client_rejects_non_4326_target_crs() -> None:
    """target_crs other than EPSG:4326 raises ValueError (the server returns 4326 only)."""
    client = VQTessera(server_url="http://localhost:8000")
    with pytest.raises(ValueError, match="EPSG:4326"):
        client.fetch_mosaic_for_region((0.0, 50.0, 0.1, 50.05), target_crs="EPSG:3857")
