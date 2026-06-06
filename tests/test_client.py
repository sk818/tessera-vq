"""Tests for tessera_vq.client.VQTessera and its reconstruction helpers."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import numpy as np
import pytest
from affine import Affine

from tessera_vq.client import (
    NoCoverageError,
    QuantizedStructure,
    VQTessera,
    _reconstruct,
    _structure_from_npz,
    reconstruct_from_structure,
)
from tessera_vq.entropy import rle_encode_stack


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
    pos_arr = np.asarray(positions, dtype=np.int32) if n else np.zeros((0, 2), dtype=np.int32)
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


def _make_rvq_npz(
    t: int = 16,
    k1: int = 4,
    k2: int = 4,
    full_h: int = 32,
    full_w: int = 48,
    positions: list[tuple[int, int]] | None = None,
) -> bytes:
    """Build an NPZ matching the bolt-on's /quantized_rvq payload shape."""
    if positions is None:
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    n = len(positions)
    rng = np.random.default_rng(1)
    cb1 = rng.standard_normal((n, k1, 128)).astype(np.float32)
    cb2 = rng.standard_normal((n, k2, 128)).astype(np.float32)
    idx1 = rng.integers(0, k1, size=(n, t, t), dtype=np.uint8)
    idx2 = rng.integers(0, k2, size=(n, t, t), dtype=np.uint8)
    v1, l1, r1 = rle_encode_stack(idx1)
    pos_arr = np.asarray(positions, dtype=np.int32) if n else np.zeros((0, 2), dtype=np.int32)
    meta = np.array([t, k1, k2, 2024, full_h, full_w], dtype=np.int32)
    buf = io.BytesIO()
    np.savez(
        buf,
        codebooks1=cb1,
        idx1_values=v1,
        idx1_lengths=l1.astype(np.uint32),
        idx1_runs=r1.astype(np.int32),
        codebooks2=cb2,
        indices2=idx2,
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


def test_structure_from_npz_single_stage() -> None:
    """``_structure_from_npz`` round-trips single-stage NPZ fields onto QuantizedStructure."""
    bbox = (0.0, 50.0, 0.1, 50.05)
    struct = _structure_from_npz(_make_npz(t=16, k=4, full_h=32, full_w=48), bbox)
    assert isinstance(struct, QuantizedStructure)
    assert struct.is_rvq is False
    assert struct.codebooks2 is None and struct.indices2 is None
    assert struct.tile_size == 16
    assert struct.k1 == 4 and struct.k2 is None
    assert struct.metric == "euclidean"
    assert struct.mosaic_shape == (32, 48)
    assert struct.bbox == bbox
    assert struct.year == 2024
    assert struct.positions.shape == (6, 2)


def test_structure_from_npz_rvq() -> None:
    """``_structure_from_npz`` decodes RVQ NPZ with both codebooks/indices stacks."""
    bbox = (0.0, 50.0, 0.1, 50.05)
    struct = _structure_from_npz(_make_rvq_npz(t=16, k1=4, k2=8, full_h=32, full_w=48), bbox)
    assert struct.is_rvq is True
    assert struct.codebooks2 is not None and struct.indices2 is not None
    assert struct.codebooks1.shape == (6, 4, 128)
    assert struct.codebooks2.shape == (6, 8, 128)
    assert struct.k1 == 4 and struct.k2 == 8


def test_reconstruct_raises_no_coverage_on_zero_tile_npz() -> None:
    """An NPZ with zero positions triggers NoCoverageError, not an empty mosaic."""
    bbox = (0.0, 50.0, 0.1, 50.05)
    npz = _make_npz(positions=[])
    with pytest.raises(NoCoverageError, match="0 tiles"):
        _reconstruct(npz, bbox)


def test_reconstruct_from_structure_raises_on_all_nan_mosaic() -> None:
    """If tiles exist but the truncated output is 0x0, mosaic is all-NaN -> raise."""
    bbox = (0.0, 50.0, 0.1, 50.05)
    # full_h=16, t=32 -> out_h = (16//32)*32 = 0 -> 0-sized mosaic dim
    struct = _structure_from_npz(_make_npz(t=32, full_h=16, full_w=16), bbox)
    with pytest.raises(NoCoverageError):
        reconstruct_from_structure(struct)


def test_reconstruct_rvq_round_trip_shape() -> None:
    """RVQ NPZ reconstructs to the expected truncated mosaic shape via ``_reconstruct``."""
    bbox = (0.0, 50.0, 0.1, 50.05)
    npz = _make_rvq_npz(t=16, full_h=32, full_w=48)
    mosaic, transform, crs = _reconstruct(npz, bbox)
    assert mosaic.shape == (32, 48, 128)
    assert crs == "EPSG:4326"
    assert isinstance(transform, Affine)
    assert np.isfinite(mosaic).all()  # full grid covered by the default positions


class _StubResponse:
    """Minimal ``urlopen`` context-manager stub returning a fixed body."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _StubResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _http_error_422(message: str) -> urllib.error.HTTPError:
    """Build an HTTPError(422) with a JSON ``{"error": ...}`` body the client can decode."""
    body = json.dumps({"error": message}).encode()
    return urllib.error.HTTPError(
        url="http://test/quantized",
        code=422,
        msg="Unprocessable Entity",
        hdrs=None,  # type: ignore[arg-type]
        fp=io.BytesIO(body),
    )


def test_post_translates_422_to_no_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_post`` maps a server HTTP 422 to NoCoverageError with the server's message."""

    def _raise_422(_req: Any, timeout: float = 0.0) -> Any:  # noqa: ARG001
        raise _http_error_422("no all-finite t=256 tile fits the reprojected region")

    monkeypatch.setattr("urllib.request.urlopen", _raise_422)
    gt = VQTessera(server_url="http://test", t=256, k=4)
    with pytest.raises(NoCoverageError, match="no all-finite"):
        gt.fetch_quantized_structure((0.0, 50.0, 0.001, 50.001))


def test_post_propagates_non_422_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-422 HTTP errors pass through as urllib.error.HTTPError."""

    def _raise_500(_req: Any, timeout: float = 0.0) -> Any:  # noqa: ARG001
        raise urllib.error.HTTPError(
            url="http://test/quantized",
            code=500,
            msg="Server Error",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error":"boom"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", _raise_500)
    gt = VQTessera(server_url="http://test", t=32, k=4)
    with pytest.raises(urllib.error.HTTPError):
        gt.fetch_quantized_structure((0.0, 50.0, 0.001, 50.001))


def test_fetch_quantized_structure_returns_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 NPZ response is decoded into a populated QuantizedStructure."""
    npz = _make_npz(t=16, full_h=32, full_w=48)

    def _ok(_req: Any, timeout: float = 0.0) -> _StubResponse:  # noqa: ARG001
        return _StubResponse(npz)

    monkeypatch.setattr("urllib.request.urlopen", _ok)
    gt = VQTessera(server_url="http://test", t=16, k=4)
    struct = gt.fetch_quantized_structure((0.0, 50.0, 0.1, 50.05))
    assert struct.is_rvq is False
    assert struct.positions.shape == (6, 2)
    assert struct.mosaic_shape == (32, 48)


def test_fetch_quantized_structure_rvq_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``k2`` is set, fetch_quantized_structure decodes an RVQ NPZ."""
    npz = _make_rvq_npz(t=16, k1=4, k2=4, full_h=32, full_w=48)

    def _ok(_req: Any, timeout: float = 0.0) -> _StubResponse:  # noqa: ARG001
        return _StubResponse(npz)

    monkeypatch.setattr("urllib.request.urlopen", _ok)
    gt = VQTessera(server_url="http://test", t=16, k=4, k2=4)
    struct = gt.fetch_quantized_structure((0.0, 50.0, 0.1, 50.05))
    assert struct.is_rvq is True
    assert struct.k1 == 4 and struct.k2 == 4


def test_fetch_mosaic_for_region_raises_on_server_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: server 422 surfaces as NoCoverageError from fetch_mosaic_for_region."""

    def _raise_422(_req: Any, timeout: float = 0.0) -> Any:  # noqa: ARG001
        raise _http_error_422("no all-finite t=256 tile fits the reprojected region")

    monkeypatch.setattr("urllib.request.urlopen", _raise_422)
    gt = VQTessera(server_url="http://test", t=256, k=4, k2=4)
    with pytest.raises(NoCoverageError):
        gt.fetch_mosaic_for_region((0.0, 50.0, 0.001, 50.001))
