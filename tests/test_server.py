"""Tests for tessera_vq.server: bbox-size guardrails and no-coverage 422."""

from __future__ import annotations

import io

import numpy as np
import numpy.typing as npt
import pytest

from tessera_vq import server
from tessera_vq.codebook_codec import dequantize_codebook_uint8
from tessera_vq.entropy import rle_decode_stack
from tessera_vq.server import _bbox_size_km, _check_bbox_size, _no_tiles_message


def test_bbox_size_km_known_cambridge_box() -> None:
    """A 0.01-degree square near Cambridge (lat 52) is ~0.7 km x ~1.1 km."""
    width_km, height_km = _bbox_size_km((0.145, 52.045, 0.155, 52.055))
    # 0.01 deg lat = ~1.113 km; 0.01 deg lon at lat 52 = ~0.686 km.
    assert abs(height_km - 1.113) < 0.05
    assert abs(width_km - 0.686) < 0.05


def test_check_bbox_size_accepts_small_bbox() -> None:
    """A small bbox returns no error."""
    assert _check_bbox_size((0.145, 52.045, 0.155, 52.055)) is None


def test_check_bbox_size_rejects_huge_bbox() -> None:
    """A multi-degree bbox is rejected with an informative error message."""
    msg = _check_bbox_size((-2.0, 50.0, 2.0, 54.0))  # ~280 km tall, ~270 km wide
    assert msg is not None
    assert "too large" in msg
    assert "TESSERA_VQ_MAX_BBOX_KM" in msg


def test_no_tiles_message_includes_dims_and_t() -> None:
    """Diagnostic message names the reprojected region size and the requested t."""
    msg = _no_tiles_message((398, 603), t=256)
    assert "t=256" in msg
    assert "603x398" in msg  # message uses (width=cols, height=rows)
    assert "smaller t" in msg or "larger bbox" in msg


def _patch_read_region(monkeypatch: pytest.MonkeyPatch, window: npt.NDArray[np.float32]) -> None:
    """Replace tessera_vq.server.read_region with a stub returning ``window``."""
    monkeypatch.setattr(
        server,
        "read_region",
        lambda bbox, year: (window, "test"),
    )


def test_quantized_returns_422_when_no_tiles_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """``/quantized`` returns 422 + diagnostic when t exceeds the all-finite area."""
    rng = np.random.default_rng(0)
    window = rng.standard_normal((10, 10, 128)).astype(np.float32)
    _patch_read_region(monkeypatch, window)
    client = server.app.test_client()
    resp = client.post(
        "/quantized",
        json={"bbox": [0.0, 50.0, 0.001, 50.001], "t": 32, "k": 4},
    )
    assert resp.status_code == 422
    body = resp.get_json()
    assert "error" in body
    assert "t=32" in body["error"]
    assert "10x10" in body["error"]


def test_quantized_rvq_returns_422_when_no_tiles_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/quantized_rvq`` returns 422 + diagnostic when t exceeds the all-finite area."""
    rng = np.random.default_rng(0)
    window = rng.standard_normal((10, 10, 128)).astype(np.float32)
    _patch_read_region(monkeypatch, window)
    client = server.app.test_client()
    resp = client.post(
        "/quantized_rvq",
        json={"bbox": [0.0, 50.0, 0.001, 50.001], "t": 32, "k1": 4, "k2": 4},
    )
    assert resp.status_code == 422
    body = resp.get_json()
    assert "error" in body
    assert "t=32" in body["error"]


def test_quantized_rvq_returns_422_when_all_candidate_tiles_have_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RVQ also 422s when candidate tiles exist dimensionally but every one is NaN-cut.

    Regression for the original bug report: Cambridge-shape reprojected window where
    the source's UTM-to-EPSG:4326 reprojection introduced NaN strips that cut every
    candidate tile. Old behaviour: silent ``n_tiles=0`` NPZ. New behaviour: 422.
    """
    rng = np.random.default_rng(0)
    window = rng.standard_normal((398, 603, 128)).astype(np.float32)
    window[:, 250:260] = np.nan  # vertical NaN strip cutting both candidate tiles at t=256
    _patch_read_region(monkeypatch, window)
    client = server.app.test_client()
    resp = client.post(
        "/quantized_rvq",
        json={"bbox": [0.1025, 52.1751, 0.1758, 52.22], "t": 256, "k1": 256, "k2": 256},
    )
    assert resp.status_code == 422
    body = resp.get_json()
    assert "t=256" in body["error"]


def test_quantized_succeeds_when_tiles_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: 200 + NPZ when t fits the all-finite window."""
    rng = np.random.default_rng(0)
    window = rng.standard_normal((64, 64, 128)).astype(np.float32)
    _patch_read_region(monkeypatch, window)
    client = server.app.test_client()
    resp = client.post(
        "/quantized",
        json={"bbox": [0.0, 50.0, 0.001, 50.001], "t": 32, "k": 4},
    )
    assert resp.status_code == 200
    with np.load(io.BytesIO(resp.data)) as data:
        assert data["positions"].shape[0] == 4  # 64/32 = 2 -> 2x2 = 4 tiles
        assert data["codebooks"].shape == (4, 4, 128)


def test_quantized_rvq_succeeds_when_tiles_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control for the RVQ endpoint."""
    rng = np.random.default_rng(0)
    window = rng.standard_normal((64, 64, 128)).astype(np.float32)
    _patch_read_region(monkeypatch, window)
    client = server.app.test_client()
    resp = client.post(
        "/quantized_rvq",
        json={"bbox": [0.0, 50.0, 0.001, 50.001], "t": 32, "k1": 4, "k2": 4},
    )
    assert resp.status_code == 200
    with np.load(io.BytesIO(resp.data)) as data:
        assert data["positions"].shape[0] == 4
        # codebooks ship as per-dim uint8 (q + lo/hi), not float32
        assert "codebooks1" not in data.files and "codebooks1_q" in data.files
        assert data["codebooks1_q"].dtype == np.uint8
        cb1 = dequantize_codebook_uint8(
            data["codebooks1_q"], data["codebooks1_lo"], data["codebooks1_hi"]
        )
        assert cb1.shape == (4, 4, 128) and cb1.dtype == np.float32
        # idx1 ships RLE'd (idx1_values/lengths/runs), idx2 stays raw
        assert "indices1" not in data.files and "idx1_values" in data.files
        assert data["indices2"].shape == (4, 32, 32)
        idx1 = rle_decode_stack(
            data["idx1_values"], data["idx1_lengths"], data["idx1_runs"], 32, 32
        )
        assert idx1.shape == (4, 32, 32)
        assert int(idx1.max()) < 4
