"""Tests for tessera_vq.server: bbox-size guardrails and no-coverage 422."""

from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from affine import Affine

from tessera_vq import server
from tessera_vq.codebook_codec import dequantize_codebook_uint8
from tessera_vq.server import _bbox_size_km, _check_bbox_size, _no_tiles_message
from tessera_vq.tile_cache import TileCache


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


def test_threads_has_headroom_over_max_concurrency() -> None:
    """_compute_slot() wraps the whole handler including read_region()'s
    network-bound geotessera/S3 fetch, so waitress needs enough worker threads
    to actually dispatch _MAX_CONCURRENCY requests concurrently -- otherwise
    _COMPUTE_SEM's clean 429 load-shedding is unreachable and requests instead
    queue silently inside waitress (observed live on tee.cl as opaque
    client-side timeouts, "Task queue depth is N" in the logs, with waitress's
    old unconfigured default of 4 threads vs. _MAX_CONCURRENCY=22)."""
    assert server._THREADS > server._MAX_CONCURRENCY


def test_threads_reads_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """TESSERA_VQ_THREADS, like the other TESSERA_VQ_* knobs, is settable
    without a code change -- reload the module with the env var set rather
    than reaching into the private constant, since it's computed at import
    time."""
    import importlib

    monkeypatch.setenv("TESSERA_VQ_THREADS", "77")
    try:
        importlib.reload(server)
        assert server._THREADS == 77
    finally:
        importlib.reload(server)  # restore module state for later tests


def test_version_endpoint_returns_dataset_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """``/version`` reports the resolved Tessera dataset version as JSON."""
    monkeypatch.setattr(server, "get_dataset_version", lambda: "1.1")
    client = server.app.test_client()
    resp = client.get("/version")
    assert resp.status_code == 200
    assert resp.get_json() == {"dataset_version": "1.1"}


_TEST_TRANSFORM = Affine(0.0001, 0.0, 0.0, 0.0, -0.0001, 50.0)


def _patch_read_region(monkeypatch: pytest.MonkeyPatch, window: npt.NDArray[np.float32]) -> None:
    """Replace tessera_vq.server.read_region with a stub returning ``window``."""
    monkeypatch.setattr(
        server,
        "read_region",
        lambda bbox, year: (window, _TEST_TRANSFORM, "test"),
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

    The NaN band here (columns 200-400) is wider than the original bug report's --
    with t=256 tiling now covering the last row/column via a pulled-back tile
    (tile_pixel_offset; see tessera_vq.sweep), the candidate columns are
    [0,256), [256,512) AND [347,603), and a strip has to reach from inside the
    first into inside the third to defeat all three (a narrower strip no longer
    can -- see test_quantized_rvq_succeeds_when_edge_tile_dodges_a_narrow_nan_strip
    directly below, which is now a *positive* control instead).
    """
    rng = np.random.default_rng(0)
    window = rng.standard_normal((398, 603, 128)).astype(np.float32)
    window[:, 200:400] = np.nan  # wide enough to cut all three candidate columns at t=256
    _patch_read_region(monkeypatch, window)
    client = server.app.test_client()
    resp = client.post(
        "/quantized_rvq",
        json={"bbox": [0.1025, 52.1751, 0.1758, 52.22], "t": 256, "k1": 256, "k2": 256},
    )
    assert resp.status_code == 422
    body = resp.get_json()
    assert "t=256" in body["error"]


def test_quantized_rvq_succeeds_when_edge_tile_dodges_a_narrow_nan_strip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A narrow NaN strip that used to cut every t=256 candidate column now leaves
    one standing: the last column's tile is pulled back to end at the window's
    true right edge (tile_pixel_offset), landing at columns [347, 603) for this
    603px-wide window -- outside the [250, 260) strip that hits the two
    fixed-stride columns [0, 256) and [256, 512). Positive-side-effect regression
    for the tiling fix: real, valid ground that used to be either dropped
    (postcard/viewport truncation) or -- as here -- the difference between a
    hard 422 and a served (if partial) result, depending on exactly where a
    NaN strip happened to fall relative to the old fixed tile grid.
    """
    rng = np.random.default_rng(0)
    window = rng.standard_normal((398, 603, 128)).astype(np.float32)
    window[:, 250:260] = np.nan
    _patch_read_region(monkeypatch, window)
    client = server.app.test_client()
    resp = client.post(
        "/quantized_rvq",
        json={"bbox": [0.1025, 52.1751, 0.1758, 52.22], "t": 256, "k1": 256, "k2": 256},
    )
    assert resp.status_code == 200
    with np.load(io.BytesIO(resp.data)) as data:
        assert data["positions"].shape[0] > 0


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


def test_quantized_embeds_read_region_transform_as_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/quantized`` ships ``read_region``'s real transform as ``origin``, not a guess.

    ``real_transform`` is anchored well outside the requested bbox
    (``[0.0, 50.0, 0.001, 50.001]``) so a server that re-derived the anchor from the
    request instead of propagating ``read_region``'s own transform would be caught.
    """
    rng = np.random.default_rng(0)
    window = rng.standard_normal((64, 64, 128)).astype(np.float32)
    real_transform = Affine(0.00009, 0.0, -0.014, 0.0, -0.00009, 50.081)
    monkeypatch.setattr(
        server,
        "read_region",
        lambda bbox, year: (window, real_transform, "test"),  # noqa: ARG005
    )
    client = server.app.test_client()
    resp = client.post(
        "/quantized",
        json={"bbox": [0.0, 50.0, 0.001, 50.001], "t": 32, "k": 4},
    )
    assert resp.status_code == 200
    with np.load(io.BytesIO(resp.data)) as data:
        assert "origin" in data.files
        origin_lon, origin_lat, dx, dy = data["origin"]
        assert origin_lon == real_transform.c
        assert origin_lat == real_transform.f
        assert dx == real_transform.a
        assert dy == real_transform.e


def test_quantized_rvq_embeds_read_region_transform_as_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/quantized_rvq`` also ships ``read_region``'s real transform as ``origin``."""
    rng = np.random.default_rng(0)
    window = rng.standard_normal((64, 64, 128)).astype(np.float32)
    real_transform = Affine(0.00009, 0.0, -0.014, 0.0, -0.00009, 50.081)
    monkeypatch.setattr(
        server,
        "read_region",
        lambda bbox, year: (window, real_transform, "test"),  # noqa: ARG005
    )
    client = server.app.test_client()
    resp = client.post(
        "/quantized_rvq",
        json={"bbox": [0.0, 50.0, 0.001, 50.001], "t": 32, "k1": 4, "k2": 4},
    )
    assert resp.status_code == 200
    with np.load(io.BytesIO(resp.data)) as data:
        assert "origin" in data.files
        origin_lon, origin_lat, dx, dy = data["origin"]
        assert origin_lon == real_transform.c
        assert origin_lat == real_transform.f
        assert dx == real_transform.a
        assert dy == real_transform.e


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
        # idx1/idx2 ship as raw uint8 planes (DEFLATE-compressed NPZ), not RLE
        assert "idx1_values" not in data.files and "indices1" in data.files
        assert data["indices2"].shape == (4, 32, 32)
        idx1 = data["indices1"]
        assert idx1.shape == (4, 32, 32)
        assert int(idx1.max()) < 4


def test_quantized_rvq_cache_serves_second_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a cache configured, an identical second request is served without recompute."""
    rng = np.random.default_rng(0)
    window = rng.standard_normal((64, 64, 128)).astype(np.float32)
    calls = {"n": 0}

    def counting_read(bbox: object, year: object) -> tuple[npt.NDArray[np.float32], Affine, str]:
        calls["n"] += 1
        return window, _TEST_TRANSFORM, "test"

    monkeypatch.setattr(server, "read_region", counting_read)
    monkeypatch.setattr(server, "_CACHE", TileCache(tmp_path, 10**9))
    client = server.app.test_client()
    body = {"bbox": [0.0, 50.0, 0.001, 50.001], "t": 32, "k1": 4, "k2": 4}
    r1 = client.post("/quantized_rvq", json=body)
    r2 = client.post("/quantized_rvq", json=body)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.data == r2.data  # identical bytes
    assert calls["n"] == 1  # second request hit the cache (no recompute)


def test_quantized_rvq_429_when_compute_slots_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no free compute slot, a (cache-miss) request sheds load with 429 + Retry-After."""
    rng = np.random.default_rng(0)
    window = rng.standard_normal((64, 64, 128)).astype(np.float32)
    _patch_read_region(monkeypatch, window)
    sem = threading.BoundedSemaphore(1)
    sem.acquire()  # take the only slot so the handler can't get one
    monkeypatch.setattr(server, "_COMPUTE_SEM", sem)
    monkeypatch.setattr(server, "_CACHE", None)
    client = server.app.test_client()
    resp = client.post(
        "/quantized_rvq",
        json={"bbox": [0.0, 50.0, 0.001, 50.001], "t": 32, "k1": 4, "k2": 4},
    )
    assert resp.status_code == 429  # noqa: PLR2004
    assert resp.headers.get("Retry-After") == "2"
    sem.release()
