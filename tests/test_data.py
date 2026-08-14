"""Tests for tessera_vq.data: dataset-version normalization and read_region."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from affine import Affine

from tessera_vq import data
from tessera_vq.data import _normalize_dataset_version, read_region


def test_normalize_dataset_version_v_prefixed() -> None:
    assert _normalize_dataset_version("v1") == "1.0"
    assert _normalize_dataset_version("v1.1") == "1.1"
    assert _normalize_dataset_version("v2") == "2.0"


def test_normalize_dataset_version_bare_number() -> None:
    assert _normalize_dataset_version("1") == "1.0"
    assert _normalize_dataset_version("1.1") == "1.1"
    assert _normalize_dataset_version("2") == "2.0"


def test_normalize_dataset_version_already_normalized() -> None:
    assert _normalize_dataset_version("1.0") == "1.0"
    assert _normalize_dataset_version("v1.0") == "1.0"


# --- read_region: does it retain the geographic context (affine transform) that
# geotessera/zarr_utils computed for the *returned* mosaic, rather than discarding it
# or fabricating one from the caller's requested bounds? ---

_BOUNDS = (0.0, 50.0, 0.1, 50.05)
_YEAR = 2024

# Deliberately anchored away from _BOUNDS's own top-left corner (0.0, 50.05), so a test
# that accidentally re-derived the anchor from bounds instead of propagating the real
# transform would be caught rather than passing by coincidence.
_REAL_TRANSFORM = Affine(0.00009, 0.0, -0.014, 0.0, -0.00009, 50.081)


class _StubGtz:
    """Minimal zarr handle stand-in; only its identity matters to read_region."""


def _patch_zarr_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the zarr branch off so bbox-path tests don't touch the real zarr_utils."""
    monkeypatch.setattr(data.zarr_utils, "get_zarr", lambda: None)


@pytest.mark.skipif(  # type: ignore[misc]  # pytest's decorator isn't fully typed under strict mypy
    not data._USE_ZARR,
    reason="zarr path disabled (2026-08-14, unreliable source.coop shard reads) -- "
    "see data._USE_ZARR. Re-enable once it's flipped back on.",
)
def test_read_region_zarr_covered_propagates_real_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """zarr-covered path: transform comes from ``read_region_chunked``, not ``bounds``.

    The zarr multi-chunk read can extend past the requested bounds at chunk-grid
    edges, so the anchor zarr_utils computed for what it actually returned is the
    only correct one -- read_region must pass it through, not fabricate one.
    """
    mosaic = np.zeros((20, 30, 128), dtype=np.float32)
    monkeypatch.setattr(data.zarr_utils, "get_zarr", _StubGtz)
    monkeypatch.setattr(
        data.zarr_utils,
        "probe_zarr_coverage",
        lambda gtz, bounds, year: True,  # noqa: ARG005
    )
    monkeypatch.setattr(
        data.zarr_utils,
        "read_region_chunked",
        lambda gtz, bounds, year: (mosaic, _REAL_TRANSFORM, "native"),  # noqa: ARG005
    )
    result_mosaic, transform, path = read_region(_BOUNDS, _YEAR)
    assert path == "zarr"
    assert transform == _REAL_TRANSFORM
    assert transform.c != _BOUNDS[0] or transform.f != _BOUNDS[3]  # not fabricated from bounds
    np.testing.assert_array_equal(result_mosaic, mosaic)


def test_read_region_bbox_path_propagates_real_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bbox-fallback path: transform comes from ``fetch_mosaic_for_region``, not ``bounds``.

    geotessera's ``fetch_mosaic_for_region`` returns the union of whichever source
    tiles overlap ``bounds`` (not a ``bounds``-exact crop), so its own transform is
    the only accurate anchor for what it actually returned.
    """
    _patch_zarr_unavailable(monkeypatch)
    mosaic = np.zeros((20, 30, 128), dtype=np.float32)

    class _Gt:
        def fetch_mosaic_for_region(
            self, bounds: Any, year: Any, target_crs: str
        ) -> tuple[Any, Any, str]:
            assert target_crs == "EPSG:4326"
            return mosaic, _REAL_TRANSFORM, "bbox"

    result_mosaic, transform, path = read_region(_BOUNDS, _YEAR, gt=_Gt())
    assert path == "bbox"
    assert transform == _REAL_TRANSFORM
    assert transform.c != _BOUNDS[0] or transform.f != _BOUNDS[3]  # not fabricated from bounds
    np.testing.assert_array_equal(result_mosaic, mosaic)


def test_read_region_falls_back_to_bbox_when_zarr_mosaic_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """zarr reports coverage but returns no mosaic -> falls through to the bbox path.

    Guards against a regression where the fallback forgets to also propagate
    the bbox path's transform once the zarr branch has been entered.
    """
    mosaic = np.zeros((20, 30, 128), dtype=np.float32)
    monkeypatch.setattr(data.zarr_utils, "get_zarr", _StubGtz)
    monkeypatch.setattr(
        data.zarr_utils,
        "probe_zarr_coverage",
        lambda gtz, bounds, year: True,  # noqa: ARG005
    )
    monkeypatch.setattr(
        data.zarr_utils,
        "read_region_chunked",
        lambda gtz, bounds, year: (None, None, "native"),  # noqa: ARG005
    )

    class _Gt:
        def fetch_mosaic_for_region(
            self,
            bounds: Any,
            year: Any,
            target_crs: str,  # noqa: ARG002
        ) -> tuple[Any, Any, str]:
            return mosaic, _REAL_TRANSFORM, "bbox"

    result_mosaic, transform, path = read_region(_BOUNDS, _YEAR, gt=_Gt())
    assert path == "bbox"
    assert transform == _REAL_TRANSFORM
    np.testing.assert_array_equal(result_mosaic, mosaic)


def test_read_region_empty_returns_none_transform(monkeypatch: pytest.MonkeyPatch) -> None:
    """No coverage from either path -> ``(None, None, "empty")``, not a stale transform."""
    _patch_zarr_unavailable(monkeypatch)

    class _Gt:
        def fetch_mosaic_for_region(
            self,
            bounds: Any,
            year: Any,
            target_crs: str,  # noqa: ARG002
        ) -> tuple[None, None, str]:
            return None, None, "bbox"

    result_mosaic, transform, path = read_region(_BOUNDS, _YEAR, gt=_Gt())
    assert result_mosaic is None
    assert transform is None
    assert path == "empty"
