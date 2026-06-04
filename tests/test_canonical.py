"""Tests for tessera_vq.canonical: bbox loader + path-aware read helper.

The loader test exercises the .X5 alignment invariant on the actual YAML; the
read tests monkeypatch the underlying geotessera/zarr calls so we never touch
the network or the real Tessera store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tessera_vq import canonical
from tessera_vq.canonical import (
    CanonicalBbox,
    load_canonical_bboxes,
    read_canonical_window,
    select_path,
)

REPO_YAML = Path(__file__).resolve().parents[1] / "config" / "canonical_bboxes.yaml"


def test_load_canonical_bboxes_returns_100_aligned_entries() -> None:
    """The shipped YAML loads to exactly 100 entries on the .X5 grid."""
    bboxes = load_canonical_bboxes(REPO_YAML)
    assert len(bboxes) == 100  # noqa: PLR2004
    for b in bboxes:
        # All coords end in .X5 so a +/- 0.045 deg box stays inside one tile
        assert round((b.lon % 0.1) * 1000) == 50  # noqa: PLR2004
        assert round((b.lat % 0.1) * 1000) == 50  # noqa: PLR2004


def test_load_canonical_bboxes_rejects_misaligned(tmp_path: Path) -> None:
    """A YAML with a coord that isn't on .X5 raises ValueError."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "bboxes:\n  - {name: misaligned, lon: 0.0, lat: 50.05, biome: x, continent: EU}\n"
    )
    with pytest.raises(ValueError, match=r"\.X5"):
        load_canonical_bboxes(bad)


def _patch_zarr(monkeypatch: pytest.MonkeyPatch, *, available: bool) -> None:
    """Stub out get_zarr / probe_zarr_coverage so select_path is deterministic."""
    monkeypatch.setattr(canonical.zarr_utils, "get_zarr", lambda: object() if available else None)
    monkeypatch.setattr(canonical.zarr_utils, "probe_zarr_coverage", lambda *_a, **_k: available)


def test_select_path_returns_bbox_when_zarr_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No zarr handle -> always falls back to the bbox path, no probing attempted."""
    monkeypatch.setattr(canonical.zarr_utils, "get_zarr", lambda: None)
    b = CanonicalBbox(name="x", lon=-3.05, lat=52.05, biome="x", continent="EU")
    assert select_path(b, 2024) == "bbox"


def test_select_path_returns_zarr_when_covered(monkeypatch: pytest.MonkeyPatch) -> None:
    """zarr_utils says covered -> path is "zarr"."""
    _patch_zarr(monkeypatch, available=True)
    b = CanonicalBbox(name="x", lon=-3.05, lat=52.05, biome="x", continent="EU")
    assert select_path(b, 2024) == "zarr"


def test_read_canonical_window_returns_unavailable_on_geotessera_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If geotessera.fetch_mosaic_for_region raises (no tiles for bbox+year),
    read_canonical_window swallows the exception and reports unavailable.

    Regression: the v1 full-run crashed on bbox 34 (South Island NZ) with
    geotessera ValueError("No embedding tiles found for bbox ... in year 2024")
    because read_region propagated the exception instead of None-ing out.
    """
    _patch_zarr(monkeypatch, available=False)

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("No embedding tiles found for bbox ... in year 2024")

    monkeypatch.setattr("tessera_vq.canonical.read_region", _raise)
    b = CanonicalBbox(name="NZ", lon=170.05, lat=-43.05, biome="x", continent="OC")
    mosaic, path = read_canonical_window(b, 2024)
    assert mosaic is None
    assert path == "unavailable"


def test_read_canonical_window_returns_bbox_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path through the bbox-fallback branch: mosaic returned, path='bbox'."""
    _patch_zarr(monkeypatch, available=False)
    fake = np.zeros((100, 100, 128), dtype=np.float32)
    monkeypatch.setattr("tessera_vq.canonical.read_region", lambda *_a, **_k: (fake, "bbox"))
    b = CanonicalBbox(name="x", lon=22.05, lat=-1.05, biome="x", continent="AF")
    mosaic, path = read_canonical_window(b, 2024)
    assert mosaic is not None
    assert mosaic.shape == (100, 100, 128)
    assert path == "bbox"
