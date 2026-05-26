"""Tests for tessera_vq.server helpers (bbox-size guardrails)."""

from tessera_vq.server import _bbox_size_km, _check_bbox_size


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
