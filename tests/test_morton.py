"""Smoke test for tessera_vq.morton. Real cross-checks against pymorton land in Phase 4."""

import tessera_vq.morton


def test_morton_importable() -> None:
    """The module imports cleanly at Phase 0 (stub)."""
    assert tessera_vq.morton is not None
