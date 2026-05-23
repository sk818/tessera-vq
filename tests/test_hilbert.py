"""Smoke test for tessera_vq.hilbert. Real cross-checks against hilbertcurve land in Phase 4."""

import tessera_vq.hilbert


def test_hilbert_importable() -> None:
    """The module imports cleanly at Phase 0 (stub)."""
    assert tessera_vq.hilbert is not None
