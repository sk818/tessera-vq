"""Smoke test for tessera_vq.quantize. Real quantisation tests land in Phase 3."""

import tessera_vq.quantize


def test_quantize_importable() -> None:
    """The module imports cleanly at Phase 0 (stub)."""
    assert tessera_vq.quantize is not None
