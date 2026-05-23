"""Smoke test for tessera_vq.metrics. Epps-Pulley/Wasserstein checks land in Phases 2-3."""

import tessera_vq.metrics


def test_metrics_importable() -> None:
    """The module imports cleanly at Phase 0 (stub)."""
    assert tessera_vq.metrics is not None
