"""Data loaders for Tessera embeddings, Pool A diagnostics, and downstream tasks.

Implemented in Phase 1 (docs/spec.md) over geotessera (zarr via vendored zarr_utils
from ucam-eo/tee, with a bounding-box fallback): ``read_window``,
``iter_pool_a_windows``, ``sample_isotropy_points``, ``load_downstream``. Land-only
sampling from geotessera coverage; no embeddings persisted. Stub at Phase 0.
"""
