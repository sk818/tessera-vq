"""Tessera VQ — per-tile vector quantisation for Tessera embedding compression.

Library code for the compression study. Phase-by-phase implementation lives in the
modules below and is driven by the entry points in ``scripts/``. See ``docs/spec.md``.

The public bolt-on client surface is re-exported here for convenience: ``VQTessera``
for the geotessera-compatible client, ``QuantizedStructure`` for the per-tile
payload returned by ``fetch_quantized_structure``, ``NoCoverageError`` for the
"no embeddings here" failure mode, and the ``Distance`` literal type.
"""

from tessera_vq.client import (
    Distance,
    NoCoverageError,
    QuantizedStructure,
    VQTessera,
    reconstruct_from_structure,
)

__version__ = "0.3.2"

__all__ = [
    "Distance",
    "NoCoverageError",
    "QuantizedStructure",
    "VQTessera",
    "__version__",
    "reconstruct_from_structure",
]
