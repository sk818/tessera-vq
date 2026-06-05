"""Root conftest: anchors the repo root on sys.path so tests can import ``scripts.*``.

pytest prepends the directory containing the top-level conftest to ``sys.path`` in
its default (prepend) import mode, making both ``tessera_vq`` and the (non-package)
``scripts`` directory importable from test modules.
"""
