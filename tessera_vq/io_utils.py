"""IO helpers, including provenance-tagged Parquet writes.

Implemented in Phase 1 (docs/spec.md): ``write_parquet_with_provenance`` stamps every
output with ``git_sha``, ``seed``, ``timestamp_utc``, and ``config_hash`` (see CLAUDE.md).
Stub at Phase 0.
"""
