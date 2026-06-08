"""Tests for tessera_vq.tile_cache: durable size-capped byte cache (WS-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tessera_vq.tile_cache import TileCache


def test_compute_once_then_hit(tmp_path: Path) -> None:
    """First call computes; subsequent calls hit the store (compute not re-run)."""
    cache = TileCache(tmp_path, max_bytes=10_000)
    calls = {"n": 0}

    def compute() -> bytes:
        calls["n"] += 1
        return b"payload"

    assert cache.get_or_compute("k1", compute) == b"payload"
    assert cache.get_or_compute("k1", compute) == b"payload"
    assert calls["n"] == 1  # computed exactly once
    assert cache.get("k1") == b"payload"


def test_miss_returns_none(tmp_path: Path) -> None:
    """An absent key reads as a miss."""
    assert TileCache(tmp_path, max_bytes=10_000).get("nope") is None


def test_compute_exception_stores_nothing(tmp_path: Path) -> None:
    """If compute raises (e.g. no tiles), the error propagates and nothing is cached."""
    cache = TileCache(tmp_path, max_bytes=10_000)

    def boom() -> bytes:
        raise ValueError("no tiles")

    with pytest.raises(ValueError, match="no tiles"):
        cache.get_or_compute("k", boom)
    assert cache.get("k") is None
    assert cache.total_bytes() == 0


def test_distinct_keys_isolated(tmp_path: Path) -> None:
    """Different keys store independent payloads."""
    cache = TileCache(tmp_path, max_bytes=10_000)
    cache.get_or_compute("a", lambda: b"AAAA")
    cache.get_or_compute("b", lambda: b"BBBBBB")
    assert cache.get("a") == b"AAAA"
    assert cache.get("b") == b"BBBBBB"


def test_lru_eviction_at_cap(tmp_path: Path) -> None:
    """Over the cap, the least-recently-used entry is evicted; recent ones survive."""
    cache = TileCache(tmp_path, max_bytes=250)  # holds ~2 of the 100-byte payloads
    blob = bytes(100)
    cache.get_or_compute("old", lambda: blob)
    cache.get_or_compute("mid", lambda: blob)
    cache.get("old")  # touch 'old' so 'mid' becomes the LRU
    cache.get_or_compute("new", lambda: blob)  # 300 > 250 -> evict one (the LRU = 'mid')
    assert cache.total_bytes() <= 250
    assert cache.get("old") is not None
    assert cache.get("new") is not None
    assert cache.get("mid") is None


def test_atomic_no_tmp_files_left(tmp_path: Path) -> None:
    """A successful write leaves the final file and no stray temp files."""
    cache = TileCache(tmp_path, max_bytes=10_000)
    cache.get_or_compute("k", lambda: b"data")
    leftovers = [p.name for p in tmp_path.rglob("*") if ".tmp-" in p.name]
    assert leftovers == []
