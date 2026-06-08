"""Durable, size-capped on-disk cache for quantized tile payloads (WS-2).

Compute-once-keep-forever store for the bolt-on: a cache hit skips both the geotessera
read and the per-tile k-means. Unlike an LRU tuned for temporal locality, it is durable
up to a byte cap (default 500 GB) and only LRU-evicts when full -- so for demand that
fits under the cap it is effectively permanent.

Keyed by an opaque string (the server uses the canonicalized request: bbox + year +
t/k1/k2/metric/seed + format version), so the grid-aligned requests TEE issues reuse the
same Tessera tiles. Bytes in, bytes out -- the value is the response NPZ.

Concurrency: a per-key lock collapses a thundering herd of identical cold requests to a
single compute (waitress is one multi-threaded process, so in-process locks suffice).
Writes are atomic (temp file + ``os.replace``). LRU recency is the file mtime, refreshed
on hit; eviction (rare -- only at the cap) scans and deletes oldest-first.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
from collections.abc import Callable
from pathlib import Path


class TileCache:
    """On-disk byte cache with per-key locking and LRU eviction at a size cap."""

    def __init__(self, root: str | Path, max_bytes: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self._locks: dict[str, threading.Lock] = {}
        self._master = threading.Lock()

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()
        return self.root / h[:2] / f"{h}.npz"

    def _lock_for(self, key: str) -> threading.Lock:
        with self._master:
            lk = self._locks.get(key)
            if lk is None:
                lk = threading.Lock()
                self._locks[key] = lk
            return lk

    def get(self, key: str) -> bytes | None:
        """Return cached bytes (refreshing LRU recency) or ``None`` on miss."""
        p = self._path(key)
        if not p.exists():
            return None
        with contextlib.suppress(OSError):
            os.utime(p, None)  # bump mtime so this entry is "recently used"
        return p.read_bytes()

    def get_or_compute(self, key: str, compute: Callable[[], bytes]) -> bytes:
        """Return cached bytes, else ``compute()`` once (under a per-key lock) and store.

        If ``compute`` raises, the exception propagates and nothing is stored (callers
        use this for the no-tiles/422 case).
        """
        hit = self.get(key)
        if hit is not None:
            return hit
        with self._lock_for(key):
            hit = self.get(key)  # double-check: another thread may have filled it
            if hit is not None:
                return hit
            data = compute()
            self._write_atomic(key, data)
            self._evict_if_over_cap()
            return data

    def _write_atomic(self, key: str, data: bytes) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def total_bytes(self) -> int:
        """Current on-disk size of the cache (sum of stored payloads)."""
        return sum(f.stat().st_size for f in self.root.rglob("*.npz"))

    def _evict_if_over_cap(self) -> None:
        """Delete least-recently-used entries until under ``max_bytes`` (oldest mtime first)."""
        entries = [(f.stat().st_mtime, f.stat().st_size, f) for f in self.root.rglob("*.npz")]
        total = sum(s for _m, s, _f in entries)
        if total <= self.max_bytes:
            return
        for _mtime, size, f in sorted(entries):  # ascending mtime -> oldest first
            with contextlib.suppress(OSError):
                f.unlink()
                total -= size
            if total <= self.max_bytes:
                break
