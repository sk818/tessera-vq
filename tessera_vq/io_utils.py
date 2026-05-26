"""IO helpers: config loading and provenance-tagged Parquet writes.

Per CLAUDE.md every Parquet output carries ``git_sha``, ``seed``, ``timestamp_utc``,
and ``config_hash``. Use :func:`write_parquet_with_provenance` for all writes.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the resolved ``config.yaml`` (values are heterogeneous, hence ``Any``)."""
    with open(path, encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)
    return config


def config_hash(path: str | Path = "config.yaml") -> str:
    """SHA-256 of the raw ``config.yaml`` bytes (for provenance)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_sha(*, short: bool = True) -> str:
    """Short git SHA of HEAD, or ``"unknown"`` outside a git repo."""
    args = ["git", "rev-parse", "HEAD"]
    if short:
        args.insert(2, "--short")
    try:
        out = subprocess.run(args, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.stdout.strip()


def write_parquet_with_provenance(
    df: pd.DataFrame,
    path: str | Path,
    *,
    seed: int,
    config_path: str | Path = "config.yaml",
) -> Path:
    """Append provenance columns to ``df`` and write it to Parquet at ``path``."""
    out = df.copy()
    out["git_sha"] = git_sha()
    out["seed"] = seed
    out["timestamp_utc"] = datetime.now(UTC).isoformat()
    out["config_hash"] = config_hash(config_path)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    return dest
