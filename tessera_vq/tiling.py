"""Jittered, NaN-aware tile selection from a large window (WS-0b).

Large tiles (t up to 1024) only fit cleanly inside a window bigger than the tile,
so the sampler reads a ~12 km window (window_px ~= 1200) at a canonical bbox centre
and then picks 1-2 tiles from its finite interior. Two requirements drive this
module (points 3-4 of the research plan):

- **Jitter** the tile origin around the window centre by up to +-t/2 (clamped
  in-bounds, seeded), so tile placement varies across bboxes and stays in the
  finite interior. At large t a ~12 km window fits one such tile (the supervisor
  confirmed 1 tile/bbox is fine and the statistics are insensitive to it).
- **Avoid NaN edges.** Tessera windows carry no-data (NaN) pixels near coverage
  edges; per-tile k-means cannot consume NaN, so we score candidate tiles by their
  finite-pixel fraction and keep the most-finite, non-overlapping ones.

Pure numpy: no geotessera / zarr, so it unit-tests on synthetic windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class TileSample:
    """One selected tile: its top-left ``(row, col)`` origin, finite fraction, data."""

    row: int
    col: int
    finite_frac: float
    tile: npt.NDArray[np.float32]


def finite_mask(window: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
    """``(H, W)`` boolean: a pixel is finite iff its whole 128-vector is finite."""
    return cast("npt.NDArray[np.bool_]", ~np.isnan(window).any(axis=-1))


def _candidate_origins(
    h: int, w: int, t: int, n: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """``n`` top-left origins jittered by +-t/2 about the window centre, clamped."""
    cr, cc = (h - t) // 2, (w - t) // 2
    j = t // 2
    rs = np.clip(cr + rng.integers(-j, j + 1, size=n), 0, h - t)
    cs = np.clip(cc + rng.integers(-j, j + 1, size=n), 0, w - t)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for r, c in zip(rs.tolist(), cs.tolist(), strict=True):
        if (r, c) not in seen:
            seen.add((r, c))
            out.append((int(r), int(c)))
    return out


def extract_finite_tiles(
    window: npt.NDArray[np.float32],
    t: int,
    *,
    n_tiles: int = 2,
    seed: int = 42,
    min_finite_frac: float = 1.0,
    n_candidates: int = 24,
) -> list[TileSample]:
    """Pick up to ``n_tiles`` jittered, non-overlapping, most-finite ``t x t`` tiles.

    Candidates are scored by finite fraction; those below ``min_finite_frac`` are
    rejected (default 1.0 -> fully finite, matching the any-NaN-drop policy of the
    serving path). Selection is greedy from the most-finite candidate, skipping any
    that overlaps an already-chosen tile. Returns ``[]`` if the window is smaller
    than ``t`` or no candidate clears the threshold.
    """
    h, w = window.shape[0], window.shape[1]
    if h < t or w < t:
        return []
    rng = np.random.default_rng(seed)
    fmask = finite_mask(window)
    scored = sorted(
        (
            (float(fmask[r : r + t, c : c + t].mean()), r, c)
            for r, c in _candidate_origins(h, w, t, n_candidates, rng)
        ),
        reverse=True,
    )
    chosen: list[TileSample] = []
    for frac, r, c in scored:
        if frac < min_finite_frac:
            break
        if any(abs(r - s.row) < t and abs(c - s.col) < t for s in chosen):
            continue
        tile = window[r : r + t, c : c + t].astype(np.float32, copy=False)
        chosen.append(TileSample(row=r, col=c, finite_frac=frac, tile=tile))
        if len(chosen) >= n_tiles:
            break
    return chosen
