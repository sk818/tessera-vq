"""Distributional and reconstruction metrics.

Phase 2 normality diagnostics: :func:`epps_pulley` (BHEP / Epps-Pulley statistic with
beta=1, implemented from the formula since scipy lacks it) and :func:`shapiro_wilk`
(scipy wrapper). The Wasserstein-1 projection metric is added in Phase 3.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import shapiro

# Default chunk size for the Epps-Pulley double sum (bounds memory, not result).
_EP_BLOCK = 2048
# Minimum sample size for a meaningful Epps-Pulley statistic.
_MIN_SAMPLES = 8


def epps_pulley(
    samples_1d: npt.ArrayLike,
    mu: float | None = None,
    sigma: float | None = None,
    *,
    block: int = _EP_BLOCK,
) -> float:
    """Epps-Pulley (BHEP, beta=1) normality statistic; larger => less Gaussian.

    By default the sample is standardised by its own mean and std (a *composite*
    normality test, which is what the isotropy diagnostic wants). Pass ``mu``/``sigma``
    to standardise against a fixed N(mu, sigma) instead. (Spec lists mu=0, sigma=1
    defaults; we use ``None`` to mean "estimate from the sample".)
    """
    x = np.asarray(samples_1d, dtype=np.float64).ravel()
    n = x.size
    if n < _MIN_SAMPLES:
        raise ValueError("epps_pulley needs at least 8 samples")
    loc = float(x.mean()) if mu is None else mu
    scale = float(x.std(ddof=0)) if sigma is None else sigma
    if scale <= 0:
        raise ValueError("epps_pulley: zero/negative scale")
    y = (x - loc) / scale
    single = float(np.exp(-(y * y) / 4.0).sum())
    pair = 0.0
    for i in range(0, n, block):
        d = y[i : i + block][:, None] - y[None, :]
        pair += float(np.exp(-(d * d) / 2.0).sum())
    return float(pair / n - np.sqrt(2.0) * single + n / np.sqrt(3.0))


def shapiro_wilk(samples_1d: npt.ArrayLike) -> tuple[float, float]:
    """Shapiro-Wilk normality test; returns ``(statistic, p_value)``.

    scipy's implementation is unreliable for n > 5000, so callers should subsample.
    """
    x = np.asarray(samples_1d, dtype=np.float64).ravel()
    res = shapiro(x)
    return float(res.statistic), float(res.pvalue)
