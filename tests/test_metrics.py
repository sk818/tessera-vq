"""Tests for tessera_vq.metrics: Epps-Pulley and Shapiro-Wilk normality diagnostics."""

import numpy as np
import pytest

from tessera_vq.metrics import epps_pulley, shapiro_wilk


def test_epps_pulley_gaussian_is_small() -> None:
    """On standard-normal data the Epps-Pulley statistic sits near its null value."""
    rng = np.random.default_rng(0)
    stat = epps_pulley(rng.standard_normal(4000))
    assert 0.0 < stat < 5.0


def test_epps_pulley_separates_gaussian_from_exponential() -> None:
    """A clearly non-Gaussian sample yields a much larger statistic."""
    rng = np.random.default_rng(1)
    gauss = epps_pulley(rng.standard_normal(4000))
    expo = epps_pulley(rng.exponential(1.0, 4000))
    assert expo > 5.0 * gauss


def test_epps_pulley_invariant_to_location_scale() -> None:
    """Composite test: shifting/scaling Gaussian data barely moves the statistic."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal(4000)
    base = epps_pulley(x)
    shifted = epps_pulley(3.0 + 10.0 * x)
    assert abs(base - shifted) < 0.5


def test_epps_pulley_rejects_tiny_samples() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        epps_pulley(np.zeros(4))


def test_shapiro_wilk_distinguishes_normal_from_skewed() -> None:
    rng = np.random.default_rng(3)
    _, p_gauss = shapiro_wilk(rng.standard_normal(3000))
    _, p_expo = shapiro_wilk(rng.exponential(1.0, 3000))
    assert p_gauss > 0.01
    assert p_expo < 0.01
