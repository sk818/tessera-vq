"""Tests for tessera_vq.data: dataset-version normalization."""

from __future__ import annotations

from tessera_vq.data import _normalize_dataset_version


def test_normalize_dataset_version_v_prefixed() -> None:
    assert _normalize_dataset_version("v1") == "1.0"
    assert _normalize_dataset_version("v1.1") == "1.1"
    assert _normalize_dataset_version("v2") == "2.0"


def test_normalize_dataset_version_bare_number() -> None:
    assert _normalize_dataset_version("1") == "1.0"
    assert _normalize_dataset_version("1.1") == "1.1"
    assert _normalize_dataset_version("2") == "2.0"


def test_normalize_dataset_version_already_normalized() -> None:
    assert _normalize_dataset_version("1.0") == "1.0"
    assert _normalize_dataset_version("v1.0") == "1.0"
