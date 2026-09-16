"""Tests for the rank-aware IO gate in clustered/jobset tasks."""

from __future__ import annotations

import pytest

from flyte._internal.runtime.io import _is_nonzero_rank_clustered_worker

_CLUSTERED_MARKERS = ("TORCHELASTIC_RUN_ID", "FLYTE_CLUSTERED_WORKER")


def _clear_markers(monkeypatch):
    for k in _CLUSTERED_MARKERS:
        monkeypatch.delenv(k, raising=False)


def test_regular_task_uploads_even_with_stray_rank(monkeypatch):
    """A non-clustered task that happens to have RANK set must NOT be gated (no data loss)."""
    _clear_markers(monkeypatch)
    monkeypatch.setenv("RANK", "3")
    assert _is_nonzero_rank_clustered_worker() is False


def test_regular_task_no_rank(monkeypatch):
    _clear_markers(monkeypatch)
    monkeypatch.delenv("RANK", raising=False)
    assert _is_nonzero_rank_clustered_worker() is False


@pytest.mark.parametrize("marker", _CLUSTERED_MARKERS)
def test_clustered_rank0_uploads(monkeypatch, marker):
    _clear_markers(monkeypatch)
    monkeypatch.setenv(marker, "1")
    monkeypatch.setenv("RANK", "0")
    assert _is_nonzero_rank_clustered_worker() is False


@pytest.mark.parametrize("marker", _CLUSTERED_MARKERS)
def test_clustered_nonzero_rank_skips(monkeypatch, marker):
    _clear_markers(monkeypatch)
    monkeypatch.setenv(marker, "1")
    monkeypatch.setenv("RANK", "2")
    assert _is_nonzero_rank_clustered_worker() is True
