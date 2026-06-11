"""Tests for the rank-aware IO gate in clustered/jobset tasks."""

from __future__ import annotations

from flyte._internal.runtime.io import _is_nonzero_rank_clustered_worker


def test_regular_task_uploads_even_with_stray_rank(monkeypatch):
    """A non-clustered task that happens to have RANK set must NOT be gated (no data loss)."""
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    monkeypatch.setenv("RANK", "3")
    assert _is_nonzero_rank_clustered_worker() is False


def test_regular_task_no_rank(monkeypatch):
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    assert _is_nonzero_rank_clustered_worker() is False


def test_clustered_rank0_uploads(monkeypatch):
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "run-1")
    monkeypatch.setenv("RANK", "0")
    assert _is_nonzero_rank_clustered_worker() is False


def test_clustered_nonzero_rank_skips(monkeypatch):
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "run-1")
    monkeypatch.setenv("RANK", "2")
    assert _is_nonzero_rank_clustered_worker() is True
