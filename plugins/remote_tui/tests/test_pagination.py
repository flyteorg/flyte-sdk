"""Tests for list pagination helpers."""

from __future__ import annotations

from flyteplugins.remote_tui._client import PAGE_SIZE, _slice_page


def test_slice_page_first_page_has_next():
    items = list(range(30))
    paged = _slice_page(items, page=0, page_size=PAGE_SIZE)
    assert len(paged.items) == PAGE_SIZE
    assert paged.has_next is True
    assert paged.page == 0


def test_slice_page_last_partial_page():
    items = list(range(PAGE_SIZE + 10))
    paged = _slice_page(items, page=1, page_size=PAGE_SIZE)
    assert len(paged.items) == 10
    assert paged.has_next is False


def test_slice_page_empty():
    paged = _slice_page([], page=0, page_size=PAGE_SIZE)
    assert paged.items == []
    assert paged.has_next is False
