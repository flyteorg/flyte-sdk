"""Tests for property/block helpers and event normalization."""

from __future__ import annotations

from flyteplugins.notion import (
    bulleted_block,
    checkbox_property,
    date_property,
    extract_title,
    heading_block,
    number_property,
    paragraph_block,
    select_property,
    title_property,
    to_do_block,
)
from flyteplugins.notion._events import NotionEvent, events_from_pages


def test_property_helpers():
    assert title_property("x") == {"title": [{"text": {"content": "x"}}]}
    assert number_property(3) == {"number": 3}
    assert select_property("Done") == {"select": {"name": "Done"}}
    assert checkbox_property(True) == {"checkbox": True}
    assert date_property("2024-01-01", end="2024-02-01") == {"date": {"start": "2024-01-01", "end": "2024-02-01"}}


def test_block_helpers():
    assert paragraph_block("hi")["type"] == "paragraph"
    assert heading_block("hi", level=1)["type"] == "heading_1"
    assert heading_block("hi", level=9)["type"] == "heading_3"
    assert bulleted_block("hi")["type"] == "bulleted_list_item"
    assert to_do_block("task", checked=True)["to_do"]["checked"] is True


def test_extract_title():
    props = {
        "Status": {"type": "select"},
        "Name": {"type": "title", "title": [{"plain_text": "Hello "}, {"plain_text": "world"}]},
    }
    assert extract_title(props) == "Hello world"
    assert extract_title({}) == ""


def test_event_dedupe_key_changes_with_edit_time():
    e1 = NotionEvent(page_id="p1", last_edited_time="2024-06-01T00:00:00Z")
    e2 = NotionEvent(page_id="p1", last_edited_time="2024-06-01T00:00:00Z")
    e3 = NotionEvent(page_id="p1", last_edited_time="2024-06-02T00:00:00Z")
    assert e1.dedupe_key() == e2.dedupe_key()
    assert e1.dedupe_key() != e3.dedupe_key()


def test_events_from_pages():
    pages = [
        {"id": "p1", "title": "A", "url": "u1", "last_edited_time": "t1", "parent_id": "db1"},
        {"id": "p2", "title": "B", "url": "u2", "last_edited_time": "t2", "parent_id": "db1"},
    ]
    events = events_from_pages(pages, database_id="db1")
    assert len(events) == 2
    assert events[0].event_type == "page.edited"
    assert events[0].page_id == "p1"
    assert events[1].database_id == "db1"
