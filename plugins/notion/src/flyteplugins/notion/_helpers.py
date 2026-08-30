"""Helpers for building Notion property values and blocks.

Notion's API requires property values and block objects in a verbose nested
format. These helpers produce the right shape for the common types so tasks
can create and update pages without hand-rolling JSON.
"""

from __future__ import annotations

from typing import Any


def title_property(text: str) -> dict[str, Any]:
    """A `title` property value."""
    return {"title": [{"text": {"content": text}}]}


def rich_text_property(text: str) -> dict[str, Any]:
    """A `rich_text` property value."""
    return {"rich_text": [{"text": {"content": text}}]}


def number_property(value: float | int) -> dict[str, Any]:
    """A `number` property value."""
    return {"number": value}


def select_property(name: str) -> dict[str, Any]:
    """A `select` property value (the option must exist on the database)."""
    return {"select": {"name": name}}


def multi_select_property(names: list[str]) -> dict[str, Any]:
    """A `multi_select` property value (options must exist on the database)."""
    return {"multi_select": [{"name": name} for name in names]}


def checkbox_property(value: bool) -> dict[str, Any]:
    """A `checkbox` property value."""
    return {"checkbox": value}


def date_property(start: str, end: str | None = None) -> dict[str, Any]:
    """A `date` property value; `start`/`end` are ISO 8601 date strings."""
    date: dict[str, Any] = {"start": start}
    if end:
        date["end"] = end
    return {"date": date}


def url_property(url: str) -> dict[str, Any]:
    """A `url` property value."""
    return {"url": url}


def email_property(email: str) -> dict[str, Any]:
    """An `email` property value."""
    return {"email": email}


# ----------------------------------------------------------------------
# blocks
# ----------------------------------------------------------------------


def _rich_text(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": text}}]


def paragraph_block(text: str) -> dict[str, Any]:
    """A paragraph block."""
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rich_text(text)}}


def heading_block(text: str, level: int = 2) -> dict[str, Any]:
    """A heading block (`level` 1-3)."""
    key = f"heading_{min(max(level, 1), 3)}"
    return {"object": "block", "type": key, key: {"rich_text": _rich_text(text)}}


def bulleted_block(text: str) -> dict[str, Any]:
    """A bulleted list item block."""
    return {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": _rich_text(text)}}


def to_do_block(text: str, checked: bool = False) -> dict[str, Any]:
    """A to-do block."""
    return {"object": "block", "type": "to_do", "to_do": {"rich_text": _rich_text(text), "checked": checked}}


# ----------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------


def extract_title(properties: dict[str, Any]) -> str:
    """Return the text of the first `title`-typed property, if any."""
    for prop in properties.values():
        if prop.get("type") == "title":
            return extract_rich_text(prop.get("title") or [])
    return ""


def extract_rich_text(rich_text: list[dict[str, Any]]) -> str:
    """Concatenate the `plain_text` fragments of a rich-text array."""
    return "".join(fragment.get("plain_text", "") for fragment in rich_text)
