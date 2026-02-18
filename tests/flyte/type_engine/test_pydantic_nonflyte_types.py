"""
Special test cases to test the pydantic non-flyte type transformer.

1. Tuple containing TypedDict elements not round-tripping (elements stayed as Pydantic wrappers)
2. NotRequired fields broken with `from __future__ import annotations` (__required_keys__ wrong)
3. Self-referential TypedDicts failing (None placeholder used as a type)
"""

from __future__ import annotations

from typing import List, TypedDict

import pytest
from typing_extensions import NotRequired

from flyte.types._type_engine import TypeEngine


# Functional form so annotations are real types, not strings from `from __future__ import annotations`
Coord = TypedDict("Coord", {"x": float, "y": float})


class Node(TypedDict):
    value: str
    children: NotRequired[List[Node]]


class Message(TypedDict):
    text: str
    tags: NotRequired[List[str]]


@pytest.mark.asyncio
async def test_tuple_containing_typeddict_elements():
    """Tuple elements that are TypedDicts must be dicts after round-trip, not Pydantic wrappers."""
    pt = tuple[Coord, Coord]
    lt = TypeEngine.to_literal_type(pt)
    value = (Coord(x=1.0, y=2.0), Coord(x=3.0, y=4.0))

    lv = await TypeEngine.to_literal(value, pt, lt)
    result = await TypeEngine.to_python_value(lv, pt)

    assert result[0]["x"] == 1.0
    assert result[1]["y"] == 4.0


@pytest.mark.asyncio
async def test_notrequired_field_with_future_annotations():
    """NotRequired fields must be optional even with `from __future__ import annotations`."""
    pt = Message
    lt = TypeEngine.to_literal_type(pt)

    # Without optional field
    value_without = Message(text="hello")
    lv = await TypeEngine.to_literal(value_without, pt, lt)
    result = await TypeEngine.to_python_value(lv, pt)
    assert result == {"text": "hello"}
    assert "tags" not in result

    # With optional field
    value_with = Message(text="hello", tags=["a", "b"])
    lv = await TypeEngine.to_literal(value_with, pt, lt)
    result = await TypeEngine.to_python_value(lv, pt)
    assert result == {"text": "hello", "tags": ["a", "b"]}


@pytest.mark.asyncio
async def test_self_referential_typeddict():
    """Self-referential TypedDicts must round-trip correctly."""
    pt = Node
    lt = TypeEngine.to_literal_type(pt)
    value = Node(value="root", children=[Node(value="child1"), Node(value="child2")])

    lv = await TypeEngine.to_literal(value, pt, lt)
    result = await TypeEngine.to_python_value(lv, pt)

    assert result["value"] == "root"
    assert len(result["children"]) == 2
    assert result["children"][0]["value"] == "child1"
