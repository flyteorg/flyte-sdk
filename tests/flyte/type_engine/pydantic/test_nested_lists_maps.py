"""Tests for nested Pydantic/dataclass types with lists, dicts, and optionals.

These tests verify the gaps identified in PR #426 where nested type detection
did not account for list[NestedModel], dict[str, NestedModel], and
Optional[NestedModel] via $ref in JSON schemas.

Ref: https://github.com/flyteorg/flyte/issues/6887
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import pytest
from pydantic import BaseModel

from flyte.types import TypeEngine

# -- Pydantic models --


class Tag(BaseModel):
    name: str
    value: int


class Item(BaseModel):
    title: str
    tags: List[Tag]


class Inventory(BaseModel):
    items: List[Item]
    metadata: Dict[str, Tag]


class Profile(BaseModel):
    name: str
    address: Optional[Tag] = None


# -- Dataclass models --


@dataclass(eq=True)
class DCTag:
    name: str
    value: int


@dataclass(eq=True)
class DCItem:
    title: str
    tags: List[DCTag]


@dataclass(eq=True)
class DCInventory:
    items: List[DCItem]
    metadata: Dict[str, DCTag]


# -- Tests: list[NestedModel] --


@pytest.mark.asyncio
async def test_pydantic_list_of_nested_models():
    """list[Tag] should roundtrip correctly through the type engine."""
    input_val = Item(title="test", tags=[Tag(name="a", value=1), Tag(name="b", value=2)])
    lit = TypeEngine.to_literal_type(Item)
    lv = await TypeEngine.to_literal(input_val, python_type=Item, expected=lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    # Initialize with list of dicts (as would come from deserialized JSON)
    v = guessed(title="test", tags=[{"name": "a", "value": 1}, {"name": "b", "value": 2}])
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv

    pv = await TypeEngine.to_python_value(new_lv, Item)
    assert pv == input_val


@pytest.mark.asyncio
async def test_dataclass_list_of_nested_models():
    """list[DCTag] should roundtrip correctly through the type engine."""
    input_val = DCItem(title="test", tags=[DCTag(name="a", value=1), DCTag(name="b", value=2)])
    lit = TypeEngine.to_literal_type(DCItem)
    lv = await TypeEngine.to_literal(input_val, python_type=DCItem, expected=lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    v = guessed(title="test", tags=[{"name": "a", "value": 1}, {"name": "b", "value": 2}])
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv

    pv = await TypeEngine.to_python_value(new_lv, DCItem)
    assert pv == input_val


@pytest.mark.asyncio
async def test_pydantic_list_of_nested_with_mixed_init():
    """Mixing dicts and objects in a list should both work."""
    guessed = TypeEngine.guess_python_type(TypeEngine.to_literal_type(Item))

    v = guessed(
        title="mixed",
        tags=[Tag(name="obj", value=1), {"name": "dict", "value": 2}],
    )
    lit = TypeEngine.to_literal_type(Item)
    lv = await TypeEngine.to_literal(v, guessed, lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, Item)
    assert pv.tags[0].name == "obj"
    assert pv.tags[1].name == "dict"


# -- Tests: dict[str, NestedModel] --


@pytest.mark.asyncio
async def test_pydantic_dict_of_nested_models():
    """dict[str, Tag] should roundtrip correctly through the type engine."""
    input_val = Inventory(
        items=[Item(title="i1", tags=[Tag(name="t", value=1)])],
        metadata={"primary": Tag(name="p", value=10), "secondary": Tag(name="s", value=20)},
    )
    lit = TypeEngine.to_literal_type(Inventory)
    lv = await TypeEngine.to_literal(input_val, python_type=Inventory, expected=lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    v = guessed(
        items=[{"title": "i1", "tags": [{"name": "t", "value": 1}]}],
        metadata={"primary": {"name": "p", "value": 10}, "secondary": {"name": "s", "value": 20}},
    )
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv

    pv = await TypeEngine.to_python_value(new_lv, Inventory)
    assert pv == input_val


@pytest.mark.asyncio
async def test_dataclass_dict_of_nested_models():
    """dict[str, DCTag] should roundtrip correctly through the type engine."""
    input_val = DCInventory(
        items=[DCItem(title="i1", tags=[DCTag(name="t", value=1)])],
        metadata={"primary": DCTag(name="p", value=10)},
    )
    lit = TypeEngine.to_literal_type(DCInventory)
    lv = await TypeEngine.to_literal(input_val, python_type=DCInventory, expected=lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    v = guessed(
        items=[{"title": "i1", "tags": [{"name": "t", "value": 1}]}],
        metadata={"primary": {"name": "p", "value": 10}},
    )
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv

    pv = await TypeEngine.to_python_value(new_lv, DCInventory)
    assert pv == input_val


# -- Tests: Optional[NestedModel] --


@pytest.mark.asyncio
async def test_pydantic_optional_nested_model_with_value():
    """Optional[Tag] with a value should roundtrip correctly."""
    input_val = Profile(name="Alice", address=Tag(name="home", value=42))
    lit = TypeEngine.to_literal_type(Profile)
    lv = await TypeEngine.to_literal(input_val, python_type=Profile, expected=lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    v = guessed(name="Alice", address={"name": "home", "value": 42})
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv

    pv = await TypeEngine.to_python_value(new_lv, Profile)
    assert pv == input_val


@pytest.mark.asyncio
async def test_pydantic_optional_nested_model_with_none():
    """Optional[Tag] with None should roundtrip correctly."""
    input_val = Profile(name="Bob", address=None)
    lit = TypeEngine.to_literal_type(Profile)
    lv = await TypeEngine.to_literal(input_val, python_type=Profile, expected=lit)

    assert lit
    assert lv

    pv = await TypeEngine.to_python_value(lv, Profile)
    assert pv.name == "Bob"
    assert pv.address is None


@pytest.mark.asyncio
async def test_pydantic_optional_nested_model_omitted():
    """Optional[Tag] should default to None when not provided."""
    guessed = TypeEngine.guess_python_type(TypeEngine.to_literal_type(Profile))
    # Should not raise TypeError for missing 'address'
    v = guessed(name="Charlie")
    assert v.name == "Charlie"
    assert v.address is None


# -- Tests: empty collections --


@pytest.mark.asyncio
async def test_pydantic_empty_list_of_nested():
    """Empty list[Tag] should roundtrip correctly."""
    input_val = Item(title="empty", tags=[])
    lit = TypeEngine.to_literal_type(Item)
    lv = await TypeEngine.to_literal(input_val, python_type=Item, expected=lit)

    assert lv

    pv = await TypeEngine.to_python_value(lv, Item)
    assert pv.title == "empty"
    assert pv.tags == []


@pytest.mark.asyncio
async def test_pydantic_empty_dict_of_nested():
    """Empty dict[str, Tag] should roundtrip correctly."""
    input_val = Inventory(
        items=[],
        metadata={},
    )
    lit = TypeEngine.to_literal_type(Inventory)
    lv = await TypeEngine.to_literal(input_val, python_type=Inventory, expected=lit)

    assert lv

    pv = await TypeEngine.to_python_value(lv, Inventory)
    assert pv.items == []
    assert pv.metadata == {}
