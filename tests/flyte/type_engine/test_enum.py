from enum import Enum
from typing import List, Literal

import pytest

from flyte._interface import literal_to_enum
from flyte.types import TypeEngine


class Foo(Enum):
    A = "AAA"
    B = "BBB"
    C = "CCC"


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@pytest.mark.asyncio
async def test_enums():
    lit = TypeEngine.to_literal_type(Foo)
    lv = await TypeEngine.to_literal(Foo.B, Foo, lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed
    v = guessed["B"]
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv
    pv = await TypeEngine.to_python_value(new_lv, Foo)
    assert pv
    assert pv == Foo.B


@pytest.mark.asyncio
async def test_literal_string_serialization():
    """Test that Literal with string values can be serialized without errors.

    Before the fix, serializing Literal["low", "medium", "high"] would fail
    because the code incorrectly tried to access the 'name' attribute on strings.
    """
    # Literal types are converted to Enums internally during task interface construction
    IntensityLiteral = Literal["low", "medium", "high"]
    Intensity = literal_to_enum(IntensityLiteral)

    # Get the literal type
    lit = TypeEngine.to_literal_type(Intensity)
    assert lit.enum_type.values == ["low", "medium", "high"]  # Enum names are uppercased

    # Test serialization with enum values (the typical case)
    for name in ["low", "medium", "high"]:
        lv = await TypeEngine.to_literal(name, Intensity, lit)
        assert lv
        # The literal should store the enum name
        assert lv.scalar.primitive.string_value == name

        # Test roundtrip conversion
        # For LiteralEnum types, to_python_value returns the string value directly
        pv = await TypeEngine.to_python_value(lv, Intensity)
        assert pv == name


@pytest.mark.asyncio
async def test_enum_assert_type_accepts_name():
    """Test that assert_type accepts enum names (e.g. 'RED') as valid values.

    When using the CLI (e.g. flyte run --c '["RED"]'), inputs are passed as strings
    matching the enum name. Previously assert_type only checked enum values ('red'),
    causing a TypeTransformerFailedError for valid enum names.
    """
    lit = TypeEngine.to_literal_type(Color)
    # Names are used on the wire for regular enums
    assert lit.enum_type.values == ["RED", "GREEN", "BLUE"]

    # Passing enum name as string should be accepted by assert_type and to_literal
    lv = await TypeEngine.to_literal("RED", Color, lit)
    assert lv.scalar.primitive.string_value == "RED"
    pv = await TypeEngine.to_python_value(lv, Color)
    assert pv == Color.RED

    # Passing an actual enum instance should still work
    lv2 = await TypeEngine.to_literal(Color.GREEN, Color, lit)
    assert lv2.scalar.primitive.string_value == "GREEN"


@pytest.mark.asyncio
async def test_enum_in_list_accepts_name():
    """Test that List[Enum] accepts enum names as strings (as passed by the CLI)."""
    list_lit = TypeEngine.to_literal_type(List[Color])
    lv = await TypeEngine.to_literal(["RED", "BLUE"], List[Color], list_lit)
    assert lv.collection.literals[0].scalar.primitive.string_value == "RED"
    assert lv.collection.literals[1].scalar.primitive.string_value == "BLUE"
