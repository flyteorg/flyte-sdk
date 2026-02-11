from enum import Enum
from typing import Literal

import pytest

from flyte._interface import literal_to_enum
from flyte.types import TypeEngine


class Foo(Enum):
    A = "AAA"
    B = "BBB"
    C = "CCC"


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


