from enum import Enum, IntEnum, IntFlag
from typing import List, Literal

import pytest

from flyte._interface import literal_to_enum
from flyte.types import TypeEngine, TypeTransformerFailedError


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


class Severity(IntEnum):
    """A rubric: the members are ordered, and the ordering is the point."""

    NONE = 0
    MINOR = 1
    SERIOUS = 2
    BLOCKING = 3


class Perm(IntFlag):
    READ = 1
    WRITE = 2


@pytest.mark.asyncio
async def test_int_enum_roundtrips_by_name():
    """An IntEnum crosses a task boundary as its member name, like any other enum."""
    lit = TypeEngine.to_literal_type(Severity)
    assert lit.enum_type.values == ["NONE", "MINOR", "SERIOUS", "BLOCKING"]

    lv = await TypeEngine.to_literal(Severity.SERIOUS, Severity, lit)
    assert lv.scalar.primitive.string_value == "SERIOUS"

    pv = await TypeEngine.to_python_value(lv, Severity)
    assert pv is Severity.SERIOUS  # the member itself, not a bare int
    assert pv == 2 and pv + 1 == 3  # and it is still an int
    assert Severity.BLOCKING > pv  # ordering survives, which is why IntEnum was chosen


@pytest.mark.asyncio
async def test_int_enum_accepts_name_from_cli():
    """`flyte run --severity SERIOUS` passes a string, as it does for string enums."""
    lit = TypeEngine.to_literal_type(Severity)
    lv = await TypeEngine.to_literal("SERIOUS", Severity, lit)
    assert lv.scalar.primitive.string_value == "SERIOUS"
    assert await TypeEngine.to_python_value(lv, Severity) is Severity.SERIOUS


@pytest.mark.asyncio
async def test_int_enum_in_list():
    list_lit = TypeEngine.to_literal_type(List[Severity])
    lv = await TypeEngine.to_literal([Severity.NONE, Severity.BLOCKING], List[Severity], list_lit)
    assert [x.scalar.primitive.string_value for x in lv.collection.literals] == ["NONE", "BLOCKING"]
    assert await TypeEngine.to_python_value(lv, List[Severity]) == [Severity.NONE, Severity.BLOCKING]


def test_flag_enum_is_rejected_with_a_reason():
    """A composite flag member cannot be looked up by name, so it cannot come back."""
    composite = Perm.READ | Perm.WRITE
    # Python <= 3.10 leaves a composite unnamed; 3.11+ names it "READ|WRITE". Either
    # way the name does not address the member, which is the reason for the rejection.
    assert composite.name in (None, "READ|WRITE")
    with pytest.raises(KeyError):
        Perm[composite.name]

    with pytest.raises(TypeTransformerFailedError, match="Flag enum"):
        TypeEngine.to_literal_type(Perm)


def test_non_string_non_int_enum_is_still_rejected():
    class Weird(Enum):
        A = 1.5
        B = 2.5

    with pytest.raises(TypeTransformerFailedError, match="Only string-valued enums and IntEnum"):
        TypeEngine.to_literal_type(Weird)


def test_empty_enum_reports_itself_instead_of_indexerror():
    """An enum with no members used to fall out of a list comprehension."""

    class Empty(Enum):
        pass

    with pytest.raises(TypeTransformerFailedError, match="no members"):
        TypeEngine.to_literal_type(Empty)


@pytest.mark.asyncio
async def test_int_enum_alias_resolves_to_its_canonical_member():
    class Aliased(IntEnum):
        PRIMARY = 0
        SECONDARY = 1
        ALIAS = 1  # an alias of SECONDARY, not a separate member

    lit = TypeEngine.to_literal_type(Aliased)
    assert lit.enum_type.values == ["PRIMARY", "SECONDARY"]  # aliases are not members

    lv = await TypeEngine.to_literal(Aliased.ALIAS, Aliased, lit)
    assert lv.scalar.primitive.string_value == "SECONDARY"
    assert await TypeEngine.to_python_value(lv, Aliased) is Aliased.SECONDARY


def test_mixed_value_enum_is_rejected_when_it_reaches_the_wire():
    """values[0] decides the schema, so a mixed enum only fails on the offending member."""

    class Mixed(Enum):
        A = "a"
        B = 2

    TypeEngine.to_literal_type(Mixed)  # str first: the schema is accepted

    class IntFirst(Enum):
        A = 1
        B = "b"

    with pytest.raises(TypeTransformerFailedError, match="Only string-valued enums and IntEnum"):
        TypeEngine.to_literal_type(IntFirst)
