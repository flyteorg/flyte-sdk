"""A parameterized dataclass is still a dataclass.

`Box[Colour]` is a generic *alias*, not a class, so `dataclasses.is_dataclass()`
and `dataclasses.fields()` both reject it. Before these were resolved to their
origin, such a type fell through every check in `get_transformer` and landed on
pickle -- silently, for a type Flyte can carry as a struct.
"""

import enum
from dataclasses import dataclass, field
from typing import Generic, List, Optional, TypeVar

import pytest

from flyte.types import TypeEngine

T = TypeVar("T")


class Colour(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Rung(enum.IntEnum):
    LOW = 0
    HIGH = 1


@dataclass
class Box(Generic[T]):
    value: T
    weight: float = 0.0
    tags: dict = field(default_factory=dict)


@dataclass
class Crate:
    boxed: Box[Colour]
    rung: Box[Rung]


def test_parameterized_dataclass_is_not_pickled():
    assert TypeEngine.get_transformer(Box[Colour]).name == "Object-Dataclass-Transformer"
    assert TypeEngine.to_literal_type(Box[Colour]).simple  # a STRUCT, not a blob


@pytest.mark.asyncio
async def test_parameterized_dataclass_roundtrips():
    lt = TypeEngine.to_literal_type(Box[Colour])
    val = Box(Colour.RED, 1.5, {"a": 1})
    back = await TypeEngine.to_python_value(await TypeEngine.to_literal(val, Box[Colour], lt), Box[Colour])
    assert back == val
    assert back.value is Colour.RED  # the enum member, not a string


@pytest.mark.asyncio
async def test_parameterized_dataclass_coerces_from_a_dict():
    """The dict -> dataclass coercion applies here too, defaults included."""
    lt = TypeEngine.to_literal_type(Box[Colour])
    lv = await TypeEngine.to_literal({"value": "blue"}, Box[Colour], lt)
    back = await TypeEngine.to_python_value(lv, Box[Colour])
    assert back == Box(Colour.BLUE, 0.0, {})


@pytest.mark.asyncio
async def test_int_enum_inside_a_parameterized_dataclass():
    lt = TypeEngine.to_literal_type(Box[Rung])
    back = await TypeEngine.to_python_value(await TypeEngine.to_literal(Box(Rung.HIGH), Box[Rung], lt), Box[Rung])
    assert back.value is Rung.HIGH


@pytest.mark.asyncio
async def test_nested_parameterized_fields_still_work():
    lt = TypeEngine.to_literal_type(Crate)
    val = Crate(boxed=Box(Colour.RED), rung=Box(Rung.LOW))
    back = await TypeEngine.to_python_value(await TypeEngine.to_literal(val, Crate, lt), Crate)
    assert back == val


@pytest.mark.asyncio
async def test_ordinary_generics_are_unaffected():
    """The origin resolution must not change List/Dict/Optional handling."""
    for tp, val in ((List[int], [1, 2]), (Optional[str], "x"), (dict, {"a": "b"})):
        lt = TypeEngine.to_literal_type(tp)
        assert await TypeEngine.to_python_value(await TypeEngine.to_literal(val, tp, lt), tp) == val


# --------------------------------------------------------- defensive checks


def test_helper_resolves_annotated_and_leaves_everything_else_alone():
    from typing import Annotated, Dict

    from flyte.types._type_engine import _dataclass_class

    assert _dataclass_class(Box[Colour]) is Box
    assert _dataclass_class(Annotated[Box[Colour], "note"]) is Box  # strips the annotation
    assert _dataclass_class(Box) is Box
    # anything that is not a parameterized dataclass comes back untouched, including
    # values that are not types at all -- callers pass those through here.
    for tp in (List[int], Dict[str, str], Optional[Box[Colour]], dict, None, "not a type"):
        assert _dataclass_class(tp) == tp


@pytest.mark.asyncio
async def test_parameterized_dataclass_inside_containers():
    """Union/List/Dict must keep delegating, not get swallowed by the new branch."""
    from typing import Dict

    for tp, val in (
        (Optional[Box[Colour]], Box(Colour.RED)),
        (Optional[Box[Colour]], None),
        (List[Box[Colour]], [Box(Colour.RED), Box(Colour.BLUE)]),
        (Dict[str, Box[Colour]], {"a": Box(Colour.RED)}),
    ):
        lt = TypeEngine.to_literal_type(tp)
        assert await TypeEngine.to_python_value(await TypeEngine.to_literal(val, tp, lt), tp) == val


def test_a_registered_transformer_for_the_origin_still_wins():
    """The new branch is a last resort; it must not shadow a user's transformer."""
    from flyte.types import TypeTransformer

    @dataclass
    class Parcel(Generic[T]):
        value: T

    assert TypeEngine.get_transformer(Parcel[int]).name == "Object-Dataclass-Transformer"

    class _Custom(TypeTransformer):
        def __init__(self):
            super().__init__(name="custom-parcel", t=Parcel)

        def get_literal_type(self, t):
            raise NotImplementedError

        async def to_literal(self, *a):
            raise NotImplementedError

        async def to_python_value(self, *a):
            raise NotImplementedError

    TypeEngine.register(_Custom())
    try:
        assert TypeEngine.get_transformer(Parcel).name == "custom-parcel"
        assert TypeEngine.get_transformer(Parcel[int]).name == "custom-parcel"
    finally:
        TypeEngine._REGISTRY.pop(Parcel, None)


@pytest.mark.asyncio
async def test_unencodable_payload_fails_like_its_non_generic_twin():
    """Pins an intended behaviour change: such a type used to reach pickle.

    Recognising a parameterized dataclass means it is now encoded as a struct, so a
    payload mashumaro cannot handle fails -- exactly as the non-generic equivalent
    always has, instead of silently pickling.
    """

    class Opaque:
        def __init__(self, x):
            self.x = x

    @dataclass
    class PlainBox:
        value: Opaque

    async def roundtrip(tp, val):
        lt = TypeEngine.to_literal_type(tp)
        return await TypeEngine.to_python_value(await TypeEngine.to_literal(val, tp, lt), tp)

    with pytest.raises(Exception) as generic_err:
        await roundtrip(Box[Opaque], Box(Opaque(1)))
    with pytest.raises(Exception) as plain_err:
        await roundtrip(PlainBox, PlainBox(Opaque(1)))
    assert type(generic_err.value) is type(plain_err.value)
