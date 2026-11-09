"""Tests for converting values against a remote task's interface, whose types are rebuilt from its registered schema."""

import dataclasses
import enum
import typing
from unittest import mock

import pydantic
import pytest
from flyteidl2.core import interface_pb2, types_pb2
from flyteidl2.task import common_pb2

from flyte._internal.runtime import convert
from flyte.models import NativeInterface
from flyte.types import TypeEngine, guess_interface
from flyte.types._interface import UnresolvedType, infer_type_from_value


class Point(pydantic.BaseModel):
    x: int
    y: int = 0


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


@dataclasses.dataclass
class DCPoint:
    x: int
    y: int = 0


def _typed_interface(outputs: typing.Optional[dict] = None, **inputs) -> interface_pb2.TypedInterface:
    def variables(types: dict) -> interface_pb2.VariableMap:
        return interface_pb2.VariableMap(
            variables=[
                interface_pb2.VariableEntry(key=k, value=interface_pb2.Variable(type=TypeEngine.to_literal_type(t)))
                for k, t in types.items()
            ]
        )

    return interface_pb2.TypedInterface(inputs=variables(inputs), outputs=variables(outputs or {}))


def _failing_guess(predicate):
    """Patch `TypeEngine.guess_python_type` to raise for the literal types matching `predicate`."""
    real = TypeEngine.guess_python_type

    def guess(lt):
        if predicate(lt):
            raise RecursionError("maximum recursion depth exceeded")
        return real(lt)

    return mock.patch.object(TypeEngine, "guess_python_type", side_effect=guess)


def _is_point(lt: types_pb2.LiteralType) -> bool:
    return lt.HasField("metadata") and "Point" in str(lt.metadata)


# ---------------------------------------------------------------------------
# guess_interface: one variable whose type cannot be rebuilt must not fail the whole interface.
# ---------------------------------------------------------------------------


def test_guess_interface_falls_back_per_variable():
    with _failing_guess(_is_point):
        iface = guess_interface(
            _typed_interface(outputs={"o0": typing.List[Point]}, p=Point, ps=typing.List[Point], n=int)
        )
    # The STRUCT that cannot be rebuilt becomes a dict, inside its containers; other variables are untouched.
    assert iface.inputs["p"][0] is dict
    assert iface.inputs["ps"][0] == typing.List[dict]
    assert iface.inputs["n"][0] is int
    assert iface.outputs["o0"] == typing.List[dict]


def test_guess_interface_unresolved_variable():
    # A non-STRUCT type that cannot be rebuilt has no structural fallback; it becomes an UnresolvedType that
    # keeps the registered literal type, and the rest of the interface is still usable.
    with _failing_guess(lambda lt: lt.WhichOneof("type") == "enum_type"):
        iface = guess_interface(_typed_interface(color=Color, n=int))
    color_type = iface.inputs["color"][0]
    assert issubclass(color_type, UnresolvedType)
    assert TypeEngine.to_literal_type(color_type).WhichOneof("type") == "enum_type"
    assert iface.inputs["n"][0] is int


@pytest.mark.asyncio
async def test_unresolved_input_raises_naming_the_variable():
    with _failing_guess(lambda lt: lt.WhichOneof("type") == "enum_type"):
        iface = guess_interface(_typed_interface(color=Color, n=int))
    with pytest.raises(TypeError, match="'color' could not be rebuilt"):
        await convert.convert_from_native_to_inputs(iface, color=Color.RED, n=1)


@pytest.mark.asyncio
async def test_fallback_struct_round_trips_as_dict():
    # The dict fallback writes the same msgpack payload as the model, and reads an output back as a dict.
    expected = await convert.convert_from_native_to_inputs(guess_interface(_typed_interface(p=Point)), p=Point(x=1))
    with _failing_guess(_is_point):
        iface = guess_interface(_typed_interface(outputs={"o0": Point}, p=Point))
    actual = await convert.convert_from_native_to_inputs(iface, p={"x": 1, "y": 0})
    assert actual.proto_inputs.literals[0].value == expected.proto_inputs.literals[0].value

    outputs = convert.Outputs(
        proto_outputs=common_pb2.Outputs(
            literals=[common_pb2.NamedLiteral(name="o0", value=actual.proto_inputs.literals[0].value)]
        )
    )
    assert await convert.convert_outputs_to_native(iface, outputs) == {"x": 1, "y": 0}


# ---------------------------------------------------------------------------
# Value-directed types: a model or dataclass instance converts through its own class.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "declared", "expected"),
    [
        (Point(x=1), Point, Point),
        (DCPoint(x=1), DCPoint, DCPoint),
        (Point(x=1), typing.Optional[Point], typing.Optional[Point]),
        ([Point(x=1), Point(x=2)], typing.List[Point], typing.List[Point]),
        ({"a": Point(x=1)}, typing.Dict[str, Point], typing.Dict[str, Point]),
        ([Point(x=1)], typing.Optional[typing.List[Point]], typing.Optional[typing.List[Point]]),
    ],
)
def test_infer_type_from_value(value, declared, expected):
    assert infer_type_from_value(value, TypeEngine.to_literal_type(declared)) == expected


@pytest.mark.parametrize(
    ("value", "declared"),
    [
        ({"x": 1}, Point),  # a dict carries no class: left to the rebuilt type
        (DCPoint(x=1), Point),  # registered by another transformer
        (Point(x=1), DCPoint),
        (None, typing.Optional[Point]),
        ([], typing.List[Point]),  # nothing to take the element type from
        ([Point(x=1), DCPoint(x=1)], typing.List[Point]),
        (Point(x=1), typing.Union[Point, int]),  # only Optional unions are followed
        (3, int),
        (Point, Point),  # the class itself, not an instance
    ],
)
def test_infer_type_from_value_declines(value, declared):
    assert infer_type_from_value(value, TypeEngine.to_literal_type(declared)) is None


@pytest.mark.asyncio
async def test_remote_inputs_convert_through_the_values_class():
    iface = guess_interface(_typed_interface(p=Point, ps=typing.List[Point], maybe=typing.Optional[Point], n=int))
    seen = {}
    real_to_literal = TypeEngine.to_literal

    async def spy(python_val, python_type, expected):
        seen[id(python_val)] = python_type
        return await real_to_literal(python_val, python_type, expected)

    p, ps = Point(x=1), [Point(x=2)]
    with mock.patch.object(TypeEngine, "to_literal", side_effect=spy):
        await convert.convert_from_native_to_inputs(iface, p=p, ps=ps, maybe=None, n=3)
    assert seen[id(p)] is Point
    assert seen[id(ps)] == typing.List[Point]


@pytest.mark.asyncio
async def test_remote_model_input_works_when_its_type_cannot_be_rebuilt():
    expected = await convert.convert_from_native_to_inputs(guess_interface(_typed_interface(p=Point)), p=Point(x=5))
    with _failing_guess(_is_point):
        iface = guess_interface(_typed_interface(p=Point))
    actual = await convert.convert_from_native_to_inputs(iface, p=Point(x=5))
    assert actual.proto_inputs.literals[0].value == expected.proto_inputs.literals[0].value


@pytest.mark.asyncio
async def test_local_interface_keeps_declared_types():
    # Only a remote interface (with registered literal types) lets the value choose its type.
    def fn(p: Point) -> None: ...

    iface = NativeInterface.from_callable(fn)
    assert iface._remote_input_types is None
    seen = []
    real_to_literal = TypeEngine.to_literal

    async def spy(python_val, python_type, expected):
        seen.append(python_type)
        return await real_to_literal(python_val, python_type, expected)

    with mock.patch.object(TypeEngine, "to_literal", side_effect=spy):
        await convert.convert_from_native_to_inputs(iface, p=Point(x=1))
    assert seen == [Point]
