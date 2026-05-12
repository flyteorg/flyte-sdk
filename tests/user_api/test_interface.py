from __future__ import annotations

import enum
import inspect
import sys
from typing import Literal, Tuple, get_origin

import pytest

import flyte
from flyte.models import NativeInterface
from flyte.types import TypeEngine

env = flyte.TaskEnvironment("test")


@env.task
async def my_task(x: int) -> str:
    return f"Task {x}"


@env.task
async def my_task2(x: int, y: int = 10, z: int | None = None) -> Tuple[str, int]:
    return f"Task {x}, {y}, {z}", x + y + (z if z is not None else 0)


@env.task
async def main(n: int) -> list[str]:
    """
    Run my_task in parallel for the range of n.
    """
    return []


def test_interface() -> None:
    """
    Test the interface of the tasks.
    """
    assert my_task.interface.inputs == {"x": (int, inspect.Parameter.empty)}
    assert my_task.interface.outputs == {"o0": str}

    assert my_task2.interface.inputs == {"x": (int, inspect.Parameter.empty), "y": (int, 10), "z": (int | None, None)}
    assert my_task2.interface.outputs == {"o0": str, "o1": int}

    assert main.interface.inputs == {"n": (int, inspect.Parameter.empty)}
    assert main.interface.outputs == {"o0": list[str]}


def test_num_required_inputs() -> None:
    """
    Test the number of required inputs for the tasks.
    """
    assert my_task.interface.num_required_inputs() == 1
    assert my_task2.interface.num_required_inputs() == 1
    assert main.interface.num_required_inputs() == 1


@pytest.mark.asyncio
async def test_num_required_inputs_remote_defaults() -> None:
    """
    Test the number of required inputs for the tasks with remote defaults.
    """
    interface = NativeInterface.from_types(
        {"x": (int, inspect.Parameter.empty), "y": (int, 10), "z": (int | None, None)},
        {"o0": str, "o1": int},
    )
    assert interface.num_required_inputs() == 1

    interface = NativeInterface.from_types(
        {"x": (int, inspect.Parameter.empty), "y": (int, NativeInterface.has_default)},
        {"o0": str},
        {"y": await TypeEngine.to_literal(10, int, None)},
    )
    assert interface.num_required_inputs() == 1
    assert "y" in interface._remote_defaults


@pytest.mark.asyncio
async def test_native_interface_from_types_missing_defauls() -> None:
    with pytest.raises(ValueError):
        NativeInterface.from_types(
            {"x": (int, inspect.Parameter.empty), "y": (int, NativeInterface.has_default)},
            {"o0": str},
        )

    with pytest.raises(ValueError):
        NativeInterface.from_types(
            {"x": (int, inspect.Parameter.empty), "y": (int, NativeInterface.has_default)},
            {"o0": str},
            {},
        )

    with pytest.raises(ValueError):
        NativeInterface.from_types(
            {"x": (int, inspect.Parameter.empty), "y": (int, NativeInterface.has_default)},
            {"o0": str},
            {"x": await flyte.types.TypeEngine.to_literal(10, int, None)},
        )


@pytest.mark.asyncio
async def test_native_interface_with_union_type() -> None:
    interface = NativeInterface.from_types(
        {"x": (int | str, inspect.Parameter.empty)},
        {"o0": int},
    )
    repr = interface.__repr__()
    assert repr == "(x: int | str) -> o0: int:"
    assert interface.inputs == {"x": (int | str, inspect.Parameter.empty)}
    assert interface.outputs == {"o0": int}


Intensity = Literal["low", "medium", "high"]


def call_test(i: Intensity) -> Tuple[Intensity, str]:
    return i, f"Intensity is {i}"


def test_native_interface_literal():
    interface = NativeInterface.from_callable(call_test)
    assert interface.__repr__() == "(i: LiteralEnum) -> (o0: LiteralEnum, o1: str):"
    assert interface.inputs is not None
    assert "i" in interface.inputs
    assert get_origin(interface.inputs["i"][0]) is None
    assert issubclass(interface.inputs["i"][0], enum.Enum)

    assert get_origin(interface.outputs["o0"]) is None
    assert issubclass(interface.outputs["o0"], enum.Enum)
    assert issubclass(interface.outputs["o1"], str)


IntLiteral = Literal[1, 2, 3]


def call_test_int(i: IntLiteral) -> Tuple[IntLiteral, str]:
    return i, f"Intensity is {i}"


def test_native_interface_int_literal():
    interface = NativeInterface.from_callable(call_test_int)
    # IF python version < 3.11 Any is typing.Any
    if sys.version_info >= (3, 11):
        assert interface.__repr__() == "(i: Any) -> (o0: Any, o1: str):"
    else:
        assert interface.__repr__() == "(i: typing.Any) -> (o0: typing.Any, o1: str):"
    assert interface.inputs is not None
    assert "i" in interface.inputs
    assert get_origin(interface.inputs["i"][0]) is None

    assert get_origin(interface.outputs["o0"]) is None
    assert issubclass(interface.outputs["o1"], str)
