from __future__ import annotations

import inspect
import typing
from typing import Annotated, Dict, List, NamedTuple, Tuple, TypeVar

from flyte._interface import (
    extract_return_annotation,
)


def test_tuple_basic():
    class A: ...

    def func() -> Tuple[A, int]: ...

    def func2() -> tuple[A, Annotated[int, 42]]: ...

    sig = inspect.signature(func)
    assert sig.return_annotation == Tuple[A, int]
    assert extract_return_annotation(sig.return_annotation) == {
        "o0": A,
        "o1": int,
    }

    sig = inspect.signature(func2)
    assert sig.return_annotation == tuple[A, Annotated[int, 42]]
    assert extract_return_annotation(sig.return_annotation) == {
        "o0": A,
        "o1": Annotated[int, 42],
    }

    def func3(): ...

    sig = inspect.signature(func3)
    assert extract_return_annotation(sig.return_annotation) == {}

    nt1 = NamedTuple("NT1", x_str=str, y_int=int)

    def func4() -> nt1: ...

    sig = inspect.signature(func4)
    assert extract_return_annotation(sig.return_annotation) == {
        "x_str": str,
        "y_int": int,
    }


def test_extract_only():
    nt1 = NamedTuple("NT1", x_str=str, y_int=int)

    def x() -> nt1: ...

    return_types = extract_return_annotation(typing.get_type_hints(x).get("return", None))
    assert len(return_types) == 2
    assert return_types["x_str"] is str
    assert return_types["y_int"] is int

    def t() -> List[int]: ...

    return_type = extract_return_annotation(typing.get_type_hints(t).get("return", None))
    assert len(return_type) == 1
    assert return_type["o0"]._name == "List"
    assert return_type["o0"].__origin__ is list

    def t() -> Dict[str, int]: ...

    return_type = extract_return_annotation(typing.get_type_hints(t).get("return", None))
    assert len(return_type) == 1
    assert return_type["o0"]._name == "Dict"
    assert return_type["o0"].__origin__ is dict

    def t(a: int, b: str) -> typing.Tuple[int, str]: ...

    return_type = extract_return_annotation(typing.get_type_hints(t).get("return", None))
    assert len(return_type) == 2
    assert return_type["o0"] is int
    assert return_type["o1"] is str

    def t(a: int, b: str) -> (int, str): ...

    return_type = extract_return_annotation(typing.get_type_hints(t).get("return", None))
    assert len(return_type) == 2
    assert return_type["o0"] is int
    assert return_type["o1"] is str

    def t(a: int, b: str) -> str: ...

    return_type = extract_return_annotation(typing.get_type_hints(t).get("return", None))
    assert len(return_type) == 1
    assert return_type["o0"] is str

    def t(a: int, b: str): ...

    return_type = extract_return_annotation(typing.get_type_hints(t).get("return", None))
    assert len(return_type) == 0

    def t(a: int, b: str) -> None: ...

    return_type = extract_return_annotation(typing.get_type_hints(t).get("return", None))
    assert len(return_type) == 0

    def t(a: int, b: str) -> List[int]: ...

    return_type = extract_return_annotation(typing.get_type_hints(t).get("return", None))
    assert len(return_type) == 1
    assert return_type["o0"] is List[int]

    def t(a: int, b: str) -> Dict[str, int]: ...

    return_type = extract_return_annotation(typing.get_type_hints(t).get("return", None))
    assert len(return_type) == 1
    assert return_type["o0"] is Dict[str, int]

    VST = TypeVar("VST")

    def t(a: int, b: str) -> VST:  # type: ignore
        ...

    return_type = extract_return_annotation(typing.get_type_hints(t).get("return", None))
    assert len(return_type) == 1
    assert return_type["o0"] == VST
