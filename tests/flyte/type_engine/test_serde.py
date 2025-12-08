import typing

from flyte._interface import extract_return_annotation
from flyte._internal.runtime.types_serde import transform_variable_map


def test_unnamed_typing_tuple():
    def z(a: int, b: str) -> typing.Tuple[int, str]:
        return 5, "hello world"

    result = transform_variable_map(extract_return_annotation(typing.get_type_hints(z).get("return", None)))
    assert result["o0"].type.simple == 1
    assert result["o1"].type.simple == 3


def test_regular_tuple():
    def q(a: int, b: str) -> (int, str):
        return 5, "hello world"

    result = transform_variable_map(extract_return_annotation(typing.get_type_hints(q).get("return", None)))
    assert result["o0"].type.simple == 1
    assert result["o1"].type.simple == 3


def test_single_output_new_decorator():
    def q(a: int, b: str) -> int:
        return a + len(b)

    result = transform_variable_map(extract_return_annotation(typing.get_type_hints(q).get("return", None)))
    assert result["o0"].type.simple == 1


def test_sig_files():
    from flyteidl2.core import types_pb2

    from flyte.io._file import File

    def q() -> File: ...

    result = transform_variable_map(extract_return_annotation(typing.get_type_hints(q).get("return", None)))
    assert isinstance(result["o0"].type.blob, types_pb2.BlobType)


def test_file_types():
    import typing

    from flyte.io._file import File

    svg = typing.TypeVar("svg")

    def t1() -> File[svg]: ...

    return_type = extract_return_annotation(typing.get_type_hints(t1).get("return", None))
    o0 = return_type["o0"]
    assert issubclass(o0, File)
