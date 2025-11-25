import typing
from dataclasses import replace

from flyte._docstring import Docstring
from flyte._interface import extract_return_annotation
from flyte._internal.runtime.types_serde import transform_native_to_typed_interface, transform_variable_map
from flyte.models import NativeInterface


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


def test_transform_interface_to_typed_interface_with_docstring():
    # sphinx style
    def z(a: int, b: str) -> typing.Tuple[int, str]:
        """
        function z

        :param a: foo
        :param b: bar
        :return: ramen
        """
        return 1, "hello world"

    our_interface = NativeInterface.from_callable(z)
    our_interface = replace(our_interface, docstring=Docstring(callable_=z))
    typed_interface = transform_native_to_typed_interface(our_interface)
    print(typed_interface)
    assert typed_interface.inputs.variables.get("a").description == "foo"
    assert typed_interface.inputs.variables.get("b").description == "bar"
    # Output descriptions are not parsed for unnamed returns in Sphinx style
    assert typed_interface.outputs.variables.get("o1").description == "o1"

    # # numpy style, multiple return values, shared descriptions
    # def z(a: int, b: str) -> typing.Tuple[int, str]:
    #     """
    #     function z
    #
    #     Parameters
    #     ----------
    #     a : int
    #         foo
    #     b : str
    #         bar
    #
    #     Returns
    #     -------
    #     out1, out2 : tuple
    #         ramen
    #     """
    #     return 1, "hello world"
    #
    # our_interface = transform_function_to_interface(z, Docstring(callable_=z))
    # typed_interface = transform_interface_to_typed_interface(our_interface)
    # assert typed_interface.inputs.get("a").description == "foo"
    # assert typed_interface.inputs.get("b").description == "bar"
    # assert typed_interface.outputs.get("o0").description == "ramen"
    # assert typed_interface.outputs.get("o1").description == "ramen"

    # numpy style, multiple return values, named
    # def z(a: int, b: str) -> typing.NamedTuple("NT", x_str=str, y_int=int):
    #     """
    #     function z
    #
    #     Parameters
    #     ----------
    #     a : int
    #         foo
    #     b : str
    #         bar
    #
    #     Returns
    #     -------
    #     x_str : str
    #         description for x_str
    #     y_int : int
    #         description for y_int
    #     """
    #     return 1, "hello world"
    #
    # our_interface = transform_function_to_interface(z, Docstring(callable_=z))
    # typed_interface = transform_interface_to_typed_interface(our_interface)
    # assert typed_interface.inputs.get("a").description == "foo"
    # assert typed_interface.inputs.get("b").description == "bar"
    # assert typed_interface.outputs.get("x_str").description == "description for x_str"
    # assert typed_interface.outputs.get("y_int").description == "description for y_int"
