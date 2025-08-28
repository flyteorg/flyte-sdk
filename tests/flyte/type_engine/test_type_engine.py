import datetime
import os
import sys
import tempfile
import typing
from collections import OrderedDict
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, auto
from typing import Dict, List

import pytest
from flyteidl.core import errors_pb2, literals_pb2, types_pb2
from flyteidl.core.literals_pb2 import (
    Literal,
    LiteralCollection,
    LiteralMap,
    Primitive,
    Scalar,
)
from flyteidl.core.types_pb2 import (
    LiteralType,
    SimpleType,
)
from mashumaro.mixins.json import DataClassJSONMixin
from mashumaro.mixins.orjson import DataClassORJSONMixin
from pydantic import BaseModel
from typing_extensions import Annotated

from flyte._context import internal_ctx
from flyte.io._dataframe import DataFrame, DataFrameTransformerEngine
from flyte.io._dir import Dir
from flyte.io._file import File
from flyte.types._pickle import FlytePickle, FlytePickleTransformer
from flyte.types._type_engine import (
    BoolTransformer,
    DataclassTransformer,
    DatetimeTransformer,
    DictTransformer,
    EnumTransformer,
    FloatTransformer,
    IntTransformer,
    ListTransformer,
    LiteralsResolver,
    NoneTransformer,
    SimpleTransformer,
    StrTransformer,
    TimedeltaTransformer,
    TypeEngine,
    TypeTransformer,
    TypeTransformerFailedError,
    UnionTransformer,
    _check_and_covert_float,
    convert_mashumaro_json_schema_to_python_class,
    strict_type_hint_matching,
)

T = typing.TypeVar("T")


def test_check_and_convert_float():
    lit = literals_pb2.Literal(
        scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(float_value=3.332)),
    )
    result = _check_and_covert_float(lit)
    assert result == 3.332
    assert isinstance(result, float)

    lit = literals_pb2.Literal(
        scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=3)),
    )
    result = _check_and_covert_float(lit)
    assert result == 3.0
    assert isinstance(result, float)

    lit = literals_pb2.Literal(
        collection=literals_pb2.LiteralCollection(
            literals=[
                literals_pb2.Literal(
                    scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(float_value=3.332)),
                ),
            ],
        ),
    )
    with pytest.raises(TypeTransformerFailedError):
        _check_and_covert_float(lit)


def test_type_engine():
    t = int
    lt = TypeEngine.to_literal_type(t)
    assert lt.simple == types_pb2.SimpleType.INTEGER

    t = typing.Dict[str, typing.List[typing.Dict[str, timedelta]]]
    lt = TypeEngine.to_literal_type(t)
    assert lt.map_value_type.collection_type.map_value_type.simple == types_pb2.SimpleType.DURATION


def test_named_tuple():
    t = typing.NamedTuple("Outputs", [("x_str", str), ("y_int", int)])
    var_map = TypeEngine.named_tuple_to_variable_map(t)
    assert var_map.variables["x_str"].type.simple == types_pb2.SimpleType.STRING
    assert var_map.variables["y_int"].type.simple == types_pb2.SimpleType.INTEGER


def test_type_resolution():
    assert type(TypeEngine.get_transformer(typing.List[int])) is ListTransformer
    assert type(TypeEngine.get_transformer(typing.List)) is ListTransformer
    assert type(TypeEngine.get_transformer(list)) is ListTransformer

    assert type(TypeEngine.get_transformer(typing.Dict[str, int])) is DictTransformer
    assert type(TypeEngine.get_transformer(typing.Dict)) is DictTransformer
    assert type(TypeEngine.get_transformer(dict)) is DictTransformer
    assert type(TypeEngine.get_transformer(Annotated[dict, OrderedDict(allow_pickle=True)])) is DictTransformer

    assert type(TypeEngine.get_transformer(int)) is SimpleTransformer
    assert type(TypeEngine.get_transformer(datetime.date)) is SimpleTransformer

    # todo: fill in flyte file/dir
    # assert type(TypeEngine.get_transformer(os.PathLike)) == FlyteFilePathTransformer
    assert type(TypeEngine.get_transformer(FlytePickle)) is FlytePickleTransformer
    assert type(TypeEngine.get_transformer(typing.Any)) is FlytePickleTransformer


@pytest.mark.asyncio
async def test_simple_transformer():
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    lit = await DatetimeTransformer.to_literal(now, datetime.datetime, LiteralType(simple=SimpleType.DATETIME))
    pv = await DatetimeTransformer.to_python_value(lit, datetime.datetime)
    assert now == pv

    val = 42
    lit = await IntTransformer.to_literal(val, int, LiteralType(simple=SimpleType.INTEGER))
    pv = await IntTransformer.to_python_value(lit, int)
    assert val == pv

    val = 3.14159
    lit = await FloatTransformer.to_literal(val, float, LiteralType(simple=SimpleType.FLOAT))
    pv = await FloatTransformer.to_python_value(lit, float)
    assert val == pytest.approx(pv)

    val = True
    lit = await BoolTransformer.to_literal(val, bool, LiteralType(simple=SimpleType.BOOLEAN))
    pv = await BoolTransformer.to_python_value(lit, bool)
    assert val == pv

    val = "flyte"
    lit = await StrTransformer.to_literal(val, str, LiteralType(simple=SimpleType.STRING))
    pv = await StrTransformer.to_python_value(lit, str)
    assert val == pv

    val = datetime.timedelta(hours=5, minutes=30)
    lit = await TimedeltaTransformer.to_literal(val, datetime.timedelta, LiteralType(simple=SimpleType.DURATION))
    pv = await TimedeltaTransformer.to_python_value(lit, datetime.timedelta)
    assert val == pv

    val = None
    lit = await NoneTransformer.to_literal(val, type(None), LiteralType(simple=SimpleType.NONE))
    pv = await NoneTransformer.to_python_value(lit, type(None))
    assert pv is None


# def test_file_formats_getting_literal_type():
#     transformer = TypeEngine.get_transformer(File)
#
#     lt = transformer.get_literal_type(File)
#     assert lt.blob.format == ""
#
#     # Works with formats that we define
#     lt = transformer.get_literal_type(File["txt"])
#     assert lt.blob.format == "txt"
#
#     lt = transformer.get_literal_type(File[typing.TypeVar("jpg")])
#     assert lt.blob.format == "jpg"
#
#     # Empty default to the default
#     lt = transformer.get_literal_type(File)
#     assert lt.blob.format == ""
#
#     lt = transformer.get_literal_type(File[typing.TypeVar(".png")])
#     assert lt.blob.format == "png"


@pytest.mark.asyncio
async def test_file_format_getting_python_value():
    transformer = TypeEngine.get_transformer(File)

    temp_dir = tempfile.mkdtemp(prefix="temp_example_")
    file_path = os.path.join(temp_dir, "file.txt")
    with open(file_path, "w") as file1:  # noqa: ASYNC230
        file1.write("hello world")
    lv = Literal(
        scalar=Scalar(
            blob=literals_pb2.Blob(
                metadata=literals_pb2.BlobMetadata(
                    type=types_pb2.BlobType(format="txt", dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE)
                ),
                uri=file_path,
            )
        )
    )

    pv = await transformer.to_python_value(lv, expected_python_type=File)
    assert isinstance(pv, File)
    # assert pv.extension() == "txt"


@pytest.mark.asyncio
async def test_list_of_dict_getting_python_value():
    transformer = TypeEngine.get_transformer(typing.List)
    lv = literals_pb2.Literal(
        collection=LiteralCollection(
            literals=[Literal(map=LiteralMap(literals={"foo": Literal(scalar=Scalar(primitive=Primitive(integer=1)))}))]
        )
    )

    pv = await transformer.to_python_value(lv, expected_python_type=typing.List[typing.Dict[str, int]])
    assert isinstance(pv, list)


@pytest.mark.asyncio
async def test_generic_backwards():
    from google.protobuf import json_format, struct_pb2

    @dataclass
    class Foo: ...

    generic = json_format.Parse("{}", struct_pb2.Struct())
    lv = Literal(collection=LiteralCollection(literals=[Literal(scalar=Scalar(generic=generic))]))

    transformer = TypeEngine.get_transformer(List[Foo])
    v = await transformer.to_python_value(lv, expected_python_type=List[Foo])
    assert isinstance(v, list)
    assert isinstance(v[0], Foo)


@pytest.mark.asyncio
async def test_list_of_single_dataclass():
    @dataclass
    class Bar:
        v: typing.Optional[typing.List[int]]
        w: typing.Optional[typing.List[float]]

    @dataclass
    class Foo:
        a: typing.Optional[typing.List[str]]
        b: Bar

    foo = Foo(a=["abc", "def"], b=Bar(v=[1, 2, 99], w=[3.1415, 2.7182]))

    lt = TypeEngine.to_literal_type(Foo)
    lv = await TypeEngine.to_literal(foo, Foo, lt)
    print(lv)

    llv = Literal(collection=LiteralCollection(literals=[lv]))

    transformer = TypeEngine.get_transformer(typing.List)
    pv = await transformer.to_python_value(llv, expected_python_type=typing.List[Foo])
    assert pv[0].a == ["abc", "def"]
    assert pv[0].b == Bar(v=[1, 2, 99], w=[3.1415, 2.7182])


@pytest.mark.asyncio
async def test_list_of_dataclassjsonmixin_getting_python_value():
    from mashumaro.jsonschema import build_json_schema

    @dataclass
    class Bar(DataClassJSONMixin):
        v: typing.Union[int, None]
        w: typing.Optional[str]
        x: float
        y: str
        z: typing.Dict[str, bool]

    @dataclass
    class Foo(DataClassJSONMixin):
        u: typing.Optional[int]
        v: typing.Optional[int]
        w: int
        x: typing.List[int]
        y: typing.Dict[str, str]
        z: Bar

    foo = Foo(
        u=5,
        v=None,
        w=1,
        x=[1],
        y={"hello": "10"},
        z=Bar(v=3, w=None, x=1.0, y="hello", z={"world": False}),
    )
    llt = TypeEngine.to_literal_type(Foo)
    lv = await TypeEngine.to_literal(foo, Foo, llt)
    lv = Literal(collection=LiteralCollection(literals=[lv]))

    transformer = TypeEngine.get_transformer(typing.List)

    schema = build_json_schema(typing.cast(DataClassJSONMixin, Foo)).to_dict()
    foo_class = convert_mashumaro_json_schema_to_python_class(schema, "FooSchema")

    guessed_pv = await transformer.to_python_value(lv, expected_python_type=typing.List[foo_class])
    pv = await transformer.to_python_value(lv, expected_python_type=typing.List[Foo])
    assert isinstance(guessed_pv, list)
    assert guessed_pv[0].u == pv[0].u
    assert guessed_pv[0].v == pv[0].v
    assert guessed_pv[0].w == pv[0].w
    assert guessed_pv[0].x == pv[0].x
    assert guessed_pv[0].y == pv[0].y
    assert guessed_pv[0].z.x == pv[0].z.x
    assert type(guessed_pv[0].u) is int
    assert guessed_pv[0].v is None
    assert type(guessed_pv[0].w) is int
    assert type(guessed_pv[0].z.v) is int
    assert type(guessed_pv[0].z.x) is float
    assert guessed_pv[0].z.v == pv[0].z.v
    assert guessed_pv[0].z.y == pv[0].z.y
    assert guessed_pv[0].z.z == pv[0].z.z
    # todo: add this if needed outside of test
    # assert pv[0] == dataclass_from_dict(Foo, asdict(guessed_pv[0]))
    # assert dataclasses.is_dataclass(foo_class)


def test_dict_type():
    lt = TypeEngine.to_literal_type(dict)
    print(lt)


@pytest.mark.asyncio
async def test_dict_transformer(ctx_with_test_raw_data_path):
    import flyte

    await flyte.init.aio()
    d = DictTransformer()

    untyped_dict_lt = TypeEngine.to_literal_type(dict)

    def assert_struct(lit: LiteralType):
        assert lit is not None
        assert lit.simple == SimpleType.STRUCT

    def recursive_assert(
        lt: LiteralType,
        expected: LiteralType,
        expected_depth: int = 1,
        curr_depth: int = 0,
    ):
        assert curr_depth <= expected_depth
        assert lt is not None
        if not lt.HasField("map_value_type"):
            assert lt == expected
            return
        recursive_assert(lt.map_value_type, expected, expected_depth, curr_depth + 1)

    # Type inference
    assert_struct(d.get_literal_type(dict))
    assert_struct(d.get_literal_type(Annotated[dict, OrderedDict(allow_pickle=True)]))
    assert_struct(d.get_literal_type(typing.Dict[int, int]))
    recursive_assert(d.get_literal_type(typing.Dict[str, str]), LiteralType(simple=SimpleType.STRING))
    recursive_assert(
        d.get_literal_type(typing.Dict[str, int]),
        LiteralType(simple=SimpleType.INTEGER),
    )
    recursive_assert(
        d.get_literal_type(typing.Dict[str, datetime.datetime]),
        LiteralType(simple=SimpleType.DATETIME),
    )
    recursive_assert(
        d.get_literal_type(typing.Dict[str, datetime.timedelta]),
        LiteralType(simple=SimpleType.DURATION),
    )
    recursive_assert(
        d.get_literal_type(typing.Dict[str, datetime.date]),
        LiteralType(simple=SimpleType.DATETIME),
    )
    recursive_assert(
        d.get_literal_type(typing.Dict[str, dict]),
        untyped_dict_lt,
    )
    recursive_assert(
        d.get_literal_type(typing.Dict[str, typing.Dict[str, str]]),
        LiteralType(simple=SimpleType.STRING),
        expected_depth=2,
    )
    recursive_assert(
        d.get_literal_type(typing.Dict[str, typing.Dict[int, str]]),
        untyped_dict_lt,
        expected_depth=2,
    )
    recursive_assert(
        d.get_literal_type(typing.Dict[str, typing.Dict[str, typing.Dict[str, str]]]),
        LiteralType(simple=SimpleType.STRING),
        expected_depth=3,
    )
    recursive_assert(
        d.get_literal_type(typing.Dict[str, typing.Dict[str, typing.Dict[str, dict]]]),
        untyped_dict_lt,
        expected_depth=3,
    )
    recursive_assert(
        d.get_literal_type(typing.Dict[str, typing.Dict[str, typing.Dict[int, dict]]]),
        untyped_dict_lt,
        expected_depth=2,
    )

    lit = await d.to_literal({}, typing.Dict, untyped_dict_lt)
    pv = await d.to_python_value(lit, typing.Dict)
    assert pv == {}

    lit_empty = Literal(map=LiteralMap(literals={}))
    pv_empty = await d.to_python_value(lit_empty, typing.Dict[str, str])
    assert pv_empty == {}

    # Literal to python
    with pytest.raises(TypeError):
        await d.to_python_value(Literal(scalar=Scalar(primitive=Primitive(integer=10))), dict)
    with pytest.raises(TypeError):
        await d.to_python_value(Literal(), dict)
    with pytest.raises(TypeError):
        await d.to_python_value(Literal(map=LiteralMap(literals={"x": None})), dict)
    with pytest.raises(TypeError):
        await d.to_python_value(Literal(map=LiteralMap(literals={"x": None})), typing.Dict[int, str])

    with pytest.raises(TypeError):
        await d.to_literal(
            {"x": datetime.datetime(2024, 5, 5)},
            dict,
            LiteralType(simple=SimpleType.STRUCT),
        )

    lv = await d.to_literal(
        {"x": datetime.datetime(2024, 5, 5)},
        Annotated[dict, OrderedDict(allow_pickle=True)],
        LiteralType(simple=SimpleType.STRUCT),
    )
    assert lv.metadata["format"] == "pickle"
    assert await d.to_python_value(lv, dict) == {"x": datetime.datetime(2024, 5, 5)}

    await d.to_python_value(
        Literal(map=LiteralMap(literals={"x": Literal(scalar=Scalar(primitive=Primitive(integer=1)))})),
        typing.Dict[str, int],
    )

    lv = await d.to_literal(
        {"x": "hello"},
        dict,
        LiteralType(simple=SimpleType.STRUCT),
    )

    assert await d.to_python_value(lv, dict) == {"x": "hello"}


def test_convert_mashumaro_json_schema_to_python_class():
    from mashumaro.mixins.json import DataClassJSONMixin
    from pydantic.dataclasses import dataclasses

    @dataclass
    class Foo(DataClassJSONMixin):
        x: int
        y: str

    from mashumaro.jsonschema import build_json_schema

    schema = build_json_schema(typing.cast(DataClassJSONMixin, Foo)).to_dict()
    foo_class = convert_mashumaro_json_schema_to_python_class(schema, "FooSchema")
    foo = foo_class(x=1, y="hello")
    foo.x = 2
    assert foo.x == 2
    assert foo.y == "hello"
    with pytest.raises(AttributeError):
        _ = foo.c
    assert dataclasses.is_dataclass(foo_class)


@pytest.mark.asyncio
async def test_list_transformer():
    l0 = Literal(scalar=Scalar(primitive=Primitive(integer=3)))
    l1 = Literal(scalar=Scalar(primitive=Primitive(integer=4)))
    lc = LiteralCollection(literals=[l0, l1])
    lit = Literal(collection=lc)

    xx = await TypeEngine.to_python_value(lit, typing.List[int])
    assert xx == [3, 4]


@pytest.mark.asyncio
async def test_protos():
    pb = errors_pb2.ContainerError(code="code", message="message")
    lt = TypeEngine.to_literal_type(errors_pb2.ContainerError)
    assert lt.simple == SimpleType.STRUCT
    assert lt.metadata["pb_type"] == "flyteidl.core.errors_pb2.ContainerError"

    lit = await TypeEngine.to_literal(pb, errors_pb2.ContainerError, lt)
    new_python_val = await TypeEngine.to_python_value(lit, errors_pb2.ContainerError)
    assert new_python_val == pb

    # Test error
    l0 = Literal(scalar=Scalar(primitive=Primitive(integer=4)))
    with pytest.raises(AssertionError):
        await TypeEngine.to_python_value(l0, errors_pb2.ContainerError)

    default_proto = errors_pb2.ContainerError()
    lit = await TypeEngine.to_literal(default_proto, errors_pb2.ContainerError, lt)
    assert lit.HasField("scalar")
    assert lit.scalar.HasField("generic")
    new_python_val = await TypeEngine.to_python_value(lit, errors_pb2.ContainerError)
    assert new_python_val == default_proto


def test_guessing_basic():
    b = types_pb2.LiteralType(simple=types_pb2.SimpleType.BOOLEAN)
    pt = TypeEngine.guess_python_type(b)
    assert pt is bool

    lt = types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER)
    pt = TypeEngine.guess_python_type(lt)
    assert pt is int

    lt = types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING)
    pt = TypeEngine.guess_python_type(lt)
    assert pt is str

    lt = types_pb2.LiteralType(simple=types_pb2.SimpleType.DURATION)
    pt = TypeEngine.guess_python_type(lt)
    assert pt is timedelta

    lt = types_pb2.LiteralType(simple=types_pb2.SimpleType.DATETIME)
    pt = TypeEngine.guess_python_type(lt)
    assert pt is datetime.datetime

    lt = types_pb2.LiteralType(simple=types_pb2.SimpleType.FLOAT)
    pt = TypeEngine.guess_python_type(lt)
    assert pt is float

    lt = types_pb2.LiteralType(simple=types_pb2.SimpleType.NONE)
    pt = TypeEngine.guess_python_type(lt)
    assert pt is type(None)

    lt = types_pb2.LiteralType(
        blob=types_pb2.BlobType(
            format=FlytePickleTransformer.PYTHON_PICKLE_FORMAT,
            dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE,
        )
    )
    # TODO: fix it: AssertionError: assert <class 'flyte.io._structured_dataset._structured_dataset.StructuredDataset'>
    #  is FlytePickle
    # pt = TypeEngine.guess_python_type(lt)
    # assert pt is FlytePickle


def test_guessing_containers():
    b = types_pb2.LiteralType(simple=types_pb2.SimpleType.BOOLEAN)
    lt = types_pb2.LiteralType(collection_type=b)
    pt = TypeEngine.guess_python_type(lt)
    assert pt == typing.List[bool]

    dur = types_pb2.LiteralType(simple=types_pb2.SimpleType.DURATION)
    lt = types_pb2.LiteralType(map_value_type=dur)
    pt = TypeEngine.guess_python_type(lt)
    assert pt == typing.Dict[str, timedelta]


@pytest.mark.asyncio
async def test_zero_floats():
    l0 = Literal(scalar=Scalar(primitive=Primitive(integer=0)))
    l1 = Literal(scalar=Scalar(primitive=Primitive(float_value=0.0)))

    assert await TypeEngine.to_python_value(l0, float) == 0
    assert await TypeEngine.to_python_value(l1, float) == 0


def test_dataclass_transformer_with_dataclassjsonmixin():
    @dataclass
    class InnerStruct(DataClassJSONMixin):
        a: int
        b: typing.Optional[str]
        c: typing.List[int]

    @dataclass
    class TestStruct(DataClassJSONMixin):
        s: InnerStruct
        m: typing.Dict[str, str]

    schema = {
        "type": "object",
        "title": "TestStruct",
        "properties": {
            "s": {
                "type": "object",
                "title": "InnerStruct",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "c": {"type": "array", "items": {"type": "integer"}},
                },
                "additionalProperties": False,
                "required": ["a", "b", "c"],
            },
            "m": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "propertyNames": {"type": "string"},
            },
        },
        "additionalProperties": False,
        "required": ["s", "m"],
    }

    tf = DataclassTransformer()
    t = tf.get_literal_type(TestStruct)
    assert t is not None
    assert t.simple is not None
    assert t.simple == SimpleType.STRUCT
    assert t.metadata is not None
    assert t.metadata == schema

    t = TypeEngine.to_literal_type(TestStruct)
    assert t is not None
    assert t.simple is not None
    assert t.simple == SimpleType.STRUCT
    assert t.metadata is not None
    assert t.metadata == schema


# @mock.patch("flytekit.core.data_persistence.FileAccessProvider.async_put_data")
# def test_dataclass_with_postponed_annotation(mock_put_data):
#     remote_path = "s3://tmp/file"
#     mock_put_data.return_value = remote_path
#
#     @dataclass
#     class Data:
#         a: int
#         f: "FlyteFile"
#
#     ctx = internal_ctx()
#     tf = DataclassTransformer()
#     t = tf.get_literal_type(Data)
#     assert t.simple == SimpleType.STRUCT
#     with tempfile.TemporaryDirectory() as tmp:
#         test_file = os.path.join(tmp, "abc.txt")
#         with open(test_file, "w") as f:
#             f.write("123")
#
#         pv = Data(a=1, f=FlyteFile(test_file, remote_path=remote_path))
#         lt = tf.to_literal(ctx, pv, Data, t)
#         msgpack_bytes = lt.scalar.binary.value
#         dict_obj = msgpack.loads(msgpack_bytes)
#         assert dict_obj["f"]["path"] == remote_path
#
#
# @mock.patch("flytekit.core.data_persistence.FileAccessProvider.async_put_data")
# def test_optional_flytefile_in_dataclass(mock_upload_dir):
#     mock_upload_dir.return_value = True
#
#     @dataclass
#     class A(DataClassJsonMixin):
#         a: int
#
#     @dataclass
#     class TestFileStruct(DataClassJsonMixin):
#         a: FlyteFile
#         b: typing.Optional[FlyteFile]
#         b_prime: typing.Optional[FlyteFile]
#         c: typing.Union[FlyteFile, None]
#         c_prime: typing.Union[None, int, bool, FlyteFile]
#         d: typing.List[FlyteFile]
#         e: typing.List[typing.Optional[FlyteFile]]
#         e_prime: typing.List[typing.Optional[FlyteFile]]
#         f: typing.Dict[str, FlyteFile]
#         g: typing.Dict[str, typing.Optional[FlyteFile]]
#         g_prime: typing.Dict[str, typing.Optional[FlyteFile]]
#         h: typing.Optional[FlyteFile] = None
#         h_prime: typing.Optional[FlyteFile] = None
#         i: typing.Optional[A] = None
#         i_prime: typing.Optional[A] = field(default_factory=lambda: A(a=99))
#         j: typing.Union[int, FlyteFile] = 0
#
#     remote_path = "s3://tmp/file"
#     # set the return value to the remote path since that's what put_data does
#     mock_upload_dir.return_value = remote_path
#     with tempfile.TemporaryFile() as f:
#         f.write(b"abc")
#         f1 = FlyteFile("f1", remote_path=remote_path)
#         o = TestFileStruct(
#             a=f1,
#             b=f1,
#             b_prime=None,
#             c=f1,
#             c_prime=f1,
#             d=[f1],
#             e=[f1],
#             e_prime=[None],
#             f={"a": f1},
#             g={"a": f1},
#             g_prime={"a": None},
#             h=f1,
#             i=A(a=42),
#             j=remote_path,
#         )
#
#         ctx = internal_ctx()
#         tf = DataclassTransformer()
#         lt = tf.get_literal_type(TestFileStruct)
#         lv = tf.to_literal(ctx, o, TestFileStruct, lt)
#
#         msgpack_bytes = lv.scalar.binary.value
#         dict_obj = msgpack.loads(msgpack_bytes)
#
#         assert dict_obj["a"]["path"] == remote_path
#         assert dict_obj["b"]["path"] == remote_path
#         assert dict_obj["b_prime"] is None
#         assert dict_obj["c"]["path"] == remote_path
#         assert dict_obj["c_prime"]["path"] == remote_path
#         assert dict_obj["d"][0]["path"] == remote_path
#         assert dict_obj["e"][0]["path"] == remote_path
#         assert dict_obj["e_prime"][0] is None
#         assert dict_obj["f"]["a"]["path"] == remote_path
#         assert dict_obj["g"]["a"]["path"] == remote_path
#         assert dict_obj["g_prime"]["a"] is None
#         assert dict_obj["h"]["path"] == remote_path
#         assert dict_obj["h_prime"] is None
#         assert dict_obj["i"]["a"] == 42
#         assert dict_obj["i_prime"]["a"] == 99
#         assert dict_obj["j"]["path"] == remote_path
#
#         ot = tf.to_python_value(ctx, lv=lv, expected_python_type=TestFileStruct)
#
#         assert o.a.remote_path == ot.a.remote_source
#         assert o.b.remote_path == ot.b.remote_source
#         assert ot.b_prime is None
#         assert o.c.remote_path == ot.c.remote_source
#         assert o.c_prime.remote_path == ot.c_prime.remote_source
#         assert o.d[0].remote_path == ot.d[0].remote_source
#         assert o.e[0].remote_path == ot.e[0].remote_source
#         assert o.e_prime == [None]
#         assert o.f["a"].remote_path == ot.f["a"].remote_source
#         assert o.g["a"].remote_path == ot.g["a"].remote_source
#         assert o.g_prime == {"a": None}
#         assert o.h.remote_path == ot.h.remote_source
#         assert ot.h_prime is None
#         assert o.i == ot.i
#         assert o.i_prime == A(a=99)
#         assert o.j == FlyteFile(remote_path)
#
#
# @mock.patch("flytekit.core.data_persistence.FileAccessProvider.async_put_data")
# def test_optional_flytefile_in_dataclassjsonmixin(mock_upload_dir):
#     @dataclass
#     class A_optional_flytefile(DataClassJSONMixin):
#         a: int
#
#     @dataclass
#     class TestFileStruct_optional_flytefile(DataClassJSONMixin):
#         a: FlyteFile
#         b: typing.Optional[FlyteFile]
#         b_prime: typing.Optional[FlyteFile]
#         c: typing.Union[FlyteFile, None]
#         d: typing.List[FlyteFile]
#         e: typing.List[typing.Optional[FlyteFile]]
#         e_prime: typing.List[typing.Optional[FlyteFile]]
#         f: typing.Dict[str, FlyteFile]
#         g: typing.Dict[str, typing.Optional[FlyteFile]]
#         g_prime: typing.Dict[str, typing.Optional[FlyteFile]]
#         h: typing.Optional[FlyteFile] = None
#         h_prime: typing.Optional[FlyteFile] = None
#         i: typing.Optional[A_optional_flytefile] = None
#         i_prime: typing.Optional[A_optional_flytefile] = field(default_factory=lambda: A_optional_flytefile(a=99))
#
#     remote_path = "s3://tmp/file"
#     mock_upload_dir.return_value = remote_path
#
#     with tempfile.TemporaryFile() as f:
#         f.write(b"abc")
#         f1 = FlyteFile("f1", remote_path=remote_path)
#         o = TestFileStruct_optional_flytefile(
#             a=f1,
#             b=f1,
#             b_prime=None,
#             c=f1,
#             d=[f1],
#             e=[f1],
#             e_prime=[None],
#             f={"a": f1},
#             g={"a": f1},
#             g_prime={"a": None},
#             h=f1,
#             i=A_optional_flytefile(a=42),
#         )
#
#         ctx = internal_ctx()
#         tf = DataclassTransformer()
#         lt = tf.get_literal_type(TestFileStruct_optional_flytefile)
#         lv = tf.to_literal(ctx, o, TestFileStruct_optional_flytefile, lt)
#
#         msgpack_bytes = lv.scalar.binary.value
#         dict_obj = msgpack.loads(msgpack_bytes)
#
#         assert dict_obj["a"]["path"] == remote_path
#         assert dict_obj["b"]["path"] == remote_path
#         assert dict_obj["b_prime"] is None
#         assert dict_obj["c"]["path"] == remote_path
#         assert dict_obj["d"][0]["path"] == remote_path
#         assert dict_obj["e"][0]["path"] == remote_path
#         assert dict_obj["e_prime"][0] is None
#         assert dict_obj["f"]["a"]["path"] == remote_path
#         assert dict_obj["g"]["a"]["path"] == remote_path
#         assert dict_obj["g_prime"]["a"] is None
#         assert dict_obj["h"]["path"] == remote_path
#         assert dict_obj["h_prime"] is None
#         assert dict_obj["i"]["a"] == 42
#         assert dict_obj["i_prime"]["a"] == 99
#
#         ot = tf.to_python_value(ctx, lv=lv, expected_python_type=TestFileStruct_optional_flytefile)
#
#         assert o.a.remote_path == ot.a.remote_source
#         assert o.b.remote_path == ot.b.remote_source
#         assert ot.b_prime is None
#         assert o.c.remote_path == ot.c.remote_source
#         assert o.d[0].remote_path == ot.d[0].remote_source
#         assert o.e[0].remote_path == ot.e[0].remote_source
#         assert o.e_prime == [None]
#         assert o.f["a"].remote_path == ot.f["a"].remote_source
#         assert o.g["a"].remote_path == ot.g["a"].remote_source
#         assert o.g_prime == {"a": None}
#         assert o.h.remote_path == ot.h.remote_source
#         assert ot.h_prime is None
#         assert o.i == ot.i
#         assert o.i_prime == A_optional_flytefile(a=99)


@pytest.mark.asyncio
async def test_flyte_file_in_dataclassjsonmixin():
    # There's an equality check below, but ordering might be off so override to compare correctly.
    @dataclass(eq=False)
    class TestInnerFileStruct(DataClassJSONMixin):
        b: File
        c: typing.Dict[str, File]
        d: typing.List[File]
        e: typing.Dict[str, File]

        __hash__ = None  # Explicit hashing disabled

        def __eq__(self, other):
            return self.b == other.b and self.c == other.c and self.d == other.d and self.e == other.e

    @dataclass(eq=False)
    class TestFileStruct:
        a: File
        b: TestInnerFileStruct

        __hash__ = None  # Explicit hashing disabled

        def __eq__(self, other):
            return self.a == other.a and self.b == other.b

    remote_path = "s3://tmp/file.txt"
    f1 = File.from_existing_remote(remote_path=remote_path)
    f2 = File(path="/tmp/file")
    f2._remote_source = remote_path
    o = TestFileStruct(
        a=f1,
        b=TestInnerFileStruct(
            b=f1,
            c={"hello": f1},
            d=[f2],
            e={"hello": f2},
        ),
    )

    tf = DataclassTransformer()
    lt = tf.get_literal_type(TestFileStruct)
    lv = await tf.to_literal(o, TestFileStruct, lt)

    rehydrated_pt = tf.guess_python_type(lt)
    ot = await tf.to_python_value(lv=lv, expected_python_type=rehydrated_pt)
    assert o == ot


@pytest.mark.asyncio
async def test_flyte_directory_in_dataclassjsonmixin():
    @dataclass(eq=False)
    class TestInnerFileStruct:
        b: Dir
        c: typing.Dict[str, Dir]
        d: typing.List[Dir]
        e: typing.Dict[str, Dir]

        __hash__ = None  # Explicit hashing disabled

        def __eq__(self, other):
            return self.b == other.b and self.c == other.c and self.d == other.d and self.e == other.e

    @dataclass(eq=False)
    class TestFileStruct(DataClassJSONMixin):
        a: Dir
        b: TestInnerFileStruct

        __hash__ = None  # Explicit hashing disabled

        def __eq__(self, other):
            return self.a == other.a and self.b == other.b

    remote_path = "s3://tmp/file"
    tempdir = tempfile.mkdtemp(prefix="flyte-")
    f1 = Dir(path=tempdir)
    f1._remote_source = remote_path
    f2 = Dir(path=remote_path)
    o = TestFileStruct(
        a=f1,
        b=TestInnerFileStruct(
            b=f1,
            c={"hello": f1},
            d=[f2],
            e={"hello": f2},
        ),
    )

    tf = DataclassTransformer()
    lt = tf.get_literal_type(TestFileStruct)
    lv = await tf.to_literal(o, TestFileStruct, lt)
    assert lv.scalar.binary.tag == "msgpack"
    ot1 = await tf.to_python_value(lv=lv, expected_python_type=TestFileStruct)

    rehydrated_pt = tf.guess_python_type(lt)
    ot2 = await tf.to_python_value(lv=lv, expected_python_type=rehydrated_pt)
    assert o == ot1
    assert o == ot2


@pytest.mark.skipif("pandas" not in sys.modules, reason="Pandas is not installed.")
@pytest.mark.asyncio
async def test_structured_dataset_in_dataclass(ctx_with_test_raw_data_path):
    import pandas as pd
    from pandas._testing import assert_frame_equal
    from pydantic.dataclasses import dataclass as pyd_dataclass

    df = pd.DataFrame({"Name": ["Tom", "Joseph"], "Age": [20, 22]})
    People = Annotated[DataFrame, "parquet", OrderedDict(Name=str, Age=int)]

    @pyd_dataclass
    class InnerDatasetStruct:
        a: DataFrame
        b: typing.List[Annotated[DataFrame, "parquet"]]
        c: typing.Dict[str, Annotated[DataFrame, OrderedDict(Name=str, Age=int)]]

    @pyd_dataclass
    class DatasetStruct:
        a: People
        b: InnerDatasetStruct

    sd = DataFrame(val=df, file_format="parquet")
    o = DatasetStruct(a=sd, b=InnerDatasetStruct(a=sd, b=[sd], c={"hello": sd}))

    tf = DataclassTransformer()
    lt = tf.get_literal_type(DatasetStruct)
    lv = await tf.to_literal(o, DatasetStruct, lt)
    ot = await tf.to_python_value(lv=lv, expected_python_type=DatasetStruct)

    return_df = ot.a.open(pd.DataFrame)
    return_df = await return_df.all()
    assert_frame_equal(df, return_df)

    return_df = ot.b.a.open(pd.DataFrame)
    return_df = await return_df.all()
    assert_frame_equal(df, return_df)

    return_df = ot.b.b[0].open(pd.DataFrame)
    return_df = await return_df.all()
    assert_frame_equal(df, return_df)

    return_df = ot.b.c["hello"].open(pd.DataFrame)
    return_df = await return_df.all()
    assert_frame_equal(df, return_df)

    assert "parquet" == ot.a.file_format
    assert "parquet" == ot.b.a.file_format
    assert "parquet" == ot.b.b[0].file_format
    assert "parquet" == ot.b.c["hello"].file_format


# Enums should have string values
class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class MultiInheritanceColor(str, Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()


# Enums with integer values are not supported
class UnsupportedEnumValues(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


@pytest.mark.asyncio
@pytest.mark.skipif("pyarrow" not in sys.modules, reason="pyarrow is not installed.")
@pytest.mark.skipif("pandas" not in sys.modules, reason="Pandas is not installed.")
async def test_structured_dataset_type(ctx_with_test_raw_data_path):
    import pandas as pd
    import pyarrow as pa
    from pandas._testing import assert_frame_equal

    ctx = internal_ctx()

    name = "Name"
    age = "Age"
    data = {name: ["Tom", "Joseph"], age: [20, 22]}
    superset_cols = OrderedDict(Name=str, Age=int)
    subset_cols = OrderedDict(Name=str)
    df = pd.DataFrame(data)

    tf = TypeEngine.get_transformer(DataFrame)
    lt = tf.get_literal_type(Annotated[DataFrame, superset_cols, "parquet"])
    assert lt.structured_dataset_type is not None

    lv = await tf.to_literal(df, pd.DataFrame, lt)
    assert ctx.raw_data.path in lv.scalar.structured_dataset.uri
    metadata = lv.scalar.structured_dataset.metadata
    assert metadata.structured_dataset_type.format == "parquet"
    v1 = await tf.to_python_value(lv, pd.DataFrame)
    v2 = await tf.to_python_value(lv, pa.Table)
    assert_frame_equal(df, v1)
    assert_frame_equal(df, v2.to_pandas())

    subset_lt = tf.get_literal_type(Annotated[DataFrame, subset_cols, "parquet"])
    assert subset_lt.structured_dataset_type is not None

    subset_lv = await tf.to_literal(df, pd.DataFrame, subset_lt)
    assert ctx.raw_data.path in subset_lv.scalar.structured_dataset.uri
    v1 = await tf.to_python_value(subset_lv, pd.DataFrame)
    v2 = await tf.to_python_value(subset_lv, pa.Table)
    subset_data = pd.DataFrame({name: ["Tom", "Joseph"]})
    assert_frame_equal(subset_data, v1)
    assert_frame_equal(subset_data, v2.to_pandas())

    empty_lt = tf.get_literal_type(Annotated[DataFrame, "parquet"])
    assert empty_lt.structured_dataset_type is not None
    empty_lv = await tf.to_literal(df, pd.DataFrame, empty_lt)
    v1 = await tf.to_python_value(empty_lv, pd.DataFrame)
    v2 = await tf.to_python_value(empty_lv, pa.Table)
    assert_frame_equal(df, v1)
    assert_frame_equal(df, v2.to_pandas())


@pytest.mark.asyncio
async def test_enum_type():
    t = TypeEngine.to_literal_type(Color)
    assert t is not None
    assert t.HasField("enum_type")
    assert t.enum_type.values == [c.value for c in Color]

    g = TypeEngine.guess_python_type(t)
    assert [e.value for e in g] == [e.value for e in Color]

    lv = await TypeEngine.to_literal(Color.RED, Color, TypeEngine.to_literal_type(Color))
    assert lv
    assert lv.scalar
    assert lv.scalar.primitive.string_value == "red"

    v = await TypeEngine.to_python_value(lv, Color)
    assert v
    assert v == Color.RED

    v = await TypeEngine.to_python_value(lv, str)
    assert v
    assert v == "red"

    with pytest.raises(ValueError):
        await TypeEngine.to_python_value(
            Literal(scalar=Scalar(primitive=Primitive(string_value=str(Color.RED)))),
            Color,
        )

    with pytest.raises(ValueError):
        await TypeEngine.to_python_value(Literal(scalar=Scalar(primitive=Primitive(string_value="bad"))), Color)

    with pytest.raises(AssertionError):
        TypeEngine.to_literal_type(UnsupportedEnumValues)


def test_multi_inheritance_enum_type():
    tfm = TypeEngine.get_transformer(MultiInheritanceColor)
    assert isinstance(tfm, EnumTransformer)


def union_type_tags_unique(t: LiteralType):
    seen = set()
    for x in t.union_type.variants:
        if x.structure.tag in seen:
            return False
        seen.add(x.structure.tag)

    return True


@pytest.mark.asyncio
async def test_union_type():
    pt = typing.Union[str, int]
    lt = TypeEngine.to_literal_type(pt)
    pt_604 = str | int
    lt_604 = TypeEngine.to_literal_type(pt_604)
    assert lt == lt_604
    assert lt.union_type.variants == [
        LiteralType(simple=SimpleType.STRING, structure=types_pb2.TypeStructure(tag="str")),
        LiteralType(simple=SimpleType.INTEGER, structure=types_pb2.TypeStructure(tag="int")),
    ]
    assert union_type_tags_unique(lt)

    lv = await TypeEngine.to_literal(3, pt, lt)
    v = await TypeEngine.to_python_value(lv, pt)
    assert lv.scalar.union.type.structure.tag == "int"
    assert lv.scalar.union.value.scalar.primitive.integer == 3
    assert v == 3

    lv = await TypeEngine.to_literal("hello", pt, lt)
    v = await TypeEngine.to_python_value(lv, pt)
    assert lv.scalar.union.type.structure.tag == "str"
    assert lv.scalar.union.value.scalar.primitive.string_value == "hello"
    assert v == "hello"


def test_assert_dataclassjsonmixin_type():
    @dataclass
    class ArgsAssert(DataClassJSONMixin):
        x: int
        y: typing.Optional[str]

    @dataclass
    class SchemaArgsAssert(DataClassJSONMixin):
        x: typing.Optional[ArgsAssert]

    pt = SchemaArgsAssert
    lt = TypeEngine.to_literal_type(pt)
    gt = TypeEngine.guess_python_type(lt)
    pv = SchemaArgsAssert(x=ArgsAssert(x=3, y="hello"))
    DataclassTransformer().assert_type(gt, pv)
    DataclassTransformer().assert_type(SchemaArgsAssert, pv)

    @dataclass
    class Bar(DataClassJSONMixin):
        x: int

    pv = Bar(x=3)
    with pytest.raises(
        TypeTransformerFailedError,
        match="Type of Val '<class 'int'>' is not an instance of <class '.*.ArgsAssert'>",
    ):
        DataclassTransformer().assert_type(gt, pv)


def test_union_transformer():
    assert UnionTransformer.is_optional_type(typing.Optional[int])
    assert UnionTransformer.is_optional_type(int | None)
    assert not UnionTransformer.is_optional_type(str)
    assert UnionTransformer.get_sub_type_in_optional(typing.Optional[int]) is int
    assert UnionTransformer.get_sub_type_in_optional(int | None) is int
    assert not UnionTransformer.is_optional_type(typing.Union[int, str])
    assert UnionTransformer.is_optional_type(typing.Union[int, None])


def test_union_guess_type():
    ut = UnionTransformer()
    t = ut.guess_python_type(
        LiteralType(
            union_type=types_pb2.UnionType(
                variants=[
                    LiteralType(simple=SimpleType.STRING),
                    LiteralType(simple=SimpleType.INTEGER),
                ]
            )
        )
    )
    assert t == typing.Union[str, int]


@pytest.mark.asyncio
async def test_union_type_with_annotated():
    pt = typing.Union[
        Annotated[str, "hello"],
        Annotated[int, "world"],
    ]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.union_type.variants == [
        LiteralType(
            simple=SimpleType.STRING,
            structure=types_pb2.TypeStructure(tag="str"),
        ),
        LiteralType(
            simple=SimpleType.INTEGER,
            structure=types_pb2.TypeStructure(tag="int"),
        ),
    ]
    assert union_type_tags_unique(lt)

    lv = await TypeEngine.to_literal(3, pt, lt)
    v = await TypeEngine.to_python_value(lv, pt)
    assert lv.scalar.union.type.structure.tag == "int"
    assert lv.scalar.union.value.scalar.primitive.integer == 3
    assert v == 3

    lv = await TypeEngine.to_literal("hello", pt, lt)
    v = await TypeEngine.to_python_value(lv, pt)
    assert lv.scalar.union.type.structure.tag == "str"
    assert lv.scalar.union.value.scalar.primitive.string_value == "hello"
    assert v == "hello"


@pytest.mark.asyncio
async def test_annotated_union_type():
    pt = Annotated[typing.Union[str, int], {"hello": "world"}]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.union_type.variants == [
        LiteralType(simple=SimpleType.STRING, structure=types_pb2.TypeStructure(tag="str")),
        LiteralType(simple=SimpleType.INTEGER, structure=types_pb2.TypeStructure(tag="int")),
    ]
    assert union_type_tags_unique(lt)

    lv = await TypeEngine.to_literal(3, pt, lt)
    v = await TypeEngine.to_python_value(lv, pt)
    assert lv.scalar.union.type.structure.tag == "int"
    assert lv.scalar.union.value.scalar.primitive.integer == 3
    assert v == 3

    lv = await TypeEngine.to_literal("hello", pt, lt)
    v = await TypeEngine.to_python_value(lv, pt)
    assert lv.scalar.union.type.structure.tag == "str"
    assert lv.scalar.union.value.scalar.primitive.string_value == "hello"
    assert v == "hello"


@pytest.mark.asyncio
async def test_union_type_simple():
    pt = typing.Union[str, int]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.union_type.variants == [
        LiteralType(simple=SimpleType.STRING, structure=types_pb2.TypeStructure(tag="str")),
        LiteralType(simple=SimpleType.INTEGER, structure=types_pb2.TypeStructure(tag="int")),
    ]
    lv = await TypeEngine.to_literal(3, pt, lt)
    assert lv.scalar.HasField("union")
    assert lv.scalar.union.type.structure.tag == "int"
    assert len(lv.scalar.union.type.structure.dataclass_type) == 0


@pytest.mark.asyncio
async def test_union_containers():
    pt = typing.Union[typing.List[typing.Dict[str, typing.List[int]]], typing.Dict[str, typing.List[int]], int]
    lt = TypeEngine.to_literal_type(pt)

    list_of_maps_of_list_ints = [
        {"first_map_a": [42], "first_map_b": [42, 2]},
        {
            "second_map_c": [33],
            "second_map_d": [9, 99],
        },
    ]
    map_of_list_ints = {
        "ll_1": [1, 23, 3],
        "ll_2": [4, 5, 6],
    }
    lv = await TypeEngine.to_literal(list_of_maps_of_list_ints, pt, lt)
    assert lv.scalar.union.type.structure.tag == "Typed List"
    lv = await TypeEngine.to_literal(map_of_list_ints, pt, lt)
    assert lv.scalar.union.type.structure.tag == "Typed Dict"


@pytest.mark.asyncio
async def test_optional_type():
    pt = typing.Optional[int]
    lt = TypeEngine.to_literal_type(pt)
    pt_604 = int | None
    lt_604 = TypeEngine.to_literal_type(pt_604)
    assert lt == lt_604
    assert lt.union_type.variants == [
        LiteralType(simple=SimpleType.INTEGER, structure=types_pb2.TypeStructure(tag="int")),
        LiteralType(simple=SimpleType.NONE, structure=types_pb2.TypeStructure(tag="none")),
    ]
    assert union_type_tags_unique(lt)

    lv = await TypeEngine.to_literal(3, pt, lt)
    v = await TypeEngine.to_python_value(lv, pt)
    assert lv.scalar.union.type.structure.tag == "int"
    assert lv.scalar.union.value.scalar.primitive.integer == 3
    assert v == 3

    lv = await TypeEngine.to_literal(None, pt, lt)
    v = await TypeEngine.to_python_value(lv, pt)
    assert lv.scalar.union.type.structure.tag == "none"
    assert lv.scalar.union.value.scalar.none_type == literals_pb2.Void()
    assert v is None


@pytest.mark.asyncio
async def test_union_from_unambiguous_literal():
    from flyte.io._dir import Dir
    from flyte.io._file import File

    pt = typing.Union[str, int]
    lt = TypeEngine.to_literal_type(pt)
    assert lt.union_type.variants == [
        LiteralType(simple=SimpleType.STRING, structure=types_pb2.TypeStructure(tag="str")),
        LiteralType(simple=SimpleType.INTEGER, structure=types_pb2.TypeStructure(tag="int")),
    ]
    assert union_type_tags_unique(lt)

    lv = await TypeEngine.to_literal(3, int, lt)
    assert lv.scalar.primitive.integer == 3

    v = await TypeEngine.to_python_value(lv, pt)
    assert v == 3

    pt = typing.Union[File, Dir]
    temp_dir = tempfile.mkdtemp(prefix="temp_example_")
    file_path = os.path.join(temp_dir, "file.txt")
    with open(file_path, "w") as file1:  # noqa: ASYNC230
        file1.write("hello world")

    lt = TypeEngine.to_literal_type(File)
    lv = await TypeEngine.to_literal(File(path=file_path), File, lt)
    v = await TypeEngine.to_python_value(lv, pt)
    assert isinstance(v, File)
    lv = await TypeEngine.to_literal(v, File, lt)
    assert os.path.isfile(lv.scalar.blob.uri)

    lt = TypeEngine.to_literal_type(Dir)
    lv = await TypeEngine.to_literal(Dir(path=temp_dir), Dir, lt)
    v = await TypeEngine.to_python_value(lv, pt)
    assert isinstance(v, Dir)
    lv = await TypeEngine.to_literal(v, Dir, lt)
    assert os.path.isdir(lv.scalar.blob.uri)


# def test_union_custom_transformer():
#     class MyInt:
#         def __init__(self, x: int):
#             self.val = x
#
#         def __eq__(self, other):
#             if not isinstance(other, MyInt):
#                 return False
#             return other.val == self.val
#
#     TypeEngine.register(
#         SimpleTransformer(
#             "MyInt",
#             MyInt,
#             LiteralType(simple=SimpleType.INTEGER),
#             lambda x: Literal(scalar=Scalar(primitive=Primitive(integer=x.val))),
#             lambda x: MyInt(x.scalar.primitive.integer),
#         )
#     )
#
#     pt = typing.Union[int, MyInt]
#     lt = TypeEngine.to_literal_type(pt)
#     assert lt.union_type.variants == [
#         LiteralType(simple=SimpleType.INTEGER, structure=TypeStructure(tag="int")),
#         LiteralType(simple=SimpleType.INTEGER, structure=TypeStructure(tag="MyInt")),
#     ]
#     assert union_type_tags_unique(lt)
#
#     ctx = FlyteContextManager.current_context()
#     lv = TypeEngine.to_literal(ctx, 3, pt, lt)
#     v = TypeEngine.to_python_value(ctx, lv, pt)
#     assert lv.scalar.flyte.stored_type.structure.tag == "int"
#     assert lv.scalar.flyte.value.scalar.primitive.integer == 3
#     assert v == 3
#
#     lv = TypeEngine.to_literal(ctx, MyInt(10), pt, lt)
#     v = TypeEngine.to_python_value(ctx, lv, pt)
#     assert lv.scalar.flyte.stored_type.structure.tag == "MyInt"
#     assert lv.scalar.flyte.value.scalar.primitive.integer == 10
#     assert v == MyInt(10)
#
#     lv = TypeEngine.to_literal(ctx, 4, int, LiteralType(simple=SimpleType.INTEGER))
#     assert lv.scalar.primitive.integer == 4
#     try:
#         TypeEngine.to_python_value(ctx, lv, pt)
#     except TypeError as e:
#         assert "Ambiguous choice of variant" in str(e)
#
#     del TypeEngine._REGISTRY[MyInt]
#
#
# def test_union_custom_transformer_sanity_check():
#     class UnsignedInt:
#         def __init__(self, x: int):
#             self.val = x
#
#         def __eq__(self, other):
#             if not isinstance(other, UnsignedInt):
#                 return False
#             return other.val == self.val
#
#     # This transformer will not work in the implicit wrapping case
#     class UnsignedIntTransformer(TypeTransformer[UnsignedInt]):
#         def __init__(self):
#             super().__init__("UnsignedInt", UnsignedInt)
#
#         def get_literal_type(self, t: typing.Type[T]) -> LiteralType:
#             return LiteralType(simple=SimpleType.INTEGER)
#
#         def to_literal(
#                 self,
#                 ctx: FlyteContext,
#                 python_val: T,
#                 python_type: typing.Type[T],
#                 expected: LiteralType,
#         ) -> Literal:
#             if type(python_val) != int:
#                 raise TypeTransformerFailedError("Expected an integer")
#
#             if python_val < 0:
#                 raise TypeTransformerFailedError("Expected a non-negative integer")
#
#             return Literal(scalar=Scalar(primitive=Primitive(integer=python_val)))
#
#         def to_python_value(self, ctx: FlyteContext, lv: Literal, expected_python_type: typing.Type[T]) -> Literal:
#             val = lv.scalar.primitive.integer
#             return UnsignedInt(0 if val < 0 else val)  # type: ignore
#
#     TypeEngine.register(UnsignedIntTransformer())
#
#     pt = typing.Union[int, UnsignedInt]
#     lt = TypeEngine.to_literal_type(pt)
#     assert lt.union_type.variants == [
#         LiteralType(simple=SimpleType.INTEGER, structure=TypeStructure(tag="int")),
#         LiteralType(simple=SimpleType.INTEGER, structure=TypeStructure(tag="UnsignedInt")),
#     ]
#     assert union_type_tags_unique(lt)
#
#     ctx = FlyteContextManager.current_context()
#     with pytest.raises(TypeError, match="Ambiguous choice of variant for flyte type"):
#         TypeEngine.to_literal(ctx, 3, pt, lt)
#
#     del TypeEngine._REGISTRY[UnsignedInt]
#
#
# def test_union_of_lists():
#     pt = typing.Union[typing.List[int], typing.List[str]]
#     lt = TypeEngine.to_literal_type(pt)
#     assert lt.union_type.variants == [
#         LiteralType(
#             collection_type=LiteralType(simple=SimpleType.INTEGER),
#             structure=TypeStructure(tag="Typed List"),
#         ),
#         LiteralType(
#             collection_type=LiteralType(simple=SimpleType.STRING),
#             structure=TypeStructure(tag="Typed List"),
#         ),
#     ]
#     # Tags are deliberately NOT unique because they are not required to encode the deep type structure,
#     # only the top-level type transformer choice
#     #
#     # The stored typed will be used to differentiate flyte variants and must produce a unique choice.
#     assert not union_type_tags_unique(lt)
#
#     ctx = FlyteContextManager.current_context()
#     lv = TypeEngine.to_literal(ctx, ["hello", "world"], pt, lt)
#     v = TypeEngine.to_python_value(ctx, lv, pt)
#     assert lv.scalar.flyte.stored_type.structure.tag == "Typed List"
#     assert [x.scalar.primitive.string_value for x in lv.scalar.flyte.value.collection.literals] == ["hello", "world"]
#     assert v == ["hello", "world"]
#
#     lv = TypeEngine.to_literal(ctx, [1, 3], pt, lt)
#     v = TypeEngine.to_python_value(ctx, lv, pt)
#     assert lv.scalar.flyte.stored_type.structure.tag == "Typed List"
#     assert [x.scalar.primitive.integer for x in lv.scalar.flyte.value.collection.literals] == [1, 3]
#     assert v == [1, 3]
#
#
# @pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP604 requires >=3.10.")
# def test_list_of_unions():
#     pt = typing.List[typing.Union[str, int]]
#     lt = TypeEngine.to_literal_type(pt)
#     pt_604 = typing.List[str | int]
#     lt_604 = TypeEngine.to_literal_type(pt_604)
#     assert lt == lt_604
#     # todo(maximsmol): seems like the order here is non-deterministic
#     assert lt.collection_type.union_type.variants == [
#         LiteralType(simple=SimpleType.STRING, structure=TypeStructure(tag="str")),
#         LiteralType(simple=SimpleType.INTEGER, structure=TypeStructure(tag="int")),
#     ]
#     assert union_type_tags_unique(lt.collection_type)  # tags are deliberately NOT unique
#
#     ctx = FlyteContextManager.current_context()
#     lv = TypeEngine.to_literal(ctx, ["hello", 123, "world"], pt, lt)
#     v = TypeEngine.to_python_value(ctx, lv, pt)
#     lv_604 = TypeEngine.to_literal(ctx, ["hello", 123, "world"], pt_604, lt_604)
#     v_604 = TypeEngine.to_python_value(ctx, lv_604, pt_604)
#     assert [x.scalar.flyte.stored_type.structure.tag for x in lv.collection.literals] == ["str", "int", "str"]
#     assert v == v_604 == ["hello", 123, "world"]
#
#
# def test_pickle_type():
#     class Foo(object):
#         def __init__(self, number: int):
#             self.number = number
#
#     lt = TypeEngine.to_literal_type(FlytePickle)
#     assert lt.blob.format == FlytePickleTransformer.PYTHON_PICKLE_FORMAT
#     assert lt.blob.dimensionality == BlobType.BlobDimensionality.SINGLE
#
#     ctx = FlyteContextManager.current_context()
#     lv = TypeEngine.to_literal(ctx, Foo(1), FlytePickle, lt)
#     assert flyte_tmp_dir in lv.scalar.blob.uri
#
#     transformer = FlytePickleTransformer()
#     gt = transformer.guess_python_type(lt)
#     pv = transformer.to_python_value(ctx, lv, expected_python_type=gt)
#     assert Foo(1).number == pv.number
#
#     with pytest.raises(AssertionError, match="Cannot pickle None Value"):
#         lt = TypeEngine.to_literal_type(typing.Optional[typing.Any])
#         TypeEngine.to_literal(ctx, None, FlytePickle, lt)
#
#     with pytest.raises(
#             AssertionError,
#             match="Expected value of type <class 'NoneType'> but got '1' of type <class 'int'>",
#     ):
#         lt = TypeEngine.to_literal_type(typing.Optional[typing.Any])
#         TypeEngine.to_literal(ctx, 1, type(None), lt)
#
#     lt = TypeEngine.to_literal_type(typing.Optional[typing.Any])
#     TypeEngine.to_literal(ctx, 1, typing.Optional[typing.Any], lt)
#
#
# def test_enum_in_dataclass():
#     @dataclass
#     class Datum(DataClassJsonMixin):
#         x: int
#         y: Color
#
#     from mashumaro.jsonschema import build_json_schema
#     lt = TypeEngine.to_literal_type(Datum)
#     assert lt.metadata == build_json_schema(Datum).to_dict()
#
#     transformer = DataclassTransformer()
#     ctx = internal_ctx()
#     datum = Datum(5, Color.RED)
#     lv = transformer.to_literal(ctx, datum, Datum, lt)
#     gt = transformer.guess_python_type(lt)
#     pv = transformer.to_python_value(ctx, lv, expected_python_type=gt)
#     assert datum.x == pv.x
#     assert datum.y.value == pv.y
#
#
# def test_enum_in_dataclassjsonmixin():
#     @dataclass
#     class Datum(DataClassJSONMixin):
#         x: int
#         y: Color
#
#     lt = TypeEngine.to_literal_type(Datum)
#     from mashumaro.jsonschema import build_json_schema
#
#     schema = build_json_schema(typing.cast(DataClassJSONMixin, Datum)).to_dict()
#     assert lt.metadata == schema
#
#     transformer = DataclassTransformer()
#     ctx = internal_ctx()
#     datum = Datum(5, Color.RED)
#     lv = transformer.to_literal(ctx, datum, Datum, lt)
#     gt = transformer.guess_python_type(lt)
#     pv = transformer.to_python_value(ctx, lv, expected_python_type=gt)
#     assert datum.x == pv.x
#     assert datum.y.value == pv.y
#
#
# @pytest.mark.parametrize(
#     "python_value,python_types,expected_literal_map",
#     [
#         (
#                 {"a": [1, 2, 3]},
#                 {"a": typing.List[int]},
#                 LiteralMap(
#                     literals={
#                         "a": Literal(
#                             collection=LiteralCollection(
#                                 literals=[
#                                     Literal(scalar=Scalar(primitive=Primitive(integer=1))),
#                                     Literal(scalar=Scalar(primitive=Primitive(integer=2))),
#                                     Literal(scalar=Scalar(primitive=Primitive(integer=3))),
#                                 ]
#                             )
#                         )
#                     }
#                 ),
#         ),
#         (
#                 {"p1": {"k1": "v1", "k2": "2"}},
#                 {"p1": typing.Dict[str, str]},
#                 LiteralMap(
#                     literals={
#                         "p1": Literal(
#                             map=LiteralMap(
#                                 literals={
#                                     "k1": Literal(scalar=Scalar(primitive=Primitive(string_value="v1"))),
#                                     "k2": Literal(scalar=Scalar(primitive=Primitive(string_value="2"))),
#                                 },
#                             )
#                         )
#                     }
#                 ),
#         ),
#         (
#                 {"p1": "s3://tmp/file.jpeg"},
#                 {"p1": JPEGImageFile},
#                 LiteralMap(
#                     literals={
#                         "p1": Literal(
#                             scalar=Scalar(
#                                 blob=Blob(
#                                     metadata=BlobMetadata(
#                                         type=BlobType(
#                                             format="jpeg",
#                                             dimensionality=BlobType.BlobDimensionality.SINGLE,
#                                         )
#                                     ),
#                                     uri="s3://tmp/file.jpeg",
#                                 )
#                             )
#                         )
#                     }
#                 ),
#         ),
#     ],
# )
# def test_dict_to_literal_map(python_value, python_types, expected_literal_map):
#     ctx = internal_ctx()
#
#     assert TypeEngine.dict_to_literal_map(ctx, python_value, python_types) == expected_literal_map
#
#
# def test_dict_to_literal_map_with_wrong_input_type():
#     ctx = internal_ctx()
#     input = {"a": 1}
#     guessed_python_types = {"a": str}
#     with pytest.raises(user_exceptions.FlyteTypeException):
#         TypeEngine.dict_to_literal_map(ctx, input, guessed_python_types)
#
#
# def test_nested_annotated():
#     """
#     Test to show that nested Annotated types are flattened.
#     """
#     pt = Annotated[Annotated[int, "inner-annotation"], "outer-annotation"]
#     lt = TypeEngine.to_literal_type(pt)
#     assert lt.simple == types_pb2.SimpleType.INTEGER
#
#     ctx = FlyteContextManager.current_context()
#     lv = TypeEngine.to_literal(ctx, 42, pt, lt)
#     v = TypeEngine.to_python_value(ctx, lv, pt)
#     assert v == 42
#
#
# TestSchema = FlyteSchema[kwtypes(some_str=str)]  # type: ignore
#
#
# @dataclass
# class InnerResult(DataClassJsonMixin):
#     number: int
#     schema: TestSchema  # type: ignore
#
#
# @dataclass
# class Result(DataClassJsonMixin):
#     result: InnerResult
#     schema: TestSchema  # type: ignore
#
#
# def get_unsupported_complex_literals_tests():
#     if sys.version_info < (3, 9):
#         return [
#             typing_extensions.Annotated[typing.Dict[int, str], FlyteAnnotation({"foo": "bar"})],
#             typing_extensions.Annotated[typing.Dict[str, str], FlyteAnnotation({"foo": "bar"})],
#             typing_extensions.Annotated[Color, FlyteAnnotation({"foo": "bar"})],
#             typing_extensions.Annotated[Result, FlyteAnnotation({"foo": "bar"})],
#         ]
#     return [
#         typing_extensions.Annotated[dict, FlyteAnnotation({"foo": "bar"})],
#         typing_extensions.Annotated[dict[int, str], FlyteAnnotation({"foo": "bar"})],
#         typing_extensions.Annotated[typing.Dict[int, str], FlyteAnnotation({"foo": "bar"})],
#         typing_extensions.Annotated[dict[str, str], FlyteAnnotation({"foo": "bar"})],
#         typing_extensions.Annotated[typing.Dict[str, str], FlyteAnnotation({"foo": "bar"})],
#         typing_extensions.Annotated[Color, FlyteAnnotation({"foo": "bar"})],
#         typing_extensions.Annotated[Result, FlyteAnnotation({"foo": "bar"})],
#     ]
#
#
# @pytest.mark.parametrize(
#     "t",
#     get_unsupported_complex_literals_tests(),
# )
# def test_unsupported_complex_literals(t):
#     with pytest.raises(ValueError):
#         TypeEngine.to_literal_type(t)
#
#
# @dataclass
# class DataclassTest(DataClassJsonMixin):
#     a: int
#     b: str
#
#
#
#


@pytest.mark.asyncio
async def test_guess_of_dataclassjsonmixin():
    @dataclass
    class Foo(DataClassJSONMixin):
        x: int
        y: str
        z: typing.Dict[str, int]

        def hello(self): ...

    lt = TypeEngine.to_literal_type(Foo)
    foo = Foo(1, "hello", {"world": 3})
    lv = await TypeEngine.to_literal(foo, Foo, lt)
    lit_dict = {"a": lv}
    lr = LiteralsResolver(lit_dict)
    assert await lr.get("a", Foo) == foo
    assert hasattr(await lr.get("a", Foo), "hello") is True


@pytest.mark.asyncio
async def test_flyte_dir_in_union():
    pt = typing.Union[str, Dir, File]
    lt = TypeEngine.to_literal_type(pt)
    tf = UnionTransformer()

    pv = tempfile.mkdtemp(prefix="flyte-")
    lv = await tf.to_literal(Dir(path=pv), pt, lt)
    ot = await tf.to_python_value(lv=lv, expected_python_type=pt)
    assert ot is not None

    pv = "s3://bucket/key"
    lv = await tf.to_literal(File(path=pv), pt, lt)
    ot = await tf.to_python_value(lv=lv, expected_python_type=pt)
    assert ot is not None

    pv = "hello"
    lv = await tf.to_literal(pv, pt, lt)
    ot = await tf.to_python_value(lv=lv, expected_python_type=pt)
    assert ot == "hello"


# def test_file_ext_with_flyte_file_existing_file():
#     assert JPEGImageFile.extension() == "jpeg"
#
#
# def test_file_ext_convert_static_method():
#     TAR_GZ = Annotated[str, FileExt("tar.gz")]
#     item = FileExt.check_and_convert_to_str(TAR_GZ)
#     assert item == "tar.gz"
#
#     str_item = FileExt.check_and_convert_to_str("csv")
#     assert str_item == "csv"
#
#
# def test_file_ext_with_flyte_file_new_file():
#     TAR_GZ = Annotated[str, FileExt("tar.gz")]
#     flyte_file = FlyteFile[TAR_GZ]
#     assert flyte_file.extension() == "tar.gz"
#
#
# class WrongType:
#     def __init__(self, num: int):
#         self.num = num
#
#
# def test_file_ext_with_flyte_file_wrong_type():
#     WRONG_TYPE = Annotated[int, WrongType(2)]
#     with pytest.raises(ValueError) as e:
#         FlyteFile[WRONG_TYPE]
#     assert str(e.value) == "Underlying type of File Extension must be of type <str>"
#
#
# @pytest.mark.parametrize(
#     "t,expected",
#     [
#         (list, False),
#         (Annotated[int, "tag"], True),
#         (Annotated[typing.List[str], "a", "b"], True),
#         (Annotated[typing.Dict[int, str], FlyteAnnotation({"foo": "bar"})], True),
#     ],
# )
# def test_is_annotated(t, expected):
#     assert is_annotated(t) == expected
#
#
# @pytest.mark.parametrize(
#     "t,expected",
#     [
#         (typing.List, typing.List),
#         (Annotated[int, "tag"], int),
#         (Annotated[typing.List[str], "a", "b"], typing.List[str]),
#     ],
# )
# def test_get_underlying_type(t, expected):
#     assert get_underlying_type(t) == expected
#
#
# @pytest.mark.parametrize(
#     "t,expected,allow_pickle",
#     [
#         (None, (None, None), False),
#         (typing.Dict, (), False),
#         (typing.Dict[str, str], (str, str), False),
#         (
#             Annotated[typing.Dict[str, str], kwtypes(allow_pickle=True)],
#             (str, str),
#             True,
#         ),
#         (typing.Dict[Annotated[str, "a-tag"], int], (Annotated[str, "a-tag"], int), False),
#     ],
# )
# def test_dict_get(t, expected, allow_pickle):
#     assert DictTransformer.extract_types(t) == expected
#     assert DictTransformer.is_pickle(t) == allow_pickle


@pytest.mark.asyncio
async def test_DataclassTransformer_to_literal():
    import msgpack

    @dataclass
    class MyDataClassMashumaro(DataClassJSONMixin):
        x: int

    @dataclass
    class MyDataClassMashumaroORJSON(DataClassORJSONMixin):
        x: int

    @dataclass
    class MyDataClass:
        x: int

    transformer = DataclassTransformer()

    my_dat_class_mashumaro = MyDataClassMashumaro(5)
    my_dat_class_mashumaro_orjson = MyDataClassMashumaroORJSON(5)
    my_data_class = MyDataClass(5)

    lt_normal = TypeEngine.to_literal_type(MyDataClassMashumaro)
    lv_mashumaro = await transformer.to_literal(my_dat_class_mashumaro, MyDataClassMashumaro, lt_normal)
    assert lv_mashumaro is not None
    msgpack_bytes = lv_mashumaro.scalar.binary.value
    dict_obj = msgpack.loads(msgpack_bytes)
    assert dict_obj["x"] == 5

    lt_or = TypeEngine.to_literal_type(MyDataClassMashumaroORJSON)
    lv_mashumaro_orjson = await transformer.to_literal(
        my_dat_class_mashumaro_orjson,
        MyDataClassMashumaroORJSON,
        lt_or,
    )
    assert lv_mashumaro_orjson is not None
    msgpack_bytes = lv_mashumaro_orjson.scalar.binary.value
    dict_obj = msgpack.loads(msgpack_bytes)
    assert dict_obj["x"] == 5

    lv = await transformer.to_literal(my_data_class, MyDataClass, TypeEngine.to_literal_type(MyDataClass))
    assert lv is not None
    msgpack_bytes = lv.scalar.binary.value
    dict_obj = msgpack.loads(msgpack_bytes)
    assert dict_obj["x"] == 5


@pytest.mark.asyncio
async def test_DataclassTransformer_with_discriminated_subtypes():
    from mashumaro.config import BaseConfig
    from mashumaro.types import Discriminator

    from flyte._task_environment import TaskEnvironment

    class SubclassTypes(str, Enum):
        BASE = auto()
        CLASS_A = auto()
        CLASS_B = auto()

    @dataclass(kw_only=True)
    class BaseClass(DataClassJSONMixin):
        class Config(BaseConfig):
            discriminator = Discriminator(
                field="subclass_type",
                include_subtypes=True,
            )

        base_attribute: int
        subclass_type: SubclassTypes = SubclassTypes.BASE

    @dataclass(kw_only=True)
    class ClassA(BaseClass):
        class_a_attribute: str  # type: ignore[misc]
        subclass_type: SubclassTypes = SubclassTypes.CLASS_A

    @dataclass(kw_only=True)
    class ClassB(BaseClass):
        class_b_attribute: float  # type: ignore[misc]
        subclass_type: SubclassTypes = SubclassTypes.CLASS_B

    env = TaskEnvironment(name="test-dc-discriminator")

    @env.task
    async def assert_class_and_return(instance: BaseClass) -> BaseClass:
        assert hasattr(instance, "class_a_attribute") or hasattr(instance, "class_b_attribute")
        return instance

    class_a = ClassA(base_attribute=4, class_a_attribute="hello")
    assert "class_a_attribute" in class_a.to_json()
    res_1 = await assert_class_and_return(class_a)
    assert res_1.base_attribute == 4
    assert isinstance(res_1, ClassA)
    assert res_1.class_a_attribute == "hello"

    class_b = ClassB(base_attribute=4, class_b_attribute=-2.5)
    assert "class_b_attribute" in class_b.to_json()
    res_2 = await assert_class_and_return(class_b)
    assert res_2.base_attribute == 4
    assert isinstance(res_2, ClassB)
    assert res_2.class_b_attribute == -2.5


@pytest.mark.asyncio
async def test_DataclassTransformer_guess_python_type():
    @dataclass
    class DatumMashumaroORJSON(DataClassORJSONMixin):
        x: int
        y: Color
        z: datetime.datetime

    @dataclass
    class DatumMashumaro(DataClassJSONMixin):
        x: int
        y: Color

    @dataclass
    class DatumDataclassJson(DataClassJSONMixin):
        x: int
        y: Color

    @dataclass
    class DatumDataclass:
        x: int
        y: Color

    @dataclass
    class DatumDataUnion:
        data: typing.Union[str, float]

    transformer = TypeEngine.get_transformer(DatumDataUnion)

    lt = TypeEngine.to_literal_type(DatumDataUnion)
    datum_dataunion = DatumDataUnion(data="s3://my-file")
    lv = await transformer.to_literal(datum_dataunion, DatumDataUnion, lt)
    gt = transformer.guess_python_type(lt)
    pv = await transformer.to_python_value(lv, expected_python_type=DatumDataUnion)
    assert datum_dataunion.data == pv.data

    datum_dataunion = DatumDataUnion(data="0.123")
    lv = await transformer.to_literal(datum_dataunion, DatumDataUnion, lt)
    gt = transformer.guess_python_type(lt)
    pv = await transformer.to_python_value(lv, expected_python_type=gt)
    assert datum_dataunion.data == pv.data

    lt = TypeEngine.to_literal_type(DatumDataclass)
    datum_dataclass = DatumDataclass(5, Color.RED)
    lv = await transformer.to_literal(datum_dataclass, DatumDataclass, lt)
    gt = transformer.guess_python_type(lt)
    pv = await transformer.to_python_value(lv, expected_python_type=gt)
    assert datum_dataclass.x == pv.x
    assert datum_dataclass.y.value == pv.y

    lt = TypeEngine.to_literal_type(DatumDataclassJson)
    datum = DatumDataclassJson(5, Color.RED)
    lv = await transformer.to_literal(datum, DatumDataclassJson, lt)
    gt = transformer.guess_python_type(lt)
    pv = await transformer.to_python_value(lv, expected_python_type=gt)
    assert datum.x == pv.x
    assert datum.y.value == pv.y

    lt = TypeEngine.to_literal_type(DatumMashumaro)
    datum_mashumaro = DatumMashumaro(5, Color.RED)
    lv = await transformer.to_literal(datum_mashumaro, DatumMashumaro, lt)
    gt = transformer.guess_python_type(lt)
    pv = await transformer.to_python_value(lv, expected_python_type=gt)
    assert datum_mashumaro.x == pv.x
    assert datum_mashumaro.y.value == pv.y

    lt = TypeEngine.to_literal_type(DatumMashumaroORJSON)
    now = datetime.datetime.now()
    datum_mashumaro_orjson = DatumMashumaroORJSON(5, Color.RED, now)
    lv = await transformer.to_literal(datum_mashumaro_orjson, DatumMashumaroORJSON, lt)
    gt = transformer.guess_python_type(lt)
    pv = await transformer.to_python_value(lv, expected_python_type=gt)
    assert datum_mashumaro_orjson.x == pv.x
    assert datum_mashumaro_orjson.y.value == pv.y
    assert datum_mashumaro_orjson.z.isoformat() == pv.z


def test_dataclass_encoder_and_decoder_registry():
    import flyte
    from flyte._task_environment import TaskEnvironment

    flyte.init()
    env = TaskEnvironment(name="test-dc-registry")
    iterations = 10

    @dataclass
    class Datum:
        x: int
        y: str
        z: typing.Dict[int, int]
        w: List[int]

    @env.task
    async def create_dataclasses() -> List[Datum]:
        return [Datum(x=1, y="1", z={1: 1}, w=[1, 1, 1, 1])]

    @env.task
    async def concat_dataclasses(x: List[Datum], y: List[Datum]) -> List[Datum]:
        return x + y

    @env.task
    async def dynamic_wf() -> List[Datum]:
        all_dataclasses: List[Datum] = []
        for _ in range(iterations):
            data = await create_dataclasses()
            all_dataclasses = await concat_dataclasses(x=all_dataclasses, y=data)
        return all_dataclasses

    @env.task
    async def wf() -> List[Datum]:
        return await dynamic_wf()

    datum_list = flyte.run(wf)
    assert len(datum_list.outputs()) == iterations

    transformer = TypeEngine.get_transformer(Datum)
    assert transformer._msgpack_encoder.get(Datum)
    assert transformer._msgpack_decoder.get(Datum)


def test_ListTransformer_get_sub_type():
    assert ListTransformer.get_sub_type_or_none(typing.List[str]) is str


def test_ListTransformer_get_sub_type_as_none():
    assert ListTransformer.get_sub_type_or_none(type([])) is None


# def test_union_file_directory():
#     lt = TypeEngine.to_literal_type(FlyteFile)
#     s3_file = "s3://my-file"
#
#     transformer = FlyteFilePathTransformer()
#     ctx = internal_ctx()
#     lv = transformer.to_literal(ctx, s3_file, FlyteFile, lt)
#
#     union_trans = UnionTransformer()
#     pv = union_trans.to_python_value(ctx, lv, typing.Union[FlyteFile, FlyteDirectory])
#     assert pv._remote_source == s3_file
#
#     s3_dir = "s3://my-dir"
#     transformer = FlyteDirToMultipartBlobTransformer()
#     ctx = internal_ctx()
#     lv = transformer.to_literal(ctx, s3_dir, FlyteFile, lt)
#
#     pv = union_trans.to_python_value(ctx, lv, typing.Union[FlyteFile, FlyteDirectory])
#     assert pv._remote_source == s3_dir
#
#
@pytest.mark.parametrize(
    "pt,pv",
    [
        (bool, True),
        (bool, False),
        (int, 42),
        (str, "hello"),
        (Annotated[int, "tag"], 42),
        (typing.List[int], [1, 2, 3]),
        (typing.List[str], ["a", "b", "c"]),
        (typing.List[Color], [Color.RED, Color.GREEN, Color.BLUE]),
        (typing.List[Annotated[int, "tag"]], [1, 2, 3]),
        (typing.List[Annotated[str, "tag"]], ["a", "b", "c"]),
        (typing.Dict[int, str], {1: "a", 2: "b", 3: "c"}),
        (typing.Dict[str, int], {"a": 1, "b": 2, "c": 3}),
        (typing.Dict[str, typing.List[int]], {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        (typing.Dict[str, typing.Dict[int, str]], {"a": {1: "a", 2: "b", 3: "c"}, "b": {4: "d", 5: "e", 6: "f"}}),
        (typing.Union[int, str], 42),
        (typing.Union[int, str], "hello"),
        (typing.Union[typing.List[int], typing.List[str]], [1, 2, 3]),
        (typing.Union[typing.List[int], typing.List[str]], ["a", "b", "c"]),
        (typing.Union[typing.List[int], str], [1, 2, 3]),
        (typing.Union[typing.List[int], str], "hello"),
        ((typing.Union[dict, str]), {"a": 1, "b": 2, "c": 3}),
        ((typing.Union[dict, str]), "hello"),
    ],
)
@pytest.mark.asyncio
async def test_offloaded_literal(tmp_path, pt, pv):
    lt = TypeEngine.to_literal_type(pt)
    to_be_offloaded_lv = await TypeEngine.to_literal(pv, pt, lt)

    # Write offloaded_lv as bytes to a temp file
    with open(f"{tmp_path}/offloaded_proto.pb", "wb") as f:  # noqa: ASYNC230
        f.write(to_be_offloaded_lv.SerializeToString())

    literal = Literal(
        offloaded_metadata=literals_pb2.LiteralOffloadedMetadata(
            uri=f"{tmp_path}/offloaded_proto.pb",
            inferred_type=lt,
        ),
    )

    loaded_pv = await TypeEngine.to_python_value(literal, pt)
    assert loaded_pv == pv


@pytest.mark.asyncio
async def test_offloaded_literal_with_inferred_type():
    lt = TypeEngine.to_literal_type(str)
    offloaded_literal_missing_uri = Literal(
        offloaded_metadata=literals_pb2.LiteralOffloadedMetadata(
            inferred_type=lt,
        ),
    )
    with pytest.raises(AssertionError):
        await TypeEngine.to_python_value(offloaded_literal_missing_uri, str)


# def test_offloaded_literal_flytefile(tmp_path):
#     ctx = internal_ctx()
#     lt = TypeEngine.to_literal_type(FlyteFile)
#     to_be_offloaded_lv = TypeEngine.to_literal(ctx, "s3://my-file", FlyteFile, lt)
#
#     # Write offloaded_lv as bytes to a temp file
#     with open(f"{tmp_path}/offloaded_proto.pb", "wb") as f:
#         f.write(to_be_offloaded_lv.to_flyte_idl().SerializeToString())
#
#     literal = Literal(
#         offloaded_metadata=LiteralOffloadedMetadata(
#             uri=f"{tmp_path}/offloaded_proto.pb",
#             inferred_type=lt,
#         ),
#     )
#
#     loaded_pv = TypeEngine.to_python_value(ctx, literal, FlyteFile)
#     assert loaded_pv._remote_source == "s3://my-file"
#
#
# def test_offloaded_literal_flytedirectory(tmp_path):
#     ctx = internal_ctx()
#     lt = TypeEngine.to_literal_type(FlyteDirectory)
#     to_be_offloaded_lv = TypeEngine.to_literal(ctx, "s3://my-dir", FlyteDirectory, lt)
#
#     # Write offloaded_lv as bytes to a temp file
#     with open(f"{tmp_path}/offloaded_proto.pb", "wb") as f:
#         f.write(to_be_offloaded_lv.to_flyte_idl().SerializeToString())
#
#     literal = Literal(
#         offloaded_metadata=LiteralOffloadedMetadata(
#             uri=f"{tmp_path}/offloaded_proto.pb",
#             inferred_type=lt,
#         ),
#     )
#
#     loaded_pv: FlyteDirectory = TypeEngine.to_python_value(ctx, literal, FlyteDirectory)
#     assert loaded_pv._remote_source == "s3://my-dir"


@pytest.mark.asyncio
async def test_dataclass_none_output_input_deserialization():
    from flyte._task_environment import TaskEnvironment

    env = TaskEnvironment(name="test-dc-serde")

    @dataclass
    class OuterWorkflowInput(DataClassJSONMixin):
        input: float

    @dataclass
    class OuterWorkflowOutput(DataClassJSONMixin):
        nullable_output: float | None = None

    @dataclass
    class InnerWorkflowInput(DataClassJSONMixin):
        input: float

    @dataclass
    class InnerWorkflowOutput(DataClassJSONMixin):
        nullable_output: float | None = None

    @env.task
    async def inner_task(input: float) -> float | None:
        if input == 0.0:
            return None
        return input

    @env.task
    async def wrap_inner_inputs(input: float) -> InnerWorkflowInput:
        return InnerWorkflowInput(input=input)

    @env.task
    async def wrap_inner_outputs(output: float | None) -> InnerWorkflowOutput:
        return InnerWorkflowOutput(nullable_output=output)

    @env.task
    async def wrap_outer_outputs(output: float | None) -> OuterWorkflowOutput:
        return OuterWorkflowOutput(nullable_output=output)

    @env.task
    async def inner_workflow(input: InnerWorkflowInput) -> InnerWorkflowOutput:
        return await wrap_inner_outputs(output=await inner_task(input=input.input))

    @env.task
    async def outer_workflow(input: OuterWorkflowInput) -> OuterWorkflowOutput:
        inner_outputs = await inner_workflow(input=await wrap_inner_inputs(input=input.input))
        return await wrap_outer_outputs(output=inner_outputs.nullable_output)

    float_value_output = await outer_workflow(OuterWorkflowInput(input=1.0))
    float_value_output = float_value_output.nullable_output
    assert float_value_output == 1.0, f"Float value was {float_value_output}, not 1.0 as expected"
    none_value_output = await outer_workflow(OuterWorkflowInput(input=0.0))
    none_value_output = none_value_output.nullable_output
    assert none_value_output is None, f"None value was {none_value_output}, not None as expected"


#
#
# @pytest.mark.serial
# def test_lazy_import_transformers_concurrently():
#     # Configure the mocks similar to https://stackoverflow.com/questions/29749193/python-unit-testing-with-two-mock-objects-how-to-verify-call-order
#     after_import_mock, mock_register = mock.Mock(), mock.Mock()
#     mock_wrapper = mock.Mock()
#     mock_wrapper.mock_register = mock_register
#     mock_wrapper.after_import_mock = after_import_mock
#
#     with mock.patch.object(StructuredDatasetTransformerEngine, "register", new=mock_register):
#         def run():
#             TypeEngine.lazy_import_transformers()
#             after_import_mock()
#
#         N = 5
#         with ThreadPoolExecutor(max_workers=N) as executor:
#             futures = [executor.submit(run) for _ in range(N)]
#             [f.result() for f in futures]
#
#         assert mock_wrapper.mock_calls[-1] == mock.call.after_import_mock()
#         expected_number_of_register_calls = len(mock_wrapper.mock_calls) - N
#         assert sum([mock_call[0] == "mock_register" for mock_call in mock_wrapper.mock_calls]) \
#           == expected_number_of_register_calls
#         assert all([mock_call[0] == "mock_register" for mock_call in
#                     mock_wrapper.mock_calls[:int(len(mock_wrapper.mock_calls)/N)-1]])


@pytest.mark.asyncio
async def test_option_list_with_pipe():
    pt = list[int] | None
    lt = TypeEngine.to_literal_type(pt)

    lit = await TypeEngine.to_literal([1, 2, 3], pt, lt)
    assert lit.scalar.union.value.collection.literals[2].scalar.primitive.integer == 3

    await TypeEngine.to_literal(None, pt, lt)

    with pytest.raises(TypeTransformerFailedError):
        await TypeEngine.to_literal([1, 2, "3"], pt, lt)


@pytest.mark.asyncio
async def test_option_list_with_pipe_2():
    pt = list[list[dict[str, str]] | None] | None
    lt = TypeEngine.to_literal_type(pt)

    lit = await TypeEngine.to_literal([[{"a": "one"}], None, [{"b": "two"}]], pt, lt)
    uv = lit.scalar.union.value
    assert uv is not None
    assert len(uv.collection.literals) == 3
    first = uv.collection.literals[0]
    assert first.scalar.union.value.collection.literals[0].map.literals["a"].scalar.primitive.string_value == "one"

    assert len(lt.union_type.variants) == 2
    v1 = lt.union_type.variants[0]
    assert len(v1.collection_type.union_type.variants) == 2
    assert v1.collection_type.union_type.variants[0].collection_type.map_value_type.simple == SimpleType.STRING

    with pytest.raises(TypeTransformerFailedError):
        await TypeEngine.to_literal([[{"a": "one"}], None, [{"b": 3}]], pt, lt)


@pytest.mark.asyncio
async def test_generic_errors_and_empty():
    # Test dictionaries
    pt = dict[str, str]
    lt = TypeEngine.to_literal_type(pt)

    await TypeEngine.to_literal({}, pt, lt)
    await TypeEngine.to_literal({"a": "b"}, pt, lt)

    with pytest.raises(TypeTransformerFailedError):
        await TypeEngine.to_literal({"a": 3}, pt, lt)

    with pytest.raises(ValueError):
        await TypeEngine.to_literal({3: "a"}, pt, lt)

    # Test lists
    pt = list[str]
    lt = TypeEngine.to_literal_type(pt)
    await TypeEngine.to_literal([], pt, lt)
    await TypeEngine.to_literal(["a"], pt, lt)

    with pytest.raises(TypeTransformerFailedError):
        await TypeEngine.to_literal({"a": 3}, pt, lt)

    with pytest.raises(TypeTransformerFailedError):
        await TypeEngine.to_literal([3], pt, lt)


def generate_type_engine_transformer_comprehensive_tests():
    # Test dataclasses
    # @dataclass
    # class DataClass(DataClassJsonMixin):
    #     a: int
    #     b: str

    class Test:
        a: str
        b: int

    T = typing.TypeVar("T")

    class TestGeneric(typing.Generic[T]):
        a: str
        b: int

    # Test annotated types
    AnnotatedInt = Annotated[int, "tag"]
    AnnotatedFloat = Annotated[float, "tag"]
    AnnotatedStr = Annotated[str, "tag"]
    AnnotatedBool = Annotated[bool, "tag"]
    AnnotatedList = Annotated[List[str], "tag"]
    AnnotatedDict = Annotated[Dict[str, str], "tag"]
    Annotatedx3Int = Annotated[Annotated[Annotated[int, "tag"], "tag2"], "tag3"]

    # Test generics
    ListInt = List[int]
    ListStr = List[str]
    DictIntStr = Dict[str, str]
    ListAnnotatedInt = List[AnnotatedInt]
    DictAnnotatedIntStr = Dict[str, AnnotatedStr]

    # Test regular types
    Int = int
    Str = str

    CallableType = typing.Callable[[int, str], int]
    CallableTypeAnnotated = Annotated[CallableType, "tag"]
    CallableTypeList = List[CallableType]

    # IteratorType = typing.Iterator[int]
    # IteratorTypeAnnotated = Annotated[IteratorType, "tag"]
    # IteratorTypeList = List[IteratorType]

    People = Annotated[DataFrame, "parquet", OrderedDict(Name=str, Age=int)]
    PeopleDeepAnnotated = Annotated[Annotated[DataFrame, "parquet", OrderedDict(Name=str, Age=int)], "tag"]

    AnyType = typing.Any
    AnyTypeAnnotated = Annotated[AnyType, "tag"]
    AnyTypeAnnotatedList = List[AnyTypeAnnotated]

    UnionType = typing.Union[int, str]
    UnionTypeAnnotated = Annotated[UnionType, "tag"]

    OptionalType = typing.Optional[int]
    OptionalTypeAnnotated = Annotated[OptionalType, "tag"]

    WineType = Annotated[DataFrame, OrderedDict(alcohol=float, malic_acid=float)]
    WineTypeList = List[WineType]
    WineTypeListList = List[WineTypeList]
    WineTypeDict = Dict[str, WineType]

    IntPickle = Annotated[int, FlytePickleTransformer()]
    AnnotatedIntPickle = Annotated[Annotated[int, "tag"], FlytePickleTransformer()]

    # Test combinations
    return [
        # (DataClass, DataclassTransformer),
        (AnnotatedInt, IntTransformer),
        (AnnotatedFloat, FloatTransformer),
        (AnnotatedStr, StrTransformer),
        (Annotatedx3Int, IntTransformer),
        (ListInt, ListTransformer),
        (ListStr, ListTransformer),
        (DictIntStr, DictTransformer),
        (Int, IntTransformer),
        (Str, StrTransformer),
        (AnnotatedBool, BoolTransformer),
        (AnnotatedList, ListTransformer),
        (AnnotatedDict, DictTransformer),
        (ListAnnotatedInt, ListTransformer),
        (DictAnnotatedIntStr, DictTransformer),
        (CallableType, FlytePickleTransformer),
        (CallableTypeAnnotated, FlytePickleTransformer),
        (CallableTypeList, ListTransformer),
        # (IteratorType, IteratorTransformer),
        # (IteratorTypeAnnotated, IteratorTransformer),
        # (IteratorTypeList, ListTransformer),
        (People, DataFrameTransformerEngine),
        (PeopleDeepAnnotated, DataFrameTransformerEngine),
        (WineType, DataFrameTransformerEngine),
        (WineTypeList, ListTransformer),
        (AnyType, FlytePickleTransformer),
        (AnyTypeAnnotated, FlytePickleTransformer),
        (UnionType, UnionTransformer),
        (UnionTypeAnnotated, UnionTransformer),
        (OptionalType, UnionTransformer),
        (OptionalTypeAnnotated, UnionTransformer),
        (Test, FlytePickleTransformer),
        (TestGeneric, FlytePickleTransformer),
        (typing.Iterable[int], FlytePickleTransformer),
        (typing.Sequence[int], FlytePickleTransformer),
        (IntPickle, FlytePickleTransformer),
        (AnnotatedIntPickle, FlytePickleTransformer),
        # (typing.Iterator[JSON], JSONIteratorTransformer),
        # (JSONIterator, JSONIteratorTransformer),
        (AnyTypeAnnotatedList, ListTransformer),
        (WineTypeListList, ListTransformer),
        (WineTypeDict, DictTransformer),
    ]


@pytest.mark.parametrize("t, expected_transformer", generate_type_engine_transformer_comprehensive_tests())
def test_type_engine_get_transformer_comprehensive(t, expected_transformer):
    """
    This test will test various combinations like dataclasses, annotated types, generics and regular types and
    assert the right transformers are returned.
    """
    if isinstance(expected_transformer, SimpleTransformer):
        underlying_type = expected_transformer.base_type
        assert isinstance(TypeEngine.get_transformer(t), SimpleTransformer)
        assert TypeEngine.get_transformer(t).base_type == underlying_type
    else:
        assert isinstance(TypeEngine.get_transformer(t), expected_transformer)


@pytest.mark.parametrize(
    "t, expected_variants",
    [
        (int | float, [int, float]),
        (int | float | None, [int, float, type(None)]),
        (int | float | str, [int, float, str]),
    ],
)
def test_union_type_comprehensive_604(t, expected_variants):
    """
    This test will test various combinations like dataclasses, annotated types, generics and regular types and
    assert the right transformers are returned.
    """
    transformer = TypeEngine.get_transformer(t)
    assert isinstance(transformer, UnionTransformer)
    lt = transformer.get_literal_type(t)
    assert [TypeEngine.guess_python_type(i) for i in lt.union_type.variants] == expected_variants


@pytest.mark.parametrize(
    "t, expected_variants",
    [
        (typing.Union[int, str], [int, str]),
        (typing.Union[str, None], [str, type(None)]),
        (typing.Optional[int], [int, type(None)]),
    ],
)
def test_union_comprehensive(t, expected_variants):
    """
    This test will test various combinations like dataclasses, annotated types, generics and regular types and
    assert the right transformers are returned.
    """
    transformer = TypeEngine.get_transformer(t)
    assert isinstance(transformer, UnionTransformer)
    lt = transformer.get_literal_type(t)
    assert [TypeEngine.guess_python_type(i) for i in lt.union_type.variants] == expected_variants


@pytest.mark.skipif("pandas" not in sys.modules, reason="Pandas is not installed.")
@pytest.mark.asyncio
async def test_structured_dataset_collection(ctx_with_test_raw_data_path):
    WineType = Annotated[DataFrame, OrderedDict(alcohol=float, malic_acid=float)]
    WineTypeList = List[WineType]
    WineTypeListList = List[WineTypeList]

    import pandas as pd

    df = pd.DataFrame({"alcohol": [1.0, 2.0], "malic_acid": [2.0, 3.0]})

    await TypeEngine.to_literal(DataFrame(df), WineType, TypeEngine.to_literal_type(WineType))

    transformer = TypeEngine.get_transformer(WineTypeListList)
    assert isinstance(transformer, ListTransformer)
    lt = transformer.get_literal_type(WineTypeListList)
    cols = lt.collection_type.collection_type.structured_dataset_type.columns
    assert cols[0].name == "alcohol"
    assert cols[0].literal_type.simple == SimpleType.FLOAT
    assert cols[1].name == "malic_acid"
    assert cols[1].literal_type.simple == SimpleType.FLOAT

    sd = DataFrame(df, format="parquet")
    lv = await TypeEngine.to_literal([[sd]], WineTypeListList, lt)
    assert lv is not None

    lv = await TypeEngine.to_literal([[DataFrame(df)]], WineTypeListList, lt)
    assert lv is not None


@pytest.mark.skipif("pandas" not in sys.modules, reason="Pandas is not installed.")
@pytest.mark.asyncio
async def test_structured_dataset_mismatch():
    import pandas as pd

    df = pd.DataFrame({"alcohol": [1.0, 2.0], "malic_acid": [2.0, 3.0]})
    transformer = TypeEngine.get_transformer(DataFrame)
    with pytest.raises(TypeTransformerFailedError):
        await transformer.to_literal(df, DataFrame, TypeEngine.to_literal_type(DataFrame))

    with pytest.raises(TypeTransformerFailedError):
        await TypeEngine.to_literal(df, DataFrame, TypeEngine.to_literal_type(DataFrame))


def test_register_dataclass_override():
    """
    Test to confirm that a dataclass transformer can be overridden by a user defined transformer
    """

    # We register a type transformer for the top-level user-defined dataclass
    @dataclass
    class ParentDC: ...

    @dataclass
    class ChildDC(ParentDC): ...

    class ParentDCTransformer(TypeTransformer[ParentDC]):
        def __init__(self):
            super().__init__("ParentDC Transformer", ParentDC)

    # Register a type transformer for the parent dataclass
    TypeEngine.register(ParentDCTransformer())

    # Confirm that the transformer for ChildDC is the same as the ParentDC
    assert TypeEngine.get_transformer(ChildDC) == TypeEngine.get_transformer(ParentDC)

    # Confirm that the transformer for ChildDC is not flyte's default dataclass transformer
    @dataclass
    class RegularDC: ...

    assert TypeEngine.get_transformer(ChildDC) != TypeEngine.get_transformer(RegularDC)
    assert TypeEngine.get_transformer(RegularDC) == TypeEngine._DATACLASS_TRANSFORMER

    del TypeEngine._REGISTRY[ParentDC]


def test_strict_type_matching():
    # should correctly return the more specific transformer
    class MyInt:
        def __init__(self, x: int):
            self.val = x

        __hash__ = None  # Explicit hashing disabled

        def __eq__(self, other):
            if not isinstance(other, MyInt):
                return False
            return other.val == self.val

    lt = LiteralType(simple=SimpleType.INTEGER)
    TypeEngine.register(
        SimpleTransformer(
            "MyInt",
            MyInt,
            lt,
            lambda x: Literal(scalar=Scalar(primitive=Primitive(integer=x.val))),
            lambda x: MyInt(x.scalar.primitive.integer),
        )
    )

    pt_guess = IntTransformer.guess_python_type(lt)
    assert pt_guess is int
    pt_better_guess = strict_type_hint_matching(MyInt(3), lt)
    assert pt_better_guess is MyInt

    del TypeEngine._REGISTRY[MyInt]


def test_strict_type_matching_error():
    xs: typing.List[float] = [0.1, 0.2, 0.3, 0.4, -99999.7]
    lt = TypeEngine.to_literal_type(typing.List[float])
    with pytest.raises(ValueError):
        strict_type_hint_matching(xs, lt)


@pytest.mark.asyncio
async def test_dict_transformer_annotated_type():
    # Test case 1: Regular Dict type
    regular_dict = {"a": 1, "b": 2}
    regular_dict_type = Dict[str, int]
    expected_type = TypeEngine.to_literal_type(regular_dict_type)

    # This should work fine
    literal1 = await TypeEngine.to_literal(regular_dict, regular_dict_type, expected_type)
    assert literal1.map.literals["a"].scalar.primitive.integer == 1
    assert literal1.map.literals["b"].scalar.primitive.integer == 2

    # Test case 2: Annotated Dict type
    annotated_dict = {"x": 10, "y": 20}
    annotated_dict_type = Annotated[Dict[str, int], "some_metadata"]
    expected_type = TypeEngine.to_literal_type(annotated_dict_type)

    literal2 = await TypeEngine.to_literal(annotated_dict, annotated_dict_type, expected_type)
    assert literal2.map.literals["x"].scalar.primitive.integer == 10
    assert literal2.map.literals["y"].scalar.primitive.integer == 20

    # Test case 3: Nested Annotated Dict type
    nested_dict = {"outer": {"inner": 42}}
    nested_dict_type = Dict[str, Annotated[Dict[str, int], "inner_metadata"]]
    expected_type = TypeEngine.to_literal_type(nested_dict_type)

    literal3 = await TypeEngine.to_literal(nested_dict, nested_dict_type, expected_type)
    assert literal3.map.literals["outer"].map.literals["inner"].scalar.primitive.integer == 42


def test_type_casting_for_union():
    import flyte

    env = flyte.TaskEnvironment("test")

    @env.task
    def complex_task(data: dict[str, list[str] | None] | None) -> dict[str, list[str] | None] | None:
        return data

    flyte.init()
    return flyte.run(complex_task, data={"key": ["value1", "value2"]})


@pytest.mark.asyncio
async def test_union_pydantic():
    class People(BaseModel):
        name: str

    class Animal(BaseModel):
        name: str

    class Model1(BaseModel):
        action: typing.Literal["a", "a+", "at", "a+t"]

    class Model2(BaseModel):
        actions: typing.List[typing.Union[People, Animal]]

    UnionModel = Model1 | Model2

    # Test serialization and deserialization of Model1
    model1_instance = Model1(action="a")
    literal_type = TypeEngine.to_literal_type(UnionModel)
    literal_value = await TypeEngine.to_literal(model1_instance, UnionModel, literal_type)

    # Check that the correct variant was chosen
    assert literal_value.scalar.union.type.structure.tag.endswith("Pydantic Transformer")

    python_value = await TypeEngine.to_python_value(literal_value, UnionModel)
    assert python_value == model1_instance
