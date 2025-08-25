import inspect
import time
from dataclasses import dataclass
from typing import Awaitable, Optional, Tuple, Union

import pytest
from flyteidl.core.interface_pb2 import TypedInterface, Variable, VariableMap
from flyteidl.core.literals_pb2 import (
    Literal,
    LiteralCollection,
    LiteralMap,
    Primitive,
    Scalar,
)
from flyteidl.core.types_pb2 import (
    BlobType,
    EnumType,
    LiteralType,
    SimpleType,
    StructuredDatasetType,
    UnionType,
)

import flyte._internal.runtime.convert as convert
from flyte._internal.runtime.convert import Inputs, generate_sub_action_id_and_output_path
from flyte._internal.runtime.types_serde import transform_native_to_typed_interface
from flyte._protos.workflow import run_definition_pb2
from flyte._protos.workflow import run_definition_pb2 as _run_definition_pb2
from flyte.models import ActionID, NativeInterface, RawDataPath, TaskContext
from flyte.report import Report
from flyte.types import TypeEngine

test_cases = [
    (None, "cc6zwnxnmf3chm008fxfwv9g8"),
    ((NativeInterface.from_types({"x": (int, inspect.Parameter.empty)}, {}), (1,)), "2twhoypqmosoh4eepzui8954a"),
    ((NativeInterface.from_types({"x": (int, inspect.Parameter.empty)}, {}), (2,)), "bhqw2g4fyit5uczmjocduphnc"),
    ((NativeInterface.from_types({"x": (int, inspect.Parameter.empty)}, {}), (3,)), "et7s2yhynbrhdtsawc2wny9o6"),
    ((NativeInterface.from_types({"x": (int, inspect.Parameter.empty)}, {}), (4,)), "5nf5f0zrm2jkqcijzjls1pgfh"),
]


@pytest.fixture(params=test_cases)
async def generate_inputs(request) -> Tuple[Inputs, str]:
    if request.param[0] is None:
        return Inputs.empty(), request.param[1]
    interface, args = request.param[0]
    hash_val = request.param[1]
    inputs = await convert.convert_from_native_to_inputs(interface, *args)
    return inputs, hash_val


@pytest.mark.asyncio
async def test_generate_sub_action_id_and_output_path_consistency_task_name(generate_inputs: Awaitable):
    """
    This test checks that the algorithm is consistent and has not changed.
    """
    tctx = TaskContext(
        action=ActionID(name="test_action", run_name="xyz", project="test_project", domain="test_domain"),
        run_base_dir="s3://test-bucket/metadata/v2/test_project/test_domain/xyz",
        version="v1",
        raw_data_path=RawDataPath(path="s3://test-bucket/raw_data/test_project/test_domain/xyz"),
        output_path="s3://test-bucket/output/test_project/test_domain/xyz",
        report=Report(name="test"),
    )

    inputs, expected_hash = await generate_inputs
    serialized_inputs = inputs.proto_inputs.SerializeToString(deterministic=True)
    inputs_hash = convert.generate_inputs_hash(serialized_inputs)
    sub_action_id, path = generate_sub_action_id_and_output_path(
        tctx=tctx,
        task_spec_or_name="test_task",
        inputs_hash=inputs_hash,
        invoke_seq=1,
    )
    assert sub_action_id.name == expected_hash
    assert path is not None


def test_1M_action_name_with_min_diff():
    """
    This test checks that the algorithm can handle a large number of actions with minimal differences in their
    sequence numbers only.
    """
    start = time.perf_counter()

    tctx = TaskContext(
        action=ActionID(name="test_action", run_name="xyz", project="test_project", domain="test_domain"),
        run_base_dir="s3://test-bucket/metadata/v2/test_project/test_domain/xyz",
        version="v1",
        raw_data_path=RawDataPath(path="s3://test-bucket/raw_data/test_project/test_domain/xyz"),
        output_path="s3://test-bucket/output/test_project/test_domain/xyz",
        report=Report(name="test"),
    )
    prev_action_name = set({})
    serialized_inputs = Inputs.empty().proto_inputs.SerializeToString(deterministic=True)
    inputs_hash = convert.generate_inputs_hash(serialized_inputs)

    for i in range(1000000):
        sub_action_id, path = generate_sub_action_id_and_output_path(
            tctx=tctx,
            task_spec_or_name="t1",
            inputs_hash=inputs_hash,
            invoke_seq=i,
        )
        assert sub_action_id.name not in prev_action_name
        prev_action_name.add(sub_action_id.name)

    duration = time.perf_counter() - start
    print(f"\nTest duration: {duration:.4f} seconds")


@pytest.mark.asyncio
async def test_generate_cache_key_hash():
    """
    This test checks that the cache key hash generation matches that of the server side cache generation
    """

    interface = NativeInterface.from_types(
        {"int": (int, inspect.Parameter.empty), "str": (str, inspect.Parameter.empty)}, {}
    )
    typed_interface = transform_native_to_typed_interface(interface)
    args = (100, "hello world")

    inputs = await convert.convert_from_native_to_inputs(interface, *args)
    serialized_inputs = inputs.proto_inputs.SerializeToString(deterministic=True)
    inputs_hash = convert.generate_inputs_hash(serialized_inputs)

    task_name = "test_task"
    cache_key = convert.generate_cache_key_hash(task_name, inputs_hash, typed_interface, "v1", [], inputs.proto_inputs)
    assert cache_key == "5rqRLYOr9qd84OWUkfS0lT94IZ/Q0kH00c5LMKgsLNk="


# Run 10 times to make sure ordering is consistent
@pytest.mark.parametrize("_", range(10))
@pytest.mark.asyncio
async def test_generate_cache_key_hash_consistency(_):
    """
    This test checks that the cache key hash generation is consistent.
    """

    @dataclass
    class DataClassExample:
        field1: int
        field2: str

    interface = NativeInterface.from_types(
        {"x": (int, inspect.Parameter.empty), "dc": (DataClassExample, inspect.Parameter.empty)}, {}
    )
    typed_interface = transform_native_to_typed_interface(interface)
    args = (1, DataClassExample(field1=42, field2="example"))

    inputs = await convert.convert_from_native_to_inputs(interface, *args)
    serialized_inputs = inputs.proto_inputs.SerializeToString(deterministic=True)
    inputs_hash = convert.generate_inputs_hash(serialized_inputs)

    task_name = "test_task"
    cache_key = convert.generate_cache_key_hash(task_name, inputs_hash, typed_interface, "v1", [], inputs.proto_inputs)
    assert cache_key == "gKUZZV2XwUZbIUl9tfLC7+n8tREvPR9jq9vzPEHqOKg="


@pytest.mark.asyncio
async def test_generate_cache_key_ignored_input():
    interface = NativeInterface.from_types(
        {"x": (int, inspect.Parameter.empty), "ignore1": (float, inspect.Parameter.empty)}, {}
    )
    typed_interface = transform_native_to_typed_interface(interface)
    args = (1, 3.14)

    inputs = await convert.convert_from_native_to_inputs(interface, *args)
    serialized_inputs = inputs.proto_inputs.SerializeToString(deterministic=True)
    inputs_hash = convert.generate_inputs_hash(serialized_inputs)

    task_name = "test_task"
    cache_key_1 = convert.generate_cache_key_hash(
        task_name, inputs_hash, typed_interface, "v1", ["ignore1"], inputs.proto_inputs
    )

    inputs_2 = await convert.convert_from_native_to_inputs(interface, 1, 2.71828)
    serialized_inputs_2 = inputs.proto_inputs.SerializeToString(deterministic=True)
    inputs_hash_2 = convert.generate_inputs_hash(serialized_inputs_2)
    cache_key_2 = convert.generate_cache_key_hash(
        task_name, inputs_hash_2, typed_interface, "v1", ["ignore1"], inputs_2.proto_inputs
    )

    assert cache_key_1 == cache_key_2


@pytest.mark.asyncio
async def test_convert_from_native_to_inputs_empty():
    def empty_func():
        pass

    interface = NativeInterface.from_callable(empty_func)
    result = await convert.convert_from_native_to_inputs(interface)

    assert isinstance(result, Inputs)
    assert len(result.proto_inputs.literals) == 0


@pytest.mark.asyncio
async def test_convert_from_native_to_inputs_mixed_args():
    def func_mixed_args(x: int, y: str, z: float):
        pass

    interface = NativeInterface.from_callable(func_mixed_args)
    result = await convert.convert_from_native_to_inputs(interface, 42, y="hello", z=3.14)

    assert isinstance(result, Inputs)
    assert len(result.proto_inputs.literals) == 3

    literals_dict = {lit.name: lit for lit in result.proto_inputs.literals}
    assert "x" in literals_dict
    assert "y" in literals_dict
    assert "z" in literals_dict


@pytest.mark.asyncio
async def test_convert_from_native_to_inputs_with_defaults():
    def func_with_defaults(x: int, y: str = "default_value"):
        pass

    interface = NativeInterface.from_callable(func_with_defaults)
    result = await convert.convert_from_native_to_inputs(interface, 42)

    assert isinstance(result, Inputs)
    assert len(result.proto_inputs.literals) == 2

    literals_dict = {lit.name: lit for lit in result.proto_inputs.literals}
    assert "x" in literals_dict
    assert "y" in literals_dict


@pytest.mark.asyncio
async def test_convert_from_native_to_inputs_missing_required_inputs():
    def func_required(x: int, y: str):
        pass

    interface = NativeInterface.from_callable(func_required)

    with pytest.raises(ValueError, match="Missing required inputs: y"):
        await convert.convert_from_native_to_inputs(interface, 42)


@pytest.mark.asyncio
async def test_convert_from_native_to_inputs_ordering_preserved():
    def func_ordered(a: int, b: str, c: float):
        pass

    interface = NativeInterface.from_callable(func_ordered)
    result = await convert.convert_from_native_to_inputs(interface, a=1, b="test", c=2.5)

    assert isinstance(result, Inputs)
    assert len(result.proto_inputs.literals) == 3

    literal_names = [lit.name for lit in result.proto_inputs.literals]
    assert literal_names == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_convert_from_native_to_inputs_optional_types():
    # Test the | style
    def func_optional(required: int, optional: str | None = None, more_optional: Optional[float] = None):
        pass

    interface = NativeInterface.from_callable(func_optional)

    # Test with optional parameter provided
    result = await convert.convert_from_native_to_inputs(interface, required=42, more_optional=3.14)

    assert isinstance(result, Inputs)
    assert len(result.proto_inputs.literals) == 3

    literal_names = [lit.name for lit in result.proto_inputs.literals]
    assert literal_names == ["required", "optional", "more_optional"]


@pytest.mark.asyncio
async def test_convert_from_native_to_inputs_union_with_none():
    # Test the Union type hint
    def func_union_none(required: int, maybe_str: Union[str, None] = None):
        pass

    interface = NativeInterface.from_callable(func_union_none)

    # Test Union[T, None] which is equivalent to Optional[T]
    result = await convert.convert_from_native_to_inputs(interface, required=42, maybe_str="value")

    assert isinstance(result, Inputs)
    assert len(result.proto_inputs.literals) == 2

    literals_dict = {lit.name: lit for lit in result.proto_inputs.literals}
    assert "required" in literals_dict
    assert "maybe_str" in literals_dict


@pytest.mark.asyncio
async def test_convert_from_native_to_inputs_missing_required_with_defaults():
    # Missing required parameter when some have defaults
    def func_missing_required(required1: int, required2: str, optional1: float = 3.14):
        pass

    interface = NativeInterface.from_callable(func_missing_required)

    # Only provide one required parameter
    with pytest.raises(ValueError, match="Missing required inputs: required2"):
        await convert.convert_from_native_to_inputs(interface, required1=42)


@pytest.mark.asyncio
async def test_convert_from_native_to_inputs_mixed_positional_with_defaults():
    def func_mixed_positional(pos1: int, pos2: str, pos3: float = 1.0, kw1: bool = False):
        pass

    interface = NativeInterface.from_callable(func_mixed_positional)

    # Mix positional and keyword arguments with defaults
    result = await convert.convert_from_native_to_inputs(interface, 42, "hello", kw1=True)

    assert isinstance(result, Inputs)
    assert len(result.proto_inputs.literals) == 4

    # Check the order of literals matches function parameter order
    literal_names = [lit.name for lit in result.proto_inputs.literals]
    assert literal_names == ["pos1", "pos2", "pos3", "kw1"]

    # Verify all expected parameters are present
    literals_dict = {lit.name: lit for lit in result.proto_inputs.literals}
    assert "pos1" in literals_dict
    assert "pos2" in literals_dict
    assert "pos3" in literals_dict  # Should have default value
    assert "kw1" in literals_dict  # Should have overridden value


int_literal = _run_definition_pb2.NamedLiteral(
    name="int", value=Literal(scalar=Scalar(primitive=Primitive(integer=100)))
)

str_literal = _run_definition_pb2.NamedLiteral(
    name="str", value=Literal(scalar=Scalar(primitive=Primitive(string_value="hello world")))
)

list_literal = _run_definition_pb2.NamedLiteral(
    name="list",
    value=Literal(
        collection=LiteralCollection(
            literals=[
                Literal(scalar=Scalar(primitive=Primitive(string_value="hello"))),
                Literal(scalar=Scalar(primitive=Primitive(string_value="world"))),
            ]
        )
    ),
)

map_literal = _run_definition_pb2.NamedLiteral(
    name="map",
    value=Literal(
        map=LiteralMap(
            literals={
                "first": Literal(scalar=Scalar(primitive=Primitive(string_value="hello"))),
                "second": Literal(scalar=Scalar(primitive=Primitive(string_value="world"))),
            }
        )
    ),
)


@pytest.mark.parametrize(
    "name,inputs,expected_hash",
    [
        (
            "integer",
            _run_definition_pb2.Inputs(
                literals=[
                    int_literal,
                ],
            ),
            "+G832X6Jj7yD8eHKddg8qch8Ks275cSVouLtHO/GBtU=",
        ),
        (
            "string",
            _run_definition_pb2.Inputs(
                literals=[
                    str_literal,
                ]
            ),
            "SgmCD8fWRAnYWXz8qD3oLlH4brlZSgbF3rIgryyJoTA=",
        ),
        (
            "collection",
            _run_definition_pb2.Inputs(
                literals=[
                    list_literal,
                ]
            ),
            "ZzcHFxip4lXwEs4qrLsR0btMhTXrAXRQyUDclhg5mgw=",
        ),
        (
            "map",
            _run_definition_pb2.Inputs(
                literals=[
                    map_literal,
                ]
            ),
            "/R0HHuOwV7kageMb5L83BMWly/XrvJftRqAuIHjMuC4=",
        ),
        (
            "mixed inputs",
            _run_definition_pb2.Inputs(literals=[int_literal, str_literal, list_literal, map_literal]),
            "yTrdSRmRJsnEsbSeR3M9IWDI15oue8AFP4YWf7IhpFE=",
        ),
        ("empty input", _run_definition_pb2.Inputs(), "47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="),
        ("nil input", None, ""),
    ],
)
@pytest.mark.asyncio
def test_generate_inputs_hash_from_proto(name, inputs, expected_hash):
    """
    This test checks that the input hash generation matches that of the server side
    """
    actual = convert.generate_inputs_hash_from_proto(inputs)
    assert actual == expected_hash


@pytest.mark.parametrize(
    "name,interface,expected_hash",
    [
        (
            "integer",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(simple=SimpleType.INTEGER),
                        )
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(simple=SimpleType.INTEGER),
                        )
                    }
                ),
            ),
            "58JU0tE+NylwXlWV5HtOgajWkrbhcqKFsbXdX/QXNPM=",
        ),
        (
            "string",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(simple=SimpleType.STRING),
                        )
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(simple=SimpleType.STRING),
                        )
                    }
                ),
            ),
            "HAPF+vmah1Zt0RLi0cBmewehzCnkvOAbMfmvFO9H3LE=",
        ),
        (
            "float",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(simple=SimpleType.FLOAT),
                        )
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(simple=SimpleType.FLOAT),
                        )
                    }
                ),
            ),
            "qoEbscaX4yyh7pZ8TGgPxrH7dpEFrdMnWNlagZmUdAs=",
        ),
        (
            "boolean",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(simple=SimpleType.BOOLEAN),
                        )
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(simple=SimpleType.BOOLEAN),
                        )
                    }
                ),
            ),
            "WdPjloDgYp6PIg7/gaVq2jL4lNsLjGXjJUdT4XzyBis=",
        ),
        (
            "blob_type",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(
                                blob=BlobType(format="csv", dimensionality=BlobType.BlobDimensionality.SINGLE)
                            ),
                        )
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(
                                blob=BlobType(format="csv", dimensionality=BlobType.BlobDimensionality.SINGLE)
                            ),
                        )
                    }
                ),
            ),
            "DEmszHKzr6b/darsO4qwndUGwtT3yriahy4h2A2+vJU=",
        ),
        (
            "collection_type",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(collection_type=LiteralType(simple=SimpleType.INTEGER)),
                        )
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(collection_type=LiteralType(simple=SimpleType.INTEGER)),
                        )
                    }
                ),
            ),
            "uKGETsnLL4WR+vTE4I2XppcgQ8H/p+TrcDzoISsbmnM=",
        ),
        (
            "map_type",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(map_value_type=LiteralType(simple=SimpleType.STRING)),
                        )
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(map_value_type=LiteralType(simple=SimpleType.STRING)),
                        )
                    }
                ),
            ),
            "OgRavSp74HNEZdW/5SSn0eg/wU5PeEO52sbfNpl06/I=",
        ),
        (
            "enum_type",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(enum_type=EnumType(values=["PENDING", "RUNNING", "COMPLETED", "FAILED"])),
                        )
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(enum_type=EnumType(values=["PENDING", "RUNNING", "COMPLETED", "FAILED"])),
                        )
                    }
                ),
            ),
            "MoJfuK5E44Xy5bhH1ZSm+Hr0v6XUb9uS4h/ZVj1wUcE=",
        ),
        (
            "union_type",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(
                                union_type=UnionType(
                                    variants=[
                                        LiteralType(simple=SimpleType.INTEGER),
                                        LiteralType(simple=SimpleType.STRING),
                                    ]
                                )
                            ),
                        )
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(
                                union_type=UnionType(
                                    variants=[
                                        LiteralType(simple=SimpleType.INTEGER),
                                        LiteralType(simple=SimpleType.STRING),
                                    ]
                                )
                            ),
                        )
                    }
                ),
            ),
            "FHbQNnyP0k7IJt3Jpp8lfgJ/RU8EZEEcYXPuc2ieV5s=",
        ),
        (
            "structured_dataset",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(
                                structured_dataset_type=StructuredDatasetType(
                                    columns=[
                                        StructuredDatasetType.DatasetColumn(
                                            name="feature1", literal_type=LiteralType(simple=SimpleType.FLOAT)
                                        ),
                                        StructuredDatasetType.DatasetColumn(
                                            name="feature2", literal_type=LiteralType(simple=SimpleType.INTEGER)
                                        ),
                                    ],
                                    format="parquet",
                                )
                            ),
                        ),
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(
                                structured_dataset_type=StructuredDatasetType(
                                    columns=[
                                        StructuredDatasetType.DatasetColumn(
                                            name="feature1", literal_type=LiteralType(simple=SimpleType.FLOAT)
                                        ),
                                        StructuredDatasetType.DatasetColumn(
                                            name="feature2", literal_type=LiteralType(simple=SimpleType.INTEGER)
                                        ),
                                    ],
                                    format="parquet",
                                )
                            ),
                        ),
                    }
                ),
            ),
            "kqwSBWQVv4KQ7hzCh4R/8+tErcPkjW7aunm6X0lxFFc=",
        ),
        (
            "empty_interface",
            TypedInterface(inputs=VariableMap(variables={}), outputs=VariableMap(variables={})),
            "j44Ly/GI2FgswKx1w3LUwxY0q/AlHmyoLxfXsDBc9H8=",
        ),
        (
            "nested_collection",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(
                                collection_type=LiteralType(collection_type=LiteralType(simple=SimpleType.FLOAT))
                            ),
                        )
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(
                                collection_type=LiteralType(collection_type=LiteralType(simple=SimpleType.FLOAT))
                            ),
                        )
                    }
                ),
            ),
            "YYCb/LJXF/eKhRfyHXEx9icYagyQ+HTqrZuO6Xui9qs=",
        ),
        (
            "complex_map",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(
                                map_value_type=LiteralType(collection_type=LiteralType(simple=SimpleType.INTEGER))
                            ),
                        )
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(
                                map_value_type=LiteralType(collection_type=LiteralType(simple=SimpleType.INTEGER))
                            ),
                        )
                    }
                ),
            ),
            "1DpgVTQynJAY+4RUedCTOFcSxcirq6Q72EIMidmIXBQ=",
        ),
        (
            "multiple_inputs_and_outputs",
            TypedInterface(
                inputs=VariableMap(
                    variables={
                        "input_1": Variable(
                            type=LiteralType(simple=SimpleType.INTEGER),
                        ),
                        "input_2": Variable(
                            type=LiteralType(simple=SimpleType.STRING),
                        ),
                    }
                ),
                outputs=VariableMap(
                    variables={
                        "output_1": Variable(
                            type=LiteralType(simple=SimpleType.INTEGER),
                        ),
                        "output_2": Variable(
                            type=LiteralType(simple=SimpleType.STRING),
                        ),
                    }
                ),
            ),
            "dmOxMQ5/OKGLvtPFNOG4XcNrcXyaWw9bGaRQqHmk2uw=",
        ),
        ("nil_interface", None, ""),
    ],
)
@pytest.mark.asyncio
def test_generate_interface_hash(name, interface, expected_hash):
    """
    This test checks that the interface hash generation matches that of the server side
    """
    actual = convert.generate_interface_hash(interface)
    assert actual == expected_hash


@pytest.mark.asyncio
def test_generate_interface_hash_order_independence():
    interface1 = TypedInterface(
        inputs=VariableMap(
            variables={
                "a": Variable(type=LiteralType(simple=SimpleType.INTEGER)),
                "b": Variable(type=LiteralType(simple=SimpleType.STRING)),
            }
        )
    )

    interface2 = TypedInterface(
        inputs=VariableMap(
            variables={
                "b": Variable(type=LiteralType(simple=SimpleType.STRING)),
                "a": Variable(type=LiteralType(simple=SimpleType.INTEGER)),
            }
        )
    )

    hash1 = convert.generate_interface_hash(interface1)
    hash2 = convert.generate_interface_hash(interface2)

    assert hash1 == hash2


@pytest.mark.asyncio
async def test_convert_upload_default_inputs_empty():
    """
    convert_upload_default_inputs should return empty list when no defaults are present.
    """
    interface = NativeInterface.from_types({}, {})
    result = await convert.convert_upload_default_inputs(interface)
    assert result == []


@pytest.mark.asyncio
async def test_convert_upload_default_inputs_with_defaults():
    """
    convert_upload_default_inputs should convert default inputs into NamedParameter objects.
    """

    def func(a: int = 10, b: str = "default", c: float | None = None):
        pass

    interface = NativeInterface.from_callable(func)
    result = await convert.convert_upload_default_inputs(interface)

    # Expect one NamedParameter per default, in signature order
    assert [p.name for p in result] == ["a", "b"]

    named = {p.name: p for p in result}
    # a -> integer literal == 10
    assert named["a"].parameter.required is False
    assert named["a"].parameter.default.scalar.primitive.integer == 10
    # b -> string literal == "default"
    assert named["b"].parameter.required is False
    assert named["b"].parameter.default.scalar.primitive.string_value == "default"


@pytest.mark.asyncio
async def test_convert_upload_default_inputs_remote_interface():
    """
    convert_upload_default_inputs should handle remote interfaces correctly.
    """
    interface = NativeInterface.from_types(
        {"x": (int, inspect.Parameter.empty), "y": (str, NativeInterface.has_default)},
        {},
        default_inputs={
            "y": await TypeEngine.to_literal("hello", str, TypeEngine.to_literal_type(str)),
        },
    )
    result = await convert.convert_from_native_to_inputs(interface, 42)

    assert result is not None
    assert result.proto_inputs.literals == [
        run_definition_pb2.NamedLiteral(
            name="x",
            value=await TypeEngine.to_literal(42, int, TypeEngine.to_literal_type(int)),
        ),
        run_definition_pb2.NamedLiteral(
            name="y",
            value=await TypeEngine.to_literal("hello", str, TypeEngine.to_literal_type(str)),
        ),
    ]
