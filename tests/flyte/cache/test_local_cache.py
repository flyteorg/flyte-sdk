import pytest
from flyteidl2.core.interface_pb2 import TypedInterface, Variable, VariableMap
from flyteidl2.core.literals_pb2 import Literal, Primitive, Scalar
from flyteidl2.core.types_pb2 import LiteralType, SimpleType
from flyteidl2.task import common_pb2

from flyte._cache.local_cache import LocalTaskCache
from flyte._internal.runtime import convert


@pytest.mark.asyncio
async def test_set_and_get_cache():
    task_name = "test_task"
    cache_version = "v1"
    ignore_input_vars = []

    proto_inputs = convert.Inputs(
        proto_inputs=common_pb2.Inputs(
            literals=[common_pb2.NamedLiteral(name="x", value=Literal(scalar=Scalar(primitive=Primitive(integer=42))))]
        )
    )
    inputs_hash = convert.generate_inputs_hash_from_proto(proto_inputs.proto_inputs)

    task_interface = TypedInterface(
        inputs=VariableMap(variables={"x": Variable(type=LiteralType(simple=SimpleType.INTEGER))})
    )

    expected_output = convert.Outputs(
        proto_outputs=common_pb2.Outputs(
            literals=[
                common_pb2.NamedLiteral(
                    name="result", value=Literal(scalar=Scalar(primitive=Primitive(string_value="cached_result")))
                )
            ]
        )
    )

    cache_key = convert.generate_cache_key_hash(
        task_name, inputs_hash, task_interface, cache_version, ignore_input_vars, proto_inputs.proto_inputs
    )

    cached_output = await LocalTaskCache.get(cache_key)
    assert cached_output is None

    await LocalTaskCache.set(cache_key, expected_output)

    cached_output = await LocalTaskCache.get(cache_key)
    assert cached_output is not None
    assert cached_output == expected_output


@pytest.mark.asyncio
async def test_set_overwrites_existing():
    task_name = "test_task"
    cache_version = "v1"
    ignore_input_vars = []

    proto_inputs = convert.Inputs(
        proto_inputs=common_pb2.Inputs(
            literals=[common_pb2.NamedLiteral(name="x", value=Literal(scalar=Scalar(primitive=Primitive(integer=42))))]
        )
    )
    inputs_hash = convert.generate_inputs_hash_from_proto(proto_inputs.proto_inputs)

    task_interface = TypedInterface(
        inputs=VariableMap(variables={"x": Variable(type=LiteralType(simple=SimpleType.INTEGER))})
    )

    first_output = convert.Outputs(
        proto_outputs=common_pb2.Outputs(
            literals=[
                common_pb2.NamedLiteral(
                    name="result", value=Literal(scalar=Scalar(primitive=Primitive(string_value="first_result")))
                )
            ]
        )
    )

    second_output = convert.Outputs(
        proto_outputs=common_pb2.Outputs(
            literals=[
                common_pb2.NamedLiteral(
                    name="result", value=Literal(scalar=Scalar(primitive=Primitive(string_value="second_result")))
                )
            ]
        )
    )

    cache_key = convert.generate_cache_key_hash(
        task_name, inputs_hash, task_interface, cache_version, ignore_input_vars, proto_inputs.proto_inputs
    )

    # Set the cache for the first time
    await LocalTaskCache.set(cache_key, first_output)
    result = await LocalTaskCache.get(cache_key)
    assert result == first_output

    # Set the cache with same key, should overwrite the original value
    await LocalTaskCache.set(cache_key, second_output)
    result = await LocalTaskCache.get(cache_key)
    assert result == second_output


@pytest.mark.asyncio
async def test_clear_cache():
    task_name = "test_task"
    cache_version = "v1"
    ignore_input_vars = []

    proto_inputs = convert.Inputs(
        proto_inputs=common_pb2.Inputs(
            literals=[common_pb2.NamedLiteral(name="x", value=Literal(scalar=Scalar(primitive=Primitive(integer=42))))]
        )
    )
    inputs_hash = convert.generate_inputs_hash_from_proto(proto_inputs.proto_inputs)

    task_interface = TypedInterface(
        inputs=VariableMap(variables={"x": Variable(type=LiteralType(simple=SimpleType.INTEGER))})
    )

    output = convert.Outputs(
        proto_outputs=common_pb2.Outputs(
            literals=[
                common_pb2.NamedLiteral(
                    name="result", value=Literal(scalar=Scalar(primitive=Primitive(string_value="test_result")))
                )
            ]
        )
    )

    cache_key = convert.generate_cache_key_hash(
        task_name, inputs_hash, task_interface, cache_version, ignore_input_vars, proto_inputs.proto_inputs
    )

    await LocalTaskCache.set(cache_key, output)

    result = await LocalTaskCache.get(cache_key)
    assert result is not None

    await LocalTaskCache.clear()

    # Ensure the cache is cleared
    result = await LocalTaskCache.get(cache_key)
    assert result is None
