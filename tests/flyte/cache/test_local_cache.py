import pytest
from flyteidl.core.interface_pb2 import TypedInterface, Variable, VariableMap
from flyteidl.core.literals_pb2 import Literal, Primitive, Scalar
from flyteidl.core.types_pb2 import LiteralType, SimpleType

from flyte._cache.local_cache import LocalTaskCache
from flyte._internal.runtime import convert
from flyte._protos.workflow import run_definition_pb2


@pytest.mark.asyncio
async def test_set_and_get_cache():
    task_name = "test_task"
    cache_version = "v1"
    ignore_input_vars = []

    proto_inputs = convert.Inputs(
        proto_inputs=run_definition_pb2.Inputs(
            literals=[
                run_definition_pb2.NamedLiteral(name="x", value=Literal(scalar=Scalar(primitive=Primitive(integer=42))))
            ]
        )
    )
    inputs_hash = convert.generate_inputs_hash_from_proto(proto_inputs.proto_inputs)

    task_interface = TypedInterface(
        inputs=VariableMap(variables={"x": Variable(type=LiteralType(simple=SimpleType.INTEGER))})
    )

    expected_output = convert.Outputs(
        proto_outputs=run_definition_pb2.Outputs(
            literals=[
                run_definition_pb2.NamedLiteral(
                    name="result", value=Literal(scalar=Scalar(primitive=Primitive(string_value="cached_result")))
                )
            ]
        )
    )

    cached_output = await LocalTaskCache.get(
        task_name, inputs_hash, proto_inputs, task_interface, cache_version, ignore_input_vars
    )
    assert cached_output is None

    await LocalTaskCache.set(
        task_name, inputs_hash, proto_inputs, task_interface, cache_version, ignore_input_vars, expected_output
    )

    cached_output = await LocalTaskCache.get(
        task_name, inputs_hash, proto_inputs, task_interface, cache_version, ignore_input_vars
    )
    assert cached_output is not None
    assert cached_output == expected_output


@pytest.mark.asyncio
async def test_set_overwrites_existing():
    task_name = "test_task"
    cache_version = "v1"
    ignore_input_vars = []

    proto_inputs = convert.Inputs(
        proto_inputs=run_definition_pb2.Inputs(
            literals=[
                run_definition_pb2.NamedLiteral(name="x", value=Literal(scalar=Scalar(primitive=Primitive(integer=42))))
            ]
        )
    )
    inputs_hash = convert.generate_inputs_hash_from_proto(proto_inputs.proto_inputs)

    task_interface = TypedInterface(
        inputs=VariableMap(variables={"x": Variable(type=LiteralType(simple=SimpleType.INTEGER))})
    )

    first_output = convert.Outputs(
        proto_outputs=run_definition_pb2.Outputs(
            literals=[
                run_definition_pb2.NamedLiteral(
                    name="result", value=Literal(scalar=Scalar(primitive=Primitive(string_value="first_result")))
                )
            ]
        )
    )

    second_output = convert.Outputs(
        proto_outputs=run_definition_pb2.Outputs(
            literals=[
                run_definition_pb2.NamedLiteral(
                    name="result", value=Literal(scalar=Scalar(primitive=Primitive(string_value="second_result")))
                )
            ]
        )
    )

    # Set the cache for the first time
    await LocalTaskCache.set(
        task_name, inputs_hash, proto_inputs, task_interface, cache_version, ignore_input_vars, first_output
    )
    result = await LocalTaskCache.get(
        task_name, inputs_hash, proto_inputs, task_interface, cache_version, ignore_input_vars
    )
    assert result == first_output

    # Set the cache with same key, should overwrite the original value
    await LocalTaskCache.set(
        task_name, inputs_hash, proto_inputs, task_interface, cache_version, ignore_input_vars, second_output
    )
    result = await LocalTaskCache.get(
        task_name, inputs_hash, proto_inputs, task_interface, cache_version, ignore_input_vars
    )
    assert result == second_output


@pytest.mark.asyncio
async def test_clear_cache():
    task_name = "test_task"
    cache_version = "v1"
    ignore_input_vars = []

    proto_inputs = convert.Inputs(
        proto_inputs=run_definition_pb2.Inputs(
            literals=[
                run_definition_pb2.NamedLiteral(name="x", value=Literal(scalar=Scalar(primitive=Primitive(integer=42))))
            ]
        )
    )
    inputs_hash = convert.generate_inputs_hash_from_proto(proto_inputs.proto_inputs)

    task_interface = TypedInterface(
        inputs=VariableMap(variables={"x": Variable(type=LiteralType(simple=SimpleType.INTEGER))})
    )

    output = convert.Outputs(
        proto_outputs=run_definition_pb2.Outputs(
            literals=[
                run_definition_pb2.NamedLiteral(
                    name="result", value=Literal(scalar=Scalar(primitive=Primitive(string_value="test_result")))
                )
            ]
        )
    )

    await LocalTaskCache.set(
        task_name, inputs_hash, proto_inputs, task_interface, cache_version, ignore_input_vars, output
    )

    result = await LocalTaskCache.get(
        task_name, inputs_hash, proto_inputs, task_interface, cache_version, ignore_input_vars
    )
    assert result is not None

    await LocalTaskCache.clear()

    # Ensure the cache is cleared
    result = await LocalTaskCache.get(
        task_name, inputs_hash, proto_inputs, task_interface, cache_version, ignore_input_vars
    )
    assert result is None
