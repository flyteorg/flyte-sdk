"""Tests for input_context serialization in convert module."""
import pytest

from flyte._internal.runtime.convert import convert_from_native_to_inputs
from flyte.models import NativeInterface


@pytest.mark.asyncio
async def test_convert_inputs_with_context():
    """Test that input context is properly serialized in inputs."""
    interface = NativeInterface(inputs={}, outputs={})
    context = {"env": "production", "region": "us-west-2"}

    inputs = await convert_from_native_to_inputs(interface, input_context=context)

    # Verify context is in the proto
    assert len(inputs.proto_inputs.context) == 2
    context_dict = {kv.key: kv.value for kv in inputs.proto_inputs.context}
    assert context_dict == context


@pytest.mark.asyncio
async def test_convert_inputs_without_context():
    """Test that inputs work without context."""
    interface = NativeInterface(inputs={}, outputs={})

    inputs = await convert_from_native_to_inputs(interface)

    # Verify no context
    assert len(inputs.proto_inputs.context) == 0


@pytest.mark.asyncio
async def test_convert_inputs_with_parameters_and_context():
    """Test that both parameters and context are serialized."""
    interface = NativeInterface(inputs={"x": (int, ...)}, outputs={})
    context = {"env": "production"}

    inputs = await convert_from_native_to_inputs(interface, x=42, input_context=context)

    # Verify both literals and context are present
    assert len(inputs.proto_inputs.literals) == 1
    assert inputs.proto_inputs.literals[0].name == "x"
    assert len(inputs.proto_inputs.context) == 1
    assert inputs.proto_inputs.context[0].key == "env"
    assert inputs.proto_inputs.context[0].value == "production"


@pytest.mark.asyncio
async def test_inputs_context_property():
    """Test the Inputs.context property."""
    interface = NativeInterface(inputs={}, outputs={})
    context = {"key1": "value1", "key2": "value2"}

    inputs = await convert_from_native_to_inputs(interface, input_context=context)

    # Test the context property
    result_context = inputs.context
    assert result_context == context


@pytest.mark.asyncio
async def test_empty_context():
    """Test that empty context dict works correctly."""
    interface = NativeInterface(inputs={}, outputs={})

    inputs = await convert_from_native_to_inputs(interface, input_context={})

    # Verify no context
    assert len(inputs.proto_inputs.context) == 0
    assert inputs.context == {}


@pytest.mark.asyncio
async def test_context_with_empty_interface():
    """Test context serialization when interface has no inputs."""
    interface = NativeInterface(inputs={}, outputs={})
    context = {"project": "my-project", "entity": "my-entity"}

    inputs = await convert_from_native_to_inputs(interface, input_context=context)

    # Should use the empty interface path but still include context
    assert len(inputs.proto_inputs.literals) == 0
    assert len(inputs.proto_inputs.context) == 2
    assert inputs.context == context
