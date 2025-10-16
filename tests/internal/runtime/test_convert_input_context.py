"""Tests for custom_context serialization in convert module."""

import pytest

from flyte._context import internal_ctx
from flyte._internal.runtime.convert import convert_from_native_to_inputs
from flyte.models import ActionID, NativeInterface, RawDataPath, TaskContext
from flyte.report import Report


@pytest.fixture
def task_context_with_custom_context():
    """Create a task context with custom_context for testing."""
    return TaskContext(
        action=ActionID(name="test"),
        run_base_dir="test",
        output_path="test",
        raw_data_path=RawDataPath(path=""),
        version="",
        report=Report("test"),
        custom_context={"env": "production", "region": "us-west-2"},
    )


@pytest.mark.asyncio
async def test_convert_inputs_with_context(task_context_with_custom_context):
    """Test that input context is properly serialized in inputs."""
    interface = NativeInterface(inputs={}, outputs={})

    ctx = internal_ctx()
    with ctx.replace_task_context(task_context_with_custom_context):
        inputs = await convert_from_native_to_inputs(interface)

        # Verify context is in the proto
        assert len(inputs.proto_inputs.context) == 2
        context_dict = {kv.key: kv.value for kv in inputs.proto_inputs.context}
        assert context_dict == {"env": "production", "region": "us-west-2"}


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

    tctx = TaskContext(
        action=ActionID(name="test"),
        run_base_dir="test",
        output_path="test",
        raw_data_path=RawDataPath(path=""),
        version="",
        report=Report("test"),
        custom_context={"env": "production"},
    )

    ctx = internal_ctx()
    with ctx.replace_task_context(tctx):
        inputs = await convert_from_native_to_inputs(interface, x=42)

        # Verify both literals and context are present
        assert len(inputs.proto_inputs.literals) == 1
        assert inputs.proto_inputs.literals[0].name == "x"
        assert len(inputs.proto_inputs.context) == 1
        assert inputs.proto_inputs.context[0].key == "env"
        assert inputs.proto_inputs.context[0].value == "production"


@pytest.mark.asyncio
async def test_inputs_context_property(task_context_with_custom_context):
    """Test the Inputs.context property."""
    interface = NativeInterface(inputs={}, outputs={})

    ctx = internal_ctx()
    with ctx.replace_task_context(task_context_with_custom_context):
        inputs = await convert_from_native_to_inputs(interface)

        # Test the context property
        result_context = inputs.context
        assert result_context == {"env": "production", "region": "us-west-2"}


@pytest.mark.asyncio
async def test_empty_context():
    """Test that empty context dict works correctly."""
    interface = NativeInterface(inputs={}, outputs={})

    inputs = await convert_from_native_to_inputs(interface)

    # Verify no context
    assert len(inputs.proto_inputs.context) == 0
    assert inputs.context == {}


@pytest.mark.asyncio
async def test_context_with_empty_interface(task_context_with_custom_context):
    """Test context serialization when interface has no inputs."""
    interface = NativeInterface(inputs={}, outputs={})

    ctx = internal_ctx()
    with ctx.replace_task_context(task_context_with_custom_context):
        inputs = await convert_from_native_to_inputs(interface)

        # Should use the empty interface path but still include context
        assert len(inputs.proto_inputs.literals) == 0
        assert len(inputs.proto_inputs.context) == 2
        assert inputs.context == {"env": "production", "region": "us-west-2"}
