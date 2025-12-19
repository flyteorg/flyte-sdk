"""
Unit tests for _DelayedValue, RunOutput, and AppEndpoint classes.

These tests verify the delayed value classes for app inputs without using mocks
for the initialization checks (unit tests), and with mocks for the async
materialization methods that require remote connections (integration tests).
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import flyte.io
from flyte.app._parameter import (
    PARAMETER_TYPE_MAP,
    AppEndpoint,
    Parameter,
    RunOutput,
    SerializableParameter,
    SerializableParameterCollection,
    _DelayedValue,
)

# =============================================================================
# Tests for _DelayedValue base class
# =============================================================================


def test_delayed_value_type_mapping_from_str_class():
    """
    GOAL: Verify that str type is mapped to 'string' during validation.

    Tests that the model_validator correctly converts the Python str type
    to the serialized 'string' type.
    """

    class ConcreteDelayedValue(_DelayedValue):
        async def materialize(self):
            return "test"

    # Creating with str type should be mapped to "string"
    dv = ConcreteDelayedValue(type=str)
    assert dv.type == "string"


def test_delayed_value_type_mapping_from_file_class():
    """
    GOAL: Verify that flyte.io.File type is mapped to 'file' during validation.
    """

    class ConcreteDelayedValue(_DelayedValue):
        async def materialize(self):
            return flyte.io.File(path="test.txt")

    dv = ConcreteDelayedValue(type=flyte.io.File)
    assert dv.type == "file"


def test_delayed_value_type_mapping_from_dir_class():
    """
    GOAL: Verify that flyte.io.Dir type is mapped to 'directory' during validation.
    """

    class ConcreteDelayedValue(_DelayedValue):
        async def materialize(self):
            return flyte.io.Dir(path="test_dir")

    dv = ConcreteDelayedValue(type=flyte.io.Dir)
    assert dv.type == "directory"


def test_delayed_value_type_already_serialized():
    """
    GOAL: Verify that already serialized type strings pass through unchanged.

    When type is already "string", "file", or "directory", it should remain unchanged.
    """

    class ConcreteDelayedValue(_DelayedValue):
        async def materialize(self):
            return "test"

    dv_string = ConcreteDelayedValue(type="string")
    assert dv_string.type == "string"

    dv_file = ConcreteDelayedValue(type="file")
    assert dv_file.type == "file"

    dv_dir = ConcreteDelayedValue(type="directory")
    assert dv_dir.type == "directory"


@pytest.mark.asyncio
async def test_delayed_value_get_returns_string_directly():
    """
    GOAL: Verify that get() returns string values directly from materialize().
    """

    class ConcreteDelayedValue(_DelayedValue):
        async def materialize(self):
            return "materialized_value"

    dv = ConcreteDelayedValue(type="string")
    result = await dv.get()
    assert result == "materialized_value"


@pytest.mark.asyncio
async def test_delayed_value_get_returns_path_from_file():
    """
    GOAL: Verify that get() extracts .path from File objects returned by materialize().
    """

    class ConcreteDelayedValue(_DelayedValue):
        async def materialize(self):
            return flyte.io.File(path="/path/to/file.txt")

    dv = ConcreteDelayedValue(type="file")
    result = await dv.get()
    assert result.path == "/path/to/file.txt"


@pytest.mark.asyncio
async def test_delayed_value_get_returns_path_from_dir():
    """
    GOAL: Verify that get() extracts .path from Dir objects returned by materialize().
    """

    class ConcreteDelayedValue(_DelayedValue):
        async def materialize(self):
            return flyte.io.Dir(path="/path/to/directory")

    dv = ConcreteDelayedValue(type="directory")
    result = await dv.get()
    assert result.path == "/path/to/directory"


@pytest.mark.asyncio
async def test_delayed_value_get_raises_on_invalid_type():
    """
    GOAL: Verify that get() raises AssertionError for invalid materialized types.
    """

    class ConcreteDelayedValue(_DelayedValue):
        async def materialize(self):
            return 12345  # Invalid type

    dv = ConcreteDelayedValue(type="string")
    with pytest.raises(AssertionError, match="Materialized value must be a string"):
        await dv.get()


@pytest.mark.asyncio
async def test_delayed_value_materialize_not_implemented():
    """
    GOAL: Verify that base class materialize() raises NotImplementedError.
    """
    dv = _DelayedValue(type="string")
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        await dv.materialize()


# =============================================================================
# Tests for RunOutput class
# =============================================================================


def test_run_output_with_run_name():
    """
    GOAL: Verify RunOutput can be created with a run_name.

    Tests basic instantiation with run_name and default values.
    """
    ro = RunOutput(type="string", run_name="my-run-123")
    assert ro.run_name == "my-run-123"
    assert ro.task_name is None
    assert ro.task_version is None
    assert ro.task_auto_version is None
    assert ro.getter == (0,)


def test_run_output_with_task_name_auto_version():
    """
    GOAL: Verify RunOutput can be created with task_name and auto_version.

    Tests instantiation with task_name and auto_version for latest task.
    """
    ro = RunOutput(type="file", task_name="my-task", task_auto_version="latest")
    assert ro.task_name == "my-task"
    assert ro.task_auto_version == "latest"
    assert ro.run_name is None
    assert ro.task_version is None


def test_run_output_with_task_name_and_version():
    """
    GOAL: Verify RunOutput can be created with task_name and explicit version.

    Tests instantiation with task_name and task_version (auto_version=None).
    """
    ro = RunOutput(type="directory", task_name="my-task", task_version="v1.0.0", task_auto_version=None)
    assert ro.task_name == "my-task"
    assert ro.task_version == "v1.0.0"
    assert ro.task_auto_version is None


def test_run_output_with_custom_getter():
    """
    GOAL: Verify RunOutput can use custom getter tuple for nested output access.

    Tests that getter can be customized to access nested output structures.
    """
    ro = RunOutput(type="string", run_name="my-run", getter=("output_key", 0, "nested"))
    assert ro.getter == ("output_key", 0, "nested")


def test_run_output_type_mapping():
    """
    GOAL: Verify RunOutput correctly maps Python types to serialized types.

    Tests that the inherited model_validator from _DelayedValue works correctly.
    """
    ro_str = RunOutput(type=str, run_name="run-1")
    assert ro_str.type == "string"

    ro_file = RunOutput(type=flyte.io.File, run_name="run-2")
    assert ro_file.type == "file"

    ro_dir = RunOutput(type=flyte.io.Dir, run_name="run-3")
    assert ro_dir.type == "directory"


def test_run_output_model_dump_json():
    """
    GOAL: Verify RunOutput can be serialized to JSON.

    Tests that model_dump_json produces valid JSON for transport.
    """
    ro = RunOutput(type="string", run_name="my-run-123", getter=(0,))
    json_str = ro.model_dump_json()
    assert "my-run-123" in json_str
    assert "string" in json_str


def test_run_output_model_validate_json():
    """
    GOAL: Verify RunOutput can be deserialized from JSON.

    Tests round-trip serialization/deserialization.
    """
    ro = RunOutput(type="string", run_name="my-run-123", getter=(0, "key"))
    json_str = ro.model_dump_json()

    ro_restored = RunOutput.model_validate_json(json_str)
    assert ro_restored.run_name == "my-run-123"
    assert ro_restored.type == "string"
    assert ro_restored.getter == (0, "key")


@pytest.mark.asyncio
async def test_run_output_materialize_with_run_name():
    """
    GOAL: Verify RunOutput materializes correctly with run_name.

    Uses mocks to simulate the remote API calls.
    """
    # Create mock objects
    mock_run_details = MagicMock()
    mock_run_details.outputs = AsyncMock(return_value=["output_value_0", "output_value_1"])

    mock_run = MagicMock()
    mock_run.details = MagicMock()
    mock_run.details.aio = AsyncMock(return_value=mock_run_details)

    # Patch the remote module and is_initialized check
    with patch("flyte.remote.Run") as MockRun, patch("flyte._initialize.is_initialized", return_value=True):
        MockRun.get = MagicMock()
        MockRun.get.aio = AsyncMock(return_value=mock_run)

        ro = RunOutput(type="string", run_name="my-run-123", getter=(0,))
        result = await ro.materialize()

        assert result == "output_value_0"
        MockRun.get.aio.assert_called_once_with("my-run-123")


@pytest.mark.asyncio
async def test_run_output_materialize_with_task_name_auto_version():
    """
    GOAL: Verify RunOutput materializes correctly with task_name and auto_version.

    Uses mocks to simulate the remote API calls for task lookup.
    """
    # Create mock objects
    mock_task_details = MagicMock()
    mock_task_details.version = "v1.2.3"

    mock_task = MagicMock()
    mock_task.fetch = MagicMock()
    mock_task.fetch.aio = AsyncMock(return_value=mock_task_details)

    mock_run_details = MagicMock()
    mock_run_details.outputs = AsyncMock(return_value={"result": "task_output"})

    mock_run = MagicMock()
    mock_run.details = MagicMock()
    mock_run.details.aio = AsyncMock(return_value=mock_run_details)

    async def mock_listall_aio(*args, **kwargs):
        yield mock_run

    with (
        patch("flyte.remote.Task") as MockTask,
        patch("flyte.remote.Run") as MockRun,
        patch("flyte._initialize.is_initialized", return_value=True),
    ):
        MockTask.get = MagicMock(return_value=mock_task)
        MockRun.listall = MagicMock()
        MockRun.listall.aio = mock_listall_aio

        ro = RunOutput(type="string", task_name="my-task", task_auto_version="latest", getter=("result",))
        result = await ro.materialize()

        assert result == "task_output"
        MockTask.get.assert_called_once_with("my-task", version=None, auto_version="latest")


@pytest.mark.asyncio
async def test_run_output_materialize_with_task_version():
    """
    GOAL: Verify RunOutput materializes correctly with explicit task_version.

    Uses mocks to verify task_version is used directly without fetching.
    """
    mock_run_details = MagicMock()
    mock_run_details.outputs = AsyncMock(return_value=["output_0"])

    mock_run = MagicMock()
    mock_run.details = MagicMock()
    mock_run.details.aio = AsyncMock(return_value=mock_run_details)

    async def mock_listall_aio(*args, **kwargs):
        yield mock_run

    with (
        patch("flyte.remote.Run") as MockRun,
        patch("flyte._initialize.is_initialized", return_value=True),
    ):
        MockRun.listall = MagicMock()
        MockRun.listall.aio = mock_listall_aio

        ro = RunOutput(type="string", task_name="my-task", task_version="v1.0.0", task_auto_version=None)
        result = await ro.materialize()

        assert result == "output_0"
        # Verify listall was used (it's an async generator so we can't use assert_called_once)


# =============================================================================
# Tests for AppEndpoint class
# =============================================================================


def test_app_endpoint_basic_creation():
    """
    GOAL: Verify AppEndpoint can be created with basic parameters.

    Tests instantiation with app_name and default values.
    """
    ae = AppEndpoint(app_name="upstream-app")
    assert ae.app_name == "upstream-app"
    assert ae.public is False
    assert ae.type == "string"


def test_app_endpoint_public():
    """
    GOAL: Verify AppEndpoint can be created with public=True.

    Tests that public endpoint flag is correctly set.
    """
    ae = AppEndpoint(app_name="upstream-app", public=True)
    assert ae.public is True


def test_app_endpoint_type_is_always_string():
    """
    GOAL: Verify AppEndpoint type is always 'string'.

    AppEndpoint always returns a URL string, so type is fixed to 'string'.
    """
    ae = AppEndpoint(app_name="upstream-app")
    assert ae.type == "string"


def test_app_endpoint_model_dump_json():
    """
    GOAL: Verify AppEndpoint can be serialized to JSON.

    Tests that model_dump_json produces valid JSON for transport.
    """
    ae = AppEndpoint(app_name="upstream-app", public=True)
    json_str = ae.model_dump_json()
    assert "upstream-app" in json_str
    assert "true" in json_str.lower()


def test_app_endpoint_model_validate_json():
    """
    GOAL: Verify AppEndpoint can be deserialized from JSON.

    Tests round-trip serialization/deserialization.
    """
    ae = AppEndpoint(app_name="upstream-app", public=True)
    json_str = ae.model_dump_json()

    ae_restored = AppEndpoint.model_validate_json(json_str)
    assert ae_restored.app_name == "upstream-app"
    assert ae_restored.public is True
    assert ae_restored.type == "string"


@pytest.mark.asyncio
async def test_app_endpoint_materialize_private():
    """
    GOAL: Verify AppEndpoint materializes private endpoint from env var.

    Tests that private endpoints are constructed using the env var pattern.
    """
    with (
        patch.dict(os.environ, {"INTERNAL_APP_ENDPOINT_PATTERN": "http://{app_fqdn}.internal:8080"}),
        patch("flyte._initialize.is_initialized", return_value=True),
    ):
        ae = AppEndpoint(app_name="upstream-app", public=False)
        result = await ae.materialize()

        assert result == "http://upstream-app.internal:8080"


@pytest.mark.asyncio
async def test_app_endpoint_materialize_public():
    """
    GOAL: Verify AppEndpoint materializes public endpoint via remote App.

    Uses mocks to simulate the remote API call for public endpoint.
    """
    mock_app = MagicMock()
    mock_app.endpoint = "https://upstream-app.example.com"

    with (
        patch("flyte.remote.App") as MockApp,
        patch("flyte._initialize.is_initialized", return_value=True),
    ):
        MockApp.get = MagicMock(return_value=mock_app)

        ae = AppEndpoint(app_name="upstream-app", public=True)
        result = await ae.materialize()

        assert result == "https://upstream-app.example.com"
        MockApp.get.assert_called_once_with("upstream-app")


@pytest.mark.asyncio
async def test_app_endpoint_materialize_private_no_env_var():
    """
    GOAL: Verify AppEndpoint raises error when private and env var not set.

    Tests that ValueError is raised when trying to create private endpoint
    without the INTERNAL_APP_ENDPOINT_PATTERN environment variable.
    """
    # Ensure the env var is not set
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("flyte._initialize.is_initialized", return_value=True),
    ):
        ae = AppEndpoint(app_name="upstream-app", public=False)

        with pytest.raises(ValueError, match="INTERNAL_APP_ENDPOINT_PATTERN"):
            await ae.materialize()


# =============================================================================
# Tests for Parameter class with delayed values
# =============================================================================


def test_input_with_run_output():
    """
    GOAL: Verify Parameter can accept RunOutput as a value.

    Tests that Parameter properly validates and stores RunOutput delayed values.
    """
    ro = RunOutput(type="string", run_name="my-run-123")
    param = Parameter(name="model_path", value=ro)

    assert param.name == "model_path"
    assert isinstance(param.value, RunOutput)
    assert param.value.run_name == "my-run-123"


def test_input_with_app_endpoint():
    """
    GOAL: Verify Parameter can accept AppEndpoint as a value.

    Tests that Parameter properly validates and stores AppEndpoint delayed values.
    """
    ae = AppEndpoint(app_name="upstream-app")
    param = Parameter(name="api_url", value=ae)

    assert param.name == "api_url"
    assert isinstance(param.value, AppEndpoint)
    assert param.value.app_name == "upstream-app"


def test_input_with_string_value():
    """
    GOAL: Verify Parameter still works with plain string values.

    Tests backward compatibility with string values.
    """
    param = Parameter(name="config", value="config.yaml")

    assert param.name == "config"
    assert param.value == "config.yaml"


def test_input_with_file_value():
    """
    GOAL: Verify Parameter works with flyte.io.File values.

    Tests that File objects are accepted as parameter values.
    """
    file = flyte.io.File(path="s3://bucket/file.txt")
    param = Parameter(name="model", value=file)

    assert param.name == "model"
    assert isinstance(param.value, flyte.io.File)


def test_input_with_dir_value():
    """
    GOAL: Verify Parameter works with flyte.io.Dir values.

    Tests that Dir objects are accepted as parameter values.
    """
    dir_val = flyte.io.Dir(path="s3://bucket/data/")
    param = Parameter(name="data", value=dir_val)

    assert param.name == "data"
    assert isinstance(param.value, flyte.io.Dir)


# =============================================================================
# Tests for SerializableParameter with delayed values
# =============================================================================


def test_serializable_input_from_run_output():
    """
    GOAL: Verify SerializableParameter.from_parameter() handles RunOutput correctly.

    Tests that RunOutput is serialized to JSON and type is preserved.
    """
    ro = RunOutput(type="file", run_name="my-run-123", getter=(0,))
    param = Parameter(name="model", value=ro)

    serialized = SerializableParameter.from_parameter(param)

    assert serialized.name == "model"
    assert serialized.type == "file"
    # Value should be JSON serialized
    assert "my-run-123" in serialized.value
    assert serialized.download is False


def test_serializable_input_from_run_output_with_mount():
    """
    GOAL: Verify SerializableParameter.from_parameter() handles RunOutput with mount.

    Tests that download is set to True when mount is specified.
    """
    ro = RunOutput(type="file", run_name="my-run-123")
    param = Parameter(name="model", value=ro, mount="/mnt/model")

    serialized = SerializableParameter.from_parameter(param)

    assert serialized.name == "model"
    assert serialized.download is True
    assert serialized.dest == "/mnt/model"


def test_serializable_input_from_app_endpoint():
    """
    GOAL: Verify SerializableParameter.from_parameter() handles AppEndpoint correctly.

    Tests that AppEndpoint is serialized to JSON with type 'string'.
    """
    ae = AppEndpoint(app_name="upstream-app", public=True)
    param = Parameter(name="api_url", value=ae, env_var="API_URL")

    serialized = SerializableParameter.from_parameter(param)

    assert serialized.name == "api_url"
    assert serialized.type == "string"
    assert serialized.env_var == "API_URL"
    # Value should be JSON serialized
    assert "upstream-app" in serialized.value


def test_serializable_input_collection_with_mixed_values():
    """
    GOAL: Verify SerializableParameterCollection handles mixed parameter types.

    Tests that a collection with string, File, Dir, RunOutput, and AppEndpoint
    all serialize correctly.
    """
    parameters = [
        Parameter(name="config", value="config.yaml"),
        Parameter(name="model_file", value=flyte.io.File(path="s3://bucket/model.pkl"), mount="/mnt/model"),
        Parameter(name="data_dir", value=flyte.io.Dir(path="s3://bucket/data/"), mount="/mnt/data"),
        Parameter(name="run_output", value=RunOutput(type="string", run_name="run-123")),
        Parameter(name="api_url", value=AppEndpoint(app_name="upstream-app")),
    ]

    collection = SerializableParameterCollection.from_parameters(parameters)

    assert len(collection.parameters) == 5
    assert collection.parameters[0].type == "string"
    assert collection.parameters[0].value == "config.yaml"
    assert collection.parameters[1].type == "file"
    assert collection.parameters[1].download is True
    assert collection.parameters[2].type == "directory"
    assert collection.parameters[2].download is True
    assert collection.parameters[3].type == "string"  # RunOutput type
    assert collection.parameters[4].type == "string"  # AppEndpoint type


def test_serializable_input_collection_transport_roundtrip():
    """
    GOAL: Verify SerializableParameterCollection can be transported and restored.

    Tests the full round-trip: parameters -> serialization -> transport -> deserialization.
    """
    parameters = [
        Parameter(name="config", value="config.yaml"),
        Parameter(name="model", value=RunOutput(type="file", run_name="run-123")),
        Parameter(name="api", value=AppEndpoint(app_name="upstream-app")),
    ]

    collection = SerializableParameterCollection.from_parameters(parameters)
    transport_str = collection.to_transport

    restored = SerializableParameterCollection.from_transport(transport_str)

    assert len(restored.parameters) == 3
    assert restored.parameters[0].name == "config"
    assert restored.parameters[0].value == "config.yaml"
    assert restored.parameters[1].name == "model"
    assert "run-123" in restored.parameters[1].value  # JSON contains run_name
    assert restored.parameters[2].name == "api"
    assert "upstream-app" in restored.parameters[2].value  # JSON contains app_name
