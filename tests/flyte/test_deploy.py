from __future__ import annotations

import inspect
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, patch

import pytest

import flyte
from flyte._build import ImageBuild
from flyte._deploy import (
    DeploymentPlan,
    _build_image_bg,
    _build_images,
    _get_documentation_entity,
    _update_interface_inputs_and_outputs_docstring,
)
from flyte._docstring import Docstring
from flyte._internal.imagebuild.image_builder import ImageCache
from flyte._internal.runtime.types_serde import transform_native_to_typed_interface
from flyte.models import NativeInterface


def test_get_description_entity_both_descriptions_truncated():
    # Create descriptions that exceed both limits
    env_desc = "a" * 300
    short_desc = "c" * 300
    long_desc = "d" * 3000
    env = flyte.TaskEnvironment(name="test_env", image="python:3.10", description=env_desc)

    @env.task()
    async def task_both_exceed(x: int) -> int:
        return x * 2

    # Create a mock docstring with both descriptions exceeding limits
    mock_parsed_docstring = Mock()
    mock_parsed_docstring.short_description = short_desc
    mock_parsed_docstring.long_description = long_desc

    docstring = Docstring()
    docstring._parsed_docstring = mock_parsed_docstring
    # Use replace since NativeInterface is frozen
    task_both_exceed.interface = replace(task_both_exceed.interface, docstring=docstring)

    result = _get_documentation_entity(task_both_exceed)

    # Verify truncation with ...(tr.) suffix
    assert env.description == "a" * 247 + "...(tr.)"
    assert len(env.description) == 255
    assert result.short_description == "c" * 247 + "...(tr.)"
    assert len(result.short_description) == 255
    assert result.long_description == "d" * 2040 + "...(tr.)"
    assert len(result.long_description) == 2048


def test_get_description_entity_none_values():
    env = flyte.TaskEnvironment(name="test_env", image="python:3.10")

    @env.task()
    async def task_no_docstring(x: int) -> int:
        return x * 2

    # Create a mock docstring with None descriptions
    mock_parsed_docstring = Mock()
    mock_parsed_docstring.short_description = None
    mock_parsed_docstring.long_description = None

    docstring = Docstring()
    docstring._parsed_docstring = mock_parsed_docstring
    # Use replace since NativeInterface is frozen
    task_no_docstring.interface = replace(task_no_docstring.interface, docstring=docstring)

    result = _get_documentation_entity(task_no_docstring)

    # Verify None values are handled correctly
    # Note: protobuf converts None to empty string for string fields
    assert env.description is None
    assert result.short_description == ""
    assert result.long_description == ""


def test_update_interface_with_docstring():
    docstring_text = """
    A test function.

    Args:
        x: The input value
        y: Another input

    Returns:
        The result
    """

    interface = NativeInterface(
        inputs={"x": (int, inspect.Parameter.empty), "y": (str, inspect.Parameter.empty)},
        outputs={"o0": int},
        docstring=Docstring(docstring=docstring_text),
    )

    typed_interface = transform_native_to_typed_interface(interface)

    # Before update, descriptions should be empty
    inputs_dict = {entry.key: entry.value for entry in typed_interface.inputs.variables}
    assert inputs_dict["x"].description == ""
    assert inputs_dict["y"].description == ""

    # Update descriptions
    result = _update_interface_inputs_and_outputs_docstring(typed_interface, interface)

    # After update, descriptions should be set
    result_inputs = {entry.key: entry.value for entry in result.inputs.variables}
    assert result_inputs["x"].description == "The input value"
    assert result_inputs["y"].description == "Another input"


def test_update_interface_no_docstring():
    interface = NativeInterface(
        inputs={"x": (int, inspect.Parameter.empty)},
        outputs={"o0": int},
        docstring=None,
    )

    typed_interface = transform_native_to_typed_interface(interface)
    result = _update_interface_inputs_and_outputs_docstring(typed_interface, interface)

    # Descriptions should remain empty
    result_inputs = {entry.key: entry.value for entry in result.inputs.variables}
    result_outputs = {entry.key: entry.value for entry in result.outputs.variables}
    assert result_inputs["x"].description == ""
    assert result_outputs["o0"].description == ""


def test_update_interface_empty_interface():
    interface = NativeInterface(
        inputs={},
        outputs={},
        docstring=None,
    )

    typed_interface = transform_native_to_typed_interface(interface)
    result = _update_interface_inputs_and_outputs_docstring(typed_interface, interface)

    # Should not raise any errors
    assert len(result.inputs.variables) == 0
    assert len(result.outputs.variables) == 0


def test_update_interface_partial_descriptions():
    docstring_text = """
    A test function.

    Args:
        x: The input value
    """

    interface = NativeInterface(
        inputs={"x": (int, inspect.Parameter.empty), "y": (str, inspect.Parameter.empty)},
        outputs={"o0": int},
        docstring=Docstring(docstring=docstring_text),
    )

    typed_interface = transform_native_to_typed_interface(interface)
    result = _update_interface_inputs_and_outputs_docstring(typed_interface, interface)

    # Only x should have description
    result_inputs = {entry.key: entry.value for entry in result.inputs.variables}
    assert result_inputs["x"].description == "The input value"
    assert result_inputs["y"].description == ""


def test_update_interface_mismatched_names():
    docstring_text = """
    A test function.

    Args:
        name: The user's name
        age: The user's age
    """

    # Interface has different parameter names
    interface = NativeInterface(
        inputs={"x": (int, inspect.Parameter.empty), "y": (str, inspect.Parameter.empty)},
        outputs={"o0": int},
        docstring=Docstring(docstring=docstring_text),
    )

    typed_interface = transform_native_to_typed_interface(interface)
    result = _update_interface_inputs_and_outputs_docstring(typed_interface, interface)

    # Descriptions should not be set (names don't match)
    result_inputs = {entry.key: entry.value for entry in result.inputs.variables}
    assert result_inputs["x"].description == ""
    assert result_inputs["y"].description == ""


# ---------------------------------------------------------------------------
# _build_image_bg and _build_images tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_image_bg_captures_remote_run_url():
    """Build URL is extracted from remote_run.url when using the remote builder."""
    image = flyte.Image.from_base("python:3.10")
    mock_run = Mock()
    mock_run.url = "https://console.union.ai/runs/abc123"
    mock_result = ImageBuild(uri="registry/my-image:sha256abc", remote_run=mock_run)

    with patch("flyte._build.build") as mock_build:
        mock_build.aio = AsyncMock(return_value=mock_result)
        env_name, uri, build_url = await _build_image_bg("my-env", image)

    assert env_name == "my-env"
    assert uri == "registry/my-image:sha256abc"
    assert build_url == "https://console.union.ai/runs/abc123"


@pytest.mark.asyncio
async def test_build_image_bg_no_url_for_local_build():
    """Build URL is None when using the local builder (remote_run is None)."""
    image = flyte.Image.from_base("python:3.10")
    mock_result = ImageBuild(uri="registry/my-image:sha256abc", remote_run=None)

    with patch("flyte._build.build") as mock_build:
        mock_build.aio = AsyncMock(return_value=mock_result)
        _env_name, _uri, build_url = await _build_image_bg("my-env", image)

    assert build_url is None


@pytest.mark.asyncio
async def test_build_images_stores_build_run_urls_in_cache():
    """build_run_urls in ImageCache is populated when remote builder provides a URL."""
    image = flyte.Image.from_base("python:3.10")
    env = flyte.TaskEnvironment(name="my-env", image=image)
    plan = DeploymentPlan(envs={"my-env": env})

    mock_run = Mock()
    mock_run.url = "https://console.union.ai/runs/abc123"
    mock_result = ImageBuild(uri="registry/my-image:sha256abc", remote_run=mock_run)

    with patch("flyte._build.build") as mock_build:
        mock_build.aio = AsyncMock(return_value=mock_result)
        cache: ImageCache = await _build_images(plan)

    assert cache.image_lookup["my-env"] == "registry/my-image:sha256abc"
    assert cache.build_run_urls["my-env"] == "https://console.union.ai/runs/abc123"


@pytest.mark.asyncio
async def test_build_images_no_build_run_urls_for_local_build():
    """build_run_urls in ImageCache is empty when local builder is used."""
    image = flyte.Image.from_base("python:3.10")
    env = flyte.TaskEnvironment(name="my-env", image=image)
    plan = DeploymentPlan(envs={"my-env": env})

    mock_result = ImageBuild(uri="registry/my-image:sha256abc", remote_run=None)

    with patch("flyte._build.build") as mock_build:
        mock_build.aio = AsyncMock(return_value=mock_result)
        cache: ImageCache = await _build_images(plan)

    assert cache.image_lookup["my-env"] == "registry/my-image:sha256abc"
    assert cache.build_run_urls == {}
