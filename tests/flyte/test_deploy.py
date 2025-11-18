from __future__ import annotations

from unittest.mock import Mock

import flyte
from flyte._deploy import _get_documentation_entity
from flyte._docstring import Docstring


def test_get_description_entity_truncates_short_description():
    env = flyte.TaskEnvironment(name="test_env", image="python:3.10")

    # Create a description exceeds 255 characters limit
    long_short_desc = "a" * 300

    @env.task()
    async def task_with_long_short_desc(x: int) -> int:
        return x * 2

    # Create a mock docstring with long short description
    mock_parsed_docstring = Mock()
    mock_parsed_docstring.short_description = long_short_desc
    mock_parsed_docstring.long_description = None

    # Create Docstring object and set the parsed docstring
    docstring = Docstring()
    docstring._parsed_docstring = mock_parsed_docstring
    task_with_long_short_desc.interface.docstring = docstring

    result = _get_documentation_entity(task_with_long_short_desc)

    assert result.short_description is not None
    assert len(result.short_description) == 255


def test_get_description_entity_truncates_long_description():
    env = flyte.TaskEnvironment(name="test_env", image="python:3.10")

    # Create a long description that exceeds 2048 characters limit
    long_long_desc = "b" * 3000

    @env.task()
    async def task_with_long_long_desc(x: int) -> int:
        return x * 2

    # Create a mock docstring with long long description
    mock_parsed_docstring = Mock()
    mock_parsed_docstring.short_description = None
    mock_parsed_docstring.long_description = long_long_desc

    docstring = Docstring()
    docstring._parsed_docstring = mock_parsed_docstring
    task_with_long_long_desc.interface.docstring = docstring

    result = _get_documentation_entity(task_with_long_long_desc)

    assert result.long_description is not None
    assert len(result.long_description) == 2048
    assert result.long_description == "b" * 2048


def test_get_description_entity_both_truncated():
    env = flyte.TaskEnvironment(name="test_env", image="python:3.10")

    # Create descriptions that exceed both limits
    short_desc = "c" * 300
    long_desc = "d" * 3000

    @env.task()
    async def task_both_exceed(x: int) -> int:
        return x * 2

    # Create a mock docstring with both descriptions exceeding limits
    mock_parsed_docstring = Mock()
    mock_parsed_docstring.short_description = short_desc
    mock_parsed_docstring.long_description = long_desc

    docstring = Docstring()
    docstring._parsed_docstring = mock_parsed_docstring
    task_both_exceed.interface.docstring = docstring

    result = _get_documentation_entity(task_both_exceed)

    # Verify truncation
    assert result.short_description == "c" * 255
    assert len(result.short_description) == 255
    assert result.long_description == "d" * 2048
    assert len(result.long_description) == 2048


def test_get_description_entity_within_limits():
    env = flyte.TaskEnvironment(name="test_env", image="python:3.10")

    # Create descriptions within limits
    short_desc = "e" * 200
    long_desc = "f" * 1500

    @env.task()
    async def task_within_limits(x: int) -> int:
        return x * 2

    # Manually set descriptions within limits
    task_within_limits.interface.docstring._parsed_docstring.short_description = short_desc
    task_within_limits.interface.docstring._parsed_docstring.long_description = long_desc

    result = _get_documentation_entity(task_within_limits)

    # Verify no truncation occurred
    assert result.short_description == short_desc
    assert len(result.short_description) == 200
    assert result.long_description == long_desc
    assert len(result.long_description) == 1500


def test_get_description_entity_exact_limits():
    env = flyte.TaskEnvironment(name="test_env", image="python:3.10")

    # Create descriptions at exact limits
    short_desc = "g" * 255
    long_desc = "h" * 2048

    @env.task()
    async def task_at_limits(x: int) -> int:
        return x * 2

    # Manually set descriptions at exact limits
    task_at_limits.interface.docstring._parsed_docstring.short_description = short_desc
    task_at_limits.interface.docstring._parsed_docstring.long_description = long_desc

    result = _get_documentation_entity(task_at_limits)

    # Verify exact limits are preserved
    assert result.short_description == short_desc
    assert len(result.short_description) == 255
    assert result.long_description == long_desc
    assert len(result.long_description) == 2048


def test_get_description_entity_none_values():
    env = flyte.TaskEnvironment(name="test_env", image="python:3.10")

    @env.task()
    async def task_no_docstring(x: int) -> int:
        return x * 2

    # Set docstring to None
    task_no_docstring.interface.docstring._parsed_docstring.short_description = None
    task_no_docstring.interface.docstring._parsed_docstring.long_description = None

    result = _get_documentation_entity(task_no_docstring)

    # Verify None values are handled correctly
    assert result.short_description is None
    assert result.long_description is None
