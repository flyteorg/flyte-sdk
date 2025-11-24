from __future__ import annotations

from unittest.mock import Mock

import flyte
from flyte._deploy import _get_documentation_entity
from flyte._docstring import Docstring


def test_get_description_entity_both_truncated():
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
    task_both_exceed.interface.docstring = docstring

    result = _get_documentation_entity(task_both_exceed)

    # Verify truncation
    assert env.description == "a" * 255
    assert len(env.description) == 255
    assert result.short_description == "c" * 255
    assert len(result.short_description) == 255
    assert result.long_description == "d" * 2048
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
    assert env.description is None
    assert result.short_description is None
    assert result.long_description is None
