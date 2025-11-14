from __future__ import annotations

import flyte
from flyte._deploy import _get_documentation_entity


def test_get_description_entity_truncates_short_description():
    env = flyte.TaskEnvironment(name="test_env", image="python:3.10")

    # Create a description exceeds 255 characters limit
    long_short_desc = "a" * 300

    @env.task()
    async def task_with_long_short_desc(x: int) -> int:
        return x * 2

    # Manually set a long short description for testing
    task_with_long_short_desc.interface.docstring._parsed_docstring.short_description = long_short_desc

    result = _get_documentation_entity(task_with_long_short_desc)

    assert result.short_description is not None
    assert len(result.short_description) == 255


def test_get_description_entity_exact_limits():
    env = flyte.TaskEnvironment(name="test_env", image="python:3.10")

    # Create descriptions at exact limits
    short_desc = "c" * 300
    long_desc = "d" * 3000

    @env.task()
    async def task_at_limits(x: int) -> int:
        return x * 2

    # Manually set descriptions at exact limits
    task_at_limits.interface.docstring._parsed_docstring.short_description = short_desc
    task_at_limits.interface.docstring._parsed_docstring.long_description = long_desc

    result = _get_documentation_entity(task_at_limits)

    assert result.short_description == short_desc
    assert len(result.short_description) == 255
    assert result.long_description == long_desc
    assert len(result.long_description) == 2048
