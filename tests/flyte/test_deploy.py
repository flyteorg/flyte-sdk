from __future__ import annotations

import inspect
import sys
import types
from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

import flyte
from flyte._deploy import (
    _check_duplicate_env,
    _get_documentation_entity,
    _recursive_discover,
    _update_interface_inputs_and_outputs_docstring,
    plan_deploy,
)
from flyte._docstring import Docstring
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
# _check_duplicate_env / plan_deploy — duplicate detection tests
# ---------------------------------------------------------------------------


def _make_module(name: str, file: str, env: flyte.TaskEnvironment) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = file
    mod.env = env
    return mod


@pytest.fixture()
def dual_import_envs():
    """Two distinct env objects with the same name, each registered to a module
    that points at the same physical file (the classic src/ dual-import scenario)."""
    env1 = flyte.TaskEnvironment(name="my_env", image="python:3.10")
    env2 = flyte.TaskEnvironment(name="my_env", image="python:3.10")
    mod1 = _make_module("my_module.envs", "/project/src/my_module/envs.py", env1)
    mod2 = _make_module("src.my_module.envs", "/project/src/my_module/envs.py", env2)
    modules = {"my_module.envs": mod1, "src.my_module.envs": mod2}
    return env1, env2, modules


def test_check_duplicate_env_dual_import(dual_import_envs):
    """Same physical file imported under two module names → dual-import hint."""
    env1, env2, modules = dual_import_envs
    with (
        patch.dict(sys.modules, modules),
        patch("flyte._deploy.os.path.samefile", return_value=True),
        pytest.raises(ValueError, match="imported twice under different module names"),
    ):
        _check_duplicate_env(env1, env2)


def test_check_duplicate_env_true_duplicate():
    """Two envs with the same name from genuinely different files → plain duplicate error."""
    env1 = flyte.TaskEnvironment(name="my_env", image="python:3.10")
    env2 = flyte.TaskEnvironment(name="my_env", image="python:3.10")
    mod1 = _make_module("module_a.envs", "/project/module_a/envs.py", env1)
    mod2 = _make_module("module_b.envs", "/project/module_b/envs.py", env2)
    with (
        patch.dict(sys.modules, {"module_a.envs": mod1, "module_b.envs": mod2}),
        patch("flyte._deploy.os.path.samefile", return_value=False),
        pytest.raises(ValueError, match="Duplicate environment name 'my_env'"),
    ):
        _check_duplicate_env(env1, env2)


def test_plan_deploy_dual_import_raises(dual_import_envs):
    """plan_deploy surfaces the dual-import error when the same env name appears twice."""
    env1, env2, modules = dual_import_envs
    with (
        patch.dict(sys.modules, modules),
        patch("flyte._deploy.os.path.samefile", return_value=True),
        pytest.raises(ValueError, match="imported twice under different module names"),
    ):
        plan_deploy(env1, env2)


def test_recursive_discover_dual_import_raises(dual_import_envs):
    """_recursive_discover surfaces the dual-import error via the identity guard."""
    env1, env2, modules = dual_import_envs
    with (
        patch.dict(sys.modules, modules),
        patch("flyte._deploy.os.path.samefile", return_value=True),
        pytest.raises(ValueError, match="imported twice under different module names"),
    ):
        _recursive_discover({"my_env": env1}, env2)
