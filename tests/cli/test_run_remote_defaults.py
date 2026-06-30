"""
CLI tests for deployed-task input defaults and partial ``--inputs`` JSON.

Covers:

- ``_has_default`` sentinel handling (remote defaults must not leak into click)
- Pydantic / dataclass models with field-level defaults on ``--inputs``
- Shallow merge of partial JSON with the task-arg default before validation
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pydantic
import pytest
from click.testing import CliRunner
from pydantic import Field

import flyte.remote
from flyte.cli._params import JsonParamType, to_click_option
from flyte.cli._run import RunArguments, RunRemoteTaskCommand, run
from flyte.models import NativeInterface
from flyte.types import TypeEngine
from flyte.types._type_engine import convert_mashumaro_json_schema_to_python_class

# ---------------------------------------------------------------------------
# Helpers to build a fake remote `TaskDetails`-like object with real defaults
# ---------------------------------------------------------------------------


def _build_remote_interface(
    defaults: dict[str, tuple[type, Any]],
    required: dict[str, type] | None = None,
) -> NativeInterface:
    """
    Build a `NativeInterface` mimicking what `guess_interface` produces for a
    deployed task: defaults are encoded as the `_has_default` sentinel with the
    actual literal values living in `_remote_defaults`.
    """
    required = required or {}

    async def _build():
        remote_defaults = {}
        for name, (py_type, value) in defaults.items():
            remote_defaults[name] = await TypeEngine.to_literal(value, py_type, TypeEngine.to_literal_type(py_type))

        inputs: dict[str, tuple[type, Any]] = {}
        for name, py_type in required.items():
            inputs[name] = (py_type, inspect.Parameter.empty)
        for name, (py_type, _value) in defaults.items():
            inputs[name] = (py_type, NativeInterface.has_default)

        return NativeInterface.from_types(inputs, {}, remote_defaults)

    return asyncio.run(_build())


class _FakeTaskDetails:
    """Stand-in for `flyte.remote.TaskDetails` with a real `NativeInterface`."""

    def __init__(self, interface: NativeInterface):
        self._interface = interface

    @property
    def interface(self) -> NativeInterface:
        return self._interface


class _FakeLazyEntity:
    """Stand-in for `flyte.remote.LazyEntity`; returns our fake details."""

    def __init__(self, details: _FakeTaskDetails):
        self._details = details

    def fetch(self) -> _FakeTaskDetails:
        return self._details


def _patched_task_get(details: _FakeTaskDetails):
    return patch.object(flyte.remote.Task, "get", return_value=_FakeLazyEntity(details))


# ---------------------------------------------------------------------------
# Tests targeting `to_click_option`: this is the lowest-level guard. The CLI
# must never end up handing the `_has_default` sentinel class to click as a
# default value, because click will instantiate it.
# ---------------------------------------------------------------------------


def test_to_click_option_rejects_has_default_sentinel_class():
    """
    Regression: `to_click_option` must not propagate `NativeInterface.has_default`
    (a callable class) into `click.Option(default=...)`. If it does, click
    silently calls `_has_default()` and string-formats the resulting *instance*.

    The option's default must therefore be either ``None`` or a properly
    resolved Python value — never the sentinel class itself.
    """
    from flyteidl2.core.interface_pb2 import Variable
    from flyteidl2.core.types_pb2 import LiteralType, SimpleType

    literal_var = Variable(type=LiteralType(simple=SimpleType.STRING), description="font")

    option = to_click_option(
        input_name="font",
        literal_var=literal_var,
        python_type=str,
        default_val=NativeInterface.has_default,
    )

    # The sentinel must not leak through as the click default — neither as the
    # class itself nor as a stringified instance/class of `_has_default`.
    assert option.default is not NativeInterface.has_default
    rendered_default = repr(option.default)
    assert "_has_default" not in rendered_default, (
        f"`_has_default` sentinel leaked into click default: {rendered_default!r}"
    )


# ---------------------------------------------------------------------------
# Tests targeting `RunRemoteTaskCommand.get_params`: ensures the synthesized
# click options use the *real* remote default values from `_remote_defaults`.
# ---------------------------------------------------------------------------


def _make_remote_command(task_name: str = "fake.task", version: str | None = None) -> RunRemoteTaskCommand:
    return RunRemoteTaskCommand(
        task_name=task_name,
        run_args=RunArguments(project="p", domain="d"),
        version=version,
        name=task_name,
    )


@pytest.mark.parametrize(
    "py_type,default_value",
    [
        (str, "head"),
        (str, "hello, flyte"),
        (int, 42),
        (bool, True),
    ],
)
def test_get_params_resolves_primitive_remote_defaults(py_type, default_value):
    """
    For a deployed task with primitive defaults, `RunRemoteTaskCommand.get_params`
    must resolve the literal stored in `_remote_defaults` into a concrete Python
    value and hand that to click as the option default. The result must never be
    the `_has_default` sentinel.
    """
    interface = _build_remote_interface(defaults={"x": (py_type, default_value)})
    details = _FakeTaskDetails(interface)
    cmd = _make_remote_command()

    with _patched_task_get(details), patch("flyte.cli._common.initialize_config"):
        import click as _click

        ctx = _click.Context(cmd)
        params = cmd.get_params(ctx)

    by_name = {p.name: p for p in params if isinstance(p, _click.Option)}
    assert "x" in by_name, f"Expected --x option in {list(by_name)}"
    opt = by_name["x"]
    assert opt.default == default_value, f"--x default should resolve to {default_value!r}, got {opt.default!r}"
    assert "_has_default" not in repr(opt.default)


def test_get_params_honors_pinned_version_during_input_discovery():
    """
    Regression (A3): ``RunRemoteTaskCommand.get_params`` must build the input form from the *pinned*
    version, exactly like the execution path. Dropping the version here resolves ``latest`` instead,
    which builds options from the wrong interface and -- if latest's output type can't be
    reconstructed -- crashes input discovery before a run of the pinned version even starts.
    """
    interface = _build_remote_interface(defaults={"x": (str, "head")})
    details = _FakeTaskDetails(interface)
    cmd = _make_remote_command(version="v-pinned-123")

    with _patched_task_get(details) as mock_get, patch("flyte.cli._common.initialize_config"):
        import click as _click

        cmd.get_params(_click.Context(cmd))

    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs.get("version") == "v-pinned-123", (
        f"get_params must pass the pinned version to Task.get, got {mock_get.call_args!r}"
    )


def test_help_output_shows_real_default_not_has_default_class():
    """
    End-to-end check: `flyte run deployed-task <name> --help` must render the
    actual default value (e.g. ``standard``), never
    ``<class 'flyte.models._has_default'>`` or ``<flyte.models._has_default object ...>``.
    """
    interface = _build_remote_interface(
        defaults={
            "message": (str, "hello, flyte"),
            "font": (str, "standard"),
        }
    )
    details = _FakeTaskDetails(interface)

    runner = CliRunner()
    with _patched_task_get(details), patch("flyte.cli._common.initialize_config"):
        result = runner.invoke(
            run,
            ["deployed-task", "hello_flyte.hello_flyte_task", "--help"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert "_has_default" not in result.output, f"`_has_default` sentinel leaked into --help output:\n{result.output}"
    # Sanity: the actual resolved defaults should be visible somewhere in --help.
    assert "standard" in result.output
    assert "hello, flyte" in result.output


# ---------------------------------------------------------------------------
# Defense-in-depth: even if a caller manages to pass the `_has_default`
# sentinel (class *or* instance) through to `convert_from_native_to_inputs`,
# the runtime must fall back to the `_remote_defaults` literal instead of
# trying to serialize the sentinel as if it were a real user value.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sentinel",
    [
        NativeInterface.has_default,  # the class itself
        NativeInterface.has_default(),  # an instance, as click would produce
    ],
    ids=["class", "instance"],
)
async def test_convert_from_native_to_inputs_handles_sentinel_in_kwargs(sentinel):
    """
    If the `_has_default` sentinel (class or instance) shows up as a value in
    `kwargs`, `convert_from_native_to_inputs` must transparently substitute the
    matching literal from `interface._remote_defaults` rather than attempting
    to serialize the sentinel itself.
    """
    from flyte._internal.runtime.convert import convert_from_native_to_inputs

    default_literal = await TypeEngine.to_literal("standard", str, TypeEngine.to_literal_type(str))
    interface = NativeInterface.from_types(
        {"font": (str, NativeInterface.has_default)},
        {},
        {"font": default_literal},
    )

    result = await convert_from_native_to_inputs(interface, font=sentinel)
    literals = {lit.name: lit.value for lit in result.proto_inputs.literals}
    assert "font" in literals
    assert literals["font"].scalar.primitive.string_value == "standard"


class HelloFlyteInputManifest(pydantic.BaseModel):
    message: str = Field(default="hello, flyte")
    font: str = Field(default="standard")


def test_guessed_pydantic_type_accepts_partial_json():
    """
    Remote tasks guess Pydantic inputs as dynamic dataclasses from JSON schema metadata.
    Field-level defaults must be applied so partial ``--inputs`` JSON validates.
    """
    lt = TypeEngine.to_literal_type(HelloFlyteInputManifest)
    guessed = TypeEngine.guess_python_type(lt)
    param_type = JsonParamType(guessed, default_value=HelloFlyteInputManifest())
    result = param_type.convert(json.dumps({"message": "hello, niels"}), None, None)
    assert result.message == "hello, niels"
    assert result.font == "standard"


def test_json_param_type_merges_partial_with_task_default():
    """Partial object CLI values merge with the task-arg default before validation."""
    param_type = JsonParamType(HelloFlyteInputManifest, default_value=HelloFlyteInputManifest())
    result = param_type.convert(
        json.dumps({"message": "hello, niels"}),
        None,
        None,
    )
    assert result.message == "hello, niels"
    assert result.font == "standard"


def test_convert_mashumaro_schema_honors_pydantic_field_defaults():
    schema = HelloFlyteInputManifest.model_json_schema()
    cls = convert_mashumaro_json_schema_to_python_class(schema, "HelloFlyteInputManifest")
    from mashumaro.codecs.json import JSONDecoder

    decoded = JSONDecoder(cls).decode(json.dumps({"message": "hello, niels"}))
    assert decoded.message == "hello, niels"
    assert decoded.font == "standard"


@dataclass
class DataclassInput:
    message: str = "hello"
    font: str = "standard"


def test_guessed_dataclass_type_accepts_partial_json():
    """Remote dataclass inputs must honor JSON-schema field defaults on partial ``--inputs``."""
    lt = TypeEngine.to_literal_type(DataclassInput)
    guessed = TypeEngine.guess_python_type(lt)
    param_type = JsonParamType(guessed, default_value=DataclassInput())
    result = param_type.convert(json.dumps({"message": "hi"}), None, None)
    assert result.message == "hi"
    assert result.font == "standard"


def test_json_param_type_merges_partial_with_dataclass_task_default():
    param_type = JsonParamType(DataclassInput, default_value=DataclassInput())
    result = param_type.convert(json.dumps({"message": "hi"}), None, None)
    assert result.message == "hi"
    assert result.font == "standard"


# ---------------------------------------------------------------------------
# Deployed-task path: Pydantic ``inputs`` model with field defaults (customer repro)
# ---------------------------------------------------------------------------


def test_get_params_resolves_pydantic_model_remote_default():
    """``--inputs`` default for a remote Pydantic model is materialized as JSON for click."""
    default_model = HelloFlyteInputManifest()
    interface = _build_remote_interface(defaults={"inputs": (HelloFlyteInputManifest, default_model)})
    details = _FakeTaskDetails(interface)
    cmd = _make_remote_command("deploy_task_test.hello_flyte_task")

    with _patched_task_get(details), patch("flyte.cli._common.initialize_config"):
        import click as _click

        ctx = _click.Context(cmd)
        params = cmd.get_params(ctx)

    by_name = {p.name: p for p in params if isinstance(p, _click.Option)}
    assert "inputs" in by_name
    opt = by_name["inputs"]
    assert isinstance(opt.default, str)
    assert "hello, flyte" in opt.default
    assert "standard" in opt.default
    assert "_has_default" not in opt.default


def test_help_output_shows_pydantic_inputs_json_default():
    """``--help`` for a Pydantic ``inputs`` param shows the resolved model default JSON."""
    interface = _build_remote_interface(
        defaults={"inputs": (HelloFlyteInputManifest, HelloFlyteInputManifest())},
    )
    details = _FakeTaskDetails(interface)
    runner = CliRunner()

    with _patched_task_get(details), patch("flyte.cli._common.initialize_config"):
        result = runner.invoke(
            run,
            ["deployed-task", "deploy_task_test.hello_flyte_task", "--help"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert "_has_default" not in result.output
    # Rich help may wrap the default JSON across lines; check key parts separately.
    assert "hello" in result.output
    assert "flyte" in result.output
    assert "standard" in result.output
    assert '"font":"standard"' in result.output or '"font": "standard"' in result.output


def test_remote_partial_pydantic_inputs_converts_via_click_param_type():
    """
    End-to-end CLI param conversion for deployed tasks: partial ``--inputs`` JSON
    must validate after merging with the remote task default.
    """
    interface = _build_remote_interface(
        defaults={"inputs": (HelloFlyteInputManifest, HelloFlyteInputManifest())},
    )
    details = _FakeTaskDetails(interface)
    cmd = _make_remote_command()

    with _patched_task_get(details), patch("flyte.cli._common.initialize_config"):
        import click as _click

        ctx = _click.Context(cmd)
        params = cmd.get_params(ctx)

    by_name = {p.name: p for p in params if isinstance(p, _click.Option)}
    inputs_opt = by_name["inputs"]
    # ``to_click_option`` stores the default as model_dump_json for STRUCT + pydantic
    assert isinstance(inputs_opt.default, str)
    result = inputs_opt.type.convert(json.dumps({"message": "hello, niels"}), inputs_opt, ctx)
    assert result.message == "hello, niels"
    assert result.font == "standard"


def test_json_param_type_merges_partial_with_json_string_task_default():
    """STRUCT click defaults are JSON strings; merge must still apply field overrides."""
    default_json = HelloFlyteInputManifest().model_dump_json()
    param_type = JsonParamType(HelloFlyteInputManifest, default_value=default_json)
    result = param_type.convert(json.dumps({"message": "override"}), None, None)
    assert result.message == "override"
    assert result.font == "standard"


def test_json_param_type_default_dict_strips_schema_without_field_defaults():
    """
    Stripped schemas (all properties required, no per-field defaults) still work when
    the task-arg default is merged before validation.
    """
    stripped_schema = {
        "type": "object",
        "title": "HelloFlyteInputManifest",
        "properties": {
            "message": {"type": "string"},
            "font": {"type": "string"},
        },
        "required": ["message", "font"],
        "additionalProperties": False,
    }
    cls = convert_mashumaro_json_schema_to_python_class(stripped_schema, "HelloFlyteInputManifest")
    param_type = JsonParamType(cls, default_value=HelloFlyteInputManifest())
    result = param_type.convert(json.dumps({"message": "only message"}), None, None)
    assert result.message == "only message"
    assert result.font == "standard"
