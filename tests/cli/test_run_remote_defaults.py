"""
Reproduction tests for the `_has_default` regression on `flyte run deployed-task`.

When a deployed task has default input values, the SDK reconstructs the remote
`NativeInterface` using `NativeInterface.has_default` (the `_has_default` class)
as a sentinel marker — with the actual literal default value stored in
`NativeInterface._remote_defaults`.

The CLI's `get_params` used to pass this sentinel *class* straight through to
`click.Option(default=...)`. Click treats callable defaults as factories, so it
would instantiate `_has_default()` and convert the resulting instance with
`STRING.convert`, producing a literal string like
``"<flyte.models._has_default object at 0x...>"``. That garbage string then
- showed up in ``--help`` as the rendered default, and
- was shipped over the wire as the actual input value, causing the deployed
  pod to fail with confusing errors like ``FontNotFound: <flyte.models._has_default object at 0x...>``.

These tests pin both the CLI rendering and the runtime materialization paths so
the regression cannot return.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import flyte.remote
from flyte.cli._params import to_click_option
from flyte.cli._run import RunArguments, RunRemoteTaskCommand, run
from flyte.models import NativeInterface
from flyte.types import TypeEngine

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


def _make_remote_command(task_name: str = "fake.task") -> RunRemoteTaskCommand:
    return RunRemoteTaskCommand(
        task_name=task_name,
        run_args=RunArguments(project="p", domain="d"),
        version=None,
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
