from __future__ import annotations

import rich_click as click
from rich.markdown import Markdown

import flyte.remote as remote
from flyte.cli import _common as common


@click.group(name="signal")
def signal():
    """
    Signal an event waiting on a paused condition action.
    """


@signal.command(cls=common.CommandBase)
@click.argument("run-name", type=str, required=True)
@click.argument("action-name", type=str, required=True)
@click.argument("value", type=str, required=False, default=None)
@click.pass_obj
def event(
    cfg: common.CLIConfig,
    run_name: str,
    action_name: str,
    value: str | None = None,
    project: str | None = None,
    domain: str | None = None,
):
    """
    Signal a paused condition action.

    The condition's declared payload type and prompt are read from the
    backend. If VALUE is omitted the condition's prompt is displayed and a
    typed interactive prompt is shown to collect the payload. When VALUE is
    provided it's coerced to the expected type (``true``/``false`` for bool,
    integer literals for int, decimal literals for float, any string for str).
    """
    cfg.init(project=project, domain=domain)

    action = remote.Action.get(run_name=run_name, name=action_name)
    details = action._details
    if details is None or not details.pb2.HasField("condition"):
        raise click.ClickException(f"Action '{action_name}' in run '{run_name}' is not a signalable condition.")
    cond = details.pb2.condition

    ev = remote.Event(pb2=action.pb2)
    expected = ev.expected_type
    if expected is None:
        raise click.ClickException(f"Backend did not expose expected payload type for condition '{action_name}'.")

    console = common.get_console()
    _display_prompt(console, cond.prompt, cond.prompt_type)

    if value is None:
        payload = _prompt_for_value(expected)
    else:
        payload = _coerce(value, expected)
        console.print(f"signaling with: {payload!r}")

    with console.status(
        f"Signaling event on action '{action_name}' (run '{run_name}')...",
        spinner=common.safe_spinner("dots"),
    ):
        ev.signal(payload)
    console.print(f"Event on action '{action_name}' has been signaled.")


_BOOL_TRUE = {"true", "1", "yes", "y", "t"}
_BOOL_FALSE = {"false", "0", "no", "n", "f"}


def _coerce(value: str, expected: type) -> bool | int | float | str:
    if expected is bool:
        v = value.strip().lower()
        if v in _BOOL_TRUE:
            return True
        if v in _BOOL_FALSE:
            return False
        raise click.BadParameter(
            f"could not parse {value!r} as bool (expected one of {sorted(_BOOL_TRUE | _BOOL_FALSE)})",
            param_hint="VALUE",
        )
    if expected is int:
        try:
            return int(value)
        except ValueError as e:
            raise click.BadParameter(f"could not parse {value!r} as int: {e}", param_hint="VALUE") from e
    if expected is float:
        try:
            return float(value)
        except ValueError as e:
            raise click.BadParameter(f"could not parse {value!r} as float: {e}", param_hint="VALUE") from e
    if expected is str:
        return value
    raise click.ClickException(f"unsupported expected type {expected!r}")


def _display_prompt(console, prompt: str, prompt_type: int) -> None:
    """Render the condition's prompt before asking for input.

    Falls back to plain text when prompt_type is unset or unknown.
    """
    if not prompt:
        return
    from flyteidl2.workflow import run_definition_pb2

    if prompt_type == run_definition_pb2.CONDITION_PROMPT_TYPE_MARKDOWN:
        console.print(Markdown(prompt))
    else:
        console.print(prompt)


def _prompt_for_value(expected: type) -> bool | int | float | str:
    """Interactive prompt for a payload, typed by the condition's declared type."""
    if expected is bool:
        return click.confirm("Approve?", default=True)
    if expected is int:
        return click.prompt("Value", type=int)
    if expected is float:
        return click.prompt("Value", type=float)
    if expected is str:
        return click.prompt("Value", type=str)
    raise click.ClickException(f"unsupported expected type {expected!r}")
