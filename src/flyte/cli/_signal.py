import rich_click as click

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
@click.argument("value", type=str, required=True)
@click.pass_obj
def event(
    cfg: common.CLIConfig,
    run_name: str,
    action_name: str,
    value: str,
    project: str | None = None,
    domain: str | None = None,
):
    """
    Signal a paused condition action with VALUE.

    The condition's declared payload type is read from the backend, so VALUE
    is coerced accordingly: ``true``/``false`` for bool, integer literals for
    int, decimal literals for float, and any string for str.
    """
    cfg.init(project=project, domain=domain)

    action = remote.Action.get(run_name=run_name, name=action_name)
    ev = remote.Event(pb2=action.pb2)
    expected = ev.expected_type
    if expected is None:
        raise click.ClickException(
            f"Action '{action_name}' in run '{run_name}' is not a signalable condition, "
            f"or the backend does not expose its expected payload type."
        )

    payload = _coerce(value, expected)

    console = common.get_console()
    with console.status(
        f"Signaling event on action '{action_name}' (run '{run_name}') with {payload!r}...",
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
