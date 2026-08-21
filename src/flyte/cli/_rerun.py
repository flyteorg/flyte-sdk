"""`flyte rerun <run>` — re-run an existing run with its own code + exact inputs.

Counterpart to `flyte run`, which launches *local* code: `rerun` re-launches an *existing* run,
fetching its task + inputs from the platform, so no local code is needed.

* `flyte rerun <run>` creates a whole new run with the same inputs and re-executes the whole
  workflow again, subject to global caching.
* `flyte rerun <run> --recover` creates a whole new run with the same inputs but reuses the prior
  run's succeeded actions, re-executing only what failed or never ran.
* `flyte rerun <run> --action-name <action>` re-runs just that one action from the run: the new
  run is rooted at that action's task, executed with the exact inputs it received. Mutually
  exclusive with `--recover`.
* `flyte rerun <run> --x 2` changes input parameters. The task's inputs become options on this
  command exactly as they do for `flyte run` — same literal-type-to-click-type conversion — except
  that the interface is read off the source run instead of local code, and every option is
  optional: an input left out keeps the prior run's value. This composes with `--recover`.

Rerun always replays the source run's *code* as-is — substituting local code is `flyte fork`,
reserved for flyteplugins-union.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rich_click as click
from click.core import ParameterSource

from . import _common as common

# Keyword arguments of `rerun()` itself. A task input by one of these names cannot be forwarded as
# a keyword, so it never becomes an option here; it stays reachable from the Python API only via a
# run whose interface does not collide. rerun's own option names are excluded separately, straight
# off the declared parameters, so the two lists cannot drift apart.
_RESERVED_INPUT_NAMES = frozenset({"run_name", "action_name", "recover", "force_rerun_actions"})


def _parse_kv(items: Tuple[str, ...], flag: str) -> Optional[Dict[str, str]]:
    """Parse repeated `KEY=VALUE` flag values into a dict (None if none given)."""
    if not items:
        return None
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise click.BadParameter(f"Invalid {flag} value {item!r}: expected KEY=VALUE.")
        key, value = item.split("=", 1)
        if not key:
            raise click.BadParameter(f"Invalid {flag} value {item!r}: key must not be empty.")
        parsed[key] = value
    return parsed


async def _fetch_source_interface(run_name: str, action_name: str):
    """The typed interface of the action that supplies the task, or None if it carries no task.

    Rerun never uses local code, so the interface the input options are built from lives on the
    platform: it is read off the same action `rerun()` sources the task spec from.
    """
    from flyte.remote._action import ActionDetails
    from flyte.remote._run import RunDetails

    if action_name == "a0":
        action_details = (await RunDetails.get.aio(name=run_name)).action_details
    else:
        action_details = await ActionDetails.get.aio(run_name=run_name, name=action_name)
    if not action_details.pb2.HasField("task"):
        return None
    return action_details.pb2.task.task_template.interface


def _to_optional_option(name: str, var, python_type) -> click.Option:
    """`flyte run`'s option for this input, made optional.

    `flyte run` marks an input required when the task declares no default, and shows that default
    in `--help`. Neither fits rerun: the fallback for an input the user leaves out is the source
    run's value, not the task's default and not an error.
    """
    from ._params import FlyteLiteralConverter, to_click_option

    if FlyteLiteralConverter(literal_type=var.type, python_type=python_type).is_bool():
        # to_click_option only offers `--x/--no-x` when the task's default is True. Here either
        # value has to be expressible, because "not passed" already means "keep the prior value".
        return click.Option(
            param_decls=[f"--{name}/--no-{name}"],
            default=None,
            required=False,
            help=var.description,
        )
    option = to_click_option(name, var, python_type, None)
    option.required = False
    option.default = None
    option.show_default = False
    return option


def _input_options(interface, taken_names: set[str], taken_opts: set[str]) -> List[click.Parameter]:
    """One option per input of the source task, skipping the ones a CLI cannot express."""
    from flyte.types._interface import guess_interface

    native_inputs = guess_interface(interface).inputs
    options: List[click.Parameter] = []
    for entry in interface.inputs.variables:
        name, var = entry.key, entry.value
        # An input that collides with one of rerun's own options (or with a keyword `rerun()`
        # already takes) cannot be an option here without shadowing it. Everything else about the
        # rerun still works; only that one input keeps the prior run's value.
        if name in taken_names or name in _RESERVED_INPUT_NAMES or f"--{name}" in taken_opts:
            continue
        if name not in native_inputs:
            continue
        python_type, _ = native_inputs[name]
        try:
            options.append(_to_optional_option(name, var, python_type))
        except Exception:
            # e.g. an uppercase input name, which click cannot express as an option, or a type
            # with no click equivalent. Same outcome: that input keeps the prior run's value.
            continue
    return options


def _peek_declared_args(args: Sequence[str], declared_params: List[click.Parameter]):
    """Read rerun's own arguments out of `args`, tolerating input options not yet built.

    Returns the parsed values plus whatever was left over. Nothing left over means no input was
    passed, which is what makes the extra round trip to the platform skippable.
    """
    shadow = click.Command(
        "rerun",
        params=list(declared_params),
        add_help_option=False,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )
    try:
        ctx = shadow.make_context("rerun", list(args), resilient_parsing=True)
    except Exception:
        return {}, []
    return ctx.params, list(ctx.args)


class RerunCommand(click.RichCommand):
    """`flyte rerun`, with an option per input of the source run's task.

    Those inputs come from the platform rather than from local code, so they can only be
    discovered once RUN_NAME has been read off the command line — hence the two-pass parse:
    rerun's own options first, then the interface, then the real parse with both sets of options.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._declared_params: List[click.Parameter] = list(self.params)
        self._input_params: List[click.Parameter] = []

    def get_params(self, ctx: click.Context) -> List[click.Parameter]:
        # Note this may be called several times by click (parsing, help, completion).
        self.params = [*self._declared_params, *self._input_params]
        return super().get_params(ctx)

    def parse_args(self, ctx: click.Context, args: List[str]) -> List[str]:
        self._input_params = self._build_input_params(ctx, args)
        return super().parse_args(ctx, args)

    def _build_input_params(self, ctx: click.Context, args: List[str]) -> List[click.Parameter]:
        declared, leftover = _peek_declared_args(args, self._declared_params)
        run_name = declared.get("run_name")
        # No leftover means every token was consumed by rerun's own options, so no input was
        # passed and there is nothing for the interface to interpret (`--help` counts as leftover,
        # so `flyte rerun <run> --help` does list the run's inputs).
        if not run_name or not leftover:
            return []
        try:
            common.initialize_config(ctx, project=declared.get("project"), domain=declared.get("domain"))
            interface = asyncio.run(_fetch_source_interface(run_name, declared.get("action_name") or "a0"))
        except Exception as e:
            # `--help` has to render even with no config or no connectivity; it just cannot list
            # the run's inputs in that case.
            if "--help" in args:
                return []
            raise click.UsageError(
                f"Could not read the inputs of {run_name} from the platform, so "
                f"{' '.join(leftover)} cannot be interpreted: {e}"
            ) from e
        if interface is None:
            return []
        taken_names = {p.name for p in self._declared_params if p.name}
        taken_opts = {o for p in self._declared_params for o in [*p.opts, *p.secondary_opts]}
        return _input_options(interface, taken_names, taken_opts)


@click.command("rerun", cls=RerunCommand)
@click.argument("run_name", required=True)
@click.option("-p", "--project", default=None, help="Project for the new run (defaults to config).")
@click.option("-d", "--domain", default=None, help="Domain for the new run (defaults to config).")
@click.option("--name", default=None, help="Name for the new run (a random name is generated if unset).")
@click.option("-e", "--env", "env", multiple=True, help="Env var KEY=VALUE for the new run. Repeatable.")
@click.option("--label", "label", multiple=True, help="Label KEY=VALUE for the new run. Repeatable.")
@click.option("--follow", "-f", is_flag=True, default=False, help="Stream the parent action logs after launch.")
@click.option(
    "--recover",
    is_flag=True,
    default=False,
    help="Reuse the prior run's succeeded actions, re-running only what failed or never ran. Remote-only.",
)
@click.option(
    "--action-name",
    "action_name",
    default=None,
    help="Re-run only this action from the run, instead of the whole run: the new run is rooted "
    "at that action's task with the inputs it received. Cannot be combined with --recover. "
    "List names with `flyte get action <run>`.",
)
@click.option(
    "--force-rerun-action",
    "force_rerun_action",
    multiple=True,
    help="With --recover: name of an action to re-execute even though it succeeded in the "
    "source run. Repeatable. A listed parent re-enqueues its children (list them too to "
    "force the whole subtree); unknown names are ignored.",
)
@click.option(
    "--allow-missing-outputs",
    "allow_missing_outputs",
    is_flag=True,
    default=False,
    help="Proceed when the source run's outputs were cleaned up from storage, using its inputs "
    "URI directly. The inputs cannot be verified from the client — if they were deleted too, "
    "the new run fails at runtime.",
)
@click.pass_context
def rerun(
    ctx: click.Context,
    run_name: str,
    project: Optional[str],
    domain: Optional[str],
    name: Optional[str],
    env: Tuple[str, ...],
    label: Tuple[str, ...],
    follow: bool,
    recover: bool,
    action_name: Optional[str],
    force_rerun_action: Tuple[str, ...],
    allow_missing_outputs: bool,
    **task_inputs: Any,
) -> None:
    """Re-run an existing run RUN_NAME with its original code and inputs.

    Fetches the prior run's task + inputs from the platform (no local code needed) and launches a
    new run that returns the same way `flyte run` does. `--recover` reuses the prior run's
    succeeded actions, re-running only what failed or never ran; `--force-rerun-action` forces
    named actions to re-execute anyway.

    `--action-name` narrows the whole thing to a single action: the new run is rooted at that
    action's task, run with the exact inputs it received inside RUN_NAME. That is always a plain
    re-execution, so it cannot be combined with `--recover`.

    The task's own inputs are options too, built from RUN_NAME's interface just as `flyte run`
    builds them from local code — run `flyte rerun RUN_NAME --help` to list them. Every input left
    out keeps RUN_NAME's value. With `--recover`, the new run starts from the changed inputs but
    still reuses RUN_NAME's succeeded actions, which keep the outputs they produced under the
    *original* inputs — force the ones that must re-execute with `--force-rerun-action`.

    Examples:

        $ flyte rerun rxyz
        $ flyte rerun rxyz --name retry-1 --follow
        $ flyte rerun rxyz --recover
        $ flyte rerun rxyz --recover --force-rerun-action a3 --force-rerun-action a7
        $ flyte rerun rxyz --action-name a3
        $ flyte rerun rxyz --help
        $ flyte rerun rxyz --n 10 --cfg '{"lr": 0.1}'
        $ flyte rerun rxyz --recover --n 10 --force-rerun-action a3
    """
    if force_rerun_action and not recover:
        raise click.UsageError("--force-rerun-action requires --recover")
    if action_name and recover:
        raise click.UsageError(
            "--action-name cannot be combined with --recover: recovery matches succeeded actions "
            "from the source run by name, and a run rooted at a single action has a different "
            "action tree. Re-run the action on its own, or recover the whole run."
        )
    # Only inputs actually typed on the command line are changes; the rest are absent on purpose,
    # so that they keep the source run's value rather than the task's default.
    new_inputs = {
        key: value for key, value in task_inputs.items() if ctx.get_parameter_source(key) is ParameterSource.COMMANDLINE
    }
    config = common.initialize_config(ctx, project=project, domain=domain)
    asyncio.run(
        _execute(
            run_name,
            name,
            env,
            label,
            follow,
            recover,
            action_name,
            force_rerun_action,
            new_inputs,
            allow_missing_outputs,
            config,
        )
    )


async def _execute(
    run_name: str,
    name: Optional[str],
    env: Tuple[str, ...],
    label: Tuple[str, ...],
    follow: bool,
    recover: bool,
    action_name: Optional[str],
    force_rerun_action: Tuple[str, ...],
    new_inputs: Dict[str, Any],
    allow_missing_outputs: bool,
    config: common.CLIConfig,
) -> None:
    import flyte
    from flyte._status import status

    console = common.get_console()
    try:
        target = f"{run_name}/{action_name}" if action_name else run_name
        status.step(f"{'Recovering' if recover else 'Re-running'} {target}...")
        runner = flyte.with_runcontext(
            mode="remote",
            name=name,
            env_vars=_parse_kv(env, "--env"),
            labels=_parse_kv(label, "--label"),
        )
        result = await runner.rerun.aio(
            run_name,
            action_name=action_name or "a0",
            recover=recover,
            force_rerun_actions=force_rerun_action or None,
            allow_missing_source_outputs=allow_missing_outputs,
            **new_inputs,
        )
    except Exception as e:
        console.print(f"[red]✕ {'Recovery' if recover else 'Re-run'} failed:[/red] {e}")
        return

    if config.output_format in ("json", "table-simple"):
        run_info = f"Created Run: {result.name}"
    else:
        run_info = f"[green bold]Created Run: {result.name}[/green bold]"
    console.print(common.get_panel("Recover" if recover else "Rerun", run_info, config.output_format))
    common.print_url(console, result.url, of=config.output_format)

    if follow:
        status.step("Waiting for log stream...")
        await result.show_logs.aio(max_lines=30, show_ts=True, raw=False)
