"""Teach the run-addressed CLI commands to find runs on other profiles.

Every command taking a run name has the problem `flyte get run` has: a run on a non-ambient
profile is invisible to it. That includes the ones that *write* -- `rerun`, `abort` -- not just
the reads.

This is resolution, not routing: the policy picks where a *new* run goes; these find where an
*existing* one already is. A rerun must go to the control plane holding the source run, so it
never consults the policy.

Hooks rather than replacement commands, so the core command still runs and this only decides which
profile it runs under.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import rich_click as click

from ._resolve import resolve_run_profile

__all__ = [
    "peek_argument",
    "resolve_profile_at_parse",
    "resolve_profile_for",
    "route_abort_action",
    "route_abort_run",
    "route_get_action",
    "route_get_condition",
    "route_get_io",
    "route_get_logs",
    "route_get_run",
    "route_rerun",
    "route_signal_condition",
]


def _switch_profile(ctx: click.Context, run_name: str) -> None:
    """Point `ctx` at the profile holding `run_name`, if the name says which one that is.

    Swaps the profile on the CLI's own config object rather than entering `flyte.use_profile`, so
    the command builds exactly one client, for the right control plane.

    A name this plugin did not mint leaves the command alone, on the default profile.
    """
    import flyte.config as config

    obj = ctx.obj
    if obj is None:
        return

    source = obj.config.source
    profile = resolve_run_profile(run_name, source)
    if profile and profile != obj.config.profile:
        ctx.obj = obj.replace(config=config.auto(source, profile=profile), profile=profile)


def resolve_profile_for(param: str) -> Callable[[click.Command], click.Command]:
    """Hook that resolves the profile at invoke time, once Click has parsed the arguments.

    Right for any command that does not need config before `invoke` -- which is all of them except
    `rerun`; see `resolve_profile_at_parse`.

    Args:
        param: Name of the Click parameter carrying the run name (`name` on `get run`, `run_name`
            everywhere else).
    """

    def hook(command: click.Command) -> click.Command:
        original_invoke = command.invoke

        def wrapper(ctx: click.Context):
            run_name = ctx.params.get(param)
            # `get run` with no name is a listing, not a lookup -- nothing to resolve.
            if run_name:
                _switch_profile(ctx, run_name)
            return original_invoke(ctx)

        command.invoke = wrapper  # type: ignore[method-assign]
        return command

    return hook


def peek_argument(command: click.Command, args: List[str], param: str) -> Optional[str]:
    """Read a positional argument straight out of raw argv, before Click parses.

    Skips options and their values, so `flyte rerun --project p my-run` yields `my-run`, not `p`.
    """
    takes_value = set()
    for p in command.params:
        if isinstance(p, click.Option) and not p.is_flag and not p.count:
            takes_value.update(p.opts)
            takes_value.update(p.secondary_opts)

    positional = [p.name for p in command.params if isinstance(p, click.Argument)]
    if param not in positional:
        return None
    wanted = positional.index(param)

    seen: List[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--":
            seen.extend(args[i + 1 :])
            break
        if token.startswith("-") and token != "-":
            # `--opt=value` carries its value; `--opt value` consumes the next token.
            if "=" not in token and token in takes_value:
                i += 2
                continue
            i += 1
            continue
        seen.append(token)
        i += 1

    return seen[wanted] if wanted < len(seen) else None


def resolve_profile_at_parse(param: str) -> Callable[[click.Command], click.Command]:
    """Hook that resolves the profile before Click parses the command's arguments.

    `rerun` reads the source run's interface during `parse_args` to turn `--some-input v` into a
    typed option. That read goes to the control plane holding the run, so the profile must be
    settled before parsing, not at invoke time.
    """

    def hook(command: click.Command) -> click.Command:
        original_parse_args = command.parse_args

        def wrapper(ctx: click.Context, args: List[str]):
            run_name = peek_argument(command, args, param)
            if run_name:
                try:
                    _switch_profile(ctx, run_name)
                except Exception as e:
                    # `--help` must render without config or connectivity, and a resolution failure
                    # should surface as the command's own error, not as a plugin traceback.
                    click.echo(f"Warning: could not resolve a profile for {run_name!r}: {e}", err=True)
            return original_parse_args(ctx, args)

        command.parse_args = wrapper  # type: ignore[method-assign]
        return command

    return hook


# Reads.
route_get_run = resolve_profile_for("name")
route_get_logs = resolve_profile_for("run_name")
route_get_action = resolve_profile_for("run_name")
route_get_io = resolve_profile_for("run_name")
route_get_condition = resolve_profile_for("run_name")

# Writes against an existing run. These need resolving just as much as the reads do -- more so,
# since aborting or rerunning against the wrong control plane simply fails.
route_abort_run = resolve_profile_for("run_name")
route_abort_action = resolve_profile_for("run_name")
route_signal_condition = resolve_profile_for("run_name")
route_rerun = resolve_profile_at_parse("run_name")
