"""Route `flyte run` by hooking the CLI, with no support needed from the SDK.

`flyte run tasks.py my_task --x 1` resolves through a chain of Click groups -- `run` is a group of
files, each file a group of tasks -- so the task is only known at the leaf. This wraps
`get_command` at each level so whatever the chain produces gets wrapped too. At the leaf both the
task object and its parsed arguments are in hand, which is what makes data-location routing
possible here.

`flyte.run(...)` from a script or notebook never goes through Click, so this never sees it; use
`flyteplugins.routing.run(...)` there. That split is the trade for the SDK carrying no routing
seam at all.
"""

from __future__ import annotations

from typing import Any, Optional

import rich_click as click

from ._router import route
from ._types import RoutingContext, RoutingDecision

__all__ = ["route_run"]


def _decide(cmd: click.Command, ctx: click.Context) -> Optional[RoutingDecision]:
    """Build the policy's context from what the leaf command already holds."""
    import flyte.config as config

    obj = ctx.obj
    if obj is None:
        return None

    task = getattr(cmd, "obj", None)
    parent_env = getattr(task, "parent_env", None)
    env = parent_env() if callable(parent_env) else None
    resources = getattr(task, "resources", None) or getattr(env, "resources", None)

    run_args = getattr(cmd, "run_args", None)

    return route(
        RoutingContext(
            profiles=tuple(config.list_profiles(obj.config.source)),
            active_profile=obj.config.profile,
            project=obj.config.task.project,
            domain=obj.config.task.domain,
            task_name=getattr(task, "name", None),
            resources=resources,
            inputs=dict(ctx.params),
            run_name=getattr(run_args, "name", None),
        )
    )


def _wrap_leaf(cmd: click.Command) -> click.Command:
    """Wrap a leaf run command so the policy runs just before it submits."""
    original_invoke = cmd.invoke

    def invoke(ctx: click.Context):
        import flyte

        try:
            decision = _decide(cmd, ctx)
        except Exception as e:
            # A policy fault must not take down the run. Submitting to the ambient profile is the
            # documented fallback, and it is what would happen with the plugin uninstalled.
            click.echo(f"Warning: routing policy failed, using the default profile: {e}", err=True)
            return original_invoke(ctx)

        if decision is None:
            return original_invoke(ctx)

        run_args = getattr(cmd, "run_args", None)
        previous = None
        if run_args is not None:
            previous = (run_args.name, list(run_args.label or []))
            if decision.run_name and not run_args.name:
                run_args.name = decision.run_name
            if decision.labels:
                # `--label KEY=VALUE`, parsed later by RunArguments.parsed_labels(). Appending
                # leaves any label the caller passed ahead of ours, so theirs wins on conflict.
                existing = {item.split("=", 1)[0] for item in (run_args.label or []) if "=" in item}
                run_args.label = [
                    *(run_args.label or []),
                    *(f"{k}={v}" for k, v in decision.labels.items() if k not in existing),
                ]

        try:
            if decision.profile is None or decision.profile == ctx.obj.config.profile:
                return original_invoke(ctx)
            with flyte.use_profile(decision.profile, config_file=ctx.obj.config.source):
                return original_invoke(ctx)
        finally:
            # The command object is reused across invocations in-process (tests, a REPL), so a
            # name minted for this run must not leak into the next one and collide.
            if run_args is not None and previous is not None:
                run_args.name, run_args.label = previous[0], previous[1]

    cmd.invoke = invoke  # type: ignore[method-assign]
    return cmd


def _wrap(cmd: Any) -> Any:
    """Wrap a command, following groups down to the leaf.

    `run` yields file groups, which yield task commands; only the last knows what is being run, so
    each level rewrites the next level's `get_command`.
    """
    if isinstance(cmd, click.Group):
        original_get = cmd.get_command

        def get_command(ctx, name):
            found = original_get(ctx, name)
            return _wrap(found) if found is not None else None

        cmd.get_command = get_command  # type: ignore[method-assign]
        return cmd
    return _wrap_leaf(cmd)


def route_run(command: click.Command) -> click.Command:
    """Entry point for the `run` CLI hook."""
    return _wrap(command)
