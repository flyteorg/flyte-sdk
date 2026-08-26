"""Routing for submissions that do not go through the CLI.

`flyte.run(...)` from a script, notebook or orchestrator never goes through Click, so the CLI hook
never sees it. These are drop-in replacements applying the same policy -- a mechanical swap of the
module name:

    from flyteplugins import routing

    routing.run(my_task, "s3://bucket/x")                       # was flyte.run(...)
    routing.with_runcontext(version="v2").run(my_task, x=1)     # was flyte.with_runcontext(...)

Anything still calling `flyte.run` submits to the default profile. If a scheduler submits your
production runs, route them through here or pin them with a profile -- do not assume they inherit
the CLI's behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import flyte
import flyte.config as config

from ._router import route
from ._types import RoutingContext, RoutingDecision

__all__ = ["decide", "run", "with_runcontext"]


def _session_config_file(config_file):
    """Default to the config file this session was initialized from.

    Without this, `routing.run(...)` after `flyte.init_from_config(path)` would re-run the default
    config search and route against a different file -- or, finding none, not route at all.
    """
    if config_file is not None:
        return config_file
    from flyte._initialize import _get_init_config

    init_cfg = _get_init_config()
    return init_cfg.source_config_path if init_cfg is not None else None


def _named_inputs(task: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve positional arguments to names through the task's interface.

    The policy hashes by name, so `run(t, "s3://x")` and `run(t, dataset="s3://x")` must describe
    identically -- placement must not depend on how the call was written.
    """
    interface = getattr(task, "native_interface", None)
    if interface is not None:
        try:
            return interface.convert_to_kwargs(*args, **kwargs)
        except Exception:
            # Genuinely wrong arguments are reported by the submit path, with a better message
            # than routing could give. Route on what we can name and let it get there.
            pass
    return dict(kwargs)


def decide(
    task: Any,
    *args: Any,
    config_file=None,
    profile: Optional[str] = None,
    run_name: Optional[str] = None,
    **kwargs: Any,
) -> Optional[RoutingDecision]:
    """Apply the policy without submitting, to see where a run would go."""
    config_file = _session_config_file(config_file)
    cfg = config.auto(config_file, profile=profile)
    parent_env = getattr(task, "parent_env", None)
    env = parent_env() if callable(parent_env) else None

    return route(
        RoutingContext(
            profiles=tuple(config.list_profiles(config_file)),
            active_profile=profile or config.get_active_profile(),
            project=cfg.task.project,
            domain=cfg.task.domain,
            task_name=getattr(task, "name", None),
            resources=getattr(task, "resources", None) or getattr(env, "resources", None),
            inputs=_named_inputs(task, args, kwargs),
            run_name=run_name,
        )
    )


class _RoutedRunner:
    """What `with_runcontext` returns: a runner that routes before it submits.

    Passes every `flyte.with_runcontext` option straight through, adding only the routed profile,
    the run name and the decision's labels.
    """

    def __init__(self, config_file=None, profile: Optional[str] = None, **runcontext: Any) -> None:
        self._config_file = _session_config_file(config_file)
        self._profile = profile
        self._runcontext = runcontext

    def run(self, task: Any, *args: Any, **kwargs: Any):
        decision = decide(
            task,
            *args,
            config_file=self._config_file,
            profile=self._profile,
            run_name=self._runcontext.get("name"),
            **kwargs,
        )

        opts = dict(self._runcontext)
        if self._profile is not None:
            # An explicitly pinned profile is honoured even though the policy declines on it.
            with flyte.use_profile(self._profile, config_file=self._config_file):
                return flyte.with_runcontext(**opts).run(task, *args, **kwargs)

        if decision is None or decision.profile is None:
            return (
                flyte.with_runcontext(**opts).run(task, *args, **kwargs) if opts else flyte.run(task, *args, **kwargs)
            )

        if decision.run_name and not opts.get("name"):
            opts["name"] = decision.run_name
        if decision.labels:
            # Caller labels win on conflict, matching the CLI hook.
            merged = dict(decision.labels)
            merged.update(opts.get("labels") or {})
            opts["labels"] = merged

        with flyte.use_profile(decision.profile, config_file=self._config_file):
            return flyte.with_runcontext(**opts).run(task, *args, **kwargs)


def with_runcontext(config_file=None, profile: Optional[str] = None, **runcontext: Any) -> _RoutedRunner:
    """Drop-in for `flyte.with_runcontext` that routes the run.

    Plus `config_file` and `profile`, to point at a specific config or pin a profile outright.
    """
    return _RoutedRunner(config_file=config_file, profile=profile, **runcontext)


def run(task: Any, *args: Any, **kwargs: Any):
    """Drop-in for `flyte.run` that routes the run.

    Positional and keyword arguments both work. For run options, use `with_runcontext(...).run()`.
    """
    return _RoutedRunner().run(task, *args, **kwargs)
