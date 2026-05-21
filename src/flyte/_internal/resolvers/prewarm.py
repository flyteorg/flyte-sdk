"""Resolver for auto-synthesized prewarm tasks.

A prewarm task is a hidden no-op that the SDK attaches to every reusable
TaskEnvironment so that `env.prewarm()` can submit a cheap sub-action whose
sole side-effect is to trigger `GetOrCreateEnvironment` on the backend
(see `cloud/flyte/fasttask/plugin/plugin.go`). The dummy function lives in
the SDK rather than in user code, so the worker needs a resolver that can
materialize a task without importing the user's module.
"""

from __future__ import annotations

import pathlib
from typing import List, Optional

from flyte._internal.resolvers.common import Resolver
from flyte._task import AsyncFunctionTaskTemplate, TaskTemplate
from flyte.models import NativeInterface


def prewarm_task_short_name(env_name: str) -> str:
    """Per-env short name so the UI shows which env each prewarm belongs to.

    Prefix-form (`prewarm_<env>`) makes prewarm tasks sort together when
    listing a project's tasks alphabetically.
    """
    return f"prewarm_{env_name}"


def prewarm_task_full_name(env_name: str) -> str:
    return f"{env_name}.{prewarm_task_short_name(env_name)}"


async def _prewarm_noop() -> int:
    """No-op coroutine; serves only as a vehicle to spin up the actor pool."""
    return 0


class PrewarmTaskResolver(Resolver):
    """Resolver that materializes a prewarm task from its env name alone.

    The synthesized task carries no user code, so on the worker side we can
    rebuild it from `["env_name", <env_name>]` without touching the code
    bundle. The wire `TaskTemplate` proto carries the image, ReusePolicy,
    secrets, env_vars, etc. — the worker only needs a callable to execute.
    """

    @property
    def import_path(self) -> str:
        return "flyte._internal.resolvers.prewarm.PrewarmTaskResolver"

    def load_task(self, loader_args: List[str]) -> TaskTemplate:
        # loader_args is ["env_name", <env_name>]
        if len(loader_args) < 2 or loader_args[0] != "env_name":
            raise ValueError(f"PrewarmTaskResolver expected ['env_name', <name>], got {loader_args}")
        env_name = loader_args[1]
        return AsyncFunctionTaskTemplate(
            func=_prewarm_noop,
            name=prewarm_task_full_name(env_name),
            short_name=prewarm_task_short_name(env_name),
            interface=NativeInterface.from_callable(_prewarm_noop),
            parent_env_name=env_name,
            task_resolver=self,
        )

    def loader_args(self, task: TaskTemplate, root_dir: Optional[pathlib.Path] = None) -> List[str]:
        return ["env_name", task.parent_env_name or ""]
