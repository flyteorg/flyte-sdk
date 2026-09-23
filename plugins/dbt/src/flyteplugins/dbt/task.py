from __future__ import annotations

import inspect
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from flyte.extend import RuntimeTaskTemplate
from flyte.models import NativeInterface

from flyteplugins.dbt.runner import (
    DbtEventCallback,
    DbtNodeResult,
    invoke_dbt,
)

if TYPE_CHECKING:
    from flyte import TaskEnvironment


@dataclass(kw_only=True)
class DbtTask(RuntimeTaskTemplate):
    """A Flyte task that maps one dbtRunner.invoke(cli_args) call to one task."""

    callbacks: list[DbtEventCallback | str] = field(default_factory=list)
    trace_node_events: bool = True

    def __init__(
        self,
        *,
        name: str,
        task_environment: Optional[TaskEnvironment] = None,
        callbacks: list[DbtEventCallback | str] | None = None,
        trace_node_events: bool = True,
        **kwargs: Any,
    ):
        self.callbacks = list(callbacks or [])
        self.trace_node_events = trace_node_events
        from flyteplugins.dbt.resolver import DbtTaskResolver

        task_name = f"{task_environment.name}.{name}" if task_environment else name
        interface = kwargs.pop(
            "interface",
            NativeInterface(
                inputs={"cli_args": (list[str], inspect.Parameter.empty)},
                outputs={"results": list[DbtNodeResult]},
            ),
        )
        parent_env = kwargs.pop("parent_env", weakref.ref(task_environment) if task_environment else None)
        parent_env_name = kwargs.pop("parent_env_name", task_environment.name if task_environment else None)

        super().__init__(
            name=task_name,
            interface=interface,
            image=kwargs.pop("image", task_environment.image if task_environment else "auto"),
            resources=kwargs.pop("resources", task_environment.resources if task_environment else None),
            cache=kwargs.pop("cache", task_environment.cache if task_environment else "disable"),
            reusable=kwargs.pop("reusable", task_environment.reusable if task_environment else None),
            env_vars=kwargs.pop("env_vars", task_environment.env_vars if task_environment else None),
            secrets=kwargs.pop("secrets", task_environment.secrets if task_environment else None),
            service_account=kwargs.pop(
                "service_account", task_environment.service_account if task_environment else None
            ),
            pod_template=kwargs.pop("pod_template", task_environment.pod_template if task_environment else None),
            queue=kwargs.pop("queue", task_environment.queue if task_environment else None),
            interruptible=kwargs.pop(
                "interruptible", task_environment.interruptible if task_environment else False
            ),
            short_name=kwargs.pop("short_name", name if task_environment else ""),
            task_type=kwargs.pop("task_type", "dbt"),
            _call_as_synchronous=kwargs.pop("_call_as_synchronous", True),
            parent_env=parent_env,
            parent_env_name=parent_env_name,
            task_resolver=kwargs.pop("task_resolver", DbtTaskResolver()),
            **kwargs,
        )

        if task_environment is not None:
            task_environment._tasks[task_name] = self

    def forward(self, *args: Any, **kwargs: Any) -> list[DbtNodeResult]:
        kwargs = self.interface.convert_to_kwargs(*args, **kwargs)
        return invoke_dbt(
            kwargs["cli_args"],
            callbacks=self.callbacks,
            trace_node_events=self.trace_node_events,
        )

    async def execute(self, *args: Any, **kwargs: Any) -> list[DbtNodeResult]:
        kwargs = self.interface.convert_to_kwargs(*args, **kwargs)

        from flyte._utils.asyncify import run_sync_in_thread

        return await run_sync_in_thread(
            invoke_dbt,
            kwargs["cli_args"],
            self.callbacks,
            trace_node_events=self.trace_node_events,
        )
