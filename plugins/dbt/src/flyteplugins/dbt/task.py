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
    include_default_callback: bool = True

    def __init__(
        self,
        *,
        name: str,
        task_environment: Optional[TaskEnvironment] = None,
        callbacks: list[DbtEventCallback | str] | None = None,
        include_default_callback: bool = True,
        **kwargs: Any,
    ):
        self.callbacks = list(callbacks or [])
        self.include_default_callback = include_default_callback
        from flyteplugins.dbt.resolver import DbtTaskResolver

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
            name=name,
            interface=interface,
            task_type=kwargs.pop("task_type", "dbt"),
            _call_as_synchronous=kwargs.pop("_call_as_synchronous", True),
            parent_env=parent_env,
            parent_env_name=parent_env_name,
            task_resolver=kwargs.pop("task_resolver", DbtTaskResolver()),
            **kwargs,
        )

    def forward(self, *args: Any, **kwargs: Any) -> list[DbtNodeResult]:
        kwargs = self.interface.convert_to_kwargs(*args, **kwargs)
        return invoke_dbt(
            kwargs["cli_args"],
            callbacks=self.callbacks,
            include_default_callback=self.include_default_callback,
        )

    async def execute(self, *args: Any, **kwargs: Any) -> list[DbtNodeResult]:
        kwargs = self.interface.convert_to_kwargs(*args, **kwargs)

        from flyte._utils.asyncify import run_sync_in_thread

        return await run_sync_in_thread(
            invoke_dbt,
            kwargs["cli_args"],
            self.callbacks,
            include_default_callback=self.include_default_callback,
        )
