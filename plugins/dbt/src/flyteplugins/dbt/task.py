from __future__ import annotations

import inspect
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from flyte.extend import TaskTemplate
from flyte.models import NativeInterface, SerializationContext

from flyteplugins.dbt.runner import (
    DbtEventCallback,
    DbtNodeResult,
    callback_import_paths,
    invoke_dbt,
)

if TYPE_CHECKING:
    from flyte import TaskEnvironment


@dataclass(kw_only=True)
class DbtTask(TaskTemplate):
    """A Flyte task that maps one dbtRunner.invoke(cli_args) call to one task."""

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
        super().__init__(
            name=name,
            interface=NativeInterface(
                inputs={"cli_args": (list[str], inspect.Parameter.empty)},
                outputs={"results": list[DbtNodeResult]},
            ),
            task_type="dbt",
            _call_as_synchronous=True,
            parent_env=weakref.ref(task_environment) if task_environment else None,
            parent_env_name=task_environment.name if task_environment else None,
            **kwargs,
        )

    def custom_config(self, sctx: SerializationContext) -> dict[str, Any]:
        return {}

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

    def container_args(self, serialize_context: SerializationContext) -> list[str]:
        from flyteplugins.dbt.resolver import DbtTaskResolver

        resolver = DbtTaskResolver()

        args = [
            "a0",
            "--inputs",
            serialize_context.input_path,
            "--outputs-path",
            serialize_context.output_path,
            "--version",
            serialize_context.version,
            "--raw-data-path",
            "{{.rawOutputDataPrefix}}",
            "--checkpoint-path",
            "{{.checkpointOutputPrefix}}",
            "--prev-checkpoint",
            "{{.prevCheckpointPrefix}}",
            "--run-name",
            "{{.runName}}",
            "--name",
            "{{.actionName}}",
            "--run-start-time",
            "{{.runStartTime}}",
        ]

        if serialize_context.image_cache and serialize_context.image_cache.serialized_form:
            args = [
                *args,
                "--image-cache",
                serialize_context.image_cache.serialized_form,
            ]
        elif serialize_context.image_cache:
            args = [
                *args,
                "--image-cache",
                serialize_context.image_cache.to_transport,
            ]

        if serialize_context.code_bundle:
            if serialize_context.code_bundle.tgz:
                args = [*args, "--tgz", f"{serialize_context.code_bundle.tgz}"]
            elif serialize_context.code_bundle.pkl:
                args = [*args, "--pkl", f"{serialize_context.code_bundle.pkl}"]
            args = [*args, "--dest", f"{serialize_context.code_bundle.destination or '.'}"]

        return [
            *args,
            "--resolver",
            resolver.import_path,
            *resolver.loader_args(task=self, root_dir=serialize_context.root_dir),
        ]

    def callback_import_paths(self) -> list[str]:
        return callback_import_paths(self.callbacks)
