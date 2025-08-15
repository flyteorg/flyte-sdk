from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, cast

import click
from click import Context, Parameter
from rich.console import Console
from typing_extensions import get_args

from .._code_bundle._utils import CopyFiles
from .._task import TaskTemplate
from ..remote import Run
from . import _common as common
from ._common import CLIConfig
from ._params import to_click_option


@lru_cache()
def _initialize_config(ctx: Context, project: str, domain: str):
    obj: CLIConfig | None = ctx.obj
    if obj is None:
        import flyte.config

        obj = CLIConfig(flyte.config.auto(), ctx)

    obj.init(project, domain)
    return obj


@lru_cache()
def _list_tasks(
    ctx: Context,
    project: str,
    domain: str,
    by_task_name: str | None = None,
    by_task_env: str | None = None,
) -> list[str]:
    import flyte.remote

    _initialize_config(ctx, project, domain)
    return [task.name for task in flyte.remote.Task.listall(by_task_name=by_task_name, by_task_env=by_task_env)]


@dataclass
class RunArguments:
    project: str = field(
        default=cast(str, common.PROJECT_OPTION.default), metadata={"click.option": common.PROJECT_OPTION}
    )
    domain: str = field(
        default=cast(str, common.DOMAIN_OPTION.default), metadata={"click.option": common.DOMAIN_OPTION}
    )
    local: bool = field(
        default=False,
        metadata={
            "click.option": click.Option(
                ["--local"],
                is_flag=True,
                help="Run the task locally",
            )
        },
    )
    copy_style: CopyFiles = field(
        default="loaded_modules",
        metadata={
            "click.option": click.Option(
                ["--copy-style"],
                type=click.Choice(get_args(CopyFiles)),
                default="loaded_modules",
                help="Copy style to use when running the task",
            )
        },
    )
    name: str | None = field(
        default=None,
        metadata={
            "click.option": click.Option(
                ["--name"],
                type=str,
                help="Name of the run. If not provided, a random name will be generated.",
            )
        },
    )
    follow: bool = field(
        default=False,
        metadata={
            "click.option": click.Option(
                ["--follow", "-f"],
                is_flag=True,
                default=False,
                help="Wait and watch logs for the parent action. If not provided, the CLI will exit after "
                "successfully launching a remote execution with a link to the UI.",
            )
        },
    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RunArguments:
        return cls(**d)

    @classmethod
    def options(cls) -> List[click.Option]:
        """
        Return the set of base parameters added to run subcommand.
        """
        return [common.get_option_from_metadata(f.metadata) for f in fields(cls) if f.metadata]


class RunTaskCommand(click.Command):
    def __init__(self, obj_name: str, obj: Any, run_args: RunArguments, *args, **kwargs):
        self.obj_name = obj_name
        self.obj = cast(TaskTemplate, obj)
        self.run_args = run_args
        kwargs.pop("name", None)
        super().__init__(obj_name, *args, **kwargs)

    def invoke(self, ctx: Context):
        obj: CLIConfig = _initialize_config(ctx, self.run_args.project, self.run_args.domain)

        async def _run():
            import flyte

            r = await flyte.with_runcontext(
                copy_style=self.run_args.copy_style,
                mode="local" if self.run_args.local else "remote",
                name=self.run_args.name,
            ).run.aio(self.obj, **ctx.params)
            if isinstance(r, Run) and r.action is not None:
                console = Console()
                console.print(
                    common.get_panel(
                        "Run",
                        f"[green bold]Created Run: {r.name} [/green bold] "
                        f"(Project: {r.action.action_id.run.project}, Domain: {r.action.action_id.run.domain})\n"
                        f"➡️  [blue bold][link={r.url}]{r.url}[/link][/blue bold]",
                        obj.output_format,
                    )
                )
                if self.run_args.follow:
                    console.print(
                        "[dim]Log streaming enabled, will wait for task to start running "
                        "and log stream to be available[/dim]"
                    )
                    await r.show_logs.aio(max_lines=30, show_ts=True, raw=False)

        asyncio.run(_run())

    def get_params(self, ctx: Context) -> List[Parameter]:
        # Note this function may be called multiple times by click.
        task = self.obj
        from .._internal.runtime.types_serde import transform_native_to_typed_interface

        interface = transform_native_to_typed_interface(task.native_interface)
        if interface is None:
            return super().get_params(ctx)
        inputs_interface = task.native_interface.inputs

        params: List[Parameter] = []
        for name, var in interface.inputs.variables.items():
            default_val = None
            if inputs_interface[name][1] is not inspect._empty:
                default_val = inputs_interface[name][1]
            params.append(to_click_option(name, var, inputs_interface[name][0], default_val))

        self.params = params
        return super().get_params(ctx)


class TaskPerFileGroup(common.ObjectsPerFileGroup):
    """
    Group that creates a command for each task in the current directory that is not __init__.py.
    """

    def __init__(self, filename: Path, run_args: RunArguments, *args, **kwargs):
        if filename.is_absolute():
            filename = filename.relative_to(Path.cwd())
        super().__init__(*(filename, *args), **kwargs)
        self.run_args = run_args

    def _filter_objects(self, module: ModuleType) -> Dict[str, Any]:
        return {k: v for k, v in module.__dict__.items() if isinstance(v, TaskTemplate)}

    def _get_command_for_obj(self, ctx: click.Context, obj_name: str, obj: Any) -> click.Command:
        obj = cast(TaskTemplate, obj)
        return RunTaskCommand(
            obj_name=obj_name,
            obj=obj,
            help=obj.docs.__help__str__() if obj.docs else None,
            run_args=self.run_args,
        )


class RunReferenceTaskCommand(click.Command):
    def __init__(self, task_name: str, run_args: RunArguments, version: str | None, *args, **kwargs):
        self.task_name = task_name
        self.run_args = run_args
        self.version = version

        super().__init__(*args, **kwargs)

    def invoke(self, ctx: click.Context):
        obj: CLIConfig = _initialize_config(ctx, self.run_args.project, self.run_args.domain)

        async def _run():
            import flyte
            import flyte.remote

            task = flyte.remote.Task.get(self.task_name, version=self.version, auto_version="latest")

            r = await flyte.with_runcontext(
                copy_style=self.run_args.copy_style,
                mode="local" if self.run_args.local else "remote",
                name=self.run_args.name,
            ).run.aio(task, **ctx.params)
            if isinstance(r, Run) and r.action is not None:
                console = Console()
                console.print(
                    common.get_panel(
                        "Run",
                        f"[green bold]Created Run: {r.name} [/green bold] "
                        f"(Project: {r.action.action_id.run.project}, Domain: {r.action.action_id.run.domain})\n"
                        f"➡️  [blue bold][link={r.url}]{r.url}[/link][/blue bold]",
                        obj.output_format,
                    )
                )
                if self.run_args.follow:
                    console.print(
                        "[dim]Log streaming enabled, will wait for task to start running "
                        "and log stream to be available[/dim]"
                    )
                    await r.show_logs.aio(max_lines=30, show_ts=True, raw=False)

        asyncio.run(_run())

    def get_params(self, ctx: Context) -> List[Parameter]:
        # Note this function may be called multiple times by click.
        import flyte.remote
        from flyte._internal.runtime.types_serde import transform_native_to_typed_interface

        _initialize_config(ctx, self.run_args.project, self.run_args.domain)

        task = flyte.remote.Task.get(self.task_name, auto_version="latest")
        task_details = task.fetch()

        interface = transform_native_to_typed_interface(task_details.interface)
        if interface is None:
            return super().get_params(ctx)
        inputs_interface = task_details.interface.inputs

        params: List[Parameter] = []
        for name, var in interface.inputs.variables.items():
            default_val = None
            if inputs_interface[name][1] is not inspect._empty:
                default_val = inputs_interface[name][1]
            params.append(to_click_option(name, var, inputs_interface[name][0], default_val))

        self.params = params
        return super().get_params(ctx)


class ReferenceEnvGroup(common.GroupBase):
    def __init__(self, name: str, *args, run_args, env: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.env = env
        self.run_args = run_args

    def list_commands(self, ctx):
        return _list_tasks(ctx, self.run_args.project, self.run_args.domain, by_task_env=self.env)

    def get_command(self, ctx, name):
        return RunReferenceTaskCommand(
            task_name=name,
            run_args=self.run_args,
            name=name,
            version=None,
            help=f"Run reference task '{name}' from the Flyte backend",
        )


class ReferenceTaskGroup(common.GroupBase):
    """
    Group that creates a command for each reference task in the current directory that is not __init__.py.
    """

    def __init__(self, name: str, *args, run_args, tasks: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.run_args = run_args

    def list_commands(self, ctx):
        # list envs of all reference tasks
        envs = []
        for task in _list_tasks(ctx, self.run_args.project, self.run_args.domain):
            env = task.split(".")[0]
            if env not in envs:
                envs.append(env)
        return envs

    @staticmethod
    def _parse_task_name(task_name: str) -> tuple[str, str | None, str | None]:
        import re

        pattern = r"^([^.:]+)(?:\.([^:]+))?(?::(.+))?$"
        match = re.match(pattern, task_name)
        if not match:
            raise click.BadParameter(f"Invalid task name format: {task_name}")
        return match.group(1), match.group(2), match.group(3)

    def _env_is_task(self, ctx: click.Context, env: str) -> bool:
        # check if the env name is the full task name, since sometimes task
        # names don't have an environment prefix
        tasks = [*_list_tasks(ctx, self.run_args.project, self.run_args.domain, by_task_name=env)]
        return len(tasks) > 0

    def get_command(self, ctx, name):
        env, task, version = self._parse_task_name(name)

        match env, task, version:
            case env, None, None:
                if self._env_is_task(ctx, env):
                    # this handles cases where task names do not have a environment prefix
                    task_name = env
                    return RunReferenceTaskCommand(
                        task_name=task_name,
                        run_args=self.run_args,
                        name=task_name,
                        version=None,
                        help=f"Run reference task `{task_name}` from the Flyte backend",
                    )
                else:
                    return ReferenceEnvGroup(
                        name=name,
                        run_args=self.run_args,
                        env=env,
                        help=f"Run reference tasks in the `{env}` environment from the Flyte backend",
                    )
            case env, task, None:
                task_name = f"{env}.{task}"
                return RunReferenceTaskCommand(
                    task_name=task_name,
                    run_args=self.run_args,
                    name=task_name,
                    version=None,
                    help=f"Run reference task '{task_name}' from the Flyte backend",
                )
            case env, task, version:
                task_name = f"{env}.{task}"
                return RunReferenceTaskCommand(
                    task_name=task_name,
                    run_args=self.run_args,
                    version=version,
                    name=f"{task_name}:{version}",
                    help=f"Run reference task '{task_name}' from the Flyte backend",
                )
            case _:
                raise click.BadParameter(f"Invalid task name format: {task_name}")


class TaskFiles(common.FileGroup):
    """
    Group that creates a command for each file in the current directory that is not __init__.py.
    """

    common_options_enabled = False

    def __init__(
        self,
        *args,
        directory: Path | None = None,
        **kwargs,
    ):
        if "params" not in kwargs:
            kwargs["params"] = []
        kwargs["params"].extend(RunArguments.options())
        super().__init__(*args, directory=directory, **kwargs)

    def get_command(self, ctx, cmd_name):
        run_args = RunArguments.from_dict(ctx.params)

        if cmd_name == "reference-task":
            return ReferenceTaskGroup(
                name=cmd_name,
                run_args=run_args,
                help="Run reference task from the Flyte backend",
            )

        fp = Path(cmd_name)
        if not fp.exists():
            raise click.BadParameter(f"File {cmd_name} does not exist")
        if fp.is_dir():
            return TaskFiles(
                directory=fp,
                help=f"Run `*.py` file inside in the {fp} directory",
            )
        return TaskPerFileGroup(
            filename=fp,
            run_args=run_args,
            name=cmd_name,
            help=f"Run functions decorated with `env.task` in {cmd_name}",
        )


run = TaskFiles(
    name="run",
    help="""
Run a task from a python file or reference a remote task.

To run a remote task that already exists in Flyte, use the reference-task command:

Example usage:

```bash
flyte run --project my-project --domain development hello.py my_task --arg1 value1 --arg2 value2
```

Arguments to the run command are provided right after the `run` command and before the file name.
For example, the command above specifies the project and domain.

To run a task locally, use the `--local` flag. This will run the task in the local environment instead of the remote
Flyte environment:

```bash
flyte run --local hello.py my_task --arg1 value1 --arg2 value2
```

To run tasks that you've already deployed to Flyte, use the reference-task command:

```bash
flyte run reference-task my_env.my_task --arg1 value1 --arg2 value2
```

You can specify the `--config` flag to point to a specific Flyte cluster:

```bash
flyte run --config my-config.yaml reference-task ...
```

You can discover what reference tasks are available by running:

```bash
flyte run reference-task
```

Other arguments to the run command are listed below.

Arguments for the task itself are provided after the task name and can be retrieved using `--help`. For example:

```bash
flyte run hello.py my_task --help
```
""",
)
