from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, cast

import rich_click as click
from typing_extensions import get_args

from .._code_bundle._utils import CopyFiles
from .._task import TaskTemplate
from ..remote import Run
from . import _common as common
from ._common import CLIConfig, initialize_config
from ._params import to_click_option

RUN_REMOTE_CMD = "deployed-task"


@lru_cache()
def _initialize_config(ctx: click.Context, project: str, domain: str, root_dir: str | None = None):
    obj: CLIConfig | None = ctx.obj
    if obj is None:
        import flyte.config

        obj = CLIConfig(flyte.config.auto(), ctx)

    obj.init(project, domain, root_dir)
    return obj


@lru_cache()
def _list_tasks(
    ctx: click.Context,
    project: str,
    domain: str,
    by_task_name: str | None = None,
    by_task_env: str | None = None,
) -> list[str]:
    import flyte.remote

    common.initialize_config(ctx, project, domain)
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
    root_dir: str | None = field(
        default=None,
        metadata={
            "click.option": click.Option(
                ["--root-dir"],
                type=str,
                help="Override the root source directory, helpful when working with monorepos.",
            )
        },
    )
    raw_data_path: str | None = field(
        default=None,
        metadata={
            "click.option": click.Option(
                ["--raw-data-path"],
                type=str,
                help="Override the output prefix used to store offloaded data types. e.g. s3://bucket/",
            )
        },
    )
    service_account: str | None = field(
        default=None,
        metadata={
            "click.option": click.Option(
                ["--service-account"],
                type=str,
                help="Kubernetes service account. If not provided, the configured default will be used",
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
    image: List[str] = field(
        default_factory=list,
        metadata={
            "click.option": click.Option(
                ["--image"],
                type=str,
                multiple=True,
                help="Image to be used in the run. Format: imagename=imageuri. Can be specified multiple times.",
            )
        },
    )
    no_sync_local_sys_paths: bool = field(
        default=True,
        metadata={
            "click.option": click.Option(
                ["--no-sync-local-sys-paths"],
                is_flag=True,
                flag_value=True,
                default=False,
                help="Disable synchronization of local sys.path entries under the root directory "
                "to the remote container.",
            )
        },
    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RunArguments:
        modified = {k: v for k, v in d.items() if k in {f.name for f in fields(cls)}}
        return cls(**modified)

    @classmethod
    def options(cls) -> List[click.Option]:
        """
        Return the set of base parameters added to run subcommand.
        """
        return [common.get_option_from_metadata(f.metadata) for f in fields(cls) if f.metadata]


class RunTaskCommand(click.RichCommand):
    def __init__(self, obj_name: str, obj: Any, run_args: RunArguments, *args, **kwargs):
        self.obj_name = obj_name
        self.obj = cast(TaskTemplate, obj)
        self.run_args = run_args
        kwargs.pop("name", None)
        super().__init__(obj_name, *args, **kwargs)

    def invoke(self, ctx: click.Context):
        obj: CLIConfig = initialize_config(
            ctx,
            self.run_args.project,
            self.run_args.domain,
            self.run_args.root_dir,
            tuple(self.run_args.image) or None,
            not self.run_args.no_sync_local_sys_paths,
        )

        async def _run():
            import flyte

            console = common.get_console()
            r = await flyte.with_runcontext(
                copy_style=self.run_args.copy_style,
                mode="local" if self.run_args.local else "remote",
                name=self.run_args.name,
                raw_data_path=self.run_args.raw_data_path,
                service_account=self.run_args.service_account,
                log_format=obj.log_format,
            ).run.aio(self.obj, **ctx.params)
            if self.run_args.local:
                console.print(
                    common.get_panel(
                        "Local Run",
                        f"[green]Completed Local Run, data stored in path: {r.url} [/green] \n"
                        f"➡️  Outputs: {r.outputs()}",
                        obj.output_format,
                    )
                )
                return
            if isinstance(r, Run) and r.action is not None:
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

    def get_params(self, ctx: click.Context) -> List[click.Parameter]:
        # Note this function may be called multiple times by click.
        task = self.obj
        from .._internal.runtime.types_serde import transform_native_to_typed_interface

        interface = transform_native_to_typed_interface(task.native_interface)
        if interface is None:
            return super().get_params(ctx)
        inputs_interface = task.native_interface.inputs

        params: List[click.Parameter] = []
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

    def list_commands(self, ctx):
        common.initialize_config(
            ctx,
            self.run_args.project,
            self.run_args.domain,
            self.run_args.root_dir,
            sync_local_sys_paths=not self.run_args.no_sync_local_sys_paths,
        )
        return super().list_commands(ctx)

    def get_command(self, ctx, obj_name):
        common.initialize_config(
            ctx,
            self.run_args.project,
            self.run_args.domain,
            self.run_args.root_dir,
            sync_local_sys_paths=not self.run_args.no_sync_local_sys_paths,
        )
        return super().get_command(ctx, obj_name)

    def _get_command_for_obj(self, ctx: click.Context, obj_name: str, obj: Any) -> click.Command:
        obj = cast(TaskTemplate, obj)
        return RunTaskCommand(
            obj_name=obj_name,
            obj=obj,
            help=obj.docs.__help__str__() if obj.docs else None,
            run_args=self.run_args,
        )


class RunReferenceTaskCommand(click.RichCommand):
    def __init__(self, task_name: str, run_args: RunArguments, version: str | None, *args, **kwargs):
        self.task_name = task_name
        self.run_args = run_args
        self.version = version

        super().__init__(*args, **kwargs)

    def invoke(self, ctx: click.Context):
        obj: CLIConfig = common.initialize_config(
            ctx,
            self.run_args.project,
            self.run_args.domain,
            self.run_args.root_dir,
            tuple(self.run_args.image) or None,
            not self.run_args.no_sync_local_sys_paths,
        )

        async def _run():
            import flyte.remote

            task = flyte.remote.Task.get(self.task_name, version=self.version, auto_version="latest")
            console = common.get_console()

            r = await flyte.with_runcontext(
                copy_style=self.run_args.copy_style,
                mode="local" if self.run_args.local else "remote",
                name=self.run_args.name,
            ).run.aio(task, **ctx.params)
            if isinstance(r, Run) and r.action is not None:
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

    def get_params(self, ctx: click.Context) -> List[click.Parameter]:
        # Note this function may be called multiple times by click.
        import flyte.remote
        from flyte._internal.runtime.types_serde import transform_native_to_typed_interface

        common.initialize_config(
            ctx,
            self.run_args.project,
            self.run_args.domain,
            sync_local_sys_paths=not self.run_args.no_sync_local_sys_paths,
        )

        task = flyte.remote.Task.get(self.task_name, auto_version="latest")
        task_details = task.fetch()

        interface = transform_native_to_typed_interface(task_details.interface)
        if interface is None:
            return super().get_params(ctx)
        inputs_interface = task_details.interface.inputs

        params: List[click.Parameter] = []
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
            help=f"Run deployed task '{name}' from the Flyte backend",
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

    def list_commands(self, ctx):
        v = [
            RUN_REMOTE_CMD,
            *super().list_commands(ctx),
        ]
        return v

    def get_command(self, ctx, cmd_name):
        run_args = RunArguments.from_dict(ctx.params)
        if cmd_name == RUN_REMOTE_CMD:
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
                help=f"Run `*.py` file inside the {fp} directory",
            )
        return TaskPerFileGroup(
            filename=fp,
            run_args=run_args,
            name=cmd_name,
            help=f"Run functions decorated with `env.task` in {cmd_name}",
        )


run = TaskFiles(
    name="run",
    help=f"""
Run a task from a python file or deployed task.

To run a remote task that already exists in Flyte, use the {RUN_REMOTE_CMD} command:

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

You can provide image mappings with `--image` flag. This allows you to specify
the image URI for the task environment during CLI execution without changing
the code. Any images defined with `Image.from_ref_name("name")` will resolve to the
corresponding URIs you specify here.

```bash
flyte run hello.py my_task --image my_image=ghcr.io/myorg/my-image:v1.0
```

If the image name is not provided, it is regarded as a default image and will
be used when no image is specified in TaskEnvironment:

```bash
flyte run hello.py my_task --image ghcr.io/myorg/default-image:latest
```

You can specify multiple image arguments:

```bash
flyte run hello.py my_task --image ghcr.io/org/default:latest --image gpu=ghcr.io/org/gpu:v2.0
```

To run tasks that you've already deployed to Flyte, use the {RUN_REMOTE_CMD} command:

```bash
flyte run {RUN_REMOTE_CMD} my_env.my_task --arg1 value1 --arg2 value2
```

To run a specific version of a deployed task, use the `env.task:version` syntax:

```bash
flyte run {RUN_REMOTE_CMD} my_env.my_task:xyz123 --arg1 value1 --arg2 value2
```

You can specify the `--config` flag to point to a specific Flyte cluster:

```bash
flyte run --config my-config.yaml {RUN_REMOTE_CMD} ...
```

You can discover what deployed tasks are available by running:

```bash
flyte run {RUN_REMOTE_CMD}
```

Other arguments to the run command are listed below.

Arguments for the task itself are provided after the task name and can be retrieved using `--help`. For example:

```bash
flyte run hello.py my_task --help
```
""",
)
