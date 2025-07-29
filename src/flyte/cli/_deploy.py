from dataclasses import dataclass, field, fields
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, cast, get_args

import click
from click import Context

import flyte

from .._code_bundle._utils import CopyFiles
from . import _common as common
from ._common import CLIConfig


@dataclass
class DeployArguments:
    project: str = field(
        default=cast(str, common.PROJECT_OPTION.default), metadata={"click.option": common.PROJECT_OPTION}
    )
    domain: str = field(
        default=cast(str, common.DOMAIN_OPTION.default), metadata={"click.option": common.DOMAIN_OPTION}
    )
    version: str = field(
        default="",
        metadata={
            "click.option": click.Option(
                ["--version"],
                type=str,
                help="Version of the environment to deploy",
            )
        },
    )
    dry_run: bool = field(default=False, metadata={"click.option": common.DRY_RUN_OPTION})
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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeployArguments":
        return cls(**d)

    @classmethod
    def options(cls) -> List[click.Option]:
        """
        Return the set of base parameters added to every flyte run workflow subcommand.
        """
        return [common.get_option_from_metadata(f.metadata) for f in fields(cls) if f.metadata]


class DeployEnvCommand(click.Command):
    def __init__(self, obj_name: str, obj: Any, deploy_args: DeployArguments, *args, **kwargs):
        self.obj_name = obj_name
        self.obj = obj
        self.deploy_args = deploy_args
        super().__init__(*args, **kwargs)

    def invoke(self, ctx: Context):
        from rich.console import Console

        console = Console()
        console.print(f"Deploying root - environment: {self.obj_name}")
        obj: CLIConfig = ctx.obj
        obj.init(self.deploy_args.project, self.deploy_args.domain)
        with console.status("Deploying...", spinner="dots"):
            deployment = flyte.deploy(
                self.obj,
                dryrun=self.deploy_args.dry_run,
                copy_style=self.deploy_args.copy_style,
                version=self.deploy_args.version,
            )

        console.print(common.get_table("Environments", deployment.env_repr(), simple=obj.simple))
        console.print(common.get_table("Tasks", deployment.task_repr(), simple=obj.simple))


class EnvPerFileGroup(common.ObjectsPerFileGroup):
    """
    Group that creates a command for each task in the current directory that is not `__init__.py`.
    """

    def __init__(self, filename: Path, deploy_args: DeployArguments, *args, **kwargs):
        args = (filename, *args)
        super().__init__(*args, **kwargs)
        self.deploy_args = deploy_args

    def _filter_objects(self, module: ModuleType) -> Dict[str, Any]:
        return {k: v for k, v in module.__dict__.items() if isinstance(v, flyte.Environment)}

    def _get_command_for_obj(self, ctx: click.Context, obj_name: str, obj: Any) -> click.Command:
        obj = cast(flyte.Environment, obj)
        return DeployEnvCommand(
            name=obj_name,
            obj_name=obj_name,
            obj=obj,
            help=f"{obj.name}" + (f": {obj.description}" if obj.description else ""),
            deploy_args=self.deploy_args,
        )


class EnvFiles(common.FileGroup):
    """
    Group that creates a command for each file in the current directory that is not `__init__.py`.
    """

    common_options_enabled = False

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        if "params" not in kwargs:
            kwargs["params"] = []
        kwargs["params"].extend(DeployArguments.options())
        super().__init__(*args, **kwargs)

    def get_command(self, ctx, filename):
        deploy_args = DeployArguments.from_dict(ctx.params)
        return EnvPerFileGroup(
            filename=Path(filename),
            deploy_args=deploy_args,
            name=filename,
            help=f"Run, functions decorated `env.task` or instances of Tasks in {filename}",
        )


deploy = EnvFiles(
    name="deploy",
    help="""
    Deploy one or more environments from a python file.
    This command will create or update environments in the Flyte system.
    """,
)
