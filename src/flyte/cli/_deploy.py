import pathlib
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, cast, get_args

import rich_click as click

import flyte
from flyte._code_bundle._utils import CopyFiles

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
    recursive: bool = field(
        default=False,
        metadata={
            "click.option": click.Option(
                ["--recursive", "-r"],
                is_flag=True,
                help="Recursively deploy all environments in the current directory",
            )
        },
    )
    all: bool = field(
        default=False,
        metadata={
            "click.option": click.Option(
                ["--all"],
                is_flag=True,
                help="Deploy all environments in the current directory, ignoring the file name",
            )
        },
    )
    ignore_load_errors: bool = field(
        default=False,
        metadata={
            "click.option": click.Option(
                ["--ignore-load-errors", "-i"],
                is_flag=True,
                help="Ignore errors when loading environments especially when using --recursive or --all.",
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
    def from_dict(cls, d: Dict[str, Any]) -> "DeployArguments":
        return cls(**d)

    @classmethod
    def options(cls) -> List[click.Option]:
        """
        Return the set of base parameters added to every flyte run workflow subcommand.
        """
        return [common.get_option_from_metadata(f.metadata) for f in fields(cls) if f.metadata]


class DeployEnvCommand(click.RichCommand):
    def __init__(self, env_name: str, env: Any, deploy_args: DeployArguments, *args, **kwargs):
        self.env_name = env_name
        self.env = env
        self.deploy_args = deploy_args
        super().__init__(*args, **kwargs)

    def invoke(self, ctx: click.Context):
        console = common.get_console()
        console.print(f"Deploying root - environment: {self.env_name}")
        obj: CLIConfig = ctx.obj
        obj.init(
            self.deploy_args.project,
            self.deploy_args.domain,
            root_dir=self.deploy_args.root_dir,
            sync_local_sys_paths=not self.deploy_args.no_sync_local_sys_paths,
        )
        with console.status("Deploying...", spinner="dots"):
            deployment = flyte.deploy(
                self.env,
                dryrun=self.deploy_args.dry_run,
                copy_style=self.deploy_args.copy_style,
                version=self.deploy_args.version,
            )

        console.print(common.format("Environments", deployment[0].env_repr(), obj.output_format))
        console.print(common.format("Entities", deployment[0].table_repr(), obj.output_format))


class DeployEnvRecursiveCommand(click.Command):
    """
    Command to deploy all loaded environments in a directory or a file, optionally recursively.
    This command will load all python files in the directory, and deploy all environments found in them.
    If the path is a file, it will deploy all environments in that file.
    """

    def __init__(self, path: pathlib.Path, deploy_args: DeployArguments, *args, **kwargs):
        self.path = path
        self.deploy_args = deploy_args
        super().__init__(*args, **kwargs)

    def invoke(self, ctx: click.Context):
        from flyte._environment import list_loaded_environments
        from flyte._utils import load_python_modules

        obj: CLIConfig = ctx.obj
        console = common.get_console()

        # Load all python modules
        loaded_modules, failed_paths = load_python_modules(self.path, self.deploy_args.recursive)
        if failed_paths:
            console.print(f"Loaded {len(loaded_modules)} modules with, but failed to load {len(failed_paths)} paths:")
            console.print(
                common.format("Modules", [[("Path", p), ("Err", e)] for p, e in failed_paths], obj.output_format)
            )
        else:
            console.print(f"Loaded {len(loaded_modules)} modules")

        # Get newly loaded environments
        all_envs = list_loaded_environments()
        if not all_envs:
            console.print("No environments found to deploy")
            return
        console.print(common.format("Loaded Environments", [[("name", e.name)] for e in all_envs], obj.output_format))

        if not self.deploy_args.ignore_load_errors and len(failed_paths) > 0:
            raise click.ClickException(
                f"Failed to load {len(failed_paths)} files. Use --ignore-load-errors to ignore these errors."
            )
        # Now start connection and deploy all environments
        obj.init(
            self.deploy_args.project,
            self.deploy_args.domain,
            sync_local_sys_paths=not self.deploy_args.no_sync_local_sys_paths,
        )
        with console.status("Deploying...", spinner="dots"):
            deployments = flyte.deploy(
                *all_envs,
                dryrun=self.deploy_args.dry_run,
                copy_style=self.deploy_args.copy_style,
                version=self.deploy_args.version,
            )

        console.print(
            common.format("Environments", [env for d in deployments for env in d.env_repr()], obj.output_format)
        )
        console.print(common.format("Tasks", [task for d in deployments for task in d.table_repr()], obj.output_format))


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

    def list_commands(self, ctx):
        common.initialize_config(
            ctx,
            self.deploy_args.project,
            self.deploy_args.domain,
            self.deploy_args.root_dir,
            sync_local_sys_paths=not self.deploy_args.no_sync_local_sys_paths,
        )
        return super().list_commands(ctx)

    def get_command(self, ctx, obj_name):
        common.initialize_config(
            ctx,
            self.deploy_args.project,
            self.deploy_args.domain,
            self.deploy_args.root_dir,
            sync_local_sys_paths=not self.deploy_args.no_sync_local_sys_paths,
        )
        return super().get_command(ctx, obj_name)

    def _get_command_for_obj(self, ctx: click.Context, obj_name: str, obj: Any) -> click.Command:
        obj = cast(flyte.Environment, obj)
        return DeployEnvCommand(
            name=obj_name,
            env_name=obj_name,
            env=obj,
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
        directory: Path | None = None,
        **kwargs,
    ):
        if "params" not in kwargs:
            kwargs["params"] = []
        kwargs["params"].extend(DeployArguments.options())
        super().__init__(*args, directory=directory, **kwargs)

    def get_command(self, ctx, filename):
        deploy_args = DeployArguments.from_dict(ctx.params)
        fp = Path(filename)
        if not fp.exists():
            raise click.BadParameter(f"File {filename} does not exist")
        if deploy_args.recursive or deploy_args.all:
            # If recursive or all, we want to deploy all environments in the current directory
            return DeployEnvRecursiveCommand(
                path=fp,
                deploy_args=deploy_args,
                name=filename,
                help="Deploy all loaded environments from the file, or directory (optional recursively)",
            )
        if fp.is_dir():
            # If the path is a directory, we want to deploy all environments in that directory
            return EnvFiles(directory=fp)
        return EnvPerFileGroup(
            filename=fp,
            deploy_args=deploy_args,
            name=filename,
            help="Deploy a single environment and all its dependencies, from the file.",
        )


deploy = EnvFiles(
    name="deploy",
    help="""
    Deploy one or more environments from a python file.
    This command will create or update environments in the Flyte system.
    """,
)
