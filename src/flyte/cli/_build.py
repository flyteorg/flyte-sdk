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
class BuildArguments:
    copy_style: CopyFiles = field(
        default="loaded_modules",
        metadata={
            "click.option": click.Option(
                ["--copy-style"],
                type=click.Choice(get_args(CopyFiles)),
                default="loaded_modules",
                help="Copy style of the eventual deploy. Must match the deploy's --copy-style "
                "so the image content hash — and therefore the registry tag — lines up.",
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
                help="Recursively build all environments in the current directory and its subdirectories.",
            )
        },
    )
    all: bool = field(
        default=False,
        metadata={
            "click.option": click.Option(
                ["--all"],
                is_flag=True,
                help="Build the images for all environments in the file or directory, ignoring the file name.",
            )
        },
    )
    ignore_load_errors: bool = field(
        default=False,
        metadata={
            "click.option": click.Option(
                ["--ignore-load-errors", "-i"],
                is_flag=True,
                help="Ignore errors when loading environments, especially when using --recursive or --all.",
            )
        },
    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BuildArguments":
        return cls(**d)

    @classmethod
    def options(cls) -> List[click.Option]:
        """
        Return the set of base parameters added to every flyte build subcommand.
        """
        return [common.get_option_from_metadata(f.metadata) for f in fields(cls) if f.metadata]


class BuildEnvCommand(click.Command):
    def __init__(self, obj_name: str, obj: Any, build_args: BuildArguments, *args, **kwargs):
        self.obj_name = obj_name
        self.obj = obj
        self.build_args = build_args
        super().__init__(*args, **kwargs)

    def invoke(self, ctx: click.Context):
        from flyte._status import status

        obj: CLIConfig = ctx.obj
        status.step(f"Building environment: {self.obj_name}")
        obj.init(root_dir=self.build_args.root_dir)
        with common.cli_status(obj.output_format, "Building...", spinner="dots", no_progress=obj.no_progress):
            image_cache = flyte.build_images(self.obj, copy_style=self.build_args.copy_style)

        status.success(f"Environment {self.obj_name} built")
        common.print_output(common.format("Images", image_cache.repr(), obj.output_format), obj.output_format)


class BuildEnvRecursiveCommand(click.Command):
    """
    Command to build the images for all loaded environments in a file or directory, optionally recursively.

    This mirrors ``flyte deploy --all`` (see ``DeployEnvRecursiveCommand``): it loads all python modules
    found at the path, collects every ``flyte.Environment`` instantiated at import time, and builds the
    images for all of them in a single planning pass.
    """

    def __init__(self, path: pathlib.Path, build_args: BuildArguments, *args, **kwargs):
        self.path = path
        self.build_args = build_args
        super().__init__(*args, **kwargs)

    def invoke(self, ctx: click.Context):
        from flyte._environment import list_loaded_environments
        from flyte._status import status
        from flyte._utils import load_python_modules

        obj: CLIConfig = ctx.obj
        obj.init(root_dir=self.build_args.root_dir)

        root_dir = Path.cwd()
        if self.build_args.root_dir:
            root_dir = pathlib.Path(self.build_args.root_dir).resolve()

        # Load all python modules so their Environments register themselves.
        loaded_modules, failed_paths = load_python_modules(self.path, root_dir, self.build_args.recursive)
        if failed_paths:
            status.warn(f"Loaded {len(loaded_modules)} modules, but failed to load {len(failed_paths)} paths")
            common.print_output(
                common.format("Modules", [[("Path", p), ("Err", e)] for p, e in failed_paths], obj.output_format),
                obj.output_format,
            )
        else:
            status.info(f"Loaded {len(loaded_modules)} modules")

        all_envs = list_loaded_environments()
        if not all_envs:
            status.info("No environments found to build")
            return

        common.print_output(
            common.format("Loaded Environments", [[("name", e.name)] for e in all_envs], obj.output_format),
            obj.output_format,
        )

        if not self.build_args.ignore_load_errors and len(failed_paths) > 0:
            raise click.ClickException(
                f"Failed to load {len(failed_paths)} files. Use --ignore-load-errors to ignore these errors."
            )

        with common.cli_status(obj.output_format, "Building...", spinner="dots", no_progress=obj.no_progress):
            image_cache = flyte.build_images(*all_envs, copy_style=self.build_args.copy_style)

        status.success(f"Built images for {len(all_envs)} environments")
        common.print_output(common.format("Images", image_cache.repr(), obj.output_format), obj.output_format)


class EnvPerFileGroup(common.ObjectsPerFileGroup):
    """
    Group that creates a command for each task in the current directory that is not `__init__.py`.
    """

    def __init__(self, filename: Path, build_args: BuildArguments, *args, **kwargs):
        args = (filename, *args)
        super().__init__(*args, **kwargs)
        self.build_args = build_args

    def _filter_objects(self, module: ModuleType) -> Dict[str, Any]:
        return {k: v for k, v in module.__dict__.items() if isinstance(v, flyte.Environment)}

    def _get_command_for_obj(self, ctx: click.Context, obj_name: str, obj: Any) -> click.Command:
        obj = cast(flyte.Environment, obj)
        return BuildEnvCommand(
            name=obj_name,
            obj_name=obj_name,
            obj=obj,
            help=f"{obj.name}" + (f": {obj.description}" if obj.description else ""),
            build_args=self.build_args,
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
        kwargs["params"].extend(BuildArguments.options())
        super().__init__(*args, directory=directory, **kwargs)

    def get_command(self, ctx, filename):
        build_args = BuildArguments.from_dict(ctx.params)
        fp = Path(filename)
        if not fp.exists():
            raise click.BadParameter(f"File {filename} does not exist")
        if build_args.recursive or build_args.all:
            # If recursive or all, build the images for every environment in the file/directory,
            # without naming one (mirrors `flyte deploy --all`).
            return BuildEnvRecursiveCommand(
                path=fp,
                build_args=build_args,
                name=filename,
                help="Build images for all loaded environments from the file, or directory (optional recursively)",
            )
        if fp.is_dir():
            # If the path is a directory, expose the files within it as subcommands.
            return EnvFiles(directory=fp)
        return EnvPerFileGroup(
            filename=fp,
            build_args=build_args,
            name=filename,
            help=f"Run, functions decorated `env.task` or instances of Tasks in {filename}",
        )


build = EnvFiles(
    name="build",
    help="""
    Build the environments defined in a python file or directory. This will build the images associated with the
    environments.

    To build the image for a single named environment:

    ```bash
    flyte build hello.py my_env
    ```

    To build the images for all environments in a file (without naming one), use the `--all` flag:

    ```bash
    flyte build --all hello.py
    ```

    To recursively build all environments in a directory and its subdirectories, use the `--recursive` flag:

    ```bash
    flyte build --all --recursive ./src
    ```
    """,
)
