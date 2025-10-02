from dataclasses import dataclass, field, fields
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, cast

import rich_click as click

import flyte

from . import _common as common
from ._common import CLIConfig


@dataclass
class BuildArguments:
    noop: bool = field(
        default=False,
        metadata={
            "click.option": click.Option(
                ["--noop"],
                type=bool,
                help="Dummy parameter, placeholder for future use. Does not affect the build process.",
            )
        },
    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BuildArguments":
        return cls(**d)

    @classmethod
    def options(cls) -> List[click.Option]:
        """
        Return the set of base parameters added to every flyte run workflow subcommand.
        """
        return [common.get_option_from_metadata(f.metadata) for f in fields(cls) if f.metadata]


class BuildEnvCommand(click.Command):
    def __init__(self, obj_name: str, obj: Any, build_args: BuildArguments, *args, **kwargs):
        self.obj_name = obj_name
        self.obj = obj
        self.build_args = build_args
        super().__init__(*args, **kwargs)

    def invoke(self, ctx: click.Context):
        console = common.get_console()
        console.print(f"Building Environment: {self.obj_name}")
        obj: CLIConfig = ctx.obj
        obj.init()
        with console.status("Building...", spinner="dots"):
            image_cache = flyte.build_images(self.obj)

        console.print(common.format("Images", image_cache.repr(), obj.output_format))


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
        **kwargs,
    ):
        if "params" not in kwargs:
            kwargs["params"] = []
        kwargs["params"].extend(BuildArguments.options())
        super().__init__(*args, **kwargs)

    def get_command(self, ctx, filename):
        build_args = BuildArguments.from_dict(ctx.params)
        return EnvPerFileGroup(
            filename=Path(filename),
            build_args=build_args,
            name=filename,
            help=f"Run, functions decorated `env.task` or instances of Tasks in {filename}",
        )


build = EnvFiles(
    name="build",
    help="""
    Build the environments defined in a python file or directory. This will build the images associated with the
    environments.
    """,
)
