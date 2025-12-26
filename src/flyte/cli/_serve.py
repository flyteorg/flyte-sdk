from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, cast

import rich_click as click
from typing_extensions import get_args

from flyte._code_bundle._utils import CopyFiles
from flyte.app import AppEnvironment

from . import _common as common
from ._common import CLIConfig


@dataclass
class ServeArguments:
    project: str = field(
        default=cast(str, common.PROJECT_OPTION.default), metadata={"click.option": common.PROJECT_OPTION}
    )
    domain: str = field(
        default=cast(str, common.DOMAIN_OPTION.default), metadata={"click.option": common.DOMAIN_OPTION}
    )
    copy_style: CopyFiles = field(
        default="loaded_modules",
        metadata={
            "click.option": click.Option(
                ["--copy-style"],
                type=click.Choice(get_args(CopyFiles)),
                default="loaded_modules",
                help="Copy style to use when serving the app",
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
                help="Name of the app deployment. If not provided, the app environment name will be used.",
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
                help="Wait and watch logs for the app. If not provided, the CLI will exit after "
                "successfully deploying the app with a link to the UI.",
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
                help="Image to be used in the serve. Format: imagename=imageuri. Can be specified multiple times.",
            )
        },
    )
    no_sync_local_sys_paths: bool = field(
        default=False,
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
    env_var: List[str] = field(
        default_factory=list,
        metadata={
            "click.option": click.Option(
                ["--env-var", "-e"],
                type=str,
                multiple=True,
                help="Environment variable to set in the app. Format: KEY=VALUE. Can be specified multiple times. "
                "Example: --env-var LOG_LEVEL=DEBUG --env-var DATABASE_URL=postgresql://...",
            )
        },
    )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ServeArguments:
        modified = {k: v for k, v in d.items() if k in {f.name for f in fields(cls)}}
        return cls(**modified)

    @classmethod
    def options(cls) -> List[click.Option]:
        """
        Return the set of base parameters added to serve subcommand.
        """
        return [common.get_option_from_metadata(f.metadata) for f in fields(cls) if f.metadata]


class ServeAppCommand(click.RichCommand):
    def __init__(self, obj_name: str, obj: AppEnvironment, serve_args: ServeArguments, *args, **kwargs):
        self.obj_name = obj_name
        self.obj = cast(AppEnvironment, obj)
        self.serve_args = serve_args
        kwargs.pop("name", None)
        super().__init__(obj_name, *args, **kwargs)

    def invoke(self, ctx: click.Context):
        obj: CLIConfig = common.initialize_config(
            ctx,
            self.serve_args.project,
            self.serve_args.domain,
            self.serve_args.root_dir,
            tuple(self.serve_args.image) or None,
            not self.serve_args.no_sync_local_sys_paths,
        )

        async def _serve():
            import flyte

            console = common.get_console()

            # Parse env vars from CLI (format: KEY=VALUE)
            env_vars = {}
            for env_var in self.serve_args.env_var:
                if "=" in env_var:
                    key, value = env_var.split("=", 1)
                    env_vars[key] = value
                else:
                    console.print(
                        f"[yellow]Warning: Ignoring invalid env var format: {env_var} (expected KEY=VALUE)[/yellow]"
                    )

            # Use with_servecontext to configure the serve operation
            app = await flyte.with_servecontext(
                copy_style=self.serve_args.copy_style,
                project=self.serve_args.project if self.serve_args.project else None,
                domain=self.serve_args.domain if self.serve_args.domain else None,
                env_vars=env_vars if env_vars else None,
            ).serve.aio(self.obj)

            console.print(
                common.get_panel(
                    "Serve",
                    f"[green bold]App '{app.name}' is now being served[/green bold]\n"
                    f"➡️  [blue bold][link={app.url}]{app.url}[/link][/blue bold]",
                    obj.output_format,
                )
            )

            if self.serve_args.follow:
                # TODO: Implement log streaming for apps
                # This should retrieve and display logs from the running app
                # Similar to how r.show_logs.aio() works for tasks in _run.py
                console.print(
                    "[yellow]Note: Log streaming for apps is not yet implemented. "
                    "Please check the app logs via the UI.[/yellow]"
                )

        asyncio.run(_serve())


class AppPerFileGroup(common.ObjectsPerFileGroup):
    """
    Group that creates a command for each AppEnvironment in the current directory that is not __init__.py.
    """

    def __init__(self, filename: Path, serve_args: ServeArguments, *args, **kwargs):
        if filename.is_absolute():
            filename = filename.relative_to(Path.cwd())
        super().__init__(*(filename, *args), **kwargs)
        self.serve_args = serve_args

    def _filter_objects(self, module: ModuleType) -> Dict[str, Any]:
        return {k: v for k, v in module.__dict__.items() if isinstance(v, AppEnvironment)}

    def list_commands(self, ctx):
        common.initialize_config(
            ctx,
            self.serve_args.project,
            self.serve_args.domain,
            self.serve_args.root_dir,
            sync_local_sys_paths=not self.serve_args.no_sync_local_sys_paths,
        )
        return super().list_commands(ctx)

    def get_command(self, ctx, obj_name):
        common.initialize_config(
            ctx,
            self.serve_args.project,
            self.serve_args.domain,
            self.serve_args.root_dir,
            sync_local_sys_paths=not self.serve_args.no_sync_local_sys_paths,
        )
        return super().get_command(ctx, obj_name)

    def _get_command_for_obj(self, ctx: click.Context, obj_name: str, obj: Any) -> click.Command:
        obj = cast(AppEnvironment, obj)
        return ServeAppCommand(
            obj_name=obj_name,
            obj=obj,
            help=f"Serve the '{obj_name}' app environment",
            serve_args=self.serve_args,
        )


class AppFiles(common.FileGroup):
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
        kwargs["params"].extend(ServeArguments.options())
        super().__init__(*args, directory=directory, **kwargs)

    def get_command(self, ctx, cmd_name):
        serve_args = ServeArguments.from_dict(ctx.params)

        fp = Path(cmd_name)
        if not fp.exists():
            raise click.BadParameter(f"File {cmd_name} does not exist")
        if fp.is_dir():
            return AppFiles(
                directory=fp,
                help=f"Serve `*.py` file inside the {fp} directory",
            )
        return AppPerFileGroup(
            filename=fp,
            serve_args=serve_args,
            name=cmd_name,
            help=f"Serve AppEnvironment instances in {cmd_name}",
        )


serve = AppFiles(
    name="serve",
    help="""
Serve an app from a Python file using flyte.serve().

This command allows you to serve apps defined with `flyte.app.AppEnvironment`
in your Python files. The serve command will deploy the app to the Flyte backend
and start it, making it accessible via a URL.

Example usage:

```bash
flyte serve examples/apps/basic_app.py app_env
```

Arguments to the serve command are provided right after the `serve` command and before the file name.

To follow the logs of the served app, use the `--follow` flag:

```bash
flyte serve --follow examples/apps/basic_app.py app_env
```

Note: Log streaming is not yet fully implemented and will be added in a future release.

You can provide image mappings with `--image` flag. This allows you to specify
the image URI for the app environment during CLI execution without changing
the code. Any images defined with `Image.from_ref_name("name")` will resolve to the
corresponding URIs you specify here.

```bash
flyte serve --image my_image=ghcr.io/myorg/my-image:v1.0 examples/apps/basic_app.py app_env
```

If the image name is not provided, it is regarded as a default image and will
be used when no image is specified in AppEnvironment:

```bash
flyte serve --image ghcr.io/myorg/default-image:latest examples/apps/basic_app.py app_env
```

You can specify multiple image arguments:

```bash
flyte serve --image ghcr.io/org/default:latest --image gpu=ghcr.io/org/gpu:v2.0 examples/apps/basic_app.py app_env
```

You can specify the `--config` flag to point to a specific Flyte cluster:

```bash
flyte serve --config my-config.yaml examples/apps/basic_app.py app_env
```

You can override the default configured project and domain:

```bash
flyte serve --project my-project --domain development examples/apps/basic_app.py app_env
```

Other arguments to the serve command are listed below.

Note: This pattern is primarily useful for serving apps defined in tasks.
Serving deployed apps is not currently supported through this CLI command.
""",
)
