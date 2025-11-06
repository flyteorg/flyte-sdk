from __future__ import annotations

import importlib.util
import json
import logging
import os
import pathlib
import sys
from abc import abstractmethod
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Dict, Iterable, List, Literal, Optional

import rich.box
import rich.repr
import rich_click as click
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pretty_repr
from rich.table import Table
from rich.traceback import Traceback

import flyte.config
import flyte.errors
from flyte._logging import LogFormat
from flyte.config import Config

OutputFormat = Literal["table", "json", "table-simple", "json-raw"]

PREFERRED_BORDER_COLOR = "dim cyan"
PREFERRED_ACCENT_COLOR = "bold #FFD700"
HEADER_STYLE = f"{PREFERRED_ACCENT_COLOR} on black"

PROJECT_OPTION = click.Option(
    param_decls=["-p", "--project"],
    required=False,
    type=str,
    default=None,
    help="Project to which this command applies.",
    show_default=True,
)

DOMAIN_OPTION = click.Option(
    param_decls=["-d", "--domain"],
    required=False,
    type=str,
    default=None,
    help="Domain to which this command applies.",
    show_default=True,
)

DRY_RUN_OPTION = click.Option(
    param_decls=["--dry-run", "--dryrun"],
    required=False,
    type=bool,
    is_flag=True,
    default=False,
    help="Dry run. Do not actually call the backend service.",
    show_default=True,
)


def _common_options() -> List[click.Option]:
    """
    Common options that will be added to all commands and groups that inherit from CommandBase or GroupBase.
    """
    return [PROJECT_OPTION, DOMAIN_OPTION]


# This is global state for the CLI, it is manipulated by the main command
_client_secret_options = ["client-secret", "client_secret", "clientsecret", "app-credential", "app_credential"]
_device_flow_options = ["headless", "device-flow", "device_flow"]
_pkce_options = ["pkce"]
_external_command_options = ["external-command", "external_command", "externalcommand", "command", "custom"]
ALL_AUTH_OPTIONS = _client_secret_options + _device_flow_options + _pkce_options + _external_command_options


def sanitize_auth_type(auth_type: str | None) -> str:
    """
    Convert the auth type to the mode that is used by the Flyte backend.
    """
    if auth_type is None:
        return "pkce"
    if auth_type.lower() in _pkce_options:
        return "Pkce"
    if auth_type.lower() in _device_flow_options:
        return "DeviceFlow"
    if auth_type.lower() in _client_secret_options:
        return "ClientSecret"
    if auth_type.lower() in _external_command_options:
        return "ExternalCommand"
    raise ValueError(f"Unknown auth type: {auth_type}. Supported types are: {ALL_AUTH_OPTIONS}.")


@rich.repr.auto
@dataclass(frozen=True)
class CLIConfig:
    """
    This is the global state for the CLI. It is manipulated by the main command.
    """

    config: Config
    ctx: click.Context
    log_level: int | None = logging.ERROR
    log_format: LogFormat = "console"
    endpoint: str | None = None
    insecure: bool = False
    org: str | None = None
    auth_type: str | None = None
    output_format: OutputFormat = "table"

    def replace(self, **kwargs) -> CLIConfig:
        """
        Replace the global state with a new one.
        """
        return replace(self, **kwargs)

    def init(
        self,
        project: str | None = None,
        domain: str | None = None,
        root_dir: str | None = None,
        images: tuple[str, ...] | None = None,
        sync_local_sys_paths: bool = True,
    ):
        from flyte.config._config import TaskConfig

        task_cfg = TaskConfig(
            org=self.org or self.config.task.org,
            project=project if project is not None else self.config.task.project,
            domain=domain if domain is not None else self.config.task.domain,
        )

        kwargs: Dict[str, Any] = {}
        if self.endpoint:
            kwargs["endpoint"] = self.endpoint
        if self.insecure is not None:
            kwargs["insecure"] = self.insecure
        if self.auth_type:
            kwargs["auth_mode"] = sanitize_auth_type(self.auth_type)
        platform_cfg = self.config.platform.replace(**kwargs)

        updated_config = self.config.with_params(platform_cfg, task_cfg)

        flyte.init_from_config(
            updated_config,
            log_level=self.log_level,
            log_format=self.log_format,
            root_dir=pathlib.Path(root_dir) if root_dir else None,
            images=images,
            sync_local_sys_paths=sync_local_sys_paths,
        )


class InvokeBaseMixin:
    """
    Mixin to catch grpc.RpcError, flyte.RpcError, other errors and other exceptions
    and raise them as gclick.ClickException.
    """

    def invoke(self, ctx):
        import grpc

        try:
            return super().invoke(ctx)  # type: ignore
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.UNAUTHENTICATED:
                raise click.ClickException(f"Authentication failed. Please check your credentials. {e.details()}")
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise click.ClickException(f"Requested object NOT FOUND. Please check your input. Error: {e.details()}")
            if e.code() == grpc.StatusCode.ALREADY_EXISTS:
                raise click.ClickException("Resource already exists.")
            if e.code() == grpc.StatusCode.INTERNAL:
                raise click.ClickException(f"Internal server error: {e.details()}")
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                raise click.ClickException(
                    f"Service is currently unavailable. Please try again later. Error: {e.details()}"
                )
            if e.code() == grpc.StatusCode.PERMISSION_DENIED:
                raise click.ClickException(f"Permission denied. Please check your access rights. Error: {e.details()}")
            if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise click.ClickException(f"Invalid argument provided. Please check your input. Error: {e.details()}")
            raise click.ClickException(f"RPC error invoking command: {e!s}") from e
        except flyte.errors.InitializationError as e:
            raise click.ClickException(f"Initialization failed. Pass remote config for CLI. (Reason: {e})")
        except flyte.errors.BaseRuntimeError as e:
            raise click.ClickException(f"{e.kind} failure, {e.code}. {e}") from e
        except click.exceptions.Exit as e:
            # This is a normal exit, do nothing
            raise e
        except click.exceptions.NoArgsIsHelpError:
            # Do not raise an error if no arguments are passed, just show the help message.
            # https://github.com/pallets/click/pull/1489
            return None
        except Exception as e:
            if ctx.obj and ctx.obj.log_level and ctx.obj.log_level <= logging.DEBUG:
                # If the user has requested verbose output, print the full traceback
                console = get_console()
                console.print(Traceback.from_exception(type(e), e, e.__traceback__))
                exit(1)
            else:
                raise click.ClickException(f"Error invoking command: {e}") from e


class CommandBase(InvokeBaseMixin, click.RichCommand):
    """
    Base class for all commands, that adds common options to all commands if enabled.
    """

    common_options_enabled = True

    def __init__(self, *args, **kwargs):
        if "params" not in kwargs:
            kwargs["params"] = []
        if self.common_options_enabled:
            kwargs["params"].extend(_common_options())
        super().__init__(*args, **kwargs)


class GroupBase(InvokeBaseMixin, click.RichGroup):
    """
    Base class for all commands, that adds common options to all commands if enabled.
    """

    common_options_enabled = True

    def __init__(self, *args, **kwargs):
        if "params" not in kwargs:
            kwargs["params"] = []
        if self.common_options_enabled:
            kwargs["params"].extend(_common_options())
        super().__init__(*args, **kwargs)


class GroupBaseNoOptions(GroupBase):
    common_options_enabled = False


def get_option_from_metadata(metadata: MappingProxyType) -> click.Option:
    return metadata["click.option"]


def key_value_callback(_: Any, param: str, values: List[str]) -> Optional[Dict[str, str]]:
    """
    Callback for click to parse key-value pairs.
    """
    if not values:
        return None
    result = {}
    for v in values:
        if "=" not in v:
            raise click.BadParameter(f"Expected key-value pair of the form key=value, got {v}")
        k, v_ = v.split("=", 1)
        result[k.strip()] = v_.strip()
    return result


class ObjectsPerFileGroup(GroupBase):
    """
    Group that creates a command for each object in a python file.
    """

    def __init__(self, filename: Path | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if filename is None:
            raise ValueError("filename must be provided")
        if not filename.exists():
            raise click.ClickException(f"{filename} does not exists")
        self.filename = filename
        self._objs: Dict[str, Any] | None = None

    @abstractmethod
    def _filter_objects(self, module: ModuleType) -> Dict[str, Any]:
        """
        Filter the objects in the module to only include the ones we want to expose.
        """
        raise NotImplementedError

    @property
    def objs(self) -> Dict[str, Any]:
        if self._objs is not None:
            return self._objs

        module_name = os.path.splitext(os.path.basename(self.filename))[0]
        module_path = os.path.dirname(os.path.abspath(self.filename))

        spec = importlib.util.spec_from_file_location(module_name, self.filename)
        if spec is None or spec.loader is None:
            raise click.ClickException(f"Could not load module {module_name} from path [{self.filename}]")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        sys.path.append(module_path)
        spec.loader.exec_module(module)

        self._objs = self._filter_objects(module)
        if not self._objs:
            raise click.ClickException(f"No objects found in {self.filename}")
        return self._objs

    def list_commands(self, ctx):
        m = list(self.objs.keys())
        return sorted(m)

    @abstractmethod
    def _get_command_for_obj(self, ctx: click.Context, obj_name: str, obj: Any) -> click.Command: ...

    def get_command(self, ctx, obj_name):
        obj = self.objs[obj_name]
        return self._get_command_for_obj(ctx, obj_name, obj)


class FileGroup(GroupBase):
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
        super().__init__(*args, **kwargs)
        self._files = None
        self._dir = directory

    @property
    def files(self):
        if self._files is None:
            directory = self._dir or Path(".").absolute()
            # add python files
            _files = [os.fspath(p) for p in directory.glob("*.py") if p.name != "__init__.py"]

            # add directories
            _files.extend(
                [
                    os.fspath(directory / p.name)
                    for p in directory.iterdir()
                    if not p.name.startswith(("_", ".")) and p.is_dir()
                ]
            )

            # files that are in the current directory or subdirectories of the
            # current directory should be displayed as relative paths
            self._files = [
                str(Path(f).relative_to(Path.cwd())) if Path(f).is_relative_to(Path.cwd()) else f for f in _files
            ]
        return self._files

    def list_commands(self, ctx):
        return self.files

    def get_command(self, ctx, filename):
        raise NotImplementedError


def _table_format(table: Table, vals: Iterable[Any]) -> Table:
    headers = None
    has_rich_repr = False
    for p in vals:
        if hasattr(p, "__rich_repr__"):
            has_rich_repr = True
        elif not isinstance(p, (list, tuple)):
            raise ValueError("Expected a list or tuple of values, or an object with __rich_repr__ method.")
        o = list(p.__rich_repr__()) if has_rich_repr else p
        if headers is None:
            headers = [k for k, _ in o]
            for h in headers:
                table.add_column(h.capitalize(), no_wrap=True if "name" in h.casefold() else False)
        table.add_row(*[str(v) for _, v in o])
    return table


def format(title: str, vals: Iterable[Any], of: OutputFormat = "table") -> Table | Any:
    """
    Get a table from a list of values.
    """
    match of:
        case "table-simple":
            return _table_format(Table(title, box=None), vals)
        case "table":
            return _table_format(
                Table(
                    title=title,
                    box=rich.box.SQUARE_DOUBLE_HEAD,
                    header_style=HEADER_STYLE,
                    show_header=True,
                    border_style=PREFERRED_BORDER_COLOR,
                    expand=True,
                ),
                vals,
            )
        case "json":
            if not vals:
                return pretty_repr([])
            return pretty_repr([v.to_dict() for v in vals])
        case "json-raw":
            if not vals:
                return []
            return json.dumps([v.to_dict() for v in vals])

    raise click.ClickException("Unknown output format. Supported formats are: table, table-simple, json.")


def get_panel(title: str, renderable: Any, of: OutputFormat = "table") -> Panel:
    """
    Get a panel from a list of values.
    """
    if of in ["table-simple", "json"]:
        return renderable
    return Panel.fit(
        renderable,
        title=f"[{PREFERRED_ACCENT_COLOR}]{title}[/{PREFERRED_ACCENT_COLOR}]",
        border_style=PREFERRED_BORDER_COLOR,
    )


def get_console() -> Console:
    """
    Get a console that is configured to use colors if the terminal supports it.
    """
    return Console(color_system="auto", force_terminal=True, width=120)


def parse_images(cfg: Config, values: tuple[str, ...] | None) -> None:
    """
    Parse image values and update the config.

    Args:
        cfg: The Config object to write images to
        values: List of image strings in format "imagename=imageuri" or just "imageuri"
    """
    if values is None:
        return
    for value in values:
        if "=" in value:
            image_name, image_uri = value.split("=", 1)
            cfg.image.image_refs[image_name] = image_uri
        else:
            # If no name specified, use "default" as the name
            cfg.image.image_refs["default"] = value


@lru_cache()
def initialize_config(
    ctx: click.Context,
    project: str,
    domain: str,
    root_dir: str | None = None,
    images: tuple[str, ...] | None = None,
    sync_local_sys_paths: bool = True,
):
    obj: CLIConfig | None = ctx.obj
    if obj is None:
        import flyte.config

        obj = CLIConfig(flyte.config.auto(), ctx)

    obj.init(project, domain, root_dir, images, sync_local_sys_paths)
    return obj
