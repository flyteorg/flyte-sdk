import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, Union, runtime_checkable

from mashumaro.codecs.json import JSONEncoder

from flyte.app.models import AppSerializationSettings

# TODO: Add more protocols here when we deploy to another cloud providers
SUPPORTED_FS_PROTOCOLS = ["s3://", "gs://", "union://", "ufs://", "unionmeta://", "ums://", "abfs://"]

_PACKAGE_NAME_RE = re.compile(r"^[\w-]+")
_UNION_RUNTIME_NAME = "union-runtime"


def _is_union_runtime(package: str) -> bool:
    """Return True if `package` is union-runtime."""
    m = _PACKAGE_NAME_RE.match(package)
    if not m:
        return False
    name = m.group()
    return name == _UNION_RUNTIME_NAME


def _convert_union_runtime_to_serverless(packages: Optional[list[str]]) -> Optional[list[str]]:
    """Convert packages using union-runtime to union-runtime[serverless]"""
    if packages is None:
        return None

    union_runtime_length = len(_UNION_RUNTIME_NAME)
    new_packages = []
    for package in packages:
        if _is_union_runtime(package):
            version_spec = package[union_runtime_length:]
            new_packages.append(f"union-runtime[serverless]{version_spec}")
        else:
            new_packages.append(package)
    return new_packages


@dataclass
class URLQuery:
    name: str
    public: bool = False


ENV_NAME_RE = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")
APP_NAME_RE = re.compile(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*")


def _has_file_extension(file_path) -> bool:
    _, ext = os.path.splitext(file_path)
    return bool(ext)


@dataclass
class Input:
    """
    Input for application.

    :param name: Name of input.
    :param value: Value for input.
    :param env_var: Environment name to set the value in the serving environment.
    :param download: When True, the input will be automatically downloaded. This
        only works if the value refers to an item in a object store. i.e. `s3://...`
    :param mount: If `value` is a directory, then the directory will be available
        at `mount`. If `value` is a file, then the file will be downloaded into the
        `mount` directory.
    :param ignore_patterns: If `value` is a directory, then this is a list of glob
        patterns to ignore.
    """

    class Type(Enum):
        File = "file"
        Directory = "directory"
        String = "string"
        _UrlQuery = "url_query"  # Private type, users should not need this

    value: Union[str, URLQuery]
    name: Optional[str] = None
    env_var: Optional[str] = None
    type: Optional[Type] = None
    download: bool = False
    mount: Optional[str] = None
    ignore_patterns: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.env_var is not None and ENV_NAME_RE.match(self.env_var) is None:
            msg = f"env_var ({self.env_var}) is not a valid environment name for shells"
            raise ValueError(msg)

        if self.name is None:
            if isinstance(self.value, URLQuery):
                self.name = self.value.name
            elif isinstance(self.value, str):
                self.name = self.value
            else:
                msg = "If name is not provided, then the Input value must be an URLQuery, or str"
                raise ValueError(msg)


@dataclass
class MaterializedInput:
    value: str
    type: Optional[Input.Type] = None


@dataclass
class InputBackend:
    """
    Input information for the backend.
    """

    name: str
    value: str
    download: bool
    type: Optional[Input.Type] = None
    env_var: Optional[str] = None
    dest: Optional[str] = None
    ignore_patterns: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.type is None:
            if any(self.value.startswith(proto) for proto in SUPPORTED_FS_PROTOCOLS):
                if _has_file_extension(self.value):
                    self.type = Input.Type.File
                else:
                    self.type = Input.Type.Directory
            else:
                self.type = Input.Type.String

        if self.type == Input.Type.String:
            # Nothing to download for a string
            self.download = False

        # If the type is a file or directory and there is a destination, then we
        # automatically assume it is going to be downloaded
        # TODO: In the future, we may mount this, so there is no need to download it
        # with the runtime.
        if self.type in (Input.Type.File, Input.Type.Directory) and self.dest is not None:
            self.download = True

    @classmethod
    def from_input(cls, user_input: Input, settings: AppSerializationSettings) -> "InputBackend":
        if isinstance(user_input.value, str):
            value = user_input.value
            input_type = user_input.type
        else:
            try:
                materialized_input = settings.materialized_inputs[user_input.name]
                value = materialized_input.value
                input_type = materialized_input.type or user_input.type
            except KeyError:
                msg = f"Did not materialize {user_input.name}"
                raise ValueError(msg)

        return InputBackend(
            name=user_input.name,
            value=value,
            download=user_input.download,
            env_var=user_input.env_var,
            type=input_type,
            dest=user_input.mount,
            ignore_patterns=user_input.ignore_patterns,
        )


@dataclass
class ServeConfig:
    """
    Configuration for serve runtime.

    :param code_uri: Location of user code in an object store (s3://...)
    :param user_inputs: User inputs. Passed in by `app.inputs`
    """

    code_uri: str  # location of user code
    inputs: List[InputBackend]


SERVE_CONFIG_ENCODER = JSONEncoder(ServeConfig)


@dataclass
class ResolvedInclude:
    src: str
    dest: str


@runtime_checkable
class AppConfigProtocol(Protocol):
    def before_to_union_idl(self, app: "App", settings: AppSerializationSettings):
        """Modify app in place at the beginning of `App._to_union_idl`."""





