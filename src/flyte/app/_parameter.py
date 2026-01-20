from __future__ import annotations

import os
import re
import typing
from dataclasses import dataclass, field
from functools import cache, cached_property
from typing import TYPE_CHECKING, List, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, model_validator

import flyte.io
from flyte._initialize import requires_initialization
from flyte._logging import logger

if TYPE_CHECKING:
    from flyte.remote._task import AutoVersioning
else:
    AutoVersioning = Literal["latest", "current"]


ParameterTypes: TypeAlias = Union[str, flyte.io.File, flyte.io.Dir, "AppEndpoint"]
_SerializedParameterType = Literal["string", "file", "directory", "app_endpoint"]

RUNTIME_PARAMETERS_FILE = "flyte-parameters.json"


class _DelayedValue(BaseModel):
    """
    Delayed value for app parameters.
    """

    type: _SerializedParameterType

    @model_validator(mode="before")
    @classmethod
    def check_type(cls, data: typing.Any) -> typing.Any:
        if "type" in data:
            data["type"] = PARAMETER_TYPE_MAP.get(data["type"], data["type"])
        return data

    async def get(self) -> str | flyte.io.File | flyte.io.Dir | AppEndpoint:
        value = await self.materialize()
        assert isinstance(value, (str, flyte.io.File, flyte.io.Dir, AppEndpoint)), (
            f"Materialized value must be a string, file, directory or app endpoint, found {type(value)}"
        )
        return value

    async def materialize(self) -> ParameterTypes:
        raise NotImplementedError("Subclasses must implement this method")


class RunOutput(_DelayedValue):
    """
    Use a run's output for app parameters.

    This enables the declaration of an app parameter dependency on the output of
    a run, given by a specific run name, or a task name and version. If
    `task_auto_version == 'latest'`, the latest version of the task will be used.
    If `task_auto_version == 'current'`, the version will be derived from the callee
    app or task context. To get the latest task run for ephemeral task runs, set
    `task_version` and `task_auto_version` should both be set to `None` (which is the default).

    Examples:

    Get the output of a specific run:

    ```python
    run_output = RunOutput(type="directory", run_name="my-run-123")
    ```

    Get the latest output of an ephemeral task run:

    ```python
    run_output = RunOutput(type="file", task_name="env.my_task")
    ```

    Get the latest output of a deployed task run:

    ```python
    run_output = RunOutput(type="file", task_name="env.my_task", task_auto_version="latest")
    ```

    Get the output of a specific task run:

    ```python
    run_output = RunOutput(type="file", task_name="env.my_task", task_version="xyz")
    ```
    """

    run_name: str | None = None
    task_name: str | None = None
    task_version: str | None = None
    task_auto_version: AutoVersioning | None = None
    getter: tuple[typing.Any, ...] = (0,)

    def __post_init__(self):
        if self.run_name is None and self.task_name is None:
            raise ValueError("Either run_name or task_name must be provided")
        if self.run_name is not None and self.task_name is not None:
            raise ValueError("Only one of run_name or task_name must be provided")
        if self.type == "app_endpoint":
            raise ValueError("AppEndpoint is not supported as a run output")

    @requires_initialization
    async def materialize(self) -> ParameterTypes:
        if self.run_name is not None:
            return await self._materialize_with_run_name()
        elif self.task_name is not None:
            return await self._materialize_with_task_name()
        else:
            raise ValueError("Either run_name or task_name must be provided")

    async def _materialize_with_task_name(self) -> ParameterTypes:
        import flyte.errors
        from flyte.remote import Run, RunDetails, Task, TaskDetails

        assert self.task_name is not None, "task_name must be provided"
        if self.task_auto_version is not None:
            task_details: TaskDetails = await Task.get(
                self.task_name, version=self.task_version, auto_version=self.task_auto_version
            ).fetch.aio()
            task_version = task_details.version
        elif self.task_version is not None:
            task_version = self.task_version
        else:
            task_version = None

        runs = Run.listall.aio(
            in_phase=("succeeded",),
            task_name=self.task_name,
            task_version=task_version,
            limit=1,
            sort_by=("created_at", "desc"),
        )
        try:
            run = await anext(runs)
            run_details: RunDetails = await run.details.aio()
            output = await run_details.outputs()
            for getter in self.getter:
                output = output[getter]
            logger.debug("Materialized output: %s", output)
            return typing.cast(ParameterTypes, output)
        except StopAsyncIteration:
            raise flyte.errors.ParameterMaterializationError(f"No runs found for task {self.task_name}")
        except Exception as e:
            raise flyte.errors.ParameterMaterializationError(
                f"Failed to materialize output for task {self.task_name}"
            ) from e

    async def _materialize_with_run_name(self) -> ParameterTypes:
        from flyte.remote import Run, RunDetails

        run: Run = await Run.get.aio(self.run_name)
        run_details: RunDetails = await run.details.aio()
        output = await run_details.outputs()
        for getter in self.getter:
            output = output[getter]
        return typing.cast(ParameterTypes, output)


class AppEndpoint(_DelayedValue):
    """
    Embed an upstream app's endpoint as an app parameter.

    This enables the declaration of an app parameter dependency on a the endpoint of
    an upstream app, given by a specific app name. This gives the app access to
    the upstream app's endpoint as a public or private url.
    """

    app_name: str
    public: bool = False
    type: Literal["string"] = "string"

    async def materialize(self) -> AppEndpoint:
        """Returns the AppEndpoint object, the endpoint is retrieved at serving time by the fserve executable."""
        return self

    @requires_initialization
    async def _retrieve_endpoint(self) -> str:
        """Get the endpoint of the specified app at serving time."""
        from flyte.app._app_environment import INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR

        if self.public:
            from flyte.remote import App

            app = await App.get.aio(self.app_name)
            return app.endpoint

        endpoint_pattern = os.getenv(INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR)
        if endpoint_pattern is not None:
            return endpoint_pattern.format(app_fqdn=self.app_name)

        raise ValueError(
            f"Environment variable {INTERNAL_APP_ENDPOINT_PATTERN_ENV_VAR} is not set to create a private url."
        )


PARAMETER_TYPE_MAP = {
    str: "string",
    flyte.io.File: "file",
    flyte.io.Dir: "directory",
    AppEndpoint: "app_endpoint",
}


@dataclass
class Parameter:
    """
    Parameter for application.

    :param name: Name of parameter.
    :param value: Value for parameter.
    :param env_var: Environment name to set the value in the serving environment.
    :param download: When True, the parameter will be automatically downloaded. This
        only works if the value refers to an item in a object store. i.e. `s3://...`
    :param mount: If `value` is a directory, then the directory will be available
        at `mount`. If `value` is a file, then the file will be downloaded into the
        `mount` directory.
    :param ignore_patterns: If `value` is a directory, then this is a list of glob
        patterns to ignore.
    """

    name: str
    value: ParameterTypes | _DelayedValue
    env_var: Optional[str] = None
    download: bool = False
    mount: Optional[str] = None
    ignore_patterns: list[str] = field(default_factory=list)

    def __post_init__(self):
        import flyte.io

        env_name_re = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")

        if self.env_var is not None and env_name_re.match(self.env_var) is None:
            raise ValueError(f"env_var ({self.env_var}) is not a valid environment name for shells")

        if self.value and not isinstance(self.value, (str, flyte.io.File, flyte.io.Dir, RunOutput, AppEndpoint)):
            raise TypeError(
                f"Expected value to be of type str, file, dir, RunOutput or AppEndpoint, got {type(self.value)}"
            )

        if self.name is None:
            self.name = "i0"


class SerializableParameter(BaseModel):
    """
    Serializable version of Parameter.
    """

    name: str
    value: str
    download: bool
    type: _SerializedParameterType = "string"
    env_var: Optional[str] = None
    dest: Optional[str] = None
    ignore_patterns: List[str] = field(default_factory=list)

    @classmethod
    def from_parameter(cls, param: Parameter) -> "SerializableParameter":
        import flyte.io

        # param.name is guaranteed to be set by Parameter.__post_init__
        assert param.name is not None, "Parameter name should be set by __post_init__"

        tpe: _SerializedParameterType = "string"
        if isinstance(param.value, flyte.io.File):
            value = param.value.path
            tpe = "file"
            download = True if param.mount is not None else param.download
        elif isinstance(param.value, flyte.io.Dir):
            value = param.value.path
            tpe = "directory"
            download = True if param.mount is not None else param.download
        elif isinstance(param.value, RunOutput):
            value = param.value.model_dump_json()
            tpe = param.value.type
            download = True if param.mount is not None else param.download
        elif isinstance(param.value, AppEndpoint):
            value = param.value.model_dump_json()
            tpe = "app_endpoint"
            download = False
        else:
            value = typing.cast(str, param.value)
            download = False

        return cls(
            name=param.name,
            value=value,
            type=tpe,
            download=download,
            env_var=param.env_var,
            dest=param.mount,
            ignore_patterns=param.ignore_patterns,
        )


class SerializableParameterCollection(BaseModel):
    """
    Collection of parameters for application.

    :param parameters: List of parameters.
    """

    parameters: List[SerializableParameter] = field(default_factory=list)

    @cached_property
    def to_transport(self) -> str:
        import base64
        import gzip
        from io import BytesIO

        json_str = self.model_dump_json()
        buf = BytesIO()
        with gzip.GzipFile(mode="wb", fileobj=buf, mtime=0) as f:
            f.write(json_str.encode("utf-8"))
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @classmethod
    def from_transport(cls, s: str) -> SerializableParameterCollection:
        import base64
        import gzip

        compressed_val = base64.b64decode(s.encode("utf-8"))
        json_str = gzip.decompress(compressed_val).decode("utf-8")
        return cls.model_validate_json(json_str)

    @classmethod
    def from_parameters(cls, parameters: List[Parameter]) -> SerializableParameterCollection:
        return cls(parameters=[SerializableParameter.from_parameter(param) for param in parameters])


@cache
def _load_parameters() -> dict[str, str]:
    """Load parameters for application or endpoint."""
    import json
    import os

    config_file = os.getenv(RUNTIME_PARAMETERS_FILE)

    if config_file is None:
        raise ValueError("Parameters are not mounted")

    with open(config_file, "r") as f:
        parameters = json.load(f)

    return parameters


def get_parameter(name: str) -> str:
    """Get parameters for application or endpoint."""
    parameters = _load_parameters()
    return parameters[name]
