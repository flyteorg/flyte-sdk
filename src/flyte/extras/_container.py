import os
import pathlib
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from flyteidl2.core import tasks_pb2

from flyte import Image, storage
from flyte._logging import logger
from flyte._task import TaskTemplate
from flyte.io import Dir, File
from flyte.models import NativeInterface, SerializationContext


def _extract_command_key(cmd: str, **kwargs) -> List[Any] | None:
    """
    Extract the key from the command using regex.
    """
    import re

    input_regex = r"\{\{\.inputs\.([a-zA-Z0-9_]+)\}\}"
    return re.findall(input_regex, cmd)


def _extract_path_command_key(cmd: str, input_data_dir: Optional[str]) -> Optional[str]:
    """
    Extract the key from the path-like command using regex.
    """
    import re

    input_data_dir = input_data_dir or ""
    input_regex = rf"{re.escape(input_data_dir)}/([\w\-.]+)"  # captures file or dir names

    match = re.search(input_regex, cmd)
    if match:
        return match.group(1)
    return None


class ContainerTask(TaskTemplate):
    """
    This is an intermediate class that represents Flyte Tasks that run a container at execution time. This is the vast
    majority of tasks - the typical ``@task`` decorated tasks; for instance, all run a container. An example of
    something that doesn't run a container would be something like the Athena SQL task.

    :param name: Name of the task
    :param image: The container image to use for the task. This can be a string or an Image object.
    :param command: The command to run in the container. This can be a list of strings or a single string.
    :param inputs: The inputs to the task. This is a dictionary of input names to types.
    :param arguments: The arguments to pass to the command. This is a list of strings.
    :param outputs: The outputs of the task. This is a dictionary of output names to types.
    :param input_data_dir: The directory where the input data is stored. This is a string or a Path object.
    :param output_data_dir: The directory where the output data is stored. This is a string or a Path object.
    :param metadata_format: The format of the output file. This can be "JSON", "YAML", or "PROTO".
    :param local_logs: If True, logs will be printed to the console in the local execution.
    """

    MetadataFormat = Literal["JSON", "YAML", "PROTO"]

    def __init__(
        self,
        name: str,
        image: Union[str, Image],
        command: List[str],
        inputs: Optional[Dict[str, Type]] = None,
        arguments: Optional[List[str]] = None,
        outputs: Optional[Dict[str, Type]] = None,
        input_data_dir: str | pathlib.Path = "/var/inputs",
        output_data_dir: str | pathlib.Path = "/var/outputs",
        metadata_format: MetadataFormat = "JSON",
        local_logs: bool = True,
        **kwargs,
    ):
        super().__init__(
            task_type="raw-container",
            name=name,
            image=image,
            interface=NativeInterface({k: (v, None) for k, v in inputs.items()} if inputs else {}, outputs or {}),
            **kwargs,
        )
        self._image = image
        if isinstance(image, str):
            if image == "auto":
                self._image = Image.from_debian_base()
            else:
                self._image = Image.from_base(image)

        if command and any(not isinstance(c, str) for c in command):
            raise ValueError("All elements in the command list must be strings.")
        if arguments and any(not isinstance(a, str) for a in arguments):
            raise ValueError("All elements in the arguments list must be strings.")
        self._cmd = command
        self._args = arguments
        self._input_data_dir = input_data_dir
        if isinstance(input_data_dir, str):
            self._input_data_dir = pathlib.Path(input_data_dir)
        self._output_data_dir = output_data_dir
        if isinstance(output_data_dir, str):
            self._output_data_dir = pathlib.Path(output_data_dir)
        self._metadata_format = metadata_format
        self._inputs = inputs
        self._outputs = outputs
        self.local_logs = local_logs

    def _render_command_and_volume_binding(self, cmd: str, **kwargs) -> Tuple[str, Dict[str, Dict[str, str]]]:
        """
        We support template-style references to inputs, e.g., "{{.inputs.infile}}".

        For FlyteFile and FlyteDirectory commands, e.g., "/var/inputs/inputs", we extract the key from strings that
         begin with the specified `input_data_dir`.
        """
        from flyte.io import Dir, File

        volume_binding: Dict[str, Dict[str, str]] = {}
        path_k = _extract_path_command_key(cmd, str(self._input_data_dir))
        keys = [path_k] if path_k else _extract_command_key(cmd)

        command = cmd

        if keys:
            for k in keys:
                input_val = kwargs.get(k)
                # TODO: Add support file and directory transformer first
                if input_val and type(input_val) in [File, Dir]:
                    if not path_k:
                        raise AssertionError(
                            "File and Directory commands should not use the template syntax "
                            "like this: {{.inputs.infile}}\n"
                            "Please use a path-like syntax, such as: /var/inputs/infile.\n"
                            "This requirement is due to how Flyte Propeller processes template syntax inputs."
                        )
                    local_flyte_file_or_dir_path = input_val.path
                    remote_flyte_file_or_dir_path = os.path.join(self._input_data_dir, k)  # type: ignore
                    volume_binding[local_flyte_file_or_dir_path] = {
                        "bind": remote_flyte_file_or_dir_path,
                        "mode": "rw",
                    }
                else:
                    command = command.replace(f"{{{{.inputs.{k}}}}}", str(input_val))
        else:
            command = cmd

        return command, volume_binding

    def _prepare_command_and_volumes(
        self, cmd_and_args: List[str], **kwargs
    ) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
        """
        Prepares the command and volume bindings for the container based on input arguments and command templates.

        Parameters:
        - cmd_and_args (List[str]): The command and arguments to prepare.
        - **kwargs: Keyword arguments representing task inputs.

        Returns:
        - Tuple[List[str], Dict[str, Dict[str, str]]]: A tuple containing the prepared commands and volume bindings.
        """

        commands = []
        volume_bindings = {}

        for cmd in cmd_and_args:
            command, volume_binding = self._render_command_and_volume_binding(cmd, **kwargs)
            commands.append(command)
            volume_bindings.update(volume_binding)

        return commands, volume_bindings

    def _pull_image_if_not_exists(self, client, image: str):
        try:
            if not client.images.list(filters={"reference": image}):
                logger.info(f"Pulling image: {image} for container task: {self.name}")
                client.images.pull(image)
        except Exception as e:
            logger.error(f"Failed to pull image {image}: {e!s}")
            raise

    def _string_to_timedelta(self, s: str):
        import datetime
        import re

        regex = r"(?:(\d+) days?, )?(?:(\d+):)?(\d+):(\d+)(?:\.(\d+))?"
        parts = re.match(regex, s)
        if not parts:
            raise ValueError("Invalid timedelta string format")

        days = int(parts.group(1)) if parts.group(1) else 0
        hours = int(parts.group(2)) if parts.group(2) else 0
        minutes = int(parts.group(3)) if parts.group(3) else 0
        seconds = int(parts.group(4)) if parts.group(4) else 0
        microseconds = int(parts.group(5)) if parts.group(5) else 0

        return datetime.timedelta(
            days=days,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
        )

    async def _convert_output_val_to_correct_type(
        self, output_path: pathlib.Path, output_val: Any, output_type: Type
    ) -> Any:
        import datetime

        if issubclass(output_type, bool):
            return output_val.lower() != "false"
        elif issubclass(output_type, datetime.datetime):
            return datetime.datetime.fromisoformat(output_val)
        elif issubclass(output_type, datetime.timedelta):
            return self._string_to_timedelta(output_val)
        elif issubclass(output_type, File):
            return await File.from_local(output_path)
        elif issubclass(output_type, Dir):
            return await Dir.from_local(output_path)
        else:
            return output_type(output_val)

    async def _get_output(self, output_directory: pathlib.Path) -> Tuple[Any]:
        output_items = []
        if self._outputs:
            for k, output_type in self._outputs.items():
                output_path = output_directory / k
                if os.path.isfile(output_path):
                    with output_path.open("r") as f:
                        output_val = f.read()
                else:
                    output_val = None
                parsed = await self._convert_output_val_to_correct_type(output_path, output_val, output_type)
                output_items.append(parsed)
        # return a tuple so that each element is treated as a separate output.
        # this allows flyte to map the user-defined output types (dict) to individual values.
        # if we returned a list instead, it would be treated as a single output.
        return tuple(output_items)

    async def execute(self, **kwargs) -> Any:
        try:
            import docker
        except ImportError:
            raise ImportError("Docker is not installed. Please install Docker by running `pip install docker`.")

        # Normalize the input and output directories
        self._input_data_dir = os.path.normpath(self._input_data_dir) if self._input_data_dir else ""
        self._output_data_dir = os.path.normpath(self._output_data_dir) if self._output_data_dir else ""

        output_directory = storage.get_random_local_directory()
        cmd_and_args = (self._cmd or []) + (self._args or [])
        commands, volume_bindings = self._prepare_command_and_volumes(cmd_and_args, **kwargs)
        volume_bindings[str(output_directory)] = {"bind": self._output_data_dir, "mode": "rw"}

        client = docker.from_env()
        if isinstance(self._image, str):
            raise AssertionError(f"Only Image objects are supported, not strings. Got {self._image} instead.")
        uri = self._image.uri
        self._pull_image_if_not_exists(client, uri)
        print(f"Command: {commands!r}")

        container = client.containers.run(uri, command=commands, remove=True, volumes=volume_bindings, detach=True)

        # Wait for the container to finish the task
        # TODO: Add a 'timeout' parameter to control the max wait time for the container to finish the task.

        if self.local_logs:
            for log in container.logs(stream=True):
                print(f"[Local Container] {log.strip()!r}")

        container.wait()

        output = await self._get_output(output_directory)
        return output

    def data_loading_config(self, sctx: SerializationContext) -> tasks_pb2.DataLoadingConfig:
        literal_to_protobuf = {
            "JSON": tasks_pb2.DataLoadingConfig.JSON,
            "YAML": tasks_pb2.DataLoadingConfig.YAML,
            "PROTO": tasks_pb2.DataLoadingConfig.PROTO,
        }

        return tasks_pb2.DataLoadingConfig(
            input_path=str(self._input_data_dir) if self._input_data_dir else None,
            output_path=str(self._output_data_dir) if self._output_data_dir else None,
            enabled=True,
            format=literal_to_protobuf.get(self._metadata_format, "JSON"),
        )

    def container_args(self, sctx: SerializationContext) -> List[str]:
        return self._cmd + (self._args if self._args else [])
