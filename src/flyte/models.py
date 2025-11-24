from __future__ import annotations

import inspect
import os
import pathlib
import typing
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Literal, Optional, Tuple, Type

import rich.repr

from flyte._docstring import Docstring
from flyte._interface import extract_return_annotation, literal_to_enum
from flyte._logging import logger

if TYPE_CHECKING:
    from flyteidl2.core import literals_pb2

    from flyte._internal.imagebuild.image_builder import ImageCache
    from flyte.report import Report

# --- Constants ----
MAX_INLINE_IO_BYTES = 10 * 1024 * 1024  # 100 MB


def generate_random_name() -> str:
    """
    Generate a random name for the task. This is used to create unique names for tasks.
    TODO we can use unique-namer in the future, for now its just guids
    """
    from uuid import uuid4

    return str(uuid4())  # Placeholder for actual random name generation logic


@rich.repr.auto
@dataclass(frozen=True, kw_only=True)
class ActionID:
    """
    A class representing the ID of an Action, nested within a Run. This is used to identify a specific action on a task.
    """

    name: str
    run_name: str | None = None
    project: str | None = None
    domain: str | None = None
    org: str | None = None

    def __post_init__(self):
        if self.run_name is None:
            object.__setattr__(self, "run_name", self.name)

    @classmethod
    def create_random(cls):
        name = generate_random_name()
        return cls(name=name, run_name=name)

    def new_sub_action(self, name: str | None = None) -> ActionID:
        """
        Create a new sub-run with the given name. If  name is None, a random name will be generated.
        """
        if name is None:
            name = generate_random_name()
        return replace(self, name=name)

    def new_sub_action_from(self, task_call_seq: int, task_hash: str, input_hash: str, group: str | None) -> ActionID:
        """Make a deterministic name"""
        import hashlib

        from flyte._utils.helpers import base36_encode

        components = f"{self.name}-{input_hash}-{task_hash}-{task_call_seq}" + (f"-{group}" if group else "")
        logger.debug(f"----- Generating sub-action ID from components: {components}")
        # has the components into something deterministic
        bytes_digest = hashlib.md5(components.encode()).digest()
        new_name = base36_encode(bytes_digest)
        return self.new_sub_action(new_name)


@rich.repr.auto
@dataclass
class PathRewrite:
    """
    Configuration for rewriting paths during input loading.
    """

    # If set, rewrites any path starting with this prefix to the new prefix.
    old_prefix: str
    new_prefix: str

    def __post_init__(self):
        if not self.old_prefix or not self.new_prefix:
            raise ValueError("Both old_prefix and new_prefix must be non-empty strings.")
        if self.old_prefix == self.new_prefix:
            raise ValueError("old_prefix and new_prefix must be different.")

    @classmethod
    def from_str(cls, pattern: str) -> PathRewrite:
        """
        Create a PathRewrite from a string pattern of the form `old_prefix->new_prefix`.
        """
        parts = pattern.split("->")
        if len(parts) != 2:
            raise ValueError(f"Invalid path rewrite pattern: {pattern}. Expected format 'old_prefix->new_prefix'.")
        return cls(old_prefix=parts[0], new_prefix=parts[1])

    def __repr__(self) -> str:
        return f"{self.old_prefix}->{self.new_prefix}"


@rich.repr.auto
@dataclass(frozen=True, kw_only=True)
class RawDataPath:
    """
    A class representing the raw data path for a task. This is used to store the raw data for the task execution and
    also get mutations on the path.
    """

    path: str
    path_rewrite: Optional[PathRewrite] = None

    @classmethod
    def from_local_folder(cls, local_folder: str | pathlib.Path | None = None) -> RawDataPath:
        """
        Create a new context attribute object, with local path given. Will be created if it doesn't exist.
        :return: Path to the temporary directory
        """
        import tempfile

        match local_folder:
            case pathlib.Path():
                local_folder.mkdir(parents=True, exist_ok=True)
                return RawDataPath(path=str(local_folder))
            case None:
                # Create a temporary directory for data storage
                p = tempfile.mkdtemp()
                logger.debug(f"Creating temporary directory for data storage: {p}")
                pathlib.Path(p).mkdir(parents=True, exist_ok=True)
                return RawDataPath(path=p)
            case str():
                return RawDataPath(path=local_folder)
            case _:
                raise ValueError(f"Invalid local path {local_folder}")

    def get_random_remote_path(self, file_name: Optional[str] = None) -> str:
        """
        Returns a random path for uploading a file/directory to. This file/folder will not be created, it's just a path.

        :param file_name: If given, will be joined after a randomly generated portion.
        :return:
        """
        import random
        from uuid import UUID

        import fsspec
        from fsspec.utils import get_protocol

        random_string = UUID(int=random.getrandbits(128)).hex
        file_prefix = self.path

        protocol = get_protocol(file_prefix)
        if "file" in protocol:
            parent_folder = pathlib.Path(file_prefix)
            parent_folder.mkdir(exist_ok=True, parents=True)
            if file_name:
                random_folder = parent_folder / random_string
                random_folder.mkdir()
                local_path = random_folder / file_name
            else:
                local_path = parent_folder / random_string
            return str(local_path.absolute())

        fs = fsspec.filesystem(protocol)
        if file_prefix.endswith(fs.sep):
            file_prefix = file_prefix[:-1]
        remote_path = fs.sep.join([file_prefix, random_string])
        if file_name:
            remote_path = fs.sep.join([remote_path, file_name])
        return remote_path


@rich.repr.auto
@dataclass(frozen=True)
class GroupData:
    name: str


@rich.repr.auto
@dataclass(frozen=True, kw_only=True)
class TaskContext:
    """
    A context class to hold the current task executions context.
    This can be used to access various contextual parameters in the task execution by the user.

    :param action: The action ID of the current execution. This is always set, within a run.
    :param version: The version of the executed task. This is set when the task is executed by an action and will be
      set on all sub-actions.
    :param custom_context: Context metadata for the action. If an action receives context, it'll automatically pass it
      to any actions it spawns. Context will not be used for cache key computation.
    """

    action: ActionID
    version: str
    raw_data_path: RawDataPath
    input_path: str | None = None
    output_path: str
    run_base_dir: str
    report: Report
    group_data: GroupData | None = None
    checkpoints: Checkpoints | None = None
    code_bundle: CodeBundle | None = None
    compiled_image_cache: ImageCache | None = None
    data: Dict[str, Any] = field(default_factory=dict)
    mode: Literal["local", "remote", "hybrid"] = "remote"
    interactive_mode: bool = False
    custom_context: Dict[str, str] = field(default_factory=dict)

    def replace(self, **kwargs) -> TaskContext:
        if "data" in kwargs:
            rec_data = kwargs.pop("data")
            if rec_data is None:
                return replace(self, **kwargs)
            data = {}
            if self.data is not None:
                data = self.data.copy()
            data.update(rec_data)
            kwargs.update({"data": data})
        return replace(self, **kwargs)

    def __getitem__(self, key: str) -> Optional[Any]:
        return self.data.get(key)

    def is_in_cluster(self):
        """
        Check if the task is running in a cluster.
        :return: bool
        """
        return self.mode == "remote"


@rich.repr.auto
@dataclass(frozen=True, kw_only=True)
class CodeBundle:
    """
    A class representing a code bundle for a task. This is used to package the code and the inflation path.
    The code bundle computes the version of the code using the hash of the code.

    :param computed_version: The version of the code bundle. This is the hash of the code.
    :param destination: The destination path for the code bundle to be inflated to.
    :param tgz: Optional path to the tgz file.
    :param pkl: Optional path to the pkl file.
    :param downloaded_path: The path to the downloaded code bundle. This is only available during runtime, when
        the code bundle has been downloaded and inflated.
    """

    computed_version: str
    destination: str = "."
    tgz: str | None = None
    pkl: str | None = None
    downloaded_path: pathlib.Path | None = None

    # runtime_dependencies: Tuple[str, ...] = field(default_factory=tuple)  In the future if we want we could add this
    # but this messes up actors, spark etc

    def __post_init__(self):
        if self.tgz is None and self.pkl is None:
            raise ValueError("Either tgz or pkl must be provided")

    def with_downloaded_path(self, path: pathlib.Path) -> CodeBundle:
        """
        Create a new CodeBundle with the given downloaded path.
        """
        return replace(self, downloaded_path=path)


@rich.repr.auto
@dataclass(frozen=True)
class Checkpoints:
    """
    A class representing the checkpoints for a task. This is used to store the checkpoints for the task execution.
    """

    prev_checkpoint_path: str | None
    checkpoint_path: str | None


class _has_default:
    """
    A marker class to indicate that a specific input has a default value or not.
    This is used to determine if the input is required or not.
    """


@dataclass(frozen=True)
class NativeInterface:
    """
    A class representing the native interface for a task. This is used to interact with the task and its execution
    context.
    """

    inputs: Dict[str, Tuple[Type, Any]]
    outputs: Dict[str, Type]
    docstring: Optional[Docstring] = None

    # This field is used to indicate that the task has a default value for the input, but already in the
    # remote form.
    _remote_defaults: Optional[Dict[str, literals_pb2.Literal]] = field(default=None, repr=False)

    has_default: ClassVar[Type[_has_default]] = _has_default  # This can be used to indicate if a specific input

    # has a default value or not, in the case when the default value is not known. An example would be remote tasks.

    def has_outputs(self) -> bool:
        """
        Check if the task has outputs. This is used to determine if the task has outputs or not.
        """
        return self.outputs is not None and len(self.outputs) > 0

    def required_inputs(self) -> List[str]:
        """
        Get the names of the required inputs for the task. This is used to determine which inputs are required for the
        task execution.
        :return: A list of required input names.
        """
        return [k for k, v in self.inputs.items() if v[1] is inspect.Parameter.empty]

    def num_required_inputs(self) -> int:
        """
        Get the number of required inputs for the task. This is used to determine how many inputs are required for the
        task execution.
        """
        return sum(1 for t in self.inputs.values() if t[1] is inspect.Parameter.empty)

    @classmethod
    def from_types(
        cls,
        inputs: Dict[str, Tuple[Type, Type[_has_default] | Type[inspect._empty]]],
        outputs: Dict[str, Type],
        default_inputs: Optional[Dict[str, literals_pb2.Literal]] = None,
    ) -> NativeInterface:
        """
        Create a new NativeInterface from the given types. This is used to create a native interface for the task.
        :param inputs: A dictionary of input names and their types and a value indicating if they have a default value.
        :param outputs: A dictionary of output names and their types.
        :param default_inputs: Optional dictionary of default inputs for remote tasks.
        :return: A NativeInterface object with the given inputs and outputs.
        """
        for k, v in inputs.items():
            if v[1] is cls.has_default and (default_inputs is None or k not in default_inputs):
                raise ValueError(f"Input {k} has a default value but no default input provided for remote task.")
        return cls(inputs=inputs, outputs=outputs, _remote_defaults=default_inputs)

    @classmethod
    def from_callable(cls, func: Callable) -> NativeInterface:
        """
        Extract the native interface from the given function. This is used to create a native interface for the task.
        """
        # Get function parameters, defaults, varargs info (POSITIONAL_ONLY, VAR_POSITIONAL, KEYWORD_ONLY, etc.).
        sig = inspect.signature(func)

        # Extract parameter details (name, type, default value)
        param_info = {}
        try:
            # Get fully evaluated, real Python types for type checking.
            hints = typing.get_type_hints(func, include_extras=True)
        except Exception as e:
            logger.warning(f"Could not get type hints for function {func.__name__}: {e}")
            raise

        for name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise ValueError(f"Function {func.__name__} cannot have variable positional or keyword arguments.")
            if param.annotation is inspect.Parameter.empty:
                logger.warning(
                    f"Function {func.__name__} has parameter {name} without type annotation. Data will be pickled."
                )
            arg_type = hints.get(name, param.annotation)
            if typing.get_origin(arg_type) is Literal:
                param_info[name] = (literal_to_enum(arg_type), param.default)
            else:
                param_info[name] = (arg_type, param.default)

        # Get return type
        outputs = extract_return_annotation(hints.get("return", sig.return_annotation))
        return cls(inputs=param_info, outputs=outputs)

    def convert_to_kwargs(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Convert the given arguments to keyword arguments based on the native interface. This is used to convert the
        arguments to the correct types for the task execution.
        """
        # Convert positional arguments to keyword arguments
        if len(args) > len(self.inputs):
            raise ValueError(
                f"Too many positional arguments provided, expected inputs {self.inputs.keys()}, args {len(args)}"
            )
        for arg, input_name in zip(args, self.inputs.keys()):
            kwargs[input_name] = arg
        if len(kwargs) > len(self.inputs):
            raise ValueError(
                f"Too many keyword arguments provided, expected inputs {self.inputs.keys()}, args {kwargs.keys()}"
            )
        return kwargs

    def get_input_types(self) -> Dict[str, Type]:
        """
        Get the input types for the task. This is used to get the types of the inputs for the task execution.
        """
        return {k: v[0] for k, v in self.inputs.items()}

    def __repr__(self):
        """
        Returns a string representation of the task interface.
        """
        i = "("
        if self.inputs:
            initial = True
            for key, tpe in self.inputs.items():
                if not initial:
                    i += ", "
                initial = False
                tp = tpe[0] if isinstance(tpe[0], str) else getattr(tpe[0], "__name__", str(tpe[0]))
                i += f"{key}: {tp}"
                if tpe[1] is not inspect.Parameter.empty:
                    if tpe[1] is self.has_default:
                        i += " = ..."
                    else:
                        i += f" = {tpe[1]}"
        i += ")"
        if self.outputs:
            initial = True
            multi = len(self.outputs) > 1
            i += " -> "
            if multi:
                i += "("
            for key, tpe in self.outputs.items():
                if not initial:
                    i += ", "
                initial = False
                tp = tpe.__name__ if isinstance(tpe, type) else tpe
                i += f"{key}: {tp}"
            if multi:
                i += ")"
        return i + ":"


@dataclass
class SerializationContext:
    """
    This object holds serialization time contextual information, that can be used when serializing the task and
    various parameters of a tasktemplate. This is only available when the task is being serialized and can be
    during a deployment or runtime.

    :param version: The version of the task
    :param code_bundle: The code bundle for the task. This is used to package the code and the inflation path.
    :param input_path: The path to the inputs for the task. This is used to determine where the inputs will be located
    :param output_path: The path to the outputs for the task. This is used to determine where the outputs will be
     located
    """

    version: str
    project: str | None = None
    domain: str | None = None
    org: str | None = None
    code_bundle: Optional[CodeBundle] = None
    input_path: str = "{{.input}}"
    output_path: str = "{{.outputPrefix}}"
    interpreter_path: str = "/opt/venv/bin/python"
    image_cache: ImageCache | None = None
    root_dir: Optional[pathlib.Path] = None

    def get_entrypoint_path(self, interpreter_path: Optional[str] = None) -> str:
        """
        Get the entrypoint path for the task. This is used to determine the entrypoint for the task execution.
        :param interpreter_path: The path to the interpreter (python)
        """
        if interpreter_path is None:
            interpreter_path = self.interpreter_path
        return os.path.join(os.path.dirname(interpreter_path), "runtime.py")
