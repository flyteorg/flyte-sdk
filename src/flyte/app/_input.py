from __future__ import annotations

import re
import typing
from dataclasses import dataclass, field
from functools import cache, cached_property
from typing import List, Literal, Optional

from pydantic import BaseModel

if typing.TYPE_CHECKING:
    import flyte.io

RUNTIME_INPUTS_FILE = "flyte-inputs.json"

InputType = Literal["file", "directory", "string"]


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

    value: str | flyte.io.File | flyte.io.Dir
    name: Optional[str] = None
    env_var: Optional[str] = None
    download: bool = False
    mount: Optional[str] = None
    ignore_patterns: list[str] = field(default_factory=list)

    def __post_init__(self):
        import flyte.io

        env_name_re = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")

        if self.env_var is not None and env_name_re.match(self.env_var) is None:
            raise ValueError(f"env_var ({self.env_var}) is not a valid environment name for shells")

        if not isinstance(self.value, (str, flyte.io.File, flyte.io.Dir)):
            raise TypeError(f"Expected value to be of type str, file or dir, got {type(self.value)}")

        if self.name is None:
            self.name = "i0"


_SerializedInputType = Literal["file", "directory", "string"]


class SerializableInput(BaseModel):
    """
    Serializable version of Input.
    """

    name: str
    value: str
    download: bool
    type: _SerializedInputType = "string"
    env_var: Optional[str] = None
    dest: Optional[str] = None
    ignore_patterns: List[str] = field(default_factory=list)

    @classmethod
    def from_input(cls, inp: Input) -> "SerializableInput":
        import flyte.io

        # inp.name is guaranteed to be set by Input.__post_init__
        assert inp.name is not None, "Input name should be set by __post_init__"

        tpe: _SerializedInputType = "string"
        if isinstance(inp.value, flyte.io.File):
            value = inp.value.path
            tpe = "file"
            download = True if inp.mount is not None else inp.download
        elif isinstance(inp.value, flyte.io.Dir):
            value = inp.value.path
            tpe = "directory"
            download = True if inp.mount is not None else inp.download
        else:
            value = inp.value
            download = False

        return cls(
            name=inp.name,
            value=value,
            type=tpe,
            download=download,
            env_var=inp.env_var,
            dest=inp.mount,
            ignore_patterns=inp.ignore_patterns,
        )


class SerializableInputCollection(BaseModel):
    """
    Collection of inputs for application.

    :param inputs: List of inputs.
    """

    inputs: List[SerializableInput] = field(default_factory=list)

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
    def from_transport(cls, s: str) -> SerializableInputCollection:
        import base64
        import gzip

        compressed_val = base64.b64decode(s.encode("utf-8"))
        json_str = gzip.decompress(compressed_val).decode("utf-8")
        return cls.model_validate_json(json_str)

    @classmethod
    def from_inputs(cls, inputs: List[Input]) -> SerializableInputCollection:
        return cls(inputs=[SerializableInput.from_input(inp) for inp in inputs])


@cache
def _load_inputs() -> dict[str, str]:
    """Load inputs for application or endpoint."""
    import json
    import os

    config_file = os.getenv(RUNTIME_INPUTS_FILE)

    if config_file is None:
        raise ValueError("Inputs are not mounted")

    with open(config_file, "r") as f:
        inputs = json.load(f)

    return inputs


def get_input(name: str) -> str:
    """Get inputs for application or endpoint."""
    inputs = _load_inputs()
    return inputs[name]
