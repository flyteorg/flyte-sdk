import re
from dataclasses import dataclass, field
from functools import cache
from typing import Literal, Optional

import flyte.io

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
        env_name_re = re.compile("^[_a-zA-Z][_a-zA-Z0-9]*$")

        if self.env_var is not None and env_name_re.match(self.env_var) is None:
            raise ValueError(f"env_var ({self.env_var}) is not a valid environment name for shells")

        if not isinstance(self.value, (str, flyte.io.File, flyte.io.Dir)):
            raise TypeError(f"Expected value to be of type str, file or dir, got {type(self.value)}")

        if self.name is None:
            self.name = "i0"


@cache
def get_input(name: str) -> str:
    """Get inputs for application or endpoint."""
    import json
    import os

    from ._runtime import RUNTIME_CONFIG_FILE

    config_file = os.getenv(RUNTIME_CONFIG_FILE)

    with open(config_file, "r") as f:
        inputs = json.load(f)["inputs"]

    return inputs[name]
