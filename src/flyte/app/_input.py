from dataclasses import dataclass, field
from enum import Enum
from typing import Union, Optional, Literal

from flyte.app._app_environment_bk import ENV_NAME_RE

InputType = Literal["file", "directory", "string"]

@dataclass
class Input:
    """
    Input for application.

    TODO Support flyte.io.File/Dir, string, int, float, bool, etc.

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

    value: str
    name: Optional[str] = None
    env_var: Optional[str] = None
    type: InputType = "string"
    download: bool = False
    mount: Optional[str] = None
    ignore_patterns: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.env_var is not None and ENV_NAME_RE.match(self.env_var) is None:
            raise ValueError(f"env_var ({self.env_var}) is not a valid environment name for shells")

        if not isinstance(self.value, str):
            raise TypeError(f"Expected value to be of type str, got {type(self.value)}")

        if self.name is None:
            self.name = "i0"
