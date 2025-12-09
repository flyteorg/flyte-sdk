import shlex
from dataclasses import dataclass
from typing import List

import rich.repr

from flyte.app import AppEnvironment
from flyte.app._input import Input
from flyte.app._types import Port
from flyte.models import SerializationContext


@rich.repr.auto
@dataclass(init=True, repr=True)
class ConnectorEnvironment(AppEnvironment):
    type: str = "connector"
    port: int | Port = 8000

    def __post_init__(self):
        super().__post_init__()

    def container_args(self, serialize_context: SerializationContext) -> List[str]:
        return ["c0"]

    def container_cmd(
        self, serialize_context: SerializationContext, input_overrides: list[Input] | None = None
    ) -> List[str]:
        if isinstance(self.command, str):
            return shlex.split(self.command)
        return self.command
