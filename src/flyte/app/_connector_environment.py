import shlex
from dataclasses import dataclass, field
from typing import List

import rich.repr

from flyte.app import AppEnvironment
from flyte.app._parameter import Parameter
from flyte.app._types import Port
from flyte.models import SerializationContext


@rich.repr.auto
@dataclass(init=True, repr=True)
class ConnectorEnvironment(AppEnvironment):
    type: str = "Connector"
    port: int | Port = field(default=Port(port=8080, name="h2c"))

    def __post_init__(self):
        super().__post_init__()

    def container_args(self, serialize_context: SerializationContext) -> List[str]:
        if self.args is None:
            return ["c0", "--port", str(self.port.port), "--prometheus_port", "9092"]
        return super().container_args(serialize_context)

    def container_cmd(
        self, serialize_context: SerializationContext, parameter_overrides: list[Parameter] | None = None
    ) -> List[str]:
        if isinstance(self.command, str):
            return shlex.split(self.command)
        return self.command
