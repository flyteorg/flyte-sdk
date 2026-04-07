import os
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
    type: str = "connector"
    port: int | Port = field(default=Port(port=8080, name="h2c"))

    def __post_init__(self):
        super().__post_init__()

    def container_args(self, serialize_context: SerializationContext) -> List[str]:
        if self.args is None:
            if isinstance(self.port, Port):
                port = self.port.port
            else:
                port = self.port
            base_args = ["c0", "--port", str(port), "--prometheus_port", "9092"]
            if self.include:
                # Convert file paths to module names
                modules = []
                for file_path in self.include:
                    # Remove .py extension if exists
                    module_name = os.path.splitext(os.path.basename(file_path))[0]
                    modules.append(module_name)

                base_args.extend(["--modules", *modules])
            return base_args
        return super().container_args(serialize_context)

    def container_cmd(
        self, serialize_context: SerializationContext, parameter_overrides: list[Parameter] | None = None
    ) -> List[str]:
        if isinstance(self.command, str):
            return shlex.split(self.command)
        elif isinstance(self.command, list):
            return self.command
        else:
            # command is None, use default from parent class
            return super().container_cmd(serialize_context, parameter_overrides)
