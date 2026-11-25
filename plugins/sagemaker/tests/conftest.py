from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from flyteidl2.core.interface_pb2 import TypedInterface, Variable, VariableEntry, VariableMap
from flyteidl2.core.literals_pb2 import KeyValuePair
from flyteidl2.core.tasks_pb2 import Container, TaskTemplate
from flyteidl2.core.types_pb2 import LiteralType, SimpleType
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct


def template(
    task_type: str,
    custom_value: dict,
    *,
    outputs: tuple[str, ...] = ("result",),
    image: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> TaskTemplate:
    custom = Struct()
    ParseDict(custom_value, custom)
    output_type = LiteralType(simple=SimpleType.STRUCT)
    interface = TypedInterface(
        outputs=VariableMap(variables=[VariableEntry(key=name, value=Variable(type=output_type)) for name in outputs])
    )
    container = None
    if image:
        container = Container(
            image=image,
            args=args if args is not None else ["a0", "--outputs-path", "s3://bucket/run/outputs.pb"],
            env=[KeyValuePair(key=key, value=value) for key, value in (env or {}).items()],
        )
    return TaskTemplate(type=task_type, custom=custom, interface=interface, container=container)


def aws_client() -> MagicMock:
    client = MagicMock()
    client.meta = SimpleNamespace(region_name="us-west-2")
    return client
