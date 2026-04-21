from pathlib import Path

import flyte
from flyte._internal.runtime.task_serde import translate_task_to_wire
from flyte.models import SerializationContext

from flyteplugins.echo import Echo, EchoTask

ROOT_DIR = Path(__file__).resolve().parents[3]


def test_echo_task_serializes_to_echo_type():
    env = flyte.TaskEnvironment(
        name="echo-env",
        image=flyte.Image.from_debian_base(),
        plugin_config=Echo(),
    )

    @env.task
    async def echo_identity(message: str) -> str:
        return message

    assert isinstance(echo_identity, EchoTask)
    assert echo_identity.task_type == "echo"

    task_spec = translate_task_to_wire(echo_identity, SerializationContext(version="v1", root_dir=ROOT_DIR))
    assert task_spec.task_template.type == "echo"
    assert len(task_spec.task_template.custom) == 0


def test_echo_task_supports_empty_interface():
    env = flyte.TaskEnvironment(
        name="echo-empty-env",
        image=flyte.Image.from_debian_base(),
        plugin_config=Echo(),
    )

    @env.task
    async def echo_noop() -> None:
        return None

    task_spec = translate_task_to_wire(echo_noop, SerializationContext(version="v1", root_dir=ROOT_DIR))
    assert task_spec.task_template.type == "echo"
