from datetime import timedelta
from pathlib import Path

from flyteidl2.core.types_pb2 import SimpleType

import flyte
from flyte._internal.runtime.task_serde import translate_task_to_wire
from flyte.extras import Sleep, SleepTask
from flyte.models import SerializationContext

ROOT_DIR = Path(__file__).resolve().parents[3]


def test_sleep_task_serializes_to_core_sleep_type():
    env = flyte.TaskEnvironment(
        name="sleep-env",
        image=flyte.Image.from_debian_base(),
        plugin_config=Sleep(),
    )

    @env.task
    async def sleep_for(duration: timedelta) -> None:
        return None

    assert isinstance(sleep_for, SleepTask)
    assert sleep_for.task_type == "core-sleep"

    task_spec = translate_task_to_wire(sleep_for, SerializationContext(version="v1", root_dir=ROOT_DIR))
    assert task_spec.task_template.type == "core-sleep"
    assert len(task_spec.task_template.custom) == 0

    inputs = task_spec.task_template.interface.inputs.variables
    assert len(inputs) == 1
    assert inputs[0].key == "duration"
    assert inputs[0].value.type.simple == SimpleType.DURATION

    outputs = task_spec.task_template.interface.outputs.variables
    assert len(outputs) == 0


def test_sleep_task_requires_no_plugin_customization():
    env = flyte.TaskEnvironment(
        name="sleep-default-env",
        image=flyte.Image.from_debian_base(),
        plugin_config=Sleep(),
    )

    @env.task
    async def sleep_default(duration: timedelta = timedelta(seconds=5)) -> None:
        return None

    task_spec = translate_task_to_wire(sleep_default, SerializationContext(version="v1", root_dir=ROOT_DIR))
    assert task_spec.task_template.type == "core-sleep"
    assert len(task_spec.task_template.custom) == 0
