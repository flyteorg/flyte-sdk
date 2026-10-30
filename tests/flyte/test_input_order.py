"""A task's declared input order travels in its template, so remote calls can be positional."""

import asyncio
import pathlib
from unittest import mock

import pytest
from flyteidl2.task import task_definition_pb2

import flyte
import flyte.errors
from flyte._internal.runtime.task_serde import translate_task_to_wire
from flyte._internal.runtime.types_serde import INPUT_ORDER_CONFIG_KEY
from flyte.models import SerializationContext
from flyte.remote._task import TaskDetails

env = flyte.TaskEnvironment(name="input-order")


@env.task
async def power(exponent: int, base: int) -> int:
    return base**exponent


@env.task
async def no_inputs() -> int:
    return 1


def _template(task):
    return translate_task_to_wire(
        task, SerializationContext(version="v1", root_dir=pathlib.Path(__file__).parent)
    ).task_template


def test_template_records_declared_order():
    template = _template(power)
    assert template.config[INPUT_ORDER_CONFIG_KEY] == "exponent,base"
    # The interface itself stays sorted by name.
    assert [v.key for v in template.interface.inputs.variables] == ["base", "exponent"]


def test_template_records_no_inputs():
    assert _template(no_inputs).config[INPUT_ORDER_CONFIG_KEY] == ""


def _details(order=None) -> TaskDetails:
    spec = task_definition_pb2.TaskSpec()
    spec.task_template.CopyFrom(_template(power))
    del spec.task_template.config[INPUT_ORDER_CONFIG_KEY]
    if order is not None:
        spec.task_template.config[INPUT_ORDER_CONFIG_KEY] = order
    return TaskDetails(pb2=task_definition_pb2.TaskDetails(spec=spec))


def test_remote_task_input_names():
    assert _details("exponent,base").input_names == ("exponent", "base")
    assert _details("").input_names == ()
    assert _details().input_names is None


def test_remote_task_binds_positional_arguments():
    details = _details("exponent,base")
    assert details._bind_positional((2, 3), {}) == {"exponent": 2, "base": 3}
    assert details._bind_positional((2,), {"base": 3}) == {"exponent": 2, "base": 3}
    with pytest.raises(TypeError, match="multiple values"):
        details._bind_positional((2,), {"exponent": 3})
    with pytest.raises(TypeError, match="positional arguments"):
        details._bind_positional((1, 2, 3), {})


def test_remote_task_without_recorded_order_takes_keywords_only():
    with pytest.raises(flyte.errors.RemoteTaskUsageError, match="keyword"):
        asyncio.run(_details()(2, 3))


def test_remote_task_call_submits_bound_keywords():
    details = _details("exponent,base")
    controller = mock.AsyncMock()
    ctx = mock.Mock(is_task_context=mock.Mock(return_value=True))
    with (
        mock.patch("flyte.remote._task.internal_ctx", return_value=ctx),
        mock.patch("flyte._internal.controllers.get_controller", return_value=controller),
    ):
        asyncio.run(details(2, 3))
    controller.submit_task_ref.assert_awaited_once_with(details, exponent=2, base=3)
