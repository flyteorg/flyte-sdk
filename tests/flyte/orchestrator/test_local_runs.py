"""Orchestrators run on this machine see and do what they would on a cluster.

Each expectation here was checked against the remote orchestrator worker on a
cluster, which gave the same answers: the values a task call returns, how
arguments bind, and how failures are raised in the source. The rules
themselves are pinned in `test_values.py`; these run them end to end, through
`flyte.run` and real local task calls.
"""

import asyncio
from typing import Optional
from unittest import mock

import pytest
from flyteidl2.task import task_definition_pb2
from pydantic import BaseModel

import flyte
import flyte.errors
import flyte.orchestrator
from flyte.io import File

env = flyte.TaskEnvironment(name="parity")


class Point(BaseModel):
    x: int
    y: int


@env.task
async def numbers() -> dict[str, int]:
    return {"b": 2, "a": 1, "c": 3}


@env.task
async def half(x: float) -> float:
    return x / 2


@env.task
async def fail(message: str) -> int:
    raise ValueError(message)


@env.task
async def point(x: int) -> Point:
    return Point(x=x, y=x)


@env.task
async def norm(p: Point) -> int:
    return p.x + p.y


@env.task
async def write(text: str) -> File:
    f = File.new_remote(file_name="probe.txt")
    async with f.open("wb") as fh:
        await fh.write(text.encode())
    return f


@env.task
async def echo(xs: list[int]) -> list[int]:
    return xs


@env.task
async def maybe(x: Optional[int] = None, label: str = "none") -> str:
    return f"{label}:{x!r}"


@env.task
async def pair(x: int) -> tuple[int, str]:
    return x, str(x)


@env.task
async def nothing() -> None:
    return None


def run(orchestrator, **inputs):
    flyte.init()
    return flyte.with_runcontext(mode="local").run(orchestrator, **inputs).outputs()[0]


@flyte.orchestrator.task
def dict_keys() -> list[str]:
    return list(numbers().keys())


def test_a_dict_a_task_returns_is_sorted_by_key():
    assert run(dict_keys) == ["a", "b", "c"]


@flyte.orchestrator.task
def int_to_float(n: int) -> float:
    return half(n)


def test_an_int_passed_as_a_float_is_a_float():
    assert run(int_to_float, n=3) == 1.5


@flyte.orchestrator.task
def caught(message: str) -> str:
    try:
        fail(message)
    except RuntimeError as e:
        return str(e)
    return "no error"


def test_a_failed_task_is_a_runtime_error_with_its_code_and_message():
    assert run(caught, message="boom") == "ValueError: boom"


@flyte.orchestrator.task
def uncaught(message: str) -> int:
    return fail(message)


def test_an_uncaught_task_failure_fails_the_run():
    with pytest.raises(flyte.errors.RuntimeUserError, match="ValueError: boom"):
        run(uncaught, message="boom")


@flyte.orchestrator.task
def bad_calls() -> list[str]:
    errors = []
    for attempt in range(4):
        try:
            if attempt == 0:
                half("x")
            elif attempt == 1:
                half(1, 2)
            elif attempt == 2:
                half(y=1)
            else:
                half()
            errors.append("no error")
        except RuntimeError as e:
            errors.append(str(e))
    return errors


def test_calls_that_do_not_bind_are_type_errors_the_source_can_catch():
    assert run(bad_calls) == [
        "TypeError: parity.half() argument 'x': a str cannot be passed as float",
        "TypeError: parity.half() takes 1 arguments but 2 were given",
        "TypeError: parity.half() got an unexpected keyword argument 'y'",
        "TypeError: parity.half() is missing the argument 'x'",
    ]


@flyte.orchestrator.task
def plain_data_only() -> list[str]:
    seen = [repr(echo((1, 2)))]
    for value in [{1, 2}, b"x", 2**63]:
        try:
            echo([value])
            seen.append("no error")
        except TypeError as e:
            seen.append(str(e))
    return seen


def test_tuples_are_lists_and_values_that_are_not_plain_data_are_type_errors():
    assert run(plain_data_only) == [
        "[1, 2]",
        "values of type 'set' cannot be passed to or returned from a task",
        "values of type 'bytes' cannot be passed to or returned from a task",
        "values of type 'int' cannot be passed to or returned from a task",
    ]


@flyte.orchestrator.task
async def opaque_values() -> list[str]:
    p1, p2 = await asyncio.gather(point(1), point(1))
    p3 = await point(2)
    seen = [repr(p1 == p2), repr(p1 == p3), repr(p1)[:15], repr(await norm(p1))]
    try:
        await norm(1)
        seen.append("no error")
    except RuntimeError as e:
        seen.append(str(e))
    try:
        p1.x
        seen.append("no error")
    except AttributeError:
        seen.append("AttributeError")
    return seen


def test_models_are_opaque_values_equal_when_their_literals_are():
    assert run(opaque_values) == [
        "True",
        "False",
        "Opaque(literal=",
        "2",
        "TypeError: parity.norm() argument 'p': values of Flyte type struct are not supported",
        "AttributeError",
    ]


@flyte.orchestrator.task
def file_fields() -> list[str]:
    f = write("hi")
    return [f.name, f.format, repr(f.hash), repr(f.path == f[0])]


def test_a_file_is_a_named_tuple_named_after_its_path():
    assert run(file_fields) == ["probe.txt", "", "None", "True"]


@flyte.orchestrator.task
def defaults_and_outputs() -> list[str]:
    return [maybe(), maybe(3, label="three"), repr(pair(5)), repr(nothing())]


def test_defaults_fill_missing_arguments_and_several_outputs_are_a_list():
    assert run(defaults_and_outputs) == ["none:None", "three:3", "[5, '5']", "None"]


@flyte.orchestrator.task
def unknown_name() -> str:
    try:
        not_a_task(1)
    except NameError as e:
        return str(e)
    return "no error"


@flyte.orchestrator.task
def no_map() -> str:
    try:
        flyte_map("parity.half", [1])
    except NameError as e:
        return str(e)
    return "no error"


def test_names_that_are_not_tasks_are_name_errors():
    assert run(unknown_name) == "name 'not_a_task' is not defined"
    assert run(no_map) == "name 'flyte_map' is not defined"


@flyte.orchestrator.task
def inputs_seen(n: int, ratio: float, tags: dict[str, int]) -> list[str]:
    return [repr(n), repr(ratio), repr(list(tags))]


def test_inputs_are_read_as_the_worker_reads_them():
    assert run(inputs_seen, n=3, ratio=2.0, tags={"b": 1, "a": 2}) == ["3", "2.0", "['a', 'b']"]


@flyte.orchestrator.task
def returns_wrong_type() -> int:
    return "x"


@flyte.orchestrator.task
def returns_set() -> None:
    return {1}


@flyte.orchestrator.task
def returns_one(n: int) -> tuple[int, str]:
    return n


@flyte.orchestrator.task
def returns_two(n: int) -> tuple[int, str]:
    return n, str(n)


def test_the_result_must_fit_the_declared_outputs():
    with pytest.raises(flyte.errors.RuntimeUserError, match="the result: a str cannot be passed as integer"):
        run(returns_wrong_type)
    with pytest.raises(flyte.errors.RuntimeUserError, match="values of type 'set' cannot be passed"):
        run(returns_set)
    with pytest.raises(flyte.errors.RuntimeUserError, match="the orchestrator declares 2 outputs"):
        run(returns_one, n=1)
    assert run(returns_two, n=1) == 1


@pytest.mark.asyncio
async def test_a_local_tasks_spec_carries_its_defaults():
    """The worker fills a call's missing arguments from the spec, so it must carry them."""
    from flyte._run import _Runner

    @flyte.orchestrator.task
    def calls_maybe() -> str:
        return maybe()

    async def build(self, task):
        spec = task_definition_pb2.TaskSpec()
        spec.task_template.id.name = task.name
        return spec, None, "v1"

    await flyte.init.aio()
    with mock.patch.object(_Runner, "_build_task_spec_from_template", build):
        await _Runner()._build_containerless_task_spec(calls_maybe)
    import base64

    from google.protobuf.json_format import MessageToDict

    from flyte._internal.runtime.task_serde import translate_task_to_wire
    from flyte.models import SerializationContext

    custom = MessageToDict(translate_task_to_wire(calls_maybe, SerializationContext(version="v1")).task_template.custom)
    spec = task_definition_pb2.TaskSpec()
    spec.ParseFromString(base64.b64decode(custom["tasks"]["maybe"]["spec"]))
    defaults = {p.name: p.parameter.default for p in spec.default_inputs}
    assert set(defaults) == {"x", "label"}
    assert defaults["label"].scalar.primitive.string_value == "none"
