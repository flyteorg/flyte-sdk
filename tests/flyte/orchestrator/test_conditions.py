"""Tests for conditions inside `flyte.orchestrator.task`."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import flyte
import flyte.errors
import flyte.orchestrator
from flyte import new_condition
from flyte import new_condition as ask
from flyte._condition import _Condition
from flyte.durable import sleep as durable_sleep
from flyte.models import SerializationContext
from flyte.orchestrator._conditions import NEW_CONDITION, PRELUDE, WAIT_CONDITION
from flyte.sandbox._config import SandboxedConfig
from flyte.sandbox._runtime import checkout


@flyte.orchestrator.task
def approve_by_module(x: int) -> int:
    approval = flyte.new_condition("approve", prompt="Double it?", timeout=60)
    return x * 2 if approval.wait() else x


@flyte.orchestrator.task
def approve_by_name(x: int) -> int:
    # Only the function's source is shipped, so it imports what it uses.
    from datetime import timedelta

    count = new_condition("count", data_type=int, timeout=timedelta(minutes=2))
    note = ask("note", data_type=str, prompt="**Why?**", prompt_type="markdown")
    return x * count.wait() + len(note.wait())


@flyte.orchestrator.task
def no_conditions(x: int) -> int:
    return x + 1


def _custom(task) -> dict:
    asyncio.run(task.resolve_tasks())
    return task.custom_config(SerializationContext(version="v1"))


class TestDecoration:
    def test_conditions_are_not_external_refs(self):
        for task in (approve_by_module, approve_by_name):
            assert task._uses_conditions
            assert all(not refs for refs in task._external_refs.values()), task._external_refs

    def test_source_ships_with_the_prelude(self):
        source = _custom(approve_by_module)["source"]
        assert source.startswith(PRELUDE)
        assert "\nflyte = _FlyteNamespace()\n" in source
        assert source.rstrip().endswith("approve_by_module(x)")

        source = _custom(approve_by_name)["source"]
        assert "\nask = _flyte_new_condition_impl\n" in source
        assert "\nnew_condition = _flyte_new_condition_impl\n" in source
        assert "_FlyteNamespace()" not in source.split(PRELUDE, 1)[1]

    def test_a_source_without_conditions_is_shipped_unchanged(self):
        assert not no_conditions._uses_conditions
        assert _custom(no_conditions)["source"].startswith("def no_conditions")

    def test_the_prelude_changes_the_version(self):
        assert approve_by_module.source_version != approve_by_name.source_version

    def test_other_calls_are_still_rejected(self):
        with pytest.raises(flyte.errors.RuntimeUserError, match="durable_sleep is neither"):

            @flyte.orchestrator.task
            def sleeps(x: int) -> int:
                flyte.new_condition("approve").wait()
                durable_sleep(1)
                return x


async def _run_with_fake_host(source: str, inputs: dict, signals: dict) -> tuple:
    """Run *source* as the worker does, answering the prelude's calls from *signals*."""
    from pydantic_monty import AsyncFunctionSnapshot, MontyComplete

    calls = []
    async with await checkout(SandboxedConfig()) as session:
        progress = await session.feed_start(source, inputs=inputs)
        while not isinstance(progress, MontyComplete):
            assert isinstance(progress, AsyncFunctionSnapshot), progress
            calls.append((progress.function_name, tuple(progress.args), dict(progress.kwargs)))
            if progress.function_name == NEW_CONDITION:
                result = f"handle-{progress.args[0]}"
            else:
                assert progress.function_name == WAIT_CONDITION
                result = signals[progress.args[0].removeprefix("handle-")]
            progress = await progress.resume({"return_value": result})
    return progress.output, calls


class TestWorkerProtocol:
    """The calls the worker answers, as the shipped source makes them."""

    def test_module_and_name_forms_make_the_same_calls(self):
        output, calls = asyncio.run(_run_with_fake_host(approve_by_module._source_code, {"x": 3}, {"approve": True}))
        assert output == 6
        assert calls == [
            (
                NEW_CONDITION,
                ("approve",),
                {
                    "prompt": "Double it?",
                    "prompt_type": "text",
                    "data_type": "bool",
                    "description": "",
                    "timeout": 60,
                },
            ),
            (WAIT_CONDITION, ("handle-approve",), {}),
        ]

    def test_types_timedeltas_and_markdown_are_passed_on(self):
        output, calls = asyncio.run(
            _run_with_fake_host(approve_by_name._source_code, {"x": 3}, {"count": 7, "note": "ok"})
        )
        assert output == 23
        created = {args[0]: kwargs for name, args, kwargs in calls if name == NEW_CONDITION}
        assert created["count"]["data_type"] == "int"
        assert created["count"]["timeout"] == 120.0
        assert created["note"]["data_type"] == "str"
        assert created["note"]["prompt_type"] == "markdown"

    def test_an_invalid_data_type_raises_in_the_source(self):
        source = PRELUDE + "\nnew_condition = _flyte_new_condition_impl\nnew_condition('x', data_type=list)\n"
        with pytest.raises(Exception, match="Invalid data_type"):
            asyncio.run(_run_with_fake_host(source, {}, {}))


class TestLocalRun:
    """Run locally, the prelude's calls go through `flyte.new_condition` and the controller."""

    def test_conditions_are_registered_and_waited_on(self):
        signals = {"count": 4, "note": "fine"}
        registered = []

        async def register(condition):
            registered.append(condition)

        async def wait(condition: _Condition):
            return signals[condition.name]

        controller = MagicMock()
        controller.register_action = AsyncMock(side_effect=register)
        controller.wait_for_registered_action = AsyncMock(side_effect=wait)
        ctx = MagicMock()
        ctx.is_task_context.return_value = True
        with (
            patch("flyte._context.internal_ctx", return_value=ctx),
            patch("flyte._internal.controllers.get_controller", return_value=controller),
        ):
            result = asyncio.run(approve_by_name._run_sandboxed(x=5))

        assert result == 5 * 4 + len("fine")
        assert [c.name for c in registered] == ["count", "note"]
        count, note = registered
        assert count.data_type is int
        assert count._timeout_seconds == 120.0
        assert note.data_type is str
        assert note.prompt_type == "markdown"
        assert note.prompt == "**Why?**"
