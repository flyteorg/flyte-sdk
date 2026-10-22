"""
Backward compatibility for remote methods that used to be plain `async def` and are now syncified.

Sync callers get the result directly. Async callers that still write `await x.method()` keep working but get a
DeprecationWarning pointing at `.aio()`, and `await x.method.aio()` works without a warning.
"""

import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest
from flyteidl2.common import phase_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.task import common_pb2
from flyteidl2.workflow import run_definition_pb2

from flyte.remote._action import Action, ActionDetails, ActionOutputs
from flyte.remote._artifact import Artifact
from flyte.remote._run import RunDetails
from flyte.remote._trigger import Trigger
from flyte.syncify import _api as syncify_api
from flyte.syncify._api import _AwaitableCompatSyncWrapper
from flyte.types import TypeEngine

# Every method that was a plain coroutine before it was syncified.
CONVERTED_METHODS = [
    (Action, "details"),
    (Action, "wait"),
    *[
        (cls, name)
        for cls in (ActionDetails, RunDetails)
        for name in ("inputs", "outputs", "input_literals", "output_literals", "typed_inputs", "typed_outputs")
    ],
    (Artifact, "to_python"),
    (Artifact, "coerce_to_literal"),
    (Trigger, "get_details"),
]


@pytest.fixture(autouse=True)
def _reset_warned_call_sites():
    syncify_api._legacy_await_warned.clear()
    yield
    syncify_api._legacy_await_warned.clear()


async def _named(name: str, value, py_type) -> common_pb2.NamedLiteral:
    lit = await TypeEngine.to_literal(value, py_type, TypeEngine.to_literal_type(py_type))
    return common_pb2.NamedLiteral(name=name, value=lit)


def _details(x: common_pb2.NamedLiteral, o0: common_pb2.NamedLiteral) -> ActionDetails:
    """A terminal ActionDetails with seeded inputs and outputs, so no network call is made."""
    pb2 = run_definition_pb2.ActionDetails()
    pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
    resp = dataproxy_service_pb2.GetActionDataResponse(
        inputs=common_pb2.Inputs(literals=[x]),
        outputs=common_pb2.Outputs(literals=[o0]),
    )
    return ActionDetails(pb2=pb2, _action_data=resp)


async def _details_async() -> ActionDetails:
    return _details(await _named("x", 1, int), await _named("o0", 2, int))


def _details_sync() -> ActionDetails:
    return asyncio.run(_details_async())


def _legacy_warnings(record: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in record if issubclass(w.category, DeprecationWarning) and ".aio()" in str(w.message)]


@pytest.mark.parametrize(("cls", "name"), CONVERTED_METHODS, ids=lambda v: getattr(v, "__name__", v))
def test_converted_method_uses_compat_decorator(cls, name):
    assert isinstance(cls.__dict__[name], _AwaitableCompatSyncWrapper)
    assert hasattr(getattr(cls, name), "aio")


class TestActionDetailsOutputs:
    def test_sync_call_returns_the_value(self):
        details = _details_sync()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            outputs = details.outputs()
        assert isinstance(outputs, ActionOutputs)
        assert outputs.pb2.literals[0].name == "o0"
        assert _legacy_warnings(record) == []

    @pytest.mark.asyncio
    async def test_legacy_await_works_and_warns(self):
        details = await _details_async()
        with pytest.warns(DeprecationWarning, match=r"await <obj>\.outputs\.aio\(\)"):
            outputs = await details.outputs()
        assert isinstance(outputs, ActionOutputs)
        assert outputs.pb2.literals[0].name == "o0"

    @pytest.mark.asyncio
    async def test_aio_works_without_warning(self):
        details = await _details_async()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            outputs = await details.outputs.aio()
        assert isinstance(outputs, ActionOutputs)
        assert _legacy_warnings(record) == []


class TestActionDetailsTypedInputs:
    def test_sync_call_returns_the_value(self):
        details = _details_sync()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert details.typed_inputs({"x": int}) == {"x": 1}
        assert _legacy_warnings(record) == []

    @pytest.mark.asyncio
    async def test_legacy_await_works_and_warns(self):
        details = await _details_async()
        with pytest.warns(DeprecationWarning, match=r"ActionDetails\.typed_inputs"):
            assert await details.typed_inputs({"x": int}) == {"x": 1}

    @pytest.mark.asyncio
    async def test_aio_works_without_warning(self):
        details = await _details_async()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert await details.typed_inputs.aio({"x": int}) == {"x": 1}
        assert _legacy_warnings(record) == []


class TestActionDetails:
    @staticmethod
    def _action(details: ActionDetails | None = None) -> Action:
        return Action(pb2=run_definition_pb2.Action(), _details=details)

    def test_sync_call_fetches_and_returns_the_value(self, monkeypatch):
        details = _details_sync()
        fake_get_details = MagicMock(aio=AsyncMock(return_value=details))
        monkeypatch.setattr(ActionDetails, "get_details", fake_get_details)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert self._action().details() is details
        fake_get_details.aio.assert_awaited_once()
        assert _legacy_warnings(record) == []

    @pytest.mark.asyncio
    async def test_legacy_await_works_and_warns(self):
        details = await _details_async()
        with pytest.warns(DeprecationWarning, match=r"Action\.details\(\)") as record:
            assert await self._action(details).details() is details
        # The warning points at the caller's line, not at the syncify internals.
        assert record[0].filename == __file__

    @pytest.mark.asyncio
    async def test_aio_works_without_warning(self):
        details = await _details_async()
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            assert await self._action(details).details.aio() is details
        assert _legacy_warnings(record) == []


@pytest.mark.asyncio
async def test_legacy_warning_is_emitted_once_per_call_site():
    details = await _details_async()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(3):
            await details.outputs()
        await details.outputs()
    # Two distinct call sites, each warned once, despite the "always" filter.
    assert len(_legacy_warnings(record)) == 2
