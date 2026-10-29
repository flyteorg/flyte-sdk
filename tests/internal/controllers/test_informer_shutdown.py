import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from flyteidl2.common import identifier_pb2
from flyteidl2.workflow import state_service_pb2

from flyte._internal.controllers.remote._core import Controller
from flyte._internal.controllers.remote._informer import Informer, InformerCache


@pytest.mark.asyncio
@pytest.mark.parametrize("use_actions", [False, True])
async def test_stop_waits_for_stream_cleanup(use_actions):
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def watch(*args, **kwargs):
        try:
            yield state_service_pb2.WatchResponse(control_message={"sentinel": True})
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()
            cleanup_finished.set()

    client = SimpleNamespace(watch=watch, watch_for_updates=watch)
    informer = Informer(
        identifier_pb2.RunIdentifier(name="run"),
        "parent",
        asyncio.Queue(),
        client=client,
        actions_client=client if use_actions else None,
    )
    watch_task = await informer.start(timeout=1)
    stop_task = asyncio.create_task(informer.stop())
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        assert not stop_task.done()
        assert informer.watch_task is watch_task
        assert not informer.is_running()
    finally:
        allow_cleanup.set()
        await asyncio.wait_for(stop_task, timeout=1)

    assert cleanup_finished.is_set()
    assert watch_task.done()
    assert informer.watch_task is None
    await informer.stop()  # Stopping twice is safe.


@pytest.mark.asyncio
async def test_stop_during_retry_backoff():
    async def watch(*args, **kwargs):
        yield state_service_pb2.WatchResponse(control_message={"sentinel": True})
        raise RuntimeError("stream disconnected")

    informer = Informer(
        identifier_pb2.RunIdentifier(name="run"),
        "parent",
        asyncio.Queue(),
        client=SimpleNamespace(watch=watch),
        min_watch_backoff=3600,
        max_watch_backoff=3600,
    )
    task = await informer.start(timeout=1)
    await asyncio.wait_for(informer.stop(), timeout=1)
    assert task.cancelled()
    assert informer.watch_task is None
    controller = Mock()
    Controller._bg_handle_informer_error(controller, task)
    controller._set_exception.assert_not_called()
    controller._failure_event.set.assert_not_called()


@pytest.mark.asyncio
async def test_failed_watch_does_not_interrupt_cache_shutdown():
    cache = InformerCache()
    tasks = []
    for parent in ("one", "two"):
        informer = Informer(identifier_pb2.RunIdentifier(name="run"), parent, asyncio.Queue())

        async def failed_watch():
            raise RuntimeError("watch exhausted retries")

        task = asyncio.create_task(failed_watch())
        informer._watch_task = task
        cache._cache[informer.name] = informer
        tasks.append(task)
    await asyncio.sleep(0)  # Let the watches fail before stopping them.
    await cache.remove_and_stop_all()
    assert not cache._cache
    assert all(task.done() for task in tasks)
