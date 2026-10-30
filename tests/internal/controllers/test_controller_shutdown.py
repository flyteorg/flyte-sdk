import asyncio
import threading
from types import SimpleNamespace

import pytest
from flyteidl2.common import identifier_pb2

from flyte._internal.controllers.remote._action import Action
from flyte._internal.controllers.remote._controller import RemoteController
from flyte._internal.controllers.remote._core import Controller


async def make_client():
    return SimpleNamespace(state_service=None, queue_service=None, actions_service=None)


@pytest.mark.asyncio
async def test_idle_workers_stop_and_repeated_stop():
    controller = Controller(make_client(), workers=2)
    loop = controller._loop
    thread = controller._thread
    await asyncio.wait_for(asyncio.gather(controller.stop(), controller.stop()), timeout=3)
    assert not thread.is_alive()
    assert loop.is_closed()
    assert all(task.done() for task in controller._worker_tasks)
    assert controller._resource_log_task.done()
    await controller.stop()

    async def unexpected_work():
        pytest.fail("Work was accepted after shutdown")

    coro = unexpected_work()
    with pytest.raises(RuntimeError, match="stopped or stopping"):
        controller._run_coroutine_in_controller_thread(coro)
    assert coro.cr_frame is None  # Rejected coroutines must not leak.


@pytest.mark.asyncio
async def test_stop_waits_for_active_worker_cleanup(monkeypatch):
    started = threading.Event()
    cleaning = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    async def process(self, action):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await asyncio.to_thread(release.wait)
            finished.set()

    monkeypatch.setattr(Controller, "_bg_process", process)
    controller = Controller(make_client(), workers=1)
    action = Action(action_id=identifier_pb2.ActionIdentifier(name="test"), parent_action_name="parent")
    controller._loop.call_soon_threadsafe(controller._shared_queue.put_nowait, action)
    stop = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        stop = asyncio.create_task(controller.stop())
        assert await asyncio.to_thread(cleaning.wait, 2)
        assert not stop.done()
    finally:
        release.set()
        await asyncio.wait_for(stop if stop is not None else controller.stop(), timeout=3)
    assert finished.is_set()
    assert not controller._thread.is_alive()
    assert controller._loop.is_closed()


@pytest.mark.asyncio
async def test_loop_drains_pending_tasks_async_generators_and_executor():
    controller = Controller(make_client(), workers=1)
    task_cleaned = threading.Event()
    generator_cleaned = threading.Event()
    executor_started = threading.Event()
    release_executor = threading.Event()
    executor_finished = threading.Event()
    generators = []
    tasks = []

    def executor_work():
        executor_started.set()
        release_executor.wait()
        executor_finished.set()

    async def generator():
        try:
            yield 1
        finally:
            await asyncio.sleep(0)
            generator_cleaned.set()

    async def pending():
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            task_cleaned.set()

    async def setup():
        gen = generator()
        generators.append(gen)  # Keep it open until shutdown_asyncgens.
        await anext(gen)
        tasks.append(asyncio.create_task(pending()))
        asyncio.get_running_loop().run_in_executor(None, executor_work)

    stop = None
    try:
        await controller._run_coroutine_in_controller_thread(setup())
        assert await asyncio.to_thread(executor_started.wait, 2)
        stop = asyncio.create_task(controller.stop())
        assert await asyncio.to_thread(generator_cleaned.wait, 2)
        assert task_cleaned.is_set()
        assert not stop.done()
    finally:
        release_executor.set()
        await asyncio.wait_for(stop if stop is not None else controller.stop(), timeout=3)
    assert executor_finished.is_set()
    assert controller._loop.is_closed()
    assert not controller._thread.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("active_submit", [False, True])
async def test_remote_stop_drains_submit_thread(monkeypatch, active_submit):
    started = threading.Event()
    cleaned = threading.Event()

    async def submit(self, task):
        started.set()
        try:
            if active_submit:
                await asyncio.Event().wait()
        finally:
            # Core must still be usable while submit cancellation unwinds.
            await self._run_coroutine_in_controller_thread(asyncio.sleep(0))
            cleaned.set()

    monkeypatch.setattr(RemoteController, "submit", submit)
    controller = RemoteController(make_client(), workers=1)
    future = controller.submit_sync(None)
    try:
        assert await asyncio.to_thread(started.wait, 2)
        if not active_submit:
            await asyncio.wrap_future(future)
    finally:
        await asyncio.wait_for(asyncio.gather(controller.stop(), controller.stop()), timeout=3)
    assert cleaned.is_set()
    assert future.done()
    assert not controller._submit_thread.is_alive()
    assert not controller._thread.is_alive()
    assert controller._submit_loop.is_closed()
    assert controller._loop.is_closed()
    await controller.stop()
    with pytest.raises(RuntimeError, match="stopped or stopping"):
        controller.submit_sync(None)


@pytest.mark.asyncio
async def test_remote_stop_without_submit_thread():
    controller = RemoteController(make_client(), workers=1)
    await asyncio.wait_for(controller.stop(), timeout=3)
    assert controller._submit_thread is None
    assert not controller._thread.is_alive()
    with pytest.raises(RuntimeError, match="stopped or stopping"):
        await controller.submit(None)


@pytest.mark.asyncio
async def test_cancelled_remote_stop_still_waits_for_submission_cleanup(monkeypatch):
    started = threading.Event()
    cleaning = threading.Event()
    release = threading.Event()
    cleaned = threading.Event()

    async def submit(self, task):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await asyncio.to_thread(release.wait)
            await self._run_coroutine_in_controller_thread(asyncio.sleep(0))
            cleaned.set()

    monkeypatch.setattr(RemoteController, "submit", submit)
    controller = RemoteController(make_client(), workers=1)
    controller.submit_sync(None)
    stop = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        stop = asyncio.create_task(controller.stop())
        assert await asyncio.to_thread(cleaning.wait, 2)
        stop.cancel()
        await asyncio.sleep(0)
        assert not stop.done()
        assert not controller._stopping
    finally:
        release.set()
        if stop is not None:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(stop, timeout=3)
        else:
            await controller.stop()
    assert cleaned.is_set()
    assert not controller._submit_thread.is_alive()
    assert not controller._thread.is_alive()


@pytest.mark.asyncio
async def test_watch_failure_is_preserved_during_shutdown():
    from flyte.errors import RuntimeSystemError

    controller = Controller(make_client(), workers=1)
    watcher = asyncio.create_task(controller.watch_for_errors())

    async def fail():
        raise ValueError("watch failed")

    async def setup():
        task = asyncio.create_task(fail())
        task.add_done_callback(controller._bg_handle_informer_error)
        await asyncio.sleep(0)

    try:
        await controller._run_coroutine_in_controller_thread(setup())
        with pytest.raises(RuntimeSystemError, match="watch failed"):
            await asyncio.wait_for(watcher, timeout=3)
    finally:
        await asyncio.wait_for(controller.stop(), timeout=3)
    assert not controller._thread.is_alive()
    assert controller._loop.is_closed()


@pytest.mark.asyncio
async def test_worker_pool_failure_drains_loop_and_allows_stop(monkeypatch):
    fail_worker = threading.Event()
    cleanup = threading.Event()

    async def worker(self, name):
        try:
            await asyncio.to_thread(fail_worker.wait)
            raise ValueError("worker failed")
        finally:
            await asyncio.sleep(0)
            cleanup.set()

    monkeypatch.setattr(Controller, "_bg_run", worker)
    controller = Controller(make_client(), workers=1)
    fail_worker.set()
    await asyncio.wait_for(asyncio.to_thread(controller._thread.join), timeout=3)
    await controller.stop()
    assert cleanup.is_set()
    assert controller._get_exception() is not None
    assert controller._loop.is_closed()
    assert controller._resource_log_task.done()
