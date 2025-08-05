from __future__ import annotations

import asyncio
import sys
import threading
from asyncio import Event
from typing import Awaitable, Coroutine, Optional

import grpc.aio
from google.protobuf.wrappers_pb2 import StringValue

import flyte.errors
from flyte._logging import log, logger
from flyte._protos.common import identifier_pb2
from flyte._protos.workflow import (
    queue_service_pb2,
    task_definition_pb2,
)
from flyte.errors import RuntimeSystemError

from ._action import Action
from ._informer import InformerCache
from ._service_protocol import ClientSet, QueueService, StateService


class Controller:
    """
    Generic controller with high-level submit API running in a dedicated thread with its own event loop.
    All methods that begin with _bg_ are run in the controller's event loop, and will need to use
    _run_coroutine_in_controller_thread to run them in the controller's event loop.
    """

    def __init__(
        self,
        client_coro: Awaitable[ClientSet],
        workers: int = 2,
        max_system_retries: int = 5,
        resource_log_interval_sec: float = 10.0,
        min_backoff_on_err_sec: float = 0.1,
        thread_wait_timeout_sec: float = 5.0,
        enqueue_timeout_sec: float = 5.0,
    ):
        """
        Create a new controller instance.
        :param workers: Number of worker threads.
        :param max_system_retries: Maximum number of system retries.
        :param resource_log_interval_sec: Interval for logging resource stats.
        :param min_backoff_on_err_sec: Minimum backoff time on error.
        :param thread_wait_timeout_sec: Timeout for waiting for the controller thread to start.
        :param
        """
        self._informers = InformerCache()
        self._shared_queue: asyncio.Queue[Action] = asyncio.Queue(maxsize=10000)
        self._running = False
        self._resource_log_task = None
        self._workers = workers
        self._max_retries = max_system_retries
        self._resource_log_interval = resource_log_interval_sec
        self._min_backoff_on_err = min_backoff_on_err_sec
        self._thread_wait_timeout = thread_wait_timeout_sec
        self._client_coro = client_coro
        self._failure_event: Event | None = None
        self._enqueue_timeout = enqueue_timeout_sec
        self._informer_start_wait_timeout = thread_wait_timeout_sec

        # Thread management
        self._thread = None
        self._loop = None
        self._thread_ready = threading.Event()
        self._thread_exception: Optional[BaseException] = None
        self._thread_com_lock = threading.Lock()
        self._start()

    # ---------------- Public sync methods, we can add more sync methods if needed
    @log
    def submit_action_sync(self, action: Action) -> Action:
        """Synchronous version of submit that runs in the controller's event loop"""
        fut = self._run_coroutine_in_controller_thread(self._bg_submit_action(action))
        return fut.result()

    # --------------- Public async methods
    @log
    async def submit_action(self, action: Action) -> Action:
        """Public API to submit a resource and wait for completion"""
        return await self._run_coroutine_in_controller_thread(self._bg_submit_action(action))

    async def get_action(self, action_id: identifier_pb2.ActionIdentifier, parent_action_name: str) -> Optional[Action]:
        """Get the action from the informer"""
        return await self._run_coroutine_in_controller_thread(self._bg_get_action(action_id, parent_action_name))

    @log
    async def cancel_action(self, action: Action):
        return await self._run_coroutine_in_controller_thread(self._bg_cancel_action(action))

    async def _finalize_parent_action(
        self,
        run_id: identifier_pb2.RunIdentifier,
        parent_action_name: str,
        timeout: Optional[float] = None,
    ):
        """Finalize the parent run"""
        await self._run_coroutine_in_controller_thread(
            self._bg_finalize_informer(run_id=run_id, parent_action_name=parent_action_name, timeout=timeout)
        )

    def _bg_handle_informer_error(self, task: asyncio.Task):
        """Handle errors in the informer task"""
        try:
            exc = task.exception()
            if exc:
                logger.error("Informer task failed with exception", exc_info=exc)
                self._set_exception(exc)
                if self._failure_event is None:
                    raise RuntimeError("Failure event not initialized")
                self._failure_event.set()
        except asyncio.CancelledError:
            pass

    async def _bg_watch_for_errors(self):
        if self._failure_event is None:
            raise RuntimeError("Failure event not initialized")
        await self._failure_event.wait()
        logger.warning(f"Failure event received: {self._failure_event}, cleaning up informers and exiting.")

    async def watch_for_errors(self):
        """Watch for errors in the background thread"""
        await self._run_coroutine_in_controller_thread(self._bg_watch_for_errors())
        raise RuntimeSystemError(
            code="InformerWatchFailure",
            message=f"Controller thread failed with exception: {self._get_exception()}",
        )

    @log
    async def stop(self):
        """Stop the controller"""
        return await self._run_coroutine_in_controller_thread(self._bg_stop())

    # ------------- Background thread management methods
    def _set_exception(self, exc: Optional[BaseException]):
        """Set exception in the thread lock"""
        with self._thread_com_lock:
            self._thread_exception = exc

    def _get_exception(self) -> Optional[BaseException]:
        """Get exception in the thread lock"""
        with self._thread_com_lock:
            return self._thread_exception

    @log
    def _start(self):
        """Start the controller in a separate thread"""
        if self._thread and self._thread.is_alive():
            logger.warning("Controller thread is already running")
            return

        self._thread_ready.clear()
        self._set_exception(None)
        self._thread = threading.Thread(target=self._bg_thread_target, daemon=True, name="ControllerThread")
        self._thread.start()

        # Wait for the thread to be ready
        logger.info("Waiting for controller thread to be ready...")
        if not self._thread_ready.wait(timeout=self._thread_wait_timeout):
            raise TimeoutError("Controller thread failed to start in time")

        if self._get_exception():
            raise RuntimeSystemError(
                type(self._get_exception()).__name__,
                f"Controller thread startup failed: {self._get_exception()}",
            )

        logger.info(f"Controller started in thread: {self._thread.name}")

    def _run_coroutine_in_controller_thread(self, coro: Coroutine) -> asyncio.Future:
        """Run a coroutine in the controller's event loop and return the result"""
        with self._thread_com_lock:
            loop = self._loop
            if not self._loop or not self._thread or not self._thread.is_alive():
                raise RuntimeError("Controller thread is not running")

        assert self._thread.name != threading.current_thread().name, "Cannot run coroutine in the same thread"

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return asyncio.wrap_future(future)

    # ------------- Private methods that run on the background thread
    async def _bg_worker_pool(self):
        logger.debug("Starting controller worker pool")
        self._running = True
        logger.debug("Waiting for Service Client to be ready")
        client_set = await self._client_coro
        self._state_service: StateService = client_set.state_service
        self._queue_service: QueueService = client_set.queue_service
        self._resource_log_task = asyncio.create_task(self._bg_log_stats())
        # We will wait for this to signal that the thread is ready
        # Signal the main thread that we're ready
        logger.debug("Background thread initialization complete")
        self._thread_ready.set()
        if sys.version_info >= (3, 11):
            async with asyncio.TaskGroup() as tg:
                for i in range(self._workers):
                    tg.create_task(self._bg_run())
        else:
            tasks = []
            for i in range(self._workers):
                tasks.append(asyncio.create_task(self._bg_run()))
            await asyncio.gather(*tasks)

    def _bg_thread_target(self):
        """Target function for the controller thread that creates and manages its own event loop"""
        try:
            # Create a new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            logger.debug(f"Controller thread started with new event loop: {threading.current_thread().name}")

            # Create an event to signal the errors were observed in the thread's loop
            self._failure_event = Event()

            self._loop.run_until_complete(self._bg_worker_pool())
        except Exception as e:
            logger.error(f"Controller thread encountered an exception: {e}")
            self._set_exception(e)
        finally:
            if self._loop and self._loop.is_running():
                self._loop.close()
            logger.debug(f"Controller thread exiting: {threading.current_thread().name}")

    async def _bg_get_action(
        self, action_id: identifier_pb2.ActionIdentifier, parent_action_name: str
    ) -> Optional[Action]:
        """Get the action from the informer"""
        # Ensure the informer is created and wait for it to be ready
        informer = await self._informers.get_or_create(
            action_id.run,
            parent_action_name,
            self._shared_queue,
            self._state_service,
            fn=self._bg_handle_informer_error,
            timeout=self._informer_start_wait_timeout,
        )
        if informer:
            return await informer.get(action_id.name)
        return None

    async def _bg_finalize_informer(
        self,
        run_id: identifier_pb2.RunIdentifier,
        parent_action_name: str,
        timeout: Optional[float] = None,
    ):
        informer = await self._informers.remove(run_name=run_id.name, parent_action_name=parent_action_name)
        if informer:
            await informer.stop()

    @log
    async def _bg_submit_action(self, action: Action) -> Action:
        """Submit a resource and await its completion, returning the final state"""
        logger.debug(f"{threading.current_thread().name} Submitting action {action.name}")
        informer = await self._informers.get_or_create(
            action.action_id.run,
            action.parent_action_name,
            self._shared_queue,
            self._state_service,
            fn=self._bg_handle_informer_error,
            timeout=self._informer_start_wait_timeout,
        )
        await informer.submit(action)

        logger.debug(f"{threading.current_thread().name} Waiting for completion of {action.name}")
        # Wait for completion
        await informer.wait_for_action_completion(action.name)
        logger.info(f"{threading.current_thread().name} Action {action.name} completed")

        # Get final resource state and clean up
        final_resource = await informer.get(action.name)
        if final_resource is None:
            raise ValueError(f"Action {action.name} not found")
        logger.debug(f"{threading.current_thread().name} Removed completion event for action {action.name}")
        await informer.remove(action.name)  # TODO we should not remove maybe, we should keep a record of completed?
        logger.debug(f"{threading.current_thread().name} Removed action {action.name}, final={final_resource}")
        return final_resource

    async def _bg_cancel_action(self, action: Action):
        """
        Cancel an action.
        """
        if action.is_terminal():
            logger.info(f"Action {action.name} is already terminal, no need to cancel.")
            return

        started = action.is_started()
        action.mark_cancelled()
        if started:
            logger.info(f"Cancelling action: {action.name}")
            try:
                # TODO add support when the queue service supports aborting actions
                # await self._queue_service.AbortQueuedAction(
                #     queue_service_pb2.AbortQueuedActionRequest(action_id=action.action_id),
                #     wait_for_ready=True,
                # )
                logger.info(f"Successfully cancelled action: {action.name}")
            except grpc.aio.AioRpcError as e:
                if e.code() in [
                    grpc.StatusCode.NOT_FOUND,
                    grpc.StatusCode.FAILED_PRECONDITION,
                ]:
                    logger.info(f"Action {action.name} not found, assumed completed or cancelled.")
                    return
        else:
            # If the action is not started, we have to ensure it does not get launched
            logger.info(f"Action {action.name} is not started, no need to cancel.")

        informer = await self._informers.get(run_name=action.run_name, parent_action_name=action.parent_action_name)
        if informer:
            await informer.fire_completion_event(action.name)

    async def _bg_launch(self, action: Action):
        """
        Attempt to launch an action.
        """
        if not action.is_started():
            task: queue_service_pb2.TaskAction | None = None
            trace: queue_service_pb2.TraceAction | None = None
            if action.type == "task":
                if action.task is None:
                    raise flyte.errors.RuntimeSystemError(
                        "NoTaskSpec", "Task Spec not found, cannot launch Task Action."
                    )
                cache_key = None
                logger.info(f"Action {action.name} has cache version {action.cache_key}")
                if action.cache_key:
                    cache_key = StringValue(value=action.cache_key)

                task = queue_service_pb2.TaskAction(
                    id=task_definition_pb2.TaskIdentifier(
                        version=action.task.task_template.id.version,
                        org=action.task.task_template.id.org,
                        project=action.task.task_template.id.project,
                        domain=action.task.task_template.id.domain,
                        name=action.task.task_template.id.name,
                    ),
                    spec=action.task,
                    cache_key=cache_key,
                )
            elif action.type == "trace":
                trace = action.trace

            logger.debug(f"Attempting to launch action: {action.name}")
            try:
                await self._queue_service.EnqueueAction(
                    queue_service_pb2.EnqueueActionRequest(
                        action_id=action.action_id,
                        parent_action_name=action.parent_action_name,
                        task=task,
                        trace=trace,
                        input_uri=action.inputs_uri,
                        run_output_base=action.run_output_base,
                        group=action.group.name if action.group else None,
                        # Subject is not used in the current implementation
                    ),
                    wait_for_ready=True,
                    timeout=self._enqueue_timeout,
                )
                logger.info(f"Successfully launched action: {action.name}")
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.ALREADY_EXISTS:
                    logger.info(f"Action {action.name} already exists, continuing to monitor.")
                    return
                logger.exception(f"Failed to launch action: {action.name} backing off...")
                logger.debug(f"Action details: {action}")
                raise e

    @log
    async def _bg_process(self, action: Action):
        """Process resource updates"""
        logger.debug(f"Processing action: name={action.name}, started={action.is_started()}")

        if not action.is_started():
            await self._bg_launch(action)
        elif action.is_terminal():
            informer = await self._informers.get(run_name=action.run_name, parent_action_name=action.parent_action_name)
            if informer:
                await informer.fire_completion_event(action.name)
        else:
            logger.debug(f"Resource {action.name} still in progress...")

    async def _bg_log_stats(self):
        """Periodically log resource stats if debug is enabled"""
        while self._running:
            async for (
                started,
                pending,
                terminal,
            ) in self._informers.count_started_pending_terminal_actions():
                logger.info(f"Resource stats: Started={started}, Pending={pending}, Terminal={terminal}")
            await asyncio.sleep(self._resource_log_interval)

    @log
    async def _bg_run(self):
        """Run loop with resource status logging"""
        while self._running:
            logger.debug(f"{threading.current_thread().name} Waiting for resource")
            action = await self._shared_queue.get()
            logger.debug(f"{threading.current_thread().name} Got resource {action.name}")
            try:
                await self._bg_process(action)
            except Exception as e:
                logger.error(f"Error in controller loop: {e}")
                # TODO we need a better way of handling backoffs currently the entire worker coroutine backs off
                await asyncio.sleep(self._min_backoff_on_err)
                action.increment_retries()
                if action.retries > self._max_retries:
                    err = flyte.errors.RuntimeSystemError(
                        code=type(e).__name__,
                        message=f"Controller failed, system retries {action.retries}"
                        f" crossed threshold {self._max_retries}",
                    )
                    err.__cause__ = e
                    action.set_client_error(err)
                    informer = await self._informers.get(
                        run_name=action.run_name,
                        parent_action_name=action.parent_action_name,
                    )
                    if informer:
                        await informer.fire_completion_event(action.name)
                else:
                    await self._shared_queue.put(action)
            finally:
                self._shared_queue.task_done()

    @log
    async def _bg_stop(self):
        """Stop the controller"""
        self._running = False
        self._resource_log_task.cancel()
        await self._informers.remove_and_stop_all()
