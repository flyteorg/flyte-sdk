from __future__ import annotations

import asyncio
from asyncio import Queue
from typing import AsyncIterator, Callable, Dict, Optional, Tuple, cast

import grpc.aio
from flyteidl2.common import identifier_pb2
from flyteidl2.workflow import run_definition_pb2, state_service_pb2

from flyte._logging import log, logger

from ._action import Action
from ._service_protocol import StateService


class ActionCache:
    """
    Cache for actions, used to store the state of all sub-actions, launched by this parent action.
    This is coroutine-safe.
    """

    def __init__(self, parent_action_name: str):
        # Cache for actions (sub-actions)
        self._cache: Dict[str, Action] = {}
        # Completion events for actions
        self._completion_events: Dict[str, asyncio.Event] = {}
        # Lock for coroutine safety
        self._lock = asyncio.Lock()
        # Parent action name
        self._parent_action_name = parent_action_name

    async def has(self, name: str) -> bool:
        """Check if a node is in the cache"""
        async with self._lock:
            return name in self._cache

    async def observe_state(self, state: state_service_pb2.ActionUpdate) -> Action:
        """
        Add an action to the cache if it doesn't exist. This is invoked by the watch.
        """
        logger.debug(f"Observing phase {run_definition_pb2.Phase.Name(state.phase)} for {state.action_id.name}")
        if state.output_uri:
            logger.debug(f"Output URI: {state.output_uri}")
        else:
            logger.warning(
                f"{state.action_id.name} has no output URI, in phase {run_definition_pb2.Phase.Name(state.phase)}"
            )
        if state.phase == run_definition_pb2.Phase.PHASE_FAILED:
            logger.error(
                f"Action {state.action_id.name} failed with error (msg):"
                f" [{state.error if state.HasField('error') else None}]"
            )
        async with self._lock:
            if state.action_id.name in self._cache:
                self._cache[state.action_id.name].merge_state(state)
            else:
                self._cache[state.action_id.name] = Action.from_state(self._parent_action_name, state)
            return self._cache[state.action_id.name]

    async def submit(self, action: Action) -> Action:
        """
        Submit a new Action to the cache. This is invoked by the parent_action.
        """
        async with self._lock:
            if action.name in self._cache:
                self._cache[action.name].merge_in_action_from_submit(action)
            else:
                self._cache[action.name] = action
            if action.name not in self._completion_events:
                self._completion_events[action.name] = asyncio.Event()
            return self._cache[action.name]

    async def get(self, name: str) -> Action | None:
        """Get an action by its name from the cache"""
        async with self._lock:
            return self._cache.get(name, None)

    async def remove(self, name: str) -> Action | None:
        """Remove an action from the cache"""
        async with self._lock:
            return self._cache.pop(name, None)

    async def wait_for_completion(self, name: str) -> bool:
        """Wait for an action to complete"""
        async with self._lock:
            if name not in self._completion_events:
                return False
            event = self._completion_events[name]
        return await event.wait()

    async def fire_all_completion_events(self):
        """Fire all completion events"""
        async with self._lock:
            for name, event in self._completion_events.items():
                event.set()
            self._completion_events.clear()

    async def fire_completion_event(self, name: str):
        """Fire a completion event for an action"""
        async with self._lock:
            if name in self._completion_events:
                self._completion_events[name].set()

    async def count_started_pending_terminal_actions(self) -> Tuple[int, int, int]:
        """
        Get all started, pending and terminal actions.
        Started: implies they were submitted to queue service
        Pending: implies they are still not submitted to the queue service
        Terminal: implies completed (success, failure, aborted, timedout) actions
        """
        started = 0
        pending = 0
        terminal = 0
        async with self._lock:
            for name, res in self._cache.items():
                if res.is_started():
                    started += 1
                elif res.is_terminal():
                    terminal += 1
                else:
                    pending += 1
            return started, pending, terminal


class Informer:
    """Remote StateStore watcher and informer for sub-actions."""

    def __init__(
        self,
        run_id: identifier_pb2.RunIdentifier,
        parent_action_name: str,
        shared_queue: Queue,
        client: Optional[StateService] = None,
        min_watch_backoff: float = 1.0,
        max_watch_backoff: float = 30.0,
        watch_conn_timeout_sec: float = 5.0,
        max_watch_retries: int = 10,
    ):
        self.name = self.mkname(run_name=run_id.name, parent_action_name=parent_action_name)
        self.parent_action_name = parent_action_name
        self._run_id = run_id
        self._client = client
        self._action_cache = ActionCache(parent_action_name)
        self._shared_queue = shared_queue
        self._running = False
        self._watch_task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._min_watch_backoff = min_watch_backoff
        self._max_watch_backoff = max_watch_backoff
        self._watch_conn_timeout_sec = watch_conn_timeout_sec
        self._max_watch_retries = max_watch_retries

    @classmethod
    def mkname(cls, *, run_name: str, parent_action_name: str) -> str:
        """Get the name of the informer"""
        return f"{run_name}.{parent_action_name}"

    @property
    def watch_task(self) -> asyncio.Task | None:
        """Get the watch task"""
        return self._watch_task

    def is_running(self) -> bool:
        """Check if informer is running"""
        return self._running

    async def _set_ready(self):
        """Set the informer as ready"""
        self._ready.set()

    async def wait_for_cache_sync(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the informer to be ready. In the case of a timeout, it will return False.
        :param timeout: float time to wait for
        :return: bool
        """
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.error(f"Informer cache sync timed out, for {self.name}")
            return False

    async def wait_for_action_completion(self, name: str) -> bool:
        """Wait for an action to complete"""
        return await self._action_cache.wait_for_completion(name)

    async def fire_completion_event(self, name: str):
        """Fire a completion event for an action"""
        await self._action_cache.fire_completion_event(name)

    @log
    async def submit(self, action: Action):
        """Add a new resource to watch"""
        node = await self._action_cache.submit(action)
        await self._shared_queue.put(node)

    @log
    async def remove(self, name: str):
        """Remove a resource from watching"""
        await self._action_cache.remove(name)

    async def get(self, name: str) -> Action | None:
        """Get a resource by name"""
        return await self._action_cache.get(name)

    async def has(self, name: str) -> bool:
        """Check if a resource exists"""
        return await self._action_cache.has(name)

    async def watch(self):
        """
        Watch for updates on all resources - to be implemented by subclasses for watch mode
        """
        # sentinel = False
        retries = 0
        last_exc = None
        while self._running:
            if retries >= self._max_watch_retries:
                logger.error(
                    f"Informer watch failure retries crossed threshold {retries}/{self._max_watch_retries}, exiting!"
                )
                raise last_exc
            try:
                if retries >= 1:
                    logger.warning(f"Informer watch retrying, attempt {retries}/{self._max_watch_retries}")
                watcher = self._client.Watch(
                    state_service_pb2.WatchRequest(
                        parent_action_id=identifier_pb2.ActionIdentifier(
                            name=self.parent_action_name,
                            run=self._run_id,
                        ),
                    ),
                    wait_for_ready=True,
                )
                resp: state_service_pb2.WatchResponse
                async for resp in watcher:
                    retries = 0
                    if resp.control_message is not None and resp.control_message.sentinel:
                        logger.info(f"Received Sentinel, for run {self.name}")
                        await self._set_ready()
                        continue
                    node = await self._action_cache.observe_state(resp.action_update)
                    await self._shared_queue.put(node)
                    # hack to work in the absence of sentinel
            except asyncio.CancelledError:
                logger.info(f"Watch cancelled: {self.name}")
                return
            except asyncio.TimeoutError as e:
                logger.error(f"Watch timeout: {self.name}", exc_info=e)
                last_exc = e
                retries += 1
            except grpc.aio.AioRpcError as e:
                logger.exception(f"RPC error: {self.name}", exc_info=e)
                last_exc = e
                retries += 1
            except Exception as e:
                logger.exception(f"Watch error: {self.name}", exc_info=e)
                last_exc = e
                retries += 1
            backoff = min(self._min_watch_backoff * (2**retries), self._max_watch_backoff)
            logger.warning(f"Watch for {self.name} failed, retrying in {backoff} seconds...")
            await asyncio.sleep(backoff)

    @log
    async def start(self, timeout: Optional[float] = None) -> asyncio.Task:
        """Start the informer"""
        if self._running:
            logger.warning("Informer already running")
            return cast(asyncio.Task, self._watch_task)
        self._running = True
        self._watch_task = asyncio.create_task(self.watch(), name=f"InformerWatch-{self.parent_action_name}")
        await self.wait_for_cache_sync(timeout=timeout)
        return self._watch_task

    async def count_started_pending_terminal_actions(self) -> Tuple[int, int, int]:
        """Get all launched and waiting resources"""
        return await self._action_cache.count_started_pending_terminal_actions()

    @log
    async def stop(self):
        """Stop the informer"""
        self._running = False
        if self._watch_task:
            self._watch_task.cancel()
            self._watch_task = None
        logger.info("Stopped informer")


class InformerCache:
    """Cache for informers, used to store the state of all subactions for multiple parent_actions.
    This is coroutine-safe.
    """

    def __init__(self):
        self._cache: Dict[str, Informer] = {}
        self._lock = asyncio.Lock()

    @log
    async def get_or_create(
        self,
        run_id: identifier_pb2.RunIdentifier,
        parent_action_name: str,
        shared_queue: Queue,
        state_service: StateService,
        fn: Callable[[asyncio.Task], None],
        timeout: Optional[float] = None,
    ) -> Informer:
        """
        Start and add a new informer to the cache
        :param run_id: Run ID
        :param parent_action_name: Parent action name
        :param shared_queue: Shared queue
        :param state_service: State service
        :param fn: Callback function to be called when the informer is done
        :param timeout: Timeout for the informer to be ready
        :return: Tuple of informer and a boolean indicating if it was created. True if created, false if already exists.
        """
        name = Informer.mkname(run_name=run_id.name, parent_action_name=parent_action_name)
        async with self._lock:
            if name in self._cache:
                return self._cache[name]
            informer = Informer(
                run_id=run_id,
                parent_action_name=parent_action_name,
                shared_queue=shared_queue,
                client=state_service,
            )
            self._cache[informer.name] = informer
            # TODO This is a potential perf problem for large number of informers.
            # We can start in only if it is not started. Reason to do this overly optimistic is to avoid,
            # remove from removing the cache.
            task = await informer.start(timeout=timeout)
            if task is None:
                logger.error(f"Informer {name} failed to start")
                raise RuntimeError(f"Informer {name} failed to start")
            task.add_done_callback(fn)
            return informer

    @log
    async def get(self, *, run_name: str, parent_action_name: str) -> Informer | None:
        """Get an informer by name"""
        async with self._lock:
            return self._cache.get(
                Informer.mkname(run_name=run_name, parent_action_name=parent_action_name),
                None,
            )

    @log
    async def remove(self, *, run_name: str, parent_action_name: str) -> Informer | None:
        """Remove an informer from the cache"""
        async with self._lock:
            return self._cache.pop(
                Informer.mkname(run_name=run_name, parent_action_name=parent_action_name),
                None,
            )

    async def has(self, *, run_name: str, parent_action_name: str) -> bool:
        """Check if an informer exists in the cache"""
        async with self._lock:
            return Informer.mkname(run_name=run_name, parent_action_name=parent_action_name) in self._cache

    async def count_started_pending_terminal_actions(
        self,
    ) -> AsyncIterator[Tuple[int, int, int]]:
        """Log resource stats"""
        async with self._lock:
            for informer in self._cache.values():
                yield await informer.count_started_pending_terminal_actions()

    async def remove_and_stop_all(self):
        """Stop all informers and remove them from the cache"""
        async with self._lock:
            while self._cache:
                _name, informer = self._cache.popitem()
                try:
                    await informer.stop()
                except asyncio.CancelledError:
                    pass
            self._cache.clear()
