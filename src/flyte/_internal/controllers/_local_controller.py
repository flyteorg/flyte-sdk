import asyncio
import atexit
import concurrent.futures
import os
import pathlib
import shutil
import threading
from contextlib import nullcontext
from typing import Any, Callable, Tuple, TypeVar

import flyte.errors
from flyte._cache.cache import VersionParameters, cache_from_request
from flyte._context import internal_ctx
from flyte._internal.controllers import TaskCallSequencer, TraceInfo
from flyte._internal.runtime import convert
from flyte._internal.runtime.entrypoints import direct_dispatch
from flyte._internal.runtime.types_serde import transform_native_to_typed_interface
from flyte._logging import log, logger
from flyte._persistence._recorder import RunRecorder
from flyte._persistence._task_cache import LocalTaskCache
from flyte._task import AsyncFunctionTaskTemplate, TaskTemplate
from flyte._utils.helpers import _selector_policy
from flyte.models import ActionID, CheckpointPaths, NativeInterface
from flyte.remote._task import TaskDetails
from flyte.storage._storage import strip_file_header

R = TypeVar("R")

# Local retry backoff defaults for task errors during local runs. Do not
# expose these to the user since RetryStrategy only supports a count of retries.
# This is because currently the backend implements the underlying retry strategy,
# and does not allow for custom backoff strategies.
_MIN_BACKOFF_ON_ERR_SEC = 0.5
_BACKOFF_MULTIPLIER = 2.0


def _stage_prev_checkpoint_for_local_retry(checkpoint_paths: CheckpointPaths | None) -> None:
    """
    Before a local retry, copy the last attempt's checkpoint object into ``prev_checkpoint`` so
    :class:`~flyte.Checkpoint` can load it (mirrors remote behavior where the platform stages prior output).
    """
    if checkpoint_paths is None:
        return
    dest = checkpoint_paths.checkpoint_path
    prev = checkpoint_paths.prev_checkpoint_path
    if not dest or not prev:
        return
    src = pathlib.Path(strip_file_header(str(dest)))
    dst = pathlib.Path(strip_file_header(str(prev)))
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


class _TaskRunner:
    """A task runner that runs an asyncio event loop on a background thread."""

    def __init__(self) -> None:
        self.__loop: asyncio.AbstractEventLoop | None = None
        self.__runner_thread: threading.Thread | None = None
        self.__lock = threading.Lock()
        atexit.register(self._close)

    def _close(self) -> None:
        if self.__loop:
            self.__loop.stop()

    def _execute(self) -> None:
        loop = self.__loop
        assert loop is not None
        try:
            loop.run_forever()
        finally:
            loop.close()

    def get_exc_handler(self):
        def exc_handler(loop, context):
            logger.error(
                f"Taskrunner for {self.__runner_thread.name if self.__runner_thread else 'no thread'} caught"
                f" exception in {loop}: {context}"
            )

        return exc_handler

    def get_run_future(self, coro: Any) -> concurrent.futures.Future:
        """Synchronously run a coroutine on a background thread."""
        name = f"{threading.current_thread().name} : loop-runner"
        with self.__lock:
            if self.__loop is None:
                with _selector_policy():
                    self.__loop = asyncio.new_event_loop()

                exc_handler = self.get_exc_handler()
                self.__loop.set_exception_handler(exc_handler)
                self.__runner_thread = threading.Thread(target=self._execute, daemon=True, name=name)
                self.__runner_thread.start()
        fut = asyncio.run_coroutine_threadsafe(coro, self.__loop)
        return fut


class LocalController:
    def __init__(self):
        logger.debug("LocalController init")
        self._runner_map: dict[str, _TaskRunner] = {}
        self._sequencer = TaskCallSequencer()
        self._recorder = RunRecorder()
        self._registered_events: dict[str, Any] = {}

    def set_recorder(self, recorder: RunRecorder) -> None:
        self._recorder = recorder

    @log
    async def submit(self, _task: TaskTemplate, *args, **kwargs) -> Any:
        """
        Main entrypoint for submitting a task to the local controller.
        """
        ctx = internal_ctx()
        tctx = ctx.data.task_context
        if not tctx:
            raise flyte.errors.RuntimeSystemError("BadContext", "Task context not initialized")

        _ctx = ctx.new_in_driver_literal_conversion(True) if ctx.is_task_context() else nullcontext()
        with _ctx:
            inputs = await convert.convert_from_native_to_inputs(_task.native_interface, *args, **kwargs)
        inputs_hash = convert.generate_inputs_hash_from_proto(inputs.proto_inputs)
        task_interface = transform_native_to_typed_interface(_task.interface)

        task_call_seq = self._sequencer.next_seq(_task, tctx.action.name)
        sub_action_id, sub_action_output_path = convert.generate_sub_action_id_and_output_path(
            tctx, _task.name, inputs_hash, task_call_seq
        )
        sub_action_raw_data_path = tctx.raw_data_path
        # Make sure the output path exists
        pathlib.Path(sub_action_output_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(sub_action_raw_data_path.path).mkdir(parents=True, exist_ok=True)

        task_cache = cache_from_request(_task.cache)
        cache_enabled = task_cache.is_enabled()
        if isinstance(_task, AsyncFunctionTaskTemplate):
            version_parameters = VersionParameters(func=_task.func, image=_task.image)
        else:
            version_parameters = VersionParameters(func=None, image=_task.image)
        cache_version = task_cache.get_version(version_parameters)
        cache_key = convert.generate_cache_key_hash(
            _task.name,
            inputs_hash,
            task_interface,
            cache_version,
            list(task_cache.get_ignored_inputs()),
            inputs.proto_inputs,
        )

        out = None
        cache_hit = False
        # We only get output from cache if the cache behavior is set to auto
        # and run cache is not disabled
        if task_cache.behavior == "auto" and not tctx.disable_run_cache:
            out = await LocalTaskCache.get(cache_key)
            if out is not None:
                cache_hit = True
                logger.info(
                    f"Cache hit for task '{_task.name}' (version: {cache_version}), getting result from cache..."
                )

        # Build common metadata for the recorder (tracker + persistence).
        native_inputs: dict[str, Any] | None = None
        parent_id: str | None = None
        rendered_links: list[tuple[str, str]] | None = None

        if self._recorder.is_active:
            param_names = list(_task.native_interface.inputs.keys())
            native_inputs = {}
            for i, arg in enumerate(args):
                if i < len(param_names):
                    native_inputs[param_names[i]] = arg
            native_inputs.update(kwargs)

            # If the parent action isn't tracked yet, this is the top-level call
            parent_id = tctx.action.name if self._recorder.get_action(tctx.action.name) else None

            # Render log links for this action, replacing template placeholders
            # with concrete local values (see task_serde.py for remote equivalents).
            if _task.links:
                rendered_links = []
                action = tctx.action
                for link in _task.links:
                    uri = link.get_link(
                        run_name=action.run_name or "",
                        project=action.project or "",
                        domain=action.domain or "",
                        context=tctx.custom_context or {},
                        parent_action_name=action.name or "",
                        action_name=sub_action_id.name,
                        pod_name="localhost",
                    )
                    rendered_links.append((link.name, uri))

        # When run cache is disabled, never report cache hit to the TUI
        effective_cache_hit = cache_hit if not tctx.disable_run_cache else False

        self._recorder.record_start(
            action_id=sub_action_id.name,
            task_name=_task.name,
            short_name=_task.short_name if _task.short_name != _task.name else None,
            parent_id=parent_id,
            inputs=native_inputs,
            output_path=sub_action_output_path,
            has_report=_task.report,
            cache_enabled=cache_enabled,
            cache_hit=effective_cache_hit,
            disable_run_cache=tctx.disable_run_cache,
            context=tctx.custom_context or None,
            group=tctx.group_data.name if tctx.group_data else None,
            log_links=rendered_links,
        )

        if out is None:
            retries = _task.retries.count if hasattr(_task.retries, "count") else int(_task.retries)
            max_attempts = retries + 1
            err = None
            for attempt_num in range(1, max_attempts + 1):
                if attempt_num > 1:
                    _stage_prev_checkpoint_for_local_retry(tctx.checkpoint_paths)
                self._recorder.record_attempt_start(
                    action_id=sub_action_id.name,
                    attempt_num=attempt_num,
                )
                out, err = await direct_dispatch(
                    _task,
                    controller=self,
                    action=sub_action_id,
                    raw_data_path=sub_action_raw_data_path,
                    inputs=inputs,
                    version=cache_version,
                    checkpoint_paths=tctx.checkpoint_paths,
                    code_bundle=tctx.code_bundle,
                    output_path=sub_action_output_path,
                    run_base_dir=tctx.run_base_dir,
                )
                if not err:
                    self._recorder.record_attempt_complete(
                        action_id=sub_action_id.name,
                        attempt_num=attempt_num,
                        outputs=out,
                    )
                    break
                self._recorder.record_attempt_failure(
                    action_id=sub_action_id.name,
                    attempt_num=attempt_num,
                    error=str(err),
                )
                if not err.recoverable:
                    logger.warning(
                        f"Task '{_task.name}' raised a non-recoverable error on attempt "
                        f"{attempt_num}/{max_attempts}, skipping remaining retries."
                    )
                    break
                if attempt_num < max_attempts:
                    backoff = _MIN_BACKOFF_ON_ERR_SEC * (_BACKOFF_MULTIPLIER ** (attempt_num - 1))
                    logger.warning(
                        f"Task '{_task.name}' action '{sub_action_id.name}' failed on attempt "
                        f"{attempt_num}/{max_attempts}; retrying in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)

            if err:
                self._recorder.record_failure(action_id=sub_action_id.name, error=str(err))
                exc = convert.convert_error_to_native(err)
                if exc:
                    raise exc
                else:
                    raise flyte.errors.RuntimeSystemError("BadError", "Unknown error")

            # store into cache (skip when run cache is disabled)
            if cache_enabled and out is not None and not tctx.disable_run_cache:
                await LocalTaskCache.set(cache_key, out)

        self._recorder.record_complete(action_id=sub_action_id.name, outputs=out)

        if _task.native_interface.outputs:
            if out is None:
                raise flyte.errors.RuntimeSystemError("BadOutput", "Task output not captured.")
            result = await convert.convert_outputs_to_native(_task.native_interface, out)
            return result
        return None

    def submit_sync(self, _task: TaskTemplate, *args, **kwargs) -> concurrent.futures.Future:
        name = threading.current_thread().name + f"PID:{os.getpid()}"
        coro = self.submit(_task, *args, **kwargs)
        if name not in self._runner_map:
            if len(self._runner_map) > 100:
                logger.warning(
                    "More than 100 event loop runners created!!! This could be a case of runaway recursion..."
                )
            self._runner_map[name] = _TaskRunner()

        return self._runner_map[name].get_run_future(coro)

    async def finalize_parent_action(self, action: ActionID):
        pass

    async def stop(self):
        await LocalTaskCache.close()

    async def watch_for_errors(self):
        try:
            await asyncio.Event().wait()  # Wait indefinitely until cancelled
        except asyncio.CancelledError:
            return  # Return with no errors when cancelled

    async def get_action_outputs(
        self, _interface: NativeInterface, _func: Callable, *args, **kwargs
    ) -> Tuple[TraceInfo, bool]:
        """
        This method returns the outputs of the action, if it is available.
        If not available it raises a  flyte.errors.ActionNotFoundError.
        :return:
        """
        ctx = internal_ctx()
        tctx = ctx.data.task_context
        if not tctx:
            raise flyte.errors.NotInTaskContextError("BadContext", "Task context not initialized")

        converted_inputs = convert.Inputs.empty()
        if _interface.inputs:
            _ctx = ctx.new_in_driver_literal_conversion(True) if ctx.is_task_context() else nullcontext()
            with _ctx:
                converted_inputs = await convert.convert_from_native_to_inputs(_interface, *args, **kwargs)
            assert converted_inputs

        inputs_hash = convert.generate_inputs_hash_from_proto(converted_inputs.proto_inputs)
        invoke_seq_num = self._sequencer.next_seq(_func, tctx.action.name)
        action_id, action_output_path = convert.generate_sub_action_id_and_output_path(
            tctx,
            _func.__name__,
            inputs_hash,
            invoke_seq_num,
        )
        assert action_output_path

        if self._recorder.is_active:
            native_inputs: dict[str, Any] = {}
            param_names = list(_interface.inputs.keys())
            for i, arg in enumerate(args):
                if i < len(param_names):
                    native_inputs[param_names[i]] = arg
            native_inputs.update(kwargs)
            self._recorder.record_start(
                action_id=action_id.name,
                task_name=_func.__name__,
                parent_id=tctx.action.name,
                inputs=native_inputs,
                output_path=action_output_path,
            )

        return (
            TraceInfo(
                name=_func.__name__,
                action=action_id,
                interface=_interface,
                inputs_path=action_output_path,
            ),
            True,
        )

    async def record_trace(self, info: TraceInfo):
        """
        This method records the trace of the action.
        :param info: Trace information
        :return:
        """
        ctx = internal_ctx()
        tctx = ctx.data.task_context
        if not tctx:
            raise flyte.errors.NotInTaskContextError("BadContext", "Task context not initialized")

        if info.error:
            # If there is an error, convert it to a native error
            converted_error = convert.convert_from_native_to_error(info.error)
            assert converted_error
            self._recorder.record_failure(action_id=info.action.name, error=str(info.error))
        else:
            converted_outputs = None
            if info.interface.outputs and info.output:
                _ctx = ctx.new_in_driver_literal_conversion(True) if ctx.is_task_context() else nullcontext()
                with _ctx:
                    converted_outputs = await convert.convert_from_native_to_outputs(
                        info.output, info.interface, info.name
                    )
                assert converted_outputs
            self._recorder.record_complete(action_id=info.action.name, outputs=converted_outputs)
        assert info.action
        assert info.start_time
        assert info.end_time

    async def submit_task_ref(self, _task: TaskDetails, max_inline_io_bytes: int, *args, **kwargs) -> Any:
        raise flyte.errors.RemoteTaskUsageError(
            f"Remote tasks cannot be executed locally, only remotely. Found remote task {_task.name}"
        )

    async def register_event(self, event: Any):
        """
        Register an event that can be awaited. Stores the event for later retrieval.
        If the event has a webhook configured, fires it asynchronously.

        :param event: Event object to register
        """
        from flyte._event import _Event

        if not isinstance(event, _Event):
            raise TypeError(f"Expected _Event, got {type(event)}")

        logger.debug(f"Registering event: {event.name}")
        self._registered_events[event.name] = event

        if event.webhook is not None:
            await self._fire_event_webhook(event)

    async def _fire_event_webhook(self, event: Any):
        """Fire the webhook associated with an event.

        Substitutes ``{callback_uri}`` in all string values of the payload, then
        POSTs the JSON body to the webhook URL.
        """
        import httpx

        webhook = event.webhook
        callback_uri = f"local://events/{event.name}/signal"

        payload = webhook.payload
        if payload is not None:
            payload = _substitute_callback_uri(payload, callback_uri)

        logger.debug(f"Firing webhook for event '{event.name}' to {webhook.url}")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    webhook.url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                logger.debug(f"Webhook response for event '{event.name}': {resp.status_code}")
        except Exception:
            logger.exception(f"Failed to fire webhook for event '{event.name}'")

    def _get_current_action_id(self) -> str:
        ctx = internal_ctx()
        tctx = ctx.data.task_context
        if tctx is None:
            raise flyte.errors.RuntimeSystemError("BadContext", "Task context not initialized")
        return tctx.action.name

    async def wait_for_event(self, event: Any) -> Any:
        """
        Wait for an event to be signaled.

        In TUI mode, records a pending event so the TUI can render an input panel and
        blocks until the user submits a value. Without TUI, falls back to rich console prompts.

        :param event: Event object to wait for
        :return: The payload associated with the event when it is signaled
        """
        from flyte._event import _Event

        if not isinstance(event, _Event):
            raise TypeError(f"Expected _Event, got {type(event)}")

        logger.info(f"Waiting for event: {event.name}")

        action_id = self._get_current_action_id()
        pending = self._recorder.record_event_waiting(
            action_id=action_id,
            event_name=event.name,
            prompt=event.prompt,
            prompt_type=event.prompt_type,
            data_type=event.data_type,
            description=event.description,
        )

        timeout_seconds = event._timeout_seconds

        if pending is not None:
            # TUI mode: block until the TUI resolves the event
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, pending.wait_for_result, timeout_seconds)
            if pending.timed_out:
                raise flyte.errors.EventTimedoutError(
                    f"Event '{event.name}' was not signaled within {timeout_seconds} seconds."
                )
            if result is None:
                raise RuntimeError(f"Event '{event.name}' was cancelled (TUI quit).")
            return result

        # Non-TUI mode: fall back to rich console prompts
        if timeout_seconds is not None:
            loop = asyncio.get_event_loop()
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, self._prompt_event_console, event),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                raise flyte.errors.EventTimedoutError(
                    f"Event '{event.name}' was not signaled within {timeout_seconds} seconds."
                )
        return self._prompt_event_console(event)

    @staticmethod
    def _prompt_event_console(event: Any) -> Any:
        from rich.console import Console
        from rich.prompt import Confirm, Prompt

        console = Console()
        console.print(f"\n[bold cyan]Event:[/bold cyan] {event.name}")
        if event.description:
            console.print(f"[dim]{event.description}[/dim]")

        if event.data_type is bool:
            result = Confirm.ask(event.prompt, console=console)
        elif event.data_type in (int, float, str):
            while True:
                try:
                    value = Prompt.ask(event.prompt, console=console)
                    result = event.data_type(value)
                    break
                except ValueError:
                    type_name = event.data_type.__name__
                    console.print(f"[red]Please enter a valid {type_name}[/red]")
        else:
            raise ValueError(f"Unsupported data type {event.data_type}")

        logger.debug(f"Event {event.name} received value: {result}")
        return result


def _substitute_callback_uri(obj: Any, callback_uri: str) -> Any:
    """Recursively replace ``{callback_uri}`` in all string values."""
    if isinstance(obj, str):
        return obj.replace("{callback_uri}", callback_uri)
    if isinstance(obj, dict):
        return {k: _substitute_callback_uri(v, callback_uri) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_callback_uri(item, callback_uri) for item in obj]
    return obj
