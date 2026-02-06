import asyncio
import atexit
import concurrent.futures
import os
import pathlib
import threading
import io as _io
import contextlib
from typing import Any, Callable, Tuple, TypeVar

import flyte.errors
from flyte._cache.cache import VersionParameters, cache_from_request
from flyte._cache.local_cache import LocalTaskCache
from flyte._context import internal_ctx
from flyte._internal.controllers import TraceInfo
from flyte._internal.runtime import convert
from flyte._internal.runtime.entrypoints import direct_dispatch
from flyte._internal.runtime.types_serde import transform_native_to_typed_interface
from flyte._logging import log, logger
from flyte._task import AsyncFunctionTaskTemplate, TaskTemplate
from flyte._utils.helpers import _selector_policy
from flyte.models import ActionID, NativeInterface
from flyte.remote._task import TaskDetails

R = TypeVar("R")


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

    @log
    async def submit(self, _task: TaskTemplate, *args, **kwargs) -> Any:
        """
        Main entrypoint for submitting a task to the local controller.
        """
        from flyte._debug import local_ui_db

        ctx = internal_ctx()
        tctx = ctx.data.task_context
        if not tctx:
            raise flyte.errors.RuntimeSystemError("BadContext", "Task context not initialized")

        run_id = local_ui_db.get_run_id_from_context()
        task_inputs = local_ui_db.coerce_inputs(getattr(_task, "func", None), args, kwargs)
        if run_id:
            local_ui_db.ensure_run(
                run_id,
                task_inputs,
                workflow_module=getattr(getattr(_task, "func", None), "__module__", None),
                workflow_name=getattr(getattr(_task, "func", None), "__name__", None),
                raw_args=task_inputs,
            )

        start_time = local_ui_db._utc_now_iso() if run_id else ""
        timer = local_ui_db.Timer() if run_id else None
        status = "completed"
        output_value = None

        inputs = await convert.convert_from_native_to_inputs(_task.native_interface, *args, **kwargs)
        inputs_hash = convert.generate_inputs_hash_from_proto(inputs.proto_inputs)
        task_interface = transform_native_to_typed_interface(_task.interface)

        sub_action_id, sub_action_output_path = convert.generate_sub_action_id_and_output_path(
            tctx, _task.name, inputs_hash, 0
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
        captured_stdout = ""
        captured_stderr = ""
        task_row_id = -1
        if run_id:
            task_name_for_db = getattr(_task, "name", getattr(_task, "short_name", "task")).split(".")[-1]
            input_value = local_ui_db.maybe_float(next(iter(task_inputs.values()), 0.0))
            task_row_id = local_ui_db.record_task_start(
                run_id=run_id,
                name=task_name_for_db,
                action_name=sub_action_id.name,
                input_value=input_value if input_value is not None else 0.0,
                start_time=start_time,
            )
        # We only get output from cache if the cache behavior is set to auto
        if task_cache.behavior == "auto":
            out = await LocalTaskCache.get(cache_key)
            if out is not None:
                logger.info(
                    f"Cache hit for task '{_task.name}' (version: {cache_version}), getting result from cache..."
                )

        if out is None:
            stdout_buffer: _io.StringIO | None = None
            stderr_buffer: _io.StringIO | None = None
            if run_id:
                stdout_buffer = _io.StringIO()
                stderr_buffer = _io.StringIO()
            if stdout_buffer and stderr_buffer:
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    out, err = await direct_dispatch(
                        _task,
                        controller=self,
                        action=sub_action_id,
                        raw_data_path=sub_action_raw_data_path,
                        inputs=inputs,
                        version=cache_version,
                        checkpoints=tctx.checkpoints,
                        code_bundle=tctx.code_bundle,
                        output_path=sub_action_output_path,
                        run_base_dir=tctx.run_base_dir,
                    )
            else:
                out, err = await direct_dispatch(
                    _task,
                    controller=self,
                    action=sub_action_id,
                    raw_data_path=sub_action_raw_data_path,
                    inputs=inputs,
                    version=cache_version,
                    checkpoints=tctx.checkpoints,
                    code_bundle=tctx.code_bundle,
                    output_path=sub_action_output_path,
                    run_base_dir=tctx.run_base_dir,
                )
            captured_stdout = stdout_buffer.getvalue() if stdout_buffer else ""
            captured_stderr = stderr_buffer.getvalue() if stderr_buffer else ""

            if err:
                exc = convert.convert_error_to_native(err)
                if exc:
                    raise exc
                else:
                    raise flyte.errors.RuntimeSystemError("BadError", "Unknown error")

            # store into cache
            if cache_enabled and out is not None:
                await LocalTaskCache.set(cache_key, out)

        error_message: str | None = None
        try:
            if _task.native_interface.outputs:
                if out is None:
                    raise flyte.errors.RuntimeSystemError("BadOutput", "Task output not captured.")
                result = await convert.convert_outputs_to_native(_task.native_interface, out)
                output_value = local_ui_db.maybe_float(result)
            else:
                result = None
            return result
        except Exception as exc:
            status = "failed"
            error_message = repr(exc)
            raise
        finally:
            if run_id:
                end_time = local_ui_db._utc_now_iso()
                duration_ms = timer.ms() if timer else 0.0
                task_name = getattr(_task, "name", getattr(_task, "short_name", "task"))
                input_display = next(iter(task_inputs.values()), None)
                input_value = local_ui_db.maybe_float(input_display)
                report_html = local_ui_db.read_report_html(sub_action_output_path)
                log_lines = [
                    f"[{start_time}] task {task_name} start input={input_display!r}",
                    f"[{end_time}] task {task_name} end output={output_value} status={status} duration_ms={duration_ms:.0f}",
                ]
                if error_message:
                    log_lines.append(f"ERROR: {error_message}")
                if captured_stdout:
                    log_lines.append("--- stdout ---")
                    log_lines.append(captured_stdout.rstrip())
                if captured_stderr:
                    log_lines.append("--- stderr ---")
                    log_lines.append(captured_stderr.rstrip())
                log_text = "\n".join(log_lines)
                local_ui_db.record_task_finish(
                    row_id=task_row_id,
                    output_value=output_value,
                    status=status,
                    end_time=end_time,
                    duration_ms=duration_ms,
                    log_text=log_text,
                    report_html=report_html,
                )

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
            converted_inputs = await convert.convert_from_native_to_inputs(_interface, *args, **kwargs)
            assert converted_inputs

        inputs_hash = convert.generate_inputs_hash_from_proto(converted_inputs.proto_inputs)
        action_id, action_output_path = convert.generate_sub_action_id_and_output_path(
            tctx,
            _func.__name__,
            inputs_hash,
            0,
        )
        assert action_output_path
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

        if info.interface.outputs and info.output:
            # If the result is not an AsyncGenerator, convert it directly
            converted_outputs = await convert.convert_from_native_to_outputs(info.output, info.interface, info.name)
            assert converted_outputs
        elif info.error:
            # If there is an error, convert it to a native error
            converted_error = convert.convert_from_native_to_error(info.error)
            assert converted_error
        assert info.action
        assert info.start_time
        assert info.end_time

    async def submit_task_ref(self, _task: TaskDetails, max_inline_io_bytes: int, *args, **kwargs) -> Any:
        raise flyte.errors.RemoteTaskUsageError(
            f"Remote tasks cannot be executed locally, only remotely. Found remote task {_task.name}"
        )
