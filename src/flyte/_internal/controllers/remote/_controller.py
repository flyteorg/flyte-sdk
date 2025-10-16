from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, Awaitable, DefaultDict, Tuple, TypeVar

from flyteidl2.common import identifier_pb2
from flyteidl2.workflow import run_definition_pb2

import flyte
import flyte.errors
import flyte.storage as storage
from flyte._code_bundle import build_pkl_bundle
from flyte._context import internal_ctx
from flyte._internal.controllers import TraceInfo
from flyte._internal.controllers.remote._action import Action
from flyte._internal.controllers.remote._core import Controller
from flyte._internal.controllers.remote._service_protocol import ClientSet
from flyte._internal.runtime import convert, io
from flyte._internal.runtime.task_serde import translate_task_to_wire
from flyte._internal.runtime.types_serde import transform_native_to_typed_interface
from flyte._logging import logger
from flyte._task import TaskTemplate
from flyte._utils.helpers import _selector_policy
from flyte.models import MAX_INLINE_IO_BYTES, ActionID, NativeInterface, SerializationContext
from flyte.remote._task import TaskDetails

R = TypeVar("R")

MAX_TRACE_BYTES = MAX_INLINE_IO_BYTES


async def upload_inputs_with_retry(serialized_inputs: bytes, inputs_uri: str, max_bytes: int) -> None:
    """
    Upload inputs to the specified URI with error handling.

    Args:
        serialized_inputs: The serialized inputs to upload
        inputs_uri: The destination URI
        max_bytes: Maximum number of bytes to read from the input stream

    Raises:
        RuntimeSystemError: If the upload fails
    """
    if len(serialized_inputs) > max_bytes:
        raise flyte.errors.InlineIOMaxBytesBreached(
            f"Inputs exceed max_bytes limit of {max_bytes / 1024 / 1024} MB,"
            f" actual size: {len(serialized_inputs) / 1024 / 1024} MB"
        )
    try:
        # TODO Add retry decorator to this
        await storage.put_stream(serialized_inputs, to_path=inputs_uri)
    except Exception as e:
        logger.exception("Failed to upload inputs")
        raise flyte.errors.RuntimeSystemError(type(e).__name__, str(e)) from e


async def handle_action_failure(action: Action, task_name: str) -> Exception:
    """
    Handle action failure by loading error details or raising a RuntimeSystemError.

    Args:
        action: The updated action
        task_name: The name of the task

    Raises:
        Exception: The converted native exception or RuntimeSystemError
    """
    err = action.err or action.client_err
    if not err and action.phase == run_definition_pb2.PHASE_FAILED:
        logger.error(f"Server reported failure for action {action.name}, checking error file.")
        try:
            error_path = io.error_path(f"{action.run_output_base}/{action.action_id.name}/1")
            err = await io.load_error(error_path)
        except Exception as e:
            logger.exception("Failed to load error file", e)
            err = flyte.errors.RuntimeSystemError(type(e).__name__, f"Failed to load error file: {e}")
    else:
        logger.error(f"Server reported failure for action {action.action_id.name}, error: {err}")

    exc = convert.convert_error_to_native(err)
    if not exc:
        return flyte.errors.RuntimeSystemError("UnableToConvertError", f"Error in task {task_name}: {err}")
    return exc


async def load_and_convert_outputs(iface: NativeInterface, realized_outputs_uri: str, max_bytes: int) -> Any:
    """
    Load outputs from the given URI and convert them to native format.

    Args:
        iface: The Native interface
        realized_outputs_uri: The URI where outputs are stored
        max_bytes: Maximum number of bytes to read from the output file

    Returns:
        The converted native outputs
    """
    outputs_file_path = io.outputs_path(realized_outputs_uri)
    outputs = await io.load_outputs(outputs_file_path, max_bytes=max_bytes)
    return await convert.convert_outputs_to_native(iface, outputs)


def unique_action_name(action_id: ActionID) -> str:
    return f"{action_id.name}_{action_id.run_name}"


class RemoteController(Controller):
    """
    This a specialized controller that wraps the core controller and performs IO, serialization and deserialization
    """

    def __init__(
        self,
        client_coro: Awaitable[ClientSet],
        workers: int = 20,
        max_system_retries: int = 10,
    ):
        """ """
        super().__init__(
            client_coro=client_coro,
            workers=workers,
            max_system_retries=max_system_retries,
        )
        default_parent_concurrency = int(os.getenv("_F_P_CNC", "1000"))
        self._default_parent_concurrency = default_parent_concurrency
        self._parent_action_semaphore: DefaultDict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(default_parent_concurrency)
        )
        self._parent_action_task_call_sequence: DefaultDict[str, DefaultDict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._submit_loop: asyncio.AbstractEventLoop | None = None
        self._submit_thread: threading.Thread | None = None

    def generate_task_call_sequence(self, task_obj: object, action_id: ActionID) -> int:
        """
        Generate a task call sequence for the given task object and action ID.
        This is used to track the number of times a task is called within an action.
        """
        uniq = unique_action_name(action_id)
        current_action_sequencer = self._parent_action_task_call_sequence[uniq]
        current_task_id = id(task_obj)
        v = current_action_sequencer[current_task_id]
        new_seq = v + 1
        current_action_sequencer[current_task_id] = new_seq
        name = ""
        if hasattr(task_obj, "__name__"):
            name = task_obj.__name__
        elif hasattr(task_obj, "name"):
            name = task_obj.name
        logger.info(f"For action {uniq}, task {name} call sequence is {new_seq}")
        return new_seq

    async def _submit(self, _task_call_seq: int, _task: TaskTemplate, *args, **kwargs) -> Any:
        ctx = internal_ctx()
        tctx = ctx.data.task_context
        if tctx is None:
            raise flyte.errors.RuntimeSystemError("BadContext", "Task context not initialized")
        current_action_id = tctx.action

        # In the case of a regular code bundle, we will just pass it down as it is to the downstream tasks
        # It is not allowed to change the code bundle (for regular code bundles) in the middle of a run.
        code_bundle = tctx.code_bundle

        if tctx.interactive_mode or (code_bundle and code_bundle.pkl):
            logger.debug(f"Building new pkl bundle for task {_task.name}")
            code_bundle = await build_pkl_bundle(
                _task,
                upload_to_controlplane=False,
                upload_from_dataplane_base_path=tctx.run_base_dir,
            )

        inputs = await convert.convert_from_native_to_inputs(_task.native_interface, *args, **kwargs)

        root_dir = Path(code_bundle.destination).absolute() if code_bundle else Path.cwd()
        # Don't set output path in sec context because node executor will set it
        new_serialization_context = SerializationContext(
            project=current_action_id.project,
            domain=current_action_id.domain,
            org=current_action_id.org,
            code_bundle=code_bundle,
            version=tctx.version,
            # supplied version.
            # input_path=inputs_uri,
            image_cache=tctx.compiled_image_cache,
            root_dir=root_dir,
        )

        task_spec = translate_task_to_wire(_task, new_serialization_context)
        inputs_hash = convert.generate_inputs_hash_from_proto(inputs.proto_inputs)
        sub_action_id, sub_action_output_path = convert.generate_sub_action_id_and_output_path(
            tctx, task_spec, inputs_hash, _task_call_seq
        )
        logger.info(f"Sub action {sub_action_id} output path {sub_action_output_path}")

        serialized_inputs = inputs.proto_inputs.SerializeToString(deterministic=True)
        inputs_uri = io.inputs_path(sub_action_output_path)
        await upload_inputs_with_retry(serialized_inputs, inputs_uri, max_bytes=_task.max_inline_io_bytes)

        md = task_spec.task_template.metadata
        ignored_input_vars = []
        if len(md.cache_ignore_input_vars) > 0:
            ignored_input_vars = list(md.cache_ignore_input_vars)
        cache_key = None
        if task_spec.task_template.metadata and task_spec.task_template.metadata.discoverable:
            discovery_version = task_spec.task_template.metadata.discovery_version
            cache_key = convert.generate_cache_key_hash(
                _task.name,
                inputs_hash,
                task_spec.task_template.interface,
                discovery_version,
                ignored_input_vars,
                inputs.proto_inputs,
            )

        # Clear to free memory
        serialized_inputs = None  # type: ignore
        inputs_hash = None  # type: ignore

        action = Action.from_task(
            sub_action_id=identifier_pb2.ActionIdentifier(
                name=sub_action_id.name,
                run=identifier_pb2.RunIdentifier(
                    name=current_action_id.run_name,
                    project=current_action_id.project,
                    domain=current_action_id.domain,
                    org=current_action_id.org,
                ),
            ),
            parent_action_name=current_action_id.name,
            group_data=tctx.group_data,
            task_spec=task_spec,
            inputs_uri=inputs_uri,
            run_output_base=tctx.run_base_dir,
            cache_key=cache_key,
            queue=_task.queue,
        )

        try:
            logger.info(
                f"Submitting action Run:[{action.run_name}, Parent:[{action.parent_action_name}], "
                f"task:[{_task.name}], action:[{action.name}]"
            )
            n = await self.submit_action(action)
            logger.info(f"Action for task [{_task.name}] action id: {action.name}, completed!")
        except asyncio.CancelledError:
            # If the action is cancelled, we need to cancel the action on the server as well
            logger.info(f"Action {action.action_id.name} cancelled, cancelling on server")
            await self.cancel_action(action)
            raise

        # If the action is aborted, we should abort the controller as well
        if n.phase == run_definition_pb2.PHASE_ABORTED:
            logger.warning(f"Action {n.action_id.name} was aborted, aborting current Action{current_action_id.name}")
            raise flyte.errors.RunAbortedError(
                f"Action {n.action_id.name} was aborted, aborting current Action {current_action_id.name}"
            )

        if n.phase == run_definition_pb2.PHASE_TIMED_OUT:
            logger.warning(
                f"Action {n.action_id.name} timed out, raising timeout exception Action {current_action_id.name}"
            )
            raise flyte.errors.TaskTimeoutError(
                f"Action {n.action_id.name} timed out, raising exception in current Action {current_action_id.name}"
            )

        if n.has_error() or n.phase == run_definition_pb2.PHASE_FAILED:
            exc = await handle_action_failure(action, _task.name)
            raise exc

        if _task.native_interface.outputs:
            if not n.realized_outputs_uri:
                raise flyte.errors.RuntimeSystemError(
                    "RuntimeError",
                    f"Task {n.action_id.name} did not return an output path, but the task has outputs defined.",
                )
            return await load_and_convert_outputs(
                _task.native_interface, n.realized_outputs_uri, max_bytes=_task.max_inline_io_bytes
            )
        return None

    async def submit(self, _task: TaskTemplate, *args, **kwargs) -> Any:
        """
        Submit a task to the remote controller.This creates a new action on the queue service.
        """
        ctx = internal_ctx()
        tctx = ctx.data.task_context
        if tctx is None:
            raise flyte.errors.RuntimeSystemError("BadContext", "Task context not initialized")
        current_action_id = tctx.action
        task_call_seq = self.generate_task_call_sequence(_task, current_action_id)
        async with self._parent_action_semaphore[unique_action_name(current_action_id)]:
            return await self._submit(task_call_seq, _task, *args, **kwargs)

    def _sync_thread_loop_runner(self) -> None:
        """This method runs the event loop and should be invoked in a separate thread."""

        loop = self._submit_loop
        assert loop is not None
        try:
            loop.run_forever()
        finally:
            loop.close()

    def submit_sync(self, _task: TaskTemplate, *args, **kwargs) -> concurrent.futures.Future:
        """
        This function creates a cached thread and loop for the purpose of calling the submit method synchronously,
        returning a concurrent Future that can be awaited. There's no need for a lock because this function itself is
        single threaded and non-async. This pattern here is basically the trivial/degenerate case of the thread pool
        in the LocalController.
        Please see additional comments in protocol.

        :param _task:
        :param args:
        :param kwargs:
        :return:
        """
        if self._submit_thread is None:
            # Please see LocalController for the general implementation of this pattern.
            def exc_handler(loop, context):
                logger.error(f"Remote controller submit sync loop caught exception in {loop}: {context}")

            with _selector_policy():
                self._submit_loop = asyncio.new_event_loop()
                self._submit_loop.set_exception_handler(exc_handler)

            self._submit_thread = threading.Thread(
                name=f"remote-controller-{os.getpid()}-submitter",
                daemon=True,
                target=self._sync_thread_loop_runner,
            )
            self._submit_thread.start()

        coro = self.submit(_task, *args, **kwargs)
        assert self._submit_loop is not None, "Submit loop should always have been initialized by now"
        fut = asyncio.run_coroutine_threadsafe(coro, self._submit_loop)
        return fut

    async def finalize_parent_action(self, action_id: ActionID):
        """
        This method is invoked when the parent action is finished. It will finalize the run and upload the outputs
        to the control plane.
        """
        run_id = identifier_pb2.RunIdentifier(
            name=action_id.run_name,
            project=action_id.project,
            domain=action_id.domain,
            org=action_id.org,
        )
        await super()._finalize_parent_action(run_id=run_id, parent_action_name=action_id.name)
        self._parent_action_semaphore.pop(unique_action_name(action_id), None)
        self._parent_action_task_call_sequence.pop(unique_action_name(action_id), None)

    async def get_action_outputs(
        self, _interface: NativeInterface, _func: Callable, *args, **kwargs
    ) -> Tuple[TraceInfo, bool]:
        """
        This method returns the outputs of the action, if it is available.
        If not available it raises a NotFoundError.
        :param _interface: NativeInterface
        :param _func: Function name
        :param args: Arguments
        :param kwargs: Keyword arguments
        :return:
        """
        ctx = internal_ctx()
        tctx = ctx.data.task_context
        if tctx is None:
            raise flyte.errors.RuntimeSystemError("BadContext", "Task context not initialized")
        current_action_id = tctx.action

        func_name = _func.__name__
        invoke_seq_num = self.generate_task_call_sequence(_func, current_action_id)

        inputs = await convert.convert_from_native_to_inputs(_interface, *args, **kwargs)
        serialized_inputs = inputs.proto_inputs.SerializeToString(deterministic=True)
        inputs_hash = convert.generate_inputs_hash_from_proto(inputs.proto_inputs)

        sub_action_id, sub_action_output_path = convert.generate_sub_action_id_and_output_path(
            tctx, func_name, inputs_hash, invoke_seq_num
        )

        inputs_uri = io.inputs_path(sub_action_output_path)
        await upload_inputs_with_retry(serialized_inputs, inputs_uri, max_bytes=MAX_TRACE_BYTES)
        # Clear to free memory
        serialized_inputs = None  # type: ignore

        prev_action = await self.get_action(
            identifier_pb2.ActionIdentifier(
                name=sub_action_id.name,
                run=identifier_pb2.RunIdentifier(
                    name=current_action_id.run_name,
                    project=current_action_id.project,
                    domain=current_action_id.domain,
                    org=current_action_id.org,
                ),
            ),
            current_action_id.name,
        )

        if prev_action is None:
            return TraceInfo(func_name, sub_action_id, _interface, inputs_uri), False

        if prev_action.phase == run_definition_pb2.PHASE_FAILED:
            if prev_action.has_error():
                exc = convert.convert_error_to_native(prev_action.err)
                return (
                    TraceInfo(func_name, sub_action_id, _interface, inputs_uri, error=exc),
                    True,
                )
            else:
                logger.warning(f"Action {prev_action.action_id.name} failed, but no error was found, re-running trace!")
        elif prev_action.realized_outputs_uri is not None:
            o = await io.load_outputs(prev_action.realized_outputs_uri, max_bytes=MAX_TRACE_BYTES)
            outputs = await convert.convert_outputs_to_native(_interface, o)
            return (
                TraceInfo(func_name, sub_action_id, _interface, inputs_uri, output=outputs),
                True,
            )

        return TraceInfo(func_name, sub_action_id, _interface, inputs_uri), False

    async def record_trace(self, info: TraceInfo):
        """
        Record a trace action. This is used to record the trace of the action and should be called when the action
        :param info:
        :return:
        """
        ctx = internal_ctx()
        tctx = ctx.data.task_context
        if tctx is None:
            raise flyte.errors.RuntimeSystemError("BadContext", "Task context not initialized")

        current_action_id = tctx.action
        sub_run_output_path = storage.join(tctx.run_base_dir, info.action.name)
        outputs_file_path: str = ""

        if info.interface.has_outputs():
            if info.error:
                err = convert.convert_from_native_to_error(info.error)
                await io.upload_error(err.err, sub_run_output_path)
            else:
                outputs = await convert.convert_from_native_to_outputs(info.output, info.interface)
                outputs_file_path = io.outputs_path(sub_run_output_path)
                await io.upload_outputs(outputs, sub_run_output_path, max_bytes=MAX_TRACE_BYTES)

        typed_interface = transform_native_to_typed_interface(info.interface)

        trace_action = Action.from_trace(
            parent_action_name=current_action_id.name,
            action_id=identifier_pb2.ActionIdentifier(
                name=info.action.name,
                run=identifier_pb2.RunIdentifier(
                    name=current_action_id.run_name,
                    project=current_action_id.project,
                    domain=current_action_id.domain,
                    org=current_action_id.org,
                ),
            ),
            inputs_uri=info.inputs_path,
            outputs_uri=outputs_file_path,
            friendly_name=info.name,
            group_data=tctx.group_data,
            run_output_base=tctx.run_base_dir,
            start_time=info.start_time,
            end_time=info.end_time,
            typed_interface=typed_interface if typed_interface else None,
        )

        async with self._parent_action_semaphore[unique_action_name(current_action_id)]:
            try:
                logger.info(
                    f"Submitting Trace action Run:[{trace_action.run_name},"
                    f" Parent:[{trace_action.parent_action_name}],"
                    f" Trace fn:[{info.name}], action:[{info.action.name}]"
                )
                await self.submit_action(trace_action)
                logger.info(f"Trace Action for [{info.name}] action id: {info.action.name}, completed!")
            except asyncio.CancelledError:
                # If the action is cancelled, we need to cancel the action on the server as well
                raise

    async def _submit_task_ref(self, invoke_seq_num: int, _task: TaskDetails, *args, **kwargs) -> Any:
        ctx = internal_ctx()
        tctx = ctx.data.task_context
        if tctx is None:
            raise flyte.errors.RuntimeSystemError("BadContext", "Task context not initialized")
        current_action_id = tctx.action
        task_name = _task.name

        native_interface = _task.interface
        pb_interface = _task.pb2.spec.task_template.interface

        inputs = await convert.convert_from_native_to_inputs(native_interface, *args, **kwargs)
        inputs_hash = convert.generate_inputs_hash_from_proto(inputs.proto_inputs)
        sub_action_id, sub_action_output_path = convert.generate_sub_action_id_and_output_path(
            tctx, task_name, inputs_hash, invoke_seq_num
        )

        serialized_inputs = inputs.proto_inputs.SerializeToString(deterministic=True)
        inputs_uri = io.inputs_path(sub_action_output_path)
        await upload_inputs_with_retry(serialized_inputs, inputs_uri, _task.max_inline_io_bytes)
        # cache key - task name, task signature, inputs, cache version
        cache_key = None
        md = _task.pb2.spec.task_template.metadata
        ignored_input_vars = []
        if len(md.cache_ignore_input_vars) > 0:
            ignored_input_vars = list(md.cache_ignore_input_vars)
        if md and md.discoverable:
            discovery_version = md.discovery_version
            cache_key = convert.generate_cache_key_hash(
                task_name,
                inputs_hash,
                pb_interface,
                discovery_version,
                ignored_input_vars,
                inputs.proto_inputs,
            )

        # Clear to free memory
        serialized_inputs = None  # type: ignore
        inputs_hash = None  # type: ignore

        action = Action.from_task(
            sub_action_id=identifier_pb2.ActionIdentifier(
                name=sub_action_id.name,
                run=identifier_pb2.RunIdentifier(
                    name=current_action_id.run_name,
                    project=current_action_id.project,
                    domain=current_action_id.domain,
                    org=current_action_id.org,
                ),
            ),
            parent_action_name=current_action_id.name,
            group_data=tctx.group_data,
            task_spec=_task.pb2.spec,
            inputs_uri=inputs_uri,
            run_output_base=tctx.run_base_dir,
            cache_key=cache_key,
            queue=None,
        )

        try:
            logger.info(
                f"Submitting action Run:[{action.run_name}, Parent:[{action.parent_action_name}], "
                f"task:[{task_name}], action:[{action.name}]"
            )
            n = await self.submit_action(action)
            logger.info(f"Action for task [{task_name}] action id: {action.name}, completed!")
        except asyncio.CancelledError:
            # If the action is cancelled, we need to cancel the action on the server as well
            logger.info(f"Action {action.action_id.name} cancelled, cancelling on server")
            await self.cancel_action(action)
            raise

        if n.has_error() or n.phase == run_definition_pb2.PHASE_FAILED:
            exc = await handle_action_failure(action, task_name)
            raise exc

        if native_interface.outputs:
            if not n.realized_outputs_uri:
                raise flyte.errors.RuntimeSystemError(
                    "RuntimeError",
                    f"Task {n.action_id.name} did not return an output path, but the task has outputs defined.",
                )
            return await load_and_convert_outputs(native_interface, n.realized_outputs_uri, _task.max_inline_io_bytes)
        return None

    async def submit_task_ref(self, _task: TaskDetails, *args, **kwargs) -> Any:
        ctx = internal_ctx()
        tctx = ctx.data.task_context
        if tctx is None:
            raise flyte.errors.RuntimeSystemError("BadContext", "Task context not initialized")
        current_action_id = tctx.action
        task_call_seq = self.generate_task_call_sequence(_task, current_action_id)
        async with self._parent_action_semaphore[unique_action_name(current_action_id)]:
            return await self._submit_task_ref(task_call_seq, _task, *args, **kwargs)
