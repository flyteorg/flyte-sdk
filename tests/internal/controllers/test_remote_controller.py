import asyncio
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.core import execution_pb2
from flyteidl2.task import common_pb2

import flyte
import flyte.errors
import flyte.report
from flyte._context import internal_ctx
from flyte._internal.controllers.remote._action import Action
from flyte._internal.controllers.remote._controller import RemoteController
from flyte._internal.controllers.remote._core import Controller
from flyte._internal.controllers.remote._service_protocol import ClientSet
from flyte._internal.runtime.convert import Outputs
from flyte.models import ActionID, CodeBundle, RawDataPath, TaskContext
from flyte.types import TypeEngine

env = flyte.TaskEnvironment("test")


@env.task
async def t1():
    pass


@env.task
async def t2() -> str:
    return "test"


@pytest.mark.asyncio
async def test_submit_task():
    await flyte.init.aio()

    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    action = Action(
        parent_action_name="test_parent_action",
        action_id=identifier_pb2.ActionIdentifier(
            name="test_action",
        ),
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
    )

    with (
        patch(
            "flyte._internal.controllers.remote._controller.upload_inputs_with_retry",
            new_callable=AsyncMock,
        ) as mock_upload_inputs,
        patch("flyte._internal.runtime.io.load_outputs", new_callable=AsyncMock) as mock_load_outputs,
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ) as mock_submit_and_wait,
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent
        # Ensure the mock returns a valid value
        mock_submit_and_wait.return_value = action

        ctx = internal_ctx()
        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        tctx = TaskContext(
            action=ActionID(name="test"),
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/run_base",
            report=flyte.report.Report(name="test_report"),
            # set code bundle to have a dummy root_dir to circumvent pytest setting the wrong cwd
            code_bundle=CodeBundle(
                computed_version="vcode-bundle",
                destination=this_dir_str,
                tgz="dummy.tgz",
            ),
        )
        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            result = await controller.submit(t1)

        mock_upload_inputs.assert_called_once()
        mock_submit_and_wait.assert_called_once()
        mock_load_outputs.assert_not_called()
        assert result is None


@pytest.mark.asyncio
async def test_submit_with_outputs():
    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    action = Action(
        parent_action_name="test_parent_action",
        action_id=identifier_pb2.ActionIdentifier(
            name="test_action",
        ),
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
        run_output_base="/tmp/outputs/base",
        realized_outputs_uri="/tmp/outputs/realized",
    )

    with (
        patch(
            "flyte._internal.controllers.remote._controller.upload_inputs_with_retry",
            new_callable=AsyncMock,
        ) as mock_upload_inputs,
        patch("flyte._internal.runtime.io.load_outputs", new_callable=AsyncMock) as mock_load_outputs,
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ) as mock_submit_and_wait,
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent

        # Ensure the mock returns a valid value
        mock_submit_and_wait.return_value = action
        mock_load_outputs.return_value = Outputs(
            proto_outputs=common_pb2.Outputs(
                literals=[
                    common_pb2.NamedLiteral(
                        name="o0",
                        value=await TypeEngine.to_literal("test", str, TypeEngine.to_literal_type(str)),
                    )
                ]
            )
        )

        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        ctx = internal_ctx()
        tctx = TaskContext(
            action=ActionID(name="test"),
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/tmp/outputs/base",
            report=flyte.report.Report(name="test_report"),
            code_bundle=CodeBundle(
                computed_version="vcode-bundle",
                destination=this_dir_str,
                tgz="dummy.tgz",
            ),
        )
        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            result = await controller.submit(t2)

        mock_upload_inputs.assert_called_once()
        mock_submit_and_wait.assert_called_once()
        mock_load_outputs.assert_called_with("/tmp/outputs/realized/outputs.pb", max_bytes=10485760)
        assert result == "test"


@pytest.mark.asyncio
async def test_submit_task_with_error():
    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    action = Action(
        parent_action_name="test_parent_action",
        action_id=identifier_pb2.ActionIdentifier(
            name="test_action",
        ),
        phase=phase_pb2.ACTION_PHASE_FAILED,
        client_err=Exception("Task failed"),
        run_output_base="/tmp/outputs/base",
    )

    with (
        patch(
            "flyte._internal.controllers.remote._controller.upload_inputs_with_retry",
            new_callable=AsyncMock,
        ) as mock_upload_inputs,
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ) as mock_submit_and_wait,
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent

        # Ensure the mock returns a valid value
        mock_submit_and_wait.return_value = action

        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        ctx = internal_ctx()
        tctx = TaskContext(
            action=ActionID(name="test"),
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/tmp/outputs/base",
            report=flyte.report.Report(name="test_report"),
            code_bundle=CodeBundle(
                computed_version="vcode-bundle",
                destination=this_dir_str,
                tgz="dummy.tgz",
            ),
        )
        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            # The returned action carries client_err — that real error must propagate, not be
            # swallowed into a generic "Error in task ...: None" fallback.
            with pytest.raises(Exception, match="Task failed"):
                await controller.submit(t1)

        mock_upload_inputs.assert_called_once()
        mock_submit_and_wait.assert_called_once()


@pytest.mark.asyncio
async def test_submit_task_error_from_observed_action_on_recovery():
    """Regression: on recovery, the failed sub-action's error is observed on the action object
    that submit_and_wait_for_action *returns* (the cache/`from_state` object), NOT on the local
    stub that ``_submit`` builds via ``Action.from_task``.

    The first run happens to work because the stub is also the cached object (submit-before-watch),
    so ``merge_state`` lands the error on it. On a retry the watch observes the pre-existing FAILED
    action first, so the returned ``n`` carries the error while the local stub stays empty. The
    propagated exception must reflect the *real* user error, not a generic
    ``UnableToConvertError: Error in task ...: None``.
    """

    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    # `n` — the observed/cached action returned by submit_and_wait_for_action. Distinct object
    # from the stub that _submit builds internally; it is the only one carrying the failure.
    observed = Action(
        parent_action_name="test_parent_action",
        action_id=identifier_pb2.ActionIdentifier(name="test_action"),
        phase=phase_pb2.ACTION_PHASE_FAILED,
        err=execution_pb2.ExecutionError(
            code="BadRequest",
            message="400 Resources exceeded during query execution: Memory exceeded while reading Parquet",
            kind=execution_pb2.ExecutionError.USER,
        ),
        run_output_base="/tmp/outputs/base",
    )

    with (
        patch(
            "flyte._internal.controllers.remote._controller.upload_inputs_with_retry",
            new_callable=AsyncMock,
        ),
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ) as mock_submit_and_wait,
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent
        mock_submit_and_wait.return_value = observed

        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        ctx = internal_ctx()
        tctx = TaskContext(
            action=ActionID(name="test"),
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/tmp/outputs/base",
            report=flyte.report.Report(name="test_report"),
            code_bundle=CodeBundle(
                computed_version="vcode-bundle",
                destination=this_dir_str,
                tgz="dummy.tgz",
            ),
        )
        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            with pytest.raises(flyte.errors.RuntimeUserError, match="Resources exceeded") as exc_info:
                await controller.submit(t1)

        # The propagated error must be the real USER error, not the generic dropped-error fallback.
        assert "UnableToConvertError" not in str(exc_info.value)
        assert "None" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_finalize_parent_action():
    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    mock_action_id = ActionID(
        name="parent_action",
        run_name="root_run",
        project="project",
        domain="domain",
        org="org",
    )

    with patch(
        "flyte._internal.controllers.remote._core.Controller._finalize_parent_action",
        new_callable=AsyncMock,
    ) as mock_finalize_action:
        controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
        await controller.finalize_parent_action(mock_action_id)

        mock_finalize_action.assert_called_once()


@pytest.mark.asyncio
async def test_defaultdict():
    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)

    assert controller._parent_action_semaphore["test_key"] is not None


@pytest.mark.asyncio
async def test_record_trace_with_int_zero_output():
    """Test record_trace when trace info has output as int = 0"""
    await flyte.init.aio()

    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    with (
        patch("flyte._internal.runtime.io.upload_outputs", new_callable=AsyncMock) as mock_upload_outputs,
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ) as mock_submit_and_wait,
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent

        # Create a simple interface with int output
        from flyte.models import NativeInterface

        interface = NativeInterface(inputs={}, outputs={"result": int})

        # Create trace info with int output = 0
        from flyte._internal.controllers import TraceInfo
        from flyte.models import ActionID

        trace_info = TraceInfo(
            name="test_function",
            action=ActionID(name="test_action"),
            interface=interface,
            inputs_path="/tmp/inputs",
            output=0,  # Test with int = 0
        )

        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        ctx = internal_ctx()
        tctx = TaskContext(
            action=ActionID(name="parent_action"),
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/tmp/outputs/base",
            report=flyte.report.Report(name="test_report"),
            code_bundle=CodeBundle(
                computed_version="vcode-bundle",
                destination=this_dir_str,
                tgz="dummy.tgz",
            ),
        )

        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            await controller.record_trace(trace_info)

        # Verify that convert_from_native_to_outputs was called with 0
        mock_upload_outputs.assert_called_once()
        mock_submit_and_wait.assert_called_once()


@pytest.mark.asyncio
async def test_record_trace_with_optional_none_output():
    """Test record_trace when trace info has optional return value as None"""
    await flyte.init.aio()

    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    with (
        patch("flyte._internal.runtime.io.upload_outputs", new_callable=AsyncMock) as mock_upload_outputs,
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ) as mock_submit_and_wait,
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent

        # Create interface with optional output
        from typing import Optional

        from flyte.models import NativeInterface

        interface = NativeInterface(inputs={}, outputs={"result": Optional[str]})

        # Create trace info with None output
        from flyte._internal.controllers import TraceInfo
        from flyte.models import ActionID

        trace_info = TraceInfo(
            name="test_function_optional",
            action=ActionID(name="test_action_optional"),
            interface=interface,
            inputs_path="/tmp/inputs",
            output=None,  # Test with None output
        )

        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        ctx = internal_ctx()
        tctx = TaskContext(
            action=ActionID(name="parent_action"),
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/tmp/outputs/base",
            report=flyte.report.Report(name="test_report"),
            code_bundle=CodeBundle(
                computed_version="vcode-bundle",
                destination=this_dir_str,
                tgz="dummy.tgz",
            ),
        )

        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            await controller.record_trace(trace_info)

        # Verify that convert_from_native_to_outputs was called with None
        mock_upload_outputs.assert_called_once()
        mock_submit_and_wait.assert_called_once()


@pytest.mark.asyncio
async def test_record_trace_with_error():
    """Test record_trace when trace info contains an error"""
    await flyte.init.aio()

    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    with (
        patch("flyte._internal.runtime.convert.convert_from_native_to_error") as mock_convert_error,
        patch("flyte._internal.runtime.io.upload_error", new_callable=AsyncMock) as mock_upload_error,
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ) as mock_submit_and_wait,
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent
        mock_convert_error.return_value = AsyncMock(err=AsyncMock())

        # Create interface with output
        from flyte.models import NativeInterface

        interface = NativeInterface(inputs={}, outputs={"result": str})

        # Create trace info with error
        from flyte._internal.controllers import TraceInfo
        from flyte.models import ActionID

        test_error = ValueError("Test error")
        trace_info = TraceInfo(
            name="test_function_with_error",
            action=ActionID(name="test_action_with_error"),
            interface=interface,
            inputs_path="/tmp/inputs",
            error=test_error,  # Test with error instead of output
        )

        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        ctx = internal_ctx()
        tctx = TaskContext(
            action=ActionID(name="parent_action"),
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/tmp/outputs/base",
            report=flyte.report.Report(name="test_report"),
            code_bundle=CodeBundle(
                computed_version="vcode-bundle",
                destination=this_dir_str,
                tgz="dummy.tgz",
            ),
        )

        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            await controller.record_trace(trace_info)

        # Verify that convert_from_native_to_error was called with the error
        mock_convert_error.assert_called_once_with(test_error)
        mock_upload_error.assert_called_once()
        mock_submit_and_wait.assert_called_once()


@pytest.mark.asyncio
async def test_bg_run_slowdown_error_translated_clean():
    # Create instance without running __init__
    controller = object.__new__(Controller)

    # Minimal attributes needed by _bg_run
    controller._running = True  # start the controller in running
    controller._shared_queue = asyncio.Queue()
    controller._max_retries = 1
    controller._min_backoff_on_err = 0.1
    controller._max_backoff_on_err = 0.1

    informer = AsyncMock()
    controller._informers = MagicMock()
    controller._informers.get = AsyncMock(return_value=informer)

    class FakeAction:
        def __init__(self):
            self.name = "A"
            self.run_name = "run"
            self.parent_action_name = "parent"
            self.retries = 1  # exceeds limit → triggers outer handler
            self.set_client_error = MagicMock()

    action = FakeAction()
    await controller._shared_queue.put(action)

    async def fake_bg_process(action):
        controller._running = False  # Run once only, switch it off here.
        raise flyte.errors.SlowDownError("boom")

    controller._bg_process = fake_bg_process

    await controller._bg_run(worker_id="w1")

    # Assertions
    action.set_client_error.assert_called_once()
    err = action.set_client_error.call_args[0][0]
    assert action.retries == 2

    assert isinstance(err, flyte.errors.RuntimeSystemError)
    assert isinstance(err.__cause__, flyte.errors.SlowDownError)

    informer.fire_completion_event.assert_awaited_once_with("A")


@pytest.mark.asyncio
async def test_record_trace_uses_task_action_when_in_trace_scope():
    """When tctx.action has been swapped by @trace, record_trace must submit with
    parent_action_name = tctx.task_action.name (the real running task's action),
    not tctx.action.name (the outer trace's pseudo-action)."""
    await flyte.init.aio()

    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    with (
        patch("flyte._internal.runtime.io.upload_outputs", new_callable=AsyncMock),
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ) as mock_submit_action,
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent

        from flyte._internal.controllers import TraceInfo
        from flyte.models import NativeInterface

        interface = NativeInterface(inputs={}, outputs={"result": int})
        trace_info = TraceInfo(
            name="inner_trace_fn",
            action=ActionID(name="inner_trace_action"),
            interface=interface,
            inputs_path="/tmp/inputs",
            output=1,
        )

        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        ctx = internal_ctx()
        # Simulate state inside a nested trace: `action` is the outer trace's pseudo-action,
        # `task_action` still points at the real running task's action.
        tctx = TaskContext(
            action=ActionID(name="outer_trace_action"),
            task_action=ActionID(name="real_container_action"),
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/tmp/outputs/base",
            report=flyte.report.Report(name="test_report"),
            code_bundle=CodeBundle(
                computed_version="vcode-bundle",
                destination=this_dir_str,
                tgz="dummy.tgz",
            ),
        )

        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            await controller.record_trace(trace_info)

        mock_submit_action.assert_called_once()
        submitted_action = mock_submit_action.call_args[0][0]
        assert submitted_action.parent_action_name == "real_container_action"


@pytest.mark.asyncio
async def test_record_trace_falls_back_to_action_when_task_action_unset():
    """TaskContexts that don't set task_action must still work: __post_init__ pins
    task_action to tctx.action, so record_trace records under tctx.action.name."""
    await flyte.init.aio()

    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    with (
        patch("flyte._internal.runtime.io.upload_outputs", new_callable=AsyncMock),
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ) as mock_submit_action,
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent

        from flyte._internal.controllers import TraceInfo
        from flyte.models import NativeInterface

        interface = NativeInterface(inputs={}, outputs={"result": int})
        trace_info = TraceInfo(
            name="fn",
            action=ActionID(name="trace_action"),
            interface=interface,
            inputs_path="/tmp/inputs",
            output=1,
        )

        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        ctx = internal_ctx()
        tctx = TaskContext(
            action=ActionID(name="legacy_action"),
            # task_action omitted → defaults to action via __post_init__
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/tmp/outputs/base",
            report=flyte.report.Report(name="test_report"),
            code_bundle=CodeBundle(
                computed_version="vcode-bundle",
                destination=this_dir_str,
                tgz="dummy.tgz",
            ),
        )

        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            await controller.record_trace(trace_info)

        mock_submit_action.assert_called_once()
        submitted_action = mock_submit_action.call_args[0][0]
        assert submitted_action.parent_action_name == "legacy_action"


@pytest.mark.asyncio
async def test_get_action_outputs_uses_task_action_when_in_trace_scope():
    """When tctx.action has been swapped by @trace, get_action_outputs must look up
    the existing trace record under parent = tctx.task_action.name, not tctx.action.name."""
    await flyte.init.aio()

    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    with (
        patch(
            "flyte._internal.controllers.remote._controller.upload_inputs_with_retry",
            new_callable=AsyncMock,
        ),
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.get_action",
            new_callable=AsyncMock,
        ) as mock_get_action,
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent
        mock_get_action.return_value = None  # no prior record → return TraceInfo, False

        from flyte.models import NativeInterface

        interface = NativeInterface(inputs={}, outputs={"result": int})

        async def fn() -> int:
            return 1

        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        ctx = internal_ctx()
        tctx = TaskContext(
            action=ActionID(name="outer_trace_action", run_name="r", project="p", domain="d", org="o"),
            task_action=ActionID(name="real_container_action", run_name="r", project="p", domain="d", org="o"),
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/tmp/outputs/base",
            report=flyte.report.Report(name="test_report"),
            code_bundle=CodeBundle(
                computed_version="vcode-bundle",
                destination=this_dir_str,
                tgz="dummy.tgz",
            ),
        )

        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            _info, ok = await controller.get_action_outputs(interface, fn)

        assert ok is False
        mock_get_action.assert_awaited_once()
        # parent_action_name is the second positional arg to get_action
        passed_parent = mock_get_action.call_args[0][1]
        assert passed_parent == "real_container_action"


@pytest.mark.asyncio
async def test_record_trace_semaphore_keyed_on_task_action():
    """The submit semaphore for trace records must be keyed by the running task's
    action so concurrent traces inside one container serialize under the right informer,
    not under each (different) trace pseudo-action."""
    await flyte.init.aio()

    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    with (
        patch("flyte._internal.runtime.io.upload_outputs", new_callable=AsyncMock),
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
        ),
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent

        from flyte._internal.controllers import TraceInfo
        from flyte._internal.controllers.remote._controller import unique_action_name
        from flyte.models import NativeInterface

        interface = NativeInterface(inputs={}, outputs={"result": int})
        trace_info = TraceInfo(
            name="fn",
            action=ActionID(name="trace_action"),
            interface=interface,
            inputs_path="/tmp/inputs",
            output=1,
        )

        task_action = ActionID(name="real_container_action", run_name="r", project="p", domain="d", org="o")
        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        ctx = internal_ctx()
        tctx = TaskContext(
            action=ActionID(name="outer_trace_action", run_name="r", project="p", domain="d", org="o"),
            task_action=task_action,
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/tmp/outputs/base",
            report=flyte.report.Report(name="test_report"),
            code_bundle=CodeBundle(
                computed_version="vcode-bundle",
                destination=this_dir_str,
                tgz="dummy.tgz",
            ),
        )

        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            assert unique_action_name(task_action) not in controller._parent_action_semaphore
            await controller.record_trace(trace_info)
            # The semaphore should have been created under the task_action key, not the trace's action key.
            assert unique_action_name(task_action) in controller._parent_action_semaphore
            assert unique_action_name(tctx.action) not in controller._parent_action_semaphore


@pytest.mark.asyncio
async def test_submit_task_parent_is_action_for_regular_task():
    """For a regular task (no @trace), task_action defaults to action, so a submitted sub-task
    parents under tctx.action.name — i.e. the change is a no-op outside trace scope."""
    await flyte.init.aio()

    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    action = Action(
        parent_action_name="ignored",
        action_id=identifier_pb2.ActionIdentifier(name="test_action"),
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
    )

    with (
        patch("flyte._internal.controllers.remote._controller.upload_inputs_with_retry", new_callable=AsyncMock),
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
            return_value=action,
        ) as mock_submit_and_wait,
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent
        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        ctx = internal_ctx()
        tctx = TaskContext(
            action=ActionID(name="real_task"),
            # task_action omitted → defaults to action via __post_init__
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/run_base",
            report=flyte.report.Report(name="test_report"),
            code_bundle=CodeBundle(computed_version="vcode-bundle", destination=this_dir_str, tgz="dummy.tgz"),
        )
        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            await controller.submit(t1)

        mock_submit_and_wait.assert_called_once()
        submitted = mock_submit_and_wait.call_args[0][0]
        assert submitted.parent_action_name == "real_task"


@pytest.mark.asyncio
async def test_submit_task_reparents_to_task_action_inside_trace():
    """When tctx.action has been swapped by @trace, a submitted sub-task must parent under
    tctx.task_action.name (the real running task), and the submit semaphore must be keyed by it."""
    await flyte.init.aio()
    from flyte._internal.controllers.remote._controller import unique_action_name

    async def make_client() -> ClientSet:
        return AsyncMock()  # type: ignore

    action = Action(
        parent_action_name="ignored",
        action_id=identifier_pb2.ActionIdentifier(name="test_action"),
        phase=phase_pb2.ACTION_PHASE_SUCCEEDED,
    )

    with (
        patch("flyte._internal.controllers.remote._controller.upload_inputs_with_retry", new_callable=AsyncMock),
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_and_wait_for_action",
            new_callable=AsyncMock,
            return_value=action,
        ) as mock_submit_and_wait,
        patch("flyte._initialize.get_init_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent
        this_dir_str = str(pathlib.Path(__file__).parent.absolute())
        ctx = internal_ctx()
        tctx = TaskContext(
            action=ActionID(name="outer_trace_action"),
            task_action=ActionID(name="real_container_action"),
            raw_data_path=RawDataPath(path="test"),
            output_path="/tmp",
            version="v1",
            run_base_dir="/run_base",
            report=flyte.report.Report(name="test_report"),
            code_bundle=CodeBundle(computed_version="vcode-bundle", destination=this_dir_str, tgz="dummy.tgz"),
        )
        with ctx.replace_task_context(tctx):
            controller = RemoteController(client_coro=make_client(), workers=2, max_system_retries=2)
            await controller.submit(t1)

        mock_submit_and_wait.assert_called_once()
        submitted = mock_submit_and_wait.call_args[0][0]
        assert submitted.parent_action_name == "real_container_action"
        # The submit semaphore is keyed by the real task action, not the trace pseudo-action.
        assert unique_action_name(tctx.task_action) in controller._parent_action_semaphore
        assert unique_action_name(tctx.action) not in controller._parent_action_semaphore
