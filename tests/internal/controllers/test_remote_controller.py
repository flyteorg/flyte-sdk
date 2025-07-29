import pathlib

import pytest
from mock.mock import AsyncMock, patch

import flyte
import flyte.report
from flyte._context import internal_ctx
from flyte._internal.controllers.remote._action import Action
from flyte._internal.controllers.remote._controller import RemoteController
from flyte._internal.controllers.remote._service_protocol import ClientSet
from flyte._internal.runtime.convert import Outputs
from flyte._protos.common import identifier_pb2
from flyte._protos.workflow import run_definition_pb2
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
        phase=run_definition_pb2.Phase.PHASE_SUCCEEDED,
    )

    with (
        patch(
            "flyte._internal.controllers.remote._controller.upload_inputs_with_retry",
            new_callable=AsyncMock,
        ) as mock_upload_inputs,
        patch("flyte._internal.runtime.io.load_outputs", new_callable=AsyncMock) as mock_load_outputs,
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_action",
            new_callable=AsyncMock,
        ) as mock_submit_action,
        patch("flyte._initialize.get_common_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent
        # Ensure the mock returns a valid value
        mock_submit_action.return_value = action

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
        mock_submit_action.assert_called_once()
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
        phase=run_definition_pb2.Phase.PHASE_SUCCEEDED,
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
            "flyte._internal.controllers.remote._controller.RemoteController.submit_action",
            new_callable=AsyncMock,
        ) as mock_submit_action,
        patch("flyte._initialize.get_common_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent

        # Ensure the mock returns a valid value
        mock_submit_action.return_value = action
        mock_load_outputs.return_value = Outputs(
            proto_outputs=run_definition_pb2.Outputs(
                literals=[
                    run_definition_pb2.NamedLiteral(
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
        mock_submit_action.assert_called_once()
        mock_load_outputs.assert_called_with("/tmp/outputs/realized/outputs.pb")
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
        phase=run_definition_pb2.Phase.PHASE_FAILED,
        client_err=Exception("Task failed"),
        run_output_base="/tmp/outputs/base",
    )

    with (
        patch(
            "flyte._internal.controllers.remote._controller.upload_inputs_with_retry",
            new_callable=AsyncMock,
        ) as mock_upload_inputs,
        patch(
            "flyte._internal.controllers.remote._controller.RemoteController.submit_action",
            new_callable=AsyncMock,
        ) as mock_submit_action,
        patch("flyte._initialize.get_common_config") as mock_get_common_config,
    ):
        mock_get_common_config.return_value.root_dir = pathlib.Path(__file__).parent

        # Ensure the mock returns a valid value
        mock_submit_action.return_value = action

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
            with pytest.raises(Exception, match="Error in task"):
                await controller.submit(t1)

        mock_upload_inputs.assert_called_once()
        mock_submit_action.assert_called_once()


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
