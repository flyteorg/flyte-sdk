"""Tests for ConnectError handling across remote modules."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.app import app_definition_pb2
from flyteidl2.common import identifier_pb2
from flyteidl2.workflow import run_definition_pb2

import flyte.errors

# --- Action.abort (instance method, @syncify) ---


class TestActionAbortErrors:
    def _make_action(self):
        """Create an Action instance with a real pb2 containing a proper ActionIdentifier."""
        from flyte.remote._action import Action

        action_id = identifier_pb2.ActionIdentifier(
            run=identifier_pb2.RunIdentifier(
                org="test-org",
                project="p",
                domain="d",
                name="test-run",
            ),
            name="test-action",
        )
        pb2 = run_definition_pb2.Action(id=action_id)
        action = Action.__new__(Action)
        action.pb2 = pb2
        action._details = None
        return action

    @pytest.mark.asyncio
    async def test_not_found_returns_none(self):
        """Action.abort with NOT_FOUND is a no-op (action already gone)."""
        mock_client = MagicMock()
        mock_client.run_service.abort_action = AsyncMock(side_effect=ConnectError(Code.NOT_FOUND, "not found"))
        action = self._make_action()
        with patch("flyte.remote._action.get_client", return_value=mock_client):
            result = await action.abort.aio()
        assert result is None

    @pytest.mark.asyncio
    async def test_other_error_propagates(self):
        """Action.abort with non-NOT_FOUND code re-raises."""
        mock_client = MagicMock()
        mock_client.run_service.abort_action = AsyncMock(side_effect=ConnectError(Code.INTERNAL, "server error"))
        action = self._make_action()
        with patch("flyte.remote._action.get_client", return_value=mock_client):
            with pytest.raises(ConnectError):
                await action.abort.aio()


# --- Run.abort (instance method, @syncify) ---


class TestRunAbortErrors:
    @pytest.mark.asyncio
    async def test_not_found_returns_none(self):
        """Run.abort with NOT_FOUND is a no-op."""
        from flyte.remote._run import Run

        mock_client = MagicMock()
        mock_client.run_service.abort_run = AsyncMock(side_effect=ConnectError(Code.NOT_FOUND, "not found"))
        # Build a Run without triggering __post_init__ (which requires HasField("action"))
        run = Run.__new__(Run)
        run_id = identifier_pb2.RunIdentifier(
            org="test-org",
            project="p",
            domain="d",
            name="test-run",
        )
        action_id = identifier_pb2.ActionIdentifier(run=run_id, name="a0")
        action_pb2 = run_definition_pb2.Action(id=action_id)
        run.pb2 = run_definition_pb2.Run(action=action_pb2)
        run.action = MagicMock()
        run._details = None
        run._preserve_original_types = False
        run._debug_url = None
        with patch("flyte.remote._run.get_client", return_value=mock_client):
            result = await run.abort.aio()
        assert result is None


# --- App.delete ---


class TestAppDeleteErrors:
    @pytest.mark.asyncio
    async def test_not_found_returns_none(self):
        """App.delete with NOT_FOUND is idempotent."""
        from flyte.remote._app import App

        mock_client = MagicMock()
        mock_client.app_service.delete = AsyncMock(side_effect=ConnectError(Code.NOT_FOUND, "not found"))
        mock_cfg = MagicMock()
        mock_cfg.org = "test-org"
        mock_cfg.project = "p"
        mock_cfg.domain = "d"
        with (
            patch("flyte.remote._app.ensure_client"),
            patch("flyte.remote._app.get_client", return_value=mock_client),
            patch("flyte.remote._app.get_init_config", return_value=mock_cfg),
        ):
            result = await App.delete.aio(name="test-app", project="p", domain="d")
        assert result is None


# --- App.create upsert (ALREADY_EXISTS + ABORTED) ---


class TestAppCreateErrors:
    def _make_app_proto(self):
        app = app_definition_pb2.App()
        app.metadata.id.name = "test-app"
        app.metadata.id.project = "p"
        app.metadata.id.domain = "d"
        return app

    @pytest.mark.asyncio
    async def test_already_exists_falls_through_to_replace(self):
        """App.create with ALREADY_EXISTS calls App.replace."""
        from flyte.remote._app import App

        mock_client = MagicMock()
        mock_client.app_service.create = AsyncMock(side_effect=ConnectError(Code.ALREADY_EXISTS, "exists"))
        mock_replace_result = MagicMock(spec=App)
        with (
            patch("flyte.remote._app.ensure_client"),
            patch("flyte.remote._app.get_client", return_value=mock_client),
            patch(
                "flyte.remote._app.App.replace",
                new=MagicMock(aio=AsyncMock(return_value=mock_replace_result)),
            ) as mock_replace,
        ):
            await App.create.aio(app=self._make_app_proto())
            mock_replace.aio.assert_called_once()

    @pytest.mark.asyncio
    async def test_aborted_falls_through_to_replace(self):
        """App.create with ABORTED calls App.replace."""
        from flyte.remote._app import App

        mock_client = MagicMock()
        mock_client.app_service.create = AsyncMock(side_effect=ConnectError(Code.ABORTED, "aborted on server"))
        mock_replace_result = MagicMock(spec=App)
        with (
            patch("flyte.remote._app.ensure_client"),
            patch("flyte.remote._app.get_client", return_value=mock_client),
            patch(
                "flyte.remote._app.App.replace",
                new=MagicMock(aio=AsyncMock(return_value=mock_replace_result)),
            ) as mock_replace,
        ):
            await App.create.aio(app=self._make_app_proto())
            mock_replace.aio.assert_called_once()


# --- TaskDetails.get -> LazyEntity -> deferred_get ---


class TestTaskDetailsErrors:
    @pytest.mark.asyncio
    async def test_not_found_raises_remote_task_not_found(self):
        """TaskDetails.get with NOT_FOUND raises RemoteTaskNotFoundError."""
        from flyte.remote._task import TaskDetails

        mock_client = MagicMock()
        mock_client.task_service.get_task_details = AsyncMock(
            side_effect=ConnectError(Code.NOT_FOUND, "task not found")
        )
        mock_cfg = MagicMock()
        mock_cfg.org = "test-org"
        mock_cfg.project = "p"
        mock_cfg.domain = "d"
        with (
            patch("flyte.remote._task.ensure_client"),
            patch("flyte.remote._task.get_client", return_value=mock_client),
            patch("flyte.remote._task.get_init_config", return_value=mock_cfg),
        ):
            lazy = TaskDetails.get(name="my-task", project="p", domain="d", version="v1")
            with pytest.raises(flyte.errors.RemoteTaskNotFoundError):
                await lazy.fetch.aio()  # resolving the LazyEntity triggers deferred_get


# --- _deploy_task ALREADY_EXISTS ---


class TestDeployErrors:
    @pytest.mark.asyncio
    async def test_deploy_already_exists_returns_deployed_task(self):
        """When deploy_task raises ALREADY_EXISTS, deployment is idempotent."""
        from flyte._deploy import DeployedTask, _deploy_task

        mock_client = MagicMock()
        mock_client.task_service.deploy_task = AsyncMock(
            side_effect=ConnectError(Code.ALREADY_EXISTS, "already deployed")
        )
        mock_cfg = MagicMock()
        mock_cfg.org = "test-org"
        mock_cfg.project = "p"
        mock_cfg.domain = "d"

        # Create a mock task template
        mock_task = MagicMock()
        mock_task.name = "my-task"
        mock_task.source_file = "test.py"
        mock_task.parent_env_name = "default"
        mock_task.image = "test:latest"  # string, not Image instance
        mock_task.triggers = []
        mock_task.parent_env = MagicMock(return_value=MagicMock(description=None))
        mock_task.interface = MagicMock()
        mock_task.native_interface = MagicMock()
        mock_task.native_interface.docstring = None

        # Create a mock serialization context
        mock_sc = MagicMock()
        mock_sc.version = "v1"
        mock_sc.output_path = "/tmp"
        mock_sc.image_cache = MagicMock()

        # Mock the wire-translation to return a spec with expected attributes
        mock_spec = MagicMock()
        mock_spec.task_template.id.org = "test-org"
        mock_spec.task_template.id.project = "p"
        mock_spec.task_template.id.domain = "d"
        mock_spec.task_template.id.version = "v1"
        mock_spec.task_template.id.name = "my-task"
        mock_spec.task_template.HasField.return_value = True
        mock_spec.task_template.container.args = ["--module", "test", "--fn", "my_task"]
        mock_spec.default_inputs = []

        # Patch the protobuf request constructors that reject MagicMock inputs
        mock_deploy_request = MagicMock()
        with (
            patch("flyte._deploy.ensure_client"),
            patch("flyte._deploy.get_client", return_value=mock_client),
            patch("flyte._deploy.get_init_config", return_value=mock_cfg),
            patch("flyte._deploy.status") as mock_status,
            patch(
                "flyte._internal.runtime.convert.convert_upload_default_inputs",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "flyte._internal.runtime.task_serde.translate_task_to_wire",
                return_value=mock_spec,
            ),
            patch("flyte._deploy._get_documentation_entity", return_value=MagicMock()),
            patch(
                "flyte._deploy._update_interface_inputs_and_outputs_docstring",
                return_value=mock_spec.task_template.interface,
            ),
            patch(
                "flyteidl2.task.task_service_pb2.DeployTaskRequest",
                return_value=mock_deploy_request,
            ),
        ):
            result = await _deploy_task(
                task=mock_task,
                serialization_context=mock_sc,
            )
            assert isinstance(result, DeployedTask)
            mock_status.info.assert_called_once()
