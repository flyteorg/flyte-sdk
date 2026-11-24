from unittest.mock import AsyncMock

import pytest
from botocore.exceptions import ClientError

from flyteplugins.emr_serverless.boto_handler import EMRServerlessHandler


@pytest.fixture
def handler() -> EMRServerlessHandler:
    return EMRServerlessHandler("us-west-2")


@pytest.mark.asyncio
class TestApplications:
    async def test_get_and_find(self, handler: EMRServerlessHandler, monkeypatch: pytest.MonkeyPatch) -> None:
        call = AsyncMock(return_value={"application": {"applicationId": "app123", "state": "STARTED"}})
        paginate = AsyncMock(return_value=[{"id": "app123", "name": "analytics"}])
        monkeypatch.setattr(handler, "_call", call)
        monkeypatch.setattr(handler, "_paginate", paginate)

        assert (await handler.get_application("app123"))["state"] == "STARTED"
        assert await handler.find_application_by_name("analytics") == "app123"
        assert await handler.find_application_by_name("missing") is None
        assert set(paginate.await_args.kwargs["states"]) == {
            "CREATING",
            "CREATED",
            "STARTING",
            "STARTED",
            "STOPPING",
            "STOPPED",
        }

    async def test_create_forwards_every_field(
        self, handler: EMRServerlessHandler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call = AsyncMock(return_value={"applicationId": "newapp"})
        monkeypatch.setattr(handler, "_call", call)
        application_id = await handler.create_application(
            name="analytics",
            release_label="emr-7.5.0",
            application_type="SPARK",
            initial_capacity={"DRIVER": {"workerCount": 1}},
            maximum_capacity={"cpu": "100vCPU"},
            network_configuration={"subnetIds": ["subnet-example"]},
            image_configuration={"imageUri": "example.invalid/image:v1"},
            tags={"team": "data"},
            architecture="ARM64",
            runtime_configuration=[{"classification": "spark"}],
            scheduler_configuration={"maxConcurrentRuns": 5},
            auto_stop_config={"enabled": True},
        )
        assert application_id == "newapp"
        kwargs = call.await_args.kwargs
        assert kwargs["releaseLabel"] == "emr-7.5.0"
        assert kwargs["architecture"] == "ARM64"
        assert kwargs["autoStopConfiguration"] == {"enabled": True}

    async def test_update_noop_and_fields(self, handler: EMRServerlessHandler, monkeypatch: pytest.MonkeyPatch) -> None:
        call = AsyncMock(return_value={})
        monkeypatch.setattr(handler, "_call", call)
        await handler.update_application(application_id="app123")
        call.assert_not_awaited()
        await handler.update_application(
            application_id="app123",
            image_configuration={"imageUri": "example.invalid/image:v2"},
            maximum_capacity={"cpu": "200vCPU"},
        )
        assert call.await_args.kwargs["imageConfiguration"]["imageUri"].endswith(":v2")

    @pytest.mark.parametrize("state", ["CREATING", "STARTING", "STOPPING"])
    async def test_transitioning_is_deferred(
        self, state: str, handler: EMRServerlessHandler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(handler, "get_application", AsyncMock(return_value={"state": state}))
        assert not await handler.start_application_if_needed("app123")

    async def test_stopped_requests_start(self, handler: EMRServerlessHandler, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(handler, "get_application", AsyncMock(return_value={"state": "STOPPED"}))
        call = AsyncMock(return_value={})
        monkeypatch.setattr(handler, "_call", call)
        assert not await handler.start_application_if_needed("app123")
        call.assert_awaited_once_with("start_application", applicationId="app123")

    async def test_started_requests_stop_for_update(
        self,
        handler: EMRServerlessHandler,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            handler,
            "get_application",
            AsyncMock(return_value={"state": "STARTED"}),
        )
        call = AsyncMock(return_value={})
        monkeypatch.setattr(handler, "_call", call)
        assert not await handler.stop_application_if_needed("app123")
        call.assert_awaited_once_with("stop_application", applicationId="app123")

    @pytest.mark.parametrize("state", ["CREATED", "STOPPED"])
    async def test_stopped_application_is_ready_for_update(
        self,
        state: str,
        handler: EMRServerlessHandler,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            handler,
            "get_application",
            AsyncMock(return_value={"state": state}),
        )
        assert await handler.stop_application_if_needed("app123")

    async def test_terminal_application_fails(
        self, handler: EMRServerlessHandler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(handler, "get_application", AsyncMock(return_value={"state": "TERMINATED"}))
        with pytest.raises(RuntimeError, match="terminal"):
            await handler.start_application_if_needed("app123")

    async def test_ensure_application_started_polls(
        self, handler: EMRServerlessHandler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        start = AsyncMock(side_effect=[False, True])
        sleep = AsyncMock()
        monkeypatch.setattr(handler, "start_application_if_needed", start)
        monkeypatch.setattr("flyteplugins.emr_serverless.boto_handler.asyncio.sleep", sleep)
        await handler.ensure_application_started("app123", poll_interval_seconds=0)
        assert start.await_count == 2
        sleep.assert_awaited_once_with(0)


@pytest.mark.asyncio
class TestJobs:
    async def test_start_job_forwards_options(
        self, handler: EMRServerlessHandler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call = AsyncMock(return_value={"jobRunId": "job123"})
        monkeypatch.setattr(handler, "_call", call)
        result = await handler.start_job_run(
            application_id="app123",
            execution_role_arn="role",
            job_driver={"sparkSubmit": {"entryPoint": "s3://bucket/main.py"}},
            configuration_overrides={"monitoringConfiguration": {}},
            tags={"team": "data"},
            execution_timeout_minutes=120,
            name="task",
            retry_policy={"maxAttempts": 3},
            client_token="stable",
        )
        assert result == "job123"
        assert call.await_args.kwargs["retryPolicy"] == {"maxAttempts": 3}
        assert call.await_args.kwargs["clientToken"] == "stable"

    async def test_cancel_terminal_is_noop(
        self, handler: EMRServerlessHandler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(handler, "get_job_run", AsyncMock(return_value={"state": "SUCCESS"}))
        call = AsyncMock()
        monkeypatch.setattr(handler, "_call", call)
        await handler.cancel_job_run("app123", "job123")
        call.assert_not_awaited()

    async def test_cancel_running(self, handler: EMRServerlessHandler, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(handler, "get_job_run", AsyncMock(return_value={"state": "RUNNING"}))
        call = AsyncMock(return_value={})
        monkeypatch.setattr(handler, "_call", call)
        await handler.cancel_job_run("app123", "job123")
        call.assert_awaited_once_with("cancel_job_run", applicationId="app123", jobRunId="job123")

    async def test_cancel_only_swallows_idempotent_errors(
        self, handler: EMRServerlessHandler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        missing = ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "GetJobRun")
        monkeypatch.setattr(handler, "get_job_run", AsyncMock(side_effect=missing))
        await handler.cancel_job_run("app123", "job123")

        denied = ClientError({"Error": {"Code": "AccessDeniedException"}}, "GetJobRun")
        monkeypatch.setattr(handler, "get_job_run", AsyncMock(side_effect=denied))
        with pytest.raises(ClientError):
            await handler.cancel_job_run("app123", "job123")
