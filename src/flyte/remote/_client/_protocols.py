from typing import AsyncIterator, Protocol

from flyteidl2.app import app_payload_pb2
from flyteidl2.auth import identity_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.project import project_service_pb2
from flyteidl2.secret import payload_pb2
from flyteidl2.task import task_service_pb2
from flyteidl2.trigger import trigger_service_pb2
from flyteidl2.workflow import run_logs_service_pb2, run_service_pb2


class ProjectDomainService(Protocol):
    async def create_project(
        self, request: project_service_pb2.CreateProjectRequest
    ) -> project_service_pb2.CreateProjectResponse: ...

    async def update_project(
        self, request: project_service_pb2.UpdateProjectRequest
    ) -> project_service_pb2.UpdateProjectResponse: ...

    async def get_project(
        self, request: project_service_pb2.GetProjectRequest
    ) -> project_service_pb2.GetProjectResponse: ...

    async def list_projects(
        self, request: project_service_pb2.ListProjectsRequest
    ) -> project_service_pb2.ListProjectsResponse: ...


class TaskService(Protocol):
    async def deploy_task(self, request: task_service_pb2.DeployTaskRequest) -> task_service_pb2.DeployTaskResponse: ...

    async def get_task_details(
        self, request: task_service_pb2.GetTaskDetailsRequest
    ) -> task_service_pb2.GetTaskDetailsResponse: ...

    async def list_tasks(self, request: task_service_pb2.ListTasksRequest) -> task_service_pb2.ListTasksResponse: ...


class AppService(Protocol):
    async def create(self, request: app_payload_pb2.CreateRequest) -> app_payload_pb2.CreateResponse: ...

    async def get(self, request: app_payload_pb2.GetRequest) -> app_payload_pb2.GetResponse: ...

    async def update(self, request: app_payload_pb2.UpdateRequest) -> app_payload_pb2.UpdateResponse: ...

    async def update_status(
        self, request: app_payload_pb2.UpdateStatusRequest
    ) -> app_payload_pb2.UpdateStatusResponse: ...

    async def delete(self, request: app_payload_pb2.DeleteRequest) -> app_payload_pb2.DeleteResponse: ...

    async def list(self, request: app_payload_pb2.ListRequest) -> app_payload_pb2.ListResponse: ...

    async def watch(self, request: app_payload_pb2.WatchRequest) -> app_payload_pb2.WatchResponse: ...

    async def lease(self, request: app_payload_pb2.LeaseRequest) -> app_payload_pb2.LeaseResponse: ...


class RunService(Protocol):
    async def create_run(self, request: run_service_pb2.CreateRunRequest) -> run_service_pb2.CreateRunResponse: ...

    async def abort_run(self, request: run_service_pb2.AbortRunRequest) -> run_service_pb2.AbortRunResponse: ...

    async def abort_action(
        self, request: run_service_pb2.AbortActionRequest
    ) -> run_service_pb2.AbortActionResponse: ...

    async def get_run_details(
        self, request: run_service_pb2.GetRunDetailsRequest
    ) -> run_service_pb2.GetRunDetailsResponse: ...

    async def watch_run_details(
        self, request: run_service_pb2.WatchRunDetailsRequest
    ) -> AsyncIterator[run_service_pb2.WatchRunDetailsResponse]: ...

    async def get_action_details(
        self, request: run_service_pb2.GetActionDetailsRequest
    ) -> run_service_pb2.GetActionDetailsResponse: ...

    async def watch_action_details(
        self, request: run_service_pb2.WatchActionDetailsRequest
    ) -> AsyncIterator[run_service_pb2.WatchActionDetailsResponse]: ...

    async def get_action_data(
        self, request: run_service_pb2.GetActionDataRequest
    ) -> run_service_pb2.GetActionDataResponse: ...

    async def list_runs(self, request: run_service_pb2.ListRunsRequest) -> run_service_pb2.ListRunsResponse: ...

    async def watch_runs(
        self, request: run_service_pb2.WatchRunsRequest
    ) -> AsyncIterator[run_service_pb2.WatchRunsResponse]: ...

    async def list_actions(
        self, request: run_service_pb2.ListActionsRequest
    ) -> run_service_pb2.ListActionsResponse: ...

    async def watch_actions(
        self, request: run_service_pb2.WatchActionsRequest
    ) -> AsyncIterator[run_service_pb2.WatchActionsResponse]: ...


class DataProxyService(Protocol):
    async def create_upload_location(
        self, request: dataproxy_service_pb2.CreateUploadLocationRequest
    ) -> dataproxy_service_pb2.CreateUploadLocationResponse: ...

    async def upload_inputs(
        self, request: dataproxy_service_pb2.UploadInputsRequest
    ) -> dataproxy_service_pb2.UploadInputsResponse: ...


class RunLogsService(Protocol):
    def tail_logs(
        self, request: run_logs_service_pb2.TailLogsRequest
    ) -> AsyncIterator[run_logs_service_pb2.TailLogsResponse]: ...


class SecretService(Protocol):
    async def create_secret(self, request: payload_pb2.CreateSecretRequest) -> payload_pb2.CreateSecretResponse: ...

    async def update_secret(self, request: payload_pb2.UpdateSecretRequest) -> payload_pb2.UpdateSecretResponse: ...

    async def get_secret(self, request: payload_pb2.GetSecretRequest) -> payload_pb2.GetSecretResponse: ...

    async def list_secrets(self, request: payload_pb2.ListSecretsRequest) -> payload_pb2.ListSecretsResponse: ...

    async def delete_secret(self, request: payload_pb2.DeleteSecretRequest) -> payload_pb2.DeleteSecretResponse: ...


class IdentityService(Protocol):
    async def user_info(self, request: identity_pb2.UserInfoRequest) -> identity_pb2.UserInfoResponse: ...


class TriggerService(Protocol):
    async def deploy_trigger(
        self, request: trigger_service_pb2.DeployTriggerRequest
    ) -> trigger_service_pb2.DeployTriggerResponse: ...

    async def get_trigger_details(
        self, request: trigger_service_pb2.GetTriggerDetailsRequest
    ) -> trigger_service_pb2.GetTriggerDetailsResponse: ...

    async def get_trigger_revision_details(
        self, request: trigger_service_pb2.GetTriggerRevisionDetailsRequest
    ) -> trigger_service_pb2.GetTriggerRevisionDetailsResponse: ...

    async def list_triggers(
        self, request: trigger_service_pb2.ListTriggersRequest
    ) -> trigger_service_pb2.ListTriggersResponse: ...

    async def get_trigger_revision_history(
        self, request: trigger_service_pb2.GetTriggerRevisionHistoryRequest
    ) -> trigger_service_pb2.GetTriggerRevisionHistoryResponse: ...

    async def update_triggers(
        self, request: trigger_service_pb2.UpdateTriggersRequest
    ) -> trigger_service_pb2.UpdateTriggersResponse: ...

    async def delete_triggers(
        self, request: trigger_service_pb2.DeleteTriggersRequest
    ) -> trigger_service_pb2.DeleteTriggersResponse: ...
