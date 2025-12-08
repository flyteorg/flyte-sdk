from typing import AsyncIterator, Protocol

from flyteidl.admin import project_attributes_pb2, project_pb2, version_pb2
from flyteidl.service import dataproxy_pb2, identity_pb2
from flyteidl2.app import app_payload_pb2
from flyteidl2.secret import payload_pb2
from flyteidl2.task import task_service_pb2
from flyteidl2.trigger import trigger_service_pb2
from flyteidl2.workflow import run_logs_service_pb2, run_service_pb2
from grpc.aio import UnaryStreamCall
from grpc.aio._typing import RequestType


class MetadataServiceProtocol(Protocol):
    async def GetVersion(self, request: version_pb2.GetVersionRequest) -> version_pb2.GetVersionResponse: ...


class ProjectDomainService(Protocol):
    async def RegisterProject(
        self, request: project_pb2.ProjectRegisterRequest
    ) -> project_pb2.ProjectRegisterResponse: ...

    async def UpdateProject(self, request: project_pb2.Project) -> project_pb2.ProjectUpdateResponse: ...

    async def GetProject(self, request: project_pb2.ProjectGetRequest) -> project_pb2.Project: ...

    async def ListProjects(self, request: project_pb2.ProjectListRequest) -> project_pb2.Projects: ...

    async def GetDomains(self, request: project_pb2.GetDomainRequest) -> project_pb2.GetDomainsResponse: ...

    async def UpdateProjectDomainAttributes(
        self, request: project_attributes_pb2.ProjectAttributesUpdateRequest
    ) -> project_pb2.ProjectUpdateResponse: ...

    async def GetProjectDomainAttributes(
        self, request: project_attributes_pb2.ProjectAttributesGetRequest
    ) -> project_attributes_pb2.ProjectAttributes: ...

    async def DeleteProjectDomainAttributes(
        self, request: project_attributes_pb2.ProjectAttributesDeleteRequest
    ) -> project_attributes_pb2.ProjectAttributesDeleteResponse: ...

    async def UpdateProjectAttributes(
        self, request: project_attributes_pb2.ProjectAttributesUpdateRequest
    ) -> project_attributes_pb2.ProjectAttributesUpdateResponse: ...

    async def GetProjectAttributes(
        self, request: project_attributes_pb2.ProjectAttributesGetRequest
    ) -> project_attributes_pb2.ProjectAttributes: ...

    async def DeleteProjectAttributes(
        self, request: project_attributes_pb2.ProjectAttributesDeleteRequest
    ) -> project_attributes_pb2.ProjectAttributesDeleteResponse: ...


class TaskService(Protocol):
    async def DeployTask(self, request: task_service_pb2.DeployTaskRequest) -> task_service_pb2.DeployTaskResponse: ...

    async def GetTaskDetails(
        self, request: task_service_pb2.GetTaskDetailsRequest
    ) -> task_service_pb2.GetTaskDetailsResponse: ...

    async def ListTasks(self, request: task_service_pb2.ListTasksRequest) -> task_service_pb2.ListTasksResponse: ...


class AppService(Protocol):
    async def Create(self, request: app_payload_pb2.CreateRequest) -> app_payload_pb2.CreateResponse: ...

    async def Get(self, request: app_payload_pb2.GetRequest) -> app_payload_pb2.GetResponse: ...

    async def Update(self, request: app_payload_pb2.UpdateRequest) -> app_payload_pb2.UpdateResponse: ...

    async def UpdateStatus(
        self, request: app_payload_pb2.UpdateStatusRequest
    ) -> app_payload_pb2.UpdateStatusResponse: ...

    async def Delete(self, request: app_payload_pb2.DeleteRequest) -> app_payload_pb2.DeleteResponse: ...

    async def List(self, request: app_payload_pb2.ListRequest) -> app_payload_pb2.ListResponse: ...

    async def Watch(self, request: app_payload_pb2.WatchRequest) -> app_payload_pb2.WatchResponse: ...

    async def Lease(self, request: app_payload_pb2.LeaseRequest) -> app_payload_pb2.LeaseResponse: ...


class RunService(Protocol):
    async def CreateRun(self, request: run_service_pb2.CreateRunRequest) -> run_service_pb2.CreateRunResponse: ...

    async def AbortRun(self, request: run_service_pb2.AbortRunRequest) -> run_service_pb2.AbortRunResponse: ...

    async def GetRunDetails(
        self, request: run_service_pb2.GetRunDetailsRequest
    ) -> run_service_pb2.GetRunDetailsResponse: ...

    async def WatchRunDetails(
        self, request: run_service_pb2.WatchRunDetailsRequest
    ) -> AsyncIterator[run_service_pb2.WatchRunDetailsResponse]: ...

    async def GetActionDetails(
        self, request: run_service_pb2.GetActionDetailsRequest
    ) -> run_service_pb2.GetActionDetailsResponse: ...

    async def WatchActionDetails(
        self, request: run_service_pb2.WatchActionDetailsRequest
    ) -> AsyncIterator[run_service_pb2.WatchActionDetailsResponse]: ...

    async def GetActionData(
        self, request: run_service_pb2.GetActionDataRequest
    ) -> run_service_pb2.GetActionDataResponse: ...

    async def ListRuns(self, request: run_service_pb2.ListRunsRequest) -> run_service_pb2.ListRunsResponse: ...

    async def WatchRuns(
        self, request: run_service_pb2.WatchRunsRequest
    ) -> AsyncIterator[run_service_pb2.WatchRunsResponse]: ...

    async def ListActions(self, request: run_service_pb2.ListActionsRequest) -> run_service_pb2.ListActionsResponse: ...

    async def WatchActions(
        self, request: run_service_pb2.WatchActionsRequest
    ) -> AsyncIterator[run_service_pb2.WatchActionsResponse]: ...


class DataProxyService(Protocol):
    async def CreateUploadLocation(
        self, request: dataproxy_pb2.CreateUploadLocationRequest
    ) -> dataproxy_pb2.CreateUploadLocationResponse: ...

    async def CreateDownloadLocation(
        self, request: dataproxy_pb2.CreateDownloadLocationRequest
    ) -> dataproxy_pb2.CreateDownloadLocationResponse: ...

    async def CreateDownloadLink(
        self, request: dataproxy_pb2.CreateDownloadLinkRequest
    ) -> dataproxy_pb2.CreateDownloadLinkResponse: ...

    async def GetData(self, request: dataproxy_pb2.GetDataRequest) -> dataproxy_pb2.GetDataResponse: ...


class RunLogsService(Protocol):
    def TailLogs(
        self, request: run_logs_service_pb2.TailLogsRequest
    ) -> UnaryStreamCall[RequestType, run_logs_service_pb2.TailLogsResponse]: ...


class SecretService(Protocol):
    async def CreateSecret(self, request: payload_pb2.CreateSecretRequest) -> payload_pb2.CreateSecretResponse: ...

    async def UpdateSecret(self, request: payload_pb2.UpdateSecretRequest) -> payload_pb2.UpdateSecretResponse: ...

    async def GetSecret(self, request: payload_pb2.GetSecretRequest) -> payload_pb2.GetSecretResponse: ...

    async def ListSecrets(self, request: payload_pb2.ListSecretsRequest) -> payload_pb2.ListSecretsResponse: ...

    async def DeleteSecret(self, request: payload_pb2.DeleteSecretRequest) -> payload_pb2.DeleteSecretResponse: ...


class IdentityService(Protocol):
    async def UserInfo(self, request: identity_pb2.UserInfoRequest) -> identity_pb2.UserInfoResponse: ...


class TriggerService(Protocol):
    async def DeployTrigger(
        self, request: trigger_service_pb2.DeployTriggerRequest
    ) -> trigger_service_pb2.DeployTriggerResponse: ...

    async def GetTriggerDetails(
        self, request: trigger_service_pb2.GetTriggerDetailsRequest
    ) -> trigger_service_pb2.GetTriggerDetailsResponse: ...

    async def GetTriggerRevisionDetails(
        self, request: trigger_service_pb2.GetTriggerRevisionDetailsRequest
    ) -> trigger_service_pb2.GetTriggerRevisionDetailsResponse: ...

    async def ListTriggers(
        self, request: trigger_service_pb2.ListTriggersRequest
    ) -> trigger_service_pb2.ListTriggersResponse: ...

    async def GetTriggerRevisionHistory(
        self, request: trigger_service_pb2.GetTriggerRevisionHistoryRequest
    ) -> trigger_service_pb2.GetTriggerRevisionHistoryResponse: ...

    async def UpdateTriggers(
        self, request: trigger_service_pb2.UpdateTriggersRequest
    ) -> trigger_service_pb2.UpdateTriggersResponse: ...

    async def DeleteTriggers(
        self, request: trigger_service_pb2.DeleteTriggersRequest
    ) -> trigger_service_pb2.DeleteTriggersResponse: ...
