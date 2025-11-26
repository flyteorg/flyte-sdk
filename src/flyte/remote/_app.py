from __future__ import annotations

from typing import AsyncIterator, Literal, Tuple, cast

import rich.repr
from flyteidl2.app import app_definition_pb2, app_payload_pb2
from flyteidl2.common import identifier_pb2, list_pb2

from flyte._initialize import ensure_client, get_client, get_init_config
from flyte.syncify import syncify

from ._common import ToJSONMixin, filtering, sorting

WaitFor = Literal["started", "stopped"]


def _is_started(state: app_definition_pb2.Status.DeploymentStatus) -> bool:
    return state == [
        app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_ACTIVE,
        app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_STARTED,
    ]


def _is_stopped(state: app_definition_pb2.Status.DeploymentStatus) -> bool:
    return state == [
        app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_UNASSIGNED,
        app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_STOPPED,
    ]


class App(ToJSONMixin):
    pb2: app_definition_pb2.App

    def __init__(self, pb2: app_definition_pb2.App):
        self.pb2 = pb2

    @property
    def name(self) -> str:
        return self.pb2.metadata.id.name

    @property
    def revision(self) -> int:
        return self.pb2.metadata.revision

    @property
    def endpoint(self) -> str:
        return self.pb2.status.ingress.public_url

    @property
    def deployment_status(self) -> app_definition_pb2.Status.DeploymentStatus:
        """
        Get the deployment status of the app
        Returns:

        """
        if len(self.pb2.status.conditions) > 0:
            return self.pb2.status.conditions[-1].deployment_status
        else:
            return app_definition_pb2.Status.DeploymentStatus.UNKNOWN

    @property
    def desired_state(self) -> app_definition_pb2.Spec.DesiredState:
        return self.pb2.spec.desired_state

    @syncify
    async def watch(self, wait_for: WaitFor = "started") -> App:
        """
        Watch for the app to reach started or stopped state.
        :param wait_for: ["started", "stopped"]

        Returns: The app in the desired state.
        Raises: RuntimeError if the app did not reach desired state and failed!
        """

        if wait_for == "started" and _is_started(self.deployment_status):
            return self
        elif wait_for == "stopped" and _is_stopped(self.deployment_status):
            return self

        call = cast(
            AsyncIterator[app_payload_pb2.WatchResponse],
            get_client().app_service.Watch(
                request=app_payload_pb2.WatchRequest(
                    app_id=self.pb2.metadata.id,
                )
            ),
        )
        async for resp in call:
            if resp.update_event:
                updated_app = resp.update_event.updated_app
                current_status = updated_app.status.conditions[-1].deployment_status
                if current_status == app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_FAILED:
                    raise RuntimeError(f"App deployment for app {self.name} has failed!")
                if wait_for == "started" and _is_started(current_status):
                    return App(updated_app)
                elif wait_for == "stopped" and _is_stopped(current_status):
                    return App(updated_app)
        raise RuntimeError(f"App deployment for app {self.name} stalled!")

    async def _update_status(self, desired_state: app_definition_pb2.Spec.DesiredState):
        ensure_client()
        self.pb2.spec.desired_state = desired_state
        resp = await get_client().app_service.UpdateStatus(
            request=app_payload_pb2.UpdateStatusRequest(
                app=self.pb2,
            )
        )
        self.pb2 = resp.app

    @syncify
    async def start(self, wait: bool = False):
        """
        Start the app
        :param wait: Wait for the app to reach started state

        """
        if _is_started(self.deployment_status):
            return
        await self._update_status(app_definition_pb2.Spec.DESIRED_STATE_STARTED)
        if wait:
            await self.watch.aio(wait_for="started")

    @syncify
    async def stop(self, wait: bool = False):
        """
        Stop the app
        :param wait: Wait for the app to reach the stopped state
        """
        if _is_stopped(self.deployment_status):
            return
        await self._update_status(app_definition_pb2.Spec.DESIRED_STATE_STOPPED)
        if wait:
            await self.watch.aio(wait_for="stopped")

    def __rich_repr__(self) -> rich.repr.Result:
        yield "name", self.name
        yield "revision", self.revision
        yield "endpoint", self.endpoint
        yield (
            "deployment_status",
            app_definition_pb2.Status.DeploymentStatus.Name(self.deployment_status)[len("DEPLOYMENT_STATUS_") :],
        )
        yield "desired_state", app_definition_pb2.Spec.DesiredState.Name(self.desired_state)[len("DESIRED_STATE_") :]

    @syncify
    @classmethod
    async def get(
        cls,
        name: str,
        project: str | None = None,
        domain: str | None = None,
    ) -> App:
        """
        Get an app by name.

        :param name: The name of the app.
        :param project: The project of the app.
        :param domain: The domain of the app.
        :return: The app remote object.
        """
        ensure_client()
        cfg = get_init_config()
        resp = await get_client().app_service.Get(
            request=app_payload_pb2.GetRequest(
                app_id=app_definition_pb2.Identifier(
                    org=cfg.org,
                    project=project or cfg.project,
                    domain=domain or cfg.domain,
                    name=name,
                ),
            )
        )
        return cls(pb2=resp.app)

    @syncify
    @classmethod
    async def listall(
        cls,
        created_by_subject: str | None = None,
        sort_by: Tuple[str, Literal["asc", "desc"]] | None = None,
        limit: int = 100,
    ) -> AsyncIterator[App]:
        ensure_client()
        cfg = get_init_config()
        i = 0
        token = None
        sort_pb2 = sorting(sort_by)
        filters = filtering(created_by_subject)
        project = None
        if cfg.project:
            project = identifier_pb2.ProjectIdentifier(
                organization=cfg.org,
                name=cfg.project,
                domain=cfg.domain,
            )
        while True:
            req = app_payload_pb2.ListRequest(
                request=list_pb2.ListRequest(
                    limit=min(100, limit),
                    token=token,
                    sort_by=sort_pb2,
                    filters=filters,
                ),
                org=cfg.org,
                project=project,
            )
            resp = await get_client().app_service.List(
                request=req,
            )
            token = resp.token
            for a in resp.apps:
                i += 1
                if i > limit:
                    return
                yield cls(a)
            if not token:
                break
