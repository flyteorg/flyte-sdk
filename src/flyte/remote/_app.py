from __future__ import annotations

from typing import AsyncIterator, Literal, Mapping, Tuple, cast

import grpc
import rich.repr
from flyteidl2.app import app_definition_pb2, app_payload_pb2
from flyteidl2.common import identifier_pb2, list_pb2

from flyte._initialize import ensure_client, get_client, get_init_config
from flyte._logging import logger
from flyte.syncify import syncify

from ._common import ToJSONMixin, filtering, sorting

WaitFor = Literal["activated", "deactivated"]


def _is_active(state: app_definition_pb2.Status.DeploymentStatus) -> bool:
    return state in [
        app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_ACTIVE,
        app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_STARTED,
    ]


def _is_deactivated(state: app_definition_pb2.Status.DeploymentStatus) -> bool:
    return state in [
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
            return app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_UNSPECIFIED

    @property
    def desired_state(self) -> app_definition_pb2.Spec.DesiredState:
        return self.pb2.spec.desired_state

    def is_active(self) -> bool:
        return _is_active(self.deployment_status)

    def is_deactivated(self) -> bool:
        return _is_deactivated(self.deployment_status)

    @property
    def url(self) -> str:
        client = get_client()
        return client.console.app_url(
            project=self.pb2.metadata.id.project,
            domain=self.pb2.metadata.id.domain,
            app_name=self.name,
        )

    @syncify
    async def watch(self, wait_for: WaitFor = "activated") -> App:
        """
        Watch for the app to reach activated or deactivated state.
        :param wait_for: ["activated", "deactivated"]

        Returns: The app in the desired state.
        Raises: RuntimeError if the app did not reach desired state and failed!
        """

        if wait_for == "activated" and self.is_active():
            return self
        elif wait_for == "deactivated" and self.is_deactivated():
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
                if wait_for == "activated":
                    if _is_active(current_status):
                        return App(updated_app)
                    elif _is_deactivated(current_status):
                        raise RuntimeError(f"App deployment for app {self.name} has failed!")
                elif wait_for == "deactivated":
                    if _is_deactivated(current_status):
                        return App(updated_app)
        raise RuntimeError(f"App deployment for app {self.name} stalled!")

    async def _update(
        self, desired_state: app_definition_pb2.Spec.DesiredState, reason: str, wait_for: WaitFor | None = None
    ) -> App:
        new_pb2 = app_definition_pb2.App()
        new_pb2.CopyFrom(self.pb2)
        new_pb2.spec.desired_state = desired_state
        updated_app = await App.update.aio(new_pb2, reason=reason)
        if wait_for:
            await updated_app.watch.aio(wait_for)
        return updated_app

    @syncify
    async def activate(self, wait: bool = False) -> App:
        """
        Start the app
        :param wait: Wait for the app to reach started state

        """
        if self.is_active():
            return self
        return await self._update(
            app_definition_pb2.Spec.DESIRED_STATE_STARTED,
            "User requested to activate app from flyte-sdk",
            "activated" if wait else None,
        )

    @syncify
    async def deactivate(self, wait: bool = False):
        """
        Stop the app
        :param wait: Wait for the app to reach the deactivated state
        """
        if self.is_deactivated():
            return
        return await self._update(
            app_definition_pb2.Spec.DESIRED_STATE_STOPPED,
            "User requested to deactivate app from flyte-sdk",
            "deactivated" if wait else None,
        )

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
    async def update(cls, updated_app_proto: app_definition_pb2.App, reason: str) -> App:
        ensure_client()
        resp = await get_client().app_service.Update(
            request=app_payload_pb2.UpdateRequest(
                app=updated_app_proto,
                reason=reason,
            )
        )
        return App(pb2=resp.app)

    @syncify
    @classmethod
    async def replace(
        cls,
        name: str,
        updated_app_spec: app_definition_pb2.Spec,
        reason: str,
        labels: Mapping[str, str] | None = None,
        project: str | None = None,
        domain: str | None = None,
    ) -> App:
        """
        Replace an existing app's that matches the given name, with a new spec and optionally labels.
        :param name: Name of the new app
        :param updated_app_spec: Updated app spec
        :param labels: Optional labels for the new app
        :param project: Optional project for the new app
        :param domain: Optional domain for the new app
        :return: A new app
        """
        ensure_client()
        app = await cls.get.aio(name=name, project=project, domain=domain)
        updated_app_spec.creator.CopyFrom(app.pb2.spec.creator)
        new_app = app_definition_pb2.App(
            metadata=app_definition_pb2.Meta(
                id=app.pb2.metadata.id,
                revision=app.revision,
                labels=labels if labels else app.pb2.metadata.labels,
            ),
            spec=updated_app_spec,
            status=app.pb2.status,
        )
        return await cls.update.aio(new_app, reason=reason)

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

    @syncify
    @classmethod
    async def create(cls, app: app_definition_pb2.App) -> App:
        ensure_client()
        try:
            resp = await get_client().app_service.Create(app_payload_pb2.CreateRequest(app=app))
            created_app = cls(resp.app)
            logger.info(f"Deployed app {created_app.name} with revision {created_app.revision}")
            return created_app
        except grpc.aio.AioRpcError as e:
            if e.code() in [grpc.StatusCode.ABORTED, grpc.StatusCode.ALREADY_EXISTS]:
                if e.code() == grpc.StatusCode.ALREADY_EXISTS:
                    logger.warning(f"App {app.metadata.id.name} already exists, updating...")
                elif e.code() == grpc.StatusCode.ABORTED:
                    logger.warning(f"Create App {app.metadata.id.name} was aborted on server, check state!")
                return await App.replace.aio(
                    name=app.metadata.id.name,
                    labels=app.metadata.labels,
                    updated_app_spec=app.spec,
                    reason="User requested serve from sdk",
                    project=app.metadata.id.project,
                    domain=app.metadata.id.domain,
                )
            raise
