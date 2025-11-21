from __future__ import annotations

from typing import AsyncIterator, cast

from flyteidl2.app import app_definition_pb2, app_payload_pb2

from flyte._initialize import ensure_client, get_client, get_init_config
from flyte.syncify import syncify

from ._common import ToJSONMixin


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

    @classmethod
    @syncify
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

    @classmethod
    @syncify
    async def watch(cls, name: str, /, project: str | None = None, domain: str | None = None):
        ensure_client()
        cfg = get_init_config()
        call = cast(
            AsyncIterator[app_payload_pb2.WatchResponse],
            get_client().app_service.Watch(
                request=app_payload_pb2.WatchRequest(
                    app_id=app_definition_pb2.Identifier(
                        org=cfg.org,
                        project=project or cfg.project,
                        domain=domain or cfg.domain,
                        name=name,
                    ),
                )
            ),
        )
        async for resp in call:
            if resp.update_event:
                updated_app = resp.update_event.updated_app
                current_status = updated_app.status.conditions[-1].deployment_status
                if current_status == app_definition_pb2.Status.DeploymentStatus.DEPLOYMENT_STATUS_ACTIVE:
                    status_name = app_definition_pb2.Status.DeploymentStatus.Name(current_status)
                    return status_name
