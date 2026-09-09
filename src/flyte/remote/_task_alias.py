from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator

from flyteidl2.common import identity_pb2, list_pb2
from flyteidl2.task import task_definition_pb2, task_service_pb2

from flyte._initialize import ensure_client, get_client, get_init_config
from flyte.syncify import syncify

from ._common import ToJSONMixin


def _identity_name(identity: identity_pb2.EnrichedIdentity) -> str:
    """
    Render an EnrichedIdentity as a display string.

    It is a oneof of User/Application with no common name field, so mirror the
    rendering `TaskDetails` already uses for `deployed_by`.
    """
    if identity.HasField("user"):
        return identity.user.spec.email or identity.user.spec.user_handle
    if identity.HasField("application"):
        return identity.application.spec.name
    return ""


def _alias_name(task_name: str, alias: str, project: str | None, domain: str | None) -> task_service_pb2.TaskAliasName:
    cfg = get_init_config()
    return task_service_pb2.TaskAliasName(
        task_name=task_definition_pb2.TaskName(
            org=cfg.org,
            project=project or cfg.project,
            domain=domain or cfg.domain,
            name=task_name,
        ),
        alias=alias,
    )


@dataclass
class TaskAliasRevision(ToJSONMixin):
    """One entry in an alias's move history."""

    pb2: task_service_pb2.TaskAliasRevision

    @property
    def from_version(self) -> str:
        """Version the alias pointed at before this change; empty when it was created."""
        return self.pb2.from_version

    @property
    def to_version(self) -> str:
        """Version the alias pointed at after this change; empty when it was deleted."""
        return self.pb2.to_version

    @property
    def changed_by(self) -> str:
        return _identity_name(self.pb2.changed_by)

    @property
    def changed_at(self) -> datetime:
        return self.pb2.changed_at.ToDatetime()

    def __rich_repr__(self):
        yield "from", self.from_version or "-"
        yield "to", self.to_version or "(deleted)"
        yield "changed_by", self.changed_by
        yield "changed_at", self.changed_at


@dataclass
class TaskAlias(ToJSONMixin):
    """
    A mutable named pointer to an immutable task version, e.g. `prod -> v1.4.0`.

    Deploying a task never moves an alias; only `set` does. That is the whole point:
    an operator can promote a version and rely on it staying there while CI keeps
    deploying.
    """

    pb2: task_service_pb2.TaskAlias

    @property
    def alias(self) -> str:
        return self.pb2.name.alias

    @property
    def task_name(self) -> str:
        return self.pb2.name.task_name.name

    @property
    def version(self) -> str:
        """The immutable version this alias currently resolves to."""
        return self.pb2.version

    @property
    def set_by(self) -> str:
        return _identity_name(self.pb2.set_by)

    @property
    def set_at(self) -> datetime:
        return self.pb2.set_at.ToDatetime()

    def __rich_repr__(self):
        yield "alias", self.alias
        yield "version", self.version
        yield "set_by", self.set_by
        yield "set_at", self.set_at

    @syncify
    @classmethod
    async def set(
        cls,
        task_name: str,
        alias: str,
        version: str,
        project: str | None = None,
        domain: str | None = None,
    ) -> tuple[TaskAlias, str]:
        """
        Create an alias or move it to a different version — promote, or roll back.

        Returns the alias and the version it pointed at before, so callers can render
        `prod: v1.4.0 -> v1.7.0` without a second lookup. The previous version is empty
        when the alias was just created.

        Pointing an alias at a version that was never deployed is an error rather than a
        pointer that fails later at launch time.
        """
        ensure_client()
        resp = await get_client().task_service.set_task_alias(
            request=task_service_pb2.SetTaskAliasRequest(
                name=_alias_name(task_name, alias, project, domain),
                version=version,
            )
        )
        return cls(pb2=resp.alias), resp.previous_version

    @syncify
    @classmethod
    async def get(
        cls,
        task_name: str,
        alias: str,
        project: str | None = None,
        domain: str | None = None,
    ) -> TaskAlias:
        """Resolve an alias to its current version, with who moved it there and when."""
        ensure_client()
        resp = await get_client().task_service.get_task_alias(
            request=task_service_pb2.GetTaskAliasRequest(name=_alias_name(task_name, alias, project, domain))
        )
        return cls(pb2=resp.alias)

    @syncify
    @classmethod
    async def listall(
        cls,
        task_name: str,
        project: str | None = None,
        domain: str | None = None,
        limit: int = 100,
    ) -> AsyncIterator[TaskAlias]:
        """List every alias defined for a task."""
        ensure_client()
        cfg = get_init_config()
        token = None
        while True:
            resp = await get_client().task_service.list_task_aliases(
                request=task_service_pb2.ListTaskAliasesRequest(
                    task_name=task_definition_pb2.TaskName(
                        org=cfg.org,
                        project=project or cfg.project,
                        domain=domain or cfg.domain,
                        name=task_name,
                    ),
                    request=list_pb2.ListRequest(limit=limit, token=token or ""),
                )
            )
            for alias in resp.aliases:
                yield cls(pb2=alias)
            token = resp.token
            if not token:
                break

    @syncify
    @classmethod
    async def delete(
        cls,
        task_name: str,
        alias: str,
        project: str | None = None,
        domain: str | None = None,
    ) -> None:
        """Remove an alias. The versions it pointed at are unaffected."""
        ensure_client()
        await get_client().task_service.delete_task_alias(
            request=task_service_pb2.DeleteTaskAliasRequest(name=_alias_name(task_name, alias, project, domain))
        )

    @syncify
    @classmethod
    async def history(
        cls,
        task_name: str,
        alias: str,
        project: str | None = None,
        domain: str | None = None,
        limit: int = 100,
    ) -> AsyncIterator[TaskAliasRevision]:
        """Full move history for one alias, newest first."""
        ensure_client()
        token = None
        while True:
            resp = await get_client().task_service.get_task_alias_history(
                request=task_service_pb2.GetTaskAliasHistoryRequest(
                    name=_alias_name(task_name, alias, project, domain),
                    request=list_pb2.ListRequest(limit=limit, token=token or ""),
                )
            )
            for revision in resp.revisions:
                yield TaskAliasRevision(pb2=revision)
            token = resp.token
            if not token:
                break
