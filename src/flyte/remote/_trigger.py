from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import AsyncIterator

import flyte
from flyte._initialize import ensure_client, get_client, get_common_config
from flyte._internal.runtime import trigger_serde
from flyte._protos.workflow import task_definition_pb2, trigger_definition_pb2, trigger_service_pb2
from flyte.syncify import syncify

from ._common import ToJSONMixin
from .._protos.common import identifier_pb2


@dataclass
class TriggerDetails(ToJSONMixin):
    pb2: trigger_definition_pb2.TriggerDetails

    @syncify
    @classmethod
    async def get(cls, *, name: str) -> TriggerDetails:
        """
        Retrieve detailed information about a specific trigger by its name.
        """
        ensure_client()
        cfg = get_common_config()
        resp = await get_client().trigger_service.GetTriggerDetails(
            request=trigger_service_pb2.GetTriggerDetailsRequest(
                name=identifier_pb2.TriggerName(
                    name=name,
                    org=cfg.org,
                    project=cfg.project,
                    domain=cfg.domain,
                ),
            )
        )
        return cls(pb2=resp.trigger)

    @property
    def name(self) -> str:
        return self.id.name.name

    @property
    def id(self) -> identifier_pb2.TriggerIdentifier:
        return self.pb2.id

    @property
    def task_id(self) -> task_definition_pb2.TaskIdentifier:
        return self.pb2.spec.task_id

    @property
    def automation_spec(self) -> trigger_definition_pb2.TriggerAutomationSpec:
        return self.pb2.automation_spec

    @property
    def meta(self) -> trigger_definition_pb2.TriggerMetadata:
        return self.pb2.meta

    @property
    def status(self) -> trigger_definition_pb2.TriggerStatus:
        return self.pb2.status

    @property
    def is_active(self) -> bool:
        return self.pb2.spec.active

    @cached_property
    def trigger(self) -> trigger_definition_pb2.Trigger:
        return trigger_definition_pb2.Trigger(
            id=self.pb2.id,
            task_id=self.pb2.spec.task_id,
            automation_spec=self.pb2.automation_spec,
            meta=self.pb2.meta,
            status=self.pb2.status,
            active=self.pb2.spec.active,
        )


@dataclass
class Trigger(ToJSONMixin):
    pb2: trigger_definition_pb2.Trigger
    details: TriggerDetails | None = None

    @syncify
    @classmethod
    async def create(
            cls,
            trigger: flyte.Trigger,
            task_name: str,
            task_version: str | None = None,
    ) -> Trigger:
        """
        Create a new trigger in the Flyte platform.

        :param trigger: The flyte.Trigger object containing the trigger definition.
        :param task_name: Optional name of the task to associate with the trigger.
        :param task_version: Optional version of the task to associate with the trigger.
        """
        ensure_client()
        cfg = get_common_config()
        trigger_details = trigger_serde.to_trigger_details(
            task_id=task_definition_pb2.TaskIdentifier(
                name=task_name,
                project=cfg.project,
                domain=cfg.domain,
                org=cfg.org,
                version=task_version,
            ),
            t=trigger,
        )
        resp = await get_client().trigger_service.SaveTrigger(
            request=trigger_service_pb2.SaveTriggerRequest(
                trigger=trigger_details,
            )
        )

        return cls(
            pb2=trigger_definition_pb2.Trigger(automation_spec=resp.trigger.automation_spec), details=resp.trigger
        )

    @syncify
    @classmethod
    async def get(cls, *, name: str, task_name: str) -> Trigger:
        """
        Retrieve a trigger by its name and associated task name.
        """
        details = TriggerDetails.get.aio(name=name)
        return cls(
            pb2=trigger_definition_pb2.Trigger(
                id=details.id,
                task_id=details.task_id,
                automation_spec=details.details.automation_spec,
                meta=details.meta,
                status=details.status,
                active=details.is_active,
            ),
            details=details,
        )

    @syncify
    @classmethod
    async def listall(cls, task_name: str | None = None, limit: int = 100) -> AsyncIterator[Trigger]:
        """
        List all triggers associated with a specific task or all tasks if no task name is provided.
        """
        ensure_client()
        cfg = get_common_config()
        token = None
        while True:
            resp = await get_client().trigger_service.ListTriggers(
                request=trigger_service_pb2.ListTriggersRequest(
                    organization=cfg.org,
                    project=cfg.project,
                    domain=cfg.domain,
                    token=token,
                    limit=limit,
                )
            )
            token = resp.token
            for r in resp.triggers:
                yield cls(r)
            if not token:
                break

    @syncify
    @classmethod
    async def deactivate(cls, name: str, task_name: str):
        """
        Pause a trigger by its name and associated task name.
        """
        ensure_client()
        cfg = get_common_config()
        await get_client().trigger_service.UpdateTriggers(
            request=trigger_service_pb2.UpdateTriggersRequest(
                organization=cfg.org,
                project=cfg.project,
                domain=cfg.domain,
                name=name,
                state=trigger_definition_pb2.TriggerState.TRIGGER_STATE_PAUSED,
            )
        )

    @syncify
    @classmethod
    async def activate(cls, name: str, task_name: str):
        """
        Resume a paused trigger by its name and associated task name.
        """
        ensure_client()
        cfg = get_common_config()
        await get_client().trigger_service.UpdateTriggers(
            request=trigger_service_pb2.UpdateTriggersRequest(
                organization=cfg.org,
                project=cfg.project,
                domain=cfg.domain,
                name=name,
                state=trigger_definition_pb2.TriggerState.TRIGGER_STATE_ACTIVE,
            )
        )

    @syncify
    @classmethod
    async def delete(cls, name: str):
        """
        Delete a trigger by its name.
        """
        ensure_client()
        cfg = get_common_config()
        await get_client().trigger_service.DeleteTriggers(
            request=trigger_service_pb2.DeleteTriggersRequest(
                organization=cfg.org,
                project=cfg.project,
                domain=cfg.domain,
                name=name,
            )
        )

    async def get_details(self) -> TriggerDetails:
        """
        Get detailed information about this trigger.
        """
        return await TriggerDetails.get(name=self.pb2.name)

    def _rich_automation(self, automation: trigger_definition_pb2.CronAutomation):
        yield "cron", automation.expression

    def __rich_repr__(self):
        yield "name", self.pb2.name
        yield from self._rich_automation(self.pb2.automation)
        yield "description", self.pb2.description
        yield "inputs", self.pb2.inputs
        yield "env", self.pb2.env
        yield "interruptable", self.pb2.interruptable
        yield "auto_activate", self.pb2.auto_activate
        yield "created_at", self.pb2.created_at.ToDatetime() if self.pb2.HasField("created_at") else None
        yield "updated_at", self.pb2.updated_at.ToDatetime() if self.pb2.HasField("updated_at") else None
        yield "revision", self.pb2.revision
        yield "task_name", self.pb2.task_name
        yield "task_version", self.pb2.task_version
        yield "state", self.pb2.state  # paused, active, etc.
