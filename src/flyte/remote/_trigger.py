from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import flyte
from flyte._initialize import ensure_client, get_client, get_common_config
from flyte._internal.runtime import trigger_serde
from flyte._protos.workflow import task_definition_pb2, trigger_definition_pb2, trigger_service_pb2
from flyte.syncify import syncify

from ._common import ToJSONMixin


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
        resp = await get_client().triggers_service.GetTriggerDetails(  # type: ignore
            request=trigger_service_pb2.GetTriggerDetailsRequest(
                name=name,
                project=cfg.project,
                domain=cfg.domain,
                organization=cfg.org,
            )
        )
        return cls(pb2=resp.trigger)


@dataclass
class Trigger(ToJSONMixin):
    pb2: trigger_definition_pb2.Trigger
    details: trigger_definition_pb2.TriggerDetails | None = None

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
        ensure_client()
        cfg = get_common_config()
        resp = await get_client().triggers_service.GetTrigger(  # type: ignore
            request=trigger_service_pb2.GetTriggerRequest(
                name=name,
                task_name=task_name,
                project=cfg.project,
                domain=cfg.domain,
                organization=cfg.org,
            )
        )
        return cls(pb2=resp.trigger)

    @syncify
    @classmethod
    async def listall(cls, task_name: str | None = None, limit: int = 100) -> Iterable[Trigger]:
        """
        List all triggers associated with a specific task or all tasks if no task name is provided.
        """
        ensure_client()
        cfg = get_common_config()
        token = None
        while True:
            resp = await get_client().triggers_service.ListTriggers(  # type: ignore
                request=trigger_service_pb2.ListTriggersRequest(
                    project=cfg.project,
                    domain=cfg.domain,
                    organization=cfg.org,
                    task_name=task_name,
                    limit=limit,
                    token=token,
                )
            )
            token = resp.token
            for r in resp.secrets:
                yield cls(r)
            if not token:
                break

    @syncify
    @classmethod
    async def pause(cls, name: str, task_name: str):
        """
        Pause a trigger by its name and associated task name.
        """
        ensure_client()
        cfg = get_common_config()
        await get_client().triggers_service.UpdateTrigger(  # type: ignore
            request=trigger_service_pb2.UpdateTriggersRequest(
                name=name,
                task_name=task_name,
                project=cfg.project,
                domain=cfg.domain,
                organization=cfg.org,
            )
        )

    @syncify
    @classmethod
    async def resume(cls, name: str, task_name: str, catchup: bool = False):
        """
        Resume a paused trigger by its name and associated task name.
        """
        ensure_client()
        cfg = get_common_config()
        await get_client().triggers_service.UpdateTrigger(  # type: ignore
            request=trigger_service_pb2.UpdateTriggersRequest(
                name=name,
                task_name=task_name,
                project=cfg.project,
                domain=cfg.domain,
                organization=cfg.org,
                catchup=catchup,  # If True, it will run all missed executions since the last pause
            )
        )
        return

    async def details(self) -> TriggerDetails:
        pass

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
