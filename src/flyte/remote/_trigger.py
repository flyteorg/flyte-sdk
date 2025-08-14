from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from flyte._protos.workflow import run_definition_pb2
from flyte.syncify import syncify
from flyte.trigger import Trigger as TriggerDef

from .._initialize import ensure_client, get_client, get_common_config
from ._common import ToJSONMixin


@dataclass
class Trigger(ToJSONMixin):
    pb2: run_definition_pb2.Trigger

    @syncify
    @classmethod
    async def create(
        cls,
        *trigger: TriggerDef,
        task_name: str | None = None,
        task_version: str | None = None,
    ):
        """
        Create a new trigger in the Flyte platform.

        :param trigger: The TriggerDef object containing the trigger definition.
        :param task_name: Optional name of the task to associate with the trigger.
        :param task_version: Optional version of the task to associate with the trigger.
        """
        ensure_client()
        cfg = get_common_config()
        triggers = []
        for t in trigger:
            triggers.append(
                run_definition_pb2.TriggerSpec(
                    name=trigger.name,
                    automation=run_definition_pb2.CronAutomation(  # this should be a oneof the automation types
                        expression=trigger.automation.expression,
                    ),
                    description=trigger.description,
                    inputs=trigger.inputs or {},
                    env=trigger.env or {},
                    interruptable=trigger.interruptable,
                    auto_activate=trigger.auto_activate,
                )
            )
        resp = await get_client().triggers_service.CreateTrigger(  # type: ignore
            request=run_definition_pb2.CreateTriggerRequest(
                trigger=triggers,  # Note multiple triggers can be created at once
                task_name=task_name,
                task_version=task_version,
                project=cfg.project,
                domain=cfg.domain,
                organization=cfg.org,
                replace_mode=run_definition_pb2.Trigger.ReplaceMode.REPLACE_MODE_All,
                # replace all existing triggers (this can be future too)
            )
        )

        return [
            cls(pb2=t) for t in resp.triggers
        ]  # Each Trigger should have name, revision and associated task name/version, created at

    @syncify
    @classmethod
    async def get(cls, *, name: str, task_name: str) -> Trigger:
        """
        Retrieve a trigger by its name and associated task name.
        """
        ensure_client()
        cfg = get_common_config()
        resp = await get_client().triggers_service.GetTrigger(  # type: ignore
            request=run_definition_pb2.GetTriggerRequest(
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
                request=run_definition_pb2.ListTriggersRequest(
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
        await get_client().triggers_service.PauseTrigger(  # type: ignore
            request=run_definition_pb2.PauseTriggerRequest(
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
        await get_client().triggers_service.ResumeTrigger(  # type: ignore
            request=run_definition_pb2.ResumeTriggerRequest(
                name=name,
                task_name=task_name,
                project=cfg.project,
                domain=cfg.domain,
                organization=cfg.org,
                catchup=catchup,  # If True, it will run all missed executions since the last pause
            )
        )
        return

    def _rich_automation(self, automation: run_definition_pb2.CronAutomation):
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
