from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import AsyncIterator

import grpc.aio
from flyteidl2.common import identifier_pb2, list_pb2
from flyteidl2.task import common_pb2, task_definition_pb2
from flyteidl2.trigger import trigger_definition_pb2, trigger_service_pb2

import flyte
from flyte._initialize import ensure_client, get_client, get_init_config
from flyte._internal.runtime import trigger_serde
from flyte.syncify import syncify

from ._common import ToJSONMixin
from ._task import Task, TaskDetails


@dataclass
class TriggerDetails(ToJSONMixin):
    pb2: trigger_definition_pb2.TriggerDetails

    @syncify
    @classmethod
    async def get(cls, *, name: str, task_name: str) -> TriggerDetails:
        """
        Retrieve detailed information about a specific trigger by its name.
        """
        ensure_client()
        cfg = get_init_config()
        resp = await get_client().trigger_service.GetTriggerDetails(
            request=trigger_service_pb2.GetTriggerDetailsRequest(
                name=identifier_pb2.TriggerName(
                    task_name=task_name,
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
    def task_name(self) -> str:
        return self.pb2.id.name.task_name

    @property
    def automation_spec(self) -> common_pb2.TriggerAutomationSpec:
        return self.pb2.automation_spec

    @property
    def metadata(self) -> trigger_definition_pb2.TriggerMetadata:
        return self.pb2.metadata

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
            automation_spec=self.automation_spec,
            metadata=self.metadata,
            status=self.status,
            active=self.is_active,
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
        """
        ensure_client()
        cfg = get_init_config()

        # Fetch the task to ensure it exists and to get its input definitions
        try:
            lazy = (
                Task.get(name=task_name, version=task_version)
                if task_version
                else Task.get(name=task_name, auto_version="latest")
            )
            task: TaskDetails = await lazy.fetch.aio()

            task_trigger = await trigger_serde.to_task_trigger(
                t=trigger,
                task_name=task_name,
                task_inputs=task.pb2.spec.task_template.interface.inputs,
                task_default_inputs=list(task.pb2.spec.default_inputs),
            )

            resp = await get_client().trigger_service.DeployTrigger(
                request=trigger_service_pb2.DeployTriggerRequest(
                    name=identifier_pb2.TriggerName(
                        name=trigger.name,
                        task_name=task_name,
                        org=cfg.org,
                        project=cfg.project,
                        domain=cfg.domain,
                    ),
                    spec=trigger_definition_pb2.TriggerSpec(
                        active=task_trigger.spec.active,
                        inputs=task_trigger.spec.inputs,
                        run_spec=task_trigger.spec.run_spec,
                        task_version=task.version,
                    ),
                    automation_spec=task_trigger.automation_spec,
                )
            )

            details = TriggerDetails(pb2=resp.trigger)

            return cls(pb2=details.trigger, details=details)
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise ValueError(f"Task {task_name}:{task_version or 'latest'} not found") from e
            raise

    @syncify
    @classmethod
    async def get(cls, *, name: str, task_name: str) -> TriggerDetails:
        """
        Retrieve a trigger by its name and associated task name.
        """
        return await TriggerDetails.get.aio(name=name, task_name=task_name)

    @syncify
    @classmethod
    async def listall(
        cls, task_name: str | None = None, task_version: str | None = None, limit: int = 100
    ) -> AsyncIterator[Trigger]:
        """
        List all triggers associated with a specific task or all tasks if no task name is provided.
        """
        ensure_client()
        cfg = get_init_config()
        token = None
        task_name_id = None
        project_id = None
        task_id = None
        if task_name and task_version:
            task_id = task_definition_pb2.TaskIdentifier(
                name=task_name,
                project=cfg.project,
                domain=cfg.domain,
                org=cfg.org,
                version=task_version,
            )
        elif task_name:
            task_name_id = task_definition_pb2.TaskName(
                name=task_name,
                project=cfg.project,
                domain=cfg.domain,
                org=cfg.org,
            )
        else:
            project_id = identifier_pb2.ProjectIdentifier(
                organization=cfg.org,
                domain=cfg.domain,
                name=cfg.project,
            )

        while True:
            resp = await get_client().trigger_service.ListTriggers(
                request=trigger_service_pb2.ListTriggersRequest(
                    project_id=project_id,
                    task_id=task_id,
                    task_name=task_name_id,
                    request=list_pb2.ListRequest(
                        limit=limit,
                        token=token,
                    ),
                )
            )
            token = resp.token
            for r in resp.triggers:
                yield cls(r)
            if not token:
                break

    @syncify
    @classmethod
    async def update(cls, name: str, task_name: str, active: bool):
        """
        Pause a trigger by its name and associated task name.
        """
        ensure_client()
        cfg = get_init_config()
        await get_client().trigger_service.UpdateTriggers(
            request=trigger_service_pb2.UpdateTriggersRequest(
                names=[
                    identifier_pb2.TriggerName(
                        org=cfg.org,
                        project=cfg.project,
                        domain=cfg.domain,
                        name=name,
                        task_name=task_name,
                    )
                ],
                active=active,
            )
        )

    @syncify
    @classmethod
    async def delete(cls, name: str, task_name: str):
        """
        Delete a trigger by its name.
        """
        ensure_client()
        cfg = get_init_config()
        await get_client().trigger_service.DeleteTriggers(
            request=trigger_service_pb2.DeleteTriggersRequest(
                names=[
                    identifier_pb2.TriggerName(
                        org=cfg.org,
                        project=cfg.project,
                        domain=cfg.domain,
                        name=name,
                        task_name=task_name,
                    )
                ],
            )
        )

    @property
    def id(self) -> identifier_pb2.TriggerIdentifier:
        return self.pb2.id

    @property
    def name(self) -> str:
        return self.id.name.name

    @property
    def task_name(self) -> str:
        return self.id.name.task_name

    @property
    def automation_spec(self) -> common_pb2.TriggerAutomationSpec:
        return self.pb2.automation_spec

    async def get_details(self) -> TriggerDetails:
        """
        Get detailed information about this trigger.
        """
        if not self.details:
            details = await TriggerDetails.get.aio(name=self.pb2.id.name.name)
            self.details = details
        return self.details

    @property
    def is_active(self) -> bool:
        return self.pb2.active

    def _rich_automation(self, automation: common_pb2.TriggerAutomationSpec):
        if automation.type == common_pb2.TriggerAutomationSpec.type.TYPE_NONE:
            yield "none", None
        elif automation.type == common_pb2.TriggerAutomationSpec.type.TYPE_SCHEDULE:
            if automation.schedule.cron is not None:
                yield "cron", automation.schedule.cron
            elif automation.schedule.rate is not None:
                r = automation.schedule.rate
                yield (
                    "fixed_rate",
                    (
                        f"Every [{r.value}] {r.unit} starting at "
                        f"{r.start_time.ToDatetime() if automation.HasField('start_time') else 'now'}"
                    ),
                )

    def __rich_repr__(self):
        yield "task_name", self.task_name
        yield "name", self.name
        yield from self._rich_automation(self.pb2.automation_spec)
        yield "auto_activate", self.is_active
