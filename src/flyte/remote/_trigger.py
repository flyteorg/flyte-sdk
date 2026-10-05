from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import AsyncIterator

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.common import identifier_pb2, list_pb2
from flyteidl2.task import common_pb2, task_definition_pb2
from flyteidl2.trigger import trigger_definition_pb2, trigger_service_pb2

import flyte
from flyte._initialize import ensure_client, get_client, get_init_config
from flyte._internal.runtime import trigger_serde
from flyte._logging import logger
from flyte._sentry import track_operation
from flyte.syncify import syncify

from ._common import ToJSONMixin
from ._task import Task, TaskDetails


def _describe_automation(automation: common_pb2.TriggerAutomationSpec) -> str:
    """
    Render a trigger's automation specification as a single human-readable line.

    One line (rather than one field per automation kind) keeps the columns aligned when a list of
    triggers with differing automations is formatted as a table.
    """
    if automation.type == common_pb2.TriggerAutomationSpecType.TYPE_NONE:
        return "none"
    if automation.type == common_pb2.TriggerAutomationSpecType.TYPE_ARTIFACT:
        artifact = automation.artifact
        version = f"@{artifact.version}" if artifact.version else ""
        bind = f" -> {artifact.input_arg}" if artifact.input_arg else ""
        return f"on artifact: {artifact.artifact_name}{version}{bind}"
    if automation.type != common_pb2.TriggerAutomationSpecType.TYPE_SCHEDULE:
        return common_pb2.TriggerAutomationSpecType.Name(automation.type)

    schedule = automation.schedule
    # `schedule` is a message, so its sub-messages are never None; the oneof tag is what says
    # which kind of schedule was actually set.
    match schedule.WhichOneof("expression"):
        case "cron":
            return f"cron: {schedule.cron.expression} ({schedule.cron.timezone or 'UTC'})"
        case "cron_expression":
            return f"cron: {schedule.cron_expression}"
        case "rate":
            rate = schedule.rate
            unit = common_pb2.FixedRateUnit.Name(rate.unit).removeprefix("FIXED_RATE_UNIT_").lower()
            if rate.value != 1:
                unit += "s"
            start = rate.start_time.ToDatetime() if rate.HasField("start_time") else "now"
            return f"every {rate.value} {unit} starting at {start}"
        case _:
            return "schedule: unset"


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
        resp = await get_client().trigger_service.get_trigger_details(
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
        """
        Name of the trigger.
        """
        return self.id.name.name

    @property
    def id(self) -> identifier_pb2.TriggerIdentifier:
        """
        Identifier for the trigger.
        """
        return self.pb2.id

    @property
    def task_name(self) -> str:
        """
        Name of the associated task
        """
        return self.pb2.id.name.task_name

    @property
    def automation_spec(self) -> common_pb2.TriggerAutomationSpec:
        """
        Get the automation specification for the trigger (e.g., schedule, event).
        """
        return self.pb2.automation_spec

    @property
    def metadata(self) -> trigger_definition_pb2.TriggerMetadata:
        """
        Get the metadata for the trigger.
        """
        return self.pb2.metadata

    @property
    def status(self) -> trigger_definition_pb2.TriggerStatus:
        """
        Get the current status of the trigger.
        """
        return self.pb2.status

    @property
    def is_active(self) -> bool:
        """
        Check if the trigger is currently active.
        """
        return self.pb2.spec.active

    @cached_property
    def trigger(self) -> trigger_definition_pb2.Trigger:
        """
        Get the trigger protobuf object constructed from the details.
        """
        return trigger_definition_pb2.Trigger(
            id=self.pb2.id,
            automation_spec=self.automation_spec,
            metadata=self.metadata,
            status=self.status,
            active=self.is_active,
        )


@dataclass
class Trigger(ToJSONMixin):
    """
    Represents a trigger in the Flyte platform.
    """

    pb2: trigger_definition_pb2.Trigger
    details: TriggerDetails | None = None

    @syncify
    @classmethod
    async def create(
        cls,
        trigger: flyte.Trigger,
        task_name: str,
        task_version: str | None = None,
        active: bool | None = None,
    ) -> Trigger:
        """
        Create a trigger in the Flyte platform, or replace the one of the same name on the task.

        When a trigger of the same name already exists, it is replaced at its latest revision and keeps its current
        active state (so replacing a trigger someone deactivated does not silently re-activate it, and vice versa);
        `trigger.auto_activate` applies only to a new trigger, since a plain bool cannot tell an explicit choice
        from its default. Pass `active` to set the state explicitly either way.

        Args:
            trigger: The flyte.Trigger object containing the trigger definition.
            task_name: Optional name of the task to associate with the trigger.
            task_version: The task version to bind; the latest when omitted.
            active: Explicit activation state; None keeps an existing trigger's state, or uses
                `trigger.auto_activate` for a new one.
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
        except ConnectError as e:
            if e.code == Code.NOT_FOUND:
                raise ValueError(f"Task {task_name}:{task_version or 'latest'} not found") from e
            raise

        task_trigger = await trigger_serde.to_task_trigger(
            t=trigger,
            task_name=task_name,
            task_inputs=task.pb2.spec.task_template.interface.inputs,
            task_default_inputs=list(task.pb2.spec.default_inputs),
        )

        # Offload the trigger inputs out-of-band via DataProxy, but only when there is input data worth
        # offloading. At fire time the backend passes the stored URI + hash through as the run's
        # OffloadedInputData. The task was just fetched above and is already registered, so we
        # reference it by task_id.
        offloaded_input_data = None
        if task_trigger.spec.inputs.literals:
            offloaded_input_data = await trigger_serde.offload_trigger_inputs(
                task_trigger.spec.inputs,
                org=cfg.org,
                project=cfg.project,
                domain=cfg.domain,
                task_name=task_name,
                task_version=task.version,
            )

        spec = trigger_definition_pb2.TriggerSpec(
            active=task_trigger.spec.active,
            run_spec=task_trigger.spec.run_spec,
            task_version=task.version,
        )
        if offloaded_input_data is not None:
            spec.offloaded_input_data.CopyFrom(offloaded_input_data)
        else:
            # Zero trust not enabled on the backend: register with inline inputs (pre-offload flow).
            spec.inputs.CopyFrom(task_trigger.spec.inputs)

        trigger_name = identifier_pb2.TriggerName(
            name=trigger.name,
            task_name=task_name,
            org=cfg.org,
            project=cfg.project,
            domain=cfg.domain,
        )
        # Replacing a trigger of the same name needs its latest revision (optimistic locking). The lookup is
        # best-effort: if it fails for any reason other than NOT_FOUND, fall back to the pre-lookup behavior
        # (revision unset) rather than aborting the deploy.
        revision = 0
        existing_active: bool | None = None
        try:
            existing = await get_client().trigger_service.get_trigger_details(
                request=trigger_service_pb2.GetTriggerDetailsRequest(name=trigger_name)
            )
            revision = existing.trigger.id.revision or 1
            if existing.trigger.HasField("spec"):
                existing_active = existing.trigger.spec.active
        except ConnectError as e:
            if e.code != Code.NOT_FOUND:
                logger.debug(f"Trigger lookup for {trigger.name} failed ({e.code}); deploying without a revision")
        except Exception as e:
            logger.debug(f"Trigger lookup for {trigger.name} failed ({e}); deploying without a revision")
        if active is not None:
            spec.active = active
        elif existing_active is not None:
            spec.active = existing_active
        with track_operation("deploy_trigger"):
            resp = await get_client().trigger_service.deploy_trigger(
                request=trigger_service_pb2.DeployTriggerRequest(
                    name=trigger_name,
                    revision=revision or None,
                    spec=spec,
                    automation_spec=task_trigger.automation_spec,
                )
            )

        details = TriggerDetails(pb2=resp.trigger)

        return cls(pb2=details.trigger, details=details)

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
            resp = await get_client().trigger_service.list_triggers(
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
        await get_client().trigger_service.update_triggers(
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
    async def delete(cls, name: str, task_name: str, project: str | None = None, domain: str | None = None):
        """
        Delete a trigger by its name.
        """
        ensure_client()
        cfg = get_init_config()
        await get_client().trigger_service.delete_triggers(
            request=trigger_service_pb2.DeleteTriggersRequest(
                names=[
                    identifier_pb2.TriggerName(
                        org=cfg.org,
                        project=project or cfg.project,
                        domain=domain or cfg.domain,
                        name=name,
                        task_name=task_name,
                    )
                ],
            )
        )

    @property
    def id(self) -> identifier_pb2.TriggerIdentifier:
        """
        Get the unique identifier for the trigger.
        """
        return self.pb2.id

    @property
    def name(self) -> str:
        """
        Get the name of the trigger.
        """
        return self.id.name.name

    @property
    def task_name(self) -> str:
        """
        Get the name of the task associated with this trigger.
        """
        return self.id.name.task_name

    @property
    def automation_spec(self) -> common_pb2.TriggerAutomationSpec:
        """
        Get the automation specification for the trigger.
        """
        return self.pb2.automation_spec

    @property
    def url(self) -> str:
        """
        Get the console URL for viewing the trigger.
        """
        client = get_client()
        return client.console.trigger_url(
            project=self.pb2.id.name.project,
            domain=self.pb2.id.name.domain,
            task_name=self.pb2.id.name.task_name,
            trigger_name=self.name,
        )

    async def get_details(self) -> TriggerDetails:
        """
        Get detailed information about this trigger.
        """
        if not self.details:
            self.details = await TriggerDetails.get.aio(name=self.name, task_name=self.task_name)
        return self.details

    @property
    def is_active(self) -> bool:
        """
        Check if the trigger is currently active.
        """
        return self.pb2.active

    def _rich_automation(self, automation: common_pb2.TriggerAutomationSpec):
        """
        Generate rich representation fields for the automation specification.
        """
        yield "automation", _describe_automation(automation)

    def __rich_repr__(self):
        """
        Rich representation of the Trigger object for pretty printing.
        """
        yield "task_name", self.task_name
        yield "name", self.name
        yield from self._rich_automation(self.pb2.automation_spec)
        yield "auto_activate", self.is_active
