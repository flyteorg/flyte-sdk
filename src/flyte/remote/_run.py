from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncGenerator, AsyncIterator, Literal, Tuple

import grpc
import rich.repr
from flyteidl2.common import identifier_pb2, list_pb2
from flyteidl2.workflow import run_definition_pb2, run_service_pb2

from flyte._initialize import ensure_client, get_client, get_init_config
from flyte._logging import logger
from flyte.syncify import syncify

from . import Action, ActionDetails, ActionInputs, ActionOutputs
from ._action import _action_details_rich_repr, _action_rich_repr
from ._common import ToJSONMixin
from ._console import get_run_url

# @kumare3 is sadpanda, because we have to create a mirror of phase types here, because protobuf phases are ghastly
Phase = Literal[
    "queued", "waiting_for_resources", "initializing", "running", "succeeded", "failed", "aborted", "timed_out"
]


@dataclass
class Run(ToJSONMixin):
    """
    A class representing a run of a task. It is used to manage the run of a task and its state on the remote
    Union API.
    """

    pb2: run_definition_pb2.Run
    action: Action = field(init=False)
    _details: RunDetails | None = None

    def __post_init__(self):
        """
        Initialize the Run object with the given run definition.
        """
        if not self.pb2.HasField("action"):
            raise RuntimeError("Run does not have an action")
        self.action = Action(self.pb2.action)

    @syncify
    @classmethod
    async def listall(
        cls,
        in_phase: Tuple[Phase] | None = None,
        created_by_subject: str | None = None,
        sort_by: Tuple[str, Literal["asc", "desc"]] | None = None,
        limit: int = 100,
    ) -> AsyncIterator[Run]:
        """
        Get all runs for the current project and domain.

        :param in_phase: Filter runs by one or more phases.
        :param created_by_subject: Filter runs by the subject that created them. (this is not username, but the subject)
        :param sort_by: The sorting criteria for the project list, in the format (field, order).
        :param limit: The maximum number of runs to return.
        :return: An iterator of runs.
        """
        ensure_client()
        token = None
        sort_by = sort_by or ("created_at", "asc")
        sort_pb2 = list_pb2.Sort(
            key=sort_by[0],
            direction=(list_pb2.Sort.ASCENDING if sort_by[1] == "asc" else list_pb2.Sort.DESCENDING),
        )
        filters = []
        if in_phase:
            phases = [str(run_definition_pb2.Phase.Value(f"PHASE_{p.upper()}")) for p in in_phase]
            logger.debug(f"Fetching run phases: {phases}")
            if len(phases) > 1:
                filters.append(
                    list_pb2.Filter(
                        function=list_pb2.Filter.Function.VALUE_IN,
                        field="phase",
                        values=phases,
                    ),
                )
            else:
                filters.append(
                    list_pb2.Filter(
                        function=list_pb2.Filter.Function.EQUAL,
                        field="phase",
                        values=phases[0],
                    ),
                )
        if created_by_subject:
            logger.debug(f"Fetching runs created by: {created_by_subject}")
            filters.append(
                list_pb2.Filter(
                    function=list_pb2.Filter.Function.EQUAL,
                    field="created_by",
                    values=[created_by_subject],
                ),
            )

        cfg = get_init_config()
        i = 0
        while True:
            req = list_pb2.ListRequest(
                limit=min(100, limit),
                token=token,
                sort_by=sort_pb2,
                filters=filters,
            )
            resp = await get_client().run_service.ListRuns(
                run_service_pb2.ListRunsRequest(
                    request=req,
                    org=cfg.org,
                    project_id=identifier_pb2.ProjectIdentifier(
                        organization=cfg.org,
                        domain=cfg.domain,
                        name=cfg.project,
                    ),
                )
            )
            token = resp.token
            for r in resp.runs:
                i += 1
                if i > limit:
                    return
                yield cls(r)
            if not token:
                break

    @syncify
    @classmethod
    async def get(cls, name: str) -> Run:
        """
        Get the current run.

        :return: The current run.
        """
        ensure_client()
        run_details: RunDetails = await RunDetails.get.aio(name=name)
        run = run_definition_pb2.Run(
            action=run_definition_pb2.Action(
                id=run_details.action_id,
                metadata=run_details.action_details.pb2.metadata,
                status=run_details.action_details.pb2.status,
            ),
        )
        return cls(pb2=run, _details=run_details)

    @property
    def name(self) -> str:
        """
        Get the name of the run.
        """
        return self.pb2.action.id.run.name

    @property
    def phase(self) -> str:
        """
        Get the phase of the run.
        """
        return self.action.phase

    @property
    def raw_phase(self) -> run_definition_pb2.Phase:
        """
        Get the raw phase of the run.
        """
        return self.action.raw_phase

    @syncify
    async def wait(self, quiet: bool = False, wait_for: Literal["terminal", "running"] = "terminal") -> None:
        """
        Wait for the run to complete, displaying a rich progress panel with status transitions,
        time elapsed, and error details in case of failure.
        """
        return await self.action.wait(quiet=quiet, wait_for=wait_for)

    async def watch(self, cache_data_on_done: bool = False) -> AsyncGenerator[ActionDetails, None]:
        """
        Get the details of the run. This is a placeholder for getting the run details.
        """
        return self.action.watch(cache_data_on_done=cache_data_on_done)

    @syncify
    async def show_logs(
        self,
        attempt: int | None = None,
        max_lines: int = 100,
        show_ts: bool = False,
        raw: bool = False,
        filter_system: bool = False,
    ):
        await self.action.show_logs.aio(attempt, max_lines, show_ts, raw, filter_system=filter_system)

    @syncify
    async def details(self) -> RunDetails:
        """
        Get the details of the run. This is a placeholder for getting the run details.
        """
        if self._details is None or not self._details.done():
            self._details = await RunDetails.get_details.aio(self.pb2.action.id.run)
        return self._details

    @syncify
    async def inputs(self) -> ActionInputs:
        """
        Get the inputs of the run. This is a placeholder for getting the run inputs.
        """
        details = await self.details.aio()
        return await details.inputs()

    @syncify
    async def outputs(self) -> ActionOutputs:
        """
        Get the outputs of the run. This is a placeholder for getting the run outputs.
        """
        details = await self.details.aio()
        return await details.outputs()

    @property
    def url(self) -> str:
        """
        Get the URL of the run.
        """
        client = get_client()
        return get_run_url(
            client.endpoint,
            insecure=client.insecure,
            project=self.pb2.action.id.run.project,
            domain=self.pb2.action.id.run.domain,
            run_name=self.name,
        )

    @syncify
    async def abort(self):
        """
        Aborts / Terminates the run.
        """
        try:
            await get_client().run_service.AbortRun(
                run_service_pb2.AbortRunRequest(
                    run_id=self.pb2.action.id.run,
                )
            )
        except grpc.aio.AioRpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                return
            raise

    def done(self) -> bool:
        """
        Check if the run is done.
        """
        return self.action.done()

    def sync(self) -> Run:
        """
        Sync the run with the remote server. This is a placeholder for syncing the run.
        """
        return self

    # TODO add add_done_callback, maybe implement sync apis etc

    def __rich_repr__(self) -> rich.repr.Result:
        """
        Rich representation of the Run object.
        """
        yield "url", f"[blue bold][link={self.url}]link[/link][/blue bold]"
        yield from _action_rich_repr(self.pb2.action)

    def __repr__(self) -> str:
        """
        String representation of the Action object.
        """
        import rich.pretty

        return rich.pretty.pretty_repr(self)


@dataclass
class RunDetails(ToJSONMixin):
    """
    A class representing a run of a task. It is used to manage the run of a task and its state on the remote
    Union API.
    """

    pb2: run_definition_pb2.RunDetails
    action_details: ActionDetails = field(init=False)

    def __post_init__(self):
        """
        Initialize the RunDetails object with the given run definition.
        """
        self.action_details = ActionDetails(self.pb2.action)

    @syncify
    @classmethod
    async def get_details(cls, run_id: identifier_pb2.RunIdentifier) -> RunDetails:
        """
        Get the details of the run. This is a placeholder for getting the run details.
        """
        ensure_client()
        resp = await get_client().run_service.GetRunDetails(
            run_service_pb2.GetRunDetailsRequest(
                run_id=run_id,
            )
        )
        return cls(resp.details)

    @syncify
    @classmethod
    async def get(cls, name: str | None = None) -> RunDetails:
        """
        Get a run by its ID or name. If both are provided, the ID will take precedence.

        :param uri: The URI of the run.
        :param name: The name of the run.
        """
        ensure_client()
        cfg = get_init_config()
        return await RunDetails.get_details.aio(
            run_id=identifier_pb2.RunIdentifier(
                org=cfg.org,
                project=cfg.project,
                domain=cfg.domain,
                name=name,
            ),
        )

    @property
    def name(self) -> str:
        """
        Get the name of the action.
        """
        return self.action_details.run_name

    @property
    def task_name(self) -> str | None:
        """
        Get the name of the task.
        """
        return self.action_details.task_name

    @property
    def action_id(self) -> identifier_pb2.ActionIdentifier:
        """
        Get the action ID.
        """
        return self.action_details.action_id

    def done(self) -> bool:
        """
        Check if the run is in a terminal state (completed or failed). This is a placeholder for checking the
        run state.
        """
        return self.action_details.done()

    async def inputs(self) -> ActionInputs:
        """
        Placeholder for inputs. This can be extended to handle inputs from the run context.
        """
        return await self.action_details.inputs()

    async def outputs(self) -> ActionOutputs:
        """
        Placeholder for outputs. This can be extended to handle outputs from the run context.
        """
        return await self.action_details.outputs()

    def __rich_repr__(self) -> rich.repr.Result:
        """
        Rich representation of the Run object.
        """
        yield "labels", str(self.pb2.run_spec.labels)
        yield "annotations", str(self.pb2.run_spec.annotations)
        yield "env-vars", str(self.pb2.run_spec.envs)
        yield "is-interruptible", str(self.pb2.run_spec.interruptible)
        yield "cache-overwrite", self.pb2.run_spec.overwrite_cache
        yield from _action_details_rich_repr(self.pb2.action)

    def __repr__(self) -> str:
        """
        String representation of the Action object.
        """
        import rich.pretty

        return rich.pretty.pretty_repr(self)
