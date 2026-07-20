from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, Literal, Tuple

import httpx
import rich.repr
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from flyteidl2.common import identifier_pb2, list_pb2, phase_pb2
from flyteidl2.core import literals_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.workflow import run_definition_pb2, run_service_pb2
from google.protobuf import duration_pb2

from flyte._initialize import ensure_client, get_client, get_init_config
from flyte._logging import logger
from flyte.models import ActionPhase
from flyte.syncify import syncify

from . import Action, ActionDetails, ActionInputs, ActionOutputs
from ._action import _action_details_rich_repr, _action_rich_repr
from ._common import TimeFilter, ToJSONMixin, filtering, sorting, time_filtering


@dataclass
class Run(ToJSONMixin):
    """
    A class representing a run of a task. It is used to manage the run of a task and its state on the remote
    Union API.
    """

    pb2: run_definition_pb2.Run
    action: Action = field(init=False)
    _details: RunDetails | None = None
    _preserve_original_types: bool = False

    def __post_init__(self):
        """
        Initialize the Run object with the given run definition.
        """
        if not self.pb2.HasField("action"):
            raise RuntimeError("Run does not have an action")
        self.action = Action(self.pb2.action)
        self._debug_url = None

    @syncify
    @classmethod
    async def listall(
        cls,
        in_phase: Tuple[ActionPhase | str, ...] | None = None,
        task_name: str | None = None,
        task_version: str | None = None,
        created_by_subject: str | None = None,
        sort_by: Tuple[str, Literal["asc", "desc"]] | None = None,
        limit: int = 100,
        project: str | None = None,
        domain: str | None = None,
        created_at: TimeFilter | None = None,
        updated_at: TimeFilter | None = None,
        with_labels: dict[str, str] | None = None,
        with_label_keys: list[str] | None = None,
        paused_actions_only: bool = False,
    ) -> AsyncIterator[Run]:
        """
        Get all runs for the current project and domain.

        :param in_phase: Filter runs by one or more phases.
        :param task_name: Filter runs by task name.
        :param task_version: Filter runs by task version.
        :param created_by_subject: Filter runs by the subject that created them. (this is not username, but the subject)
        :param sort_by: The sorting criteria for the Run list, in the format (field, order).
        :param limit: The maximum number of runs to return.
        :param project: The project to list runs for. Defaults to the globally configured project.
        :param domain: The domain to list runs for. Defaults to the globally configured domain.
        :param created_at: Filter runs by creation time range.
        :param updated_at: Filter runs by last-update time range.
        :param with_labels: Filter runs whose labels include all of these key=value pairs (AND semantics).
        :param with_label_keys: Filter runs that have all of these label keys present (existence check).
        :param paused_actions_only: If True, only return runs that have at least one paused action
            (i.e. runs waiting on a human in the loop).
        :return: An iterator of runs.
        """
        ensure_client()
        token = None
        sort_pb2 = sorting(sort_by)
        filters = []
        if in_phase:
            phases = [
                str(p.to_protobuf_value())
                if isinstance(p, ActionPhase)
                else str(phase_pb2.ActionPhase.Value(f"ACTION_PHASE_{p.upper()}"))
                for p in in_phase
            ]
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

        if task_name:
            filters.append(
                list_pb2.Filter(
                    function=list_pb2.Filter.Function.EQUAL,
                    field="task_name",
                    values=[task_name],
                ),
            )
        if task_version:
            filters.append(
                list_pb2.Filter(
                    function=list_pb2.Filter.Function.EQUAL,
                    field="task_version",
                    values=[task_version],
                ),
            )

        filters = filtering(created_by_subject, *filters)

        if created_at:
            filters.extend(time_filtering("created_at", created_at))
        if updated_at:
            filters.extend(time_filtering("updated_at", updated_at))

        # Label filters are expressed through the generic filter mechanism using a ``labels.<key>``
        # field convention: ``--with-label k=v`` becomes an EQUAL match, ``--with-label-key k`` an
        # EXISTS match. Multiple label filters are ANDed together with the other filters.
        if with_labels:
            for k, v in with_labels.items():
                filters.append(
                    list_pb2.Filter(
                        function=list_pb2.Filter.Function.EQUAL,
                        field=f"labels.{k}",
                        values=[v],
                    ),
                )
        if with_label_keys:
            for k in with_label_keys:
                filters.append(
                    list_pb2.Filter(
                        function=list_pb2.Filter.Function.EXISTS,
                        field=f"labels.{k}",
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
            resp = await get_client().run_service.list_runs(
                run_service_pb2.ListRunsRequest(
                    request=req,
                    org=cfg.org,
                    project_id=identifier_pb2.ProjectIdentifier(
                        organization=cfg.org,
                        domain=domain or cfg.domain,
                        name=project or cfg.project,
                    ),
                    paused_actions_only=paused_actions_only,
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
    def raw_phase(self) -> phase_pb2.ActionPhase:
        """
        Get the raw phase of the run.
        """
        return self.action.raw_phase

    @syncify
    async def wait(self, quiet: bool = False, wait_for: Literal["terminal", "running"] = "terminal") -> None:
        """
        Wait for the run to complete, displaying a rich progress panel with status transitions,
        time elapsed, and error details in case of failure.

        This method updates the Run's internal state, ensuring that properties like
        `run.action.phase` reflect the final state after waiting completes.
        """
        await self.action.wait(quiet=quiet, wait_for=wait_for)
        # Update the Run's pb2.action to keep it in sync after waiting
        self.pb2.action.CopyFrom(self.action.pb2)

    async def watch(self, cache_data_on_done: bool = False) -> AsyncGenerator[ActionDetails, None]:
        """
        Watch the run for updates, updating the internal Run state with latest details.

        This method updates the Run's action state, ensuring that properties like
        `run.action.phase` reflect the current state after watching.
        """
        async for ad in self.action.watch(cache_data_on_done=cache_data_on_done):
            # The action's pb2 is already updated by Action.watch()
            # Update the Run's pb2.action to keep it in sync
            self.pb2.action.CopyFrom(self.action.pb2)
            yield ad

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
    async def get_logs(
        self,
        attempt: int | None = None,
        filter_system: bool = False,
        show_ts: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Get logs for the run as an iterator of strings.

        Can be called synchronously (returns `Iterator[str]`) or asynchronously
        via `.aio()` (returns `AsyncIterator[str]`).

        :param attempt: The attempt number to retrieve logs for (defaults to latest attempt).
        :param filter_system: If True, filter out system-generated log lines.
        :param show_ts: If True, prefix each line with an ISO-8601 timestamp.
        """
        async for line in self.action.get_logs.aio(attempt, filter_system=filter_system, show_ts=show_ts):
            yield line

    @syncify
    async def get_report(self, attempt: int | None = None, expires_in: timedelta = timedelta(hours=1)) -> str:
        """
        Get the HTML report associated with this run's root action.

        This first requests a signed download link from the data proxy for the report artifact,
        then downloads the report from that URL and returns its contents as an HTML string.

        :param attempt: The attempt number to fetch the report for. Defaults to the latest attempt.
        :param expires_in: How long the signed download link should remain valid. Defaults to 1 hour.
        :return: The report contents as an HTML string.
        """
        ensure_client()

        if attempt is None:
            details = await self.action.details()
            attempt = details.attempts

        expires_in_pb = duration_pb2.Duration()
        expires_in_pb.FromTimedelta(expires_in)
        resp = await get_client().dataproxy_service.create_download_link(
            dataproxy_service_pb2.CreateDownloadLinkRequest(
                artifact_type=dataproxy_service_pb2.ARTIFACT_TYPE_REPORT,
                action_attempt_id=identifier_pb2.ActionAttemptIdentifier(
                    action_id=self.action.action_id,
                    attempt=attempt,
                ),
                expires_in=expires_in_pb,
            )
        )

        signed_urls = list(resp.pre_signed_urls.signed_url)
        if not signed_urls:
            raise RuntimeError(f"No report is available for run '{self.name}' (attempt {attempt}).")

        async with httpx.AsyncClient() as client:
            download = await client.get(signed_urls[0])
            download.raise_for_status()
            return download.text

    @syncify
    async def details(self) -> RunDetails:
        """
        Get the details of the run. This is a placeholder for getting the run details.
        """
        if self._details is None or not self._details.done():
            self._details = await RunDetails.get_details.aio(self.pb2.action.id.run)
        self._details._preserve_original_types = self._preserve_original_types
        self._details.action_details._preserve_original_types = self._preserve_original_types
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

    @syncify
    async def output_literals(self) -> Dict[str, literals_pb2.Literal]:
        """Raw output literals of the run's action, without reconstructing types.

        See :meth:`ActionDetails.output_literals`.
        """
        details = await self.details.aio()
        return await details.output_literals()

    @syncify
    async def input_literals(self) -> Dict[str, literals_pb2.Literal]:
        """Raw input literals of the run's action, without reconstructing types.

        See :meth:`ActionDetails.input_literals`.
        """
        details = await self.details.aio()
        return await details.input_literals()

    @syncify
    async def typed_outputs(
        self,
        types: Dict[str, type],
        deserializers: Dict[type, Callable[[Any], Any]] | None = None,
    ) -> Dict[str, Any]:
        """Re-hydrate the run's requested outputs into caller-supplied types.

        See :meth:`ActionDetails.typed_outputs`.
        """
        details = await self.details.aio()
        return await details.typed_outputs(types, deserializers)

    @syncify
    async def typed_inputs(
        self,
        types: Dict[str, type],
        deserializers: Dict[type, Callable[[Any], Any]] | None = None,
    ) -> Dict[str, Any]:
        """Re-hydrate the run's requested inputs into caller-supplied types.

        See :meth:`ActionDetails.typed_inputs`.
        """
        details = await self.details.aio()
        return await details.typed_inputs(types, deserializers)

    @property
    def url(self) -> str:
        """
        Get the URL of the run.
        """
        client = get_client()
        return client.console.run_url(
            project=self.pb2.action.id.run.project,
            domain=self.pb2.action.id.run.domain,
            run_name=self.name,
        )

    @syncify
    async def get_debug_url(self) -> str:
        """
        Get the debug URL of the run. Returns `None` if the VS Code
        Debugger log entry is not yet available in the action details.
        """
        if self._debug_url is not None:
            return self._debug_url
        from flyte._debug.client import watch_for_vscode_url

        self._debug_url = await watch_for_vscode_url(self)
        return self._debug_url

    @syncify
    async def abort(self, reason: str = "Manually aborted from the SDK api."):
        """
        Aborts / Terminates the run.
        """
        try:
            await get_client().run_service.abort_run(
                run_service_pb2.AbortRunRequest(
                    run_id=self.pb2.action.id.run,
                    reason=reason,
                )
            )
        except ConnectError as e:
            if e.code == Code.NOT_FOUND:
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
        yield "labels", dict(self.pb2.labels) if self.pb2.labels else {}
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
    _preserve_original_types: bool = False

    def __post_init__(self):
        """
        Initialize the RunDetails object with the given run definition.
        """
        self.action_details = ActionDetails(self.pb2.action, _preserve_original_types=self._preserve_original_types)

    @syncify
    @classmethod
    async def get_details(cls, run_id: identifier_pb2.RunIdentifier) -> RunDetails:
        """
        Get the details of the run. This is a placeholder for getting the run details.
        """
        ensure_client()
        resp = await get_client().run_service.get_run_details(
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

    async def output_literals(self) -> Dict[str, literals_pb2.Literal]:
        """Raw output literals without reconstructing types. See :meth:`ActionDetails.output_literals`."""
        return await self.action_details.output_literals()

    async def input_literals(self) -> Dict[str, literals_pb2.Literal]:
        """Raw input literals without reconstructing types. See :meth:`ActionDetails.input_literals`."""
        return await self.action_details.input_literals()

    async def typed_outputs(
        self,
        types: Dict[str, type],
        deserializers: Dict[type, Callable[[Any], Any]] | None = None,
    ) -> Dict[str, Any]:
        """Re-hydrate requested outputs into caller-supplied types. See :meth:`ActionDetails.typed_outputs`."""
        return await self.action_details.typed_outputs(types, deserializers)

    async def typed_inputs(
        self,
        types: Dict[str, type],
        deserializers: Dict[type, Callable[[Any], Any]] | None = None,
    ) -> Dict[str, Any]:
        """Re-hydrate requested inputs into caller-supplied types. See :meth:`ActionDetails.typed_inputs`."""
        return await self.action_details.typed_inputs(types, deserializers)

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
