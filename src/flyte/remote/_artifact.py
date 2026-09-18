from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, AsyncIterator, Literal, Mapping, Sequence, Type

import rich.repr
from flyteidl2.artifact import artifact_pb2, artifact_service_pb2
from flyteidl2.common import identifier_pb2, list_pb2
from flyteidl2.core import artifact_id_pb2, literals_pb2

from flyte._initialize import ensure_client, get_client, get_init_config
from flyte.artifacts._card import Card as CoreCard
from flyte.artifacts._metadata import KIND_KEY, Kind, Metadata, resolve_attrs
from flyte.artifacts._partitions import (
    GRANULARITIES,
    Granularity,
    TimePartition,
    granularity_from_pb2,
    granularity_to_pb2,
    is_time_value,
    looks_like_time,
    parse_time,
    partitions_from_pb2,
    partitions_to_pb2,
    time_value,
    to_rfc3339,
)
from flyte.artifacts._wrapper import ArtifactWrapper, ensure_artifactable
from flyte.remote._common import ToJSONMixin
from flyte.syncify import syncify

#: Filter-field prefix the artifact service uses for one key of an artifact's
#: attrs (its `user_metadata` map), mirroring how runs key label filters off
#: "labels.".
METADATA_FIELD_PREFIX = "user_metadata."

#: Filter-field prefix for one string partition key: "partition.region".
PARTITION_FIELD_PREFIX = "partition."

#: Filter field for the time partition; values are RFC3339 timestamps.
TIME_PARTITION_FIELD = "time_partition"

#: Filter field selecting versions flagged with a partition schema mismatch.
SCHEMA_MISMATCH_FIELD = "schema_mismatch"

_LIST_PAGE_SIZE = 100
_PARTITION_VALUES_LIMIT = 1000


def _name_pb2(cfg: Any, name: str, project: str | None, domain: str | None) -> artifact_pb2.ArtifactName:
    return artifact_pb2.ArtifactName(
        org=cfg.org or "",
        project=project or cfg.project or "",
        domain=domain or cfg.domain or "",
        name=name,
    )


def _project_pb2(cfg: Any, project: str | None, domain: str | None) -> identifier_pb2.ProjectIdentifier:
    return identifier_pb2.ProjectIdentifier(
        organization=cfg.org or "",
        domain=domain or cfg.domain or "",
        name=project or cfg.project or "",
    )


def partition_filters(partitions: Mapping[str, Any]) -> list[list_pb2.Filter]:
    """
    The Filter messages for a partition selection, as `get`, `listall` and
    `partition_values` send them.

    A time value (`date`, `datetime`, `TimePartition`, or a string that reads as an
    ISO date, ISO hour or RFC3339 timestamp) becomes an EQUAL on the "time_partition"
    field with the floored RFC3339 value. A 2-tuple with time bounds becomes an
    inclusive range on that field. A list on a string key becomes VALUE_IN; the
    filter language has no OR on the time partition, so a list of time values is
    rejected. Any other value is an EQUAL on "partition.<key>"; `None` is rejected.
    """
    filters: list[list_pb2.Filter] = []
    for key, value in partitions.items():
        if value is None:
            raise ValueError(f"Partition {key!r} is None; give a value to select by")
        if isinstance(value, tuple) and len(value) == 2 and any(_is_time_bound(v) for v in value):
            lo, hi = value
            if not (_is_time_bound(lo) and _is_time_bound(hi)):
                raise ValueError(
                    f"Partition {key!r}: a 2-tuple is a time range and both bounds must be dates, datetimes "
                    f"or ISO strings; got ({lo!r}, {hi!r}). For several string values use a list."
                )
            filters.append(
                list_pb2.Filter(
                    function=list_pb2.Filter.GREATER_THAN_OR_EQUAL,
                    field=TIME_PARTITION_FIELD,
                    values=[to_rfc3339(parse_time(lo))],
                )
            )
            filters.append(
                list_pb2.Filter(
                    function=list_pb2.Filter.LESS_THAN_OR_EQUAL,
                    field=TIME_PARTITION_FIELD,
                    values=[to_rfc3339(parse_time(hi))],
                )
            )
        elif _is_time_bound(value):
            floored, _ = time_value(_as_time_value(value))
            filters.append(
                list_pb2.Filter(
                    function=list_pb2.Filter.EQUAL, field=TIME_PARTITION_FIELD, values=[to_rfc3339(floored)]
                )
            )
        elif isinstance(value, (list, tuple, set, frozenset)):
            if any(_is_time_bound(v) for v in value):
                raise ValueError(
                    f"Partition {key!r}: a list of time values cannot be selected in one query; "
                    "use a range (lo, hi) or call once per value"
                )
            values = [str(v) for v in value]
            if values:
                filters.append(
                    list_pb2.Filter(
                        function=list_pb2.Filter.VALUE_IN, field=f"{PARTITION_FIELD_PREFIX}{key}", values=values
                    )
                )
        else:
            filters.append(
                list_pb2.Filter(
                    function=list_pb2.Filter.EQUAL, field=f"{PARTITION_FIELD_PREFIX}{key}", values=[str(value)]
                )
            )
    return filters


def _is_time_bound(value: Any) -> bool:
    return is_time_value(value) or (isinstance(value, str) and looks_like_time(value))


def _as_time_value(value: Any) -> Any:
    """A time-like string becomes the value the CLI would make of it: a `date` for a bare
    ISO date (daily partition), a `datetime` otherwise (hourly)."""
    if isinstance(value, str):
        parsed = parse_time(value)
        return parsed.date() if len(value.strip()) == 10 else parsed
    return value


@dataclass(frozen=True)
class PartitionSchema:
    """
    The partition keys fixed for an artifact name: at most one time key with its
    granularity, plus string keys in declaration order. `declared` says whether
    the keys were fixed by an explicit `Artifact.declare` (True) or by the first
    version (False).
    """

    time_key: str | None
    granularity: Granularity | None
    keys: tuple[str, ...]
    declared: bool = False

    @classmethod
    def from_pb2(cls, schema: artifact_pb2.ArtifactPartitionSchema, declared: bool = False) -> PartitionSchema:
        if schema.HasField("time_partition"):
            tp = schema.time_partition
            return cls(
                time_key=tp.key,
                granularity=granularity_from_pb2(tp.granularity),
                keys=tuple(schema.partition_keys),
                declared=declared,
            )
        return cls(time_key=None, granularity=None, keys=tuple(schema.partition_keys), declared=declared)

    @property
    def all_keys(self) -> tuple[str, ...]:
        """Every partition key, time key first."""
        return ((self.time_key,) if self.time_key else ()) + self.keys

    def to_pb2(self) -> artifact_pb2.ArtifactPartitionSchema:
        schema = artifact_pb2.ArtifactPartitionSchema(partition_keys=list(self.keys))
        if self.time_key:
            schema.time_partition.CopyFrom(
                artifact_pb2.TimePartitionKey(
                    key=self.time_key, granularity=granularity_to_pb2(self.granularity or "day")
                )
            )
        return schema


def schema_from_spec(partitions: Mapping[str, Any] | None) -> PartitionSchema:
    """
    A schema from a declaration mapping: `date` (or "day") is a daily time key,
    `datetime` (or "hour") hourly, "week" / "month" the coarser ones, and `str`
    or `int` a string key.
    """
    time_key: str | None = None
    granularity: Granularity | None = None
    keys: list[str] = []
    for key, spec in (partitions or {}).items():
        if not key:
            raise ValueError("Partition keys must be non-empty strings")
        if spec is datetime:
            g: Granularity | None = "hour"
        elif spec is date:
            g = "day"
        elif isinstance(spec, str) and spec in GRANULARITIES:
            g = spec  # type: ignore[assignment]
        elif spec is str or spec is int or (isinstance(spec, str) and spec in ("str", "int")):
            g = None
        else:
            raise ValueError(
                f"Partition {key!r}: declare a time key with date, datetime, 'hour', 'day', 'week' or 'month', "
                f"or a string key with str; got {spec!r}"
            )
        if g is None:
            keys.append(key)
        elif time_key is not None:
            raise ValueError(
                f"An artifact can carry at most one time partition; both {time_key!r} and {key!r} are time keys"
            )
        else:
            time_key, granularity = key, g
    return PartitionSchema(time_key=time_key, granularity=granularity, keys=tuple(keys))


def _card_to_pb2(card: CoreCard | None) -> artifact_id_pb2.ArtifactCard | None:
    if card is None:
        return None
    return artifact_id_pb2.ArtifactCard(uri=card.uri, format=card.format, type=card.card_type)


def _current_task_source() -> artifact_pb2.ArtifactSource | None:
    """Provenance for the currently running task action, or None outside a task.

    Scope fields (org/project/domain) are left for the server to inherit from
    the artifact's own scope; an artifact can only reference an action in its
    own org/project/domain.
    """
    from flyte._context import internal_ctx

    tctx = internal_ctx().data.task_context
    if tctx is None:
        return None
    action = tctx.action
    return artifact_pb2.ArtifactSource(
        task_action=artifact_pb2.TaskActionSource(
            action=identifier_pb2.ActionIdentifier(
                run=identifier_pb2.RunIdentifier(name=action.run_name or ""),
                name=action.name,
            ),
            attempt=tctx.attempt_number,
        ),
    )


@dataclass
class Artifact(ToJSONMixin):
    """
    A published artifact in the Flyte artifact service: a typed value (stored as
    a Flyte literal) addressed by org/project/domain/name/version.
    """

    pb2: artifact_pb2.Artifact

    @property
    def name(self) -> str:
        return self.pb2.artifact_id.name.name

    @property
    def version(self) -> str:
        return self.pb2.artifact_id.version

    @property
    def tracker(self) -> str:
        """The artifact's id as a tracking string: org/project/domain/name@version."""
        n = self.pb2.artifact_id.name
        return f"{n.org}/{n.project}/{n.domain}/{n.name}@{self.version}"

    @property
    def artifact_version_id(self) -> artifact_id_pb2.ArtifactVersionId:
        """The artifact's typed identity, as stamped onto values (core.Literal.artifact_id)."""
        n = self.pb2.artifact_id.name
        return artifact_id_pb2.ArtifactVersionId(
            key=artifact_id_pb2.ArtifactKey(org=n.org, project=n.project, domain=n.domain, name=n.name),
            version=self.version,
        )

    @property
    def url(self) -> str:
        """
        Get the console URL for viewing this artifact.
        """
        n = self.pb2.artifact_id.name
        return get_client().console.artifact_url(project=n.project, domain=n.domain, name=n.name)

    @property
    def source(self) -> str:
        """Best-effort display string for the artifact's provenance (ArtifactSource)."""
        src = self.pb2.spec.source
        which = src.WhichOneof("source")
        if which == "task_action":
            ta = src.task_action
            return f"run {ta.action.run.name}/{ta.action.name} (attempt {ta.attempt})"
        if which == "external_ref":
            return src.external_ref
        return ""

    @property
    def source_run_url(self) -> str | None:
        """Console URL of the run that produced this artifact, or None if no task produced it."""
        src = self.pb2.spec.source
        if src.WhichOneof("source") != "task_action":
            return None
        n = self.pb2.artifact_id.name
        return get_client().console.run_url(
            project=n.project, domain=n.domain, run_name=src.task_action.action.run.name
        )

    @property
    def source_action_url(self) -> str | None:
        """Console URL of the action that produced this artifact, or None if no task produced it."""
        src = self.pb2.spec.source
        if src.WhichOneof("source") != "task_action":
            return None
        n, action = self.pb2.artifact_id.name, src.task_action.action
        return get_client().console.action_url(
            project=n.project, domain=n.domain, run_name=action.run.name, action_name=action.name
        )

    @property
    def partitions(self) -> dict[str, Any]:
        """
        The version's partition values by key: a `date` for a daily (or coarser)
        time partition, a `datetime` in UTC for an hourly one, strings otherwise.
        Empty for an unpartitioned artifact.
        """
        spec = self.pb2.spec
        return partitions_from_pb2(
            spec.partitions if spec.HasField("partitions") else None,
            spec.time_partition if spec.HasField("time_partition") else None,
        )

    @property
    def time_partition(self) -> artifact_id_pb2.TimePartition | None:
        """The stored time partition message (key, floored value, granularity), or None."""
        spec = self.pb2.spec
        return spec.time_partition if spec.HasField("time_partition") else None

    @property
    def schema_mismatch(self) -> str | None:
        """
        Why this version is not addressable by partition, or None when its keys
        match the artifact's schema. A flagged version is stored and visible in
        version listings but is skipped by partition filters, latest-per-partition
        listings, partition value listings and partition triggers.
        """
        if not self.pb2.HasField("partition_schema_mismatch"):
            return None
        return self.pb2.partition_schema_mismatch.message or "partition keys do not match the artifact's schema"

    @property
    def kind(self) -> Kind:
        """
        What this artifact is: "model", "data", or "generic".

        Read from the reserved `flyte.io/kind` attr. Artifacts published before that
        key existed fall back to the card's type, which was the closest thing to a
        discriminator at the time -- so an older model with a card still classifies.
        Anything with neither marker is "generic": callers get a usable answer rather
        than None, since "unlabelled" and "not a model" are the same thing here.
        """
        declared = self.pb2.spec.info.user_metadata.get(KIND_KEY)
        if declared in ("model", "data", "generic"):
            return declared  # type: ignore[return-value]

        # Card type is presentational, but before the reserved key it was the only
        # signal a publisher could leave. Note it is optional even for models:
        # flyte.prefetch.hf_model only attaches a card when the repo had a README.
        card_type = self.pb2.spec.info.card.type
        if card_type in ("model", "data", "generic"):
            return card_type  # type: ignore[return-value]

        return "generic"

    @property
    def created_by(self) -> str:
        """Best-effort display string for the creating identity (EnrichedIdentity)."""
        identity = self.pb2.created_by
        which = identity.WhichOneof("principal")
        if which == "user":
            user = identity.user
            return user.spec.email or user.id.subject
        if which == "application":
            app = identity.application
            return app.spec.name or app.id.subject
        return ""

    def __rich_repr__(self) -> rich.repr.Result:
        """
        Rich representation of the Artifact object for pretty printing.
        """
        yield "project", self.pb2.artifact_id.name.project or "-"
        yield "domain", self.pb2.artifact_id.name.domain or "-"
        yield "name", self.name
        yield "version", self.version
        yield "kind", self.kind
        parts = self.partitions
        yield "partitions", ", ".join(f"{k}={_fmt_partition(v)}" for k, v in parts.items()) if parts else "-"
        if self.schema_mismatch:
            yield "schema_mismatch", self.schema_mismatch
        yield "description", self.pb2.spec.info.description or "-"
        yield "created_at", self.pb2.created_at.ToDatetime().isoformat()
        yield "created_by", self.created_by or "-"
        yield "source", self.source or "-"

    async def to_python(self, python_type: Type | None = None) -> Any:
        """
        Materialize the artifact's stored literal back into a python value.

        Args:
            python_type: Expected python type; guessed from the stored Flyte type when omitted.
        """
        from flyte.types import TypeEngine

        pt = python_type or TypeEngine.guess_python_type(self.pb2.spec.type)
        return await TypeEngine.to_python_value(self.pb2.spec.value, pt)

    async def coerce_to_literal(self, python_type: Type | None = None) -> literals_pb2.Literal:
        """
        Coerce the artifact's stored literal to the shape `python_type` expects.

        Round-trips the stored literal through the type engine — `to_python_value`
        against the declared type, then `to_literal` — so every compatibility rule
        (Optional/union wrapping, coercions, blob dimensionality) is the engine's, not
        re-derived here, and a mismatch fails now with the transformer's error rather
        than inside the task. Cheap for offloaded values: File/Dir/DataFrame literals
        reconstruct from their uri without downloading. The artifact's identity is
        stamped on the result so provenance travels with the coerced literal.

        Args:
            python_type: Declared type to coerce to. When omitted the stored literal
                is returned as-is (it already carries the service-stamped identity).

        Raises:
            TypeTransformerFailedError: when the stored value cannot bind to
                `python_type`.
        """
        from flyte.types import TypeEngine

        stored = self.pb2.spec.value
        if python_type is None:
            return stored
        pv = await TypeEngine.to_python_value(stored, python_type)
        lit = await TypeEngine.to_literal(pv, python_type, TypeEngine.to_literal_type(python_type))
        # The round-trip produces a fresh literal; carry the service's identity over. We copy
        # the stamp, we never compute it.
        if stored.HasField("artifact_id"):
            lit.artifact_id.CopyFrom(stored.artifact_id)
        else:
            lit.artifact_id.CopyFrom(self.artifact_version_id)
        return lit

    @syncify
    @classmethod
    async def create(
        cls,
        value: Any,
        *,
        name: str | None = None,
        version: str | None = None,
        description: str | None = None,
        attrs: Mapping[str, str] | None = None,
        kind: Kind | None = None,
        card: CoreCard | None = None,
        python_type: Type | None = None,
        project: str | None = None,
        domain: str | None = None,
        external_ref: str | None = None,
        partitions: Mapping[str, Any] | None = None,
    ) -> Artifact:
        """
        Publish an artifact from the local machine.

        The value must be an offloaded asset — a flyte.io File, Dir, or DataFrame.
        It is converted with the type engine (local data is uploaded to blob
        storage first) and stored in the artifact service as a typed literal.

        Args:
            value: The File, Dir, or DataFrame to publish. May be wrapped
                with `flyte.artifacts.new(...)`; wrapper metadata seeds name/version/
                description/attrs/card, and explicit keyword arguments override it.
            name: The artifact name; required when value carries no metadata.
            version: The version to publish. Defaults to the metadata version or a random one.
            description: Optional human readable description.
            attrs: Optional free-form key/value metadata.
            kind: What the artifact is ("model", "data", "generic"). Recorded under the
                reserved `flyte.io/kind` attr and read back via `Artifact.kind`. Distinct
                from a card's type, which describes how the card renders.
            card: Optional `flyte.artifacts.Card` to attach.
            python_type: Type used for literal conversion; defaults to `type(value)`.
            project: Project to publish into; defaults to the init configuration.
            domain: Domain to publish into; defaults to the init configuration.
            external_ref: Optional opaque reference into an external system (a URI,
                model id, dataset id, ...) recorded as the artifact's source. When omitted
                and called from inside a running task, the producing task action is
                recorded automatically instead.
            partitions: Partition values keyed by partition name: a `date` is a daily
                time partition, a `datetime` an hourly one, `flyte.artifacts.TimePartition`
                names the granularity, anything else is a string partition. The keys must
                match the artifact's schema (fixed by its first version); otherwise the
                version is stored but flagged and not addressable by partition.

        Returns:
            The published Artifact.
        """
        from flyte.types import TypeEngine

        ensure_client()
        cfg = get_init_config()

        obj = value
        if type(value) is ArtifactWrapper:
            md: Metadata = value.get_flyte_metadata()
            obj = value._obj
            name = name or md.name
            version = version or md.version
            description = description if description is not None else md.description
            # resolve_attrs folds the wrapper's kind= into attrs; reading md.attrs
            # directly would drop it for values wrapped by flyte.artifacts.new().
            attrs = attrs if attrs is not None else resolve_attrs(md)
            card = card if card is not None else md.card
            partitions = partitions if partitions is not None else md.partitions
        if kind is not None:
            # Same precedence as Metadata: an explicit reserved key already in attrs
            # is deliberate and wins.
            attrs = {**(attrs or {})}
            attrs.setdefault(KIND_KEY, kind)
        if not name:
            raise ValueError(
                "An artifact name is required: pass name= or publish a value wrapped by flyte.artifacts.new()"
            )
        ensure_artifactable(obj)

        pt = python_type or type(obj)
        lt = TypeEngine.to_literal_type(pt)
        lit = await TypeEngine.to_literal(obj, pt, lt)

        if external_ref is not None:
            source: artifact_pb2.ArtifactSource | None = artifact_pb2.ArtifactSource(external_ref=external_ref)
        else:
            source = _current_task_source()
        string_partitions, time_partition = partitions_to_pb2(partitions)

        request = artifact_service_pb2.CreateArtifactRequest(
            artifact_id=artifact_pb2.ArtifactIdentifier(
                name=_name_pb2(cfg, name, project, domain),
                version=version or uuid.uuid4().hex,
            ),
            spec=artifact_pb2.ArtifactSpec(
                value=lit,
                type=lt,
                info=artifact_id_pb2.ArtifactInfo(
                    description=description or "",
                    user_metadata=dict(attrs) if attrs else None,
                    card=_card_to_pb2(card),
                ),
                source=source,
                partitions=string_partitions,
                time_partition=time_partition,
            ),
        )
        resp = await get_client().artifact_service.create_artifact(request)
        return cls(pb2=resp.artifact)

    @syncify
    @classmethod
    async def get(
        cls,
        name: str,
        version: str | Literal["latest"] = "latest",
        *,
        project: str | None = None,
        domain: str | None = None,
        partitions: Mapping[str, Any] | None = None,
        **partition_kwargs: Any,
    ) -> Artifact:
        """
        Get an artifact by its name and version, or the latest version of one partition.

        ```python
        Artifact.get("raw_events")                               # latest version overall
        Artifact.get("raw_events", version="1.0")                # a pinned version
        Artifact.get("raw_events", date=date(2026, 9, 17), region="us")  # latest of that partition
        ```

        Args:
            name: The name of the artifact.
            version: The version of the artifact; "latest" returns the most recently created version.
            project: Project to look in; defaults to the init configuration.
            domain: Domain to look in; defaults to the init configuration.
            partitions: Partition values selecting one partition, one per key of the
                artifact's schema, for keys whose names collide with this method's own
                parameters (`name`, `version`, `project`, `domain`).
            **partition_kwargs: The same, as keyword arguments; these win over `partitions`.
                A `date` matches a daily time partition, a `datetime` an hourly one, a
                string that reads as an ISO date or timestamp a time partition too (the way
                the CLI reads it), any other string a string partition. A range or a list
                selects more than one partition and is rejected here; use `listall`.
                Cannot be combined with an explicit version. Versions flagged with a
                schema mismatch never match.

        Raises:
            ValueError: when both a version and partitions are given, a value selects more
                than one partition, or no version exists for the partition.
        """
        ensure_client()
        cfg = get_init_config()

        partitions = {**(partitions or {}), **partition_kwargs}
        if partitions:
            for key, value in partitions.items():
                if isinstance(value, (list, tuple, set, frozenset)):
                    raise ValueError(
                        f"Artifact.get selects one partition, but {key!r} is a range or list ({value!r}); "
                        "use Artifact.listall(name, latest_per_partition=True, ...) for several"
                    )
            if version != "latest":
                raise ValueError(
                    "Artifact.get takes either a version or partition values, not both: a version already "
                    "names one artifact version"
                )
            request = artifact_service_pb2.ListArtifactsRequest(
                request=list_pb2.ListRequest(limit=1, filters=partition_filters(partitions)),
                project_id=_project_pb2(cfg, project, domain),
                name=name,
            )
            resp = await get_client().artifact_service.list_artifacts(request)
            if not resp.artifacts:
                selection = ", ".join(f"{k}={_fmt_partition(v)}" for k, v in partitions.items())
                raise ValueError(f"No version of artifact {name!r} exists for partition {selection}")
            return cls(pb2=resp.artifacts[0])

        request_get = artifact_service_pb2.GetArtifactRequest(name=_name_pb2(cfg, name, project, domain))
        if version != "latest":
            request_get.version = version
        resp_get = await get_client().artifact_service.get_artifact(request_get)
        return cls(pb2=resp_get.artifact)

    @syncify
    @classmethod
    async def listall(
        cls,
        name: str | None = None,
        created_after: datetime | None = None,
        limit: int = -1,
        *,
        project: str | None = None,
        domain: str | None = None,
        source_run: str | None = None,
        source_action: str | None = None,
        source_external_ref: str | None = None,
        kind: Kind | None = None,
        attrs: Mapping[str, str | Sequence[str]] | None = None,
        latest_per_partition: bool = False,
        partitions: Mapping[str, Any] | None = None,
        **partition_kwargs: Any,
    ) -> AsyncIterator[Artifact]:
        """
        List artifacts, newest first.

        ```python
        Artifact.listall("raw_events", date=(date(2026, 8, 1), date(2026, 8, 31)), latest_per_partition=True)
        Artifact.listall("raw_events", region=["us", "eu"])
        ```

        Args:
            name: Exact artifact name; when set, all versions of that artifact are listed.
            created_after: Filter artifacts created after this datetime.
            limit: The maximum number of artifacts to return. -1 for no limit.
            project: Project to list in; defaults to the init configuration.
            domain: Domain to list in; defaults to the init configuration.
            source_run: Only artifacts produced by this run.
            source_action: Only artifacts produced by this action; usually combined with source_run.
            source_external_ref: Only artifacts imported from this external reference.
            kind: Only artifacts of this kind, e.g. "model". Shorthand for filtering
                on the reserved kind attr.
            attrs: Only artifacts whose attrs match. A value may be a single string or
                a sequence, in which case any of them matches. Separate keys must all
                match.
            latest_per_partition: Return only the newest version of each distinct
                partition, so a range over the time partition yields one version per
                partition. Requires a name. Flagged versions are excluded.
            partitions: Partition selection by key, the same as the keyword form. A
                `date`/`datetime` selects one time partition; a 2-tuple `(start, end)`
                selects an inclusive range of the time partition (dates, datetimes or
                ISO strings); a list on a string key matches any of the values; any
                other value matches exactly.
            **partition_kwargs: Partition selection as keywords, e.g. `region="us"`.

        Returns:
            An async iterator of artifacts.

        Filtering happens server-side, so it pages through matches rather than
        scanning everything client-side. It requires a control plane that supports
        `user_metadata` filters; older ones reject the request rather than silently
        returning unfiltered results.
        """
        ensure_client()
        cfg = get_init_config()

        filters = []
        for field, value in (
            ("source_run", source_run),
            ("source_action", source_action),
            ("source_external_ref", source_external_ref),
        ):
            if value is not None:
                filters.append(list_pb2.Filter(function=list_pb2.Filter.EQUAL, field=field, values=[value]))
        # Both land in the same attr namespace, so kind= is folded in as one more
        # predicate rather than a separate mechanism.
        attr_filters: dict[str, list[str]] = {}
        if kind is not None:
            attr_filters[KIND_KEY] = [kind]
        for attr_key, attr_value in (attrs or {}).items():
            attr_values = [attr_value] if isinstance(attr_value, str) else list(attr_value)
            if attr_values:
                attr_filters.setdefault(attr_key, []).extend(attr_values)
        for attr_key, attr_values in attr_filters.items():
            filters.append(
                list_pb2.Filter(
                    function=list_pb2.Filter.VALUE_IN,
                    field=f"{METADATA_FIELD_PREFIX}{attr_key}",
                    values=attr_values,
                )
            )
        if created_after is not None:
            ts = created_after if created_after.tzinfo else created_after.replace(tzinfo=timezone.utc)
            filters.append(
                list_pb2.Filter(
                    function=list_pb2.Filter.GREATER_THAN,
                    field="created_at",
                    values=[ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")],
                )
            )
        selection = {**(partitions or {}), **partition_kwargs}
        if selection:
            filters.extend(partition_filters(selection))
        if latest_per_partition and name is None:
            raise ValueError("latest_per_partition requires an artifact name")

        token = ""
        remaining = limit if limit >= 0 else None
        while remaining is None or remaining > 0:
            page_size = _LIST_PAGE_SIZE if remaining is None else min(_LIST_PAGE_SIZE, remaining)
            request = artifact_service_pb2.ListArtifactsRequest(
                request=list_pb2.ListRequest(limit=page_size, token=token, filters=filters),
                project_id=_project_pb2(cfg, project, domain),
                latest_per_partition=latest_per_partition,
            )
            if name is not None:
                request.name = name
            resp = await get_client().artifact_service.list_artifacts(request)
            for artifact in resp.artifacts:
                yield cls(pb2=artifact)
                if remaining is not None:
                    remaining -= 1
                    if remaining == 0:
                        return
            token = resp.token
            if not token:
                return

    @syncify
    @classmethod
    async def list_names(
        cls,
        search: str | None = None,
        limit: int = -1,
        *,
        project: str | None = None,
        domain: str | None = None,
    ) -> AsyncIterator[ArtifactGroup]:
        """
        List distinct artifact names, one entry per name carrying the latest
        version and the total version count, newest activity first.

        Args:
            search: Substring match on the artifact name.
            limit: The maximum number of names to return. -1 for no limit.
            project: Project to list in; defaults to the init configuration.
            domain: Domain to list in; defaults to the init configuration.

        Returns:
            An async iterator of artifact groups.
        """
        ensure_client()
        cfg = get_init_config()

        filters = []
        if search:
            filters.append(list_pb2.Filter(function=list_pb2.Filter.CONTAINS, field="name", values=[search]))

        token = ""
        remaining = limit if limit >= 0 else None
        while remaining is None or remaining > 0:
            page_size = _LIST_PAGE_SIZE if remaining is None else min(_LIST_PAGE_SIZE, remaining)
            request = artifact_service_pb2.ListArtifactNamesRequest(
                request=list_pb2.ListRequest(limit=page_size, token=token, filters=filters),
                project_id=identifier_pb2.ProjectIdentifier(
                    organization=cfg.org or "",
                    domain=domain or cfg.domain or "",
                    name=project or cfg.project or "",
                ),
            )
            resp = await get_client().artifact_service.list_artifact_names(request)
            for group in resp.groups:
                yield ArtifactGroup(pb2=group)
                if remaining is not None:
                    remaining -= 1
                    if remaining == 0:
                        return
            token = resp.token
            if not token:
                return

    @syncify
    @classmethod
    async def partition_values(
        cls,
        name: str,
        key: str,
        *,
        project: str | None = None,
        domain: str | None = None,
        limit: int = _PARTITION_VALUES_LIMIT,
        partitions: Mapping[str, Any] | None = None,
        **fixed: Any,
    ) -> list[Any]:
        """
        The distinct values one partition key has among an artifact's versions,
        sorted ascending, optionally scoped by the other keys.

        ```python
        Artifact.partition_values("raw_events", "region", date=date(2026, 9, 17))  # ["eu", "us"]
        Artifact.partition_values("raw_events", "date")                             # [date(...), ...]
        ```

        Args:
            name: The artifact name.
            key: The partition key to list: a string key, or the time key, whose values
                come back as `date` (daily or coarser) or `datetime` (hourly).
            project: Project to look in; defaults to the init configuration.
            domain: Domain to look in; defaults to the init configuration.
            limit: Maximum number of values.
            partitions: Partition selection scoping the versions, for keys whose names
                collide with this method's own parameters (`name`, `key`, `project`,
                `domain`, `limit`).
            **fixed: The same, as keyword arguments (these win); as in `listall`.
        """
        ensure_client()
        cfg = get_init_config()

        fixed = {**(partitions or {}), **fixed}
        schema = await cls.get_schema.aio(name, project=project, domain=domain)
        request = artifact_service_pb2.ListPartitionValuesRequest(
            request=list_pb2.ListRequest(limit=limit, filters=partition_filters(fixed) if fixed else None),
            project_id=_project_pb2(cfg, project, domain),
            name=name,
            key=key,
        )
        resp = await get_client().artifact_service.list_partition_values(request)
        if key == schema.time_key:
            if schema.granularity == "hour":
                return [parse_time(v) for v in resp.values]
            return [parse_time(v).date() for v in resp.values]
        return list(resp.values)

    @syncify
    @classmethod
    async def declare(
        cls,
        name: str,
        partitions: Mapping[str, Any],
        *,
        project: str | None = None,
        domain: str | None = None,
    ) -> PartitionSchema:
        """
        Fix an artifact name's partition keys ahead of any version.

        ```python
        Artifact.declare("raw_events", partitions={"date": date, "region": str})
        Artifact.declare("hourly_events", partitions={"hour": datetime, "region": str})
        Artifact.declare("monthly_report", partitions={"date": "month"})
        ```

        Succeeds when the name has no schema yet or an equal one; fails with
        FAILED_PRECONDITION when a different schema is already fixed. Changing the
        keys is a new artifact name.

        Args:
            name: The artifact name.
            partitions: Key to kind: `date` or "day" for a daily time key, `datetime` or
                "hour" for hourly, "week" / "month" for coarser ones, `str` (or `int`) for
                a string key. At most one time key.
            project: Project; defaults to the init configuration.
            domain: Domain; defaults to the init configuration.
        """
        ensure_client()
        cfg = get_init_config()

        schema = schema_from_spec(partitions)
        request = artifact_service_pb2.DeclareArtifactRequest(
            name=_name_pb2(cfg, name, project, domain),
            partition_schema=schema.to_pb2(),
        )
        resp = await get_client().artifact_service.declare_artifact(request)
        return PartitionSchema.from_pb2(resp.partition_schema, declared=True)

    @syncify
    @classmethod
    async def get_schema(
        cls,
        name: str,
        *,
        project: str | None = None,
        domain: str | None = None,
    ) -> PartitionSchema:
        """
        The partition keys fixed for an artifact name. An unpartitioned artifact
        has an empty schema; a name with neither a declaration nor a version is
        NOT_FOUND.
        """
        ensure_client()
        cfg = get_init_config()

        request = artifact_service_pb2.GetArtifactSchemaRequest(name=_name_pb2(cfg, name, project, domain))
        resp = await get_client().artifact_service.get_artifact_schema(request)
        return PartitionSchema.from_pb2(resp.partition_schema, declared=resp.declared)

    @syncify
    async def delete(self) -> None:
        """
        Delete this artifact from the remote system.
        """
        raise NotImplementedError("Artifact deletion not yet implemented.")


def _fmt_partition(value: Any) -> str:
    if isinstance(value, TimePartition):
        return f"{_fmt_partition(value.value)} ({value.granularity})"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


@dataclass
class ArtifactGroup(ToJSONMixin):
    """One distinct artifact name: the latest version plus the total version count."""

    pb2: artifact_service_pb2.ArtifactGroup

    @property
    def name(self) -> str:
        return self.pb2.latest.artifact_id.name.name

    @property
    def versions(self) -> int:
        """Total number of versions published under this name."""
        return self.pb2.versions

    @property
    def latest(self) -> Artifact:
        """The most recently created version of the artifact."""
        return Artifact(pb2=self.pb2.latest)

    @property
    def partition_schema(self) -> PartitionSchema | None:
        """The artifact's partition keys, or None when it has none."""
        if not self.pb2.HasField("partition_schema"):
            return None
        schema = PartitionSchema.from_pb2(self.pb2.partition_schema)
        return schema if schema.all_keys else None

    @property
    def latest_time_partition(self) -> date | datetime | None:
        """The newest time partition among the artifact's versions, or None."""
        if not self.pb2.HasField("latest_time_partition"):
            return None
        dt = self.pb2.latest_time_partition.ToDatetime(tzinfo=timezone.utc)
        schema = self.partition_schema
        return dt if schema is not None and schema.granularity == "hour" else dt.date()

    def __rich_repr__(self) -> rich.repr.Result:
        latest = self.latest
        yield "name", self.name
        yield "versions", self.versions
        schema = self.partition_schema
        yield "partitions", ", ".join(schema.all_keys) if schema else "-"
        ltp = self.latest_time_partition
        if ltp is not None:
            yield "latest_partition", _fmt_partition(ltp)
        yield "latest_version", latest.version
        yield "description", self.pb2.latest.spec.info.description or "-"
        yield "created_at", self.pb2.latest.created_at.ToDatetime().isoformat()
        yield "source", latest.source or "-"
