from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal, Mapping, Type

import rich.repr
from flyteidl2.artifact import artifact_pb2, artifact_service_pb2
from flyteidl2.common import identifier_pb2, list_pb2
from flyteidl2.core import artifact_id_pb2, literals_pb2

from flyte._initialize import ensure_client, get_client, get_init_config
from flyte.artifacts._card import Card as CoreCard
from flyte.artifacts._metadata import KIND_KEY, Kind, Metadata, resolve_attrs
from flyte.artifacts._wrapper import ArtifactWrapper, ensure_artifactable
from flyte.remote._common import ToJSONMixin
from flyte.syncify import syncify

_LIST_PAGE_SIZE = 100


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

        request = artifact_service_pb2.CreateArtifactRequest(
            artifact_id=artifact_pb2.ArtifactIdentifier(
                name=artifact_pb2.ArtifactName(
                    org=cfg.org or "",
                    project=project or cfg.project or "",
                    domain=domain or cfg.domain or "",
                    name=name,
                ),
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
    ) -> Artifact:
        """
        Get an artifact by its name and version.

        Args:
            name: The name of the artifact.
            version: The version of the artifact; "latest" returns the most recently created version.
            project: Project to look in; defaults to the init configuration.
            domain: Domain to look in; defaults to the init configuration.
        """
        ensure_client()
        cfg = get_init_config()

        request = artifact_service_pb2.GetArtifactRequest(
            name=artifact_pb2.ArtifactName(
                org=cfg.org or "",
                project=project or cfg.project or "",
                domain=domain or cfg.domain or "",
                name=name,
            ),
        )
        if version != "latest":
            request.version = version
        resp = await get_client().artifact_service.get_artifact(request)
        return cls(pb2=resp.artifact)

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
    ) -> AsyncIterator[Artifact]:
        """
        List artifacts, newest first.

        Args:
            name: Exact artifact name; when set, all versions of that artifact are listed.
            created_after: Filter artifacts created after this datetime.
            limit: The maximum number of artifacts to return. -1 for no limit.
            project: Project to list in; defaults to the init configuration.
            domain: Domain to list in; defaults to the init configuration.
            source_run: Only artifacts produced by this run.
            source_action: Only artifacts produced by this action; usually combined with source_run.
            source_external_ref: Only artifacts imported from this external reference.

        Returns:
            An async iterator of artifacts.
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
        if created_after is not None:
            ts = created_after if created_after.tzinfo else created_after.replace(tzinfo=timezone.utc)
            filters.append(
                list_pb2.Filter(
                    function=list_pb2.Filter.GREATER_THAN,
                    field="created_at",
                    values=[ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")],
                )
            )

        token = ""
        remaining = limit if limit >= 0 else None
        while remaining is None or remaining > 0:
            page_size = _LIST_PAGE_SIZE if remaining is None else min(_LIST_PAGE_SIZE, remaining)
            request = artifact_service_pb2.ListArtifactsRequest(
                request=list_pb2.ListRequest(limit=page_size, token=token, filters=filters),
                project_id=identifier_pb2.ProjectIdentifier(
                    organization=cfg.org or "",
                    domain=domain or cfg.domain or "",
                    name=project or cfg.project or "",
                ),
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
    async def delete(self) -> None:
        """
        Delete this artifact from the remote system.
        """
        raise NotImplementedError("Artifact deletion not yet implemented.")


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

    def __rich_repr__(self) -> rich.repr.Result:
        latest = self.latest
        yield "name", self.name
        yield "versions", self.versions
        yield "latest_version", latest.version
        yield "description", self.pb2.latest.spec.info.description or "-"
        yield "created_at", self.pb2.latest.created_at.ToDatetime().isoformat()
        yield "source", latest.source or "-"
