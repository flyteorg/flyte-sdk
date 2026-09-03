from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from flyteidl2.core import artifact_id_pb2, types_pb2
from flyteidl2.task import common_pb2

from ._card import Card

#: Reserved `attrs` key naming what an artifact *is*, as opposed to how its card
#: renders. Stamped at creation and read back through `flyte.remote.Artifact.kind`;
#: callers should never reach into `user_metadata` for it themselves.
#:
#: This lives in free-form metadata rather than a typed field because the artifact
#: proto has no discriminator today, and the set of kinds is still open -- an enum
#: would have to be widened through an IDL release before anyone could even write a
#: new value. The key is deliberately narrow so it can be promoted to a typed
#: `ArtifactInfo` field later and backfilled, without touching producers.
#:
#: It is namespaced because `user_metadata` is documented as user-supplied: a bare
#: `kind` would collide with an attr a user legitimately owns, and silently
#: misclassify their artifact.
KIND_KEY = "flyte.io/kind"

Kind = Literal["model", "data", "generic"]


@dataclass(frozen=True, kw_only=True)
class ArtifactParent:
    """One parent edge for an artifact version's lineage (`Metadata.parents`).

    Every field except `version` defaults to "inherit from the child being
    published": an empty `name` means a same-name parent (an earlier version of
    this artifact), and empty scope fields inherit the child's
    org/project/domain. A bare version string is accepted anywhere an
    `ArtifactParent` is — it is shorthand for `ArtifactParent(version=...)`.

    Parents are stored as given and never resolved: a parent that hasn't been
    (or never will be) published is legal, like a git remote missing commits
    that were never pushed. Only a direct self-reference is rejected by the
    service.
    """

    version: str
    name: Optional[str] = None
    project: Optional[str] = None
    domain: Optional[str] = None
    org: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class Metadata:
    """Structured metadata for Flyte artifacts."""

    # Core tracking fields
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    attrs: Optional[typing.Mapping[str, str]] = None
    card: Optional[Card] = None
    #: What this artifact is. Merged into `attrs` under `KIND_KEY` at
    #: serialization time; an explicit `attrs["kind"]` wins, so a caller who
    #: sets the key by hand is never silently overridden.
    kind: Optional[Kind] = None
    #: Lineage: the artifact versions this version derives from, ordered with
    #: the primary parent first (git-style merge lineage; up to 32). Each entry
    #: is an `ArtifactParent` or a bare version string (shorthand for a
    #: same-name parent).
    parents: Optional[Tuple[typing.Union[str, ArtifactParent], ...]] = None
    #: When no explicit `version` is given, publish under the content hash the
    #: type transformer stamped on the literal (`Literal.hash`) instead of the
    #: backend's run-action-attempt default. Opt-in: content-addressed versions
    #: make re-publishing identical content idempotent, but they also mean
    #: unchanged content produces NO new version — only types whose literals
    #: carry a meaningful content hash should set this.
    version_from_content: bool = False

    @classmethod
    def create_model_metadata(
        cls,
        *,
        name: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        card: Optional[Card] = None,
        framework: Optional[str] = None,
        model_type: Optional[str] = None,
        architecture: Optional[str] = None,
        task: Optional[str] = None,
        modality: Tuple[str, ...] = ("text",),
        serial_format: str = "safetensors",
        attrs: Optional[typing.Mapping[str, str]] = None,
    ) -> Metadata:
        """
        Helper method to create ModelMetadata. This method sets the attrs keys specific to models.
        Extra key/values passed via `attrs` are merged in; the model-specific keys win on conflict.

        The reserved `kind` key is set to "model" here, so a model is identifiable
        without depending on the shape of the key set or on a card being attached.
        """
        merged: dict[str, str] = dict(attrs) if attrs else {}
        merged.update(
            {
                KIND_KEY: "model",
                "framework": framework or "",
                "model_type": model_type or "",
                "architecture": architecture or "",
                "task": task or "",
                "modality": ",".join(modality) if modality else "",
                "serial_format": serial_format or "",
            }
        )
        return cls(
            name=name,
            version=version,
            description=description,
            attrs=merged,
            card=card,
        )


def resolve_attrs(md: Metadata) -> dict[str, str]:
    """
    The `attrs` an artifact is published with, including the reserved kind key.

    An explicit `attrs[KIND_KEY]` beats the `kind=` field. Writing the reserved,
    namespaced key by hand is unambiguous -- nobody arrives at "flyte.io/kind" by
    accident -- so it is treated as deliberate. `kind=` is the ergonomic path and
    only fills the key in when it is absent.
    """
    attrs: dict[str, str] = dict(md.attrs) if md.attrs else {}
    if md.kind is not None:
        attrs.setdefault(KIND_KEY, md.kind)
    return attrs


def parents_to_pb2(
    parents: Optional[typing.Sequence[typing.Union[str, ArtifactParent]]],
) -> list[artifact_id_pb2.ArtifactVersionId]:
    """
    Serialize parent edges for the wire (`ArtifactSpec.parent_artifacts` /
    `ProducedArtifact.parent_artifacts`). A bare string becomes a keyless
    entry — the service inherits the child's name and scope for empty key
    fields — and an `ArtifactParent` carries a key only when it overrides
    something, keeping the common same-name case terse on the wire.
    """
    out: list[artifact_id_pb2.ArtifactVersionId] = []
    for entry in parents or ():
        parent = ArtifactParent(version=entry) if isinstance(entry, str) else entry
        key = None
        if parent.name or parent.project or parent.domain or parent.org:
            key = artifact_id_pb2.ArtifactKey(
                org=parent.org or "",
                project=parent.project or "",
                domain=parent.domain or "",
                name=parent.name or "",
            )
        out.append(artifact_id_pb2.ArtifactVersionId(key=key, version=parent.version))
    return out


def to_produced_artifact(
    md: Metadata,
    *,
    output: str,
    literal_type: types_pb2.LiteralType,
) -> common_pb2.ProducedArtifact:
    """
    Convert a `Metadata` into the first-class production declaration carried on the
    Outputs envelope (`task.Outputs.produced_artifacts`). The declaration is
    self-contained: the backend registers the artifact from it verbatim (identity
    scope and a default version come from the producing action).
    """
    card = None
    if md.card is not None:
        card = artifact_id_pb2.ArtifactCard(uri=md.card.uri, format=md.card.format, type=md.card.card_type)
    info = artifact_id_pb2.ArtifactInfo(
        description=md.description or "",
        user_metadata=resolve_attrs(md) or None,
        card=card,
    )
    return common_pb2.ProducedArtifact(
        output=output,
        name=md.name,
        version=md.version or "",
        info=info,
        type=literal_type,
        parent_artifacts=parents_to_pb2(md.parents) or None,
    )
