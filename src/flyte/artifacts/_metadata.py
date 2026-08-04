from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Optional, Tuple

from flyteidl2.core import artifact_id_pb2, types_pb2
from flyteidl2.task import common_pb2

from ._card import Card


@dataclass(frozen=True, kw_only=True)
class Metadata:
    """Structured metadata for Flyte artifacts."""

    # Core tracking fields
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    data: Optional[typing.Mapping[str, str]] = None
    card: Optional[Card] = None

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
        data: Optional[typing.Mapping[str, str]] = None,
    ) -> Metadata:
        """
        Helper method to create ModelMetadata. This method sets the data keys specific to models.
        Extra key/values passed via `data` are merged in; the model-specific keys win on conflict.
        """
        merged: dict[str, str] = dict(data) if data else {}
        merged.update(
            {
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
            data=merged,
            card=card,
        )


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
        user_metadata=dict(md.data) if md.data else None,
        card=card,
    )
    return common_pb2.ProducedArtifact(
        output=output,
        name=md.name,
        version=md.version or "",
        info=info,
        type=literal_type,
    )
