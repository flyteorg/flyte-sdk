from __future__ import annotations

import json
import typing
from dataclasses import dataclass
from typing import Optional, Tuple, cast

from ._card import Card, CardFormat, CardType


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
    ) -> Metadata:
        """
        Helper method to create ModelMetadata. This method sets the data keys specific to models.
        """
        return cls(
            name=name,
            version=version,
            description=description,
            data={
                "framework": framework or "",
                "model_type": model_type or "",
                "architecture": architecture or "",
                "task": task or "",
                "modality": ",".join(modality) if modality else "",
                "serial_format": serial_format or "",
            },
            card=card,
        )


def to_compact_json(md: Metadata) -> str:
    """
    Serialize a `Metadata` to compact, deterministic JSON for stamping into a literal's
    metadata map (under `flyte._constants.ARTIFACT_PRODUCED_KEY`). None fields are omitted
    and keys are sorted, so equal metadata always yields byte-identical JSON. The backend
    reader (leaseworker) parses exactly this shape — keep the two in sync.
    """
    payload: dict[str, typing.Any] = {"name": md.name}
    if md.version is not None:
        payload["version"] = md.version
    if md.description is not None:
        payload["description"] = md.description
    if md.data is not None:
        payload["data"] = dict(md.data)
    if md.card is not None:
        payload["card"] = {"uri": md.card.uri, "format": md.card.format, "type": md.card.card_type}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def from_compact_json(s: str) -> Metadata:
    """Inverse of `to_compact_json`."""
    payload = json.loads(s)
    card = None
    if "card" in payload:
        card = Card(
            uri=payload["card"]["uri"],
            format=cast(CardFormat, payload["card"]["format"]),
            card_type=cast(CardType, payload["card"]["type"]),
        )
    return Metadata(
        name=payload["name"],
        version=payload.get("version"),
        description=payload.get("description"),
        data=payload.get("data"),
        card=card,
    )
