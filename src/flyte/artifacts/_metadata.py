from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Optional, Tuple

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
        short_description: Optional[str] = None,
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
                "short_description": short_description or "",
            },
            card=card,
        )
