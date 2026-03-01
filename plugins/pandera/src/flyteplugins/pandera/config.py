"""Pandera validation configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ValidationConfig:
    """Configuration for Pandera validation behavior.

    Attributes:
        on_error: Determines how validation errors are handled.
            "raise" will raise the SchemaError/SchemaErrors exception.
            "warn" will log a warning and continue with the original data.
    """

    on_error: Literal["raise", "warn"] = "raise"
