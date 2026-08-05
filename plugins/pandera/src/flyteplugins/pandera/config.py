"""Pandera validation configuration."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ValidationConfig:
    """Controls behavior for pandera validation failures."""

    on_error: Literal["raise", "warn"] = "raise"
