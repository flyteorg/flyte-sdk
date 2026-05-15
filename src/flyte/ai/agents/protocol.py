"""Agent protocol for the flyte.ai.agents module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class AgentResult:
    """Outcome of a single agent invocation."""

    code: str = ""
    charts: list[str] = field(default_factory=list)
    summary: str = ""
    error: str = ""
    attempts: int = 1


@runtime_checkable
class Agent(Protocol):
    """Minimal protocol that any agent must satisfy to work with
    :class:`AgentChatAppEnvironment`.
    """

    async def run(self, message: str, history: list[dict[str, str]]) -> AgentResult:
        """Process *message* (with prior *history*) and return an :class:`AgentResult`."""
        ...

    def tool_descriptions(self) -> list[dict[str, str]]:
        """Return JSON-friendly metadata for every registered tool.

        Each dict should contain at least ``name``, ``signature``, and
        ``description`` keys.
        """
        ...
