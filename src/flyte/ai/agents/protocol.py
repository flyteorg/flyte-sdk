"""Agent protocol for the flyte.ai.agents module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .memory import MemoryStore


@dataclass
class AgentResult:
    """Outcome of a single agent invocation."""

    code: str = ""
    charts: list[str] = field(default_factory=list)
    summary: str = ""
    error: str = ""
    attempts: int = 1
    memory: "MemoryStore | None" = None


@runtime_checkable
class AgentProtocol(Protocol):
    """Minimal protocol that any agent must satisfy to work with
    :class:`AgentChatAppEnvironment`.
    """

    def run(self, message: str, memory: list[dict[str, Any]] | "MemoryStore" | None = None) -> AgentResult:
        """Process *message* (with prior *memory*) and return an :class:`AgentResult`.

        ``memory`` may be a ``list[dict]`` of prior messages (e.g. a chat
        ``history``) or a :class:`MemoryStore` for durable, cross-run state.

        Synchronous entry point. In async contexts, use ``run.aio(...)``.
        """
        ...

    def tool_descriptions(self) -> list[dict[str, str]]:
        """Return JSON-friendly metadata for every registered tool.

        Each dict should contain at least ``name``, ``signature``, and
        ``description`` keys.
        """
        ...
