from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxedConfig:
    """Configuration for a sandboxed task executed via Monty."""

    max_memory: int = 50 * 1024 * 1024  # 50 MB
    max_stack_depth: int = 256
    timeout_ms: int = 30_000
    type_check: bool = True
