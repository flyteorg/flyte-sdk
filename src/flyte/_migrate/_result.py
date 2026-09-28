from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal

MigrationStatus = Literal["converted", "converted_with_todos", "nothing_to_migrate"]

#: Every manual follow-up the converter leaves in generated code starts with this marker.
TODO_MARKER = "TODO(flyte migrate):"


class MigrationError(Exception):
    """Raised when a file cannot be migrated. Nothing is written when this is raised."""


class OutputExistsError(MigrationError):
    """Raised when the output file already exists and overwriting was not requested."""


@dataclass(frozen=True)
class Todo:
    """A `TODO(flyte migrate)` marker in the generated source."""

    line: int
    message: str


@dataclass
class MigrationResult:
    """The outcome of migrating one flytekit (v1) file to flyte (v2)."""

    source_path: Path
    output_path: Path
    status: MigrationStatus
    code: str = ""
    applied: Counter[str] = field(default_factory=Counter)
    todos: List[Todo] = field(default_factory=list)
    environments: List[str] = field(default_factory=list)

    def to_dict(self, include_code: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "source": str(self.source_path),
            "output": str(self.output_path),
            "status": self.status,
            "rules_applied": dict(sorted(self.applied.items())),
            "environments": self.environments,
            "todos": [{"line": t.line, "message": t.message} for t in self.todos],
        }
        if include_code:
            data["code"] = self.code
        return data
