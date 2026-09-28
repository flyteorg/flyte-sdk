from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import libcst as cst

EntityKind = Literal["task", "dynamic", "eager", "workflow"]

#: flytekit decorators that turn a function into a Flyte entity, by their final name segment.
ENTITY_DECORATORS: Dict[str, EntityKind] = {
    "task": "task",
    "dynamic": "dynamic",
    "eager": "eager",
    "workflow": "workflow",
}


@dataclass
class Entity:
    """A module-level function decorated with a flytekit entity decorator."""

    name: str
    kind: EntityKind
    is_async: bool
    params: List[str]


@dataclass
class MigrationContext:
    """State shared by all rules while migrating one file."""

    source_path: Path
    suffix: str
    entities: Dict[str, Entity] = field(default_factory=dict)
    #: Names imported from modules that live next to the source file (probably other flytekit modules).
    local_callables: Set[str] = field(default_factory=set)
    #: Trigger expressions (source code) to attach to an entity, keyed by entity name.
    triggers: Dict[str, List[str]] = field(default_factory=dict)
    #: Imports the generated code needs, as (module, object) pairs; object None means `import module`.
    imports: Set[Tuple[str, Optional[str]]] = field(default_factory=set)
    needs_collect_helper: bool = False
    applied: Counter[str] = field(default_factory=Counter)
    environments: List[str] = field(default_factory=list)
    #: TODOs that could not be attached to a statement; they are emitted at the top of the module.
    module_todos: List[str] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return self.source_path.stem

    def require_import(self, module: str, obj: Optional[str] = None) -> None:
        self.imports.add((module, obj))


def entity_params(node: cst.FunctionDef) -> List[str]:
    params = node.params
    return [p.name.value for p in (*params.posonly_params, *params.params, *params.kwonly_params)]
