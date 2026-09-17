"""Resolve imports of modules that live next to the file being migrated."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

import libcst as cst


def local_module_path(source_path: Path, node: cst.ImportFrom) -> Optional[Path]:
    """The file an `from x import y` statement refers to, if it sits in the source file's directory tree."""
    if node.module is None:
        return None
    base = source_path.parent
    for _ in range(max(len(node.relative) - 1, 0)):
        base = base.parent
    parts = cst.Module([]).code_for_node(node.module).split(".")
    candidates = [base.joinpath(*parts).with_suffix(".py"), base.joinpath(*parts, "__init__.py")]
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() != source_path.resolve():
            return candidate
    return None


def sync_entity_names(path: Path) -> Set[str]:
    """Names of synchronous flytekit tasks and workflows defined at the top level of a local module."""
    from ._analyze import collect_entities

    try:
        module = cst.parse_module(path.read_text())
    except (OSError, UnicodeDecodeError, cst.ParserSyntaxError):
        return set()
    return {name for name, entity in collect_entities(module).items() if not entity.is_async}
