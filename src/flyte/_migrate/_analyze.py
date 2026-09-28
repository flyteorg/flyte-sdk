"""Read-only analysis that runs before any rule: which flytekit entities a module defines and imports."""

from __future__ import annotations

from typing import Dict

import libcst as cst
from libcst.metadata import QualifiedNameProvider

from ._context import ENTITY_DECORATORS, Entity, MigrationContext, entity_params


class _EntityCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self) -> None:
        super().__init__()
        self.entities: Dict[str, Entity] = {}

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        return False

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        for decorator in node.decorators:
            expr = decorator.decorator
            target = expr.func if isinstance(expr, cst.Call) else expr
            for q in self.get_metadata(QualifiedNameProvider, target, set()):
                symbol = q.name.rsplit(".", 1)[-1]
                if q.name.startswith("flytekit.") and symbol in ENTITY_DECORATORS:
                    self.entities[node.name.value] = Entity(
                        name=node.name.value,
                        kind=ENTITY_DECORATORS[symbol],
                        is_async=node.asynchronous is not None,
                        params=entity_params(node),
                    )
                    return False
        # Only module-level functions are entities.
        return False


def collect_entities(module: cst.Module) -> Dict[str, Entity]:
    collector = _EntityCollector()
    cst.MetadataWrapper(module).visit(collector)
    return collector.entities


def analyze(module: cst.Module, ctx: MigrationContext) -> None:
    """Record the entities the module defines and the synchronous tasks it imports from local modules."""
    from ._local_modules import local_module_path, sync_entity_names

    ctx.entities.update(collect_entities(module))
    for statement in module.body:
        if not isinstance(statement, cst.SimpleStatementLine):
            continue
        for small in statement.body:
            if not isinstance(small, cst.ImportFrom) or isinstance(small.names, cst.ImportStar):
                continue
            path = local_module_path(ctx.source_path, small)
            if path is None:
                continue
            tasks = sync_entity_names(path)
            for alias in small.names:
                if isinstance(alias.name, cst.Name) and alias.name.value in tasks:
                    asname = alias.asname.name if alias.asname else alias.name
                    ctx.local_callables.add(cst.Module([]).code_for_node(asname))
