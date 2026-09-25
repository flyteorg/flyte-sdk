from __future__ import annotations

from typing import List, Optional

import libcst as cst

from .._context import MigrationContext
from .._conversions import convert_map, override_arguments
from .._cst import Rule, arguments_code, call_arguments, code


class CallsRule(Rule):
    """
    Convert call sites: `map_task(...)(...)` -> `flyte.map`, `.with_overrides(...)` -> `.override(...)`, and
    `await task(...)` in async code -> `await task.aio(...)`.
    """

    id = "calls"
    description = "map_task -> flyte.map, with_overrides -> override, awaited task calls -> .aio"

    def __init__(self, ctx: MigrationContext):
        super().__init__(ctx)
        self._async_stack: List[bool] = []

    @property
    def _in_async(self) -> bool:
        return bool(self._async_stack) and self._async_stack[-1]

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._async_stack.append(node.asynchronous is not None)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        self._async_stack.pop()
        return updated_node

    def leave_Await(self, original_node: cst.Await, updated_node: cst.Await) -> cst.BaseExpression:
        call = original_node.expression
        if not isinstance(call, cst.Call):
            return updated_node

        mapped = convert_map(self, call, is_async=True)
        if mapped is not None:
            self.applied("map_task")
            return cst.parse_expression(f"[r async for r in {mapped}]")

        converted = self._override(call, is_async=True)
        if converted is not None:
            return cst.parse_expression(f"await {converted}")

        func = call.func
        if isinstance(func, cst.Name) and self._is_sync_task(func.value):
            self.applied("aio")
            return updated_node.with_changes(
                expression=updated_node.expression.with_changes(  # type: ignore[attr-defined]
                    func=cst.Attribute(value=func, attr=cst.Name("aio"))
                )
            )
        return updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        if self._in_async:
            # Awaited calls are handled in `leave_Await`, which sees the whole `await` expression.
            return updated_node
        mapped = convert_map(self, original_node, is_async=False)
        if mapped is not None:
            self.applied("map_task")
            return cst.parse_expression(mapped)
        converted = self._override(original_node, is_async=False)
        if converted is not None:
            return cst.parse_expression(converted)
        return updated_node

    def _override(self, call: cst.Call, *, is_async: bool) -> Optional[str]:
        func = call.func
        if not (
            isinstance(func, cst.Attribute) and func.attr.value == "with_overrides" and isinstance(func.value, cst.Call)
        ):
            return None
        inner = func.value
        overrides, positional = call_arguments(call)
        if positional:
            return None
        self.applied("with_overrides")
        arguments = arguments_code(inner)
        call_suffix = f".aio({arguments})" if is_async else f"({arguments})"
        return f"{code(inner.func)}.override({override_arguments(self, overrides)}){call_suffix}"

    def _is_sync_task(self, name: str) -> bool:
        entity = self.ctx.entities.get(name)
        if entity is not None:
            return not entity.is_async
        return name in self.ctx.local_callables
