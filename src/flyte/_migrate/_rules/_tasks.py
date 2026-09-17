from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import libcst as cst
import libcst.matchers as m

from .._context import ENTITY_DECORATORS, MigrationContext
from .._conversions import convert_cache, merge_resources
from .._cst import Rule, call_arguments, code, kwargs_code, literal

#: `@task` arguments that configure the container, and therefore belong to a `TaskEnvironment` in v2.
ENV_LEVEL_ARGS = (
    "container_image",
    "requests",
    "limits",
    "resources",
    "accelerator",
    "shared_memory",
    "secret_requests",
    "environment",
    "task_config",
)
_CACHE_ARGS = ("cache", "cache_version", "cache_serialize", "cache_ignore_input_vars")
_PASSTHROUGH_ARGS = ("retries", "timeout", "interruptible", "pod_template")
_DROPPED_ARGS = {
    "deprecated": "`deprecated` has no v2 equivalent and was dropped",
    "docs": "`docs` was dropped; v2 uses the function docstring (or flyte Documentation via `docs=`)",
    "labels": "labels are set per run in v2; use flyte.with_runcontext(labels=...)",
    "annotations": "annotations are set per run in v2; use flyte.with_runcontext(annotations=...)",
    "node_dependency_hints": "`node_dependency_hints` is not needed in v2 and was dropped",
    "pickle_untyped": "`pickle_untyped` has no v2 equivalent and was dropped",
    "deck_fields": "`deck_fields` has no v2 equivalent; use flyte.report",
    "disable_deck": "`disable_deck` was dropped; v2 reports are off unless `report=True`",
    "execution_mode": "`execution_mode` has no v2 equivalent and was dropped",
    "task_resolver": "flytekit task resolvers are not compatible with v2 and were dropped",
    "failure_policy": "WorkflowFailurePolicy has no v2 equivalent; handle failures with try/except in this task",
    "on_failure": "`on_failure` has no v2 equivalent; handle failures with try/except in this task",
    "default_options": "`default_options` has no v2 equivalent; use flyte.with_runcontext",
}


@dataclass
class _Group:
    """Entities that share the same environment-level configuration."""

    fingerprint: Tuple[Tuple[str, str], ...]
    members: List[str] = field(default_factory=list)
    #: Groups whose tasks this group's entities call, in first-call order.
    depends_on: List["_Group"] = field(default_factory=list)
    name: str = ""
    variable: str = ""
    env_args: Optional[List[Tuple[str, str]]] = None


class TasksRule(Rule):
    """Convert flytekit entity decorators to `@env.task` and generate the `TaskEnvironment`s they need."""

    id = "tasks"
    description = "@task/@dynamic/@eager/@workflow -> TaskEnvironment + @env.task"

    def __init__(self, ctx: MigrationContext):
        super().__init__(ctx)
        self._groups: Dict[Tuple[Tuple[str, str], ...], _Group] = {}
        self._group_of: Dict[str, _Group] = {}

    # -- grouping ---------------------------------------------------------------------------------------------------

    def visit_Module(self, node: cst.Module) -> None:
        taken = _module_level_names(node)
        for statement in node.body:
            if not isinstance(statement, cst.FunctionDef) or statement.name.value not in self.ctx.entities:
                continue
            decorator = self._entity_decorator(statement)
            if decorator is None:
                continue
            args = _decorator_arguments(decorator)
            fingerprint = tuple((k, code(args[k])) for k in ENV_LEVEL_ARGS if k in args)
            group = self._groups.setdefault(fingerprint, _Group(fingerprint))
            group.members.append(statement.name.value)
            self._group_of[statement.name.value] = group

        # v2 deploys an environment's dependencies with it, so depend on the environments whose tasks are called.
        for statement in node.body:
            if not isinstance(statement, cst.FunctionDef) or statement.name.value not in self._group_of:
                continue
            group = self._group_of[statement.name.value]
            for called in m.findall(statement.body, m.Name()):
                callee = self._group_of.get(called.value) if isinstance(called, cst.Name) else None
                if callee is not None and callee is not group and callee not in group.depends_on:
                    group.depends_on.append(callee)

        base = _snake(self.ctx.stem)
        single = len(self._groups) == 1
        for group in self._groups.values():
            if single or not group.fingerprint:
                group.name = f"{base}_env"
            else:
                group.name = f"{base}_{_snake(group.members[0])}_env"
            group.variable = "env" if single and "env" not in taken else _identifier(group.name, taken)
            taken.add(group.variable)
            self.ctx.environments.append(group.name)

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> Union[cst.FunctionDef, cst.FlattenSentinel]:
        name = original_node.name.value
        group = self._group_of.get(name)
        decorator = self._entity_decorator(original_node)
        if group is None or decorator is None:
            return updated_node
        entity = self.ctx.entities[name]
        args = _decorator_arguments(decorator)

        if group.env_args is None:
            group.env_args = self._environment_arguments(args)

        task_args = self._task_arguments(args)
        if entity.kind == "workflow":
            task_args.insert(0, ("entrypoint", "True"))
        triggers = self.ctx.triggers.get(name, [])
        if len(triggers) == 1:
            task_args.append(("triggers", triggers[0]))
        elif triggers:
            task_args.append(("triggers", f"({', '.join(triggers)},)"))

        source = f"@{group.variable}.task({kwargs_code(task_args)})" if task_args else f"@{group.variable}.task"
        new_decorator = cst.Decorator(decorator=cst.parse_expression(source[1:]))
        index = original_node.decorators.index(decorator)
        decorators = list(updated_node.decorators)
        decorators[index] = new_decorator.with_changes(leading_lines=decorators[index].leading_lines)
        self.applied(entity.kind)
        return updated_node.with_changes(decorators=decorators)

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if not self._groups:
            return updated_node
        self.ctx.require_import("flyte")
        body = list(updated_node.body)
        first = next(
            i
            for i, statement in enumerate(body)
            if isinstance(statement, cst.FunctionDef) and statement.name.value in self._group_of
        )
        body[first:first] = self._environment_statements()
        return updated_node.with_changes(body=body)

    def _environment_statements(self) -> List[cst.SimpleStatementLine]:
        """Define environments after the ones they depend on; dependency cycles fall back to `add_dependency`."""
        remaining = list(self._groups.values())
        defined: List[_Group] = []
        deferred: List[str] = []
        statements: List[cst.SimpleStatementLine] = []
        while remaining:
            group = next((g for g in remaining if all(d in defined for d in g.depends_on)), remaining[0])
            remaining.remove(group)
            ready = [d.variable for d in group.depends_on if d in defined]
            deferred.extend(
                f"{group.variable}.add_dependency({d.variable})" for d in group.depends_on if d not in defined
            )
            statements.append(self._environment_statement(group, ready))
            defined.append(group)
        for source in deferred:
            statements.append(cst.parse_statement(source + "\n"))  # type: ignore[arg-type]
        return statements

    # -- environment ------------------------------------------------------------------------------------------------

    def _environment_arguments(self, args: Dict[str, cst.BaseExpression]) -> List[Tuple[str, str]]:
        env_args: List[Tuple[str, str]] = []
        image = args.get("container_image")
        if image is not None:
            value = literal(image)
            if isinstance(value, str) and "{{" in value:
                self.todo("flytekit image templates are not supported in v2; use flyte.Image")
            env_args.append(("image", code(image)))
        resources = merge_resources(self, args)
        if resources is not None:
            env_args.append(("resources", resources))
        if "secret_requests" in args:
            env_args.append(("secrets", code(args["secret_requests"])))
        if "environment" in args:
            env_args.append(("env_vars", code(args["environment"])))
        if "task_config" in args:
            self.todo(
                "plugin configuration types differ in v2; replace `plugin_config` with the flyteplugins equivalent"
            )
            env_args.append(("plugin_config", code(args["task_config"])))
        return env_args

    def _environment_statement(self, group: _Group, depends_on: Sequence[str]) -> cst.SimpleStatementLine:
        env_args: List[Tuple[str, str]] = [("name", f'"{group.name}"'), *(group.env_args or [])]
        if depends_on:
            env_args.append(("depends_on", f"[{', '.join(depends_on)}]"))
        statement = cst.parse_statement(f"{group.variable} = flyte.TaskEnvironment({kwargs_code(env_args)})\n")
        assert isinstance(statement, cst.SimpleStatementLine)
        return statement.with_changes(leading_lines=[cst.EmptyLine(), cst.EmptyLine()])

    # -- task -------------------------------------------------------------------------------------------------------

    def _task_arguments(self, args: Dict[str, cst.BaseExpression]) -> List[Tuple[str, str]]:
        task_args: List[Tuple[str, str]] = []
        cache = convert_cache(self, args)
        if cache is not None:
            task_args.append(("cache", cache))
        for key, node in args.items():
            if key in ENV_LEVEL_ARGS or key in _CACHE_ARGS:
                continue
            if key in _PASSTHROUGH_ARGS:
                task_args.append((key, code(node)))
            elif key == "pod_template_name":
                task_args.append(("pod_template", code(node)))
            elif key == "enable_deck":
                if literal(node) is not False:
                    task_args.append(("report", code(node)))
            elif key in _DROPPED_ARGS:
                self.todo(_DROPPED_ARGS[key])
            else:
                self.todo(f"`{key}` has no automatic v2 mapping and was dropped")
        return task_args

    def _entity_decorator(self, node: cst.FunctionDef) -> Optional[cst.Decorator]:
        for decorator in node.decorators:
            expr = decorator.decorator
            target = expr.func if isinstance(expr, cst.Call) else expr
            if self.flytekit_symbol(target) in ENTITY_DECORATORS:
                return decorator
        return None


def _decorator_arguments(decorator: cst.Decorator) -> Dict[str, cst.BaseExpression]:
    expr = decorator.decorator
    if not isinstance(expr, cst.Call):
        return {}
    kwargs, _ = call_arguments(expr)
    return kwargs


def _module_level_names(module: cst.Module) -> set:
    names = set()
    for statement in module.body:
        if isinstance(statement, (cst.FunctionDef, cst.ClassDef)):
            names.add(statement.name.value)
        elif isinstance(statement, cst.SimpleStatementLine):
            for small in statement.body:
                if isinstance(small, cst.Assign):
                    for target in small.targets:
                        if isinstance(target.target, cst.Name):
                            names.add(target.target.value)
    return names


def _snake(value: str) -> str:
    """v2 environment names must match `^[a-z0-9]+([_-][a-z0-9]+)*$`."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", value.lower())).strip("_") or "flyte"


def _identifier(name: str, taken: set) -> str:
    candidate = name if name.isidentifier() and not keyword.iskeyword(name) else f"_{name}"
    suffix = 2
    unique = candidate
    while unique in taken:
        unique = f"{candidate}_{suffix}"
        suffix += 1
    return unique
