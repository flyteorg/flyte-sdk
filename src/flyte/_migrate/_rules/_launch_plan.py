from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple, Union

import libcst as cst
import libcst.matchers as m

from .._context import MigrationContext
from .._cst import NOT_LITERAL, Rule, call_arguments, code, kwargs_code, literal

_CREATE_POSITIONAL = ("name", "workflow")
_GET_OR_CREATE_POSITIONAL = ("workflow", "name")
_HANDLED = {
    "name",
    "workflow",
    "schedule",
    "default_inputs",
    "fixed_inputs",
    "labels",
    "annotations",
    "overwrite_cache",
}


class LaunchPlanRule(Rule):
    """Convert module-level scheduled `LaunchPlan`s into `flyte.Trigger`s on the launched entity."""

    id = "launch-plan"
    description = "LaunchPlan with a schedule -> flyte.Trigger on the entrypoint task"

    def __init__(self, ctx: MigrationContext):
        super().__init__(ctx)
        self._depth = 0
        self._module: Optional[cst.Module] = None

    def visit_Module(self, node: cst.Module) -> None:
        self._module = node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._depth += 1

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        self._depth -= 1
        return updated_node

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        self._depth += 1

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        self._depth -= 1
        return updated_node

    def leave_SimpleStatementLine(
        self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
    ) -> Union[cst.SimpleStatementLine, cst.RemovalSentinel]:
        if self._depth or len(original_node.body) != 1:
            return updated_node
        statement = original_node.body[0]
        target: Optional[str] = None
        if isinstance(statement, cst.Assign) and len(statement.targets) == 1:
            if isinstance(statement.targets[0].target, cst.Name):
                target = statement.targets[0].target.value
            value = statement.value
        elif isinstance(statement, cst.Expr):
            value = statement.value
        else:
            return updated_node
        if not isinstance(value, cst.Call):
            return updated_node

        qualified = self.flytekit_name(value.func) or ""
        if qualified.endswith("LaunchPlan.create"):
            positional = _CREATE_POSITIONAL
        elif qualified.endswith("LaunchPlan.get_or_create"):
            positional = _GET_OR_CREATE_POSITIONAL
        else:
            return updated_node

        args, _ = call_arguments(value, positional)
        workflow = args.get("workflow")
        schedule = args.get("schedule")
        if not isinstance(workflow, cst.Name) or workflow.value not in self.ctx.entities:
            self.todo("LaunchPlan for an entity outside this file was not converted; attach a flyte.Trigger to it")
            return updated_node
        if schedule is None:
            self.todo(
                "LaunchPlans without a schedule have no v2 equivalent; pass inputs with flyte.with_runcontext "
                "or wrap the task"
            )
            return updated_node

        trigger = self._trigger(workflow.value, args, schedule)
        if trigger is None:
            return updated_node
        if target is not None and self._is_referenced(target):
            self.todo(f"`{target}` is still referenced; its schedule moved to a flyte.Trigger on `{workflow.value}`")
            return updated_node

        self.ctx.triggers.setdefault(workflow.value, []).append(trigger)
        self.applied("scheduled")
        return cst.RemoveFromParent()

    def _trigger(self, entity: str, args: Dict[str, cst.BaseExpression], schedule: cst.BaseExpression) -> Optional[str]:
        automation, kickoff_arg = self._automation(schedule)
        if automation is None:
            self.todo(f"could not convert schedule `{code(schedule)}` to flyte.Cron or flyte.FixedRate")
            return None

        fields: List[Tuple[str, str]] = []
        name = args.get("name")
        fields.append(("name", code(name) if name is not None else repr(f"{entity}_schedule")))
        fields.append(("automation", automation))

        inputs = self._inputs(args, kickoff_arg)
        if inputs:
            fields.append(("inputs", inputs))
        for key in ("labels", "annotations", "overwrite_cache"):
            if key in args:
                fields.append((key, code(args[key])))
        for key in args:
            if key not in _HANDLED:
                self.todo(f"LaunchPlan `{key}` was not converted; see flyte.Trigger for the v2 options")

        self.ctx.require_import("flyte")
        return f"flyte.Trigger({kwargs_code(fields)})"

    def _automation(self, schedule: cst.BaseExpression) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(schedule, cst.Call):
            return None, None
        symbol = self.flytekit_symbol(schedule.func)
        if symbol == "CronSchedule":
            kwargs, _ = call_arguments(schedule, ("cron_expression", "schedule", "offset", "kickoff_time_input_arg"))
            expression = kwargs.get("schedule", kwargs.get("cron_expression"))
            if expression is None:
                return None, None
            if "offset" in kwargs:
                self.todo("CronSchedule `offset` has no v2 equivalent and was dropped")
            return f"flyte.Cron({code(expression)})", _string(kwargs.get("kickoff_time_input_arg"))
        if symbol == "FixedRate":
            kwargs, _ = call_arguments(schedule, ("duration", "kickoff_time_input_arg"))
            duration = kwargs.get("duration")
            if duration is None:
                return None, None
            return (
                f"flyte.FixedRate(interval_minutes={interval_minutes(duration)})",
                _string(kwargs.get("kickoff_time_input_arg")),
            )
        return None, None

    def _inputs(self, args: Dict[str, cst.BaseExpression], kickoff_arg: Optional[str]) -> Optional[str]:
        entries: List[str] = []
        for key in ("default_inputs", "fixed_inputs"):
            node = args.get(key)
            if node is None:
                continue
            if isinstance(node, cst.Dict):
                entries.extend(code(element) for element in node.elements)
            else:
                entries.append(f"**{code(node)}")
        if kickoff_arg is not None:
            entries.append(f"{kickoff_arg!r}: flyte.TriggerTime")
        return "{" + ", ".join(entries) + "}" if entries else None

    def _is_referenced(self, name: str) -> bool:
        assert self._module is not None
        return len(m.findall(self._module, m.Name(value=name))) > 1


def interval_minutes(duration: cst.BaseExpression) -> str:
    """Render a `timedelta` expression as whole minutes, as a literal when it can be evaluated statically."""
    if isinstance(duration, cst.Call) and code(duration.func) in ("timedelta", "datetime.timedelta"):
        kwargs, positional = call_arguments(duration, ("days", "seconds", "microseconds", "milliseconds", "minutes"))
        values = {k: literal(v) for k, v in kwargs.items()}
        if not positional and all(isinstance(v, (int, float)) for v in values.values()):
            return str(int(datetime.timedelta(**values).total_seconds() // 60))
    return f"int(({code(duration)}).total_seconds() // 60)"


def _string(node: Optional[cst.BaseExpression]) -> Optional[str]:
    value = literal(node)
    return value if value is not NOT_LITERAL and isinstance(value, str) else None
