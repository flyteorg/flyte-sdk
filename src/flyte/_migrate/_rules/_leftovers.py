from __future__ import annotations

from typing import Union

import libcst as cst

from .._cst import Rule
from .._local_modules import local_module_path
from .._result import TODO_MARKER

_MESSAGES = {
    "conditional": "conditional() has no v2 equivalent; rewrite it as a Python if/elif/else inside the task",
    "approve": "gate nodes become flyte.new_condition(...) in v2; rewrite this approval",
    "wait_for_input": "gate nodes become flyte.new_condition(...) in v2; rewrite this signal",
    "sleep": "use `await flyte.durable.sleep(...)` inside an async task in v2",
    "reference_task": "reference a deployed v2 task with flyte.remote.Task.get(...)",
    "reference_workflow": "v2 has no workflows; reference the migrated entrypoint task with flyte.remote.Task.get(...)",
    "reference_launch_plan": "v2 has no launch plans; reference the task with flyte.remote.Task.get(...)",
    "get_reference_entity": "reference a deployed v2 task with flyte.remote.Task.get(...)",
    "LaunchPlan": "only module-level LaunchPlans with a schedule are converted (to flyte.Trigger)",
    "Deck": "use flyte.report (and `report=True` on the task) instead of Deck",
    "WorkflowFailurePolicy": "WorkflowFailurePolicy has no v2 equivalent; handle failures with try/except",
    "Workflow": "imperative workflows have no v2 equivalent; rewrite this as a task that calls other tasks",
    "ImperativeWorkflow": "imperative workflows have no v2 equivalent; rewrite this as a task that calls other tasks",
    "Email": "notifications attach to flyte.Trigger(notifications=...) using flyte.notify.Email",
    "Slack": "notifications attach to flyte.Trigger(notifications=...) using flyte.notify.Slack",
    "PagerDuty": "v2 has no PagerDuty notifier; use flyte.notify.Webhook",
    "FlyteRemote": "use flyte.init_from_config() and the flyte.remote API",
    "ImageSpec": "convert this ImageSpec to flyte.Image",
    "ContainerTask": "use flyte.extras.ContainerTask; its arguments differ from flytekit's",
    "Checkpoint": "use flyte.Checkpoint / flyte.latest_checkpoint; the API differs from flytekit's",
    "map_task": "convert this map_task to flyte.map(task, *iterables)",
}


class LeftoversRule(Rule):
    """Flag every flytekit reference no other rule converted, and imports of local modules that still use flytekit."""

    id = "leftovers"
    description = "TODOs for unconverted flytekit usage"

    def visit_Import(self, node: cst.Import) -> bool:
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        path = local_module_path(self.ctx.source_path, node)
        if path is not None and "flytekit" in path.read_text(errors="ignore"):
            module = "." * len(node.relative) + (cst.Module([]).code_for_node(node.module) if node.module else "")
            self.todo(
                f"`{module}` still uses flytekit; run `flyte migrate {path.name}` and import from its "
                f"`{path.stem}{self.ctx.suffix}` module"
            )
        return False

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> bool:
        # An earlier rule already explained what to do with this statement.
        return not any(line.comment and TODO_MARKER in line.comment.value for line in node.leading_lines)

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        return not self._flag(node)

    def visit_Name(self, node: cst.Name) -> None:
        self._flag(node)

    def _flag(self, node: Union[cst.Name, cst.Attribute]) -> bool:
        name = self.flytekit_name(node)
        if name is None or name == "flytekit":
            return False
        # `flytekit.LaunchPlan.get_or_create` is about LaunchPlan; `flytekit.core.gate.sleep` is about sleep.
        symbol = next((part for part in name.split(".")[1:] if part in _MESSAGES), None)
        self.todo(_MESSAGES[symbol] if symbol else f"`{name}` has no automatic v2 mapping")
        return True
