from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static, Tree
from textual.widgets.tree import TreeNode
from textual.worker import Worker, WorkerState

from ._tracker import ActionNode, ActionStatus, ActionTracker

_STATUS_ICON = {
    ActionStatus.RUNNING: ">",
    ActionStatus.SUCCEEDED: "ok",
    ActionStatus.FAILED: "FAIL",
}


def _elapsed(node: ActionNode) -> str:
    if node.end_time is not None:
        return f"{node.end_time - node.start_time:.2f}s"
    return ""


def _label(node: ActionNode) -> str:
    icon = _STATUS_ICON[node.status]
    elapsed = _elapsed(node)
    suffix = f" ({elapsed})" if elapsed else ""
    return f"[{icon}] {node.task_name}{suffix}"


def _pretty_json(obj: Any) -> str:
    if obj is None:
        return "(none)"
    try:
        return json.dumps(obj, indent=2, default=repr)
    except (TypeError, ValueError):
        return repr(obj)


class ActionTreeWidget(Tree[str]):
    """Left panel: navigable tree rebuilt from tracker snapshots."""

    def __init__(self, tracker: ActionTracker, **kwargs: Any) -> None:
        super().__init__("Actions", **kwargs)
        self._tracker = tracker
        self._node_map: dict[str, TreeNode[str]] = {}

    def on_mount(self) -> None:
        self.root.expand()

    def refresh_from_tracker(self) -> None:
        root_ids, children, nodes = self._tracker.snapshot()

        for rid in root_ids:
            self._sync_node(rid, self.root, children, nodes)

        for action_id, tree_node in list(self._node_map.items()):
            action = nodes.get(action_id)
            if action is not None:
                tree_node.set_label(_label(action))

    def _sync_node(
        self,
        action_id: str,
        parent: TreeNode[str],
        children: dict[str, list[str]],
        nodes: dict[str, ActionNode],
    ) -> None:
        if action_id not in self._node_map:
            action = nodes.get(action_id)
            if action is None:
                return
            tree_node = parent.add(_label(action), data=action_id, expand=True)
            self._node_map[action_id] = tree_node
        for child_id in children.get(action_id, []):
            self._sync_node(child_id, self._node_map[action_id], children, nodes)


class _DetailBox(Static):
    """A bordered box used inside the detail panel."""

    DEFAULT_CSS = """
    _DetailBox {
        border: solid $accent;
        padding: 0 1;
        margin-bottom: 1;
        height: auto;
    }
    """


class DetailPanel(VerticalScroll):
    """Right panel: separate boxes for task details, report, inputs, outputs."""

    action_id: reactive[str | None] = reactive(None)

    def __init__(self, tracker: ActionTracker, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tracker = tracker

    def compose(self) -> ComposeResult:
        yield _DetailBox(id="box-task-details")
        yield _DetailBox(id="box-report")
        yield _DetailBox(id="box-inputs")
        yield _DetailBox(id="box-outputs")

    def on_mount(self) -> None:
        self._render_detail()

    def watch_action_id(self, new_val: str | None) -> None:
        self._render_detail()

    def refresh_detail(self) -> None:
        self._render_detail()

    def _render_detail(self) -> None:
        try:
            task_box = self.query_one("#box-task-details", _DetailBox)
            report_box = self.query_one("#box-report", _DetailBox)
            inputs_box = self.query_one("#box-inputs", _DetailBox)
            outputs_box = self.query_one("#box-outputs", _DetailBox)
        except Exception:
            return

        aid = self.action_id
        if aid is None:
            task_box.update("Select an action to view details.")
            report_box.update("")
            inputs_box.update("")
            outputs_box.update("")
            task_box.border_title = "Task Details"
            report_box.border_title = "Report"
            inputs_box.border_title = "Inputs"
            outputs_box.border_title = "Outputs"
            return

        node = self._tracker.get_action(aid)
        if node is None:
            task_box.update(f"Action {aid} not found.")
            report_box.update("")
            inputs_box.update("")
            outputs_box.update("")
            return

        # -- Task Details box --
        task_box.border_title = "Task Details"
        details: list[str] = []
        details.append(f"task name:  {node.task_name}")
        details.append(f"action id:  {node.action_id}")
        details.append(f"status:     {node.status.value}")
        elapsed = _elapsed(node)
        if elapsed:
            details.append(f"duration:   {elapsed}")
        cache_str = "enabled" if node.cache_enabled else "disabled"
        if node.cache_hit:
            cache_str += " (cache hit)"
        details.append(f"cache:      {cache_str}")
        task_box.update("\n".join(details))

        # -- Report box --
        report_box.border_title = "Report"
        if node.has_report and node.output_path:
            report_box.update(f"{node.output_path}/report.html")
        else:
            report_box.update("(no report)")

        # -- Inputs box --
        inputs_box.border_title = "Inputs"
        input_parts: list[str] = []
        if node.output_path:
            input_parts.append(f"path: {node.output_path}")
            input_parts.append("")
        input_parts.append(_pretty_json(node.inputs))
        inputs_box.update("\n".join(input_parts))

        # -- Outputs / Error box --
        if node.error:
            outputs_box.border_title = "Error"
            error_parts: list[str] = []
            if node.output_path:
                error_parts.append(f"path: {node.output_path}")
                error_parts.append("")
            error_parts.append(node.error)
            outputs_box.update("\n".join(error_parts))
        else:
            outputs_box.border_title = "Outputs"
            output_parts: list[str] = []
            if node.output_path:
                output_parts.append(f"path: {node.output_path}")
                output_parts.append("")
            output_parts.append(_pretty_json(node.outputs))
            outputs_box.update("\n".join(output_parts))


class FlyteTUIApp(App[None]):
    """Interactive TUI for ``flyte run --local --tui``."""

    CSS = """
    Horizontal {
        height: 1fr;
    }
    ActionTreeWidget {
        width: 1fr;
        min-width: 30;
        border: solid green;
    }
    DetailPanel {
        width: 2fr;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        tracker: ActionTracker,
        execute_fn: Callable[[], Awaitable[Any]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._tracker = tracker
        self._execute_fn = execute_fn
        self._last_version: int = -1

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield ActionTreeWidget(self._tracker, id="action-tree")
            yield DetailPanel(self._tracker, id="detail-panel")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Flyte Local Run"
        self.sub_title = "running..."
        self._run_execution()
        self.set_interval(0.2, self._poll_tracker)

    def _run_execution(self) -> Worker[Any]:
        async def _run_in_thread() -> Any:
            return await self._execute_fn()

        return self.run_worker(_run_in_thread, thread=True)

    def _poll_tracker(self) -> None:
        v = self._tracker.version
        if v != self._last_version:
            self._last_version = v
            tree = self.query_one("#action-tree", ActionTreeWidget)
            tree.refresh_from_tracker()
            detail = self.query_one("#detail-panel", DetailPanel)
            detail.refresh_detail()

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.action_id = event.node.data

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.SUCCESS:
            self.sub_title = "completed"
        elif event.state == WorkerState.ERROR:
            self.sub_title = "failed"
