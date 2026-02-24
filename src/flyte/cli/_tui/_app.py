from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, ClassVar

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Input, Markdown, RichLog, Static, TabbedContent, TabPane, Tree
from textual.widgets.tree import TreeNode
from textual.worker import Worker, WorkerState

from ._tracker import ActionNode, ActionStatus, ActionTracker, PendingEvent

_STATUS_ICON = {
    ActionStatus.RUNNING: ("●", "dodger_blue1"),
    ActionStatus.PAUSED: ("⏸", "yellow"),
    ActionStatus.SUCCEEDED: ("✓", "green"),
    ActionStatus.FAILED: ("✗", "red"),
}


def _elapsed(node: ActionNode) -> str:
    if node.end_time is not None:
        return f"{node.end_time - node.start_time:.2f}s"
    return ""


def _cache_icon(node: ActionNode) -> str:
    if node.cache_hit:
        return " $"  # cache hit
    if node.cache_enabled:
        return " ~"  # cache enabled but miss
    return ""


def _display_name(node: ActionNode) -> str:
    return node.short_name or node.task_name


def _is_group_node(node: ActionNode) -> bool:
    return node.action_id.startswith("__group__")


def _label(node: ActionNode, children_map: dict[str, list[str]] | None = None) -> Text:
    icon_char, icon_color = _STATUS_ICON[node.status]
    label = Text()
    label.append(icon_char, style=icon_color)

    if _is_group_node(node):
        count = len(children_map.get(node.action_id, [])) if children_map else 0
        elapsed = _elapsed(node)
        suffix = f" ({elapsed})" if elapsed else ""
        label.append(f" {_display_name(node)} [{count}]{suffix}")
    else:
        cache = _cache_icon(node)
        elapsed = _elapsed(node)
        suffix = f" ({elapsed})" if elapsed else ""
        label.append(f"{cache} {_display_name(node)}{suffix}")
    return label


def _pretty_json(obj: Any) -> str:
    if obj is None:
        return "(none)"
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, indent=2, default=repr)
    except (TypeError, ValueError):
        return repr(obj)


class _LogViewer(RichLog):
    """RichLog that writes incoming Print events from Textual's stdout/stderr capture."""

    def on_print(self, event: events.Print) -> None:
        self.write(event.text)


class ActionTreeWidget(Tree[str]):
    """Left panel: navigable action tree.

    The invisible Textual root is hidden via ``show_root=False``.
    The first real action becomes the visible top-level node.
    """

    def __init__(self, tracker: ActionTracker, **kwargs: Any) -> None:
        super().__init__("Actions", **kwargs)
        self.show_root = False
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
                tree_node.set_label(_label(action, children))

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
            tree_node = parent.add(_label(action, children), data=action_id, expand=True)
            self._node_map[action_id] = tree_node
        for child_id in children.get(action_id, []):
            self._sync_node(child_id, self._node_map[action_id], children, nodes)


class _DetailBox(Static):
    """A bordered box used inside the detail panel.

    Markup is disabled so that JSON/repr content with ``[brackets]`` is
    rendered literally instead of being parsed as Rich markup tags.
    """

    DEFAULT_CSS = """
    _DetailBox {
        border: solid $accent;
        padding: 0 1;
        margin-bottom: 1;
        height: auto;
    }
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("markup", False)
        super().__init__(*args, **kwargs)


class EventInputPanel(Vertical):
    """Interactive panel shown when a task is paused waiting for an event."""

    DEFAULT_CSS = """
    EventInputPanel {
        height: auto;
        padding: 1;
    }
    EventInputPanel .event-prompt {
        margin-bottom: 1;
    }
    EventInputPanel .event-description {
        color: $text-muted;
        margin-bottom: 1;
    }
    EventInputPanel .event-buttons {
        height: auto;
        layout: horizontal;
    }
    EventInputPanel .event-buttons Button {
        margin-right: 1;
    }
    EventInputPanel .event-input-row {
        height: auto;
        layout: horizontal;
    }
    EventInputPanel .event-input-row Input {
        width: 1fr;
        margin-right: 1;
    }
    EventInputPanel .event-validation-error {
        color: red;
        height: auto;
    }
    """

    class Submitted(Message):
        """Posted when the user submits a value for a pending event."""

        def __init__(self, action_id: str, value: Any) -> None:
            super().__init__()
            self.action_id = action_id
            self.value = value

    def __init__(self, pending: PendingEvent, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pending = pending

    def compose(self) -> ComposeResult:
        pe = self._pending
        if pe.description:
            yield Static(pe.description, classes="event-description")

        if pe.prompt_type == "markdown":
            yield Markdown(pe.prompt, classes="event-prompt")
        else:
            yield Static(pe.prompt, classes="event-prompt")

        if pe.data_type is bool:
            with Horizontal(classes="event-buttons"):
                yield Button("Yes", id="event-yes", variant="success")
                yield Button("No", id="event-no", variant="error")
        else:
            type_name = pe.data_type.__name__
            with Horizontal(classes="event-input-row"):
                yield Input(placeholder=f"Enter a {type_name} value...", id="event-input")
                yield Button("Submit", id="event-submit", variant="primary")
            yield Static("", id="event-validation-error", classes="event-validation-error")

    @on(Button.Pressed, "#event-yes")
    def _on_yes(self) -> None:
        self.post_message(self.Submitted(self._pending.action_id, True))

    @on(Button.Pressed, "#event-no")
    def _on_no(self) -> None:
        self.post_message(self.Submitted(self._pending.action_id, False))

    @on(Button.Pressed, "#event-submit")
    def _on_submit(self) -> None:
        self._try_submit()

    @on(Input.Submitted, "#event-input")
    def _on_input_submitted(self) -> None:
        self._try_submit()

    def _try_submit(self) -> None:
        inp = self.query_one("#event-input", Input)
        err_label = self.query_one("#event-validation-error", Static)
        raw = inp.value.strip()
        if not raw:
            err_label.update("Value cannot be empty.")
            return
        try:
            value = self._pending.data_type(raw)
        except (ValueError, TypeError):
            type_name = self._pending.data_type.__name__
            err_label.update(f"Please enter a valid {type_name}.")
            return
        err_label.update("")
        self.post_message(self.Submitted(self._pending.action_id, value))


class DetailPanel(VerticalScroll):
    """Right panel: separate boxes for task details, inputs, outputs.

    Report box is only mounted when the selected action has a report.
    Context box is only shown when context data is present.
    """

    action_id: reactive[str | None] = reactive(None)

    def __init__(self, tracker: ActionTracker, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tracker = tracker
        self._current_event_action_id: str | None = None

    def compose(self) -> ComposeResult:
        yield _DetailBox(id="box-task-details")
        yield _DetailBox(id="box-event")
        yield _DetailBox(id="box-report")
        yield _DetailBox(id="box-log-links")
        yield _DetailBox(id="box-inputs")
        yield _DetailBox(id="box-context")
        yield _DetailBox(id="box-outputs")

    def on_mount(self) -> None:
        self._render_detail()

    def watch_action_id(self, new_val: str | None) -> None:
        self._render_detail()

    def refresh_detail(self) -> None:
        self._render_detail()

    def _hide_event_box(self, event_box: _DetailBox) -> None:
        """Remove any mounted EventInputPanel and hide the event box."""
        event_box.display = False
        if self._current_event_action_id is not None:
            for child in event_box.query(EventInputPanel):
                child.remove()
            self._current_event_action_id = None

    def _render_detail(self) -> None:
        try:
            task_box = self.query_one("#box-task-details", _DetailBox)
            event_box = self.query_one("#box-event", _DetailBox)
            report_box = self.query_one("#box-report", _DetailBox)
            log_links_box = self.query_one("#box-log-links", _DetailBox)
            inputs_box = self.query_one("#box-inputs", _DetailBox)
            context_box = self.query_one("#box-context", _DetailBox)
            outputs_box = self.query_one("#box-outputs", _DetailBox)
        except Exception:
            return

        aid = self.action_id
        if aid is None:
            task_box.update("Select an action to view details.")
            task_box.border_title = "Task Details"
            self._hide_event_box(event_box)
            report_box.display = False
            log_links_box.display = False
            inputs_box.update("")
            inputs_box.border_title = "Inputs"
            context_box.display = False
            outputs_box.update("")
            outputs_box.border_title = "Outputs"
            return

        node = self._tracker.get_action(aid)
        if node is None:
            task_box.update(f"Action {aid} not found.")
            self._hide_event_box(event_box)
            report_box.display = False
            log_links_box.display = False
            inputs_box.update("")
            context_box.display = False
            outputs_box.update("")
            return

        # -- Task Details box --
        if _is_group_node(node):
            task_box.border_title = "Group Details"
            details: list[str] = []
            details.append(f"group:   {node.task_name}")
            details.append(f"status:  {node.status.value}")
            elapsed = _elapsed(node)
            if elapsed:
                details.append(f"duration:   {elapsed}")
            task_box.update("\n".join(details))
            self._hide_event_box(event_box)
            report_box.display = False
            log_links_box.display = False
            inputs_box.display = False
            context_box.display = False
            outputs_box.display = False
            return

        task_box.border_title = "Task Details"
        details = []
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

        # -- Event box (only when paused) --
        if node.status == ActionStatus.PAUSED:
            pe = self._tracker.get_pending_event(aid)
            if pe is not None and self._current_event_action_id != aid:
                # Remove any old event panel and mount a new one
                self._hide_event_box(event_box)
                event_box.border_title = f"Event: {pe.event_name}"
                event_box.display = True
                event_box.mount(EventInputPanel(pe))
                self._current_event_action_id = aid
            elif pe is not None:
                # Already showing the right panel, just make sure it's visible
                event_box.display = True
            else:
                self._hide_event_box(event_box)
            # Hide other detail boxes when paused
            report_box.display = False
            log_links_box.display = False
            inputs_box.display = False
            context_box.display = False
            outputs_box.display = False
            return

        # Not paused — hide event box
        self._hide_event_box(event_box)

        # -- Report box (only when available) --
        if node.has_report and node.output_path:
            report_box.border_title = "Report"
            report_box.update(f"{node.output_path}/report.html")
            report_box.display = True
        else:
            report_box.display = False

        # -- Log Links box (only when available) --
        if node.log_links:
            log_links_box.border_title = "Log Links"
            log_links_box.update("\n".join(f"{name}: {uri}" for name, uri in node.log_links))
            log_links_box.display = True
        else:
            log_links_box.display = False

        # -- Inputs box --
        inputs_box.display = True
        inputs_box.border_title = "Inputs"
        input_parts: list[str] = []
        if node.output_path:
            input_parts.append(f"path: {node.output_path}")
            input_parts.append("")
        input_parts.append(_pretty_json(node.inputs))
        inputs_box.update("\n".join(input_parts))

        # -- Context box (only when available) --
        if node.context:
            context_box.border_title = "Context"
            context_box.update(_pretty_json(node.context))
            context_box.display = True
        else:
            context_box.display = False

        # -- Outputs / Error box --
        outputs_box.display = True
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


# Flyte brand purple palette
_FLYTE_PURPLE = "#7652a2"
_FLYTE_PURPLE_LIGHT = "#f7f5fd"
_FLYTE_PURPLE_DARK = "#171020"
_FLYTE_BORDER = "#DEDDE4"


class FlyteTUIApp(App[None]):
    """Interactive TUI for ``flyte run --local --tui``."""

    CSS = f"""
    Screen {{
        background: {_FLYTE_PURPLE_DARK};
    }}
    Header {{
        background: {_FLYTE_PURPLE};
        color: {_FLYTE_PURPLE_LIGHT};
    }}
    Footer {{
        background: {_FLYTE_PURPLE};
        color: {_FLYTE_PURPLE_LIGHT};
    }}
    Horizontal {{
        height: 1fr;
    }}
    ActionTreeWidget {{
        width: 1fr;
        min-width: 30;
        border: solid {_FLYTE_PURPLE};
        border-title-color: {_FLYTE_PURPLE_LIGHT};
        background: {_FLYTE_PURPLE_DARK};
        color: {_FLYTE_PURPLE_LIGHT};
    }}
    #right-tabs {{
        width: 2fr;
    }}
    DetailPanel {{
        background: {_FLYTE_PURPLE_DARK};
    }}
    #log-viewer {{
        background: {_FLYTE_PURPLE_DARK};
        color: {_FLYTE_PURPLE_LIGHT};
    }}
    TabPane {{
        padding: 0;
    }}
    Tabs {{
        background: {_FLYTE_PURPLE_DARK};
        color: {_FLYTE_PURPLE_LIGHT};
    }}
    Tab {{
        background: {_FLYTE_PURPLE_DARK};
        color: {_FLYTE_PURPLE_LIGHT};
    }}
    Tab.-active {{
        background: {_FLYTE_PURPLE};
        color: {_FLYTE_PURPLE_LIGHT};
    }}
    Underline {{
        color: {_FLYTE_PURPLE};
    }}
    _DetailBox {{
        border: solid {_FLYTE_PURPLE};
        border-title-color: {_FLYTE_PURPLE_LIGHT};
        padding: 0 1;
        margin-bottom: 1;
        height: auto;
        color: {_FLYTE_PURPLE_LIGHT};
    }}
    EventInputPanel {{
        color: {_FLYTE_PURPLE_LIGHT};
    }}
    EventInputPanel Button {{
        margin-right: 1;
    }}
    EventInputPanel Input {{
        width: 1fr;
        margin-right: 1;
    }}
    EventInputPanel .event-validation-error {{
        color: red;
    }}
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+q", "quit", "Quit", priority=True, show=False),
        Binding("d", "show_details", "Details"),
        Binding("l", "show_logs", "Logs"),
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
            tree = ActionTreeWidget(self._tracker, id="action-tree")
            tree.border_title = "Actions"
            yield tree
            with TabbedContent(initial="tab-details", id="right-tabs"):
                with TabPane("Details", id="tab-details"):
                    yield DetailPanel(self._tracker, id="detail-panel")
                with TabPane("Logs", id="tab-logs"):
                    yield _LogViewer(
                        id="log-viewer",
                        auto_scroll=True,
                        wrap=True,
                        markup=False,
                        highlight=False,
                        max_lines=10_000,
                    )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Flyte Local Run"
        self.sub_title = "running..."
        log_viewer = self.query_one("#log-viewer", _LogViewer)
        self.begin_capture_print(log_viewer)
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

    @on(EventInputPanel.Submitted)
    def _on_event_submitted(self, event: EventInputPanel.Submitted) -> None:
        self._tracker.resolve_event(event.action_id, event.value)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.SUCCESS:
            self.sub_title = "completed"
        elif event.state == WorkerState.ERROR:
            self.sub_title = "failed"

    def action_quit(self) -> None:
        # Cancel all pending events so blocked threads don't hang
        for pe in self._tracker.get_all_pending_events():
            pe.set_result(None)
        self.exit()
        # The execution worker may be blocked on synchronous calls (via
        # syncify) that Textual's worker cancellation cannot interrupt.
        # Schedule a hard exit as a safety net so the terminal is not
        # left in a broken state.
        import os
        import threading
        import time

        def _force_exit() -> None:
            time.sleep(2)
            os._exit(0)

        threading.Thread(target=_force_exit, daemon=True).start()

    def action_show_details(self) -> None:
        self.query_one("#right-tabs", TabbedContent).active = "tab-details"

    def action_show_logs(self) -> None:
        self.query_one("#right-tabs", TabbedContent).active = "tab-logs"
