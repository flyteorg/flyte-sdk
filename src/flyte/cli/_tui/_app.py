from __future__ import annotations

import functools
import html
import json
import re
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

from ._tracker import ActionNode, ActionStatus, ActionTracker, PendingCondition

_STATUS_ICON = {
    ActionStatus.RUNNING: ("●", "dodger_blue1"),
    ActionStatus.PAUSED: ("⏸", "yellow"),
    ActionStatus.SUCCEEDED: ("✓", "green"),
    ActionStatus.FAILED: ("✗", "red"),
}


def _normalize_markdown_prompt_for_tui(prompt: str) -> str:
    """Convert HTML-heavy markdown prompts into Textual-friendly markdown.

    Condition prompts may embed HTML for the web UI. The terminal Markdown widget
    ignores unknown HTML tags but still pays for their block spacing, which can
    leave large gaps (for example around ``<br>`` before a list).
    """
    text = html.unescape(prompt)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<code>(.*?)</code>", r"`\1`", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<kbd>(.*?)</kbd>", r"`\1`", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?(?:small|em|strong)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_text_prompt_for_tui(text: str) -> str:
    """Render markdown-ish condition prompts as compact plain text."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Keep bullets tight under the preceding paragraph.
    text = re.sub(r"\n\n(-\s)", r"\n\1", text)
    return text.strip()


def _condition_prompt_for_tui(prompt: str, prompt_type: str) -> tuple[str, str]:
    """Return ``(render_mode, text)`` for displaying a condition prompt in the TUI.

    ``render_mode`` is ``markdown`` or ``text``. Bullet lists are shown as plain
    text because Textual's Markdown widget adds large vertical gaps between list
    items inside narrow condition panels.
    """
    if prompt_type != "markdown":
        return "text", _format_text_prompt_for_tui(prompt)
    normalized = _normalize_markdown_prompt_for_tui(prompt)
    if re.search(r"(?m)^-\s", normalized):
        return "text", _format_text_prompt_for_tui(normalized)
    return "markdown", normalized


def _elapsed(node: ActionNode) -> str:
    if node.end_time is not None:
        return f"{node.end_time - node.start_time:.2f}s"
    return ""


def _cache_icon(node: ActionNode) -> str:
    if node.disable_run_cache:
        return "-"  # run-level override: don't show cache status
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
        attempts_suffix = f" [x{node.attempt_count}]" if node.attempt_count > 1 else ""
        label.append(f"{cache} {_display_name(node)}{attempts_suffix}{suffix}")
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


def _next_attempt_num(current: int | None, attempt_numbers: list[int], direction: int) -> int | None:
    """Return the next attempt number in the requested direction.

    direction: -1 for previous, +1 for next.
    """
    if not attempt_numbers:
        return None
    if current is None or current not in attempt_numbers:
        return attempt_numbers[-1]
    idx = attempt_numbers.index(current)
    next_idx = max(0, min(len(attempt_numbers) - 1, idx + direction))
    return attempt_numbers[next_idx]


class _LogViewer(RichLog):
    """RichLog that writes incoming Print events from Textual's stdout/stderr capture."""

    def on_print(self, event: events.Print) -> None:
        self.write(event.text)


class ActionTreeWidget(Tree[str]):
    """Left panel: navigable action tree.

    The invisible Textual root is hidden via `show_root=False`.
    The first real action becomes the visible top-level node.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("down,j", "cursor_down", "Cursor Down", show=False),
        Binding("up,k", "cursor_up", "Cursor Up", show=False),
    ]

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

    def focus_action(self, action_id: str) -> bool:
        """Move the cursor to *action_id*'s tree node. Returns True on success."""
        tree_node = self._node_map.get(action_id)
        if tree_node is None:
            return False
        self.move_cursor(tree_node)
        return True

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

    Markup is disabled so that JSON/repr content with `[brackets]` is
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


class ConditionInputPanel(Vertical):
    """Interactive panel shown when a task is paused waiting for a condition."""

    DEFAULT_CSS = """
    ConditionInputPanel {
        height: auto;
        padding: 1;
    }
    ConditionInputPanel .condition-prompt-scroll {
        max-height: 8;
        height: auto;
        margin-bottom: 1;
    }
    ConditionInputPanel .condition-prompt-scroll Markdown,
    ConditionInputPanel .condition-prompt-scroll Static {
        height: auto;
    }
    ConditionInputPanel .condition-prompt {
        height: auto;
    }
    ConditionInputPanel Markdown.condition-prompt MarkdownHeader {
        margin: 0 0 1 0;
    }
    ConditionInputPanel Markdown.condition-prompt MarkdownParagraph {
        margin: 0;
    }
    ConditionInputPanel Markdown.condition-prompt MarkdownBulletList {
        margin: 0;
    }
    ConditionInputPanel Markdown.condition-prompt MarkdownBulletList Horizontal {
        height: auto;
        margin: 0;
    }
    ConditionInputPanel .condition-description {
        color: $text-muted;
        margin-bottom: 1;
    }
    ConditionInputPanel .condition-input-row {
        height: 1;
        min-height: 1;
        layout: horizontal;
        dock: bottom;
        margin-top: 1;
    }
    ConditionInputPanel .condition-input-row Input {
        width: 1fr;
        margin-right: 1;
        height: 1;
    }
    ConditionInputPanel .condition-input-row Input.-textual-compact {
        background: $surface;
        color: $foreground;
    }
    ConditionInputPanel .condition-input-row Input.-textual-compact:focus {
        background-tint: $foreground 10%;
    }
    ConditionInputPanel .condition-buttons {
        height: auto;
        layout: horizontal;
        dock: bottom;
        margin-top: 1;
    }
    ConditionInputPanel .condition-buttons Button {
        margin-right: 1;
        color: #ffffff;
        text-style: bold;
    }
    ConditionInputPanel .condition-buttons Button:focus {
        text-style: bold;
    }
    ConditionInputPanel .condition-validation-error {
        color: red;
        height: auto;
    }
    """

    class Submitted(Message):
        """Posted when the user submits a value for a pending condition."""

        def __init__(self, action_id: str, value: Any) -> None:
            super().__init__()
            self.action_id = action_id
            self.value = value

    def __init__(self, pending: PendingCondition, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pending = pending

    def compose(self) -> ComposeResult:
        pc = self._pending
        render_mode, prompt_text = _condition_prompt_for_tui(pc.prompt, pc.prompt_type)

        if pc.description:
            yield Static(pc.description, classes="condition-description")

        if render_mode == "markdown":
            with VerticalScroll(classes="condition-prompt-scroll"):
                yield Markdown(prompt_text, classes="condition-prompt")
        else:
            yield Static(prompt_text, classes="condition-prompt", markup=False)

        if pc.data_type is bool:
            with Horizontal(classes="condition-buttons"):
                # compact=True keeps the label visible when terminals collapse 3D button height.
                yield Button("Yes", id="condition-yes", variant="success", compact=True)
                yield Button("No", id="condition-no", variant="error", compact=True)
        else:
            yield Static("", id="condition-validation-error", classes="condition-validation-error")
            with Horizontal(classes="condition-input-row"):
                # compact=True: default tall Input needs 3 rows; VS Code/Cursor collapses to 1
                # and only paints the top border, hiding the placeholder and cursor.
                yield Input(
                    placeholder=f"Enter a {pc.data_type.__name__} value...",
                    id="condition-input",
                    compact=True,
                )
                yield Button("Submit", id="condition-submit", variant="primary", compact=True)

    def on_mount(self) -> None:
        if self._pending.data_type is not bool:
            self.call_after_refresh(self._focus_value_input)

    def _focus_value_input(self) -> None:
        try:
            self.query_one("#condition-input", Input).focus()
        except Exception:
            pass

    @on(Button.Pressed, "#condition-yes")
    def _on_yes(self) -> None:
        self.post_message(self.Submitted(self._pending.action_id, True))

    @on(Button.Pressed, "#condition-no")
    def _on_no(self) -> None:
        self.post_message(self.Submitted(self._pending.action_id, False))

    @on(Button.Pressed, "#condition-submit")
    def _on_submit(self) -> None:
        self._try_submit()

    @on(Input.Submitted, "#condition-input")
    def _on_input_submitted(self) -> None:
        self._try_submit()

    def _try_submit(self) -> None:
        inp = self.query_one("#condition-input", Input)
        err_label = self.query_one("#condition-validation-error", Static)
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


class ConditionResultPanel(Vertical):
    """Read-only panel showing a resolved condition's prompt and the submitted response."""

    DEFAULT_CSS = """
    ConditionResultPanel {
        height: auto;
        padding: 1;
    }
    ConditionResultPanel .condition-prompt {
        margin-bottom: 1;
    }
    ConditionResultPanel .condition-description {
        color: $text-muted;
        margin-bottom: 1;
    }
    ConditionResultPanel .condition-response {
        color: $success;
    }
    """

    def __init__(self, pending: PendingCondition, value: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pending = pending
        self._value = value

    def compose(self) -> ComposeResult:
        pc = self._pending
        if pc.description:
            yield Static(pc.description, classes="condition-description")

        render_mode, prompt_text = _condition_prompt_for_tui(pc.prompt, pc.prompt_type)
        if render_mode == "markdown":
            yield Markdown(prompt_text, classes="condition-prompt")
        else:
            yield Static(prompt_text, classes="condition-prompt", markup=False)

        yield Static(f"response: {self._value!r}", classes="condition-response")


class DetailPanel(VerticalScroll):
    """Right panel: separate boxes for task details, inputs, outputs.

    Report box is only mounted when the selected action has a report.
    Context box is only shown when context data is present.
    """

    action_id: reactive[str | None] = reactive(None)

    def __init__(self, tracker: ActionTracker, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tracker = tracker
        # Identifies the panel currently mounted in the condition box:
        # "<action_id>|pending" or "<action_id>|resolved".
        self._current_condition_key: str | None = None
        self._selected_attempts_by_action: dict[str, int] = {}

    def compose(self) -> ComposeResult:
        yield _DetailBox(id="box-attempt-controls")
        yield _DetailBox(id="box-task-details")
        yield _DetailBox(id="box-condition")
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

    def _clear_condition_box(self, condition_box: _DetailBox) -> None:
        """Remove any mounted condition panel (interactive or read-only)."""
        if self._current_condition_key is not None:
            for input_panel in condition_box.query(ConditionInputPanel):
                input_panel.remove()
            for result_panel in condition_box.query(ConditionResultPanel):
                result_panel.remove()
            self._current_condition_key = None

    def _hide_condition_box(self, condition_box: _DetailBox) -> None:
        """Remove any mounted condition panel and hide the condition box."""
        condition_box.display = False
        self._clear_condition_box(condition_box)

    def _show_condition_panel(
        self, condition_box: _DetailBox, key: str, panel_factory: Callable[[], Any], title: str
    ) -> None:
        """Mount *panel_factory()* into the condition box unless *key* is already shown."""
        pending = key.endswith("|pending")
        if self._current_condition_key != key:
            self._clear_condition_box(condition_box)
            condition_box.border_title = title
            condition_box.update("")
            panel = panel_factory()
            condition_box.mount(panel)
            self._current_condition_key = key
        condition_box.display = True
        if pending:
            self.call_after_refresh(self._scroll_to_pending_condition, condition_box)

    def _scroll_to_pending_condition(self, condition_box: _DetailBox) -> None:
        """Keep the value input visible and focused for pending conditions."""
        try:
            input_row = condition_box.query_one(".condition-input-row")
        except Exception:
            input_row = condition_box
        self.scroll_to_widget(input_row, animate=False)
        for panel in condition_box.query(ConditionInputPanel):
            panel.call_after_refresh(panel._focus_value_input)
            break

    def _attempt_numbers_for_action(self, action_id: str) -> list[int]:
        node = self._tracker.get_action(action_id)
        if node is None or not node.attempts:
            return []
        return sorted(int(a.get("attempt_num", 0)) for a in node.attempts)

    def select_previous_attempt(self) -> bool:
        if self.action_id is None:
            return False
        attempt_numbers = self._attempt_numbers_for_action(self.action_id)
        current = self._selected_attempts_by_action.get(self.action_id)
        nxt = _next_attempt_num(current, attempt_numbers, -1)
        if nxt is None or nxt == current:
            return False
        self._selected_attempts_by_action[self.action_id] = nxt
        self._render_detail()
        return True

    def select_next_attempt(self) -> bool:
        if self.action_id is None:
            return False
        attempt_numbers = self._attempt_numbers_for_action(self.action_id)
        current = self._selected_attempts_by_action.get(self.action_id)
        nxt = _next_attempt_num(current, attempt_numbers, +1)
        if nxt is None or nxt == current:
            return False
        self._selected_attempts_by_action[self.action_id] = nxt
        self._render_detail()
        return True

    def _render_detail(self) -> None:
        try:
            attempt_box = self.query_one("#box-attempt-controls", _DetailBox)
            task_box = self.query_one("#box-task-details", _DetailBox)
            condition_box = self.query_one("#box-condition", _DetailBox)
            report_box = self.query_one("#box-report", _DetailBox)
            log_links_box = self.query_one("#box-log-links", _DetailBox)
            inputs_box = self.query_one("#box-inputs", _DetailBox)
            context_box = self.query_one("#box-context", _DetailBox)
            outputs_box = self.query_one("#box-outputs", _DetailBox)
        except Exception:
            return

        aid = self.action_id
        if aid is None:
            attempt_box.display = False
            task_box.update("Select an action to view details.")
            task_box.border_title = "Task Details"
            self._hide_condition_box(condition_box)
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
            attempt_box.display = False
            task_box.update(f"Action {aid} not found.")
            self._hide_condition_box(condition_box)
            report_box.display = False
            log_links_box.display = False
            inputs_box.update("")
            context_box.display = False
            outputs_box.update("")
            return

        attempt_box.display = bool(node.attempts)
        selected_attempt: dict[str, Any] | None = None
        if node.attempts:
            sorted_attempts = sorted(node.attempts, key=lambda a: int(a.get("attempt_num", 0)))
            attempt_numbers = [int(a.get("attempt_num", 0)) for a in sorted_attempts]

            default_attempt_num = self._selected_attempts_by_action.get(aid)
            if default_attempt_num is None:
                default_attempt_num = node.selected_attempt
            if default_attempt_num is None and node.attempts:
                default_attempt_num = int(node.attempts[-1].get("attempt_num", 0))
            if default_attempt_num is not None:
                self._selected_attempts_by_action[aid] = default_attempt_num
                selected_attempt = next(
                    (a for a in node.attempts if int(a.get("attempt_num", -1)) == default_attempt_num),
                    None,
                )
            else:
                fallback_attempt = int(node.attempts[-1].get("attempt_num", 1))
                self._selected_attempts_by_action[aid] = fallback_attempt
                selected_attempt = next(
                    (a for a in node.attempts if int(a.get("attempt_num", -1)) == fallback_attempt),
                    None,
                )
            current_num = int((selected_attempt or sorted_attempts[-1]).get("attempt_num", 1))
            total_num = len(attempt_numbers)
            attempt_box.border_title = "Attempts"
            attempt_box.update(f"[ {current_num}/{total_num} ]")

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
            self._hide_condition_box(condition_box)
            report_box.display = False
            log_links_box.display = False
            inputs_box.display = False
            context_box.display = False
            outputs_box.display = False
            attempt_box.display = False
            return

        task_box.border_title = "Task Details"
        details = []
        details.append(f"task name:  {node.task_name}")
        details.append(f"action id:  {node.action_id}")
        details.append(f"status:     {node.status.value}")
        if node.attempt_count > 0:
            details.append(f"attempts:   {node.attempt_count}")
            if selected_attempt is not None:
                details.append(f"viewing:    attempt {selected_attempt['attempt_num']}")
        elapsed = _elapsed(node)
        if elapsed:
            details.append(f"duration:   {elapsed}")
        if node.disable_run_cache:
            cache_str = "disabled (run override)"
        elif node.cache_enabled:
            cache_str = "enabled"
            if node.cache_hit:
                cache_str += " (cache hit)"
        else:
            cache_str = "disabled"
        details.append(f"cache:      {cache_str}")
        task_box.update("\n".join(details))

        # -- Condition box (interactive while paused) --
        if node.status == ActionStatus.PAUSED:
            pending = self._tracker.get_pending_condition(aid)
            if pending is not None:
                attempt_box.display = False
                task_box.display = False
                self._show_condition_panel(
                    condition_box,
                    f"{aid}|pending",
                    functools.partial(ConditionInputPanel, pending),
                    f"Condition: {pending.condition_name}",
                )
            else:
                self._hide_condition_box(condition_box)
            # Hide other detail boxes when paused
            report_box.display = False
            log_links_box.display = False
            inputs_box.display = False
            context_box.display = False
            outputs_box.display = False
            return

        # -- Condition box (read-only after the condition was resolved) --
        resolved = self._tracker.get_resolved_condition(aid)
        if resolved is not None:
            pc, value = resolved
            self._show_condition_panel(
                condition_box,
                f"{aid}|resolved",
                lambda: ConditionResultPanel(pc, value),
                f"Condition: {pc.condition_name} (signaled)",
            )
        else:
            self._hide_condition_box(condition_box)

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
        attempt_error = selected_attempt.get("error") if selected_attempt else None
        attempt_outputs = selected_attempt.get("outputs") if selected_attempt else None

        if attempt_error or node.error:
            outputs_box.border_title = "Error"
            error_parts: list[str] = []
            if node.output_path:
                error_parts.append(f"path: {node.output_path}")
                error_parts.append("")
            error_parts.append(attempt_error or node.error or "")
            outputs_box.update("\n".join(error_parts))
        else:
            outputs_box.border_title = "Outputs"
            output_parts: list[str] = []
            if node.output_path:
                output_parts.append(f"path: {node.output_path}")
                output_parts.append("")
            output_parts.append(_pretty_json(attempt_outputs if selected_attempt else node.outputs))
            outputs_box.update("\n".join(output_parts))


# Flyte brand purple palette
_FLYTE_PURPLE = "#7652a2"
_FLYTE_PURPLE_LIGHT = "#f7f5fd"
_FLYTE_PURPLE_DARK = "#171020"
_FLYTE_BORDER = "#DEDDE4"


class FlyteTUIApp(App[None]):
    """Interactive TUI for `flyte run --local --tui`."""

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
    ConditionInputPanel .condition-prompt,
    ConditionInputPanel .condition-description {{
        color: {_FLYTE_PURPLE_LIGHT};
    }}
    ConditionInputPanel .condition-buttons Button {{
        margin-right: 1;
        color: #ffffff;
        text-style: bold;
    }}
    ConditionInputPanel .condition-buttons Button:focus {{
        text-style: bold;
    }}
    ConditionInputPanel .condition-prompt-scroll {{
        max-height: 8;
        height: auto;
        margin-bottom: 1;
    }}
    ConditionInputPanel .condition-input-row {{
        height: 1;
        min-height: 1;
        layout: horizontal;
        dock: bottom;
        margin-top: 1;
    }}
    ConditionInputPanel .condition-buttons {{
        height: 1;
        min-height: 1;
        layout: horizontal;
        dock: bottom;
        margin-top: 1;
    }}
    ConditionInputPanel Markdown.condition-prompt MarkdownHeader {{
        margin: 0 0 1 0;
    }}
    ConditionInputPanel Markdown.condition-prompt MarkdownParagraph {{
        margin: 0;
    }}
    ConditionInputPanel Markdown.condition-prompt MarkdownBulletList {{
        margin: 0;
    }}
    ConditionInputPanel Input {{
        width: 1fr;
        margin-right: 1;
        height: 1;
    }}
    ConditionInputPanel Input.-textual-compact {{
        background: #1a0a2e;
        color: #ffffff;
    }}
    ConditionInputPanel Input.-textual-compact:focus {{
        background: #2a1040;
        background-tint: #ffffff 10%;
    }}
    ConditionInputPanel .condition-validation-error {{
        color: red;
    }}
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+q", "quit", "Quit", priority=True, show=False),
        Binding("d", "show_details", "Details"),
        Binding("l", "show_logs", "Logs"),
        Binding("[", "previous_attempt", "Prev Attempt"),
        Binding("]", "next_attempt", "Next Attempt"),
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
        self._seen_pending_condition_ids: set[str] = set()

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
            # Jump to a newly pending condition's sub-action so its input panel is shown.
            for pc in self._tracker.get_all_pending_conditions():
                if pc.action_id not in self._seen_pending_condition_ids:
                    self._seen_pending_condition_ids.add(pc.action_id)
                    if tree.focus_action(pc.action_id):
                        detail.action_id = pc.action_id
                    self.query_one("#right-tabs", TabbedContent).active = "tab-details"
            detail.refresh_detail()

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.action_id = event.node.data

    @on(ConditionInputPanel.Submitted)
    def _on_condition_submitted(self, event: ConditionInputPanel.Submitted) -> None:
        self._tracker.resolve_condition(event.action_id, event.value)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.SUCCESS:
            self.sub_title = "completed"
        elif event.state == WorkerState.ERROR:
            self.sub_title = "failed"

    async def action_quit(self) -> None:
        # Cancel all pending conditions so blocked threads don't hang
        for pc in self._tracker.get_all_pending_conditions():
            pc.set_result(None)
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

    def action_previous_attempt(self) -> None:
        self.query_one("#right-tabs", TabbedContent).active = "tab-details"
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.select_previous_attempt()

    def action_next_attempt(self) -> None:
        self.query_one("#right-tabs", TabbedContent).active = "tab-details"
        detail = self.query_one("#detail-panel", DetailPanel)
        detail.select_next_attempt()
