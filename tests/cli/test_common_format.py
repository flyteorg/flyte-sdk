from rich.console import Console

import flyte.cli._common as common


class _Row:
    def __init__(self, task_name: str, name: str, active: bool):
        self._task_name, self._name, self._active = task_name, name, active

    def __rich_repr__(self):
        yield "task_name", self._task_name
        yield "name", self._name
        yield "automation", "Schedule: cron 0 * * * * (UTC), kickoff arg kickoff_time"
        yield "auto_activate", self._active


def _render(monkeypatch, width: int) -> str:
    console = Console(width=width, force_terminal=False, record=True)
    monkeypatch.setattr(common, "get_console", lambda: console)
    rows = [
        _Row("hello_world.main_pipeline_task", "hourly_training_trigger", False),
        _Row("short.task", "daily", True),
    ]
    console.print(common.format("Triggers", rows))
    return console.export_text()


def test_short_values_are_not_folded_in_a_narrow_table(monkeypatch):
    # Regression: in a narrow terminal the boolean column split "False" across two lines.
    out = _render(monkeypatch, 76)
    assert "False" in out
    assert "True" in out
    truncated = "False"[:-1]
    assert f"{truncated} " not in out and f"{truncated}│" not in out


def test_wide_table_keeps_headers_on_one_line(monkeypatch):
    out = _render(monkeypatch, 160)
    assert "Auto_activate" in out
    assert "False" in out
