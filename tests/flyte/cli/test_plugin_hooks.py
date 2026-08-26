"""CLI plugin hooks wrap commands; they must not invoke them at registration time."""

from __future__ import annotations

import rich_click as click

from flyte.cli._plugins import _apply_hook_to_subcommand


def _root() -> click.Group:
    calls: list[str] = []

    @click.group()
    def root():
        pass

    @click.group()
    def get():
        pass

    @get.command("run")
    def get_run():
        calls.append("invoked")

    root.add_command(get)
    root.calls = calls  # type: ignore[attr-defined]
    return root


def test_hook_does_not_invoke_the_command_it_wraps() -> None:
    """Regression: registration used to call `original_command.callback()`, running the wrapped
    command once during plugin discovery -- before any arguments were parsed."""
    root = _root()

    def hook(command: click.Command) -> click.Command:
        return command

    _apply_hook_to_subcommand(root, "get", "run", hook)
    assert root.calls == []  # type: ignore[attr-defined]


def test_hook_replaces_the_command() -> None:
    root = _root()
    replacement = click.Command("run")

    _apply_hook_to_subcommand(root, "get", "run", lambda cmd: replacement)
    assert root.commands["get"].commands["run"] is replacement  # type: ignore[attr-defined]


def test_failing_hook_restores_the_original() -> None:
    root = _root()
    original = root.commands["get"].commands["run"]  # type: ignore[attr-defined]

    def boom(command: click.Command) -> click.Command:
        raise RuntimeError("bad hook")

    _apply_hook_to_subcommand(root, "get", "run", boom)
    assert root.commands["get"].commands["run"] is original  # type: ignore[attr-defined]
    assert root.calls == []  # type: ignore[attr-defined]


def test_hook_can_wrap_invoke() -> None:
    """The documented pattern: wrap `invoke` so Click's full machinery still runs."""
    root = _root()
    order: list[str] = []

    def hook(command: click.Command) -> click.Command:
        original_invoke = command.invoke

        def wrapper(ctx):
            order.append("before")
            result = original_invoke(ctx)
            order.append("after")
            return result

        command.invoke = wrapper  # type: ignore[method-assign]
        return command

    _apply_hook_to_subcommand(root, "get", "run", hook)
    assert order == []

    from click.testing import CliRunner

    CliRunner().invoke(root, ["get", "run"])
    assert order == ["before", "after"]
    assert root.calls == ["invoked"]  # type: ignore[attr-defined]
