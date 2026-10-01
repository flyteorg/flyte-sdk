"""The root CLI group imports a subcommand, and the plugins that extend it, only when it is used."""

import sys
from types import SimpleNamespace

import click
import pytest

from flyte.cli import _plugins
from flyte.cli._lazy_group import LazyGroup
from flyte.cli.main import _COMMANDS, main


class _EP:
    """Stand-in for an entry point that records whether it was loaded."""

    def __init__(self, name, value):
        self.name = name
        self._value = value
        self.loaded = False
        self.dist = SimpleNamespace(name="some-plugin")

    def load(self):
        self.loaded = True
        return self._value


@pytest.fixture
def plugin_entry_points(monkeypatch):
    """Install fake plugin entry points: `{group: [entry points]}`."""
    registered = {_plugins.COMMANDS_GROUP: [], _plugins.HOOKS_GROUP: []}
    monkeypatch.setattr(_plugins, "entry_points", lambda *, group: registered[group])
    return registered


@pytest.fixture
def lazy_module(tmp_path, monkeypatch):
    """A module on sys.path that defines a `greet` command and a `things` group."""
    (tmp_path / "lazy_cli_fixture.py").write_text(
        "import click\n\n"
        "@click.command()\n"
        "def greet():\n"
        "    click.echo('hello')\n\n"
        "@click.group()\n"
        "def things():\n"
        "    pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "lazy_cli_fixture"
    sys.modules.pop("lazy_cli_fixture", None)


def _group(module_name: str) -> LazyGroup:
    return LazyGroup(name="root", lazy_commands={"greet": f"{module_name}:greet", "things": f"{module_name}:things"})


def test_listing_commands_does_not_import_them(lazy_module, plugin_entry_points):
    group = _group(lazy_module)

    assert group.list_commands(None) == ["greet", "things"]
    assert lazy_module not in sys.modules


def test_command_is_imported_on_first_use(lazy_module, plugin_entry_points):
    group = _group(lazy_module)

    command = group.get_command(None, "greet")

    assert isinstance(command, click.Command)
    assert lazy_module in sys.modules
    assert group.get_command(None, "greet") is command


def test_unknown_command_resolves_to_none(lazy_module, plugin_entry_points):
    assert _group(lazy_module).get_command(None, "nope") is None


def test_plugin_command_is_listed_but_loaded_only_when_used(lazy_module, plugin_entry_points):
    extra = _EP("extra", click.Command("extra"))
    plugin_entry_points[_plugins.COMMANDS_GROUP].append(extra)
    group = _group(lazy_module)

    assert "extra" in group.list_commands(None)
    group.get_command(None, "greet")
    assert not extra.loaded

    command = group.get_command(None, "extra")
    assert extra.loaded
    assert _plugins.get_command_distribution(command) == "some-plugin"


def test_plugin_subcommands_are_attached_when_their_group_is_used(lazy_module, plugin_entry_points):
    subs = [_EP("things.widget", click.Command("widget")), _EP("things.gadget", click.Command("gadget"))]
    plugin_entry_points[_plugins.COMMANDS_GROUP].extend(subs)
    group = _group(lazy_module)

    group.get_command(None, "greet")
    assert not any(sub.loaded for sub in subs)

    things = group.get_command(None, "things")
    assert all(sub.loaded for sub in subs)
    assert {"widget", "gadget"} <= set(things.commands)


def test_hooks_for_several_subcommands_are_all_applied(lazy_module, plugin_entry_points):
    plugin_entry_points[_plugins.COMMANDS_GROUP].extend(
        [_EP("things.widget", click.Command("widget")), _EP("things.gadget", click.Command("gadget"))]
    )
    replaced = {"widget": click.Command("widget"), "gadget": click.Command("gadget")}
    plugin_entry_points[_plugins.HOOKS_GROUP].extend(
        [_EP(f"things.{name}", lambda command, name=name: replaced[name]) for name in replaced]
    )

    things = _group(lazy_module).get_command(None, "things")

    assert things.commands["widget"] is replaced["widget"]
    assert things.commands["gadget"] is replaced["gadget"]


def test_hook_is_applied_when_its_command_is_used(lazy_module, plugin_entry_points):
    replacement = click.Command("greet")
    hook = _EP("greet", lambda command: replacement)
    plugin_entry_points[_plugins.HOOKS_GROUP].append(hook)
    group = _group(lazy_module)

    group.get_command(None, "things")
    assert not hook.loaded

    assert group.get_command(None, "greet") is replacement


def test_every_registered_command_resolves():
    """A typo in `_COMMANDS` would otherwise only surface when that command is invoked."""
    for name in _COMMANDS:
        command = main.get_command(None, name)
        assert isinstance(command, click.Command), name
        assert command.name == name
