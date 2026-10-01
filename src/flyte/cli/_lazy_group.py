from __future__ import annotations

import importlib
from typing import Mapping

import rich_click as click

from . import _plugins


class LazyGroup(click.RichGroup):
    """A command group that imports each subcommand the first time it is used.

    Importing every subcommand up front makes each invocation pay for all of them: `flyte run`
    would import the code behind `flyte get`, `flyte serve` and every installed plugin before
    doing any work. Instead the group is told where each subcommand lives, as
    `{"name": "module:attribute"}`, and imports one only when click asks for it. Whatever
    plugins contribute to a command (see `_plugins`) is attached at the same moment.

    Listing the commands needs only their names. Rendering `--help` for the group loads all of
    them, because the summary of each command lives on the command itself.
    """

    def __init__(self, *args, lazy_commands: Mapping[str, str], **kwargs):
        super().__init__(*args, **kwargs)
        self._lazy_commands = dict(lazy_commands)
        self._resolved: set[str] = set()

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*self.commands, *self._lazy_commands, *_plugins.plugin_command_names()})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if cmd_name not in self._resolved:
            self._resolved.add(cmd_name)
            location = self._lazy_commands.get(cmd_name)
            if location is not None:
                module_name, attribute = location.split(":")
                self.add_command(getattr(importlib.import_module(module_name), attribute), name=cmd_name)
            _plugins.register_plugins(self, cmd_name)
        return self.commands.get(cmd_name)
