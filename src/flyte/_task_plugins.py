from __future__ import annotations

import typing
from typing import Type

import rich.repr

if typing.TYPE_CHECKING:
    from ._task import AsyncFunctionTaskTemplate

T = typing.TypeVar("T", bound="AsyncFunctionTaskTemplate")


class _Registry:
    """
    A registry for task plugins.
    """

    def __init__(self: _Registry):
        self._plugins: typing.Dict[Type, Type[typing.Any]] = {}

    def register(self, config_type: Type, plugin: Type[T]):
        """
        Register a plugin.
        """
        self._plugins[config_type] = plugin

    def find(self, config_type: Type) -> typing.Optional[Type[T]]:
        """
        Get a plugin by name.
        """
        return self._plugins.get(config_type)

    def list_plugins(self):
        """
        List all registered plugins.
        """
        return list(self._plugins.keys())

    def __rich_repr__(self) -> "rich.repr.Result":
        yield from (("Name", i) for i in self.list_plugins())

    def __repr__(self):
        return f"TaskPluginRegistry(plugins={self.list_plugins()})"


TaskPluginRegistry = _Registry()
