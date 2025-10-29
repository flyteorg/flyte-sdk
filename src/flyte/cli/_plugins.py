"""CLI Plugin System for Flyte.

This module provides a plugin system that allows external packages to:
1. Register new top-level CLI commands (e.g., flyte my-command)
2. Register new subcommands in existing groups (e.g., flyte get my-object)
3. Modify behavior of existing commands via hooks

Plugins are discovered via Python entry points.

Entry Point Groups:
- flyte.plugins.cli.commands: Register new commands
  - Entry point name "foo" -> flyte foo
  - Entry point name "get.bar" -> flyte get bar (adds subcommand to get group)

- flyte.plugins.cli.hooks: Modify existing commands
  - Entry point name "run" -> modifies flyte run
  - Entry point name "get.project" -> modifies flyte get project

Example Plugin Package:
    # In your-plugin/pyproject.toml
    [project.entry-points."flyte.plugins.cli.commands"]
    my-command = "your_plugin.cli:my_command"
    get.my-object = "your_plugin.cli:get_my_object"

    [project.entry-points."flyte.plugins.cli.hooks"]
    run = "your_plugin.hooks:modify_run"

    # In your-plugin/your_plugin/cli.py
    import rich_click as click

    @click.command()
    def my_command():
        '''My custom top-level command.'''
        click.echo("Hello from plugin!")

    @click.command()
    def get_my_object():
        '''Get my custom object.'''
        click.echo("Getting my object...")

    # In your-plugin/your_plugin/hooks.py
    def modify_run(command):
        '''Add behavior to flyte run command.'''
        original_callback = command.callback
        def wrapper(*args, **kwargs):
            # Do something before
            result = original_callback(*args, **kwargs)
            # Do something after
            return result
        command.callback = wrapper
        return command
"""

from importlib.metadata import entry_points
from typing import Callable

import rich_click as click

from flyte._logging import logger

# Type alias for command hooks
CommandHook = Callable[[click.Command], click.Command]


def discover_and_register_plugins(root_group: click.Group):
    """
    Discover all CLI plugins from installed packages and register them.

    This function:
    1. Discovers command plugins and adds them to the CLI
    2. Discovers hook plugins and applies them to existing commands

    Args:
        root_group: The root Click command group (main CLI group)
    """
    _load_command_plugins(root_group)
    _load_hook_plugins(root_group)


def _load_command_plugins(root_group: click.Group):
    """Load and register command plugins."""
    for ep in entry_points(group="flyte.plugins.cli.commands"):
        try:
            command = ep.load()
            if not isinstance(command, click.Command):
                logger.warning(
                    f"Plugin {ep.name} did not return a click.Command, got {type(command).__name__}"
                )
                continue

            # Check if this is a subcommand (contains dot notation)
            if "." in ep.name:
                group_name, command_name = ep.name.split(".", 1)
                _add_subcommand_to_group(root_group, group_name, command_name, command)
            else:
                # Top-level command
                root_group.add_command(command, name=ep.name)
                logger.info(f"Registered plugin command: flyte {ep.name}")

        except Exception as e:
            logger.error(f"Failed to load plugin command {ep.name}: {e}")


def _load_hook_plugins(root_group: click.Group):
    """Load and apply hook plugins to existing commands."""
    for ep in entry_points(group="flyte.plugins.cli.hooks"):
        try:
            hook = ep.load()
            if not callable(hook):
                logger.warning(f"Plugin hook {ep.name} is not callable")
                continue

            # Check if this is a subcommand hook (contains dot notation)
            if "." in ep.name:
                group_name, command_name = ep.name.split(".", 1)
                _apply_hook_to_subcommand(root_group, group_name, command_name, hook)
            else:
                # Top-level command hook
                _apply_hook_to_command(root_group, ep.name, hook)

        except Exception as e:
            logger.error(f"Failed to apply hook {ep.name}: {e}")


def _add_subcommand_to_group(
    root_group: click.Group,
    group_name: str,
    command_name: str,
    command: click.Command
):
    """Add a subcommand to an existing command group."""
    if group_name not in root_group.commands:
        logger.warning(
            f"Cannot add plugin subcommand '{command_name}' - group '{group_name}' does not exist"
        )
        return

    group = root_group.commands[group_name]
    if not isinstance(group, click.Group):
        logger.warning(
            f"Cannot add plugin subcommand '{command_name}' - '{group_name}' is not a command group"
        )
        return

    group.add_command(command, name=command_name)
    logger.info(f"Registered plugin subcommand: flyte {group_name} {command_name}")


def _apply_hook_to_command(root_group: click.Group, command_name: str, hook: CommandHook):
    """Apply a hook to a top-level command."""
    if command_name not in root_group.commands:
        logger.warning(
            f"Cannot apply hook - command '{command_name}' does not exist"
        )
        return

    try:
        original_command = root_group.commands[command_name]
        modified_command = hook(original_command)
        root_group.commands[command_name] = modified_command
        logger.info(f"Applied hook to command: flyte {command_name}")
    except Exception as e:
        logger.error(f"Hook failed for command {command_name}: {e}")


def _apply_hook_to_subcommand(
    root_group: click.Group,
    group_name: str,
    command_name: str,
    hook: CommandHook
):
    """Apply a hook to a subcommand within a group."""
    if group_name not in root_group.commands:
        logger.warning(
            f"Cannot apply hook - group '{group_name}' does not exist"
        )
        return

    group = root_group.commands[group_name]
    if not isinstance(group, click.Group):
        logger.warning(
            f"Cannot apply hook - '{group_name}' is not a command group"
        )
        return

    if command_name not in group.commands:
        logger.warning(
            f"Cannot apply hook - subcommand '{command_name}' does not exist in group '{group_name}'"
        )
        return

    try:
        original_command = group.commands[command_name]
        modified_command = hook(original_command)
        group.commands[command_name] = modified_command
        logger.info(f"Applied hook to subcommand: flyte {group_name} {command_name}")
    except Exception as e:
        logger.error(f"Hook failed for subcommand {group_name}.{command_name}: {e}")
