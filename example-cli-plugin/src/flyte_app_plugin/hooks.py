"""Hooks for modifying existing Flyte CLI commands.

This module provides hooks that modify the behavior of existing commands.
"""

import rich_click as click


def enhance_run_command(command: click.Command) -> click.Command:
    """
    Enhance the 'flyte run' command with pre and post execution hooks.

    This hook wraps the existing 'flyte run' command to:
    1. Print a hello message before execution
    2. Run the original command logic
    3. Print a completion message after execution

    Args:
        command: The original 'flyte run' command

    Returns:
        The modified command with enhanced behavior
    """
    # Save the original callback
    original_callback = command.callback

    # Create a wrapper function that adds our custom behavior
    def enhanced_callback(*args, **kwargs):
        # Pre-execution hook
        click.echo()
        click.echo("=" * 60)
        click.echo("👋 Hello from the App Plugin!")
        click.echo("   This message is printed BEFORE the task runs.")
        click.echo("=" * 60)
        click.echo()

        # Call the original command logic
        try:
            result = original_callback(*args, **kwargs)
        except Exception as e:
            # If something goes wrong, still print our message
            click.echo()
            click.echo("=" * 60)
            click.echo("❌ Task execution failed, but plugin caught it!")
            click.echo(f"   Error: {e}")
            click.echo("=" * 60)
            click.echo()
            raise

        # Post-execution hook
        click.echo()
        click.echo("=" * 60)
        click.echo("🎉 Task execution completed!")
        click.echo("   This message is printed AFTER the task runs.")
        click.echo("   Thanks for using the App Plugin!")
        click.echo("=" * 60)
        click.echo()

        return result

    # Replace the command's callback with our enhanced version
    command.callback = enhanced_callback

    return command
