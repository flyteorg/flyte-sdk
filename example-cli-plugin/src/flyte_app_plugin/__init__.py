"""Flyte App Plugin - Example CLI Plugin.

This plugin demonstrates how to extend the Flyte CLI by:
- Adding new commands for managing 'app' entities
- Hooking into existing commands to add custom behavior
"""

__version__ = "0.1.0"

# Export main components (optional, for programmatic use)
from .commands import create_app, delete_app, get_app
from .hooks import enhance_run_command

__all__ = [
    "get_app",
    "create_app",
    "delete_app",
    "enhance_run_command",
]
