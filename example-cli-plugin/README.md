# Flyte App Plugin

Example Flyte CLI plugin that demonstrates how to extend the Flyte CLI.

## What This Plugin Does

This plugin adds support for managing "app" entities in Flyte:

### 1. New Commands

Adds three new subcommands to existing command groups:

- **`flyte get app [NAME]`** - List all apps or get a specific app
- **`flyte create app NAME`** - Create a new app
- **`flyte delete app NAME`** - Delete an app

### 2. Command Hooks

Modifies the existing `flyte run` command to:
- Print a hello message before task execution
- Run the original task logic
- Print a completion message after execution

## Installation

### For Development (from this directory)

```bash
cd example-cli-plugin
uv pip install -e .
```

### For Production

```bash
pip install flyte-app-plugin
```

## Usage

### Get Apps

```bash
# List all apps
flyte get app

# Get a specific app
flyte get app my-app

# Get apps from a specific project
flyte get app --project my-project --domain production
```

### Create App

```bash
# Create a basic app
flyte create app my-app --project my-project

# Create with custom configuration
flyte create app my-app \
  --project my-project \
  --domain production \
  --image myregistry/myapp:latest \
  --replicas 3 \
  --description "My production application"
```

### Delete App

```bash
# Delete with confirmation prompt
flyte delete app my-app

# Force delete without confirmation
flyte delete app my-app --force

# Delete from specific project
flyte delete app my-app --project my-project --domain production
```

### Enhanced Run Command

When you run any task, you'll see the plugin's messages:

```bash
flyte run my_workflow.py my_task

# Output will include:
# ============================================================
# 👋 Hello from the App Plugin!
#    This message is printed BEFORE the task runs.
# ============================================================
#
# [... original task execution output ...]
#
# ============================================================
# 🎉 Task execution completed!
#    This message is printed AFTER the task runs.
#    Thanks for using the App Plugin!
# ============================================================
```

## Verification

To verify the plugin is installed and loaded:

```bash
# Check if 'app' subcommands appear in help
flyte get --help    # Should show 'app' command
flyte create --help # Should show 'app' command
flyte delete --help # Should show 'app' command

# Check command-specific help
flyte get app --help
flyte create app --help
flyte delete app --help

# Test the run hook
flyte run --help  # Command should work normally, with plugin hook active
```

## Plugin Structure

```
example-cli-plugin/
├── pyproject.toml              # Plugin configuration and entry points
├── README.md                   # This file
└── src/
    └── flyte_app_plugin/
        ├── __init__.py         # Package initialization
        ├── commands.py         # App management commands
        └── hooks.py            # Command hooks (run enhancement)
```

## Entry Points

The plugin registers itself via entry points in `pyproject.toml`:

```toml
[project.entry-points."flyte.plugins.cli.commands"]
"get.app" = "flyte_app_plugin.commands:get_app"
"create.app" = "flyte_app_plugin.commands:create_app"
"delete.app" = "flyte_app_plugin.commands:delete_app"

[project.entry-points."flyte.plugins.cli.hooks"]
run = "flyte_app_plugin.hooks:enhance_run_command"
```

## How It Works

1. **Entry Points Discovery**: When Flyte CLI starts, it scans for installed packages with entry points in the `flyte.plugins.cli.*` groups

2. **Command Registration**:
   - Commands with dot notation (`get.app`) are added as subcommands to existing groups
   - Commands without dots would be added as top-level commands

3. **Hook Application**:
   - Hooks are functions that receive a Click command and return a modified version
   - They can wrap callbacks, add options, or modify behavior

## Debugging

To see detailed plugin loading information:

```bash
# Use verbose mode to see plugin loading logs
flyte -vvv get app
flyte -vvv run my_task.py
```

This will show:
- Which plugins are being discovered
- Whether commands/hooks are successfully registered
- Any errors during plugin loading

## Development Tips

### Testing Changes

After modifying the plugin code:

```bash
# Reinstall in development mode
uv pip install -e .

# Test your changes
flyte get app --help
```

### Adding More Commands

1. Add the command function to `commands.py`
2. Register it in `pyproject.toml` under the appropriate entry point group
3. Reinstall the plugin

### Adding More Hooks

1. Add the hook function to `hooks.py`
2. Register it in `pyproject.toml` under `[project.entry-points."flyte.plugins.cli.hooks"]`
3. Reinstall the plugin

## Further Documentation

- [Flyte CLI Plugin System](../CLI_PLUGINS.md) - Complete plugin documentation
- [Flyte Documentation](https://docs.union.ai/flyte) - Official Flyte docs
- [Click Documentation](https://click.palletsprojects.com/) - CLI framework docs

## License

Same as Flyte SDK (Apache 2.0)
