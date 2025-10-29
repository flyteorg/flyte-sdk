# Flyte CLI Plugin System

The Flyte CLI supports a plugin system that allows external packages to extend the CLI by:
1. Adding new top-level commands
2. Adding new subcommands to existing command groups
3. Modifying the behavior of existing commands

Plugins are automatically discovered when installed via `pip install` using Python entry points.

## Creating a CLI Plugin

### 1. Project Structure

```
my-flyte-plugin/
├── pyproject.toml
└── src/
    └── my_flyte_plugin/
        ├── __init__.py
        ├── commands.py
        └── hooks.py
```

### 2. Define Entry Points

In your `pyproject.toml`:

```toml
[project]
name = "my-flyte-plugin"
version = "0.1.0"
dependencies = ["flyte>=1.0.0"]

[project.entry-points."flyte.plugins.cli.commands"]
# Top-level command: flyte my-command
my-command = "my_flyte_plugin.commands:my_command"

# Subcommand in 'get' group: flyte get my-resource
"get.my-resource" = "my_flyte_plugin.commands:get_my_resource"

# Subcommand in 'create' group: flyte create my-resource
"create.my-resource" = "my_flyte_plugin.commands:create_my_resource"

[project.entry-points."flyte.plugins.cli.hooks"]
# Hook into 'flyte run' command
run = "my_flyte_plugin.hooks:enhance_run"

# Hook into 'flyte get project' subcommand
"get.project" = "my_flyte_plugin.hooks:enhance_get_project"
```

### 3. Implement Commands

In `src/my_flyte_plugin/commands.py`:

```python
import rich_click as click

@click.command()
@click.option("--name", required=True, help="Name of the resource")
@click.option("--verbose", is_flag=True, help="Show detailed output")
def my_command(name: str, verbose: bool):
    """
    My custom top-level command.

    This command will be available as: flyte my-command
    """
    if verbose:
        click.echo(f"Running my-command with name={name}")
    click.echo(f"Hello from my plugin: {name}")


@click.command()
@click.argument("name")
@click.pass_obj
def get_my_resource(obj, name: str):
    """
    Get my custom resource.

    This command will be available as: flyte get my-resource <name>

    The obj parameter receives the CLIConfig from the main CLI context.
    """
    from flyte.cli._common import CLIConfig

    cli_config: CLIConfig = obj

    # Access global CLI configuration
    endpoint = cli_config.endpoint
    org = cli_config.org

    click.echo(f"Getting resource '{name}' from {endpoint} (org: {org})")
    # Your implementation here


@click.command()
@click.argument("name")
@click.option("--description", help="Resource description")
@click.pass_obj
def create_my_resource(obj, name: str, description: str | None):
    """
    Create my custom resource.

    This command will be available as: flyte create my-resource <name>
    """
    click.echo(f"Creating resource: {name}")
    if description:
        click.echo(f"Description: {description}")
    # Your implementation here
```

### 4. Implement Hooks

In `src/my_flyte_plugin/hooks.py`:

```python
import rich_click as click

def enhance_run(command: click.Command) -> click.Command:
    """
    Modify the 'flyte run' command.

    This hook allows you to:
    - Add new options to the command
    - Wrap the callback to add pre/post processing
    - Modify command behavior
    """
    # Save the original callback
    original_callback = command.callback

    # Create a wrapper that adds behavior
    def wrapped_callback(*args, **kwargs):
        # Pre-processing
        click.echo("🔌 Plugin: Preparing to run task...")

        # Call original command
        result = original_callback(*args, **kwargs)

        # Post-processing
        click.echo("🔌 Plugin: Task execution completed!")

        return result

    # Replace the callback
    command.callback = wrapped_callback

    # Optionally add new options
    # command.params.append(
    #     click.Option(["--my-option"], help="My custom option")
    # )

    return command


def enhance_get_project(command: click.Command) -> click.Command:
    """
    Modify the 'flyte get project' subcommand.

    This shows how to hook into subcommands within command groups.
    """
    original_callback = command.callback

    def wrapped_callback(*args, **kwargs):
        click.echo("🔌 Plugin: Enhanced project retrieval")
        return original_callback(*args, **kwargs)

    command.callback = wrapped_callback
    return command
```

## Entry Point Naming Convention

### Commands

Entry point names determine where the command appears in the CLI:

| Entry Point Name | Result | Example |
|-----------------|--------|---------|
| `my-command` | Top-level command | `flyte my-command` |
| `get.my-object` | Subcommand in `get` group | `flyte get my-object` |
| `create.my-object` | Subcommand in `create` group | `flyte create my-object` |
| `delete.my-object` | Subcommand in `delete` group | `flyte delete my-object` |

Available command groups: `run`, `get`, `create`, `delete`, `update`, `deploy`, `build`, `gen`, `abort`

### Hooks

Hook entry point names target specific commands:

| Entry Point Name | Hooks Into | Example |
|-----------------|-----------|---------|
| `run` | `flyte run` | Modify run command |
| `get` | `flyte get` | Modify get group |
| `get.project` | `flyte get project` | Modify specific subcommand |
| `create.secret` | `flyte create secret` | Modify specific subcommand |

## Installation and Testing

### Install Your Plugin

```bash
# Install in development mode
cd my-flyte-plugin
pip install -e .

# Or install from PyPI
pip install my-flyte-plugin
```

### Verify Plugin is Loaded

The plugin will be automatically discovered and loaded when you run `flyte`:

```bash
# See your new commands
flyte --help

# Your command should appear
flyte my-command --help

# Your subcommand should appear
flyte get --help
flyte get my-resource --help
```

### Debug Plugin Loading

Use verbose mode to see plugin loading messages:

```bash
flyte -vvv my-command
```

This will show debug logs including:
- Which plugins are being discovered
- Any errors during plugin loading
- When hooks are applied

## Best Practices

1. **Error Handling**: Wrap your plugin code in try/except to prevent breaking the CLI
2. **Dependencies**: Declare `flyte` as a dependency in your `pyproject.toml`
3. **Documentation**: Provide clear docstrings and help text for your commands
4. **Naming**: Use descriptive entry point names that won't conflict with other plugins
5. **Testing**: Test your plugin with different Flyte SDK versions
6. **Context Access**: Use `@click.pass_obj` to access the global `CLIConfig` object

## Advanced: Accessing Flyte SDK Features

Your plugin commands have full access to the Flyte SDK:

```python
import rich_click as click
from flyte import Remote

@click.command()
@click.pass_obj
def my_command(obj):
    """Command that uses Flyte SDK."""
    from flyte.cli._common import CLIConfig

    cli_config: CLIConfig = obj

    # Initialize Flyte SDK with CLI configuration
    remote = cli_config.config.client()

    # Use Flyte API
    projects = remote.list_projects()
    for project in projects:
        click.echo(f"Project: {project.name}")
```

## Example Plugins

### Metrics Plugin

Add custom metrics tracking to task execution:

```toml
[project.entry-points."flyte.plugins.cli.hooks"]
run = "flyte_metrics.hooks:track_run_metrics"
```

### Validation Plugin

Add validation before deploying:

```toml
[project.entry-points."flyte.plugins.cli.hooks"]
deploy = "flyte_validator.hooks:validate_before_deploy"
```

### Custom Resource Plugin

Add commands for managing custom resources:

```toml
[project.entry-points."flyte.plugins.cli.commands"]
"get.dataset" = "flyte_datasets.commands:get_dataset"
"create.dataset" = "flyte_datasets.commands:create_dataset"
"delete.dataset" = "flyte_datasets.commands:delete_dataset"
```

## Troubleshooting

### Plugin Not Discovered

1. Verify entry points are correctly defined in `pyproject.toml`
2. Reinstall the plugin: `pip install -e .`
3. Check plugin is installed: `pip list | grep my-plugin`
4. Run with verbose logging: `flyte -vvv --help`

### Hook Not Applied

1. Ensure the command/group name matches exactly
2. Check the hook function returns the modified command
3. Verify the hook is callable
4. Check logs with: `flyte -vvv <command>`

### Import Errors

1. Ensure `flyte` is in your plugin's dependencies
2. Check all imports are available
3. Use absolute imports in your plugin code

## Questions?

- Documentation: https://docs.union.ai/flyte
- Issues: https://github.com/flyteorg/flyte/issues
- Slack: https://slack.flyte.org
