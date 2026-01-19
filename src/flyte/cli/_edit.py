import os
import subprocess
import tempfile
from pathlib import Path

import rich_click as click

from flyte.cli import _common as common


@click.group()
def edit():
    pass


@edit.command(cls=common.CommandBase)
@click.pass_obj
def settings(cfg: common.CLIConfig, project: str | None, domain: str | None):
    """Edit hierarchical settings interactively.

    Opens settings in your $EDITOR, showing:
    - Local overrides (uncommented)
    - Inherited settings (commented with origin)

    To create an override: uncomment a line and/or edit its value
    To remove an override: comment out the line
    """
    import flyte.remote as remote

    # Determine scope
    scope_desc = "ROOT"
    if project and domain:
        scope_desc = f"PROJECT({domain}/{project})"
    elif domain:
        scope_desc = f"DOMAIN({domain})"
    elif project:
        raise click.BadOptionUsage("project", "to set project settings, domain is required")

    console = common.Console()
    console.print(f"[bold]Editing settings for scope:[/bold] {scope_desc}")

    # Get current settings
    try:
        settings = remote.Settings.get(project=project, domain=domain)
    except Exception as e:
        console.print(f"[red]Error fetching settings:[/red] {e}")
        raise click.Abort

    # Generate YAML for editing
    yaml_content = settings.to_yaml()

    # Get editor from environment
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))

    # Create a temporary file for editing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(yaml_content)

    try:
        # Open the editor
        result = subprocess.run([editor, str(tmp_path)], check=False)
        if result.returncode != 0:
            console.print(f"[yellow]Editor exited with code {result.returncode}[/yellow]")

        # Read the edited content
        with open(tmp_path, "r") as f:
            edited_content = f.read()

        # Check if content changed
        if edited_content.strip() == yaml_content.strip():
            console.print("[dim]No changes detected.[/dim]")
            return

        # Parse the edited YAML to extract overrides
        try:
            overrides = remote.Settings.parse_yaml(edited_content)
        except Exception as e:
            console.print(f"[red]Error parsing edited YAML:[/red] {e}")
            raise click.Abort

        # Show changes
        original_local = {s.key: s.value for s in settings.local_settings}
        added = {k: v for k, v in overrides.items() if k not in original_local}
        removed = {k: v for k, v in original_local.items() if k not in overrides}
        modified = {
            k: (original_local[k], v) for k, v in overrides.items() if k in original_local and original_local[k] != v
        }

        if added or removed or modified:
            console.print("\n[bold]Changes to apply:[/bold]")
            if added:
                console.print("[green]Added overrides:[/green]")
                for k, v in added.items():
                    console.print(f"  + {k}: {v}")
            if modified:
                console.print("[yellow]Modified overrides:[/yellow]")
                for k, (old, new) in modified.items():
                    console.print(f"  ~ {k}: {old} → {new}")
            if removed:
                console.print("[red]Removed overrides (will inherit):[/red]")
                for k, v in removed.items():
                    console.print(f"  - {k}: {v}")

            # Confirm and apply
            if click.confirm("\nApply these changes?", default=True):
                try:
                    settings.update(overrides)
                    console.print("[green]✓ Settings updated successfully[/green]")
                except Exception as e:
                    console.print(f"[red]Error updating settings:[/red] {e}")
                    raise click.Abort
            else:
                console.print("[dim]Changes discarded.[/dim]")
        else:
            console.print("[dim]No changes detected.[/dim]")

    finally:
        # Clean up temporary file
        tmp_path.unlink(missing_ok=True)
