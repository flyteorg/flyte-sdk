import datetime
import os
import subprocess
import tempfile
from pathlib import Path

import rich_click as click

from flyte.cli import _common as common

_EDIT_ERROR_START = "### !! EDIT ERROR — fix the YAML below and re-save !!"
_EDIT_ERROR_END = "### !! end edit error !!"


def _strip_error_header(content: str) -> str:
    """Remove any pre-existing error-header block so successive parse failures
    don't stack multiple headers in the editor buffer."""
    if not content.startswith(_EDIT_ERROR_START):
        return content
    end = content.find(_EDIT_ERROR_END)
    if end < 0:
        return content
    return content[end + len(_EDIT_ERROR_END) :].lstrip("\n")


def _prepend_error_header(path: Path, err_msg: str) -> None:
    """Inject a ``##``-prefixed error block above the user's buffer so the
    next editor session opens with context on what went wrong."""
    body = _strip_error_header(path.read_text())
    lines = [_EDIT_ERROR_START]
    for ln in str(err_msg).splitlines() or [""]:
        lines.append(f"## {ln}")
    lines.append(_EDIT_ERROR_END)
    lines.append("")
    path.write_text("\n".join(lines) + body)


def _save_backup(content: str) -> Path:
    """Persist the current editor buffer to ``~/.flyte/settings-edit-<ts>.yaml``
    so the user can recover unsaved work after a failure."""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target_dir = Path.home() / ".flyte"
    target_dir.mkdir(parents=True, exist_ok=True)
    backup = target_dir / f"settings-edit-{ts}.yaml"
    backup.write_text(content)
    return backup


@click.group()
def edit():
    pass


def _print_diff(console: "common.Console", overrides: dict, original_local: dict) -> bool:
    """Print a coloured summary of the override changes. Returns True when the
    diff is non-empty (i.e. there is something to apply)."""
    added = {k: v for k, v in overrides.items() if k not in original_local}
    removed = {k: v for k, v in original_local.items() if k not in overrides}
    modified = {
        k: (original_local[k], v) for k, v in overrides.items() if k in original_local and original_local[k] != v
    }
    if not (added or removed or modified):
        return False
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
    return True


@edit.command(cls=common.CommandBase)
@click.option(
    "--from-file",
    "-f",
    "from_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Apply overrides from a YAML file and skip the editor. The file can be "
    "produced by `flyte get settings` (comment markers are honoured) or be a "
    "plain YAML mapping of flat dot-notation keys to values.",
)
@click.pass_obj
def settings(cfg: common.CLIConfig, project: str | None, domain: str | None, from_file: Path | None):
    """Edit hierarchical settings interactively — or apply a YAML file directly.

    **Interactive mode** (default). Opens settings in your ``$EDITOR``. Three
    comment tiers appear:

    - ``###`` section headers and the scope line
    - ``##`` per-field descriptions and inline metadata
    - ``#`` inactive settings (uncomment the single ``#`` to activate)

    If the edited YAML fails to parse, the editor reopens with an error
    header so you can fix the syntax without losing your edits. If you
    decline to reopen — or if the server rejects the update — your buffer
    is saved under ``~/.flyte/settings-edit-<timestamp>.yaml``.

    **Non-interactive mode**: pass ``--from-file <path>`` to skip the editor
    entirely. The file's contents are parsed, the diff is printed, and the
    overrides are applied without a confirmation prompt. Ideal for
    CI/automation.
    """
    import flyte.remote as remote

    cfg.init()

    if project and not domain:
        raise click.BadOptionUsage("project", "to set project settings, domain is required")

    console = common.Console()

    try:
        settings = remote.Settings.get_settings_for_edit(project=project, domain=domain)
    except Exception as e:
        console.print(f"[red]Error fetching settings:[/red] {e}")
        raise click.Abort

    # -------------------- non-interactive path -----------------------------
    if from_file is not None:
        console.print(
            f"[bold]Applying settings for scope:[/bold] {settings.scope_description()} [dim]from {from_file}[/dim]"
        )
        file_body = from_file.read_text()
        try:
            file_overrides = remote.Settings.parse_yaml(file_body)
        except Exception as e:
            console.print(f"[red]Invalid YAML in {from_file}:[/red] {e}")
            raise click.Abort

        if not _print_diff(console, file_overrides, settings.local_overrides()):
            console.print("[dim]No changes detected.[/dim]")
            return

        try:
            settings.update_settings(file_overrides)
        except Exception as e:
            console.print(f"[red]Error updating settings:[/red] {e}")
            raise click.Abort

        console.print("[green]✓ Settings updated successfully[/green]")
        return

    # -------------------- interactive path ---------------------------------
    console.print(f"[bold]Editing settings for scope:[/bold] {settings.scope_description()}")

    yaml_content = settings.to_yaml()
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(yaml_content)

    keep_tmp = False
    try:
        # Editor + parse loop: reopen until YAML parses cleanly, or the user bails.
        overrides: dict | None = None
        edited_content = yaml_content
        while overrides is None:
            result = subprocess.run([editor, str(tmp_path)], check=False)
            if result.returncode != 0:
                console.print(f"[yellow]Editor exited with code {result.returncode}[/yellow]")

            edited_content = tmp_path.read_text()

            if _strip_error_header(edited_content).strip() == yaml_content.strip():
                console.print("[dim]No changes detected.[/dim]")
                return

            try:
                overrides = remote.Settings.parse_yaml(_strip_error_header(edited_content))
            except Exception as e:
                console.print(f"[red]Invalid YAML:[/red] {e}")
                if not click.confirm("Reopen the editor to fix it?", default=True):
                    backup = _save_backup(edited_content)
                    console.print(f"[yellow]Your edits were saved to:[/yellow] {backup}")
                    return
                _prepend_error_header(tmp_path, str(e))
                overrides = None  # loop

        if not _print_diff(console, overrides, settings.local_overrides()):
            console.print("[dim]No changes detected.[/dim]")
            return

        if not click.confirm("\nApply these changes?", default=True):
            console.print("[dim]Changes discarded.[/dim]")
            return

        try:
            settings.update_settings(overrides)
        except Exception as e:
            console.print(f"[red]Error updating settings:[/red] {e}")
            backup = _save_backup(edited_content)
            console.print(f"[yellow]Your edits were saved to:[/yellow] {backup}")
            keep_tmp = False  # backup is the durable copy; tmp can go.
            raise click.Abort

        console.print("[green]✓ Settings updated successfully[/green]")

    finally:
        if not keep_tmp:
            tmp_path.unlink(missing_ok=True)
