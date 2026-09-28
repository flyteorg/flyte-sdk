"""`flyte migrate`: convert a flytekit (v1) file into a flyte (v2) file next to it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import rich_click as click

from . import _common as common

if TYPE_CHECKING:
    from flyte._migrate import MigrationResult

#: Exit code for `--strict` when the migrated file still contains TODO markers.
EXIT_TODOS = 2


def _split_rules(values: Tuple[str, ...]) -> list[str]:
    return [rule.strip() for value in values for rule in value.split(",") if rule.strip()]


@click.command(name="migrate")
@click.argument("file_path", type=click.Path(path_type=Path, dir_okay=False))
@click.option(
    "--suffix",
    default="_v2",
    show_default=True,
    help="Suffix appended to the file name for the output file, e.g. `pipeline.py` -> `pipeline_v2.py`.",
)
@click.option("--force", is_flag=True, default=False, help="Overwrite the output file if it already exists.")
@click.option("--dry-run", is_flag=True, default=False, help="Print the migrated code instead of writing it.")
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help=f"Exit with code {EXIT_TODOS} if the migrated code still contains `TODO(flyte migrate)` markers.",
)
@click.option(
    "--rules",
    "rules",
    multiple=True,
    help="Only run these rules (comma-separated or repeated).",
)
@click.option(
    "--exclude-rules",
    "exclude_rules",
    multiple=True,
    help="Skip these rules (comma-separated or repeated).",
)
@click.pass_obj
def migrate(
    cfg: common.CLIConfig,
    file_path: Path,
    suffix: str,
    force: bool,
    dry_run: bool,
    strict: bool,
    rules: Tuple[str, ...],
    exclude_rules: Tuple[str, ...],
):
    """
    Migrate a flytekit (v1) Python file to the flyte (v2) SDK.

    The converted code is written next to the source file with a `_v2` suffix; the source file is never modified.
    Anything that cannot be converted automatically is marked with a `TODO(flyte migrate)` comment.

    ```bash
    flyte migrate workflows/pipeline.py            # writes workflows/pipeline_v2.py
    flyte migrate pipeline.py --dry-run            # print the result instead
    flyte -of json migrate pipeline.py --strict    # machine-readable report, fail on TODOs
    ```
    """
    try:
        from flyte._migrate import MigrationError, migrate_file, write_result
    except ImportError as e:
        if e.name != "libcst":
            raise
        raise click.ClickException('`flyte migrate` needs LibCST. Install it with: pip install "flyte[migrate]"')

    if not file_path.is_file():
        raise click.ClickException(f"File not found: {file_path}")

    try:
        result = migrate_file(
            file_path, suffix=suffix, rules=_split_rules(rules) or None, exclude_rules=_split_rules(exclude_rules)
        )
        if not dry_run:
            write_result(result, force=force)
    except MigrationError as e:
        raise click.ClickException(str(e)) from e

    output_format = cfg.output_format if cfg else "table"
    if output_format in ("json", "json-raw"):
        print(json.dumps(result.to_dict(include_code=dry_run), indent=2))
    elif dry_run and result.status != "nothing_to_migrate":
        click.echo(result.code, nl=False)
    else:
        _print_summary(result)

    if strict and result.todos:
        click.get_current_context().exit(EXIT_TODOS)


def _print_summary(result: MigrationResult) -> None:
    console = common.get_console()

    def line(text: str = "") -> None:
        # soft_wrap keeps `path:line` references on one line, so terminals can still link them.
        console.print(text, markup=False, highlight=False, soft_wrap=True)

    if result.status == "nothing_to_migrate":
        line(f"Nothing to migrate: {result.source_path} does not use flytekit or was already migrated.")
        return

    console.print(
        f"[green]✔[/green] Migrated {result.source_path} → {result.output_path}", highlight=False, soft_wrap=True
    )
    line()
    top = ", ".join(f"{name} x{count}" for name, count in result.applied.most_common(5))
    line(f"  Rules applied  {sum(result.applied.values()):>3}   ({top})" if top else "  Rules applied    0")
    if result.environments:
        line(f"  Environments   {len(result.environments):>3}   ({', '.join(result.environments)})")
    line(f"  TODOs          {len(result.todos):>3}")
    for todo in result.todos:
        line(f"    {result.output_path}:{todo.line}  {todo.message}")
    line()
    line("Next steps:")
    line(f"  review {result.output_path}, then run it with `flyte run --local {result.output_path} <task>`")
