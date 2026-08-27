import inspect
import json
import sys
import textwrap
from os import getcwd
from typing import Generator, Tuple

import rich_click as click

import flyte
import flyte.cli._common as common
from flyte.cli._plugins import get_command_distribution


@click.group(name="gen")
def gen():
    """
    Generate documentation.
    """


@gen.command(cls=common.CommandBase)
@click.option(
    "--type",
    "doc_type",
    type=str,
    required=True,
    help="Type of documentation (valid: markdown, json)",
)
@click.pass_obj
def docs(
    cfg: common.CLIConfig,
    doc_type: str,
    project: str | None = None,
    domain: str | None = None,
):
    """
    Generate documentation.
    """
    if doc_type == "markdown":
        markdown(cfg)
    elif doc_type == "json":
        json_tree(cfg)
    else:
        raise click.ClickException("Invalid documentation type: {}".format(doc_type))


def walk_commands(ctx: click.Context) -> Generator[Tuple[str, click.Command, click.Context], None, None]:
    """
    Recursively walk a Click command tree, starting from the given context.

    Yields:
        (full_command_path, command_object, context)
    """
    command = ctx.command

    if not isinstance(command, click.Group):
        yield ctx.command_path, command, ctx
    elif isinstance(command, common.FileGroup):
        # If the command is a FileGroup, yield its file path and the command itself
        # No need to recurse further into FileGroup as most subcommands are dynamically generated
        # The exception is TaskFiles which has the special 'deployed-task' subcommand that should be documented
        if type(command).__name__ == "TaskFiles":
            # For TaskFiles, we only want the special non-file-based subcommands like 'deployed-task'
            # Exclude all dynamic file-based commands
            try:
                names = command.list_commands(ctx)
                for name in names:
                    if name == "deployed-task":  # Only include the deployed-task command
                        try:
                            subcommand = command.get_command(ctx, name)
                            if subcommand is not None:
                                full_name = f"{ctx.command_path} {name}".strip()
                                sub_ctx = click.Context(subcommand, info_name=name, parent=ctx)
                                yield full_name, subcommand, sub_ctx
                        except click.ClickException:
                            continue
            except click.ClickException:
                pass

        yield ctx.command_path, command, ctx
    else:
        try:
            names = command.list_commands(ctx)
        except click.ClickException:
            # Some file-based commands might not have valid objects (e.g., test files)
            # Skip these gracefully
            return

        for name in names:
            try:
                subcommand = command.get_command(ctx, name)
                if subcommand is None:
                    continue

                full_name = f"{ctx.command_path} {name}".strip()
                sub_ctx = click.Context(subcommand, info_name=name, parent=ctx)
                yield full_name, subcommand, sub_ctx

                # Recurse if subcommand is a MultiCommand (i.e., has its own subcommands)
                # But skip RemoteTaskGroup as it requires a live Flyte backend to enumerate subcommands
                if isinstance(subcommand, click.Group) and type(subcommand).__name__ != "RemoteTaskGroup":
                    yield from walk_commands(sub_ctx)
            except click.ClickException:
                # Skip files/commands that can't be loaded
                continue


def _render_command(cmd_path: str, cmd: click.Command, cmd_ctx: click.Context, distribution: str | None) -> list[str]:
    """Render a single command's documentation as Markdown."""
    output = []
    cmd_path_parts = cmd_path.split(" ")

    output.append(f"{'#' * (len(cmd_path_parts) + 1)} {cmd_path}")

    # Which distribution provided the command is a fact about the command, so it
    # is reported. Who should be shown it is not, and is not decided here.
    if distribution:
        output.append("")
        output.append(f"> **Note:** This command is provided by the `{distribution}` plugin.")

    output.append("")
    usage_line = f"{cmd_path}"

    if any(isinstance(p, click.Option) for p in cmd.params):
        usage_line += " [OPTIONS]"

    if isinstance(cmd, click.Group):
        usage_line += " COMMAND [ARGS]..."
    else:
        for arg in (p for p in cmd.params if isinstance(p, click.Argument)):
            if arg.name:
                usage_line += f" {arg.name.upper()}" if arg.required else f" [{arg.name.upper()}]"

    output.append(f"**`{usage_line}`**")

    if cmd.help:
        output.append("")
        output.append(inspect.cleandoc(cmd.help))

    if not cmd.params:
        return output

    table_data = []
    for param in cmd.get_params(cmd_ctx):
        if isinstance(param, click.Option):
            all_opts = param.opts + param.secondary_opts
            opts = " ".join(f"`{opt}`" for opt in all_opts)
            default_value = ""
            if param.default is not None:
                default_value = f"`{param.default}`".replace(f"{getcwd()}/", "")
            help_text = dedent(param.help) if param.help else ""
            table_data.append([opts, f"`{param.type.name}`", default_value, help_text])

    if not table_data:
        return output

    output.append("")
    output.append("| Option | Type | Default | Description |")
    output.append("|--------|------|---------|-------------|")
    for row in table_data:
        output.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    return output


def _build_index_table(
    groups: dict[str, list[tuple[str, bool]]],
    from_plugin: dict[str, bool] | None,
    is_verb_table: bool,
) -> list[str]:
    """Build an index table (verb or noun).

    A command provided by a plugin is marked with a plus. That is a statement
    about where it came from, not about who may see it.
    """
    output = ["| Action | On |", "| ------ | -- |"] if is_verb_table else ["| Object | Action |", "| ------ | -- |"]

    for key, entries in groups.items():
        if is_verb_table:
            key_display = f"{key}⁺" if (from_plugin or {}).get(key) else key
            if not entries:
                # No subcommands to show: it takes none, or builds them
                # dynamically. Still list it, or a documented command becomes
                # unreachable from the index.
                output.append(f"| [`{key_display}`](#flyte-{key}) | - |")
            else:
                links = [f"[`{n}⁺`](#flyte-{key}-{n})" if g else f"[`{n}`](#flyte-{key}-{n})" for n, g in entries]
                output.append(f"| `{key_display}` | {', '.join(links)}  |")
        else:
            links = [f"[`{v}⁺`](#flyte-{v}-{key})" if g else f"[`{v}`](#flyte-{v}-{key})" for v, g in entries]
            output.append(f"| `{key}` | {', '.join(links)}  |")

    return output


def _option_default(param: click.Parameter):
    """A JSON-safe rendering of a parameter default.

    The working directory is stripped because it is a property of the machine
    that ran the generator, not of the CLI being described.
    """
    default = param.default
    if default is None or isinstance(default, (bool, int, float)):
        return default
    return str(default).replace(f"{getcwd()}/", "")


def _describe(cmd_path: str, cmd: click.Command, cmd_ctx: click.Context, distribution: str | None) -> dict:
    """One command as data.

    Deliberately excludes anything about presentation. Help text is normalised
    with `inspect.cleandoc` (undoing click's indentation, which is an artefact of
    how the string was written) but is otherwise verbatim: no escaping, no code
    fencing, no markup. A consumer that needs those applies its own.

    `options` includes the `--help` that click adds to every command, because a
    renderer that shows an option table needs to show it. `declares_options`
    reports separately whether the command declared any option of its own.
    """
    params = cmd.get_params(cmd_ctx)
    return {
        "path": cmd_path,
        "name": cmd_path.rsplit(" ", 1)[-1],
        "is_group": isinstance(cmd, click.Group),
        "distribution": distribution,
        # click adds `--help` to every command, so `options` being non-empty says
        # nothing about whether this command has any of its own. A renderer needs
        # that to decide whether to advertise `[OPTIONS]` in the usage line and
        # whether an option table carries information, and it cannot recover it
        # from `options` without assuming `--help` is the only thing click adds.
        "declares_options": any(isinstance(p, click.Option) for p in cmd.params),
        "help": inspect.cleandoc(cmd.help) if cmd.help else None,
        "arguments": [
            {"name": p.name, "required": p.required} for p in params if isinstance(p, click.Argument) and p.name
        ],
        "options": [
            {
                "opts": list(p.opts),
                "secondary_opts": list(p.secondary_opts),
                "type": p.type.name,
                "default": _option_default(p),
                "help": textwrap.dedent(p.help) if p.help else None,
            }
            for p in params
            if isinstance(p, click.Option)
        ],
    }


def _resolve_distribution(cmd_path: str, own: str | None, by_path: dict[str, str | None]) -> str | None:
    """A command's distribution, inheriting from its nearest stamped ancestor.

    Inheritance is not a nicety. Only 35 of the CLI's plugin-provided commands
    are entry points; `flyte explore volume` and the three `flyte undelete`
    subcommands are defined inside a plugin's own group and so carry no stamp of
    their own. Attributing them by entry point alone would report them as core,
    which is the exact defect this data exists to prevent -- and one the older
    `__module__` guess happened to get right.
    """
    if own:
        return own
    parts = cmd_path.split(" ")
    for i in range(len(parts) - 1, 0, -1):
        inherited = by_path.get(" ".join(parts[:i]))
        if inherited:
            return inherited
    return None


def json_tree(cfg: common.CLIConfig):
    """Print the command tree as JSON.

    The machine-readable half of this generator. It reports what the CLI *is*:
    the commands, their parameters, and which distribution provided each one.
    It takes no view on who should see them -- audience is a decision for the
    documentation that consumes this, not a property of the CLI.
    """
    ctx = cfg.ctx
    commands = [("flyte", ctx.command, ctx), *walk_commands(ctx)]

    own_by_path: dict[str, str | None] = {}
    seen: list[click.Command] = []
    ordered: list[tuple[str, click.Command, click.Context]] = []
    for cmd_path, cmd, cmd_ctx in commands:
        if cmd in seen:
            continue
        seen.append(cmd)
        own_by_path[cmd_path] = get_command_distribution(cmd)
        ordered.append((cmd_path, cmd, cmd_ctx))

    out = {
        "cli": "flyte",
        "version": flyte.__version__,
        "commands": [
            _describe(p, c, cc, _resolve_distribution(p, own_by_path[p], own_by_path)) for p, c, cc in ordered
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()


def markdown(cfg: common.CLIConfig):
    """Generate the CLI documentation in Markdown.

    Plain, vendor-neutral Markdown: headings, usage lines, help text and option
    tables. It describes the CLI and nothing else.

    This used to emit Hugo -- shortcodes, variant gating, two goldmark
    workarounds and a hardcoded set of audience names -- for one documentation
    site, whose build constraints an SDK maintainer had no way to verify. That
    rendering now lives in that site's own repository, which consumes
    `--type json`. See DOC-1481.
    """
    ctx = cfg.ctx

    command_data: list[tuple[str, str | None, list[str]]] = []
    verb_groups: dict[str, list[tuple[str, bool]]] = {}
    verb_from_plugin: dict[str, bool] = {}
    noun_groups: dict[str, list[tuple[str, bool]]] = {}

    processed: list[click.Command] = []
    distribution_by_path: dict[str, str | None] = {}
    ordered: list[tuple[str, click.Command, click.Context]] = []
    for cmd_path, cmd, cmd_ctx in [("flyte", ctx.command, ctx), *walk_commands(ctx)]:
        if cmd in processed:
            continue
        processed.append(cmd)
        distribution_by_path[cmd_path] = get_command_distribution(cmd)
        ordered.append((cmd_path, cmd, cmd_ctx))

    for cmd_path, cmd, cmd_ctx in ordered:
        distribution = _resolve_distribution(cmd_path, distribution_by_path[cmd_path], distribution_by_path)
        from_plugin = distribution is not None
        parts = cmd_path.split(" ")

        if len(parts) > 1:
            verb = parts[1]
            verb_from_plugin.setdefault(verb, from_plugin)
            verb_groups.setdefault(verb, [])
            if len(parts) > 2:
                verb_groups[verb].append((parts[2], from_plugin))
        if len(parts) == 3:
            noun_groups.setdefault(parts[2], []).append((parts[1], from_plugin))

        command_data.append((cmd_path, distribution, _render_command(cmd_path, cmd, cmd_ctx, distribution)))

    print()
    print("\n".join(_build_index_table(noun_groups, None, False)))
    print()
    print("\n".join(_build_index_table(verb_groups, verb_from_plugin, True)))

    distributions = sorted({d for _, d, _ in command_data if d})
    if distributions:
        print()
        print("## Plugin commands")
        print()
        print("> [!NOTE]")
        print("> Commands marked with **\u207a** are provided by a plugin rather than by Flyte")
        print("> itself, and are unavailable until that plugin is installed. Each command's")
        print("> section names the distribution that provides it.")
        print(">")
        print(f"> Installed here: {', '.join(f'`{d}`' for d in distributions)}.")

    print()
    for _cmd_path, _distribution, rendered in command_data:
        print()
        print("\n".join(rendered))

    # Flush stdout to ensure all output is written before the process exits.
    sys.stdout.flush()


def dedent(text: str) -> str:
    """
    Remove leading whitespace from a string.
    """
    return textwrap.dedent(text).strip("\n")
