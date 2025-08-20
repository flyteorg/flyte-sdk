import textwrap
from os import getcwd
from typing import Generator, Tuple

import rich_click as click

import flyte.cli._common as common


@click.group(name="gen")
def gen():
    """
    Generate documentation.
    """


@gen.command(cls=common.CommandBase)
@click.option("--type", "doc_type", type=str, required=True, help="Type of documentation (valid: markdown)")
@click.pass_obj
def docs(cfg: common.CLIConfig, doc_type: str, project: str | None = None, domain: str | None = None):
    """
    Generate documentation.
    """
    if doc_type == "markdown":
        markdown(cfg)
    else:
        raise click.ClickException("Invalid documentation type: {}".format(doc_type))


def walk_commands(ctx: click.Context) -> Generator[Tuple[str, click.Command], None, None]:
    """
    Recursively walk a Click command tree, starting from the given context.

    Yields:
        (full_command_path, command_object)
    """
    command = ctx.command

    if not isinstance(command, click.Group):
        yield ctx.command_path, command
    elif isinstance(command, common.FileGroup):
        # If the command is a FileGroup, yield its file path and the command itself
        # No need to recurse further into FileGroup as it doesn't have subcommands, they are dynamically generated
        yield ctx.command_path, command
    else:
        for name in command.list_commands(ctx):
            subcommand = command.get_command(ctx, name)
            if subcommand is None:
                continue

            full_name = f"{ctx.command_path} {name}".strip()
            yield full_name, subcommand

            # Recurse if subcommand is a MultiCommand (i.e., has its own subcommands)
            if isinstance(subcommand, click.Group):
                sub_ctx = click.Context(subcommand, info_name=name, parent=ctx)
                yield from walk_commands(sub_ctx)


def markdown(cfg: common.CLIConfig):
    """
    Generate documentation in Markdown format
    """
    ctx = cfg.ctx

    output = []
    output_verb_groups: dict[str, list[str]] = {}
    output_noun_groups: dict[str, list[str]] = {}

    processed = []
    commands = [*[("flyte", ctx.command)], *walk_commands(ctx)]
    for cmd_path, cmd in commands:
        if cmd in processed:
            # We already processed this command, skip it
            continue
        processed.append(cmd)
        output.append("")

        cmd_path_parts = cmd_path.split(" ")

        if len(cmd_path_parts) > 1:
            if cmd_path_parts[1] not in output_verb_groups:
                output_verb_groups[cmd_path_parts[1]] = []
            if len(cmd_path_parts) > 2:
                output_verb_groups[cmd_path_parts[1]].append(cmd_path_parts[2])

        if len(cmd_path_parts) == 3:
            if cmd_path_parts[2] not in output_noun_groups:
                output_noun_groups[cmd_path_parts[2]] = []
            output_noun_groups[cmd_path_parts[2]].append(cmd_path_parts[1])

        output.append(f"{'#' * (len(cmd_path_parts) + 1)} {cmd_path}")
        if cmd.help:
            output.append("")
            output.append(f"{dedent(cmd.help)}")

        if not cmd.params:
            continue

        params = cmd.get_params(click.Context(cmd))

        # Collect all data first to calculate column widths
        table_data = []
        for param in params:
            if isinstance(param, click.Option):
                # Format each option with backticks before joining
                all_opts = param.opts + param.secondary_opts
                if len(all_opts) == 1:
                    opts = f"`{all_opts[0]}`"
                else:
                    opts = "".join(
                        [
                            "{{< multiline >}}",
                            "\n".join([f"`{opt}`" for opt in all_opts]),
                            "{{< /multiline >}}",
                        ]
                    )
                default_value = ""
                if param.default is not None:
                    default_value = f"`{param.default}`"
                    default_value = default_value.replace(f"{getcwd()}/", "")
                help_text = dedent(param.help) if param.help else ""
                table_data.append([opts, f"`{param.type.name}`", default_value, help_text])

        if not table_data:
            continue

        # Add table header with proper alignment
        output.append("")
        output.append("| Option | Type | Default | Description |")
        output.append("|--------|------|---------|-------------|")

        # Add table rows with proper alignment
        for row in table_data:
            output.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    output_verb_index = []

    if len(output_verb_groups) > 0:
        output_verb_index.append("| Action | On |")
        output_verb_index.append("| ------ | -- |")
        for verb, nouns in output_verb_groups.items():
            entries = [f"[`{noun}`](#flyte-{verb}-{noun})" for noun in nouns]
            if len(entries) == 0:
                verb_link = f"[`{verb}`](#flyte-{verb})"
                output_verb_index.append(f"| {verb_link} | - |")
            else:
                output_verb_index.append(f"| `{verb}` | {', '.join(entries)}  |")

    output_noun_index = []

    if len(output_noun_groups) > 0:
        output_noun_index.append("| Object | Action |")
        output_noun_index.append("| ------ | -- |")
        for obj, actions in output_noun_groups.items():
            entries = [f"[`{action}`](#flyte-{action}-{obj})" for action in actions]
            output_noun_index.append(f"| `{obj}` | {', '.join(entries)}  |")

    print()
    print("{{< grid >}}")
    print("{{< markdown >}}")
    print("\n".join(output_noun_index))
    print("{{< /markdown >}}")
    print("{{< markdown >}}")
    print("\n".join(output_verb_index))
    print("{{< /markdown >}}")
    print("{{< /grid >}}")
    print()
    print("\n".join(output))


def dedent(text: str) -> str:
    """
    Remove leading whitespace from a string.
    """
    return textwrap.dedent(text).strip("\n")
