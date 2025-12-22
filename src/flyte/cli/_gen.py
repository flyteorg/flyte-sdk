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
                # But skip ReferenceTaskGroup as it requires a live Flyte backend to enumerate subcommands
                if isinstance(subcommand, click.Group) and type(subcommand).__name__ != "ReferenceTaskGroup":
                    yield from walk_commands(sub_ctx)
            except click.ClickException:
                # Skip files/commands that can't be loaded
                continue


def get_plugin_info(cmd: click.Command) -> tuple[bool, str | None]:
    """
    Determine if a command is from a plugin and get the plugin module name.

    Returns:
        (is_plugin, plugin_module_name)
    """
    if not cmd or not cmd.callback:
        return False, None

    module = cmd.callback.__module__
    if "flyte." not in module:
        # External plugin
        parts = module.split(".")
        if len(parts) == 1:
            return True, parts[0]
        return True, f"{parts[0]}.{parts[1]}"
    elif module.startswith("flyte.") and not module.startswith("flyte.cli"):
        # Check if it's from a flyte plugin (not core CLI)
        # Core CLI modules are: flyte.cli.*
        # Plugin modules would be things like: flyte.databricks, flyte.snowflake, etc.
        parts = module.split(".")
        if len(parts) > 1 and parts[1] not in ["cli", "remote", "core", "internal", "app"]:
            return True, f"flyte.{parts[1]}"

    return False, None


def markdown(cfg: common.CLIConfig):
    """
    Generate documentation in Markdown format
    """
    ctx = cfg.ctx

    output = []
    # Store verbs with their nouns: {verb_name: [(noun_name, is_plugin, plugin_module), ...]}
    output_verb_groups: dict[str, list[tuple[str, bool, str | None]]] = {}
    # Store verb metadata: {verb_name: (is_plugin, plugin_module)}
    verb_metadata: dict[str, tuple[bool, str | None]] = {}
    # Store nouns with their verbs: {noun_name: [(verb_name, is_plugin, plugin_module), ...]}
    output_noun_groups: dict[str, list[tuple[str, bool, str | None]]] = {}

    processed = []
    commands = [*[("flyte", ctx.command, ctx)], *walk_commands(ctx)]
    for cmd_path, cmd, cmd_ctx in commands:
        if cmd in processed:
            # We already processed this command, skip it
            continue
        processed.append(cmd)
        output.append("")

        is_plugin, plugin_module = get_plugin_info(cmd)

        cmd_path_parts = cmd_path.split(" ")

        if len(cmd_path_parts) > 1:
            verb = cmd_path_parts[1]

            # Store verb metadata
            if verb not in verb_metadata:
                verb_metadata[verb] = (is_plugin, plugin_module)

            # Initialize verb group if needed
            if verb not in output_verb_groups:
                output_verb_groups[verb] = []

            if len(cmd_path_parts) > 2:
                noun = cmd_path_parts[2]
                # Add noun to verb's list
                output_verb_groups[verb].append((noun, is_plugin, plugin_module))

        if len(cmd_path_parts) == 3:
            noun = cmd_path_parts[2]
            verb = cmd_path_parts[1]
            if noun not in output_noun_groups:
                output_noun_groups[noun] = []
            output_noun_groups[noun].append((verb, is_plugin, plugin_module))

        output.append(f"{'#' * (len(cmd_path_parts) + 1)} {cmd_path}")

        # Add plugin notice if this is a plugin command
        if is_plugin and plugin_module:
            output.append("")
            output.append(
                f"> **Note:** This command is provided by the `{plugin_module}` plugin. "
                f"See the plugin documentation for installation instructions."
            )

        # Add usage information
        output.append("")
        usage_line = f"{cmd_path}"

        # Add [OPTIONS] if command has options
        if any(isinstance(p, click.Option) for p in cmd.params):
            usage_line += " [OPTIONS]"

        # Add command-specific usage pattern
        if isinstance(cmd, click.Group):
            usage_line += " COMMAND [ARGS]..."
        else:
            # Add arguments if any
            args = [p for p in cmd.params if isinstance(p, click.Argument)]
            for arg in args:
                if arg.name:  # Check if name is not None
                    if arg.required:
                        usage_line += f" {arg.name.upper()}"
                    else:
                        usage_line += f" [{arg.name.upper()}]"

        output.append(f"**`{usage_line}`**")

        if cmd.help:
            output.append("")
            output.append(f"{dedent(cmd.help)}")

        if not cmd.params:
            continue

        params = cmd.get_params(cmd_ctx)

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

    # Generate verb index table
    output_verb_index = []
    has_plugin_verbs = False

    if len(output_verb_groups) > 0:
        output_verb_index.append("| Action | On |")
        output_verb_index.append("| ------ | -- |")
        for verb, nouns in output_verb_groups.items():
            verb_is_plugin, _ = verb_metadata.get(verb, (False, None))
            verb_display = verb
            if verb_is_plugin:
                verb_display = f"{verb}⁺"
                has_plugin_verbs = True

            if len(nouns) == 0:
                verb_link = f"[`{verb_display}`](#flyte-{verb})"
                output_verb_index.append(f"| {verb_link} | - |")
            else:
                # Create links for nouns
                noun_links = []
                for noun, noun_is_plugin, _ in nouns:
                    noun_display = noun
                    if noun_is_plugin:
                        noun_display = f"{noun}⁺"
                        has_plugin_verbs = True
                    noun_links.append(f"[`{noun_display}`](#flyte-{verb}-{noun})")
                output_verb_index.append(f"| `{verb_display}` | {', '.join(noun_links)}  |")

        if has_plugin_verbs:
            output_verb_index.append("")
            output_verb_index.append("**⁺** Plugin command - see command documentation for installation instructions")

    # Generate noun index table
    output_noun_index = []
    has_plugin_nouns = False

    if len(output_noun_groups) > 0:
        output_noun_index.append("| Object | Action |")
        output_noun_index.append("| ------ | -- |")
        for obj, actions in output_noun_groups.items():
            action_links = []
            for action, action_is_plugin, _ in actions:
                action_display = action
                if action_is_plugin:
                    action_display = f"{action}⁺"
                    has_plugin_nouns = True
                action_links.append(f"[`{action_display}`](#flyte-{action}-{obj})")
            output_noun_index.append(f"| `{obj}` | {', '.join(action_links)}  |")

        if has_plugin_nouns:
            output_noun_index.append("")
            output_noun_index.append("**⁺** Plugin command - see command documentation for installation instructions")

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
