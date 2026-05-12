import sys
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
@click.option(
    "--plugin-variants",
    "plugin_variants",
    type=str,
    default=None,
    help="Hugo variant names for plugin commands (e.g., 'union'). "
    "When set, plugin command sections and index entries are wrapped in "
    "{{< variant >}} shortcodes. Core commands appear unconditionally.",
)
@click.pass_obj
def docs(
    cfg: common.CLIConfig,
    doc_type: str,
    plugin_variants: str | None,
    project: str | None = None,
    domain: str | None = None,
):
    """
    Generate documentation.
    """
    if doc_type == "markdown":
        markdown(cfg, plugin_variants=plugin_variants)
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


def _render_command(
    cmd_path: str, cmd: click.Command, cmd_ctx: click.Context, is_plugin: bool, plugin_module: str | None
) -> list[str]:
    """Render a single command's documentation as a list of markdown lines."""
    output = []
    cmd_path_parts = cmd_path.split(" ")

    output.append(f"{'#' * (len(cmd_path_parts) + 1)} {cmd_path}")

    # Add plugin notice if this is a plugin command
    if is_plugin and plugin_module:
        output.append("")
        output.append(f"> **Note:** This command is provided by the [`{plugin_module}`](#plugin-commands) plugin.")

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
        return output

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
            # Escape Hugo shortcode delimiters that may appear in help text
            help_text = help_text.replace("{{<", r"{{&lt;").replace("{{%", r"{{&percnt;")
            table_data.append([opts, f"`{param.type.name}`", default_value, help_text])

    if not table_data:
        return output

    # Add table header with proper alignment
    output.append("")
    output.append("| Option | Type | Default | Description |")
    output.append("|--------|------|---------|-------------|")

    # Add table rows with proper alignment
    for row in table_data:
        output.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    return output


def _build_index_table(
    groups: dict[str, list[tuple[str, bool, str | None]]],
    metadata: dict[str, tuple[bool, str | None]] | None,
    is_verb_table: bool,
    include_plugins: bool,
) -> list[str]:
    """Build an index table (verb or noun), optionally filtering plugin entries.

    Args:
        groups: verb->nouns or noun->verbs mapping.
        metadata: verb->(is_plugin, module) mapping (only for verb tables).
        is_verb_table: True for verb (Action/On) table, False for noun (Object/Action) table.
        include_plugins: Whether to include plugin entries.
    """
    output = []

    if is_verb_table:
        output.append("| Action | On |")
        output.append("| ------ | -- |")
    else:
        output.append("| Object | Action |")
        output.append("| ------ | -- |")

    for key, entries in groups.items():
        if is_verb_table:
            key_is_plugin, _ = metadata.get(key, (False, None)) if metadata else (False, None)
            if key_is_plugin and not include_plugins:
                continue

            key_display = key
            if key_is_plugin and include_plugins:
                key_display = f"{key}⁺"

            # Filter entries based on include_plugins
            filtered = [(n, ip, pm) for n, ip, pm in entries if include_plugins or not ip]
            if not filtered and not key_is_plugin:
                # Verb has no non-plugin nouns — still show it if it's a core verb
                verb_link = f"[`{key_display}`](#flyte-{key})"
                output.append(f"| {verb_link} | - |")
            elif not filtered:
                continue
            else:
                noun_links = []
                for noun, noun_is_plugin, _ in filtered:
                    noun_display = noun
                    if noun_is_plugin and include_plugins:
                        noun_display = f"{noun}⁺"
                    noun_links.append(f"[`{noun_display}`](#flyte-{key}-{noun})")
                if len(filtered) == 0:
                    verb_link = f"[`{key_display}`](#flyte-{key})"
                    output.append(f"| {verb_link} | - |")
                else:
                    output.append(f"| `{key_display}` | {', '.join(noun_links)}  |")
        else:
            # Noun table
            filtered = [(v, ip, pm) for v, ip, pm in entries if include_plugins or not ip]
            if not filtered:
                continue

            action_links = []
            for action, action_is_plugin, _ in filtered:
                action_display = action
                if action_is_plugin and include_plugins:
                    action_display = f"{action}⁺"
                action_links.append(f"[`{action_display}`](#flyte-{action}-{key})")
            output.append(f"| `{key}` | {', '.join(action_links)}  |")

    return output


def markdown(cfg: common.CLIConfig, plugin_variants: str | None = None):
    """
    Generate documentation in Markdown format.

    Args:
        cfg: CLI configuration.
        plugin_variants: Space-separated Hugo variant names for plugin commands.
            When set, plugin sections are wrapped in {{< variant >}} shortcodes.
    """
    ctx = cfg.ctx

    # Collect command data
    # Each entry: (cmd_path, cmd, cmd_ctx, is_plugin, plugin_module, rendered_lines)
    command_data = []
    output_verb_groups: dict[str, list[tuple[str, bool, str | None]]] = {}
    verb_metadata: dict[str, tuple[bool, str | None]] = {}
    output_noun_groups: dict[str, list[tuple[str, bool, str | None]]] = {}

    processed = []
    commands = [*[("flyte", ctx.command, ctx)], *walk_commands(ctx)]
    for cmd_path, cmd, cmd_ctx in commands:
        if cmd in processed:
            continue
        processed.append(cmd)

        is_plugin, plugin_module = get_plugin_info(cmd)
        cmd_path_parts = cmd_path.split(" ")

        if len(cmd_path_parts) > 1:
            verb = cmd_path_parts[1]
            if verb not in verb_metadata:
                verb_metadata[verb] = (is_plugin, plugin_module)
            if verb not in output_verb_groups:
                output_verb_groups[verb] = []
            if len(cmd_path_parts) > 2:
                noun = cmd_path_parts[2]
                output_verb_groups[verb].append((noun, is_plugin, plugin_module))

        if len(cmd_path_parts) == 3:
            noun = cmd_path_parts[2]
            verb = cmd_path_parts[1]
            if noun not in output_noun_groups:
                output_noun_groups[noun] = []
            output_noun_groups[noun].append((verb, is_plugin, plugin_module))

        rendered = _render_command(cmd_path, cmd, cmd_ctx, is_plugin, plugin_module)
        command_data.append((cmd_path, is_plugin, plugin_module, rendered))

    # --- Output ---

    has_plugins = any(ip for _, ip, _, _ in command_data)
    use_variant_wrapping = plugin_variants and has_plugins

    # Index tables
    if use_variant_wrapping:
        # Core-only index (for non-plugin variants)
        core_noun_index = _build_index_table(output_noun_groups, None, False, include_plugins=False)
        core_verb_index = _build_index_table(output_verb_groups, verb_metadata, True, include_plugins=False)
        # Full index (for plugin variants)
        full_noun_index = _build_index_table(output_noun_groups, None, False, include_plugins=True)
        full_verb_index = _build_index_table(output_verb_groups, verb_metadata, True, include_plugins=True)

        # Core-only variant
        print()
        print(f"{{{{< variant {_non_plugin_variants(plugin_variants or '')} >}}}}")
        print("{{< grid >}}")
        print("{{< markdown >}}")
        print("\n".join(core_noun_index))
        print("{{< /markdown >}}")
        print("{{< markdown >}}")
        print("\n".join(core_verb_index))
        print("{{< /markdown >}}")
        print("{{< /grid >}}")
        print("{{< /variant >}}")

        # Full variant
        print(f"{{{{< variant {plugin_variants} >}}}}")
        print("{{< grid >}}")
        print("{{< markdown >}}")
        print("\n".join(full_noun_index))
        print("{{< /markdown >}}")
        print("{{< markdown >}}")
        print("\n".join(full_verb_index))
        print("{{< /markdown >}}")
        print("{{< /grid >}}")
        print("{{< /variant >}}")
    else:
        noun_index = _build_index_table(output_noun_groups, None, False, include_plugins=True)
        verb_index = _build_index_table(output_verb_groups, verb_metadata, True, include_plugins=True)
        print()
        print("{{< grid >}}")
        print("{{< markdown >}}")
        print("\n".join(noun_index))
        print("{{< /markdown >}}")
        print("{{< markdown >}}")
        print("\n".join(verb_index))
        print("{{< /markdown >}}")
        print("{{< /grid >}}")

    # Plugin commands install section (if plugins are present)
    if has_plugins:
        plugin_section = [
            "",
            "## Union-specific functionality {#plugin-commands}",
            "",
            "> [!NOTE]",
            "> Commands marked with **⁺** are provided by the `flyteplugins-union` plugin,",
            "> which adds Union-specific functionality to the Flyte CLI",
            "> (user management, RBAC, API keys).",
            "> Install it with `pip install flyteplugins-union`.",
            ">",
            "> See the [flyteplugins.union API reference](../integrations/union/_index)",
            "> for the programmatic interface.",
            "",
        ]

        if use_variant_wrapping:
            print(f"\n{{{{< variant {plugin_variants} >}}}}")
            print("{{< markdown >}}")
            print("\n".join(plugin_section))
            print("{{< /markdown >}}")
            print("{{< /variant >}}")
        else:
            print("\n".join(plugin_section))

    # Command detail sections
    print()
    for cmd_path, is_plugin, _pm, rendered in command_data:
        if use_variant_wrapping and is_plugin:
            print(f"\n{{{{< variant {plugin_variants} >}}}}")
            print("{{< markdown >}}")
            print("\n".join(rendered))
            print("{{< /markdown >}}")
            print("{{< /variant >}}")
        else:
            print()
            print("\n".join(rendered))

    # Flush stdout to ensure all output is written before the process exits.
    sys.stdout.flush()


def _non_plugin_variants(plugin_variants: str) -> str:
    """Derive core variant names from the page's variant list minus plugin variants.

    The page frontmatter declares all variants (e.g., +flyte +union).
    Plugin variants are the ones that should show plugin commands (e.g., union).
    This function returns the remaining variants (e.g., flyte).
    """
    # For now, we hardcode "flyte" as the core variant since that's the only
    # non-plugin variant. A more robust approach would read the page frontmatter.
    all_variants = {"flyte", "union"}
    plugin_set = set(plugin_variants.split())
    core = all_variants - plugin_set
    return " ".join(sorted(core))


def dedent(text: str) -> str:
    """
    Remove leading whitespace from a string.
    """
    return textwrap.dedent(text).strip("\n")
