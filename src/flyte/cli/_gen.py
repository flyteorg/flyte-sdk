import inspect
import sys
import textwrap
from os import getcwd
from typing import Generator, Tuple

import rich_click as click

import flyte.cli._common as common

# The distribution whose commands the Union-specific legend below describes.
UNION_PLUGIN_DIST = "flyteplugins.union"


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
    "{{< variant >}} shortcodes. Core commands appear unconditionally. "
    "Applies to any plugin not named in --plugin-variant-map.",
)
@click.option(
    "--plugin-variant-map",
    "plugin_variant_map",
    type=str,
    multiple=True,
    default=None,
    help="Per-plugin variant override, as 'module_prefix=variants' (repeatable). "
    "Example: --plugin-variant-map 'flyteplugins.hydra=flyte union'. "
    "Plugins are not all shipped to the same audience, so a single "
    "--plugin-variants value cannot describe them all.",
)
@click.option(
    "--variants",
    "all_variants",
    type=str,
    default="flyte union",
    help="Every Hugo variant the generated page declares (default: 'flyte union'). "
    "A command visible in all of them is emitted unwrapped.",
)
@click.pass_obj
def docs(
    cfg: common.CLIConfig,
    doc_type: str,
    plugin_variants: str | None,
    plugin_variant_map: tuple[str, ...] | None = None,
    all_variants: str = "flyte union",
    project: str | None = None,
    domain: str | None = None,
):
    """
    Generate documentation.
    """
    if doc_type == "markdown":
        markdown(
            cfg,
            plugin_variants=plugin_variants,
            plugin_variant_map=plugin_variant_map,
            all_variants=all_variants,
        )
    else:
        raise click.ClickException("Invalid documentation type: {}".format(doc_type))


def parse_variant_map(entries: tuple[str, ...] | list[str] | None) -> dict[str, frozenset[str]]:
    """
    Parse --plugin-variant-map entries into a module-prefix -> variants mapping.

    Each entry is 'module_prefix=variant [variant ...]', e.g.
    'flyteplugins.hydra=flyte union'.

    Raises:
        click.ClickException: on a malformed entry, rather than silently
            ignoring it and mis-gating the plugin it was meant to describe.
    """
    mapping: dict[str, frozenset[str]] = {}
    for entry in entries or ():
        prefix, sep, variants = entry.partition("=")
        prefix = prefix.strip()
        if not sep or not prefix or not variants.split():
            raise click.ClickException(
                f"Invalid --plugin-variant-map entry {entry!r}; expected 'module_prefix=variant [variant ...]'"
            )
        mapping[prefix] = frozenset(variants.split())
    return mapping


def resolve_variants(
    is_plugin: bool,
    plugin_module: str | None,
    default_plugin_variants: frozenset[str],
    variant_map: dict[str, frozenset[str]],
    all_variants: frozenset[str],
) -> frozenset[str]:
    """
    Determine which variants a command is visible in.

    Core commands are part of the base product and appear in every variant.
    A plugin command appears in the variants its distribution ships to: the
    --plugin-variant-map entry whose module prefix matches, else the blanket
    --plugin-variants value.

    Matching is on a module-path boundary, so 'flyteplugins.union' matches
    'flyteplugins.union.cli.role' but never 'flyteplugins.unionx'.
    """
    if not is_plugin or not plugin_module:
        return all_variants
    for prefix in sorted(variant_map, key=len, reverse=True):
        if plugin_module == prefix or plugin_module.startswith(prefix + "."):
            return variant_map[prefix]
    return default_plugin_variants


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
        output.append(_format_command_help(cmd.help))

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
                # Render aliases inline. The multiline shortcode emits raw
                # <div>/<br/> HTML, which fails Hugo's build inside a
                # {{< markdown >}} block (plugin commands) under goldmark
                # unsafe=false + --panicOnWarning.
                opts = " ".join(f"`{opt}`" for opt in all_opts)
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
    groups: dict[str, list[tuple[str, bool, frozenset[str]]]],
    metadata: dict[str, tuple[bool, frozenset[str]]] | None,
    is_verb_table: bool,
    variant: str | None,
    all_variants: frozenset[str],
) -> list[str]:
    """Build an index table (verb or noun) for one Hugo variant.

    Args:
        groups: verb->nouns or noun->verbs mapping; each entry carries the
            variants the command is visible in.
        metadata: verb->(is_plugin, variants) mapping (only for verb tables).
        is_verb_table: True for verb (Action/On) table, False for noun (Object/Action) table.
        variant: the variant being rendered, or None to include every command
            (used when no command is variant-restricted and the page needs one
            unwrapped index).
        all_variants: every variant the page declares.
    """
    output = []

    def visible(variants: frozenset[str]) -> bool:
        return variant is None or variant in variants

    def mark(name: str, is_plugin: bool) -> str:
        # The cross marks a plugin-provided command. Gating is a separate
        # question -- a plugin shipped to every variant is still plugin-provided.
        return f"{name}\u207a" if is_plugin else name

    if is_verb_table:
        output.append("| Action | On |")
        output.append("| ------ | -- |")
    else:
        output.append("| Object | Action |")
        output.append("| ------ | -- |")

    for key, entries in groups.items():
        if is_verb_table:
            key_is_plugin, key_variants = (
                metadata.get(key, (False, all_variants))
                if metadata
                else (
                    False,
                    all_variants,
                )
            )
            if not visible(key_variants):
                continue

            key_display = mark(key, key_is_plugin)

            filtered = [(n, ip, vs) for n, ip, vs in entries if visible(vs)]
            if not filtered:
                # The verb is visible in this variant but has no listable
                # subcommands -- either it takes none, or it builds them
                # dynamically. Still list it: dropping it makes a real command
                # unreachable from the index. `flyte fork` is exactly this shape,
                # and was only ever listed by accident, because it used to be
                # misdetected as core.
                verb_link = f"[`{key_display}`](#flyte-{key})"
                output.append(f"| {verb_link} | - |")
            else:
                noun_links = []
                for noun, noun_is_plugin, _ in filtered:
                    noun_display = mark(noun, noun_is_plugin)
                    noun_links.append(f"[`{noun_display}`](#flyte-{key}-{noun})")
                output.append(f"| `{key_display}` | {', '.join(noun_links)}  |")
        else:
            # Noun table
            filtered = [(v, ip, vs) for v, ip, vs in entries if visible(vs)]
            if not filtered:
                continue

            action_links = []
            for action, action_is_plugin, _ in filtered:
                action_display = mark(action, action_is_plugin)
                action_links.append(f"[`{action_display}`](#flyte-{action}-{key})")
            output.append(f"| `{key}` | {', '.join(action_links)}  |")

    return output


def markdown(
    cfg: common.CLIConfig,
    plugin_variants: str | None = None,
    plugin_variant_map: tuple[str, ...] | list[str] | None = None,
    all_variants: str = "flyte union",
):
    """
    Generate documentation in Markdown format.

    Args:
        cfg: CLI configuration.
        plugin_variants: Space-separated Hugo variant names for plugin commands
            not covered by plugin_variant_map. When set, those commands are
            wrapped in {{< variant >}} shortcodes.
        plugin_variant_map: Per-plugin overrides, each 'module_prefix=variants'.
        all_variants: Space-separated list of every variant the page declares.
    """
    ctx = cfg.ctx

    all_v = frozenset(all_variants.split())
    variant_map = parse_variant_map(plugin_variant_map)
    # No blanket value means plugins are not variant-restricted by default.
    default_plugin_variants = frozenset(plugin_variants.split()) if plugin_variants else all_v

    # Collect command data
    # Each entry: (cmd_path, cmd, cmd_ctx, is_plugin, plugin_module, rendered_lines)
    command_data = []
    output_verb_groups: dict[str, list[tuple[str, bool, frozenset[str]]]] = {}
    verb_metadata: dict[str, tuple[bool, frozenset[str]]] = {}
    output_noun_groups: dict[str, list[tuple[str, bool, frozenset[str]]]] = {}

    processed = []
    commands = [*[("flyte", ctx.command, ctx)], *walk_commands(ctx)]
    for cmd_path, cmd, cmd_ctx in commands:
        if cmd in processed:
            continue
        processed.append(cmd)

        is_plugin, plugin_module = get_plugin_info(cmd)
        variants = resolve_variants(is_plugin, plugin_module, default_plugin_variants, variant_map, all_v)
        cmd_path_parts = cmd_path.split(" ")

        if len(cmd_path_parts) > 1:
            verb = cmd_path_parts[1]
            if verb not in verb_metadata:
                verb_metadata[verb] = (is_plugin, variants)
            if verb not in output_verb_groups:
                output_verb_groups[verb] = []
            if len(cmd_path_parts) > 2:
                noun = cmd_path_parts[2]
                output_verb_groups[verb].append((noun, is_plugin, variants))

        if len(cmd_path_parts) == 3:
            noun = cmd_path_parts[2]
            verb = cmd_path_parts[1]
            if noun not in output_noun_groups:
                output_noun_groups[noun] = []
            output_noun_groups[noun].append((verb, is_plugin, variants))

        rendered = _render_command(cmd_path, cmd, cmd_ctx, is_plugin, plugin_module)
        command_data.append((cmd_path, is_plugin, plugin_module, variants, rendered))

    # --- Output ---

    has_plugins = any(ip for _, ip, _, _, _ in command_data)
    # Wrapping is needed only when something is actually restricted to a subset
    # of the page's variants. A plugin shipped to every variant needs no gate.
    use_variant_wrapping = any(vs != all_v for _, _, _, vs, _ in command_data)

    # Index tables: one per variant, each listing what that variant can see.
    if use_variant_wrapping:
        print()
        for variant in sorted(all_v):
            noun_index = _build_index_table(output_noun_groups, None, False, variant, all_v)
            verb_index = _build_index_table(output_verb_groups, verb_metadata, True, variant, all_v)
            print(f"{{{{< variant {variant} >}}}}")
            print("{{< grid >}}")
            print("{{< markdown >}}")
            print("\n".join(noun_index))
            print("{{< /markdown >}}")
            print("{{< markdown >}}")
            print("\n".join(verb_index))
            print("{{< /markdown >}}")
            print("{{< /grid >}}")
            print("{{< /variant >}}")
    else:
        noun_index = _build_index_table(output_noun_groups, None, False, None, all_v)
        verb_index = _build_index_table(output_verb_groups, verb_metadata, True, None, all_v)
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
        # (distribution, variants) for every plugin-provided command
        plugin_dists: list[tuple[str, frozenset[str]]] = [
            (dist, vs) for _, ip, pm, vs, _ in command_data if ip and (dist := _plugin_dist(pm))
        ]
        union_variants = frozenset().union(*[vs for dist, vs in plugin_dists if dist == UNION_PLUGIN_DIST]) or None
        other_dists = sorted({dist for dist, _ in plugin_dists if dist != UNION_PLUGIN_DIST})

        if union_variants is not None:
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
                "> See the [flyteplugins.union API reference](../union-plugin/_index)",
                "> for the programmatic interface.",
                "",
            ]

            _emit_section(plugin_section, union_variants, all_v)

        if other_dists:
            # Other plugins contribute commands too; say so rather than letting
            # the Union-specific note above imply every cross comes from Union.
            other_variants = frozenset().union(*[vs for dist, vs in plugin_dists if dist in other_dists])
            names = ", ".join(f"`{d}`" for d in other_dists)
            _emit_section(
                [
                    "",
                    "> [!NOTE]",
                    f"> Some commands marked with **\u207a** are provided by {names}.",
                    "",
                ],
                other_variants,
                all_v,
            )

    # Command detail sections
    print()
    for cmd_path, is_plugin, _pm, variants, rendered in command_data:
        if use_variant_wrapping and variants != all_v:
            print(f"\n{{{{< variant {' '.join(sorted(variants))} >}}}}")
            print("{{< markdown >}}")
            print("\n".join(rendered))
            print("{{< /markdown >}}")
            print("{{< /variant >}}")
        else:
            print()
            print("\n".join(rendered))

    # Flush stdout to ensure all output is written before the process exits.
    sys.stdout.flush()


def _plugin_dist(module: str | None) -> str | None:
    """Reduce a plugin module path to its distribution prefix, e.g. 'flyteplugins.union'."""
    if not module:
        return None
    parts = module.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else module


def _emit_section(lines: list[str], variants: frozenset[str], all_variants: frozenset[str]) -> None:
    """Print a block, wrapped in a variant shortcode unless it is visible everywhere."""
    if variants != all_variants:
        print(f"\n{{{{< variant {' '.join(sorted(variants))} >}}}}")
        print("{{< markdown >}}")
        print("\n".join(lines))
        print("{{< /markdown >}}")
        print("{{< /variant >}}")
    else:
        print("\n".join(lines))


def dedent(text: str) -> str:
    """
    Remove leading whitespace from a string.
    """
    return textwrap.dedent(text).strip("\n")


def _format_command_help(text: str) -> str:
    """Render a command's help text as Markdown.

    click help strings put the first line on the opening triple-quote (column 0)
    and indent the rest, which `textwrap.dedent` cannot normalize;
    `inspect.cleandoc` handles that. Any remaining indented blocks (e.g.
    `Examples:` command listings) are wrapped in fenced code blocks rather than
    left as indentation-based code blocks — indented code does not survive the
    `{{< markdown >}}` shortcode's `RenderString` and renders inconsistently.
    """
    lines = inspect.cleandoc(text).split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("    "):
            block: list[str] = []
            while i < len(lines) and (lines[i].startswith("    ") or lines[i].strip() == ""):
                block.append(lines[i])
                i += 1
            while block and block[-1].strip() == "":
                block.pop()
            out.append("```bash")
            out.extend(textwrap.dedent("\n".join(block)).split("\n"))
            out.append("```")
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)
