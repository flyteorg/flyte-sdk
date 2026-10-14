"""`--type markdown` describes the CLI, in Markdown, and nothing else.

It used to emit Hugo for one documentation site: shortcodes, `{{< variant >}}`
gating driven by a `--plugin-variants` flag, two workarounds for that site's
goldmark configuration, and a hardcoded set of audience names. An SDK maintainer
had no way to verify any of it, and a purely presentational bug in it needed an
eng PR and an SDK release to reach readers (DOC-1478).

That rendering moved to the repository that owns the site, which consumes
`--type json` (DOC-1481). What is left here is plain Markdown.

The line these tests hold: **where a command came from is a fact and stays; who
should see it is a decision and is gone.** So a plugin command still carries its
plus marker and names its distribution, and nothing anywhere decides an audience.
"""

import click
import pytest

from flyte.cli._gen import _build_index_table, markdown
from flyte.cli._plugins import _stamp_distribution


class _EP:
    def __init__(self, name):
        self.dist = type("D", (), {"name": name})()


class _Cfg:
    def __init__(self, root):
        self.ctx = click.Context(root, info_name="flyte")


def _root_with_plugin():
    root = click.Group(name="flyte")

    @root.command("run")
    @click.option("--project", help="Project.")
    def run(): ...

    fork = click.Group(name="fork")  # dispatch-only, no callback: the DOC-1478 shape
    _stamp_distribution(fork, _EP("flyteplugins-union"))
    root.add_command(fork, name="fork")

    get = click.Group(name="get")

    @get.command("api-key")
    def api_key(): ...

    _stamp_distribution(api_key, _EP("flyteplugins-union"))
    root.add_command(get, name="get")
    return root


def _render(capsys, root=None):
    markdown(_Cfg(root or _root_with_plugin()))
    return capsys.readouterr().out


# --- the point of the change -------------------------------------------------


@pytest.mark.parametrize("construct", ["{{<", "{{%", "variant", "grid"])
def test_no_hugo_reaches_the_output(capsys, construct):
    assert construct not in _render(capsys)


def test_help_text_is_not_rewritten_into_fenced_code(capsys):
    """The fencing existed because indented code did not survive one site's
    `{{< markdown >}}` shortcode. Indented blocks are valid Markdown."""
    root = click.Group(name="flyte")

    @root.command("x")
    def x():
        """Examples:

        this is indented
        """

    assert "```bash" not in _render(capsys, root)


def test_hugo_delimiters_in_help_are_left_alone(capsys):
    """Escaping them served one renderer. Here they are just characters."""
    root = click.Group(name="flyte")

    @root.command("x")
    @click.option("--x", help="Write {{< thing >}} to do it.")
    def x(): ...

    out = _render(capsys, root)
    assert "{{&lt;" not in out


# --- what must survive: provenance -------------------------------------------


def test_a_plugin_command_names_its_distribution(capsys):
    out = _render(capsys)
    assert "> **Note:** This command is provided by the `flyteplugins-union` plugin." in out


def test_a_core_command_carries_no_note(capsys):
    out = _render(capsys)
    run = out[out.index("### flyte run") :]
    assert "provided by" not in run[: run.index("**`flyte run")]


def test_plugin_commands_are_marked_in_the_index(capsys):
    out = _render(capsys)
    assert "[`api-key⁺`]" in out or "[`get⁺`]" in out


def test_the_notice_names_the_installed_distributions(capsys):
    out = _render(capsys)
    assert "## Plugin commands" in out
    assert "`flyteplugins-union`" in out
    # Not a statement about who may see the commands, and not about one vendor.
    assert "Union-specific" not in out


def test_nothing_decides_an_audience(capsys):
    """No gating survives: every command is in the one document."""
    out = _render(capsys)
    assert "### flyte fork" in out
    assert "### flyte run" in out


# --- index rules, carried over ------------------------------------------------


def test_a_verb_with_no_subcommands_is_still_listed():
    """`fork` builds its subcommands dynamically. Dropping it from the index
    makes a documented command unreachable."""
    rows = _build_index_table({"fork": []}, {"fork": True}, True)
    assert any("[`fork⁺`](#flyte-fork)" in r for r in rows)


def test_a_core_verb_with_no_subcommands_is_listed_without_a_marker():
    rows = _build_index_table({"build": []}, {"build": False}, True)
    assert any("[`build`](#flyte-build)" in r for r in rows)
    assert not any("⁺" in r for r in rows)


def test_subcommands_are_linked_under_their_verb():
    rows = _build_index_table({"get": [("run", False), ("api-key", True)]}, {"get": False}, True)
    row = next(r for r in rows if r.startswith("| `get`"))
    assert "[`run`](#flyte-get-run)" in row
    assert "[`api-key⁺`](#flyte-get-api-key)" in row
