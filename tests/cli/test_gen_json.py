"""The machine-readable half of the CLI doc generator, and the provenance it reports.

`--type json` exists so that whoever renders the CLI reference is given facts
rather than inferences. The fact that matters most is which distribution
provided each command, because the rendered page shows Union-only commands to a
different audience than open-source ones, and getting that wrong publishes a
command the reader cannot run (DOC-1478: `flyte fork`).

Two properties are easy to lose in a refactor and are pinned here:

  * Provenance is RECORDED, not derived. `_plugins.py` stamps the distribution
    at registration, where the entry point is in hand. Nothing consults
    `__module__`, which describes where code was written rather than what
    shipped it -- and which said nothing at all about a `click.Group` with no
    callback, which is how `flyte fork` reached the open-source docs.
  * Provenance is INHERITED. Only 35 plugin commands are entry points;
    `flyte explore volume` and the three `flyte undelete` subcommands live
    inside a plugin's own group and carry no stamp. Attributing by entry point
    alone reports them as core, reintroducing the defect from the other side.
"""

import os

import click
import pytest

import flyte.cli._common as common
from flyte.cli._gen import (
    _describe,
    _option_default,
    _resolve_distribution,
    walk_commands,
)
from flyte.cli._plugins import (
    PLUGIN_DISTRIBUTION_ATTR,
    _stamp_distribution,
    get_command_distribution,
)


class _EP:
    """Minimal stand-in for an entry point, which carries `.dist.name`."""

    def __init__(self, dist_name):
        self.dist = type("D", (), {"name": dist_name})() if dist_name else None


# --- stamping ----------------------------------------------------------------


def test_a_core_command_has_no_distribution():
    @click.command("run")
    def run(): ...

    assert get_command_distribution(run) is None


def test_registration_records_the_distribution():
    @click.command("fork")
    def fork(): ...

    _stamp_distribution(fork, _EP("flyteplugins-union"))
    assert get_command_distribution(fork) == "flyteplugins-union"


def test_a_callbackless_group_is_stamped_like_any_other():
    """The `flyte fork` shape. It has no callback, so `__module__` was never
    reached and it was reported as core. Registration does not care."""
    group = click.Group(name="fork")
    assert group.callback is None
    _stamp_distribution(group, _EP("flyteplugins-union"))
    assert get_command_distribution(group) == "flyteplugins-union"


def test_an_entry_point_without_distribution_metadata_is_left_unstamped():
    """Unattributed beats mislabelled: a missing distribution must not become a
    guess, or the check downstream inherits a fiction."""
    cmd = click.Group(name="x")
    _stamp_distribution(cmd, _EP(None))
    assert get_command_distribution(cmd) is None
    assert not hasattr(cmd, PLUGIN_DISTRIBUTION_ATTR)


def test_a_second_registration_does_not_reattribute():
    cmd = click.Group(name="x")
    _stamp_distribution(cmd, _EP("flyteplugins-union"))
    _stamp_distribution(cmd, _EP("flyteplugins-other"))
    assert get_command_distribution(cmd) == "flyteplugins-union"


# --- inheritance -------------------------------------------------------------


def test_own_stamp_wins():
    by_path = {"flyte fork": "flyteplugins-union"}
    assert _resolve_distribution("flyte fork", "flyteplugins-union", by_path) == "flyteplugins-union"


def test_a_subcommand_inherits_from_its_plugin_group():
    """`flyte explore volume` is not an entry point. Its parent is."""
    by_path = {"flyte explore": "flyteplugins-union", "flyte explore volume": None}
    assert _resolve_distribution("flyte explore volume", None, by_path) == "flyteplugins-union"


def test_inheritance_reaches_past_an_unstamped_ancestor():
    by_path = {"flyte a": "flyteplugins-union", "flyte a b": None, "flyte a b c": None}
    assert _resolve_distribution("flyte a b c", None, by_path) == "flyteplugins-union"


def test_a_core_subcommand_of_a_core_group_stays_core():
    by_path = {"flyte get": None, "flyte get run": None}
    assert _resolve_distribution("flyte get run", None, by_path) is None


def test_a_plugin_subcommand_of_a_core_group_keeps_its_own_stamp():
    """The dotted form: `get.api-key` attaches to the CORE `get` group. The
    parent must not launder it back to core."""
    by_path = {"flyte get": None, "flyte get api-key": "flyteplugins-union"}
    assert _resolve_distribution("flyte get api-key", "flyteplugins-union", by_path) == "flyteplugins-union"


# --- what a command looks like as data ---------------------------------------


def _ctx(cmd):
    return click.Context(cmd, info_name=cmd.name)


def test_describe_reports_options_arguments_and_provenance():
    @click.command("api-key")
    @click.argument("name", required=True)
    @click.argument("note", required=False)
    @click.option("-p", "--project", type=str, help="Project to which this applies.")
    @click.option("--flag/--no-flag", default=False)
    def cmd(): ...

    out = _describe("flyte create api-key", cmd, _ctx(cmd), "flyteplugins-union")

    assert out["path"] == "flyte create api-key"
    assert out["name"] == "api-key"
    assert out["is_group"] is False
    assert out["distribution"] == "flyteplugins-union"
    assert out["arguments"] == [
        {"name": "name", "required": True},
        {"name": "note", "required": False},
    ]
    project = next(o for o in out["options"] if "--project" in o["opts"])
    assert project["opts"] == ["-p", "--project"]
    assert project["type"] == "text"
    assert project["help"] == "Project to which this applies."
    flag = next(o for o in out["options"] if "--flag" in o["opts"])
    assert flag["secondary_opts"] == ["--no-flag"]


def test_declares_options_separates_a_command_own_options_from_click_help():
    """click adds `--help` everywhere, so `options` being non-empty says nothing.

    A renderer needs this to decide whether to advertise `[OPTIONS]` in the usage
    line and whether an option table carries information. Reconstructing it by
    assuming `--help` is the only thing click ever adds is the kind of guess this
    output exists to remove.
    """

    @click.command("bare")
    def bare(): ...

    @click.command("has-one")
    @click.option("--x")
    def has_one(): ...

    @click.command("only-an-argument")
    @click.argument("name")
    def only_arg(): ...

    bare_out = _describe("flyte bare", bare, _ctx(bare), None)
    assert bare_out["declares_options"] is False
    assert [o["opts"] for o in bare_out["options"]] == [["--help"]]

    assert _describe("flyte has-one", has_one, _ctx(has_one), None)["declares_options"] is True

    # Declares a parameter, but not an option: the two questions differ.
    arg_out = _describe("flyte only-an-argument", only_arg, _ctx(only_arg), None)
    assert arg_out["declares_options"] is False
    assert arg_out["arguments"] == [{"name": "name", "required": True}]


def test_a_group_is_marked_as_one():
    grp = click.Group(name="get")
    assert _describe("flyte get", grp, _ctx(grp), None)["is_group"] is True


def test_help_is_normalised_but_not_marked_up():
    """`inspect.cleandoc` undoes click's indentation, which is an artefact of how
    the docstring was written. Everything past that -- escaping, code fencing --
    is presentation and belongs to the consumer, not here."""

    @click.command("x")
    def cmd():
        """First line.

            indented block

        Mentions {{< shortcode >}} literally.
        """

    out = _describe("flyte x", cmd, _ctx(cmd), None)
    assert out["help"].startswith("First line.")
    assert "    indented block" in out["help"]
    assert "{{< shortcode >}}" in out["help"]
    assert "```" not in out["help"]


def test_defaults_are_json_safe_and_drop_the_build_directory():
    @click.command("x")
    @click.option("--n", type=int, default=3)
    @click.option("--on", is_flag=True, default=False)
    @click.option("--path", type=str, default=f"{os.getcwd()}/somewhere")
    def cmd(): ...

    opts = {o["opts"][0]: o["default"] for o in _describe("flyte x", cmd, _ctx(cmd), None)["options"]}
    assert opts["--n"] == 3
    assert opts["--on"] is False
    assert opts["--path"] == "somewhere"


def test_option_default_stringifies_an_arbitrary_object():
    class Sentinel:
        def __str__(self):
            return "UNSET"

    param = click.Option(["--x"], default=Sentinel())
    assert _option_default(param) == "UNSET"


# --- the walker, which stays in this repo and had no tests at all -------------


def test_walk_yields_subcommands_of_a_plain_group():
    grp = click.Group(name="get")

    @grp.command("run")
    def run(): ...

    paths = [p for p, _, _ in walk_commands(click.Context(grp, info_name="flyte"))]
    assert paths == ["flyte run"]


def test_walk_does_not_descend_into_remotetaskgroup():
    """Matched by class NAME, with no import. Enumerating it needs a live
    backend, so descending hangs or fails at doc-generation time. A rename
    upstream silently re-enables that, which is what this pins."""
    RemoteTaskGroup = type("RemoteTaskGroup", (click.Group,), {})
    inner = RemoteTaskGroup(name="task")

    @inner.command("should-not-appear")
    def hidden(): ...

    outer = click.Group(name="get")
    outer.add_command(inner, name="task")

    paths = [p for p, _, _ in walk_commands(click.Context(outer, info_name="flyte"))]
    assert "flyte task" in paths
    assert "flyte task should-not-appear" not in paths


def test_walk_does_not_enumerate_a_filegroups_files(tmp_path):
    """A FileGroup builds one subcommand per .py file in the working directory.
    Those are the user's files, not the CLI's surface, so the walk stops."""
    (tmp_path / "user_script.py").write_text("")
    fg = common.FileGroup(name="run", directory=tmp_path)
    outer = click.Group(name="root")
    outer.add_command(fg, name="run")

    paths = [p for p, _, _ in walk_commands(click.Context(outer, info_name="flyte"))]
    assert "flyte run" in paths
    assert not any("user_script" in p for p in paths)


@pytest.mark.parametrize("name", ["TaskFiles", "RemoteTaskGroup"])
def test_the_private_type_names_the_walker_matches_on_still_exist(name):
    """`walk_commands` matches these by `type(...).__name__`, so a rename breaks
    doc generation with no import to fail first. If this test fails, the walker
    needs updating -- that is the point of it failing here rather than silently
    at the next regen."""
    import flyte.cli._common as c
    import flyte.cli._run as r

    assert any(getattr(m, name, None) is not None for m in (c, r)), (
        f"{name} was not found; walk_commands matches it by class-name string"
    )
