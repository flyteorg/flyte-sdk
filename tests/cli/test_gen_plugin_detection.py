"""`get_plugin_info` must key off where a command was defined, not whether it has a callback.

A `click.Group` that only dispatches to subcommands has `callback is None`. That is an
ordinary construct and says nothing about provenance, but the detection used to return
early on it — reporting plugin-provided groups as core. Downstream that dropped the
plugin marker and the "provided by" note from the generated CLI reference, and, because
the variant filter keys off the same flag, emitted the command into the OSS Flyte
variant.

`flyte fork` hit this: `flyteplugins-union` registers it as
`fork = ForkFiles(name="fork", ...)`, a Group subclass instantiated directly. `flyte debug`
from the same package did not, only because it is a decorated function and so carries a
callback. The two differ in construction style alone.
"""

import click

from flyte.cli._gen import get_plugin_info


def _plugin_group(module: str) -> click.Group:
    """A dispatch-only group defined in `module`, as a plugin registers one."""
    return type("ForkFiles", (click.Group,), {"__module__": module})(name="fork")


def _command(module: str) -> click.Command:
    @click.command("c")
    def _c() -> None: ...

    _c.callback.__module__ = module
    return _c


def test_plugin_group_without_callback_is_detected():
    """The regression: was (False, None), so `flyte fork` shipped as core."""
    assert get_plugin_info(_plugin_group("flyteplugins.union.cli.fork")) == (
        True,
        "flyteplugins.union",
    )


def test_plugin_command_with_callback_still_detected():
    """Control: `flyte debug`, same package, decorated. Must not regress."""
    assert get_plugin_info(_command("flyteplugins.union.cli.debug")) == (
        True,
        "flyteplugins.union",
    )


def test_core_command_stays_core():
    assert get_plugin_info(_command("flyte.cli._run")) == (False, None)


def test_core_group_subclass_without_callback_stays_core():
    """A dispatch-only group defined inside core must not become a plugin."""
    CoreGroup = type("CoreGroup", (click.Group,), {"__module__": "flyte.cli._run"})
    assert get_plugin_info(CoreGroup(name="g")) == (False, None)


def test_unsubclassed_group_is_not_reported_as_a_click_plugin():
    """`click.Group()` resolves to click's own module, which is not provenance.

    Without the guard this returns (True, "click.core") — a plugin named after click.
    """
    assert get_plugin_info(click.Group(name="g")) == (False, None)


def test_none_is_handled():
    assert get_plugin_info(None) == (False, None)


def test_plugin_verb_with_no_subcommands_is_still_indexed():
    """Detecting a plugin group must not delete it from the index.

    A group that builds its subcommands dynamically has no entries to filter.
    The index used to list such a verb only when it was NOT plugin-provided, so
    correcting `fork`'s detection would have moved it from "listed in the OSS
    index" (wrong table) to "listed in no index at all" (worse).
    """
    from flyte.cli._gen import _build_index_table

    groups = {"fork": []}
    metadata = {"fork": (True, "flyteplugins.union.cli.fork")}
    rows = "\n".join(_build_index_table(groups, metadata, True, include_plugins=True))
    assert "fork⁺" in rows
    assert "#flyte-fork" in rows
