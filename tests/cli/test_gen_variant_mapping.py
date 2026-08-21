"""Tests for per-plugin variant resolution in the CLI doc generator.

The generator used to apply one --plugin-variants value to every plugin, which
silently mis-gates any plugin shipped to a different audience than the others.
"""

import pytest
import rich_click as click

from flyte.cli._gen import _build_index_table, parse_variant_map, resolve_variants

ALL = frozenset({"flyte", "union"})
UNION_ONLY = frozenset({"union"})


def test_core_commands_appear_in_every_variant():
    assert resolve_variants(False, None, UNION_ONLY, {}, ALL) == ALL


def test_unmapped_plugin_falls_back_to_the_blanket_value():
    got = resolve_variants(True, "flyteplugins.union.cli.role", UNION_ONLY, {}, ALL)
    assert got == UNION_ONLY


def test_mapped_plugin_overrides_the_blanket_value():
    # The regression this change exists for: an OSS plugin must not be hidden
    # from the OSS variant just because another plugin is Union-only.
    vmap = {"flyteplugins.hydra": ALL}
    assert resolve_variants(True, "flyteplugins.hydra._cli", UNION_ONLY, vmap, ALL) == ALL
    # ...while the Union plugin stays restricted.
    assert resolve_variants(True, "flyteplugins.union.cli.user", UNION_ONLY, vmap, ALL) == UNION_ONLY


def test_prefix_matching_respects_module_boundaries():
    vmap = {"flyteplugins.union": UNION_ONLY}
    # A different distribution that merely shares a string prefix must not match.
    assert resolve_variants(True, "flyteplugins.unionx.cli", ALL, vmap, ALL) == ALL


def test_exact_module_matches_its_own_prefix():
    vmap = {"flyteplugins.hydra": ALL}
    assert resolve_variants(True, "flyteplugins.hydra", UNION_ONLY, vmap, ALL) == ALL


def test_longest_prefix_wins():
    vmap = {"flyteplugins": UNION_ONLY, "flyteplugins.hydra": ALL}
    assert resolve_variants(True, "flyteplugins.hydra._cli", UNION_ONLY, vmap, ALL) == ALL
    assert resolve_variants(True, "flyteplugins.other._cli", ALL, vmap, ALL) == UNION_ONLY


def test_plugin_with_no_module_is_treated_as_core():
    assert resolve_variants(True, None, UNION_ONLY, {}, ALL) == ALL


@pytest.mark.parametrize("entry", ["flyteplugins.hydra", "=flyte union", "flyteplugins.hydra=", ""])
def test_malformed_map_entries_are_rejected_loudly(entry):
    # Silently dropping a malformed entry would mis-gate the plugin it named,
    # which is the failure mode this option exists to prevent.
    with pytest.raises(click.ClickException):
        parse_variant_map([entry])


def test_map_parsing_round_trip():
    got = parse_variant_map(["flyteplugins.hydra=flyte union", "flyteplugins.union=union"])
    assert got == {"flyteplugins.hydra": ALL, "flyteplugins.union": UNION_ONLY}


def test_none_and_empty_parse_to_an_empty_map():
    assert parse_variant_map(None) == {}
    assert parse_variant_map(()) == {}


def test_index_table_hides_commands_not_in_the_rendered_variant():
    groups = {"get": [("run", False, ALL), ("api-key", True, UNION_ONLY)]}
    metadata = {"get": (False, ALL)}

    flyte_rows = "\n".join(_build_index_table(groups, metadata, True, "flyte", ALL))
    union_rows = "\n".join(_build_index_table(groups, metadata, True, "union", ALL))

    assert "api-key" not in flyte_rows
    assert "run" in flyte_rows
    assert "api-key" in union_rows


def test_index_table_marks_plugin_commands_wherever_they_are_visible():
    # A plugin shipped to both variants is still plugin-provided, so it keeps
    # its marker in the OSS index rather than looking like a core command.
    groups = {"hydra": [("run", True, ALL)]}
    metadata = {"hydra": (True, ALL)}
    rows = "\n".join(_build_index_table(groups, metadata, True, "flyte", ALL))
    assert "hydra⁺" in rows


def test_core_verb_with_no_visible_nouns_is_still_listed():
    # 'flyte create' is a core verb whose subcommands are all plugin-provided;
    # dropping it from the OSS index would lose a real command.
    groups = {"create": [("api-key", True, UNION_ONLY)]}
    metadata = {"create": (False, ALL)}
    rows = "\n".join(_build_index_table(groups, metadata, True, "flyte", ALL))
    assert "flyte-create" in rows
    assert "api-key" not in rows


def test_plugin_verb_with_no_subcommands_is_still_listed():
    # A plugin group that builds its subcommands dynamically has no entries to
    # filter. Dropping it leaves a documented command with no way to reach it
    # from the index -- `flyte fork` and `flyte debug` are both this shape.
    groups = {"fork": []}
    metadata = {"fork": (True, UNION_ONLY)}
    rows = "\n".join(_build_index_table(groups, metadata, True, "union", ALL))
    assert "fork\u207a" in rows
    assert "#flyte-fork" in rows


def test_a_verb_absent_from_this_variant_is_not_listed():
    groups = {"fork": []}
    metadata = {"fork": (True, UNION_ONLY)}
    rows = "\n".join(_build_index_table(groups, metadata, True, "flyte", ALL))
    assert "fork" not in rows
