"""Tests for surfacing ActionMetadata.relation (related_to + relation_type) in run/action output.

Relation-bearing cases are gated on the flyteidl2 build shipping ActionMetadata.relation; the
unset-relation behavior (empty column, None property) must hold on every build.
"""

import pytest
from flyteidl2.workflow import run_definition_pb2

from flyte.remote._action import (
    _RELATION_SUPPORTED,
    Action,
    ActionDetails,
    _action_details_rich_repr,
    _action_rich_repr,
    _relation_repr,
)

needs_relation = pytest.mark.skipif(not _RELATION_SUPPORTED, reason="flyteidl2 build lacks ActionMetadata.relation")


def test_relation_repr_empty_when_unset():
    assert _relation_repr(run_definition_pb2.ActionMetadata()) == ""


def test_action_rich_repr_always_has_related_to_column():
    action = run_definition_pb2.Action()
    keys = [k for k, _ in _action_rich_repr(action)]
    assert "related to" in keys
    assert dict(_action_rich_repr(action))["related to"] == ""

    details = run_definition_pb2.ActionDetails()
    keys = [k for k, _ in _action_details_rich_repr(details)]
    assert "related to" in keys
    assert dict(_action_details_rich_repr(details))["related to"] == ""


def test_relation_property_none_when_unset():
    assert Action(pb2=run_definition_pb2.Action()).relation is None
    assert ActionDetails(pb2=run_definition_pb2.ActionDetails()).relation is None


def _metadata_with_relation(kind) -> run_definition_pb2.ActionMetadata:
    from flyteidl2.common import identifier_pb2
    from flyteidl2.common import run_pb2 as common_run_pb2

    return run_definition_pb2.ActionMetadata(
        relation=common_run_pb2.Relation(
            related_to=identifier_pb2.RunIdentifier(name="parent-run"),
            relation_type=kind,
        )
    )


@needs_relation
@pytest.mark.parametrize("kind,expected", [(1, "rerun of parent-run"), (2, "recover of parent-run")])
def test_relation_repr_formats_kind_and_parent(kind, expected):
    assert _relation_repr(_metadata_with_relation(kind)) == expected


@needs_relation
def test_relation_appears_in_rich_reprs():
    metadata = _metadata_with_relation(1)
    action = run_definition_pb2.Action(metadata=metadata)
    assert dict(_action_rich_repr(action))["related to"] == "rerun of parent-run"

    details = run_definition_pb2.ActionDetails(metadata=metadata)
    assert dict(_action_details_rich_repr(details))["related to"] == "rerun of parent-run"


@needs_relation
def test_relation_property_returns_proto():
    action = Action(pb2=run_definition_pb2.Action(metadata=_metadata_with_relation(2)))
    assert action.relation is not None
    assert action.relation.related_to.name == "parent-run"
    assert action.relation.relation_type == 2

    details = ActionDetails(pb2=run_definition_pb2.ActionDetails(metadata=_metadata_with_relation(2)))
    assert details.relation is not None
    assert details.relation.related_to.name == "parent-run"
