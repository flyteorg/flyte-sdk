"""ACTION_PHASE_RECOVERED must be terminal and success-equivalent everywhere the SDK
classifies phases — a recovered child never runs in this run, so the watch stream is the
only signal; if it isn't terminal the controller re-watches forever and the run hangs.
"""

from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.core import literals_pb2
from flyteidl2.workflow import state_service_pb2

from flyte._internal.controllers.remote._action import Action


def _action_id(name: str = "a1") -> identifier_pb2.ActionIdentifier:
    return identifier_pb2.ActionIdentifier(
        name=name,
        run=identifier_pb2.RunIdentifier(org="org", project="proj", domain="dev", name="run1"),
    )


def test_recovered_is_terminal():
    action = Action(action_id=_action_id(), parent_action_name="a0")
    action.phase = phase_pb2.ACTION_PHASE_RECOVERED
    assert action.is_terminal()


def test_merge_state_recovered_update_resolves_with_source_outputs():
    """Live-update path: a RECOVERED ActionUpdate makes the action terminal, success (no
    error), with output_uri pointing at the SOURCE run's outputs (consumed as-is)."""
    action = Action(action_id=_action_id(), parent_action_name="a0")
    action.merge_state(
        state_service_pb2.ActionUpdate(
            action_id=_action_id(),
            phase=phase_pb2.ACTION_PHASE_RECOVERED,
            output_uri="s3://bucket/source-run/a1/outputs",
        )
    )
    assert action.is_terminal()
    assert action.err is None
    assert action.realized_outputs_uri == "s3://bucket/source-run/a1/outputs"


def test_from_state_recovered_snapshot_resolves():
    """Snapshot path: a RECOVERED action first observed via the watch's initial snapshot
    (never submitted locally) is immediately terminal."""
    action = Action.from_state(
        "a0",
        state_service_pb2.ActionUpdate(
            action_id=_action_id(),
            phase=phase_pb2.ACTION_PHASE_RECOVERED,
            output_uri="s3://bucket/source-run/a1/outputs",
        ),
    )
    assert action.is_terminal()
    assert action.err is None
    assert action.realized_outputs_uri == "s3://bucket/source-run/a1/outputs"


def test_merge_state_recovered_condition_carries_value():
    """A recovered condition delivers its signaled Literal inline on ActionUpdate.value."""
    action = Action(action_id=_action_id("cond1"), parent_action_name="a0", type="condition")
    lit = literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(boolean=True)))
    action.merge_state(
        state_service_pb2.ActionUpdate(
            action_id=_action_id("cond1"),
            phase=phase_pb2.ACTION_PHASE_RECOVERED,
            value=lit,
        )
    )
    assert action.is_terminal()
    assert action.condition_output == lit


def test_remote_action_done_check_recovered():
    from flyte.remote._action import _action_done_check

    assert _action_done_check(phase_pb2.ACTION_PHASE_RECOVERED)
