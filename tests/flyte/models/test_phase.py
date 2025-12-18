import pytest
from flyteidl2.common import phase_pb2

from flyte.models import ActionPhase


class TestActionPhase:
    def test_enum_values(self):
        """Test all enum values are correct."""
        assert ActionPhase.QUEUED.value == "queued"
        assert ActionPhase.WAITING_FOR_RESOURCES.value == "waiting_for_resources"
        assert ActionPhase.INITIALIZING.value == "initializing"
        assert ActionPhase.RUNNING.value == "running"
        assert ActionPhase.SUCCEEDED.value == "succeeded"
        assert ActionPhase.FAILED.value == "failed"
        assert ActionPhase.ABORTED.value == "aborted"
        assert ActionPhase.TIMED_OUT.value == "timed_out"

    def test_string_comparison(self):
        """Test that enum can be compared to strings."""
        assert ActionPhase.QUEUED == "queued"
        assert ActionPhase.SUCCEEDED == "succeeded"
        assert ActionPhase.FAILED == "failed"

    def test_is_terminal(self):
        """Test is_terminal property."""
        assert ActionPhase.SUCCEEDED.is_terminal is True
        assert ActionPhase.FAILED.is_terminal is True
        assert ActionPhase.ABORTED.is_terminal is True
        assert ActionPhase.TIMED_OUT.is_terminal is True

        assert ActionPhase.QUEUED.is_terminal is False
        assert ActionPhase.RUNNING.is_terminal is False
        assert ActionPhase.INITIALIZING.is_terminal is False
        assert ActionPhase.WAITING_FOR_RESOURCES.is_terminal is False

    def test_to_protobuf_name(self):
        """Test conversion to protobuf enum name."""
        assert ActionPhase.QUEUED.to_protobuf_name() == "ACTION_PHASE_QUEUED"
        assert ActionPhase.WAITING_FOR_RESOURCES.to_protobuf_name() == "ACTION_PHASE_WAITING_FOR_RESOURCES"
        assert ActionPhase.SUCCEEDED.to_protobuf_name() == "ACTION_PHASE_SUCCEEDED"
        assert ActionPhase.TIMED_OUT.to_protobuf_name() == "ACTION_PHASE_TIMED_OUT"

    def test_to_protobuf_value(self):
        """Test conversion to protobuf enum value."""
        assert ActionPhase.QUEUED.to_protobuf_value() == 1
        assert ActionPhase.WAITING_FOR_RESOURCES.to_protobuf_value() == 2
        assert ActionPhase.INITIALIZING.to_protobuf_value() == 3
        assert ActionPhase.RUNNING.to_protobuf_value() == 4
        assert ActionPhase.SUCCEEDED.to_protobuf_value() == 5
        assert ActionPhase.FAILED.to_protobuf_value() == 6
        assert ActionPhase.ABORTED.to_protobuf_value() == 7
        assert ActionPhase.TIMED_OUT.to_protobuf_value() == 8

    def test_from_protobuf(self):
        """Test creation from protobuf enum."""
        assert ActionPhase.from_protobuf(phase_pb2.ACTION_PHASE_QUEUED) == ActionPhase.QUEUED
        assert (
            ActionPhase.from_protobuf(phase_pb2.ACTION_PHASE_WAITING_FOR_RESOURCES) == ActionPhase.WAITING_FOR_RESOURCES
        )
        assert ActionPhase.from_protobuf(phase_pb2.ACTION_PHASE_SUCCEEDED) == ActionPhase.SUCCEEDED
        assert ActionPhase.from_protobuf(phase_pb2.ACTION_PHASE_TIMED_OUT) == ActionPhase.TIMED_OUT

    def test_from_protobuf_unspecified_raises(self):
        """Test that UNSPECIFIED phase raises error."""
        with pytest.raises(ValueError, match="Cannot convert UNSPECIFIED"):
            ActionPhase.from_protobuf(phase_pb2.ACTION_PHASE_UNSPECIFIED)

    def test_enum_iteration(self):
        """Test that all 8 phases are present."""
        phases = list(ActionPhase)
        assert len(phases) == 8

    def test_string_conversion(self):
        """Test that enum values behave as strings."""
        # The enum inherits from str, so the value itself is a string
        assert ActionPhase.QUEUED.value == "queued"
        assert ActionPhase.SUCCEEDED.value == "succeeded"
        assert ActionPhase.RUNNING.value == "running"

        # Can be used in string contexts
        assert ActionPhase.QUEUED == "queued"
        assert f"{ActionPhase.SUCCEEDED}" == "ActionPhase.SUCCEEDED"  # str() gives enum name

    def test_enum_can_be_used_in_sets(self):
        """Test that enum can be used in sets and dictionaries."""
        phases_set = {ActionPhase.QUEUED, ActionPhase.RUNNING, ActionPhase.SUCCEEDED}
        assert ActionPhase.QUEUED in phases_set
        assert ActionPhase.FAILED not in phases_set

    def test_enum_comparison(self):
        """Test enum comparison."""
        assert ActionPhase.QUEUED == ActionPhase.QUEUED
        assert ActionPhase.QUEUED != ActionPhase.RUNNING

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion from enum to protobuf and back."""
        for phase in ActionPhase:
            # Convert to protobuf enum using the value
            pb_phase = phase_pb2.ActionPhase.Value(phase.to_protobuf_name())
            # Convert back to ActionPhase
            result = ActionPhase.from_protobuf(pb_phase)
            assert result == phase
