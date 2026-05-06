"""Tests for action phase transition functionality."""

import datetime
from datetime import timezone

from flyteidl2.common import phase_pb2
from flyteidl2.workflow import run_definition_pb2
from google.protobuf.timestamp_pb2 import Timestamp

from flyte.models import ActionPhase
from flyte.remote._action import ActionDetails, PhaseTransitionInfo


class TestPhaseTransitionInfo:
    """Test the PhaseTransitionInfo dataclass."""

    def test_duration_with_end_time(self):
        """Test duration calculation when end_time is set."""
        start = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime.datetime(2024, 1, 1, 12, 5, 30, tzinfo=timezone.utc)

        transition = PhaseTransitionInfo(
            phase=ActionPhase.QUEUED,
            start_time=start,
            end_time=end,
        )

        assert transition.duration == datetime.timedelta(seconds=330)

    def test_duration_without_end_time(self):
        """Test duration calculation when end_time is None (ongoing phase)."""
        start = datetime.datetime.now(timezone.utc) - datetime.timedelta(seconds=10)

        transition = PhaseTransitionInfo(
            phase=ActionPhase.RUNNING,
            start_time=start,
            end_time=None,
        )

        # Duration should be approximately 10 seconds
        assert 9 <= transition.duration.total_seconds() <= 11


class TestActionDetailsGetPhaseTransitions:
    """Test get_phase_transitions method on ActionDetails."""

    def test_complete_lifecycle(self):
        """Test getting phase transitions for a complete action lifecycle."""
        details_pb2 = run_definition_pb2.ActionDetails()

        # Set up status
        start_ts = Timestamp()
        start_ts.FromDatetime(datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        details_pb2.status.start_time.CopyFrom(start_ts)
        details_pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
        details_pb2.status.attempts = 1

        # Create an attempt with phase transitions
        attempt = run_definition_pb2.ActionAttempt()
        attempt.attempt = 1
        attempt.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
        attempt.start_time.CopyFrom(start_ts)

        # Add phase transitions
        base_time = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Queued: 0-30s
        pt1 = run_definition_pb2.PhaseTransition()
        pt1.phase = phase_pb2.ACTION_PHASE_QUEUED
        ts1 = Timestamp()
        ts1.FromDatetime(base_time)
        pt1.start_time.CopyFrom(ts1)
        ts1_end = Timestamp()
        ts1_end.FromDatetime(base_time + datetime.timedelta(seconds=30))
        pt1.end_time.CopyFrom(ts1_end)
        attempt.phase_transitions.append(pt1)

        # Initializing: 30-60s
        pt2 = run_definition_pb2.PhaseTransition()
        pt2.phase = phase_pb2.ACTION_PHASE_INITIALIZING
        ts2 = Timestamp()
        ts2.FromDatetime(base_time + datetime.timedelta(seconds=30))
        pt2.start_time.CopyFrom(ts2)
        ts2_end = Timestamp()
        ts2_end.FromDatetime(base_time + datetime.timedelta(seconds=60))
        pt2.end_time.CopyFrom(ts2_end)
        attempt.phase_transitions.append(pt2)

        # Running: 60-200s
        pt3 = run_definition_pb2.PhaseTransition()
        pt3.phase = phase_pb2.ACTION_PHASE_RUNNING
        ts3 = Timestamp()
        ts3.FromDatetime(base_time + datetime.timedelta(seconds=60))
        pt3.start_time.CopyFrom(ts3)
        ts3_end = Timestamp()
        ts3_end.FromDatetime(base_time + datetime.timedelta(seconds=200))
        pt3.end_time.CopyFrom(ts3_end)
        attempt.phase_transitions.append(pt3)

        details_pb2.attempts.append(attempt)
        action_details = ActionDetails(pb2=details_pb2)

        result = action_details.get_phase_transitions()

        assert len(result) == 3
        assert result[0].phase == ActionPhase.QUEUED
        assert result[0].duration == datetime.timedelta(seconds=30)
        assert result[1].phase == ActionPhase.INITIALIZING
        assert result[1].duration == datetime.timedelta(seconds=30)
        assert result[2].phase == ActionPhase.RUNNING
        assert result[2].duration == datetime.timedelta(seconds=140)

    def test_no_transitions(self):
        """Test getting phase transitions when none exist."""
        details_pb2 = run_definition_pb2.ActionDetails()

        start_ts = Timestamp()
        start_ts.FromDatetime(datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        details_pb2.status.start_time.CopyFrom(start_ts)
        details_pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
        details_pb2.status.attempts = 1

        attempt = run_definition_pb2.ActionAttempt()
        attempt.attempt = 1
        attempt.start_time.CopyFrom(start_ts)
        details_pb2.attempts.append(attempt)

        action_details = ActionDetails(pb2=details_pb2)
        result = action_details.get_phase_transitions()

        assert result == []

    def test_specific_attempt(self):
        """Test getting phase transitions for a specific attempt number."""
        details_pb2 = run_definition_pb2.ActionDetails()

        start_ts = Timestamp()
        start_ts.FromDatetime(datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        details_pb2.status.start_time.CopyFrom(start_ts)
        details_pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
        details_pb2.status.attempts = 2

        base_time = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Attempt 1 - quick failure
        attempt1 = run_definition_pb2.ActionAttempt()
        attempt1.attempt = 1
        attempt1.start_time.CopyFrom(start_ts)
        pt1 = run_definition_pb2.PhaseTransition()
        pt1.phase = phase_pb2.ACTION_PHASE_QUEUED
        ts1 = Timestamp()
        ts1.FromDatetime(base_time)
        pt1.start_time.CopyFrom(ts1)
        ts1_end = Timestamp()
        ts1_end.FromDatetime(base_time + datetime.timedelta(seconds=10))
        pt1.end_time.CopyFrom(ts1_end)
        attempt1.phase_transitions.append(pt1)
        details_pb2.attempts.append(attempt1)

        # Attempt 2 - longer run
        attempt2 = run_definition_pb2.ActionAttempt()
        attempt2.attempt = 2
        attempt2.start_time.CopyFrom(start_ts)
        pt2 = run_definition_pb2.PhaseTransition()
        pt2.phase = phase_pb2.ACTION_PHASE_QUEUED
        ts2 = Timestamp()
        ts2.FromDatetime(base_time + datetime.timedelta(seconds=10))
        pt2.start_time.CopyFrom(ts2)
        ts2_end = Timestamp()
        ts2_end.FromDatetime(base_time + datetime.timedelta(seconds=50))
        pt2.end_time.CopyFrom(ts2_end)
        attempt2.phase_transitions.append(pt2)
        details_pb2.attempts.append(attempt2)

        action_details = ActionDetails(pb2=details_pb2)

        # Get attempt 1
        result1 = action_details.get_phase_transitions(attempt=1)
        assert len(result1) == 1
        assert result1[0].duration == datetime.timedelta(seconds=10)

        # Get attempt 2
        result2 = action_details.get_phase_transitions(attempt=2)
        assert len(result2) == 1
        assert result2[0].duration == datetime.timedelta(seconds=40)


class TestActionDetailsPhaseDurations:
    """Test phase duration properties on ActionDetails."""

    def test_phase_durations_property(self):
        """Test the phase_durations property returns correct mapping."""
        details_pb2 = run_definition_pb2.ActionDetails()

        start_ts = Timestamp()
        start_ts.FromDatetime(datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        details_pb2.status.start_time.CopyFrom(start_ts)
        details_pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
        details_pb2.status.attempts = 1

        attempt = run_definition_pb2.ActionAttempt()
        attempt.attempt = 1
        attempt.start_time.CopyFrom(start_ts)

        base_time = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Add transitions
        pt1 = run_definition_pb2.PhaseTransition()
        pt1.phase = phase_pb2.ACTION_PHASE_QUEUED
        ts1 = Timestamp()
        ts1.FromDatetime(base_time)
        pt1.start_time.CopyFrom(ts1)
        ts1_end = Timestamp()
        ts1_end.FromDatetime(base_time + datetime.timedelta(seconds=45))
        pt1.end_time.CopyFrom(ts1_end)
        attempt.phase_transitions.append(pt1)

        pt2 = run_definition_pb2.PhaseTransition()
        pt2.phase = phase_pb2.ACTION_PHASE_RUNNING
        ts2 = Timestamp()
        ts2.FromDatetime(base_time + datetime.timedelta(seconds=45))
        pt2.start_time.CopyFrom(ts2)
        ts2_end = Timestamp()
        ts2_end.FromDatetime(base_time + datetime.timedelta(seconds=100))
        pt2.end_time.CopyFrom(ts2_end)
        attempt.phase_transitions.append(pt2)

        details_pb2.attempts.append(attempt)
        action_details = ActionDetails(pb2=details_pb2)

        durations = action_details.phase_durations

        assert len(durations) == 2
        assert durations[ActionPhase.QUEUED] == datetime.timedelta(seconds=45)
        assert durations[ActionPhase.RUNNING] == datetime.timedelta(seconds=55)

    def test_queued_time(self):
        """Test queued_time property."""
        details_pb2 = run_definition_pb2.ActionDetails()

        start_ts = Timestamp()
        start_ts.FromDatetime(datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        details_pb2.status.start_time.CopyFrom(start_ts)
        details_pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
        details_pb2.status.attempts = 1

        attempt = run_definition_pb2.ActionAttempt()
        attempt.attempt = 1
        attempt.start_time.CopyFrom(start_ts)

        base_time = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        pt = run_definition_pb2.PhaseTransition()
        pt.phase = phase_pb2.ACTION_PHASE_QUEUED
        ts = Timestamp()
        ts.FromDatetime(base_time)
        pt.start_time.CopyFrom(ts)
        ts_end = Timestamp()
        ts_end.FromDatetime(base_time + datetime.timedelta(seconds=25))
        pt.end_time.CopyFrom(ts_end)
        attempt.phase_transitions.append(pt)

        details_pb2.attempts.append(attempt)
        action_details = ActionDetails(pb2=details_pb2)

        assert action_details.queued_time == datetime.timedelta(seconds=25)

    def test_queued_time_returns_none_when_not_queued(self):
        """Test queued_time returns None when action was never queued."""
        details_pb2 = run_definition_pb2.ActionDetails()

        start_ts = Timestamp()
        start_ts.FromDatetime(datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        details_pb2.status.start_time.CopyFrom(start_ts)
        details_pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
        details_pb2.status.attempts = 1

        attempt = run_definition_pb2.ActionAttempt()
        attempt.attempt = 1
        attempt.start_time.CopyFrom(start_ts)
        details_pb2.attempts.append(attempt)

        action_details = ActionDetails(pb2=details_pb2)

        assert action_details.queued_time is None

    def test_initializing_time(self):
        """Test initializing_time property."""
        details_pb2 = run_definition_pb2.ActionDetails()

        start_ts = Timestamp()
        start_ts.FromDatetime(datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        details_pb2.status.start_time.CopyFrom(start_ts)
        details_pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
        details_pb2.status.attempts = 1

        attempt = run_definition_pb2.ActionAttempt()
        attempt.attempt = 1
        attempt.start_time.CopyFrom(start_ts)

        base_time = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        pt = run_definition_pb2.PhaseTransition()
        pt.phase = phase_pb2.ACTION_PHASE_INITIALIZING
        ts = Timestamp()
        ts.FromDatetime(base_time)
        pt.start_time.CopyFrom(ts)
        ts_end = Timestamp()
        ts_end.FromDatetime(base_time + datetime.timedelta(seconds=40))
        pt.end_time.CopyFrom(ts_end)
        attempt.phase_transitions.append(pt)

        details_pb2.attempts.append(attempt)
        action_details = ActionDetails(pb2=details_pb2)

        assert action_details.initializing_time == datetime.timedelta(seconds=40)

    def test_running_time(self):
        """Test running_time property."""
        details_pb2 = run_definition_pb2.ActionDetails()

        start_ts = Timestamp()
        start_ts.FromDatetime(datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        details_pb2.status.start_time.CopyFrom(start_ts)
        details_pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
        details_pb2.status.attempts = 1

        attempt = run_definition_pb2.ActionAttempt()
        attempt.attempt = 1
        attempt.start_time.CopyFrom(start_ts)

        base_time = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        pt = run_definition_pb2.PhaseTransition()
        pt.phase = phase_pb2.ACTION_PHASE_RUNNING
        ts = Timestamp()
        ts.FromDatetime(base_time)
        pt.start_time.CopyFrom(ts)
        ts_end = Timestamp()
        ts_end.FromDatetime(base_time + datetime.timedelta(seconds=145))
        pt.end_time.CopyFrom(ts_end)
        attempt.phase_transitions.append(pt)

        details_pb2.attempts.append(attempt)
        action_details = ActionDetails(pb2=details_pb2)

        assert action_details.running_time == datetime.timedelta(seconds=145)

    def test_waiting_for_resources_time(self):
        """Test waiting_for_resources_time property."""
        details_pb2 = run_definition_pb2.ActionDetails()

        start_ts = Timestamp()
        start_ts.FromDatetime(datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        details_pb2.status.start_time.CopyFrom(start_ts)
        details_pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
        details_pb2.status.attempts = 1

        attempt = run_definition_pb2.ActionAttempt()
        attempt.attempt = 1
        attempt.start_time.CopyFrom(start_ts)

        base_time = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        pt = run_definition_pb2.PhaseTransition()
        pt.phase = phase_pb2.ACTION_PHASE_WAITING_FOR_RESOURCES
        ts = Timestamp()
        ts.FromDatetime(base_time)
        pt.start_time.CopyFrom(ts)
        ts_end = Timestamp()
        ts_end.FromDatetime(base_time + datetime.timedelta(seconds=30))
        pt.end_time.CopyFrom(ts_end)
        attempt.phase_transitions.append(pt)

        details_pb2.attempts.append(attempt)
        action_details = ActionDetails(pb2=details_pb2)

        assert action_details.waiting_for_resources_time == datetime.timedelta(seconds=30)

    def test_all_phase_times_combined(self):
        """Test all phase time properties together."""
        details_pb2 = run_definition_pb2.ActionDetails()

        start_ts = Timestamp()
        start_ts.FromDatetime(datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc))
        details_pb2.status.start_time.CopyFrom(start_ts)
        details_pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
        details_pb2.status.attempts = 1

        attempt = run_definition_pb2.ActionAttempt()
        attempt.attempt = 1
        attempt.start_time.CopyFrom(start_ts)

        base_time = datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Queued: 10s
        pt1 = run_definition_pb2.PhaseTransition()
        pt1.phase = phase_pb2.ACTION_PHASE_QUEUED
        ts1 = Timestamp()
        ts1.FromDatetime(base_time)
        pt1.start_time.CopyFrom(ts1)
        ts1_end = Timestamp()
        ts1_end.FromDatetime(base_time + datetime.timedelta(seconds=10))
        pt1.end_time.CopyFrom(ts1_end)
        attempt.phase_transitions.append(pt1)

        # Waiting: 15s
        pt2 = run_definition_pb2.PhaseTransition()
        pt2.phase = phase_pb2.ACTION_PHASE_WAITING_FOR_RESOURCES
        ts2 = Timestamp()
        ts2.FromDatetime(base_time + datetime.timedelta(seconds=10))
        pt2.start_time.CopyFrom(ts2)
        ts2_end = Timestamp()
        ts2_end.FromDatetime(base_time + datetime.timedelta(seconds=25))
        pt2.end_time.CopyFrom(ts2_end)
        attempt.phase_transitions.append(pt2)

        # Initializing: 30s
        pt3 = run_definition_pb2.PhaseTransition()
        pt3.phase = phase_pb2.ACTION_PHASE_INITIALIZING
        ts3 = Timestamp()
        ts3.FromDatetime(base_time + datetime.timedelta(seconds=25))
        pt3.start_time.CopyFrom(ts3)
        ts3_end = Timestamp()
        ts3_end.FromDatetime(base_time + datetime.timedelta(seconds=55))
        pt3.end_time.CopyFrom(ts3_end)
        attempt.phase_transitions.append(pt3)

        # Running: 145s
        pt4 = run_definition_pb2.PhaseTransition()
        pt4.phase = phase_pb2.ACTION_PHASE_RUNNING
        ts4 = Timestamp()
        ts4.FromDatetime(base_time + datetime.timedelta(seconds=55))
        pt4.start_time.CopyFrom(ts4)
        ts4_end = Timestamp()
        ts4_end.FromDatetime(base_time + datetime.timedelta(seconds=200))
        pt4.end_time.CopyFrom(ts4_end)
        attempt.phase_transitions.append(pt4)

        details_pb2.attempts.append(attempt)
        action_details = ActionDetails(pb2=details_pb2)

        assert action_details.queued_time == datetime.timedelta(seconds=10)
        assert action_details.waiting_for_resources_time == datetime.timedelta(seconds=15)
        assert action_details.initializing_time == datetime.timedelta(seconds=30)
        assert action_details.running_time == datetime.timedelta(seconds=145)

        # Verify total
        total = (
            action_details.queued_time
            + action_details.waiting_for_resources_time
            + action_details.initializing_time
            + action_details.running_time
        )
        assert total == datetime.timedelta(seconds=200)
