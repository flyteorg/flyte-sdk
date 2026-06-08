from datetime import datetime, timezone

from flyteidl2.core import literals_pb2
from flyteidl2.task import common_pb2
from google.protobuf import timestamp_pb2

from flyte._internal.runtime.convert import Inputs
from flyte._internal.runtime.taskrunner import _inject_kickoff_time_from_run_start
from flyte._internal.runtime.trigger_serde import KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY

RUN_START = datetime(2026, 6, 4, 12, 0, tzinfo=timezone.utc)


def _datetime_literal(dt: datetime) -> literals_pb2.Literal:
    ts = timestamp_pb2.Timestamp()
    ts.FromDatetime(dt)
    return literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(datetime=ts)))


def test_inject_kickoff_time_appends_and_strips_reserved_key():
    """When the kickoff arg has no literal yet, it is appended from run_start_time and the reserved
    context key is removed while other context entries are preserved."""
    inputs = Inputs(
        proto_inputs=common_pb2.Inputs(
            literals=[
                common_pb2.NamedLiteral(
                    name="count",
                    value=literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=5))),
                )
            ],
            context=[
                literals_pb2.KeyValuePair(key=KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY, value="start_time"),
                literals_pb2.KeyValuePair(key="team", value="ml"),
            ],
        )
    )

    out = _inject_kickoff_time_from_run_start(inputs, RUN_START)

    lits = {lit.name: lit.value for lit in out.proto_inputs.literals}
    assert lits["start_time"].scalar.primitive.datetime.ToDatetime(tzinfo=timezone.utc) == RUN_START
    assert lits["count"].scalar.primitive.integer == 5
    # Reserved key stripped, user-supplied context retained.
    assert out.context == {"team": "ml"}


def test_inject_kickoff_time_overrides_existing_placeholder():
    """An existing (placeholder) literal for the kickoff arg is overwritten with run_start_time."""
    inputs = Inputs(
        proto_inputs=common_pb2.Inputs(
            literals=[common_pb2.NamedLiteral(name="start_time", value=_datetime_literal(datetime(1970, 1, 1)))],
            context=[literals_pb2.KeyValuePair(key=KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY, value="start_time")],
        )
    )

    out = _inject_kickoff_time_from_run_start(inputs, RUN_START)

    assert len(out.proto_inputs.literals) == 1
    assert out.proto_inputs.literals[0].value.scalar.primitive.datetime.ToDatetime(tzinfo=timezone.utc) == RUN_START


def test_inject_kickoff_time_noop_without_reserved_key():
    """Non-triggered runs (no reserved context key) are left untouched."""
    inputs = Inputs(
        proto_inputs=common_pb2.Inputs(
            literals=[],
            context=[literals_pb2.KeyValuePair(key="team", value="ml")],
        )
    )

    out = _inject_kickoff_time_from_run_start(inputs, RUN_START)

    assert len(out.proto_inputs.literals) == 0
    assert out.context == {"team": "ml"}


def test_inject_kickoff_time_normalizes_non_utc_to_utc():
    """A non-UTC aware run_start_time is normalized to UTC before being written as a timestamp."""
    from datetime import timedelta

    inputs = Inputs(
        proto_inputs=common_pb2.Inputs(
            literals=[],
            context=[literals_pb2.KeyValuePair(key=KICKOFF_TIME_INPUT_ARG_CONTEXT_KEY, value="start_time")],
        )
    )

    plus_two = datetime(2026, 6, 4, 14, 0, tzinfo=timezone(timedelta(hours=2)))  # == 12:00 UTC
    out = _inject_kickoff_time_from_run_start(inputs, plus_two)
    got = out.proto_inputs.literals[0].value.scalar.primitive.datetime.ToDatetime(tzinfo=timezone.utc)
    assert got == RUN_START
