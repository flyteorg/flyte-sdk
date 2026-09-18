"""The partition value rule: how Python values become an artifact's partition identity."""

from datetime import date, datetime, timedelta, timezone

import pytest
from flyteidl2.core import artifact_id_pb2, types_pb2

import flyte.artifacts as artifacts
from flyte.artifacts._metadata import Metadata, to_produced_artifact
from flyte.artifacts._partitions import (
    TimePartition,
    default_time_key,
    floor_time,
    parse_time,
    partitions_from_pb2,
    partitions_to_pb2,
    to_rfc3339,
)

STR = types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING)


class TestValueRule:
    def test_date_is_a_daily_time_partition_under_its_own_key(self):
        strings, tp = partitions_to_pb2({"date": date(2026, 8, 1), "region": "us"})
        assert tp is not None
        assert tp.key == "date"
        assert tp.granularity == artifact_id_pb2.Granularity.DAY
        assert tp.value.time_value.ToDatetime(tzinfo=timezone.utc) == datetime(2026, 8, 1, tzinfo=timezone.utc)
        assert strings is not None
        assert {k: v.static_value for k, v in strings.value.items()} == {"region": "us"}

    def test_datetime_is_hourly_and_floored_to_utc(self):
        pst = timezone(timedelta(hours=-8))
        _, tp = partitions_to_pb2({"hour": datetime(2026, 8, 1, 13, 45, 9, tzinfo=pst)})
        assert tp.granularity == artifact_id_pb2.Granularity.HOUR
        assert tp.value.time_value.ToDatetime(tzinfo=timezone.utc) == datetime(2026, 8, 1, 21, tzinfo=timezone.utc)

    def test_naive_datetime_means_utc(self):
        _, tp = partitions_to_pb2({"hour": datetime(2026, 8, 1, 13, 45)})
        assert tp.value.time_value.ToDatetime(tzinfo=timezone.utc) == datetime(2026, 8, 1, 13, tzinfo=timezone.utc)

    def test_anything_else_is_a_string(self):
        strings, tp = partitions_to_pb2({"shard": 7, "algo": "gbm", "flag": True})
        assert tp is None
        assert {k: v.static_value for k, v in strings.value.items()} == {"shard": "7", "algo": "gbm", "flag": "True"}

    def test_two_time_values_are_rejected(self):
        with pytest.raises(ValueError, match="at most one time partition"):
            partitions_to_pb2({"date": date(2026, 8, 1), "hour": datetime(2026, 8, 1, 3)})

    def test_empty_key_is_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            partitions_to_pb2({"": "x"})

    def test_none_and_empty_produce_nothing(self):
        assert partitions_to_pb2(None) == (None, None)
        assert partitions_to_pb2({}) == (None, None)

    @pytest.mark.parametrize(
        "granularity, expected",
        [
            ("hour", datetime(2026, 8, 13, 15, tzinfo=timezone.utc)),
            ("day", datetime(2026, 8, 13, tzinfo=timezone.utc)),
            ("week", datetime(2026, 8, 10, tzinfo=timezone.utc)),  # the Monday
            ("month", datetime(2026, 8, 1, tzinfo=timezone.utc)),
        ],
    )
    def test_explicit_granularity_floors(self, granularity, expected):
        value = datetime(2026, 8, 13, 15, 42, 7, tzinfo=timezone.utc)
        assert floor_time(value, granularity) == expected
        _, tp = partitions_to_pb2({"date": TimePartition(value, granularity)})
        assert tp.value.time_value.ToDatetime(tzinfo=timezone.utc) == expected
        assert artifact_id_pb2.Granularity.Name(tp.granularity) == granularity.upper()

    def test_time_partition_wrapper_validates(self):
        with pytest.raises(ValueError, match="granularity"):
            TimePartition(date(2026, 8, 1), "fortnight")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            TimePartition("2026-08-01", "day")  # type: ignore[arg-type]

    def test_default_time_key(self):
        assert default_time_key("hour") == "hour"
        assert default_time_key("day") == "date"
        assert default_time_key("month") == "date"


class TestRoundTrip:
    def test_daily_comes_back_as_date(self):
        strings, tp = partitions_to_pb2({"date": date(2026, 8, 1), "region": "us"})
        assert partitions_from_pb2(strings, tp) == {"date": date(2026, 8, 1), "region": "us"}

    def test_hourly_comes_back_as_utc_datetime(self):
        strings, tp = partitions_to_pb2({"hour": datetime(2026, 8, 1, 13, 30)})
        assert partitions_from_pb2(strings, tp) == {"hour": datetime(2026, 8, 1, 13, tzinfo=timezone.utc)}

    def test_week_and_month_come_back_as_date(self):
        _, tp = partitions_to_pb2({"date": TimePartition(date(2026, 8, 13), "week")})
        assert partitions_from_pb2(None, tp) == {"date": date(2026, 8, 10)}
        _, tp = partitions_to_pb2({"date": TimePartition(date(2026, 8, 13), "month")})
        assert partitions_from_pb2(None, tp) == {"date": date(2026, 8, 1)}

    def test_missing_key_defaults_from_granularity(self):
        # Older producers may leave the key empty; the reader agrees on the default.
        _, tp = partitions_to_pb2({"x": datetime(2026, 8, 1, 3)})
        tp.key = ""
        assert list(partitions_from_pb2(None, tp)) == ["hour"]

    def test_parse_and_format(self):
        assert to_rfc3339(date(2026, 8, 1)) == "2026-08-01T00:00:00Z"
        assert parse_time("2026-08-01") == datetime(2026, 8, 1, tzinfo=timezone.utc)
        assert parse_time("2026-08-01T13") == datetime(2026, 8, 1, 13, tzinfo=timezone.utc)
        assert parse_time("2026-08-01T13:00:00Z") == datetime(2026, 8, 1, 13, tzinfo=timezone.utc)
        assert parse_time(datetime(2026, 8, 1, 13, tzinfo=timezone(timedelta(hours=2)))) == datetime(
            2026, 8, 1, 11, tzinfo=timezone.utc
        )


class TestMetadata:
    def test_metadata_validates_partitions_eagerly(self):
        with pytest.raises(ValueError, match="at most one time partition"):
            Metadata(name="x", partitions={"a": date(2026, 1, 1), "b": date(2026, 1, 2)})

    def test_model_metadata_carries_partitions(self):
        md = Metadata.create_model_metadata(name="m", partitions={"date": date(2026, 8, 1), "algo": "gbm"})
        assert md.partitions == {"date": date(2026, 8, 1), "algo": "gbm"}

    def test_to_produced_artifact_carries_partitions(self):
        decl = to_produced_artifact(
            Metadata(name="raw_events", partitions={"date": date(2026, 8, 1), "region": "eu"}),
            output="o0",
            literal_type=STR,
        )
        assert decl.time_partition.key == "date"
        assert decl.time_partition.granularity == artifact_id_pb2.Granularity.DAY
        assert decl.partitions.value["region"].static_value == "eu"

    def test_to_produced_artifact_without_partitions_leaves_fields_unset(self):
        decl = to_produced_artifact(Metadata(name="plain"), output="o0", literal_type=STR)
        assert not decl.HasField("partitions")
        assert not decl.HasField("time_partition")

    def test_wrapper_metadata_keeps_partitions(self):
        from flyte.io import File

        md = Metadata(name="raw_events", partitions={"date": date(2026, 8, 1)})
        wrapped = artifacts.new(File(path="s3://b/x"), md)
        assert wrapped.get_flyte_metadata().partitions == {"date": date(2026, 8, 1)}

    def test_exports(self):
        assert artifacts.TimePartition is TimePartition
