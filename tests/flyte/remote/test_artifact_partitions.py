"""flyte.remote.Artifact partition API: what goes on the wire, and what comes back."""

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.artifact import artifact_pb2, artifact_service_pb2
from flyteidl2.common import list_pb2
from flyteidl2.core import artifact_id_pb2
from google.protobuf import timestamp_pb2

from flyte.artifacts import TimePartition
from flyte.artifacts._partitions import partitions_to_pb2
from flyte.remote import Artifact, PartitionSchema
from flyte.remote._artifact import (
    PARTITION_FIELD_PREFIX,
    TIME_PARTITION_FIELD,
    ArtifactGroup,
    partition_filters,
    schema_from_spec,
)


def _cfg():
    return MagicMock(org="test-org", project="proj", domain="dev")


def _patched(client):
    return (
        patch("flyte.remote._artifact.ensure_client"),
        patch("flyte.remote._artifact.get_init_config", return_value=_cfg()),
        patch("flyte.remote._artifact.get_client", return_value=client),
    )


def _stored(name="raw_events", version="v1", partitions=None, mismatch: str | None = None) -> artifact_pb2.Artifact:
    strings, tp = partitions_to_pb2(partitions)
    a = artifact_pb2.Artifact(
        artifact_id=artifact_pb2.ArtifactIdentifier(
            name=artifact_pb2.ArtifactName(org="test-org", project="proj", domain="dev", name=name),
            version=version,
        ),
        spec=artifact_pb2.ArtifactSpec(partitions=strings, time_partition=tp),
    )
    if mismatch:
        a.partition_schema_mismatch.message = mismatch
    return a


def _by_field(filters, field):
    return [f for f in filters if f.field == field]


class TestPartitionFilters:
    def test_time_value_is_equal_on_time_partition(self):
        (f,) = partition_filters({"date": date(2026, 8, 1)})
        assert (f.field, f.function, list(f.values)) == (
            TIME_PARTITION_FIELD,
            list_pb2.Filter.EQUAL,
            ["2026-08-01T00:00:00Z"],
        )

    def test_datetime_is_floored_to_the_hour(self):
        (f,) = partition_filters({"hour": datetime(2026, 8, 1, 13, 59)})
        assert list(f.values) == ["2026-08-01T13:00:00Z"]

    def test_explicit_granularity_floors_too(self):
        (f,) = partition_filters({"date": TimePartition(date(2026, 8, 13), "month")})
        assert list(f.values) == ["2026-08-01T00:00:00Z"]

    def test_range_is_two_bounds(self):
        lo, hi = partition_filters({"date": (date(2026, 8, 1), "2026-08-31")})
        assert (lo.field, lo.function, list(lo.values)) == (
            TIME_PARTITION_FIELD,
            list_pb2.Filter.GREATER_THAN_OR_EQUAL,
            ["2026-08-01T00:00:00Z"],
        )
        assert (hi.field, hi.function, list(hi.values)) == (
            TIME_PARTITION_FIELD,
            list_pb2.Filter.LESS_THAN_OR_EQUAL,
            ["2026-08-31T00:00:00Z"],
        )

    def test_string_scalar_is_equal_on_partition_key(self):
        (f,) = partition_filters({"region": "us"})
        assert (f.field, f.function, list(f.values)) == (
            f"{PARTITION_FIELD_PREFIX}region",
            list_pb2.Filter.EQUAL,
            ["us"],
        )

    def test_string_list_is_value_in(self):
        (f,) = partition_filters({"region": ["us", "eu"]})
        assert (f.function, list(f.values)) == (list_pb2.Filter.VALUE_IN, ["us", "eu"])

    def test_two_strings_in_a_tuple_are_values_not_a_range(self):
        (f,) = partition_filters({"region": ("us", "eu")})
        assert f.field == f"{PARTITION_FIELD_PREFIX}region"
        assert f.function == list_pb2.Filter.VALUE_IN

    def test_ints_are_strings(self):
        (f,) = partition_filters({"shard": 3})
        assert list(f.values) == ["3"]


class TestGet:
    @pytest.mark.asyncio
    async def test_get_by_partition_lists_newest_first_limit_one(self):
        client = MagicMock()
        client.artifact_service.list_artifacts = AsyncMock(
            return_value=artifact_service_pb2.ListArtifactsResponse(
                artifacts=[_stored(version="v2", partitions={"date": date(2026, 8, 1), "region": "us"})]
            )
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            a = await Artifact.get.aio("raw_events", date=date(2026, 8, 1), region="us")

        req = client.artifact_service.list_artifacts.await_args[0][0]
        assert req.name == "raw_events"
        assert req.request.limit == 1
        assert req.project_id.name == "proj"
        assert _by_field(req.request.filters, TIME_PARTITION_FIELD)[0].values == ["2026-08-01T00:00:00Z"]
        assert _by_field(req.request.filters, f"{PARTITION_FIELD_PREFIX}region")[0].values == ["us"]
        assert a.version == "v2"
        assert a.partitions == {"date": date(2026, 8, 1), "region": "us"}
        client.artifact_service.get_artifact.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_by_partition_not_found_is_a_clear_error(self):
        client = MagicMock()
        client.artifact_service.list_artifacts = AsyncMock(
            return_value=artifact_service_pb2.ListArtifactsResponse(artifacts=[])
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3, pytest.raises(ValueError, match=r"No version of artifact 'raw_events' .* date=2026-08-01"):
            await Artifact.get.aio("raw_events", date=date(2026, 8, 1))

    @pytest.mark.asyncio
    async def test_version_and_partitions_conflict(self):
        client = MagicMock()
        p1, p2, p3 = _patched(client)
        with p1, p2, p3, pytest.raises(ValueError, match="either a version or partition values"):
            await Artifact.get.aio("raw_events", version="v1", date=date(2026, 8, 1))

    @pytest.mark.asyncio
    async def test_plain_get_still_uses_get_artifact(self):
        client = MagicMock()
        client.artifact_service.get_artifact = AsyncMock(
            return_value=artifact_service_pb2.GetArtifactResponse(artifact=_stored())
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            a = await Artifact.get.aio("raw_events")
        assert a.version == "v1"
        assert a.partitions == {}
        assert a.schema_mismatch is None


class TestListall:
    async def _captured(self, **kwargs) -> artifact_service_pb2.ListArtifactsRequest:
        client = MagicMock()
        client.artifact_service.list_artifacts = AsyncMock(
            return_value=artifact_service_pb2.ListArtifactsResponse(artifacts=[], token="")
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            async for _ in Artifact.listall.aio(**kwargs):
                pass
        return client.artifact_service.list_artifacts.await_args[0][0]

    @pytest.mark.asyncio
    async def test_range_and_latest_per_partition(self):
        req = await self._captured(
            name="raw_events", date=(date(2026, 8, 1), date(2026, 8, 31)), latest_per_partition=True
        )
        assert req.latest_per_partition is True
        assert [f.function for f in _by_field(req.request.filters, TIME_PARTITION_FIELD)] == [
            list_pb2.Filter.GREATER_THAN_OR_EQUAL,
            list_pb2.Filter.LESS_THAN_OR_EQUAL,
        ]

    @pytest.mark.asyncio
    async def test_partitions_mapping_and_kwargs_merge(self):
        req = await self._captured(name="raw_events", partitions={"region": ["us", "eu"]}, algo="gbm")
        assert _by_field(req.request.filters, f"{PARTITION_FIELD_PREFIX}region")[0].function == list_pb2.Filter.VALUE_IN
        assert _by_field(req.request.filters, f"{PARTITION_FIELD_PREFIX}algo")[0].values == ["gbm"]

    @pytest.mark.asyncio
    async def test_latest_per_partition_needs_a_name(self):
        with pytest.raises(ValueError, match="requires an artifact name"):
            await self._captured(latest_per_partition=True)

    @pytest.mark.asyncio
    async def test_no_partition_filters_by_default(self):
        req = await self._captured(name="raw_events")
        assert req.latest_per_partition is False
        assert not req.request.filters


class TestPartitionValues:
    @pytest.mark.asyncio
    async def test_time_key_values_parse_by_granularity(self):
        client = MagicMock()
        client.artifact_service.get_artifact_schema = AsyncMock(
            return_value=artifact_service_pb2.GetArtifactSchemaResponse(
                partition_schema=artifact_pb2.ArtifactPartitionSchema(
                    time_partition=artifact_pb2.TimePartitionKey(
                        key="date", granularity=artifact_id_pb2.Granularity.DAY
                    ),
                    partition_keys=["region"],
                )
            )
        )
        client.artifact_service.list_partition_values = AsyncMock(
            return_value=artifact_service_pb2.ListPartitionValuesResponse(
                values=["2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z"]
            )
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            values = await Artifact.partition_values.aio("raw_events", "date", region="us")

        assert values == [date(2026, 8, 1), date(2026, 8, 2)]
        req = client.artifact_service.list_partition_values.await_args[0][0]
        assert (req.name, req.key) == ("raw_events", "date")
        assert _by_field(req.request.filters, f"{PARTITION_FIELD_PREFIX}region")[0].values == ["us"]

    @pytest.mark.asyncio
    async def test_string_key_values_are_strings(self):
        client = MagicMock()
        client.artifact_service.get_artifact_schema = AsyncMock(
            return_value=artifact_service_pb2.GetArtifactSchemaResponse(
                partition_schema=artifact_pb2.ArtifactPartitionSchema(partition_keys=["region"])
            )
        )
        client.artifact_service.list_partition_values = AsyncMock(
            return_value=artifact_service_pb2.ListPartitionValuesResponse(values=["eu", "us"])
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            values = await Artifact.partition_values.aio("raw_events", "region")
        assert values == ["eu", "us"]

    @pytest.mark.asyncio
    async def test_hourly_key_values_are_datetimes(self):
        client = MagicMock()
        client.artifact_service.get_artifact_schema = AsyncMock(
            return_value=artifact_service_pb2.GetArtifactSchemaResponse(
                partition_schema=artifact_pb2.ArtifactPartitionSchema(
                    time_partition=artifact_pb2.TimePartitionKey(
                        key="hour", granularity=artifact_id_pb2.Granularity.HOUR
                    )
                )
            )
        )
        client.artifact_service.list_partition_values = AsyncMock(
            return_value=artifact_service_pb2.ListPartitionValuesResponse(values=["2026-08-01T13:00:00Z"])
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            values = await Artifact.partition_values.aio("events", "hour")
        assert values == [datetime(2026, 8, 1, 13, tzinfo=timezone.utc)]


class TestSchema:
    def test_schema_from_spec(self):
        s = schema_from_spec({"date": date, "region": str, "shard": int})
        assert s == PartitionSchema(time_key="date", granularity="day", keys=("region", "shard"))
        assert schema_from_spec({"hour": datetime}).granularity == "hour"
        assert schema_from_spec({"date": "month"}).granularity == "month"
        assert schema_from_spec({"date": "week", "algo": "str"}).keys == ("algo",)
        assert schema_from_spec(None) == PartitionSchema(time_key=None, granularity=None, keys=())

    def test_schema_from_spec_rejects(self):
        with pytest.raises(ValueError, match="at most one time partition"):
            schema_from_spec({"date": date, "hour": datetime})
        with pytest.raises(ValueError, match="declare a time key"):
            schema_from_spec({"date": float})

    def test_schema_pb2_round_trip(self):
        s = PartitionSchema(time_key="date", granularity="week", keys=("region",))
        assert PartitionSchema.from_pb2(s.to_pb2()) == s
        assert s.all_keys == ("date", "region")
        assert PartitionSchema.from_pb2(artifact_pb2.ArtifactPartitionSchema()).all_keys == ()

    @pytest.mark.asyncio
    async def test_declare_sends_the_schema(self):
        client = MagicMock()
        client.artifact_service.declare_artifact = AsyncMock(
            return_value=artifact_service_pb2.DeclareArtifactResponse(
                partition_schema=PartitionSchema("date", "day", ("region",)).to_pb2()
            )
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            s = await Artifact.declare.aio("raw_events", {"date": date, "region": str})

        req = client.artifact_service.declare_artifact.await_args[0][0]
        assert req.name.name == "raw_events"
        assert req.name.project == "proj"
        assert req.partition_schema.time_partition.key == "date"
        assert req.partition_schema.time_partition.granularity == artifact_id_pb2.Granularity.DAY
        assert list(req.partition_schema.partition_keys) == ["region"]
        assert s == PartitionSchema("date", "day", ("region",), declared=True)

    @pytest.mark.asyncio
    async def test_get_schema(self):
        client = MagicMock()
        client.artifact_service.get_artifact_schema = AsyncMock(
            return_value=artifact_service_pb2.GetArtifactSchemaResponse(
                partition_schema=PartitionSchema("hour", "hour", ()).to_pb2(), declared=True
            )
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            s = await Artifact.get_schema.aio("events")
        assert s == PartitionSchema("hour", "hour", (), declared=True)


class TestProperties:
    def test_partitions_and_mismatch(self):
        a = Artifact(pb2=_stored(partitions={"date": date(2026, 8, 1), "region": "eu"}, mismatch="keys differ"))
        assert a.partitions == {"date": date(2026, 8, 1), "region": "eu"}
        assert a.time_partition is not None and a.time_partition.key == "date"
        assert a.schema_mismatch == "keys differ"
        repr_items = dict(kv for kv in a.__rich_repr__() if isinstance(kv, tuple) and len(kv) == 2)
        assert repr_items["partitions"] == "date=2026-08-01, region=eu"
        assert repr_items["schema_mismatch"] == "keys differ"

    def test_mismatch_without_message_still_reads_as_flagged(self):
        pb = _stored()
        pb.partition_schema_mismatch.SetInParent()
        assert Artifact(pb2=pb).schema_mismatch

    def test_group_schema_and_latest_partition(self):
        ts = timestamp_pb2.Timestamp()
        ts.FromDatetime(datetime(2026, 9, 17, tzinfo=timezone.utc))
        g = ArtifactGroup(
            pb2=artifact_service_pb2.ArtifactGroup(
                latest=_stored(),
                versions=16,
                partition_schema=PartitionSchema("date", "day", ("region",)).to_pb2(),
                latest_time_partition=ts,
            )
        )
        assert g.partition_schema == PartitionSchema("date", "day", ("region",))
        assert g.latest_time_partition == date(2026, 9, 17)
        assert ArtifactGroup(pb2=artifact_service_pb2.ArtifactGroup(latest=_stored())).partition_schema is None


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_sends_partitions(self):
        from flyte.io import File

        client = MagicMock()
        client.artifact_service.create_artifact = AsyncMock(
            return_value=artifact_service_pb2.CreateArtifactResponse(artifact=_stored())
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            await Artifact.create.aio(
                File(path="s3://b/x"), name="raw_events", partitions={"date": date(2026, 8, 1), "region": "us"}
            )
        req = client.artifact_service.create_artifact.await_args[0][0]
        assert req.spec.time_partition.key == "date"
        assert req.spec.partitions.value["region"].static_value == "us"

    @pytest.mark.asyncio
    async def test_create_seeds_partitions_from_wrapper(self):
        import flyte.artifacts as artifacts
        from flyte.io import File

        client = MagicMock()
        client.artifact_service.create_artifact = AsyncMock(
            return_value=artifact_service_pb2.CreateArtifactResponse(artifact=_stored())
        )
        wrapped = artifacts.new(File(path="s3://b/x"), artifacts.Metadata(name="n", partitions={"region": "eu"}))
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            await Artifact.create.aio(wrapped)
        req = client.artifact_service.create_artifact.await_args[0][0]
        assert req.spec.partitions.value["region"].static_value == "eu"
        assert not req.spec.HasField("time_partition")


class TestPartitionFilterEdges:
    def test_list_of_time_values_is_rejected(self):
        with pytest.raises(ValueError, match="use a range \\(lo, hi\\) or call once per value"):
            partition_filters({"date": [date(2026, 9, 15), date(2026, 9, 16)]})
        with pytest.raises(ValueError, match="call once per value"):
            partition_filters({"date": ["2026-09-15", "2026-09-16"]})

    def test_mixed_tuple_is_rejected(self):
        with pytest.raises(ValueError, match="both bounds must be dates"):
            partition_filters({"date": (date(2026, 9, 15), "us")})
        with pytest.raises(ValueError, match="both bounds must be dates"):
            partition_filters({"date": ("2026-09-15", "us")})

    def test_string_tuple_is_a_value_list(self):
        (f,) = partition_filters({"region": ("us", "eu")})
        assert (f.field, f.function, list(f.values)) == ("partition.region", list_pb2.Filter.VALUE_IN, ["us", "eu"])

    def test_time_like_string_selects_the_time_partition(self):
        (f,) = partition_filters({"date": "2026-09-16"})
        assert (f.field, list(f.values)) == (TIME_PARTITION_FIELD, ["2026-09-16T00:00:00Z"])
        (h,) = partition_filters({"hour": "2026-09-16T13"})
        assert (h.field, list(h.values)) == (TIME_PARTITION_FIELD, ["2026-09-16T13:00:00Z"])
        (r,) = partition_filters({"hour": "2026-09-16T13:42:10Z"})
        assert list(r.values) == ["2026-09-16T13:00:00Z"], "an RFC3339 string floors to the hour"

    def test_none_is_rejected(self):
        with pytest.raises(ValueError, match="Partition 'region' is None"):
            partition_filters({"region": None})


class TestReservedKeyNames:
    @pytest.mark.asyncio
    async def test_get_accepts_keys_named_like_its_parameters(self):
        client = MagicMock()
        client.artifact_service.list_artifacts = AsyncMock(
            return_value=artifact_service_pb2.ListArtifactsResponse(
                artifacts=[_stored(partitions={"date": date(2026, 9, 16), "domain": "eu"})], token=""
            )
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            a = await Artifact.get.aio(
                "raw_events", partitions={"domain": "eu", "version": "a"}, date=date(2026, 9, 16)
            )
        req = client.artifact_service.list_artifacts.await_args[0][0]
        fields = {(f.field, tuple(f.values)) for f in req.request.filters}
        assert ("partition.domain", ("eu",)) in fields
        assert ("partition.version", ("a",)) in fields
        assert (TIME_PARTITION_FIELD, ("2026-09-16T00:00:00Z",)) in fields
        assert a.version == "v1"

    @pytest.mark.asyncio
    async def test_kwargs_win_over_the_mapping(self):
        client = MagicMock()
        client.artifact_service.list_artifacts = AsyncMock(
            return_value=artifact_service_pb2.ListArtifactsResponse(artifacts=[_stored()], token="")
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            await Artifact.get.aio("raw_events", partitions={"region": "eu"}, region="us")
        req = client.artifact_service.list_artifacts.await_args[0][0]
        assert [list(f.values) for f in _by_field(req.request.filters, "partition.region")] == [["us"]]

    @pytest.mark.asyncio
    async def test_partition_values_accepts_a_mapping(self):
        client = MagicMock()
        client.artifact_service.get_artifact_schema = AsyncMock(
            return_value=artifact_service_pb2.GetArtifactSchemaResponse(
                partition_schema=artifact_pb2.ArtifactPartitionSchema(
                    time_partition=artifact_pb2.TimePartitionKey(
                        key="date", granularity=artifact_id_pb2.Granularity.DAY
                    ),
                    partition_keys=["domain", "limit"],
                )
            )
        )
        client.artifact_service.list_partition_values = AsyncMock(
            return_value=artifact_service_pb2.ListPartitionValuesResponse(values=["a", "b"])
        )
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            values = await Artifact.partition_values.aio("raw_events", "limit", partitions={"domain": "eu"})
        assert values == ["a", "b"]
        req = client.artifact_service.list_partition_values.await_args[0][0]
        assert [(f.field, list(f.values)) for f in req.request.filters] == [("partition.domain", ["eu"])]


class TestGetRejectsMultiPartitionValues:
    @pytest.mark.asyncio
    async def test_range_and_list_point_at_listall(self):
        client = MagicMock()
        p1, p2, p3 = _patched(client)
        with p1, p2, p3:
            with pytest.raises(ValueError, match="listall"):
                await Artifact.get.aio("raw_events", date=(date(2026, 9, 1), date(2026, 9, 30)))
            with pytest.raises(ValueError, match="listall"):
                await Artifact.get.aio("raw_events", region=["us", "eu"])
        client.artifact_service.list_artifacts.assert_not_called()
