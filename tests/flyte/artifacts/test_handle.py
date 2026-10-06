"""flyte.artifacts handles, mappings, partition bindings, granularity markers and TimeRange."""

import os
import pickle
from datetime import date, datetime, timedelta, timezone

import pytest

import flyte
import flyte.artifacts as artifacts
from flyte.artifacts import Artifact, ArtifactMapping, PartitionValue, TimePartition, TimeRange
from flyte.artifacts._handle import _Granularity, dimension_record, type_name
from flyte.io import DataFrame, File

UTC = timezone.utc


def _events(**kw):
    return Artifact("events", type=DataFrame, partitions={"date": artifacts.Daily, "region": str}, **kw)


# ---------------------------------------------------------------- granularity markers


@pytest.mark.parametrize(
    "marker,name,registry,fmt,step",
    [
        (artifacts.Daily, "Daily", "day", "%Y-%m-%d", timedelta(days=1)),
        (artifacts.Hourly, "Hourly", "hour", "%Y-%m-%dT%H", timedelta(hours=1)),
        (artifacts.Weekly, "Weekly", "week", "%Y-%m-%d", timedelta(days=7)),
        (artifacts.Monthly, "Monthly", "month", "%Y-%m-%d", None),
    ],
)
def test_granularity_interface(marker, name, registry, fmt, step):
    assert isinstance(marker, _Granularity)
    assert (marker.name, marker.registry, marker.fmt, marker.step) == (name, registry, fmt, step)
    assert repr(marker) == f"artifacts.{name}"
    assert pickle.loads(pickle.dumps(marker)) is marker


@pytest.mark.parametrize(
    "marker,value,floor,formatted,nxt",
    [
        (
            artifacts.Daily,
            datetime(2026, 9, 8, 13, 5),
            datetime(2026, 9, 8, tzinfo=UTC),
            "2026-09-08",
            datetime(2026, 9, 9, tzinfo=UTC),
        ),
        (
            artifacts.Hourly,
            datetime(2026, 9, 8, 13, 5),
            datetime(2026, 9, 8, 13, tzinfo=UTC),
            "2026-09-08T13",
            datetime(2026, 9, 8, 14, tzinfo=UTC),
        ),
        (
            artifacts.Weekly,
            date(2026, 9, 10),
            datetime(2026, 9, 7, tzinfo=UTC),
            "2026-09-07",
            datetime(2026, 9, 14, tzinfo=UTC),
        ),
        (
            artifacts.Monthly,
            date(2026, 12, 10),
            datetime(2026, 12, 1, tzinfo=UTC),
            "2026-12-01",
            datetime(2027, 1, 1, tzinfo=UTC),
        ),
    ],
)
def test_granularity_floor_format_advance(marker, value, floor, formatted, nxt):
    assert marker.floor(value) == floor
    assert marker.format(value) == formatted
    assert marker.advance(floor) == nxt
    assert marker.parse(formatted) == floor


# ---------------------------------------------------------------- TimeRange


def test_trailing_time_range():
    r = TimeRange(days=30, hours=2)
    assert not r.is_absolute
    assert r.delta == timedelta(days=30, hours=2)
    assert r.to_dict() == {"days": 30, "hours": 2}
    assert TimeRange.from_dict(r.to_dict()) == r
    assert r == TimeRange(days=30, hours=2)
    assert hash(r) == hash(TimeRange(days=30, hours=2))
    assert repr(r) == "TimeRange(days=30, hours=2)"


def test_absolute_time_range():
    r = flyte.TimeRange("2026-08-01", "2026-08-03")
    assert r is not None and type(r) is TimeRange
    assert r.is_absolute
    # A date end includes that whole day.
    assert r.start == datetime(2026, 8, 1, tzinfo=UTC)
    assert r.end == datetime(2026, 8, 3, 23, 59, 59, 999999, tzinfo=UTC)
    assert r.delta == timedelta(days=3, microseconds=-1)
    assert repr(r) == "TimeRange('2026-08-01', '2026-08-03')"
    assert len(r.partitions(artifacts.Hourly)) == 72
    assert TimeRange.from_dict(r.to_dict()) == r
    assert r.partitions(artifacts.Daily) == [datetime(2026, 8, d, tzinfo=UTC) for d in (1, 2, 3)]
    assert TimeRange(date(2026, 8, 1), datetime(2026, 8, 2)).end == datetime(2026, 8, 2, tzinfo=UTC)


@pytest.mark.parametrize(
    "args,kwargs,err,match",
    [
        (("2026-08-01",), {}, ValueError, "needs both a start and an end"),
        (("2026-08-03", "2026-08-01"), {}, ValueError, "is before its start"),
        (("2026-08-01", "2026-08-03"), {"days": 1}, ValueError, "either absolute"),
        ((7, "2026-01-01"), {}, TypeError, "For a trailing window use numbers only"),
        ((7,), {"days": 1}, ValueError, "positionally .* or by keyword, not both"),
        ((-1,), {}, ValueError, "must not be negative"),
        ((), {"days": -1}, ValueError, "must not be negative"),
        ((), {"days": "3"}, TypeError, "must be numbers"),
    ],
)
def test_time_range_rejects(args, kwargs, err, match):
    with pytest.raises(err, match=match):
        TimeRange(*args, **kwargs)


@pytest.mark.parametrize(
    "args,kwargs,days,hours",
    [
        ((7,), {}, 7, 0),  # factory's positional days
        ((1, 12), {}, 1, 12),  # positional (days, hours)
        ((1.5,), {}, 1.5, 0),  # fractional days
        ((), {"days": 30}, 30, 0),
        ((), {"hours": 6}, 0, 6),
    ],
)
def test_trailing_time_range_forms(args, kwargs, days, hours):
    r = TimeRange(*args, **kwargs)
    assert (r.days, r.hours, r.is_absolute) == (days, hours, False)
    assert r.delta == timedelta(days=days, hours=hours)
    assert r == TimeRange(days=days, hours=hours)


def test_factory_positional_usage_round_trips():
    # The factory's authored form: window(date=TimeRange(7)) and to_dict/from_dict through its graph JSON.
    r = TimeRange(7)
    assert r.to_dict() == {"days": 7, "hours": 0} and TimeRange.from_dict(r.to_dict()) == r
    assert artifacts.Artifact("w", partitions={"date": artifacts.Daily}).window(date=r).to_dict()["days"] == 7


def test_trailing_range_does_not_enumerate():
    with pytest.raises(ValueError, match="Only an absolute TimeRange"):
        TimeRange(days=1).partitions(artifacts.Daily)


# ---------------------------------------------------------------- construction


def test_handle_attributes_and_source_location():
    line = __import__("inspect").currentframe().f_lineno + 1
    h = Artifact(
        "events",
        type=DataFrame,
        partitions={"date": artifacts.Daily, "region": str},
        description="d",
        kind="data",
        source=True,
        identity="content",
        project="p",
        domain="dev",
    )
    assert h.name == "events"
    assert h.type is DataFrame
    assert h.partitions == {"date": artifacts.Daily, "region": str}
    assert h.dims == ["date", "region"] and h.time_dim == "date"
    assert (h.description, h.kind, h.source, h.identity, h.project, h.domain) == (
        "d",
        "data",
        True,
        "content",
        "p",
        "dev",
    )
    assert h.expected == {}
    assert h.src_line == line
    assert h.src_path == os.path.abspath(__file__)
    assert h.src_file == os.path.relpath(__file__, os.getcwd())
    assert h.level == 2 and not h.is_bare
    assert Artifact("x").level == 1 and Artifact("x").is_bare
    assert repr(h) == "Artifact('events', type=DataFrame, partitions={date: Daily, region: str})"


def test_nothing_registers_at_construction():
    import flyte._environment as envmod

    before = len(envmod._ENVIRONMENT_REGISTRY)
    Artifact("not_registered", type=File)
    assert len(envmod._ENVIRONMENT_REGISTRY) == before


@pytest.mark.parametrize(
    "kwargs,err,match",
    [
        ({"name": ""}, ValueError, "non-empty name"),
        ({"name": "a b"}, ValueError, "may only contain letters"),
        ({"name": "app:x"}, ValueError, "reserved for entity node ids"),
        ({"name": "x", "identity": "hash"}, ValueError, "identity must be 'version' or 'content'"),
        ({"name": "x", "kind": "table"}, ValueError, "kind must be 'model', 'data' or 'generic'"),
        ({"name": "x", "partitions": {"1d": str}}, ValueError, "must be a valid identifier"),
        ({"name": "x", "partitions": {"d": float}}, TypeError, "must be typed str, int, or a granularity marker"),
        ({"name": "x", "partitions": {"a": artifacts.Daily, "b": artifacts.Hourly}}, ValueError, "at most one"),
    ],
)
def test_handle_rejects(kwargs, err, match):
    name = kwargs.pop("name")
    with pytest.raises(err, match=match):
        Artifact(name, **kwargs)


def test_to_dict_record():
    h = _events(description="Cleaned", kind="data").expect(region=["us", "eu"])
    rec = h.to_dict()
    assert rec == {
        "type": "DataFrame",
        "kind": "data",
        "source": False,
        "description": "Cleaned",
        "identity": "version",
        "dims": [{"name": "date", "kind": "time", "granularity": "day"}, {"name": "region", "kind": "str"}],
        "expected": {"region": ["us", "eu"]},
        "src_file": h.src_file,
        "src_line": h.src_line,
    }
    assert dimension_record("n", int) == {"name": "n", "kind": "int"}


def test_type_name():
    assert type_name(None) == ""
    assert type_name(list[DataFrame]) == "list[DataFrame]"
    assert type_name(dict[str, int]) == "dict[str, int]"
    assert type_name(datetime) == "datetime"


# ---------------------------------------------------------------- at / expect


def test_at_builds_metadata():
    card = artifacts.Card(uri="s3://c", format="html", card_type="model")
    h = Artifact("churn_model", type=File, partitions={"date": artifacts.Daily}, kind="model", description="m")
    md = h.at(date=datetime(2026, 9, 8, 13), card=card, version="v1")
    assert isinstance(md, artifacts.Metadata)
    assert md.name == "churn_model" and md.version == "v1" and md.card is card
    assert md.kind == "model" and md.description == "m" and md.version_from_content is False
    assert md.partitions == {"date": TimePartition(datetime(2026, 9, 8, tzinfo=UTC), "day")}


def test_at_partition_values():
    md = _events(identity="content").at(date="2026-09-08", region="us")
    assert md.partitions == {"date": TimePartition(datetime(2026, 9, 8, tzinfo=UTC), "day"), "region": "us"}
    assert md.version_from_content is True
    # A str dimension never becomes a time partition, whatever the value's type.
    h = Artifact("x", partitions={"label": str})
    assert h.at(label=date(2026, 1, 2)).partitions == {"label": "2026-01-02"}
    assert Artifact("bare").at().partitions is None


def test_at_unknown_dimension():
    with pytest.raises(
        ValueError, match=r"events.at\(\): 'events' has no partition dimension 'day'; declared: date, region"
    ):
        _events().at(day="2026-09-08")


def test_expect_returns_handle_and_records_values():
    h = _events()
    e = h.expect(region=["us", "eu"])
    assert e is not h and dict(h.expected) == {}
    assert e.expect(date="2026-09-08").expected == {"region": ("us", "eu"), "date": ("2026-09-08",)}
    with pytest.raises(ValueError, match="has no partition dimension 'zone'"):
        h.expect(zone="x")
    with pytest.raises(ValueError, match="needs at least one value"):
        h.expect(region=[])


# ---------------------------------------------------------------- mappings


def test_identity_all_window_select():
    h = _events()
    ident = h.identity_mapping
    assert (ident.kind, ident.handle, ident.is_list, ident.to_dict()) == ("identity", h, False, {"kind": "identity"})

    a = h.all("region")
    assert isinstance(a, ArtifactMapping)
    assert (a.kind, a.dim, a.window, a.range, a.name, a.is_list) == ("all", "region", None, None, "events", True)
    assert a.to_dict() == {"kind": "all", "dim": "region"}
    assert a.describe() == "all('region')"

    w = h.window(date=TimeRange(days=30))
    assert (w.kind, w.dim, w.window.days, w.range.days, w.is_list) == ("window", "date", 30, 30, True)
    assert w.to_dict() == {"kind": "window", "dim": "date", "days": 30, "hours": 0}

    s = h.select(region="us", date=date(2026, 9, 8))
    assert (s.kind, dict(s.pinned), s.is_list) == ("select", {"region": "us", "date": "2026-09-08"}, False)
    assert s.to_dict() == {"kind": "select", "values": {"region": "us", "date": "2026-09-08"}}
    assert hash(s) is not None


@pytest.mark.parametrize(
    "call,err,match",
    [
        (lambda h: h.all("zone"), ValueError, r"events.all\(\): 'events' has no partition dimension 'zone'"),
        (lambda h: h.window(), ValueError, "exactly one dimension"),
        (lambda h: h.window(date=TimeRange(days=1), region=TimeRange(days=1)), ValueError, "exactly one dimension"),
        (lambda h: h.window(region=TimeRange(days=1)), ValueError, "'region' is not a time dimension"),
        (lambda h: h.window(date=7), TypeError, "must be artifacts.TimeRange"),
        (lambda h: h.window(date=TimeRange("2026-01-01", "2026-01-02")), ValueError, "not an absolute range"),
        (lambda h: h.window(date=TimeRange()), ValueError, "non-empty TimeRange"),
        (lambda h: h.select(), ValueError, "at least one dimension"),
        (lambda h: h.select(zone="x"), ValueError, "has no partition dimension 'zone'"),
        (
            lambda h: h.get_partition_value("zone"),
            ValueError,
            r"get_partition_value\(\): 'events' has no partition dimension",
        ),
    ],
)
def test_mapping_rejects(call, err, match):
    with pytest.raises(err, match=match):
        call(_events())


def test_get_partition_value():
    h = _events()
    b = h.get_partition_value("date")
    assert isinstance(b, PartitionValue)
    assert (b.handle, b.dim, b.name, b.dimension_type) == (h, "date", "events", artifacts.Daily)


def test_partition_value_normalization():
    h = _events()
    assert h._coerce_partition_value("date", TimePartition(date(2026, 9, 8), "week")) == TimePartition(
        date(2026, 9, 8), "day"
    )
    assert h._coerce_partition_value("region", 7) == "7"
    with pytest.raises(ValueError, match="is None"):
        h._coerce_partition_value("region", None)


def test_handles_pickle():
    h = _events().expect(region=["us"])
    h2 = pickle.loads(pickle.dumps(h))
    assert h2.name == "events" and h2.partitions["date"] is artifacts.Daily and h2.expected == {"region": ("us",)}
    m = pickle.loads(pickle.dumps(h.window(date=TimeRange(days=3))))
    assert m.window == TimeRange(days=3)


def test_artifact_like_protocol_still_exported():
    from flyte.artifacts import ArtifactLike, new

    f = File(path="s3://b/k")
    assert isinstance(new(f, artifacts.Metadata(name="x")), ArtifactLike)


def test_artifact_subscript_is_the_class():
    assert artifacts.Artifact[File] is artifacts.Artifact

    def f(x: "artifacts.Artifact[File]") -> None: ...

    assert f is not None


# ------------------------------------------------------------------ 13. app labels validated before builds


@pytest.mark.parametrize(
    "marker,value,expected",
    [
        (artifacts.Daily, "2026-09-08T13:45:00Z", "2026-09-08"),
        (artifacts.Daily, datetime(2026, 9, 8, 23, 59), "2026-09-08"),
        (artifacts.Hourly, "2026-09-08T13:45:00Z", "2026-09-08T13"),
        (artifacts.Weekly, date(2026, 9, 10), "2026-09-07"),
        (artifacts.Monthly, "2026-09-18", "2026-09-01"),
    ],
)
def test_select_floors_time_values(marker, value, expected):
    h = Artifact("sel", partitions={"t": marker})
    assert dict(h.select(t=value).pinned) == {"t": expected}


def test_handle_record_carries_scope_when_set():
    scoped = Artifact("scoped", type=File, project="ml", domain="prod")
    assert {k: scoped.to_dict()[k] for k in ("project", "domain")} == {"project": "ml", "domain": "prod"}
    assert "project" not in Artifact("unscoped", type=File).to_dict()


def test_lineage_declaration_error_worker():
    from flyte.errors import LineageDeclarationError

    e = LineageDeclarationError("x")
    assert e.code == "LineageDeclarationError" and e.worker is None


@pytest.mark.parametrize("kwargs", [{"days": 1.5}, {"hours": 0.5}, {"days": 2, "hours": 6}, {"days": 7}])
def test_trailing_time_range_dict_round_trip_keeps_fractions(kwargs):
    # The window travels through lineage.bindings as JSON; the planner reads it back with from_dict.
    rng = TimeRange(**kwargs)
    back = TimeRange.from_dict(rng.to_dict())
    assert back == rng and back.delta == rng.delta
    assert TimeRange.from_dict({"days": 7.0}).days == 7 and isinstance(TimeRange.from_dict({"days": 7.0}).days, int)


def test_window_rejects_monthly_dimension():
    monthly = Artifact("monthly", partitions={"month": artifacts.Monthly})
    with pytest.raises(ValueError, match=r"'month' is Monthly, and a window needs a fixed step"):
        monthly.window(month=TimeRange(days=60))
    assert monthly.all("month").kind == "all"


def test_on_artifact_rejects_a_handle_scoped_elsewhere():
    lake = Artifact("events", partitions={"date": artifacts.Daily}, project="lake")
    with pytest.raises(ValueError, match=r"lives in project='lake'.*OnArtifact\('events'\)"):
        flyte.OnArtifact(lake)
    assert flyte.OnArtifact(Artifact("events")).name == "events"


def test_on_artifact_checks_partition_filters_against_the_handle():
    raw = Artifact("raw_events", partitions={"date": artifacts.Daily, "region": str})
    assert flyte.OnArtifact(raw, region="us").partitions == {"region": "us"}
    with pytest.raises(ValueError, match=r"no partition dimension 'regoin'; declared: date, region"):
        flyte.OnArtifact(raw, regoin="us")
    with pytest.raises(ValueError, match=r"'date' is a time dimension.*TriggeredPartition\('date'\)"):
        flyte.OnArtifact(raw, date="2026-09-08")
    # A name carries no declaration, so it is not checked.
    assert flyte.OnArtifact("raw_events", regoin="us").partitions == {"regoin": "us"}


def test_expect_formats_time_values_like_the_registry():
    h = Artifact("events", partitions={"date": artifacts.Daily, "region": str})
    h = h.expect(date=[datetime(2026, 9, 8, 13, 5), "2026-09-09"], region="us")
    assert h.expected == {"date": ("2026-09-08", "2026-09-09"), "region": ("us",)}


def test_time_range_equality_normalizes():
    assert TimeRange(days=1) == TimeRange(hours=24) == TimeRange(1)
    assert hash(TimeRange(days=1)) == hash(TimeRange(hours=24))
    assert TimeRange(days=1) != TimeRange(days=2)
    assert TimeRange("2026-08-01", "2026-08-03") == TimeRange(date(2026, 8, 1), date(2026, 8, 3))
    # A datetime end is the exact instant, not the whole day.
    assert TimeRange("2026-08-01", "2026-08-03") != TimeRange("2026-08-01", datetime(2026, 8, 3, tzinfo=UTC))


@pytest.mark.parametrize("kwargs", [{"days": float("nan")}, {"hours": float("inf")}, {"days": -1}])
def test_time_range_rejects_non_finite_and_negative(kwargs):
    with pytest.raises(ValueError):
        TimeRange(**kwargs)


def test_monthly_advance_from_month_end():
    assert artifacts.Monthly.advance(datetime(2026, 1, 31, tzinfo=UTC)) == datetime(2026, 2, 1, tzinfo=UTC)
    assert artifacts.Monthly.advance(datetime(2026, 12, 31, tzinfo=UTC)) == datetime(2027, 1, 1, tzinfo=UTC)


def test_window_must_be_whole_steps():
    h = Artifact("w", partitions={"date": artifacts.Daily})
    with pytest.raises(ValueError, match="not a whole number of Daily partitions"):
        h.window(date=TimeRange(hours=12))
    assert h.window(date=TimeRange(hours=48)).window == TimeRange(days=2)


def test_int_dimension_normalization():
    h = Artifact("n", partitions={"lag": int, "region": str})
    assert h.select(lag=7).pinned == {"lag": "7"}
    assert h.select(lag=7.0).pinned == {"lag": "7"}
    assert h.select(lag=" 7 ").pinned == {"lag": "7"}
    assert h.at(lag="7.0", region="us").partitions == {"lag": "7", "region": "us"}
    assert h.expect(lag=[1, "2"]).expected["lag"] == ("1", "2")
    for bad in (True, 7.5, "x", float("nan")):
        with pytest.raises((TypeError, ValueError)):
            h.select(lag=bad)
    # A datetime on a str dimension has one form everywhere.
    dt = datetime(2026, 9, 8, 13, 5, tzinfo=UTC)
    assert h.select(region=dt).pinned["region"] == h.at(lag=1, region=dt).partitions["region"] == dt.isoformat()


def test_handles_are_immutable_and_expect_copies():
    h = _events()
    e = h.expect(region=["us"])
    assert e is not h and e == h and hash(e) == hash(h)
    assert dict(h.expected) == {} and dict(e.expected) == {"region": ("us",)}
    with pytest.raises(AttributeError, match="immutable"):
        h.name = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        h.partitions["x"] = str  # type: ignore[index]


def test_handle_equality():
    assert _events() == _events()
    assert Artifact("a") != Artifact("b")
    assert Artifact("a", partitions={"d": str}) != Artifact("a", partitions={"d": int})
    assert Artifact("a") != Artifact.ref("a")
    assert Artifact("a", project="p") != Artifact("a")


def test_pickle_drops_source_path():
    h = _events()
    assert h.src_path
    h2 = pickle.loads(pickle.dumps(h))
    assert h2.src_path == "" and h2 == h and h2.src_line == h.src_line
    assert pickle.dumps(h2) == pickle.dumps(pickle.loads(pickle.dumps(h2)))
