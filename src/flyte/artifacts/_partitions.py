"""
Partition values: how a Python value becomes an artifact's partition identity.

An artifact version carries at most one time partition plus any number of string
partitions. The mapping key is the partition key; the value decides the kind:

- a `datetime.date` is a daily time partition,
- a `datetime.datetime` is an hourly one (floored to the hour, in UTC; naive means UTC),
- a `TimePartition(value, granularity)` names the granularity explicitly
  ("hour", "day", "week", "month"),
- anything else is a string partition, via `str()`.

The set of keys is fixed per artifact name by its first version; see
`flyte.remote.Artifact.declare` to fix it ahead of any version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal, Mapping, Optional, Tuple

from flyteidl2.core import artifact_id_pb2
from google.protobuf.timestamp_pb2 import Timestamp

Granularity = Literal["hour", "day", "week", "month"]

GRANULARITY_TO_PB2: dict[str, "artifact_id_pb2.Granularity.ValueType"] = {
    "hour": artifact_id_pb2.Granularity.HOUR,
    "day": artifact_id_pb2.Granularity.DAY,
    "week": artifact_id_pb2.Granularity.WEEK,
    "month": artifact_id_pb2.Granularity.MONTH,
}
GRANULARITY_FROM_PB2: dict[int, Granularity] = {
    artifact_id_pb2.Granularity.HOUR: "hour",
    artifact_id_pb2.Granularity.DAY: "day",
    artifact_id_pb2.Granularity.WEEK: "week",
    artifact_id_pb2.Granularity.MONTH: "month",
}

_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_HOUR = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}$")


@dataclass(frozen=True)
class TimePartition:
    """
    A time partition value with an explicit granularity.

    `date` and `datetime` values already imply "day" and "hour"; use this for
    "week" and "month", or to be explicit:

    ```python
    Metadata(name="monthly_report", partitions={"date": TimePartition(date(2026, 8, 1), "month")})
    ```
    """

    value: date | datetime
    granularity: Granularity = "day"

    def __post_init__(self):
        if self.granularity not in GRANULARITY_TO_PB2:
            raise ValueError(
                f"Unknown time partition granularity {self.granularity!r}; use one of {sorted(GRANULARITY_TO_PB2)}"
            )
        if not isinstance(self.value, (date, datetime)):
            raise TypeError(f"TimePartition value must be a date or datetime, got {type(self.value).__name__}")


def is_time_value(value: Any) -> bool:
    """True for values the partition rules treat as a time partition."""
    return isinstance(value, (date, datetime, TimePartition))


def to_utc(value: date | datetime) -> datetime:
    """A `date` becomes midnight UTC; a naive `datetime` is taken as UTC; aware ones convert."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)


def floor_time(value: date | datetime, granularity: Granularity) -> datetime:
    """The start of the partition containing `value`, in UTC."""
    dt = to_utc(value)
    if granularity == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if granularity == "day":
        return dt
    if granularity == "week":
        return dt - timedelta(days=dt.weekday())
    if granularity == "month":
        return dt.replace(day=1)
    raise ValueError(f"Unknown time partition granularity {granularity!r}")


def time_value(value: Any) -> Tuple[datetime, Granularity]:
    """Resolve a time-valued partition to its floored UTC datetime and granularity."""
    if isinstance(value, TimePartition):
        return floor_time(value.value, value.granularity), value.granularity
    if isinstance(value, datetime):
        return floor_time(value, "hour"), "hour"
    if isinstance(value, date):
        return floor_time(value, "day"), "day"
    raise TypeError(f"{type(value).__name__} is not a time partition value")


def default_time_key(granularity: Granularity) -> str:
    """The time key artifacts agree on without coordination: "hour" for hourly, "date" otherwise."""
    return "hour" if granularity == "hour" else "date"


def partitions_to_pb2(
    partitions: Optional[Mapping[str, Any]],
) -> Tuple[Optional[artifact_id_pb2.Partitions], Optional[artifact_id_pb2.TimePartition]]:
    """
    Split a partitions mapping into the wire messages: the string partitions and
    the (at most one) time partition. Raises ValueError on two time values or an
    empty key.
    """
    if not partitions:
        return None, None
    strings: dict[str, artifact_id_pb2.LabelValue] = {}
    time_partition: Optional[artifact_id_pb2.TimePartition] = None
    for key, value in partitions.items():
        if not key:
            raise ValueError("Partition keys must be non-empty strings")
        if is_time_value(value):
            if time_partition is not None:
                raise ValueError(
                    f"An artifact can carry at most one time partition; both {time_partition.key!r} and {key!r} "
                    "hold time values. Make one of them a string."
                )
            floored, granularity = time_value(value)
            ts = Timestamp()
            ts.FromDatetime(floored)
            time_partition = artifact_id_pb2.TimePartition(
                value=artifact_id_pb2.LabelValue(time_value=ts),
                granularity=GRANULARITY_TO_PB2[granularity],
                key=key,
            )
        else:
            strings[key] = artifact_id_pb2.LabelValue(static_value=str(value))
    return (artifact_id_pb2.Partitions(value=strings) if strings else None), time_partition


def time_partition_to_python(tp: artifact_id_pb2.TimePartition) -> date | datetime:
    """A `datetime` (UTC) for an hourly partition, a `date` for every other granularity."""
    dt = tp.value.time_value.ToDatetime(tzinfo=timezone.utc)
    granularity = GRANULARITY_FROM_PB2.get(tp.granularity, "day")
    if granularity == "hour":
        return dt
    return dt.date()


def partitions_from_pb2(
    partitions: Optional[artifact_id_pb2.Partitions],
    time_partition: Optional[artifact_id_pb2.TimePartition],
) -> dict[str, Any]:
    """The Python view of stored partitions: time value first, then string values."""
    out: dict[str, Any] = {}
    if time_partition is not None and time_partition.HasField("value"):
        key = time_partition.key or default_time_key(GRANULARITY_FROM_PB2.get(time_partition.granularity, "day"))
        out[key] = time_partition_to_python(time_partition)
    if partitions is not None:
        for key, lv in partitions.value.items():
            out[key] = lv.static_value
    return out


def to_rfc3339(value: date | datetime) -> str:
    """RFC3339 in UTC, the form the artifact service's time_partition filter takes."""
    return to_utc(value).isoformat().replace("+00:00", "Z")


def parse_time(text: str | date | datetime) -> datetime:
    """Accept a date, datetime, ISO date, ISO hour (`2026-08-01T13`), or RFC3339 string."""
    if isinstance(text, (date, datetime)):
        return to_utc(text)
    s = text.strip()
    if _ISO_DATE.match(s):
        return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if _ISO_HOUR.match(s):
        return datetime.strptime(s, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
    if _RFC3339.match(s):
        return to_utc(datetime.fromisoformat(s.replace("Z", "+00:00")))
    return to_utc(datetime.fromisoformat(s))


def looks_like_time(text: str) -> bool:
    """True for strings the CLI turns into time values rather than string partitions."""
    return bool(_ISO_DATE.match(text) or _ISO_HOUR.match(text) or _RFC3339.match(text))
