"""
Partition values: how a Python value becomes an artifact's partition identity.

An artifact version carries at most one time partition plus any number of string
partitions. The mapping key is the partition key; the value decides the kind:

- a `datetime.date` is a daily time partition,
- a `datetime.datetime` is an hourly one (floored to the hour, in UTC; naive means UTC),
- a `TimePartition(value, granularity)` names the granularity explicitly
  ("hour", "day", "week", "month"),
- anything else is a string partition, via `str()`: ints, floats and bools are
  stored as their string form ("7", "1.5", "True"); `None` is rejected.

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

#: Granularity names in the order the registry defines them, and their enum names.
GRANULARITIES: tuple[Granularity, ...] = ("hour", "day", "week", "month")
_GRANULARITY_ENUM_NAME: dict[str, str] = {"hour": "HOUR", "day": "DAY", "week": "WEEK", "month": "MONTH"}
_GRANULARITY_FROM_ENUM_NAME: dict[str, Granularity] = {v: k for k, v in _GRANULARITY_ENUM_NAME.items()}  # type: ignore[misc]

#: The flyteidl2 release that carries every granularity (WEEK was added there).
_IDL_RELEASE_WITH_WEEK = "2.0.46"


def granularity_to_pb2(granularity: str) -> int:
    """
    The wire enum value for a granularity name. Resolved on use rather than at
    import so that `import flyte` works against an older flyteidl2 that lacks a
    value (WEEK); only a use of that granularity fails, and says what it needs.
    """
    try:
        enum_name = _GRANULARITY_ENUM_NAME[granularity]
    except KeyError:
        raise ValueError(
            f"Unknown time partition granularity {granularity!r}; use one of {list(GRANULARITIES)}"
        ) from None
    try:
        return artifact_id_pb2.Granularity.Value(enum_name)
    except ValueError:
        raise RuntimeError(
            f"Time partition granularity {granularity!r} needs flyteidl2 >= {_IDL_RELEASE_WITH_WEEK}; the "
            f"installed flyteidl2 has no Granularity.{enum_name}"
        ) from None


def granularity_from_pb2(value: int) -> Granularity:
    """The granularity name for a wire enum value; unknown or UNSET reads as "day"."""
    try:
        return _GRANULARITY_FROM_ENUM_NAME.get(artifact_id_pb2.Granularity.Name(value), "day")
    except ValueError:
        return "day"


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
        if self.granularity not in _GRANULARITY_ENUM_NAME:
            raise ValueError(
                f"Unknown time partition granularity {self.granularity!r}; use one of {list(GRANULARITIES)}"
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
        if value is None:
            raise ValueError(
                f"Partition {key!r} is None. A partition value must be a date, datetime, TimePartition, "
                "or a value with a string form; an unset value is almost always a mistake."
            )
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
                granularity=granularity_to_pb2(granularity),
                key=key,
            )
        else:
            strings[key] = artifact_id_pb2.LabelValue(static_value=str(value))
    return (artifact_id_pb2.Partitions(value=strings) if strings else None), time_partition


def time_partition_to_python(tp: artifact_id_pb2.TimePartition) -> date | datetime:
    """A `datetime` (UTC) for an hourly partition, a `date` for every other granularity."""
    dt = tp.value.time_value.ToDatetime(tzinfo=timezone.utc)
    granularity = granularity_from_pb2(tp.granularity)
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
        key = time_partition.key or default_time_key(granularity_from_pb2(time_partition.granularity))
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
