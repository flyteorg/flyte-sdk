"""
Artifact handles: a name, a value type and partition dimensions, declared once beside the code that owns
the data and imported wherever it is produced or consumed.

A handle is an ordinary Python object. Constructing one registers nothing; deploy reads handles off the
`produces_artifacts=` and `consumes_artifacts=` keywords of the tasks that name them.

```python
import flyte.artifacts as artifacts
from flyte.io import DataFrame

events = artifacts.Artifact(
    "events",
    type=DataFrame,
    partitions={"date": artifacts.Daily, "region": str},
    description="Cleaned event stream, one partition per region per day.",
)

events.all("region")                                  # every region, for the consumer's date
events.window(date=artifacts.TimeRange(days=7))       # a trailing week, for the consumer's region
events.select(region="us")                            # one region pinned, date by identity
events.get_partition_value("date")                    # a parameter that carries the date coordinate
events.at(date=day, region="us")                      # Metadata for artifacts.new(value, ...)
```
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from typing_extensions import TypeGuard

from ._partitions import TimePartition, floor_time, parse_time

#: A lineage node id: an artifact name, or a prefixed entity id such as `app:churn-scoring`.
NODE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")

MappingKind = Literal["identity", "all", "window", "select"]
Identity = Literal["version", "content"]
HandleKind = Literal["model", "data", "generic"]


# --------------------------------------------------------------------------------------------------
# Granularity markers and time ranges
# --------------------------------------------------------------------------------------------------


def _next_month(value: datetime) -> datetime:
    year, month = (value.year + 1, 1) if value.month == 12 else (value.year, value.month + 1)
    return value.replace(year=year, month=month)


class _Granularity:
    """
    Marker for a time partition dimension: `artifacts.Daily`, `Hourly`, `Weekly` or `Monthly`.

    Attributes:
        name: The marker's name, e.g. `"Daily"`.
        step: Distance between adjacent partitions, or None when it is not fixed (months).
        fmt: `strftime` format that names one partition.
        registry: Granularity name the artifact registry uses: `hour`, `day`, `week` or `month`.
    """

    def __init__(
        self,
        name: str,
        step: Optional[timedelta],
        fmt: str,
        registry: str,
        advance: Optional[Callable[[datetime], datetime]] = None,
    ):
        self.name = name
        self.step = step
        self.fmt = fmt
        self.registry = registry
        self._advance = advance

    def __repr__(self) -> str:
        return f"artifacts.{self.name}"

    def __reduce__(self):
        # Markers are singletons: unpickling yields the module-level object, so `is` comparisons hold.
        return (_granularity_by_name, (self.name,))

    def format(self, value: date | datetime) -> str:
        """The partition name holding `value`, e.g. `"2026-09-08"` for `Daily`."""
        return self.floor(value).strftime(self.fmt)

    def parse(self, text: str | date | datetime) -> datetime:
        """Any ISO date, ISO hour or RFC3339 string (or a date/datetime), floored to this granularity."""
        return self.floor(parse_time(text))

    def floor(self, value: date | datetime) -> datetime:
        """The start of the partition holding `value`, in UTC (a naive datetime is taken as UTC)."""
        return floor_time(value, self.registry)  # type: ignore[arg-type]

    def advance(self, value: datetime) -> datetime:
        """The start of the partition after the one starting at `value`."""
        if self._advance is not None:
            return self._advance(value)
        assert self.step is not None
        return value + self.step


Daily = _Granularity("Daily", timedelta(days=1), "%Y-%m-%d", "day")
Hourly = _Granularity("Hourly", timedelta(hours=1), "%Y-%m-%dT%H", "hour")
#: Weekly partitions are named by their Monday.
Weekly = _Granularity("Weekly", timedelta(days=7), "%Y-%m-%d", "week")
#: Monthly partitions are named by their first day.
Monthly = _Granularity("Monthly", None, "%Y-%m-%d", "month", advance=_next_month)

GRANULARITY_MARKERS: Tuple[_Granularity, ...] = (Daily, Hourly, Weekly, Monthly)


def _granularity_by_name(name: str) -> _Granularity:
    for g in GRANULARITY_MARKERS:
        if g.name == name:
            return g
    raise ValueError(f"Unknown granularity {name!r}")


#: A partition dimension is typed as `str`, `int`, or a granularity marker for time.
DimensionType = Union[type, _Granularity]


def _whole(value: Any) -> Union[int, float]:
    """A JSON number as an int when it is whole (`7.0` -> `7`), else unchanged; a numeric string is parsed."""
    v = float(value) if isinstance(value, str) else value
    return int(v) if isinstance(v, float) and v.is_integer() else v


def _as_datetime(value: Any, what: str) -> datetime:
    if isinstance(value, (str, date, datetime)):
        return parse_time(value)
    raise TypeError(
        f"TimeRange {what} must be a date, datetime or ISO string, got {type(value).__name__} ({value!r}). "
        "For a trailing window use numbers only: TimeRange(30) or TimeRange(days=30)."
    )


@dataclass(frozen=True, init=False)
class TimeRange:
    """
    A range of time, in one of two forms.

    A *trailing window* is relative to the consumer's own time value and ends at it. It is what
    `Artifact.window` takes, by keyword or positionally as `(days, hours)`:

    ```python
    features.window(date=artifacts.TimeRange(days=30))
    features.window(date=artifacts.TimeRange(30))       # the same
    features.window(date=artifacts.TimeRange(1, 12))    # 1 day 12 hours
    ```

    An *absolute range* (positional start and end, inclusive) is what `flyte.materialize` takes to
    backfill a span of partitions:

    ```python
    flyte.materialize(daily_report, date=flyte.TimeRange("2026-08-01", "2026-08-31"), concurrency=50)
    ```

    Attributes:
        start: Inclusive start of an absolute range (UTC), or None for a trailing window.
        end: Inclusive end of an absolute range (UTC), or None for a trailing window.
        days: Length of a trailing window, in days (an int, or a float for fractional days).
        hours: Length of a trailing window, in hours (added to `days`).
    """

    start: Optional[datetime]
    end: Optional[datetime]
    days: Union[int, float]
    hours: Union[int, float]

    def __init__(
        self,
        start: Union[str, date, datetime, int, float, None] = None,
        end: Union[str, date, datetime, int, float, None] = None,
        *,
        days: Union[int, float] = 0,
        hours: Union[int, float] = 0,
    ):
        def _number(v: Any) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        # Positional numbers are the trailing-window form the factory has always accepted:
        # TimeRange(7) is 7 days, TimeRange(1, 12) is a day and a half.
        if _number(start) and (end is None or _number(end)):
            if days or hours:
                raise ValueError("Give a trailing TimeRange either positionally (days, hours) or by keyword, not both")
            days, hours = start, (end or 0)  # type: ignore[assignment]
            start = end = None
        if (start is None) != (end is None):
            raise ValueError("An absolute TimeRange needs both a start and an end, e.g. TimeRange('2026-08-01', ...)")
        s = _as_datetime(start, "start") if start is not None else None
        e = _as_datetime(end, "end") if end is not None else None
        if s is not None and e is not None:
            if days or hours:
                raise ValueError("A TimeRange is either absolute (start, end) or a trailing window (days=, hours=)")
            if e < s:
                raise ValueError(f"TimeRange end {end!r} is before its start {start!r}")
        else:
            if not _number(days) or not _number(hours):
                raise TypeError("TimeRange days and hours must be numbers")
            if days < 0 or hours < 0:
                raise ValueError("TimeRange days and hours must not be negative")
        object.__setattr__(self, "start", s)
        object.__setattr__(self, "end", e)
        object.__setattr__(self, "days", days)
        object.__setattr__(self, "hours", hours)

    def __repr__(self) -> str:
        if self.is_absolute:
            assert self.start is not None and self.end is not None
            return f"TimeRange({self.start.isoformat()!r}, {self.end.isoformat()!r})"
        parts = [f"{k}={v}" for k, v in (("days", self.days), ("hours", self.hours)) if v]
        return f"TimeRange({', '.join(parts)})"

    @property
    def is_absolute(self) -> bool:
        """True for a `(start, end)` range, False for a trailing window."""
        return self.start is not None

    @property
    def delta(self) -> timedelta:
        """Length of a trailing window; for an absolute range, `end - start`."""
        if self.is_absolute:
            assert self.start is not None and self.end is not None
            return self.end - self.start
        return timedelta(days=self.days, hours=self.hours)

    def to_dict(self) -> Dict[str, Any]:
        """JSON form: `{"days", "hours"}` for a window, `{"start", "end"}` (RFC3339) for a range."""
        if self.is_absolute:
            assert self.start is not None and self.end is not None
            return {"start": self.start.isoformat(), "end": self.end.isoformat()}
        return {"days": self.days, "hours": self.hours}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TimeRange":
        """Inverse of `to_dict`."""
        if d.get("start") is not None:
            return cls(d["start"], d["end"])
        # Keep fractional windows: `TimeRange(days=1.5)` must not come back as one day.
        return cls(days=_whole(d.get("days", 0) or 0), hours=_whole(d.get("hours", 0) or 0))

    def partitions(self, granularity: _Granularity) -> List[datetime]:
        """The partition starts an absolute range covers at `granularity`, in order."""
        if not self.is_absolute:
            raise ValueError("Only an absolute TimeRange(start, end) enumerates partitions")
        assert self.start is not None and self.end is not None
        out: List[datetime] = []
        cur = granularity.floor(self.start)
        last = granularity.floor(self.end)
        while cur <= last:
            out.append(cur)
            cur = granularity.advance(cur)
        return out


# --------------------------------------------------------------------------------------------------
# Source location
# --------------------------------------------------------------------------------------------------

_FLYTE_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def relative_source_path(path: str, root: Optional[str] = None) -> str:
    """`path` relative to `root` (default: the working directory) when it lies under it, else unchanged."""
    if not path or path.startswith("<"):
        return path
    base = os.path.abspath(root) if root else os.getcwd()
    abspath = os.path.abspath(path)
    try:
        if os.path.commonpath([base, abspath]) == base:
            return os.path.relpath(abspath, base)
    except ValueError:
        pass
    return path


_INSIDE_FLYTE_CACHE: Dict[str, bool] = {}
_ABSPATH_CACHE: Dict[str, str] = {}


def _is_flyte_file(filename: str) -> bool:
    inside = _INSIDE_FLYTE_CACHE.get(filename)
    if inside is None:
        if not filename or filename.startswith("<"):
            inside = True
        else:
            abspath = os.path.abspath(filename)
            try:
                inside = os.path.commonpath([_FLYTE_PKG_DIR, abspath]) == _FLYTE_PKG_DIR
            except ValueError:
                inside = False
        _INSIDE_FLYTE_CACHE[filename] = inside
    return inside


def _caller_location() -> Tuple[str, int]:
    """Absolute file and line of the first frame outside the flyte package (the user's declaration).

    Uses `sys._getframe` and a per-filename cache: no `inspect.stack()`, which reads source files.
    """
    f: Any = sys._getframe(1)
    while f is not None:
        filename = f.f_code.co_filename
        if not _is_flyte_file(filename):
            return _ABSPATH_CACHE.get(filename) or _ABSPATH_CACHE.setdefault(
                filename, os.path.abspath(filename)
            ), f.f_lineno
        f = f.f_back
    return "", 0


# --------------------------------------------------------------------------------------------------
# Handles, mappings and partition bindings
# --------------------------------------------------------------------------------------------------


def type_name(t: Any) -> str:
    """Short display name of a value type: `File`, `DataFrame`, `list[DataFrame]`; empty for None."""
    if t is None:
        return ""
    if isinstance(t, str):
        return t
    import typing

    origin = typing.get_origin(t)
    if origin is not None:
        args = typing.get_args(t)
        origin_name = getattr(origin, "__name__", str(origin))
        if origin is Union:
            return " | ".join(type_name(a) for a in args)
        if args:
            return f"{origin_name}[{', '.join(type_name(a) for a in args)}]"
        return origin_name
    if t is type(None):
        return "None"
    return getattr(t, "__name__", repr(t))


def dimension_kind(t: DimensionType) -> str:
    """`"time"` for a granularity marker, `"str"` for `str`, `"int"` for `int`."""
    if isinstance(t, _Granularity):
        return "time"
    if t is int:
        return "int"
    return "str"


class _HandleBase:
    """Plain base of every handle. `isinstance(x, _HandleBase)` (or `is_handle`) is the strict handle check."""

    __slots__ = ()


class _ArtifactMeta(type):
    """
    Backward compatibility: in flyte 2.10.0-2.10.7 `flyte.artifacts.Artifact` was a `runtime_checkable` protocol
    for artifact-like values (now `ArtifactLike`). `isinstance(x, Artifact)` stays True for such values, with a
    DeprecationWarning. Internal code uses `is_handle`, which never takes this fallback.
    """

    def __instancecheck__(cls, obj: Any) -> bool:
        if type.__instancecheck__(cls, obj):
            return True
        if cls.__name__ != "Artifact" or cls.__module__ != __name__:
            return False  # subclasses (ArtifactRef) keep the plain check
        from ._wrapper import ArtifactLike

        if isinstance(obj, ArtifactLike):
            warnings.warn(
                "isinstance(x, flyte.artifacts.Artifact) as an artifact-like protocol check is deprecated; "
                "flyte.artifacts.Artifact is now the artifact handle class. Use flyte.artifacts.ArtifactLike.",
                DeprecationWarning,
                stacklevel=2,
            )
            return True
        return False


def is_handle(obj: Any) -> TypeGuard["Artifact"]:
    """True for a real artifact handle (`Artifact` or `Artifact.ref`), never for an artifact-like value."""
    return isinstance(obj, _HandleBase)


class Artifact(_HandleBase, metaclass=_ArtifactMeta):
    """
    A handle to a named artifact: its value type, its partition dimensions, and what it is.

    Declare it once, at the top of the module that owns the data, and import it wherever the artifact is
    produced or consumed. A handle is a plain object: nothing is registered when it is constructed. Deploy
    reads handles off `@env.task(consumes_artifacts={...}, produces_artifacts=(handle,))`, so the artifact
    and its edges exist in the lineage graph before any version is published.

    ```python
    raw_events = artifacts.Artifact(
        "raw_events", type=File, partitions={"date": artifacts.Daily, "region": str}, source=True,
    )
    events = artifacts.Artifact("events", type=DataFrame, partitions={"date": artifacts.Daily, "region": str})

    @env.task(
        consumes_artifacts={
            "raw": raw_events,                                  # identity on date + region
            "date": raw_events.get_partition_value("date"),
            "region": raw_events.get_partition_value("region"),
        },
        produces_artifacts=(events,),
    )
    async def clean(raw: File, date: datetime, region: str, min_quality: int = 30) -> DataFrame: ...
    ```

    Args:
        name: The artifact name, the node id in the lineage graph. Letters, digits, `_ . : -`.
        type: The value type, e.g. `flyte.io.DataFrame` or `flyte.io.File`.
        partitions: Partition dimensions in order, each typed `str`, `int`, or a granularity marker
            (`artifacts.Daily`, `Hourly`, `Weekly`, `Monthly`) for time. At most one time dimension.
        description: Human-readable description, published with each version.
        source: True when nothing in your code produces this artifact; it lands from outside.
        kind: What the artifact is: `"model"`, `"data"` or `"generic"`.
        identity: `"version"` (default) publishes a new version on every run; `"content"` publishes under
            the content hash, so identical content is not published twice.
        project: Project the artifact lives in, when it is not the deploying project.
        domain: Domain the artifact lives in, when it is not the deploying domain.
        refresh: When to materialize it, kept fresh by the platform: an `artifacts.Refresh(event, lag=...)`, a bare
            `flyte.Cron(...)` / `flyte.FixedRate(...)` / source handle, or a list of them. Registered by a deploy
            that produces the artifact (not by modules that merely import the handle). See `Refresh`.

    Attributes:
        name: The artifact name.
        type: The declared value type, or None.
        partitions: Ordered mapping of dimension name to its type (`str`, `int` or a granularity marker).
        description: The description, or None.
        source: Whether the artifact lands from outside.
        kind: `"model"`, `"data"`, `"generic"`, or None.
        identity: `"version"` or `"content"`.
        project: The project override, or None.
        domain: The domain override, or None.
        expected: Expected values per dimension, recorded by `expect` (dimension name to tuple of str).
        refresh: The refresh policies (`artifacts.Refresh`), in declaration order.
        src_file: File that constructed the handle, relative to the working directory when under it.
        src_line: Line that constructed the handle.
    """

    #: True for `Artifact.ref(...)`: a handle to an artifact another codebase owns, restated to read it.
    reference: bool = False

    @classmethod
    def ref(
        cls,
        name: str,
        *,
        type: Any = None,
        partitions: Optional[Mapping[str, DimensionType]] = None,
        project: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> "ArtifactRef":
        """
        A handle to an artifact owned by code you cannot import: another team's repo, another project.

        State only what you need to read it (its name, partitions and, optionally, type), the way a reference
        task states another task's interface. Everything the owner decides (description, kind, `source`,
        `identity`, expected values) stays theirs, so a reference never overrides it in the lineage graph.

        ```python
        # analytics repo: features and churn_model are owned by the ML team's repo
        features = artifacts.Artifact.ref("features", partitions={"date": artifacts.Daily}, project="ml")
        churn_model = artifacts.Artifact.ref("churn_model", type=File, partitions={"date": artifacts.Daily})

        @env.task(
            consumes_artifacts={"week": features.window(date=artifacts.TimeRange(days=7)), "model": churn_model},
            produces_artifacts=(daily_report,),
        )
        async def report(week: list[DataFrame], model: File, date: datetime) -> DataFrame: ...
        ```

        A reference reads like any handle (`.window()`, `.all()`, `.select()`, `get_partition_value`,
        `flyte.OnArtifact(ref)`, `flyte.materialize(ref)`), but cannot be produced or published: to make the
        artifact here too, declare it with `artifacts.Artifact(...)`. `flyte deploy` checks the partitions
        against the registry and fails, with the line to paste, when the owner's declaration says otherwise.

        Args:
            name: The artifact name, as its owner declares it.
            type: The value type, when you want deploy to check the bound parameter against it.
            partitions: The owner's partition dimensions, in order. Omit for an unpartitioned artifact.
            project: The owner's project, when it is not the deploying project.
            domain: The owner's domain, when it is not the deploying domain.
        """
        return ArtifactRef(name, type=type, partitions=partitions, project=project, domain=domain)

    def __class_getitem__(cls, item: Any) -> type:
        """
        `Artifact[File]` keeps working as a type hint, as it did when `Artifact` was the protocol now named
        `ArtifactLike`. It returns the class itself; use `ArtifactLike` for protocol checks.
        """
        return cls

    def __init__(
        self,
        name: str,
        *,
        type: Any = None,
        partitions: Optional[Mapping[str, DimensionType]] = None,
        description: Optional[str] = None,
        source: bool = False,
        kind: Optional[HandleKind] = None,
        identity: Identity = "version",
        project: Optional[str] = None,
        domain: Optional[str] = None,
        refresh: Any = None,
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("An artifact handle needs a non-empty name")
        if not NODE_ID_RE.match(name) or ":" in name:
            raise ValueError(
                f"Artifact name {name!r} may only contain letters, digits, '_', '.' and '-' "
                "(':' is reserved for entity node ids such as 'app:<name>')"
            )
        if identity not in ("version", "content"):
            raise ValueError(f"identity must be 'version' or 'content', got {identity!r}")
        if kind is not None and kind not in ("model", "data", "generic"):
            raise ValueError(f"kind must be 'model', 'data' or 'generic', got {kind!r}")
        dims: Dict[str, DimensionType] = {}
        time_dims: List[str] = []
        for dim, t in (partitions or {}).items():
            if not isinstance(dim, str) or not dim.isidentifier():
                raise ValueError(f"Partition dimension {dim!r} of {name!r} must be a valid identifier")
            if isinstance(t, _Granularity):
                time_dims.append(dim)
            elif t not in (str, int):
                raise TypeError(
                    f"Partition dimension {dim!r} of {name!r} must be typed str, int, or a granularity marker "
                    f"(artifacts.Daily, Hourly, Weekly, Monthly); got {t!r}"
                )
            dims[dim] = t
        if len(time_dims) > 1:
            raise ValueError(
                f"{name!r} declares {len(time_dims)} time dimensions ({', '.join(time_dims)}); an artifact can "
                "carry at most one. Make the others str."
            )
        src, line = _caller_location()
        self.name = name
        self.type = type
        self.partitions: Dict[str, DimensionType] = dims
        self.description = description
        self.source = bool(source)
        self.kind = kind
        self.identity: Identity = identity
        self.project = project
        self.domain = domain
        self.expected: Dict[str, Tuple[str, ...]] = {}
        self._src_path = src
        self._src_file: Optional[str] = None  # relative form, computed on first access (getcwd is slow)
        self.src_line = line
        from ._refresh import as_policies

        self.refresh: Tuple[Any, ...] = as_policies(refresh)
        for policy in self.refresh:
            policy.spec(self)  # a policy that cannot work fails here, at the declaration

    # -- introspection -------------------------------------------------------------------------------

    def __repr__(self) -> str:
        dims = ", ".join(f"{d}: {dimension_type_name(t)}" for d, t in self.partitions.items())
        tn = type_name(self.type)
        return (
            f"Artifact({self.name!r}"
            + (f", type={tn}" if tn else "")
            + (f", partitions={{{dims}}}" if dims else "")
            + ")"
        )

    @property
    def dims(self) -> List[str]:
        """Partition dimension names, in declaration order."""
        return list(self.partitions)

    @property
    def time_dim(self) -> Optional[str]:
        """Name of the time dimension, or None."""
        for d, t in self.partitions.items():
            if isinstance(t, _Granularity):
                return d
        return None

    @property
    def src_file(self) -> str:
        """File that constructed the handle, relative to the working directory when under it."""
        if self._src_file is None:
            self._src_file = relative_source_path(self._src_path)
        return self._src_file

    @property
    def src_path(self) -> str:
        """Absolute path of the file that constructed this handle."""
        return self._src_path

    @property
    def is_bare(self) -> bool:
        """True for a name-only handle (no type, no partitions): it never conflicts with a typed one."""
        return self.type is None and not self.partitions

    @property
    def level(self) -> int:
        """Declaration ladder rung of the handle alone: 1 for a bare name, 2 with a type or partitions."""
        return 1 if self.is_bare else 2

    def to_dict(self, src_file: Optional[str] = None) -> Dict[str, Any]:
        """
        The handle record written into a task's `lineage.bindings` payload under `artifacts`.

        Args:
            src_file: The source file to record; defaults to `src_file` (relative to the working directory).
        """
        d: Dict[str, Any] = {
            "type": type_name(self.type),
            "kind": self.kind or "",
            "source": self.source,
            "description": self.description or "",
            "identity": self.identity,
            "dims": [dimension_record(dim, t) for dim, t in self.partitions.items()],
        }
        if self.expected:
            d["expected"] = {k: list(v) for k, v in self.expected.items()}
        if self.project:
            d["project"] = self.project
        if self.domain:
            d["domain"] = self.domain
        d["src_file"] = self.src_file if src_file is None else src_file
        d["src_line"] = self.src_line
        return d

    def _check_dim(self, dim: str, what: str) -> None:
        if dim not in self.partitions:
            declared = ", ".join(self.partitions) or "none"
            raise ValueError(f"{what}: {self.name!r} has no partition dimension {dim!r}; declared: {declared}")

    # -- producer side -------------------------------------------------------------------------------

    def at(
        self,
        *,
        version: Optional[str] = None,
        card: Any = None,
        description: Optional[str] = None,
        attrs: Optional[Mapping[str, str]] = None,
        parents: Optional[Sequence[Any]] = None,
        **partitions: Any,
    ):
        """
        The `Metadata` for one version of this artifact, to return as `artifacts.new(value, handle.at(...))`.

        Partition values are keyword arguments named by the handle's dimensions. A time dimension accepts a
        date, datetime or ISO string and is floored to the dimension's granularity.

        ```python
        return artifacts.new(weights, churn_model.at(date=date, card=card))
        ```

        Args:
            version: Explicit version; defaults to the platform's (or the content hash for
                `identity="content"`).
            card: Optional `artifacts.Card` rendered with the version.
            description: Overrides the handle's description for this version.
            attrs: Extra string attributes published with the version.
            parents: Versions this one derives from (see `artifacts.Metadata.parents`).
            partitions: Partition values, keyed by dimension name.

        Returns:
            An `artifacts.Metadata`.
        """
        from ._metadata import Metadata

        values: Dict[str, Any] = {}
        for dim, value in partitions.items():
            if self.partitions:
                self._check_dim(dim, f"{self.name}.at()")
            values[dim] = self._coerce_partition_value(dim, value)
        return Metadata(
            name=self.name,
            version=version,
            description=description if description is not None else self.description,
            card=card,
            kind=self.kind,
            attrs=dict(attrs) if attrs else None,
            partitions=values or None,
            parents=tuple(parents) if parents else None,
            version_from_content=self.identity == "content",
        )

    def _coerce_partition_value(self, dim: str, value: Any) -> Any:
        """Normalize one partition value the way the registry stores it for this handle's dimension."""
        t = self.partitions.get(dim)
        if isinstance(t, _Granularity):
            if isinstance(value, TimePartition):
                return TimePartition(value.value, t.registry)  # type: ignore[arg-type]
            return TimePartition(t.parse(value), t.registry)  # type: ignore[arg-type]
        if value is None:
            raise ValueError(f"Partition {dim!r} of {self.name!r} is None")
        if isinstance(value, (date, datetime)) and t is not None:
            # A str/int dimension never becomes a time partition, whatever the value's type.
            return value.isoformat()
        return value if t is None else str(value)

    def expect(self, **values: Any) -> "Artifact":
        """
        Record the values a dimension is expected to take, so a partition that never arrived is told apart
        from one whose producer failed (ladder level 5). Returns the handle itself.

        ```python
        events = artifacts.Artifact("events", partitions={"date": artifacts.Daily, "region": str}).expect(
            region=["us", "eu"]
        )
        ```

        Args:
            values: Dimension name to one value or a list of values.

        Returns:
            This handle.
        """
        for dim, v in values.items():
            self._check_dim(dim, f"{self.name}.expect()")
            seq = list(v) if isinstance(v, (list, tuple, set, frozenset)) else [v]
            if not seq:
                raise ValueError(f"{self.name}.expect(): {dim!r} needs at least one value")
            t = self.partitions[dim]
            # A time value is floored and formatted the way the registry stores it, not str(datetime).
            self.expected[dim] = tuple(t.format(t.parse(x)) if isinstance(t, _Granularity) else str(x) for x in seq)
        return self

    # -- consumer side -------------------------------------------------------------------------------

    @property
    def identity_mapping(self) -> "ArtifactMapping":
        """The identity mapping, what binding the handle itself to a parameter means."""
        return ArtifactMapping(handle=self, kind="identity")

    def all(self, dim: str) -> "ArtifactMapping":
        """
        Every observed partition along `dim`, for the consumer's values of the other shared dimensions. The
        bound parameter must be a `list[...]`.

        Args:
            dim: The dimension to collapse into a list.
        """
        self._check_dim(dim, f"{self.name}.all()")
        return ArtifactMapping(handle=self, kind="all", dim=dim)

    def window(self, **ranges: TimeRange) -> "ArtifactMapping":
        """
        A trailing window on the time dimension, ending at the consumer's own time value. The bound parameter
        must be a `list[...]`.

        ```python
        features.window(date=artifacts.TimeRange(days=30))
        ```

        Args:
            ranges: Exactly one `dimension=TimeRange(days=..., hours=...)`.
        """
        if len(ranges) != 1:
            raise ValueError("window() takes exactly one dimension, e.g. window(date=TimeRange(days=7))")
        ((dim, rng),) = ranges.items()
        self._check_dim(dim, f"{self.name}.window()")
        t = self.partitions[dim]
        if not isinstance(t, _Granularity):
            raise ValueError(f"{self.name}.window(): {dim!r} is not a time dimension")
        if t.step is None:
            # The planner expands a window in fixed steps; months have none.
            raise ValueError(
                f"{self.name}.window(): {dim!r} is Monthly, and a window needs a fixed step; "
                "declare it Daily, Hourly or Weekly, or read every month with .all()"
            )
        if not isinstance(rng, TimeRange):
            raise TypeError("window() values must be artifacts.TimeRange, e.g. TimeRange(days=7)")
        if rng.is_absolute:
            raise ValueError("window() takes a trailing TimeRange(days=..., hours=...), not an absolute range")
        if rng.delta <= timedelta(0):
            raise ValueError("window() needs a non-empty TimeRange")
        return ArtifactMapping(handle=self, kind="window", dim=dim, window=rng)

    def select(self, **pins: Any) -> "ArtifactMapping":
        """
        Pin one or more dimensions to fixed values; the rest map by identity.

        Args:
            pins: Dimension name to the pinned value.
        """
        if not pins:
            raise ValueError("select() needs at least one dimension=value")
        pinned: Dict[str, str] = {}
        for dim, v in pins.items():
            self._check_dim(dim, f"{self.name}.select()")
            t = self.partitions[dim]
            # A time value (date, datetime or ISO string) is floored and formatted like every other time value.
            pinned[dim] = t.format(t.parse(v)) if isinstance(t, _Granularity) else str(v)
        return ArtifactMapping(handle=self, kind="select", pinned=pinned)

    def materialize_on(
        self,
        event: Any,
        *,
        lag: Optional[TimeRange] = None,
        name: Optional[str] = None,
        image: Any = None,
        **filter: str,
    ) -> Any:
        """
        Keep this artifact fresh from code that does not own it: materialize it on a schedule, or on each new
        version of a source artifact. Works on an `Artifact.ref`, so a downstream team can keep fresh what it
        reads. The owner declares the same thing with `Artifact(..., refresh=...)`.

        ```python
        features = artifacts.Artifact.ref("features", partitions={"date": artifacts.Daily})
        keep_features_fresh = features.materialize_on(flyte.Cron("0 * * * *"), lag=artifacts.TimeRange(hours=1))
        ```

        Returns a generated `flyte.TaskEnvironment` (`refresh-<name>`): assign it to a module-level name and deploy
        it like any environment (`flyte deploy file.py keep_features_fresh`). Its one task calls
        `flyte.materialize`; you never define or import it.

        Args:
            event: `flyte.Cron(...)`, `flyte.FixedRate(...)` or a source `artifacts.Artifact` handle.
            lag: For a schedule, how far behind the trigger time the materialized partition is.
            name: Name of the generated task and its trigger (default `<name>_on_schedule` or `<name>_on_<source>`).
            image: Image of the generated environment. Default: the factory image when flyteplugins-union is
                installed (it honors `FLYTE_FACTORY_IMAGE` / `FLYTE_FACTORY_WHEEL`), else the default base plus
                flyteplugins-union.
            filter: For a source event, string partition values the new source version must carry.
        """
        from ._refresh import Refresh, refresh_env

        return refresh_env(self, [Refresh(event, lag=lag, name=name, **filter)], image=image)

    def get_partition_value(self, dim: str) -> "PartitionValue":
        """
        Bind a parameter to the *value* of one partition dimension, such as the date being built.

        The parameter receives the coordinate (for example `datetime(2026, 9, 8)` or `"us"`), not the
        partition's data; bind the handle itself, or a mapping such as `.all()` / `.window()`, to receive data.

        Reads against a handle that resolves to a single partition: one the task produces, or an input
        mapped by identity (or `select`). The parameter name is free: bind `as_of` to
        `raw_events.get_partition_value("date")` and it still carries the date.

        ```python
        @env.task(
            consumes_artifacts={"raw": raw_events, "date": events.get_partition_value("date")},
            produces_artifacts=(events,),
        )
        async def clean(raw: File, date: datetime) -> DataFrame: ...
        ```

        Args:
            dim: The dimension whose value the parameter carries.
        """
        self._check_dim(dim, f"{self.name}.get_partition_value()")
        return PartitionValue(handle=self, dim=dim)


class ArtifactRef(Artifact):
    """
    A handle to an artifact another codebase owns. Construct it with `Artifact.ref(...)`.

    It is an `Artifact`, so every API that reads a handle takes it. It carries only what a reader states (name,
    type, partitions, scope): the owner's description, kind, `source`, `identity` and expected values are not
    restated, and the producer-side calls (`at`, `expect`, a `produces_artifacts` slot) refuse it.
    """

    reference = True

    def __init__(
        self,
        name: str,
        *,
        type: Any = None,
        partitions: Optional[Mapping[str, DimensionType]] = None,
        project: Optional[str] = None,
        domain: Optional[str] = None,
    ):
        super().__init__(name, type=type, partitions=partitions, project=project, domain=domain)

    def __repr__(self) -> str:
        return "Artifact.ref" + super().__repr__()[len("Artifact") :]

    def _owned_elsewhere(self, what: str) -> ValueError:
        return ValueError(
            f"{self.name}.{what}: {self.name!r} is a reference (Artifact.ref) to an artifact owned elsewhere, so "
            f"it can only be read. To produce {self.name!r} here as well, declare it with artifacts.Artifact(...)."
        )

    def at(self, **kwargs: Any):  # type: ignore[override]
        raise self._owned_elsewhere("at()")

    def expect(self, **values: Any) -> "Artifact":
        raise self._owned_elsewhere("expect()")

    def to_dict(self, src_file: Optional[str] = None) -> Dict[str, Any]:
        d = super().to_dict(src_file)
        # Only what the reader stated; the platform ranks this below the owner's own declaration.
        for owned in ("kind", "description", "source", "identity"):
            d.pop(owned, None)
        d["reference"] = True
        return d


@dataclass(frozen=True)
class ArtifactMapping:
    """
    How a consumer's parameter maps onto an upstream artifact's partitions.

    Attributes:
        handle: The upstream `Artifact` handle.
        kind: `"identity"`, `"all"`, `"window"` or `"select"`.
        dim: The dimension `all` collapses or `window` spans; None otherwise.
        window: The trailing `TimeRange` of a `window` mapping; None otherwise. Also available as `range`.
        pinned: Dimension to pinned string value, for `select`.
    """

    handle: Artifact
    kind: MappingKind = "identity"
    dim: Optional[str] = None
    window: Optional[TimeRange] = None
    pinned: Mapping[str, str] = field(default_factory=dict, hash=False)

    @property
    def range(self) -> Optional[TimeRange]:
        """Alias of `window`."""
        return self.window

    @property
    def name(self) -> str:
        """The upstream artifact name."""
        return self.handle.name

    @property
    def is_list(self) -> bool:
        """True when the bound parameter receives many partitions (`all`, `window`)."""
        return self.kind in ("all", "window")

    def describe(self) -> str:
        if self.kind == "identity":
            return "identity"
        if self.kind == "all":
            return f"all({self.dim!r})"
        if self.kind == "window":
            return f"window({self.dim}={self.window!r})"
        return "select(" + ", ".join(f"{k}={v!r}" for k, v in self.pinned.items()) + ")"

    def to_dict(self) -> Dict[str, Any]:
        """The mapping object written into `lineage.bindings` (`parameters.<name>.mapping`)."""
        if self.kind == "all":
            return {"kind": "all", "dim": self.dim}
        if self.kind == "window":
            assert self.window is not None
            return {"kind": "window", "dim": self.dim, "days": self.window.days, "hours": self.window.hours}
        if self.kind == "select":
            return {"kind": "select", "values": dict(self.pinned)}
        return {"kind": "identity"}


@dataclass(frozen=True)
class PartitionValue:
    """
    The value of one partition dimension, bound to a parameter (`handle.get_partition_value(dim)`).

    The parameter receives the coordinate of the partition being built (its date, region, ...), not the
    partition's data.

    Attributes:
        handle: The `Artifact` handle whose dimension is read.
        dim: The dimension name.
    """

    handle: Artifact
    dim: str

    @property
    def name(self) -> str:
        """The artifact name the dimension belongs to."""
        return self.handle.name

    @property
    def dimension_type(self) -> DimensionType:
        """The dimension's declared type (`str`, `int` or a granularity marker)."""
        return self.handle.partitions[self.dim]


@dataclass(frozen=True)
class OutputPartition:
    """
    The instance's value of one dimension of the step's own outputs (`artifacts.partition(dim)`).

    The same binding as `handle.get_partition_value(dim)` on the produced handle that carries `dim`, without
    naming the handle; the factory's `fc.partition(dim)`. For a task that produces nothing (a sink), the
    dimension is read from an input mapped by identity or `select`.

    Attributes:
        dim: The dimension name.
    """

    dim: str

    def to_dict(self) -> Dict[str, Any]:
        """The factory spec form, `{"partition": dim}`."""
        return {"partition": self.dim}


@dataclass(frozen=True)
class RequiredParam:
    """
    A parameter every materialization must supply (`artifacts.required()`); the factory's `fc.required()`.

    Deploy records it as `{"kind": "required"}`, and the task stays plannable: the planner asks the caller
    for the value instead of treating the parameter as uncovered.
    """

    def to_dict(self) -> Dict[str, Any]:
        """The factory spec form, `{"required": true}`."""
        return {"required": True}


def partition(dim: str) -> OutputPartition:
    """
    Bind a parameter to the instance's value of `dim`, a dimension of what the task produces.

    Equivalent to `handle.get_partition_value(dim)` on the produced handle that carries `dim`, and to the
    factory's `fc.partition(dim)`. Deploy checks that some produced handle has `dim` (for a task that produces
    nothing, some input mapped by identity or `select`); at run time the parameter's value becomes that
    dimension of every produced handle that has it.

    ```python
    @env.task(consumes_artifacts={"day": artifacts.partition("date")}, produces_artifacts=(daily_report,))
    async def report(day: datetime) -> File: ...
    ```

    A parameter *named* like the dimension (`date: datetime`) with no default needs no binding at all.

    Args:
        dim: The dimension whose value the parameter carries.
    """
    if not isinstance(dim, str) or not dim:
        raise ValueError("artifacts.partition() needs a dimension name")
    return OutputPartition(dim=dim)


def required() -> RequiredParam:
    """
    Mark a parameter that every materialization must supply; the factory's `fc.required()`.

    Without it, a parameter with no default and no binding makes the task unplannable (deploy warns). With it,
    deploy records the parameter as required and the task stays plannable; `flyte.materialize(...,
    inputs={"<task>.<param>": value})` must then pass it. A direct call passes it like any argument.

    ```python
    @env.task(consumes_artifacts={"seed": artifacts.required()}, produces_artifacts=(model,))
    async def train(date: datetime, seed: int) -> File: ...
    ```
    """
    return RequiredParam()


def dimension_type_name(t: DimensionType) -> str:
    """`Daily`, `Hourly`, ... for a granularity marker; `str` / `int` otherwise."""
    if isinstance(t, _Granularity):
        return t.name
    return getattr(t, "__name__", str(t))


def dimension_record(name: str, t: DimensionType) -> Dict[str, str]:
    """`{"name", "kind"[, "granularity"]}`, the shape of a dimension in `lineage.bindings`."""
    rec = {"name": name, "kind": dimension_kind(t)}
    if isinstance(t, _Granularity):
        rec["granularity"] = t.registry
    return rec


#: Anything a `consumes_artifacts` value may be.
Binding = Union[Artifact, ArtifactMapping, PartitionValue, OutputPartition, RequiredParam]
