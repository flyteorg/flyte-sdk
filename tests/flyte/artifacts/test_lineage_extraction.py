"""Deploy-time extraction and validation of artifact declarations, and the tags they compile to."""

import json
import os
import pathlib
import re
from datetime import datetime
from typing import Optional, Tuple

import pytest
from flyteidl2.core import tasks_pb2

import flyte
import flyte.artifacts as artifacts
from flyte.artifacts._lineage import (
    BINDINGS_LABEL,
    CONSUMES_LABEL,
    PRODUCES_LABEL,
    check_conflicts,
    extract_task_lineage,
    merge_labels,
    summarize,
    validate_labels,
)
from flyte.errors import DeploymentError, LineageDeclarationError
from flyte.io import DataFrame, File
from flyte.models import SerializationContext

from . import proposal_tasks as P

ROOT = os.path.dirname(os.path.abspath(P.__file__))
SRC = "proposal_tasks.py"

DAY = {"name": "date", "kind": "time", "granularity": "day"}
REGION = {"name": "region", "kind": "str"}


def _rec(h, **over):
    base = {
        "type": artifacts._handle.type_name(h.type),
        "kind": h.kind or "",
        "source": h.source,
        "description": h.description or "",
        "identity": "version",
        "dims": [DAY, REGION] if "region" in h.partitions else [DAY],
        "src_file": SRC,
        "src_line": h.src_line,
    }
    base.update(over)
    return base


def _bindings(task):
    lin = extract_task_lineage(task, root_dir=ROOT)
    return lin, json.loads(lin.labels[BINDINGS_LABEL])


def _line(task):
    return task.func.__code__.co_firstlineno


_SOURCE = pathlib.Path(P.__file__).read_text().splitlines()


def _param_line(task, param):
    """Line of `param` in `task`'s signature, found by scanning the source (independent of the SDK)."""
    def_line = next(i for i, ln in enumerate(_SOURCE, 1) if ln.startswith(f"async def {task.func.__name__}("))
    for i in range(def_line, def_line + 10):
        if re.search(rf"\b{param}\s*:", _SOURCE[i - 1]):
            return i
    raise AssertionError(param)


def _with_src(task, params):
    return {k: {**v, "src_file": SRC, "src_line": _param_line(task, k)} for k, v in params.items()}


# ------------------------------------------------------------------ full bindings JSON, proposal tasks


def test_clean_bindings():
    lin, b = _bindings(P.clean)
    assert b == {
        "version": 1,
        "task": "ingest.clean",
        "src_file": SRC,
        "src_line": _line(P.clean),
        "level": 4,
        "produces": [{"node": "events", "position": 0}],
        "outputs": 1,
        "artifacts": {"events": _rec(P.events), "raw_events": _rec(P.raw_events)},
        "parameters": _with_src(
            P.clean,
            {
                "raw": {"kind": "artifact", "node": "raw_events", "type": "File", "mapping": {"kind": "identity"}},
                "date": {"kind": "partition", "node": "raw_events", "dim": "date", "type": "datetime"},
                "region": {"kind": "partition", "node": "raw_events", "dim": "region", "type": "str"},
                "min_quality": {"kind": "default", "type": "int", "default": 30},
            },
        ),
        "pullable": True,
        "unpullable_reason": "",
    }
    assert lin.labels[PRODUCES_LABEL] == "events"
    assert lin.labels[CONSUMES_LABEL] == "raw_events"
    assert _rec(P.raw_events)["source"] is True


def test_featurize_bindings():
    lin, b = _bindings(P.featurize)
    assert b == {
        "version": 1,
        "task": "ml.featurize",
        "src_file": SRC,
        "src_line": _line(P.featurize),
        "level": 4,
        "produces": [{"node": "features", "position": 0}],
        "outputs": 1,
        "artifacts": {"features": _rec(P.features), "events": _rec(P.events)},
        "parameters": _with_src(
            P.featurize,
            {
                "per_region": {
                    "kind": "artifact",
                    "node": "events",
                    "type": "list[DataFrame]",
                    "mapping": {"kind": "all", "dim": "region"},
                },
                "date": {"kind": "partition", "node": "features", "dim": "date", "type": "datetime"},
            },
        ),
        "pullable": True,
        "unpullable_reason": "",
    }
    # The partition binding on the produced handle does not make `features` a consumed node.
    assert lin.consumes == ["events"] and lin.produces == ["features"]


TRAIN_BINDINGS = {
    "version": 1,
    "task": "ml.train",
    "src_file": SRC,
    "level": 4,
    "produces": [{"node": "churn_model", "position": 0}],
    "outputs": 1,
    "parameters": {
        "history": {
            "kind": "artifact",
            "node": "features",
            "type": "list[DataFrame]",
            "mapping": {"kind": "window", "dim": "date", "days": 30, "hours": 0},
        },
        "date": {"kind": "partition", "node": "churn_model", "dim": "date", "type": "datetime"},
        "lr": {"kind": "default", "type": "float", "default": 0.0003},
    },
    "pullable": True,
    "unpullable_reason": "",
}


def test_train_bindings():
    lin, b = _bindings(P.train)
    assert b == {
        **TRAIN_BINDINGS,
        "parameters": _with_src(P.train, TRAIN_BINDINGS["parameters"]),
        "src_line": _line(P.train),
        "artifacts": {"churn_model": _rec(P.churn_model), "features": _rec(P.features)},
    }
    assert {k: v for k, v in lin.labels.items() if k != BINDINGS_LABEL} == {
        PRODUCES_LABEL: "churn_model",
        CONSUMES_LABEL: "features",
    }


def test_report_bindings():
    lin, b = _bindings(P.report)
    assert b == {
        "version": 1,
        "task": "analytics.report",
        "src_file": SRC,
        "src_line": _line(P.report),
        "level": 4,
        "produces": [{"node": "daily_report", "position": 0}],
        "outputs": 1,
        "artifacts": {
            "daily_report": _rec(P.daily_report),
            "features": _rec(P.features),
            "churn_model": _rec(P.churn_model),
        },
        "parameters": _with_src(
            P.report,
            {
                "week": {
                    "kind": "artifact",
                    "node": "features",
                    "type": "list[DataFrame]",
                    "mapping": {"kind": "window", "dim": "date", "days": 7, "hours": 0},
                },
                "model": {"kind": "artifact", "node": "churn_model", "type": "File", "mapping": {"kind": "identity"}},
                "date": {"kind": "partition", "node": "daily_report", "dim": "date", "type": "datetime"},
            },
        ),
        "pullable": True,
        "unpullable_reason": "",
    }
    assert lin.consumes == ["features", "churn_model"]
    assert lin.edges == [("features", "daily_report"), ("churn_model", "daily_report")]


def test_clean_adapter_bindings():
    _, b = _bindings(P.clean_adapter)
    assert b["task"] == "ingest.clean_adapter"
    assert b["parameters"] == _with_src(
        P.clean_adapter,
        {
            "raw": {"kind": "artifact", "node": "raw_events", "type": "File", "mapping": {"kind": "identity"}},
            "date": {"kind": "partition", "node": "raw_events", "dim": "date", "type": "datetime"},
            "region": {"kind": "partition", "node": "raw_events", "dim": "region", "type": "str"},
        },
    )
    assert b["outputs"] == 1
    assert b["produces"] == [{"node": "events", "position": 0}]
    assert b["artifacts"] == {"events": _rec(P.events), "raw_events": _rec(P.raw_events)}
    assert (b["pullable"], b["level"]) == (True, 4)


def test_proposal_deploy_summary_line():
    s = summarize(P.ALL, root_dir=ROOT)
    assert s.line() == "✓ 4 tasks, 5 artifact handles, 5 dependency edges resolved"
    assert s.render() == s.line()
    assert set(s.edge_set) == {
        ("raw_events", "events"),
        ("events", "features"),
        ("features", "churn_model"),
        ("features", "daily_report"),
        ("churn_model", "daily_report"),
    }
    assert summarize([P.train]).line() == "✓ 1 task, 2 artifact handles, 1 dependency edge resolved"


# ------------------------------------------------------------------ tags on the serialized TaskTemplate


def _proto(task, labels=None):
    from flyte._internal.runtime.task_serde import get_proto_task

    sc = SerializationContext(
        project="p",
        domain="d",
        version="v",
        org="o",
        root_dir=pathlib.Path(ROOT),
        labels=labels,
        emit_lineage_tags=True,
    )
    return get_proto_task(task, sc)


def test_tags_on_serialized_task_template():
    tt = _proto(P.train, labels={"owner": "ml-platform"})
    assert isinstance(tt, tasks_pb2.TaskTemplate)
    tags = dict(tt.metadata.tags)
    assert tags[PRODUCES_LABEL] == "churn_model"
    assert tags[CONSUMES_LABEL] == "features"
    assert tags["owner"] == "ml-platform"
    assert json.loads(tags[BINDINGS_LABEL])["parameters"] == _with_src(P.train, TRAIN_BINDINGS["parameters"])
    assert tt.metadata.produces_artifacts is True


def test_untyped_task_has_no_lineage_tags():
    env = flyte.TaskEnvironment(name="plain")

    @env.task
    async def t(x: int) -> int:
        return x

    tt = _proto(t)
    assert dict(tt.metadata.tags) == {}
    assert tt.metadata.produces_artifacts is False

    @env.task(produces_artifacts=True)
    async def t2(x: int) -> File:
        raise NotImplementedError

    tt2 = _proto(t2)
    assert dict(tt2.metadata.tags) == {} and tt2.metadata.produces_artifacts is True


def test_serialization_memoizes_per_task_off_the_task_object():
    from flyte.artifacts import _lineage

    _lineage._LINEAGE_CACHE.pop(id(P.train), None)
    _proto(P.train)
    first = _lineage._LINEAGE_CACHE[id(P.train)]
    _proto(P.train)
    assert _lineage._LINEAGE_CACHE[id(P.train)] is first
    _proto(P.train, labels={"x": "y"})
    assert _lineage._LINEAGE_CACHE[id(P.train)] is not first
    # Nothing is stored on the task, so nothing leaks into the cloudpickle'd deployment / version hash.
    assert not [k for k in P.train.__dict__ if "lineage" in k]
    import cloudpickle

    assert b"_lineage_tags_cache" not in cloudpickle.dumps(P.train)


def test_cache_entry_dropped_with_task():
    import gc

    from flyte.artifacts import _lineage

    env = flyte.TaskEnvironment(name="cache_gc")

    @env.task(labels={"a": "b"})
    async def t(x: int) -> int:
        return x

    _lineage.task_lineage_tags(t)
    tid = id(t)
    assert tid in _lineage._LINEAGE_CACHE
    env._tasks.clear()
    del t
    gc.collect()
    assert tid not in _lineage._LINEAGE_CACHE


# ------------------------------------------------------------------ labels


def test_env_and_task_labels_merge():
    env = flyte.TaskEnvironment(name="lbl", labels={"team": "ml", "lineage.consumes": "a"})

    @env.task(labels={"tier": "gold", "lineage.consumes": "b,a"})
    async def t(x: int) -> int:
        return x

    assert t.labels == {"tier": "gold", "lineage.consumes": "b,a"}  # env labels merge at serialization
    assert extract_task_lineage(t).labels == {"team": "ml", "tier": "gold", CONSUMES_LABEL: "a,b"}
    lin = extract_task_lineage(t, extra_labels={"team": "platform", "lineage.consumes": "c"})
    assert lin.labels == {"team": "platform", "tier": "gold", CONSUMES_LABEL: "a,b,c"}
    assert lin.bindings is None and lin.level == 0 and lin.edges == []


def test_env_labels_read_at_serialization_time():
    from flyte.artifacts._lineage import task_lineage_tags

    env = flyte.TaskEnvironment(name="late_labels")

    @env.task
    async def t(x: int) -> int:
        return x

    assert task_lineage_tags(t) == {}
    env.labels = {"team": "ml"}  # changed after decoration
    assert task_lineage_tags(t) == {"team": "ml"}
    # override(labels=) merges over the environment's labels instead of replacing them.
    o = t.override(labels={"tier": "gold", "team": "data"})
    assert task_lineage_tags(o) == {"team": "data", "tier": "gold"}


def test_from_task_and_clone_with_labels():
    from flyte.artifacts._lineage import task_lineage_tags

    loose = flyte.TaskEnvironment(name="loose_src")

    @loose.task
    async def t(x: int) -> int:
        return x

    loose._tasks.clear()
    t.parent_env = None
    env = flyte.TaskEnvironment.from_task("from_task_env", t)
    env.labels = {"team": "ml"}
    assert task_lineage_tags(t) == {"team": "ml"}
    assert env.clone_with("cloned_env", labels={"team": "data"}).labels == {"team": "data"}
    assert env.clone_with("cloned_env2").labels == {"team": "ml"}


def test_hand_written_produces_on_task():
    env = flyte.TaskEnvironment(name="lbl2")

    @env.task(labels={"lineage.produces": "dashboard_table", "lineage.consumes": "events"})
    async def t(x: int) -> int:
        return x

    lin = extract_task_lineage(t)
    assert lin.edges == [("events", "dashboard_table")]
    assert lin.resolvable_edges == []  # label-only: no bindings


def test_merge_labels_unions_lineage_keys():
    assert merge_labels({"lineage.produces": "a"}, None, {"lineage.produces": "b, a"}) == {"lineage.produces": "a,b"}


@pytest.mark.parametrize(
    "labels,entity,match",
    [
        (
            {"lineage.bindings": "{}"},
            "task",
            re.escape(
                "t: label 'lineage.bindings' is in the reserved 'lineage.' namespace; only 'lineage.consumes' and "
                "'lineage.produces' may be written by hand."
            ),
        ),
        (
            {"lineage.edges": "x"},
            "app",
            re.escape(
                "t: label 'lineage.edges' is in the reserved 'lineage.' namespace; only 'lineage.consumes' may be "
                "written by hand."
            ),
        ),
        (
            {"lineage.produces": "x"},
            "app",
            r"t: an app may not set 'lineage.produces'; deploy derives it as 'app:<app name>' \(the app's endpoint\).",
        ),
        ({"lineage.consumes": "a b"}, "task", r"contains an invalid node id 'a b'"),
        ({"team": 3}, "task", r"labels must be str to str"),
    ],
)
def test_reserved_namespace_rejected(labels, entity, match):
    with pytest.raises(LineageDeclarationError, match=match):
        validate_labels(labels, entity=entity, where="t")


@pytest.mark.parametrize(
    "labels,entity",
    [
        ({"lineage.consumes": "churn_model,app:x,task:a.b,trigger:n"}, "task"),
        ({"lineage.produces": "x"}, "task"),
        ({"lineage.consumes": "x", "team": "ml", "lineage_ish": "ok"}, "app"),
    ],
)
def test_reserved_namespace_allowed(labels, entity):
    validate_labels(labels, entity=entity, where="t")


def test_lineage_error_is_a_deployment_error():
    err = LineageDeclarationError("boom")
    assert isinstance(err, DeploymentError) and err.code == "LineageDeclarationError"


# ------------------------------------------------------------------ validation rules


h_ev = artifacts.Artifact("ev", type=DataFrame, partitions={"date": artifacts.Daily, "region": str})
h_out = artifacts.Artifact("out", type=File, partitions={"date": artifacts.Daily})
h_day = artifacts.Artifact("evd", type=DataFrame, partitions={"date": artifacts.Daily})
env_v = flyte.TaskEnvironment(name="v")


def _task(fn, produces=(h_out,), consumes=None):
    return env_v.task(produces_artifacts=produces, consumes_artifacts=consumes)(fn)


async def _one(x: DataFrame, date: datetime) -> File:
    raise NotImplementedError


async def _many(xs: list[DataFrame], date: datetime) -> File:
    raise NotImplementedError


async def _two(x: DataFrame, date: datetime) -> Tuple[File, File]:
    raise NotImplementedError


async def _nothing(x: DataFrame, date: datetime) -> None:
    raise NotImplementedError


VALID = [
    ("identity", _one, {"x": h_ev.select(region="us"), "date": h_out.get_partition_value("date")}),
    ("handle", _one, {"x": h_day, "date": h_day.get_partition_value("date")}),
    ("all", _many, {"xs": h_ev.all("region"), "date": h_out.get_partition_value("date")}),
    ("window", _many, {"xs": h_ev.window(date=artifacts.TimeRange(days=3)), "date": h_out.get_partition_value("date")}),
    ("select-partition", _one, {"x": h_ev.select(region="us"), "date": h_ev.get_partition_value("date")}),
]


@pytest.mark.parametrize("label,fn,consumes", VALID, ids=[v[0] for v in VALID])
def test_valid_declarations(label, fn, consumes):
    lin = extract_task_lineage(_task(fn, consumes=consumes))
    assert lin.pullable and lin.produces == ["out"] and lin.consumes in (["ev"], ["evd"])


INVALID = [
    (
        "unknown-key",
        _one,
        (h_out,),
        {"y": h_ev, "date": h_out.get_partition_value("date")},
        "v._one: consumes_artifacts key 'y' names no parameter of the task (parameters: x, date).",
    ),
    (
        "all-not-list",
        _one,
        (h_out,),
        {"x": h_ev.all("region")},
        "v._one: consumes_artifacts['x'] is ev.all('region'), which yields many partitions, so parameter 'x' must be "
        "typed list[...]; it is DataFrame.",
    ),
    (
        "window-not-list",
        _one,
        (h_out,),
        {"x": h_ev.window(date=artifacts.TimeRange(days=2))},
        "v._one: consumes_artifacts['x'] is ev.window(date=TimeRange(days=2)), which yields many partitions, so "
        "parameter 'x' must be typed list[...]; it is DataFrame.",
    ),
    (
        "identity-on-list",
        _many,
        (h_out,),
        {"xs": h_ev},
        "v._many: consumes_artifacts['xs'] maps ev by identity, which yields one partition, but parameter 'xs' is "
        "typed list[DataFrame]. Use ev.all(...) or ev.window(...) for a list.",
    ),
    (
        "select-on-list",
        _many,
        (h_out,),
        {"xs": h_ev.select(region="us")},
        "v._many: consumes_artifacts['xs'] maps ev by select, which yields one partition, but parameter 'xs' is "
        "typed list[DataFrame]. Use ev.all(...) or ev.window(...) for a list.",
    ),
    (
        "get-partition-windowed",
        _many,
        (h_out,),
        {"xs": h_ev.window(date=artifacts.TimeRange(days=2)), "date": h_ev.get_partition_value("date")},
        "v._many: consumes_artifacts['date'] is ev.get_partition_value('date'), but ev does not resolve to a single "
        "partition in this declaration. get_partition_value reads against an artifact the task produces or an input "
        "mapped by identity (or select), not a windowed or fanned-in input.",
    ),
    (
        "get-partition-unrelated",
        _one,
        (h_out,),
        {"date": h_ev.get_partition_value("date")},
        "v._one: consumes_artifacts['date'] is ev.get_partition_value('date'), "
        "but ev does not resolve to a single partition",
    ),
    (
        "produces-arity",
        _two,
        (h_out,),
        {"date": h_out.get_partition_value("date")},
        "v._two: produces_artifacts declares 1 position(s) (out) but the task returns 2 value(s); a tuple return is "
        "matched to the handles by position. Use None for an output that is not an artifact, e.g. "
        "produces_artifacts=(None, model).",
    ),
    (
        "produces-none-returned",
        _nothing,
        (h_out,),
        {"date": h_out.get_partition_value("date")},
        "v._nothing: produces_artifacts declares 1 position(s) (out) but the task returns 0 value(s)",
    ),
    (
        "produces-duplicate",
        _two,
        (h_out, h_out),
        {"date": h_out.get_partition_value("date")},
        "v._two: produces_artifacts names artifact 'out' twice; each output position needs its own artifact",
    ),
    (
        "bad-binding-type",
        _one,
        (h_out,),
        {"x": "ev"},
        "v._one: consumes_artifacts['x'] must be an artifacts.Artifact handle, a mapping (handle.all/window/select), "
        "handle.get_partition_value(dim), artifacts.partition(dim) or artifacts.required(); got str.",
    ),
]


@pytest.mark.parametrize("label,fn,produces,consumes,message", INVALID, ids=[v[0] for v in INVALID])
def test_invalid_declarations(label, fn, produces, consumes, message):
    t = _task(fn, produces=produces, consumes=consumes)
    with pytest.raises(LineageDeclarationError) as exc:
        extract_task_lineage(t)
    assert message in str(exc.value)


def test_get_partition_value_dim_missing_from_handle_after_mutation():
    b = h_ev.get_partition_value("region")
    other = artifacts.Artifact("ev", type=DataFrame, partitions={"date": artifacts.Daily})
    from flyte.artifacts._handle import PartitionValue

    t = _task(_one, consumes={"x": other, "date": PartitionValue(handle=other, dim="region")})
    with pytest.raises(
        LineageDeclarationError, match=r"reads dimension 'region' of ev, which has none \(declared: date\)"
    ):
        extract_task_lineage(t)
    assert b.dim == "region"


@pytest.mark.parametrize(
    "value,err",
    [
        ("yes", "must be True/False or a tuple of flyte.artifacts.Artifact handles, got str"),
        (("x",), "position 0 is a str"),
    ],
)
def test_produces_artifacts_type_checked_at_decoration(value, err):
    with pytest.raises(TypeError, match=err):
        env_v.task(produces_artifacts=value)(_one)


def test_produces_normalization():
    assert env_v.task(produces_artifacts=[h_out])(_one).produces_artifacts == (h_out,)
    assert env_v.task(produces_artifacts=())(_one).produces_artifacts is False
    assert env_v.task(produces_artifacts=h_out)(_one).produces_artifacts == (h_out,)


def test_consumes_and_labels_type_checked():
    with pytest.raises(TypeError, match=re.escape("consumes_artifacts of v._one must be a dict")):
        env_v.task(consumes_artifacts=[h_ev])(_one)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=re.escape("labels of v._one must be a Dict")):
        env_v.task(labels={"a": 1})(_one)  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="Expected labels to be of type Dict"):
        flyte.TaskEnvironment(name="badlabels", labels={"a": 1})  # type: ignore[dict-item]


def test_report_parameter_lines_and_outputs():
    _, b = _bindings(P.report)
    lines = [b["parameters"][p]["src_line"] for p in ("week", "model", "date")]
    assert lines[1] == lines[0] + 1 and lines[2] == lines[0] + 2  # one parameter per line in the signature
    assert all(b["parameters"][p]["src_file"] == SRC for p in ("week", "model", "date"))


@pytest.mark.parametrize("fn,arity", [("_one", 1), ("_two", 2), ("_nothing", 0)])
def test_outputs_arity(fn, arity):
    f = globals()[fn]
    produces = (h_out, artifacts.Artifact("second", type=File))[:arity] or False
    t = env_v.task(
        produces_artifacts=produces, consumes_artifacts={"x": h_day, "date": h_day.get_partition_value("date")}
    )(f)
    assert extract_task_lineage(t).bindings["outputs"] == arity


def test_tuple_return_by_position():
    a = artifacts.Artifact("a", type=File, partitions={"date": artifacts.Daily})
    b = artifacts.Artifact("b", type=File)
    t = _task(_two, produces=(a, b), consumes={"x": h_day, "date": a.get_partition_value("date")})
    (lin,) = [extract_task_lineage(t)]
    assert lin.bindings["produces"] == [{"node": "a", "position": 0}, {"node": "b", "position": 1}]
    assert lin.labels[PRODUCES_LABEL] == "a,b"


# ------------------------------------------------------------------ pullability


def test_not_pullable_warning_text():
    events_snapshot = artifacts.Artifact("events_snapshot", type=DataFrame, partitions={"date": artifacts.Daily})
    ingest = flyte.TaskEnvironment(name="ingest_snap")

    @ingest.task(
        produces_artifacts=(events_snapshot,),
        consumes_artifacts={"x": h_ev.select(region="us"), "as_of": events_snapshot.get_partition_value("date")},
    )
    async def snapshot(x: DataFrame, as_of: datetime, day: str) -> DataFrame:
        raise NotImplementedError

    lin = extract_task_lineage(snapshot)
    expected = (
        "ingest_snap.snapshot is not pullable: parameter 'day' has no default, no binding, and is not named like a "
        "partition dimension of events_snapshot. Give it a default, bind it (artifacts.partition(dim) for a partition "
        "value), or mark it artifacts.required() so every materialization must supply it."
    )
    assert lin.pullable is False
    assert lin.warnings() == [expected]
    assert lin.unpullable_reason == expected
    day = lin.bindings["parameters"]["day"]
    assert (day["kind"], day["type"]) == ("unbound", "str")
    assert day["src_line"] == snapshot.func.__code__.co_firstlineno + 4  # the def line, below the decorator
    assert lin.bindings["pullable"] is False
    s = summarize([snapshot])
    assert s.render() == "\n".join(
        [
            "✓ 1 task, 2 artifact handles, 1 dependency edge resolved",
            f"! {expected}",
            "  Deployed anyway. The task still runs when called directly; it cannot be a materialize target.",
        ]
    )


def test_consumer_only_task_is_a_sink():
    # `date` is implicitly bound to the identity input's date: every parameter is covered, so the sink is plannable.
    t = env_v.task(consumes_artifacts={"x": h_ev})(_nothing)
    lin = extract_task_lineage(t)
    assert lin.pullable is True and lin.bindings["pullable"] is True
    assert lin.unpullable_reason == ""
    assert lin.bindings["parameters"]["date"]["implicit"] is True
    assert lin.warnings() == []
    assert lin.produces == [] and lin.consumes == ["ev"] and lin.labels.get(PRODUCES_LABEL) is None


# ------------------------------------------------------------------ ladder


def test_ladder_levels():
    env = flyte.TaskEnvironment(name="ladder")
    x = artifacts.Artifact("x")
    typed = artifacts.Artifact("x2", type=File, partitions={"date": artifacts.Daily})
    expected = artifacts.Artifact("x3", type=File, partitions={"date": artifacts.Daily, "r": str}).expect(r=["a"])

    async def f(date: datetime) -> File:
        raise NotImplementedError

    async def g(inp: File, date: datetime) -> File:
        raise NotImplementedError

    async def g4(inp: File, date: datetime) -> File:
        raise NotImplementedError

    async def k(date: datetime) -> File:
        raise NotImplementedError

    level0 = env.task(produces_artifacts=True)(f)
    level3 = env.task(produces_artifacts=(typed,), consumes_artifacts={"date": typed.get_partition_value("date")})(g)
    level4 = env.task(
        produces_artifacts=(typed,), consumes_artifacts={"inp": x, "date": typed.get_partition_value("date")}
    )(g4)
    level5 = env.task(
        produces_artifacts=(expected,), consumes_artifacts={"date": expected.get_partition_value("date")}
    )(k)
    assert x.level == 1 and typed.level == 2
    assert extract_task_lineage(level0).level == 0
    assert extract_task_lineage(level3).level == 3
    assert extract_task_lineage(level4).level == 4
    assert extract_task_lineage(level5).level == 5
    assert json.loads(extract_task_lineage(level5).labels[BINDINGS_LABEL])["artifacts"]["x3"]["expected"] == {
        "r": ["a"]
    }
    # A bare handle never gets a record: the node exists, but is not "declared".
    assert "x" not in extract_task_lineage(level4).bindings["artifacts"]


# ------------------------------------------------------------------ conflicts


def test_conflicting_declarations_name_both_files():
    a = artifacts.Artifact("dup", type=DataFrame, partitions={"date": artifacts.Daily})
    b = artifacts.Artifact("dup", type=DataFrame, partitions={"date": artifacts.Daily, "region": str})
    with pytest.raises(LineageDeclarationError) as exc:
        check_conflicts([a, b])
    msg = str(exc.value)
    here = os.path.relpath(__file__, os.getcwd())
    assert f"artifact 'dup' is declared with different dimensions or type in {here}:{a.src_line}" in msg
    assert f"and {here}:{b.src_line}" in msg
    assert "(type=DataFrame, dims=[date: day])" in msg and "(type=DataFrame, dims=[date: day, region: str])" in msg


@pytest.mark.parametrize(
    "a,b,ok",
    [
        (
            {"type": File, "partitions": {"d": artifacts.Daily}},
            {"type": File, "partitions": {"d": artifacts.Daily}},
            True,
        ),
        ({"type": File, "partitions": {"d": artifacts.Daily}}, {"partitions": {"d": artifacts.Daily}}, True),
        ({"type": File, "partitions": {"d": artifacts.Daily}}, {}, True),
        (
            {"type": File, "partitions": {"d": artifacts.Daily}},
            {"type": DataFrame, "partitions": {"d": artifacts.Daily}},
            False,
        ),
        (
            {"type": File, "partitions": {"d": artifacts.Daily}},
            {"type": File, "partitions": {"d": artifacts.Hourly}},
            False,
        ),
        ({"type": File, "partitions": {"d": str}}, {"type": File, "partitions": {"d": int}}, False),
    ],
)
def test_conflict_matrix(a, b, ok):
    ha, hb = artifacts.Artifact("c", **a), artifacts.Artifact("c", **b)
    if ok:
        check_conflicts([ha, hb])
    else:
        with pytest.raises(LineageDeclarationError):
            check_conflicts([ha, hb])


def test_conflict_across_tasks_fails_summary():
    restated = artifacts.Artifact("features", type=DataFrame, partitions={"day": artifacts.Daily})
    env = flyte.TaskEnvironment(name="conflict")

    @env.task(consumes_artifacts={"x": restated})
    async def reader(x: DataFrame) -> None:
        raise NotImplementedError

    with pytest.raises(LineageDeclarationError, match="artifact 'features' is declared with different dimensions"):
        summarize([P.featurize, reader])


def test_override_keeps_and_replaces_declarations():
    o = P.train.override(produces_artifacts=True)
    assert o.produces_artifacts == (P.churn_model,)
    assert P.train.override(produces_artifacts=False).produces_artifacts is False
    assert P.train.override(labels={"a": "b"}).labels == {"a": "b"}
    other = artifacts.Artifact("other", type=File)
    assert P.train.override(produces_artifacts=(other,)).produces_artifacts == (other,)
    assert P.train.override(consumes_artifacts={}).consumes_artifacts == {}


def test_default_values_json():
    env = flyte.TaskEnvironment(name="defaults")

    class Weird:
        def __repr__(self):
            return "Weird()"

    @env.task(produces_artifacts=(h_out,), consumes_artifacts={"date": h_out.get_partition_value("date")})
    async def t(date: datetime, n: Optional[int] = None, tags: list[str] = ["a"], w: str = "x") -> File:
        raise NotImplementedError

    params = extract_task_lineage(t).bindings["parameters"]
    assert {k: params["n"][k] for k in ("kind", "type", "default")} == {
        "kind": "default",
        "type": "int | None",
        "default": None,
    }
    assert params["tags"]["default"] == ["a"]
    assert Weird is not None
