"""
The SDK never writes a lineage payload the Union backend refuses (cloud/lineage/validate.go, labels.go): the backend
would store the entity without its lineage while the deploy succeeds. Each limit is normalized where the SDK writes
the text, or fails `flyte deploy` with a LineageDeclarationError naming what to fix.
"""

import json
import math
import warnings
from datetime import datetime

import pytest

import flyte
import flyte.artifacts as artifacts
from flyte.app import AppEnvironment, Parameter
from flyte.artifacts._handle import (
    MAX_DESCRIPTION_BYTES,
    MAX_SRC_FILE_BYTES,
    MAX_TYPE_BYTES,
    is_lineage_ident,
    record_source_path,
    type_name,
    valid_node_id,
)
from flyte.artifacts._lineage import (
    BINDINGS_LABEL,
    _default_record,
    _encode_bindings,
    _join_reasons,
    check_bindings,
    extract_task_lineage,
    validate_labels,
)
from flyte.errors import LineageDeclarationError
from flyte.io import File

env = flyte.TaskEnvironment(name="contract")
out = artifacts.Artifact("bc_out", type=File, partitions={"date": artifacts.Daily})


def _bindings(t) -> dict:
    return json.loads(extract_task_lineage(t).labels[BINDINGS_LABEL])


# ------------------------------------------------------------------ 1. non-finite defaults


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -math.inf, [1.0, float("nan")], {"a": math.inf}])
def test_non_finite_defaults_go_to_default_repr(value):
    rec = _default_record("float", value)
    assert "default" not in rec and rec["default_repr"] == repr(value)


def test_task_with_non_finite_default_writes_strict_json():
    @env.task(produces_artifacts=(out,), consumes_artifacts={"date": out.get_partition_value("date")})
    async def nan_default(date: datetime, threshold: float = float("inf")) -> File:
        raise NotImplementedError

    raw = extract_task_lineage(nan_default).labels[BINDINGS_LABEL]
    assert "Infinity" not in raw and "NaN" not in raw
    json.loads(raw, parse_constant=lambda c: pytest.fail(f"non-JSON constant {c}"))
    assert json.loads(raw)["parameters"]["threshold"]["default_repr"] == "inf"


def test_encoder_refuses_nan():
    with pytest.raises(LineageDeclarationError, match="non-finite"):
        _encode_bindings({"parameters": {"p": {"kind": "default", "default": float("nan")}}}, "t")


# ------------------------------------------------------------------ 2. unpullable_reason cap


def test_unpullable_reason_is_capped():
    msgs = [f"message {i} " + "x" * 300 for i in range(20)]
    reason = _join_reasons(msgs)
    assert len(reason.encode("utf-8")) <= MAX_DESCRIPTION_BYTES
    assert reason.startswith("message 0 ") and reason.endswith(" more")
    kept = reason.count("message ")
    assert reason.endswith(f"and {20 - kept} more")
    assert _join_reasons(["a", "b"]) == "a; b"
    one = _join_reasons(["y" * 5000, "z"])
    assert len(one.encode("utf-8")) <= MAX_DESCRIPTION_BYTES and one.endswith("and 1 more")


def test_task_with_many_unbound_parameters_stays_under_the_cap():
    params = ", ".join(f"p{i}: int" for i in range(40))
    ns: dict = {"File": File, "datetime": datetime}
    exec(f"async def many(date: datetime, {params}) -> File: ...", ns)
    t = env.task(produces_artifacts=(out,), consumes_artifacts={"date": out.get_partition_value("date")})(ns["many"])
    lin = extract_task_lineage(t)
    assert not lin.pullable and len(lin.unpullable_messages) == 40
    assert len(_bindings(t)["unpullable_reason"].encode("utf-8")) <= MAX_DESCRIPTION_BYTES


# ------------------------------------------------------------------ 3. identifier rule


@pytest.mark.parametrize("name", ["date", "_x", "a1", "A" * 64])
def test_identifiers_accepted(name):
    assert is_lineage_ident(name)


@pytest.mark.parametrize("name", ["", "1a", "a-b", "ñ", "a" * 65, "a\n", "a b"])
def test_identifiers_rejected(name):
    assert not is_lineage_ident(name)


@pytest.mark.parametrize("dim", ["día", "d" * 65, "date\n"])
def test_dimension_names_follow_the_rule(dim):
    with pytest.raises(ValueError, match="valid identifier"):
        artifacts.Artifact("bc_dims", partitions={dim: str})


def test_task_parameter_names_follow_the_rule_when_it_declares_lineage():
    ns: dict = {"File": File, "datetime": datetime}
    exec("async def uni(date: datetime, größe: int = 1) -> File: ...", ns)
    t = env.task(produces_artifacts=(out,), consumes_artifacts={"date": out.get_partition_value("date")})(ns["uni"])
    with pytest.raises(LineageDeclarationError, match="parameter 'größe' cannot carry lineage"):
        extract_task_lineage(t)
    # Without lineage declarations the same name is fine.
    plain = env.task(ns["uni"])
    extract_task_lineage(plain)


def test_app_parameter_names_follow_the_rule():
    from flyte._deploy import lineage_summary

    model = artifacts.Artifact("bc_model", type=File)
    app = AppEnvironment(name="bc-app", parameters=[Parameter(name="model-path", value=model)])
    with pytest.raises(LineageDeclarationError) as exc:
        lineage_summary([app])
    assert "parameter 'model-path'" in str(exc.value) and "model_path" in str(exc.value)
    ok = AppEnvironment(name="bc-app2", parameters=[Parameter(name="model_path", value=model)])
    lineage_summary([ok])


# ------------------------------------------------------------------ 4. free text caps


def test_type_name_is_one_capped_line():
    class Weird:
        pass

    Weird.__name__ = "Line1\nLine2\t\x01" + "T" * 400
    tn = type_name(Weird)
    assert "\n" not in tn and "\x01" not in tn and len(tn.encode("utf-8")) <= MAX_TYPE_BYTES
    assert tn.startswith("Line1 Line2 ") and tn.endswith("…")
    assert type_name(list[File]) == "list[File]"


def test_long_description_is_truncated_once_with_a_warning():
    h = artifacts.Artifact("bc_desc", type=File, description="é" * 3000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = h.to_dict()
        h.to_dict()
    assert len(d["description"].encode("utf-8")) <= MAX_DESCRIPTION_BYTES and d["description"].endswith("…")
    assert sum("bc_desc" in str(x.message) for x in w) == 1


def test_src_file_outside_the_root_is_its_basename(tmp_path):
    inside = tmp_path / "pkg" / "mod.py"
    assert record_source_path(str(inside), str(tmp_path)) == "pkg/mod.py"
    assert record_source_path("/somewhere/else/mod.py", str(tmp_path)) == "mod.py"
    deep = str(tmp_path / ("d" * 200) / ("e" * 200) / ("f" * 200) / "mod.py")
    rel = record_source_path(deep, str(tmp_path))
    assert len(rel.encode("utf-8")) <= MAX_SRC_FILE_BYTES and rel.endswith("mod.py")


# ------------------------------------------------------------------ 5. project / domain


@pytest.mark.parametrize(
    "scope", [{"project": "ML"}, {"project": "ml_team"}, {"domain": "-dev"}, {"project": "a" * 64}]
)
def test_handle_scope_must_be_dns_labels(scope):
    with pytest.raises(ValueError, match="not a valid"):
        artifacts.Artifact("bc_scope", **scope)
    with pytest.raises(ValueError, match="not a valid"):
        artifacts.Artifact.ref("bc_scope", **scope)


def test_valid_scope_accepted():
    h = artifacts.Artifact.ref("bc_scope", project="ml-team", domain="development")
    assert h.project == "ml-team"


# ------------------------------------------------------------------ 6. node ids


@pytest.mark.parametrize(
    "node",
    ["events", "a.b-c_d", "x:y", "app:scoring", "app:ml/scoring", "task:p/env.t", "trigger:p/t/n", "task:a/b/c"],
)
def test_valid_node_ids(node):
    assert valid_node_id(node)
    validate_labels({"lineage.consumes": node}, entity="task", where="t")


@pytest.mark.parametrize(
    "node",
    [
        "hidden:p/abc",  # reserved for masked ids
        "app:",  # empty name
        "app:ml/",  # trailing slash
        "task:a//b",  # empty segment
        "task:/b",  # empty project
        "app:_ml/x",  # project must start alphanumeric
        "a/b",  # '/' only in prefixed ids
        "events\n",
        "ü",
    ],
)
def test_invalid_node_ids(node):
    assert not valid_node_id(node)
    if node.strip() != node:
        return  # label values are split and stripped first
    with pytest.raises(LineageDeclarationError, match="invalid node id"):
        validate_labels({"lineage.consumes": node}, entity="task", where="t")


def test_artifact_name_with_trailing_newline_rejected():
    with pytest.raises(ValueError, match="may only contain"):
        artifacts.Artifact("events\n")


# ------------------------------------------------------------------ 7. trigger partition keys and names


def test_on_artifact_by_name_never_warns_on_a_partition_key():
    # OnArtifact("name", ...) and TriggeredPartition("key") predate lineage: an SDK upgrade must not make them
    # warn (or import the handle machinery).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        flyte.OnArtifact("raw_events", partitions={"the-region": "us"})
        flyte.TriggeredPartition("the-date")


def test_on_artifact_by_handle_warns_on_an_unrecordable_partition_key():
    # A handle without declared partitions cannot reject the key, so it warns that the trigger is unrecordable.
    with pytest.warns(UserWarning, match="won't appear in the lineage graph"):
        flyte.OnArtifact(artifacts.Artifact("bc_bare"), partitions={"the-region": "us"})


def test_unrecordable_partition_key_fails_on_a_lineage_task():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trig = flyte.Trigger(
            name="bc_trig",
            automation=flyte.OnArtifact("raw_events", partitions={"the-region": "us"}),
            inputs={"date": flyte.TriggeredPartition("date")},
        )

    @env.task(produces_artifacts=(out,), consumes_artifacts={"date": out.get_partition_value("date")}, triggers=(trig,))
    async def triggered(date: datetime) -> File:
        raise NotImplementedError

    with pytest.raises(LineageDeclarationError, match="partition key 'the-region'"):
        extract_task_lineage(triggered)

    @env.task(triggers=(trig,))
    async def plain_triggered(date: datetime) -> str:
        return ""

    extract_task_lineage(plain_triggered)  # no lineage declared: nothing to check


def _record_warnings(monkeypatch) -> list:
    from flyte._logging import logger

    seen: list = []
    monkeypatch.setattr(logger, "warning", lambda msg, *a, **k: seen.append(str(msg)))
    return seen


def test_trigger_name_outside_the_graph_grammar_warns_at_deploy(monkeypatch):
    trig = flyte.Trigger(name="on new model!", automation=flyte.OnArtifact("bc_model"), inputs={"x": 1})

    @env.task(triggers=(trig,))
    async def named_oddly(x: int) -> str:
        return ""

    seen = _record_warnings(monkeypatch)
    extract_task_lineage(named_oddly)
    assert any("won't appear in the lineage graph" in m for m in seen)


# ------------------------------------------------------------------ backstop


def test_check_bindings_mirrors_the_backend():
    check_bindings({"task": "t", "parameters": {"p": {"kind": "default", "type": "int"}}}, "t")
    bad = [
        {"parameters": {"a-b": {"kind": "default"}}},
        {"parameters": {"p": {"kind": "artifact", "mapping": {"kind": "select", "values": {"a-b": "x"}}}}},
        {"parameters": {"p": {"kind": "artifact", "mapping": {"kind": "select", "values": {"k": "x\ny"}}}}},
        {"artifacts": {"a": {"type": "x" * 300}}},
        {"artifacts": {"a": {"project": "-bad"}}},
        {"artifacts": {"a": {"dims": [{"name": "1d"}]}}},
        {"task": "t\x00"},
    ]
    for b in bad:
        with pytest.raises(LineageDeclarationError):
            check_bindings(b, "t")


def test_select_value_must_be_short_single_line():
    h = artifacts.Artifact("bc_sel", partitions={"date": artifacts.Daily, "region": str})
    with pytest.raises(ValueError, match="one line"):
        h.select(region="x" * 300)
    with pytest.raises(ValueError, match="one line"):
        h.select(region="a\nb")


# ------------------------------------------------------------------ naive datetimes (item 16)


def test_naive_datetime_warning_only_on_handle_paths(monkeypatch):
    from flyte.artifacts import _partitions

    monkeypatch.setattr(_partitions, "_warned_naive", set())
    seen = _record_warnings(monkeypatch)
    naive = datetime(2026, 9, 8, 13, 30)
    artifacts.Metadata(name="m", partitions={"hour": naive})  # released path: silent
    _partitions.floor_time(naive, "hour")
    assert not any("treated as UTC" in m for m in seen)
    h = artifacts.Artifact("bc_naive", partitions={"hour": artifacts.Hourly})
    h.at(hour=naive)
    assert any("treated as UTC" in m for m in seen)
