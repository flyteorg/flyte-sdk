"""Deploy-time validation of produces/consumes declarations against the task signature."""

import json
import re
from datetime import date, datetime
from typing import Optional, Tuple

import pytest

import flyte
import flyte.artifacts as artifacts
from flyte.artifacts._lineage import BINDINGS_LABEL, extract_task_lineage, summarize
from flyte.errors import LineageDeclarationError
from flyte.io import DataFrame, Dir, File

env = flyte.TaskEnvironment(name="review")
out = artifacts.Artifact("rv_out", type=File, partitions={"date": artifacts.Daily})
two_dim = artifacts.Artifact("rv_two", type=File, partitions={"date": artifacts.Daily, "region": str})
counted = artifacts.Artifact("rv_count", type=File, partitions={"n": int})
up = artifacts.Artifact("rv_up", type=DataFrame, partitions={"date": artifacts.Daily, "region": str})


# ------------------------------------------------------------------ output types


class VolumeLike:
    def get_artifact_metadata(self):
        return None


@pytest.mark.parametrize("ret", [File, Dir, DataFrame, VolumeLike])
def test_artifactable_output_types_accepted(ret):
    async def f(date: datetime) -> File:
        raise NotImplementedError

    f.__annotations__["return"] = ret
    t = env.task(produces_artifacts=(out,), consumes_artifacts={"date": out.get_partition_value("date")})(f)
    assert extract_task_lineage(t).pullable


@pytest.mark.parametrize("ret,name", [(int, "int"), (str, "str"), (dict, "dict"), (Optional[File], "File | None")])
def test_non_artifact_output_rejected_at_deploy(ret, name):
    async def f(date: datetime) -> File:
        raise NotImplementedError

    f.__annotations__["return"] = ret
    t = env.task(produces_artifacts=(out,), consumes_artifacts={"date": out.get_partition_value("date")})(f)
    with pytest.raises(LineageDeclarationError) as exc:
        extract_task_lineage(t)
    assert (
        f"review.f: produces_artifacts declares output o0 as artifact 'rv_out', but it is annotated {name}; only "
        "flyte.io.File, flyte.io.Dir, flyte.io.DataFrame (or a type exposing get_artifact_metadata) can be "
        "published as an artifact."
    ) in str(exc.value)


# ------------------------------------------------------------------ partition annotations


@pytest.mark.parametrize(
    "handle,dim,ptype,ok",
    [
        (out, "date", datetime, True),
        (out, "date", date, True),
        (out, "date", Optional[datetime], True),
        (out, "date", str, False),
        (out, "date", int, False),
        (two_dim, "region", str, True),
        (two_dim, "region", int, False),
        (counted, "n", int, True),
        (counted, "n", bool, False),
        (counted, "n", str, False),
    ],
)
def test_partition_parameter_annotation_checked(handle, dim, ptype, ok):
    async def f(p: str) -> File:
        raise NotImplementedError

    f.__annotations__["p"] = ptype
    consumes = {"p": handle.get_partition_value(dim)}
    t = env.task(produces_artifacts=(handle,), consumes_artifacts=consumes)(f)
    if ok:
        extract_task_lineage(t)
        return
    kind = {"date": "time", "region": "str", "n": "int"}[dim]
    want = {"time": "datetime or date", "str": "str", "int": "int"}[kind]
    with pytest.raises(LineageDeclarationError) as exc:
        extract_task_lineage(t)
    assert (
        f"review.f: consumes_artifacts['p'] is {handle.name}.get_partition_value('{dim}'), a {kind} dimension, so "
        f"parameter 'p' must be typed {want}; it is "
    ) in str(exc.value)


# ------------------------------------------------------------------ unbound produced dimensions


def test_unbound_produced_dimension_is_unpullable_with_warning():
    async def g(date: datetime) -> File:
        raise NotImplementedError

    t = env.task(produces_artifacts=(two_dim,), consumes_artifacts={"date": two_dim.get_partition_value("date")})(g)
    lin = extract_task_lineage(t)
    msg = (
        "review.g is not pullable: dimension 'region' of rv_two is not bound by any parameter. Name a parameter "
        "'region' (with no default) and it carries the value, or bind one with artifacts.partition('region') / "
        "rv_two.get_partition_value('region'); until then, direct runs publish rv_two only when the body returns "
        "artifacts.new(value, rv_two.at(region=...))."
    )
    assert lin.pullable is False and lin.warnings() == [msg] and lin.unpullable_reason == msg
    assert json.loads(lin.labels[BINDINGS_LABEL])["pullable"] is False
    rendered = summarize([t]).render()
    deployed = "  Deployed anyway. The task still runs when called directly; it cannot be a materialize target."
    assert f"! {msg}\n{deployed}" in rendered


def test_identity_input_does_not_bind_a_produced_dimension():
    """Only get_partition_value binds a coordinate: that is what the run time reads, so deploy and run agree."""

    async def g(x: DataFrame) -> File:
        raise NotImplementedError

    t = env.task(produces_artifacts=(two_dim,), consumes_artifacts={"x": up})(g)
    lin = extract_task_lineage(t)
    assert not lin.pullable
    assert [w.split(":")[1].strip() for w in lin.warnings()] == [
        "dimension 'date' of rv_two is not bound by any parameter. Name a parameter 'date' (with no default) and it carries the value, or bind one with artifacts.partition('date') / rv_two.get_partition_value('date'); until then, direct runs publish rv_two only when the body returns artifacts.new(value, rv_two.at(date=...)).",  # noqa: E501
        "dimension 'region' of rv_two is not bound by any parameter. Name a parameter 'region' (with no default) and it carries the value, or bind one with artifacts.partition('region') / rv_two.get_partition_value('region'); until then, direct runs publish rv_two only when the body returns artifacts.new(value, rv_two.at(region=...)).",  # noqa: E501
    ]

    async def g2(x: DataFrame, date: datetime, region: str) -> File:
        raise NotImplementedError

    bound = env.task(
        produces_artifacts=(two_dim,),
        consumes_artifacts={
            "x": up,
            "date": up.get_partition_value("date"),
            "region": up.get_partition_value("region"),
        },
    )(g2)
    assert extract_task_lineage(bound).pullable


evm = artifacts.Artifact("rv_evm", type=DataFrame, partitions={"date": artifacts.Daily, "region": str})
summ = artifacts.Artifact("rv_summ", type=File, partitions={"date": artifacts.Daily})


async def _summ(x: DataFrame, date: datetime) -> File:
    raise NotImplementedError


async def _summ_list(x: list[DataFrame], date: datetime) -> File:
    raise NotImplementedError


@pytest.mark.parametrize(
    "fn,mapping,ok",
    [
        (_summ, evm, False),  # identity: region has nowhere to come from
        (_summ, evm.select(date="2026-01-01"), False),  # select pins date, region still free
        (_summ, evm.select(region="us"), True),  # region pinned
        (_summ_list, evm.all("region"), True),  # region collapsed
    ],
)
def test_single_partition_input_dims_must_exist_on_the_output(fn, mapping, ok):
    t = env.task(
        produces_artifacts=(summ,), consumes_artifacts={"x": mapping, "date": summ.get_partition_value("date")}
    )(fn)
    if ok:
        extract_task_lineage(t)
        return
    kind = mapping.kind if isinstance(mapping, artifacts.ArtifactMapping) else "identity"
    with pytest.raises(LineageDeclarationError) as exc:
        extract_task_lineage(t)
    assert str(exc.value).endswith(
        f"consumes_artifacts['x'] maps rv_evm by {kind}, but rv_evm has dimension 'region' that rv_summ does not, so "
        "nothing can choose a 'region' when building a partition. Use rv_evm.all('region') to collapse it, "
        "rv_evm.select(region=...) to pin it, or add 'region' to rv_summ."
    )


# ------------------------------------------------------------------ None placeholders (declaring a subset of outputs)

model = artifacts.Artifact("rv_model", type=File, partitions={"date": artifacts.Daily})
metrics = artifacts.Artifact("rv_metrics", type=File, partitions={"date": artifacts.Daily})


async def _four(date: datetime) -> Tuple[int, File, File, str]:
    raise NotImplementedError


def test_none_placeholders_declare_a_subset_of_outputs():
    t = env.task(
        produces_artifacts=(None, model, metrics, None),
        consumes_artifacts={"date": model.get_partition_value("date")},
    )(_four)
    lin = extract_task_lineage(t)
    b = lin.bindings
    assert b["produces"] == [{"node": "rv_model", "position": 1}, {"node": "rv_metrics", "position": 2}]
    assert b["outputs"] == 4
    assert set(b["artifacts"]) == {"rv_model", "rv_metrics"}
    assert lin.produces == ["rv_model", "rv_metrics"] and lin.pullable
    # The int and str outputs are placeholders, so the artifactable-type check skips them.


def test_none_placeholders_arity_must_match():
    t = env.task(produces_artifacts=(None, model), consumes_artifacts={"date": model.get_partition_value("date")})(
        _four
    )
    with pytest.raises(LineageDeclarationError) as exc:
        extract_task_lineage(t)
    assert (
        "review._four: produces_artifacts declares 2 position(s) (None, rv_model) but the task returns 4 value(s); a "
        "tuple return is matched to the handles by position. Use None for an output that is not an artifact, e.g. "
        "produces_artifacts=(None, model)."
    ) == str(exc.value)


def test_all_none_is_no_declaration():
    assert env.task(produces_artifacts=(None, None, None, None))(_four).produces_artifacts is False


# ------------------------------------------------------------------ deploy-wide labels


def test_deploy_wide_labels_may_not_set_produces():
    from flyte.artifacts._lineage import validate_deploy_labels

    with pytest.raises(LineageDeclarationError, match=re.escape("'lineage.produces' cannot be a deploy-wide label")):
        validate_deploy_labels({"lineage.produces": "x"})
    validate_deploy_labels({"lineage.consumes": "x", "team": "ml"})
    with pytest.raises(LineageDeclarationError, match=re.escape("reserved 'lineage.' namespace")):
        validate_deploy_labels({"lineage.edges": "x"})


# ------------------------------------------------------------------ tag cache keyed on content


def test_tag_cache_sees_in_place_edits():
    from flyte.artifacts._lineage import task_lineage_tags

    other = artifacts.Artifact("rv_other", type=DataFrame, partitions={"date": artifacts.Daily})

    async def c(x: DataFrame, date: datetime) -> File:
        raise NotImplementedError

    consumes = {"x": up.select(region="us"), "date": out.get_partition_value("date")}
    t = env.task(produces_artifacts=(out,), consumes_artifacts=consumes)(c)
    assert task_lineage_tags(t)["lineage.consumes"] == "rv_up"
    t.consumes_artifacts["x"] = other  # in-place mutation of the same dict
    assert task_lineage_tags(t)["lineage.consumes"] == "rv_other"
    out.expect(date=["2026-01-01"])
    try:
        assert json.loads(task_lineage_tags(t)[BINDINGS_LABEL])["level"] == 5  # expect() values count too
    finally:
        out.expected.clear()


# ------------------------------------------------------------------ flags, optional lists, defaults, self-loops


@pytest.mark.parametrize("value,expected", [(None, False), (0, False), (1, True), (True, True)])
def test_produces_artifacts_falsy_and_int_flags(value, expected):
    async def h(x: int) -> File:
        raise NotImplementedError

    assert env.task(produces_artifacts=value)(h).produces_artifacts is expected


def test_optional_list_accepts_all_and_window():
    async def k(xs: Optional[list[DataFrame]], date: datetime) -> File:
        raise NotImplementedError

    for mapping in (up.all("region"), up.window(date=artifacts.TimeRange(days=2))):
        t = env.task(
            produces_artifacts=(out,), consumes_artifacts={"xs": mapping, "date": out.get_partition_value("date")}
        )(k)
        assert extract_task_lineage(t).bindings["parameters"]["xs"]["mapping"]["kind"] in ("all", "window")


def test_non_json_default_goes_to_default_repr():
    class Weird:
        def __repr__(self):
            return "Weird()"

    w = Weird()

    async def m(date: datetime, w: object = w, n: int = 3) -> File:
        raise NotImplementedError

    t = env.task(produces_artifacts=(out,), consumes_artifacts={"date": out.get_partition_value("date")})(m)
    params = extract_task_lineage(t).bindings["parameters"]
    assert {k: v for k, v in params["w"].items() if not k.startswith("src_")} == {
        "kind": "default",
        "type": "object",
        "default_repr": "Weird()",
    }
    assert "default_repr" not in params["n"] and params["n"]["default"] == 3


def test_self_loop_allowed_and_marked():
    async def incremental(prev: list[File], date: datetime) -> File:
        raise NotImplementedError

    t = env.task(
        produces_artifacts=(out,),
        consumes_artifacts={
            "prev": out.window(date=artifacts.TimeRange(days=7)),
            "date": out.get_partition_value("date"),
        },
    )(incremental)
    lin = extract_task_lineage(t)
    assert lin.bindings["parameters"]["prev"]["self"] is True
    assert lin.consumes == ["rv_out"] and lin.produces == ["rv_out"]
    assert lin.resolvable_edges == []  # the backend drops self-edges; so does the summary count
    assert summarize([t]).edges == 0
    assert (
        "self"
        not in extract_task_lineage(
            env.task(
                produces_artifacts=(out,),
                consumes_artifacts={"x": up.select(region="us"), "date": out.get_partition_value("date")},
            )(_consumer)
        ).bindings["parameters"]["x"]
    )


async def _consumer(x: DataFrame, date: datetime) -> File:
    raise NotImplementedError
