"""Lineage must never break serialization, and must stay inside the backend's size limits."""

import json
import os
import pathlib

import pytest

import flyte
from flyte._internal.runtime.task_serde import get_proto_task
from flyte.artifacts import _lineage
from flyte.artifacts._lineage import (
    BINDINGS_LABEL,
    BINDINGS_SOFT_LIMIT,
    CONSUMES_LABEL,
    PRODUCES_LABEL,
    app_lineage_labels,
    check_edge_limits,
    validate_labels,
)
from flyte.errors import LineageDeclarationError
from flyte.io import File
from flyte.models import SerializationContext

from . import proposal_tasks as P

ROOT = pathlib.Path(os.path.dirname(os.path.abspath(P.__file__)))


def _sc(**kw):
    return SerializationContext(project="p", domain="d", version="v", org="o", root_dir=ROOT, **kw)


# ------------------------------------------------------------------ serde gating (item: child actions)


def test_lineage_tags_only_on_registration():
    assert dict(get_proto_task(P.train, _sc()).metadata.tags) == {}  # in-pod submit / connector: no work
    tags = dict(get_proto_task(P.train, _sc(emit_lineage_tags=True)).metadata.tags)
    assert tags[PRODUCES_LABEL] == "churn_model"


def test_lineage_tag_failure_never_breaks_serialization(monkeypatch):
    import flyte.artifacts._lineage as lineage

    def boom(*a, **k):
        raise RuntimeError("lineage bug")

    monkeypatch.setattr(lineage, "task_lineage_tags", boom)
    tt = get_proto_task(P.train, _sc(emit_lineage_tags=True))
    assert dict(tt.metadata.tags) == {}
    assert tt.metadata.produces_artifacts is True


# ------------------------------------------------------------------ size limits


def test_bindings_shed_descriptions_then_source_then_drop(caplog):
    # Each description is within the backend's 2048-byte cap; together they pass the payload's soft limit.
    arts = {f"a{i}": {"name": f"a{i}", "description": "x" * 2000} for i in range(BINDINGS_SOFT_LIMIT // 2000 + 1)}
    b = {"artifacts": arts, "parameters": {"p": {"kind": "unbound"}}}
    out = _lineage._encode_bindings(b, "t")
    assert out is not None and "description" not in json.loads(out)["artifacts"]["a0"]

    params = {f"p{i}": {"kind": "unbound", "src_file": "f" * 200, "src_line": i} for i in range(400)}
    b = {"artifacts": {}, "parameters": params}
    out = _lineage._encode_bindings(b, "t")
    assert out is not None and "src_file" not in json.loads(out)["parameters"]["p0"]

    b = {"artifacts": {}, "parameters": {f"p{i}": {"kind": "unbound", "type": "x" * 200} for i in range(400)}}
    assert _lineage._encode_bindings(b, "t") is None


def test_oversized_bindings_drop_only_the_bindings_label():
    many = {f"p{i}": {"kind": "default", "value": "v" * 300} for i in range(300)}
    labels = app_lineage_labels("big-app", consumed=["a"], bindings={"version": 1, "parameters": many})
    assert BINDINGS_LABEL not in labels
    assert labels[CONSUMES_LABEL] == "a"


@pytest.mark.parametrize(
    "labels, match",
    [
        ({CONSUMES_LABEL: ",".join(f"n{i}" for i in range(257))}, "at most 256"),
        ({CONSUMES_LABEL: "n" * 4097}, "the limit is 4096"),
        (
            {
                CONSUMES_LABEL: ",".join(f"c{i}" for i in range(100)),
                PRODUCES_LABEL: ",".join(f"p{i}" for i in range(50)),
            },
            "at most 4096 are allowed",
        ),
    ],
)
def test_edge_limits_are_readable_deploy_errors(labels, match):
    with pytest.raises(LineageDeclarationError, match=match):
        check_edge_limits(labels, "t")


def test_hand_written_label_too_many_ids():
    with pytest.raises(LineageDeclarationError, match="at most 256"):
        validate_labels({CONSUMES_LABEL: ",".join(f"n{i}" for i in range(300))}, entity="task", where="t")


def test_task_with_too_many_edges_fails_extraction():
    env = flyte.TaskEnvironment(name="too-many", labels={CONSUMES_LABEL: ",".join(f"n{i}" for i in range(200))})

    @env.task(labels={PRODUCES_LABEL: "only_one"})
    async def ok() -> File:
        raise NotImplementedError

    _lineage.extract_task_lineage(ok)  # 200 x 1 edges is fine

    @env.task(labels={PRODUCES_LABEL: ",".join(f"o{i}" for i in range(30))})
    async def too_many() -> File:
        raise NotImplementedError

    with pytest.raises(LineageDeclarationError, match="lineage edges"):
        _lineage.extract_task_lineage(too_many)
