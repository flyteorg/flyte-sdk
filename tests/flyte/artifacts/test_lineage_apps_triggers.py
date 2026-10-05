"""Handles on apps (consumes_artifacts, Parameter(value=handle), Meta.labels) and triggers (OnArtifact)."""

import json
import pathlib
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import flyte
import flyte.app
import flyte.artifacts as artifacts
from flyte.app import AppEndpoint, AppEnvironment, ArtifactValue, Parameter
from flyte.app._runtime.app_serde import translate_app_env_to_idl
from flyte.artifacts._lineage import app_env_lineage_labels, app_lineage_labels
from flyte.errors import LineageDeclarationError
from flyte.io import Dir, File
from flyte.models import SerializationContext

churn_model = artifacts.Artifact("churn_model", type=File, partitions={"date": artifacts.Daily}, kind="model")
weights = artifacts.Artifact("weights_dir", type=Dir, project="ml", domain="prod")
untyped = artifacts.Artifact("thing")


# ------------------------------------------------------------------ triggers


def test_on_artifact_accepts_handle():
    t = flyte.OnArtifact(churn_model)
    assert t.name == "churn_model" and t.partitions is None
    assert str(t) == "Artifact Trigger: on new version of churn_model"
    regional = artifacts.Artifact("regional_model", partitions={"date": artifacts.Daily, "region": str})
    p = flyte.OnArtifact(regional, region="us")
    assert p.partitions == {"region": "us"}
    assert flyte.OnArtifact("churn_model").name == "churn_model"
    trig = flyte.Trigger(
        name="revalidate", automation=flyte.OnArtifact(churn_model), inputs={"model": flyte.TriggeredArtifact}
    )
    assert trig.automation.name == "churn_model"


@pytest.mark.asyncio
async def test_on_artifact_trigger_serializes_handle_name():
    from flyte._internal.runtime.trigger_serde import to_task_trigger
    from flyte._internal.runtime.types_serde import transform_native_to_typed_interface

    env = flyte.TaskEnvironment(name="trig")
    trig = flyte.Trigger(
        name="revalidate",
        automation=flyte.OnArtifact(churn_model),
        inputs={"model": flyte.TriggeredArtifact, "threshold": 0.82},
    )

    @env.task(triggers=(trig,))
    async def validate(model: File, threshold: float) -> str:
        return ""

    iface = transform_native_to_typed_interface(validate.native_interface)
    out = await to_task_trigger(t=trig, task_name=validate.name, task_inputs=iface.inputs, task_default_inputs=[])
    art = out.automation_spec.artifact
    assert (art.artifact_name, art.input_arg) == ("churn_model", "model")


def test_on_artifact_rejects_empty():
    with pytest.raises(ValueError, match="non-empty artifact name"):
        flyte.OnArtifact("")


# ------------------------------------------------------------------ Parameter(value=handle)


@pytest.mark.parametrize(
    "handle,expected_type",
    [(churn_model, "file"), (weights, "directory"), (untyped, None)],
)
def test_parameter_with_handle_becomes_artifact_value(handle, expected_type):
    p = Parameter(name="model", value=handle, download=True, env_var="MODEL_PATH")
    assert isinstance(p.value, ArtifactValue)
    assert p.value.name == handle.name and p.value.type == expected_type
    assert (p.value.project, p.value.domain) == (handle.project, handle.domain)
    assert p.download is True and p.env_var == "MODEL_PATH"


# ------------------------------------------------------------------ AppEnvironment(consumes_artifacts=, labels=)


def test_consumes_artifacts_desugars_to_parameter():
    app = AppEnvironment(name="churn-scoring", consumes_artifacts={"model": churn_model}, labels={"team": "ml"})
    (p,) = app.parameters
    assert p.name == "model" and p.download is True
    assert isinstance(p.value, ArtifactValue) and p.value.name == "churn_model" and p.value.type == "file"
    assert app.labels == {"team": "ml"}


def test_consumes_artifacts_collision_and_type_errors():
    with pytest.raises(ValueError, match=r"consumes_artifacts\['model'\] of app 'a1' collides with a parameter"):
        AppEnvironment(
            name="a1", parameters=[Parameter(name="model", value="x")], consumes_artifacts={"model": churn_model}
        )
    with pytest.raises(
        TypeError, match=r"consumes_artifacts\['model'\] of app 'a2' must be a flyte.artifacts.Artifact"
    ):
        AppEnvironment(name="a2", consumes_artifacts={"model": "churn_model"})


def test_consumes_artifacts_desugaring_is_idempotent_on_clone():
    app = AppEnvironment(name="a3", consumes_artifacts={"model": churn_model})
    clone = app.clone_with("a4")
    assert [p.name for p in clone.parameters] == ["model"]


def test_app_lineage_labels():
    app = AppEnvironment(
        name="churn-dashboard",
        consumes_artifacts={"model": churn_model},
        parameters=[
            Parameter(name="scorer", value=AppEndpoint(app_name="churn-scoring")),
            Parameter(name="w", value=weights),
            Parameter(name="title", value="hi"),
        ],
        labels={"team": "analytics", "lineage.consumes": "daily_report"},
        depends_on=[flyte.TaskEnvironment(name="unrelated_dep")],
    )
    labels = app_env_lineage_labels(app, extra_labels={"owner": "x"})
    bindings = json.loads(labels.pop("lineage.bindings"))
    assert labels == {
        "team": "analytics",
        "owner": "x",
        "lineage.consumes": "app:churn-scoring,weights_dir,churn_model,daily_report",
    }
    assert "lineage.produces" not in labels  # derived by the backend
    # Which parameter takes which artifact: AppEndpoint and plain values are not artifact bindings.
    assert bindings["version"] == 1 and bindings["app"] == "churn-dashboard"
    assert bindings["parameters"] == {
        "w": {"kind": "artifact", "node": "weights_dir", "type": "Dir", "mapping": {"kind": "identity"}},
        "model": {"kind": "artifact", "node": "churn_model", "type": "File", "mapping": {"kind": "identity"}},
    }
    assert set(bindings["artifacts"]) == {"weights_dir", "churn_model"}
    assert bindings["artifacts"]["churn_model"]["dims"] == churn_model.to_dict()["dims"]
    assert bindings["artifacts"]["churn_model"]["kind"] == "model"


def test_app_bindings_bare_artifact_value_has_node_but_no_record():
    app = AppEnvironment(
        name="bare-app", parameters=[Parameter(name="m", value=ArtifactValue(name="ext_model", type="file"))]
    )
    bindings = json.loads(app_env_lineage_labels(app)["lineage.bindings"])
    assert bindings["parameters"] == {
        "m": {"kind": "artifact", "node": "ext_model", "type": "File", "mapping": {"kind": "identity"}}
    }
    assert bindings["artifacts"] == {}


def test_app_without_artifact_parameters_has_no_bindings_label():
    app = AppEnvironment(name="plain-app", image=flyte.Image.from_base("python:3.11"))
    assert "lineage.bindings" not in app_env_lineage_labels(app)


def test_app_that_consumes_only_by_label_records_empty_parameters():
    app = AppEnvironment(name="ep-app", parameters=[Parameter(name="s", value=AppEndpoint(app_name="other"))])
    labels = app_env_lineage_labels(app)
    assert labels["lineage.consumes"] == "app:other"
    assert json.loads(labels["lineage.bindings"])["parameters"] == {}


def test_app_rejects_authored_produces():
    app = AppEnvironment(name="bad-app", labels={"lineage.produces": "app:bad-app"})
    with pytest.raises(LineageDeclarationError, match=re.escape("an app may not set 'lineage.produces'")):
        app_env_lineage_labels(app)
    with pytest.raises(LineageDeclarationError, match=re.escape("an app may not set 'lineage.produces'")):
        app_lineage_labels("x", extra_labels={"lineage.produces": "y"})
    with pytest.raises(LineageDeclarationError, match=re.escape("reserved 'lineage.' namespace")):
        app_lineage_labels("x", labels={"lineage.bindings": "{}"})


def test_translate_app_writes_meta_labels():
    app = AppEnvironment(
        name="churn-scoring",
        image=flyte.Image.from_base("python:3.11"),
        consumes_artifacts={"model": churn_model},
        labels={"team": "ml"},
    )
    ctx = SerializationContext(
        org="o", project="p", domain="d", version="v1", root_dir=pathlib.Path.cwd(), labels={"env": "dev"}
    )
    with patch.object(ArtifactValue, "materialize", AsyncMock(return_value=File(path="s3://bucket/model.json"))):
        idl = translate_app_env_to_idl(app, ctx)
    labels = dict(idl.metadata.labels)
    bindings = json.loads(labels.pop("lineage.bindings"))
    assert labels == {
        "team": "ml",
        "env": "dev",
        "lineage.consumes": "churn_model",
        # what a redeploy may replace or remove
        "flyte.io/managed-labels": "env,lineage.bindings,lineage.consumes,team",
    }
    # The spec holds the resolved URI; the bindings label keeps which artifact the parameter takes.
    assert bindings["parameters"]["model"]["node"] == "churn_model"
    assert "churn_model" in bindings["artifacts"]


def test_translate_plain_app_has_no_labels():
    app = AppEnvironment(name="plain-app", image=flyte.Image.from_base("python:3.11"))
    ctx = SerializationContext(org="o", project="p", domain="d", version="v1", root_dir=pathlib.Path.cwd())
    assert dict(translate_app_env_to_idl(app, ctx).metadata.labels) == {}


def test_environment_labels_cloned():
    env = flyte.TaskEnvironment(name="lab", labels={"team": "ml"})
    assert env.clone_with("lab2").labels == {"team": "ml"}
    assert flyte.app is not None


def test_app_labels_validated_in_pre_build_pass():
    from flyte._deploy import lineage_summary

    app = AppEnvironment(name="rv-bad-app", labels={"lineage.produces": "app:x"})
    with pytest.raises(LineageDeclarationError, match=re.escape("an app may not set 'lineage.produces'")):
        lineage_summary([app], root_dir=None)
    ok = AppEnvironment(name="rv-ok-app", labels={"team": "ml"})
    with pytest.raises(LineageDeclarationError, match=re.escape("reserved 'lineage.' namespace")):
        lineage_summary([ok], labels={"lineage.bindings": "x"}, root_dir=None)
    lineage_summary([ok], labels={"team": "x"}, root_dir=None)


def test_deploy_rejects_bad_app_labels_before_building():
    import flyte._deploy as d

    app = AppEnvironment(name="rv-bad-app2", labels={"lineage.produces": "app:x"})
    with (
        patch.object(d, "get_init_config", return_value=MagicMock(root_dir=None, images={})),
        patch.object(d, "_build_images_for_plans") as build,
    ):
        with pytest.raises(LineageDeclarationError):
            d.deploy(app)
        build.assert_not_called()


# ------------------------------------------------------------------ 14. materialize .aio, queue, partitions=


def test_app_clone_with_labels():
    churn = artifacts.Artifact("rv_churn", type=File)
    app = AppEnvironment(name="rv-app", labels={"team": "ml"}, consumes_artifacts={"model": churn})
    clone = app.clone_with("rv-app2", labels={"team": "data"})
    assert clone.labels == {"team": "data"} and [p.name for p in clone.parameters] == ["model"]
    assert app.clone_with("rv-app3").labels == {"team": "ml"}


def test_app_clone_with_consumes_artifacts_replaces_desugared_parameters():
    from flyte.app import Parameter

    m = artifacts.Artifact("cl_model", type=File)
    n = artifacts.Artifact("cl_other", type=File)
    app = AppEnvironment(
        name="cl-app", parameters=[Parameter(name="title", value="t")], consumes_artifacts={"model": m}
    )
    assert [p.name for p in app.parameters] == ["title", "model"]
    swapped = app.clone_with("cl-app2", consumes_artifacts={"other": n})
    assert [(p.name, getattr(p.value, "name", p.value)) for p in swapped.parameters] == [
        ("title", "t"),
        ("other", "cl_other"),
    ]
    assert [p.name for p in app.clone_with("cl-app3", consumes_artifacts={}).parameters] == ["title"]
    assert [p.name for p in app.clone_with("cl-app4").parameters] == ["title", "model"]
