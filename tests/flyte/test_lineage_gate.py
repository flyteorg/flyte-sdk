"""A deploy or run that declares no lineage pays nothing for it: same version, no lineage imports, no lineage work."""

import hashlib
import pathlib
import subprocess
import sys
import textwrap
from unittest.mock import AsyncMock, Mock, patch

import cloudpickle
import pytest

import flyte
import flyte.app
from flyte._deploy import DeploymentPlan, apply
from flyte._internal.imagebuild.image_builder import ImageCache
from flyte._internal.lineage_gate import any_lineage, env_declares_lineage, is_handle, task_declares_lineage

# ---------------------------------------------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------------------------------------------


def _plain_env(name: str) -> flyte.TaskEnvironment:
    env = flyte.TaskEnvironment(name=name, image="python:3.12")

    @env.task
    async def t(x: int) -> int:
        return x

    return env


def test_plain_envs_declare_no_lineage():
    env = _plain_env("gate_plain")
    app = flyte.app.AppEnvironment(name="gate-plain-app", image="python:3.12", args="python -m http.server")
    assert not any_lineage([env, app])
    assert not any_lineage([env], labels={})
    assert not task_declares_lineage(next(iter(env.tasks.values())))


def test_produces_true_alone_is_not_lineage():
    env = flyte.TaskEnvironment(name="gate_produces_flag", image="python:3.12")

    @env.task(produces_artifacts=True)
    async def t(x: int) -> int:
        return x

    assert not any_lineage([env])


def test_deploy_labels_env_labels_and_task_labels_are_lineage():
    assert any_lineage([_plain_env("gate_deploy_labels")], labels={"team": "ml"})
    env = _plain_env("gate_env_labels")
    env.labels = {"team": "ml"}
    assert any_lineage([env])
    assert task_declares_lineage(next(iter(env.tasks.values())))  # the environment's labels reach its tasks

    tenv = flyte.TaskEnvironment(name="gate_task_labels", image="python:3.12")

    @tenv.task(labels={"lineage.consumes": "x"})
    async def t(x: int) -> int:
        return x

    assert any_lineage([tenv])


def test_depends_on_is_walked():
    dep = _plain_env("gate_dep")
    dep.labels = {"team": "ml"}
    env = flyte.TaskEnvironment(name="gate_root", image="python:3.12", depends_on=[dep])
    assert any_lineage([env])


def test_handles_and_consumes_are_lineage():
    from flyte.artifacts import Artifact

    events = Artifact("gate_events", type=int)
    penv = flyte.TaskEnvironment(name="gate_produces", image="python:3.12")

    @penv.task(produces_artifacts=(events,))
    async def p() -> int:
        return 1

    cenv = flyte.TaskEnvironment(name="gate_consumes", image="python:3.12")

    @cenv.task(consumes_artifacts={"x": events})
    async def c(x: int) -> int:
        return x

    assert any_lineage([penv]) and any_lineage([cenv])
    assert is_handle(events) and not is_handle("gate_events")


def test_app_lineage_declarations():
    from flyte.artifacts import Artifact

    base = {"image": "python:3.12", "args": "python -m http.server"}
    assert env_declares_lineage(flyte.app.AppEnvironment(name="gate-app-opt", lineage=True, **base))
    assert env_declares_lineage(flyte.app.AppEnvironment(name="gate-app-labels", labels={"a": "b"}, **base))
    model = Artifact("gate_model", type=flyte.io.File)
    assert env_declares_lineage(
        flyte.app.AppEnvironment(name="gate-app-consumes", consumes_artifacts={"model": model}, **base)
    )
    bound = flyte.app.AppEnvironment(
        name="gate-app-param", parameters=[flyte.app.Parameter(name="model", value=model)], **base
    )
    assert env_declares_lineage(bound)


# ---------------------------------------------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------------------------------------------


async def _apply_versions(env, labels_list):
    fake_bundle = Mock()
    fake_bundle.computed_version = "test-bundle-version"
    fake_cfg = Mock(root_dir=pathlib.Path("/tmp"), images={}, project="p", domain="d", org="o")
    contexts = []

    async def deployer(context):
        contexts.append(context.serialization_context)
        deployed = Mock()
        deployed.get_name.return_value = env.name
        return deployed

    image_cache = ImageCache(image_lookup={})
    with (
        patch("flyte._initialize.is_initialized", return_value=True),
        patch("flyte._deploy.get_init_config", return_value=fake_cfg),
        patch("flyte._code_bundle._includes.collect_env_include_files", return_value=[]),
        patch("flyte._code_bundle.build_code_bundle", new=AsyncMock(return_value=fake_bundle)),
        patch("flyte._deployer.get_deployer", return_value=deployer),
    ):
        for labels in labels_list:
            await apply(
                DeploymentPlan(envs={env.name: env}), "loaded_modules", True, image_cache=image_cache, labels=labels
            )
    return contexts, image_cache, fake_bundle


@pytest.mark.asyncio
async def test_label_free_version_is_unchanged_and_a_label_changes_it():
    env = _plain_env("gate_version_env")
    contexts, image_cache, bundle = await _apply_versions(env, [None, {}, {"team": "ml"}])

    # What the version was before labels existed: the envs, the bundle version and the image cache.
    h = hashlib.md5()
    h.update(cloudpickle.dumps({env.name: env}))
    h.update(bundle.computed_version.encode("utf-8"))
    h.update(cloudpickle.dumps(image_cache))
    assert contexts[0].version == h.hexdigest()
    assert contexts[1].version == h.hexdigest()
    assert contexts[2].version != h.hexdigest()
    # Lineage tags only when the plan declares lineage.
    assert [c.emit_lineage_tags for c in contexts] == [False, False, True]


@pytest.mark.asyncio
async def test_env_label_changes_version():
    env = _plain_env("gate_version_env_labels")
    (plain,), _, _ = await _apply_versions(env, [None])
    env.labels = {"team": "ml"}
    (labelled,), _, _ = await _apply_versions(env, [None])
    assert plain.version != labelled.version


# ---------------------------------------------------------------------------------------------------------------
# Deploy does no lineage work, and imports no lineage module
# ---------------------------------------------------------------------------------------------------------------


def _deploy_with_patches(*envs, **kwargs):
    apply_mock = AsyncMock(return_value=Mock(spec=[]))
    with (
        patch("flyte._deploy.get_init_config", return_value=Mock(images={}, root_dir=pathlib.Path("/tmp"))),
        patch("flyte._deploy._build_images_for_plans", new=AsyncMock(return_value=ImageCache(image_lookup={}))),
        patch("flyte._deploy.apply", new=apply_mock),
    ):
        return flyte.deploy(*envs, **kwargs), apply_mock


def test_plain_deploy_runs_no_lineage_code():
    boom = Mock(side_effect=AssertionError("lineage code ran for a deploy that declares no lineage"))
    with (
        patch("flyte.artifacts._refresh.refresh_envs", new=boom),
        patch("flyte.artifacts._refs.check_references", new=boom),
        patch("flyte._deploy.lineage_summary", new=boom),
        patch("flyte._deploy._plan_lineage_deploy", new=boom),
    ):
        _deploy_with_patches(_plain_env("gate_deploy_plain"))
    boom.assert_not_called()


def test_lineage_deploy_still_runs_lineage_code():
    env = _plain_env("gate_deploy_labelled")
    with patch("flyte._deploy.lineage_summary", wraps=flyte._deploy.lineage_summary) as summary:
        _deploy_with_patches(env, labels={"team": "ml"})
    summary.assert_called_once()


_PLAIN_DEPLOY = textwrap.dedent(
    """
    import pathlib, sys
    from unittest.mock import AsyncMock, Mock, patch

    import flyte
    import flyte.app
    from flyte._internal.imagebuild.image_builder import ImageCache
    from flyte._internal.runtime.task_serde import translate_task_to_wire
    from flyte.app._runtime.app_serde import translate_app_env_to_idl

    env = flyte.TaskEnvironment(name="plain_env", image="python:3.12")

    @env.task
    async def t(x: int) -> int:
        return x

    app = flyte.app.AppEnvironment(
        name="plain-app",
        image="python:3.12",
        args="python -m http.server",
        parameters=[flyte.app.Parameter(name="greeting", value="hi")],
    )
    seen = []

    async def deployer(context):
        sc = context.serialization_context
        seen.append(sc.emit_lineage_tags)
        e = context.environment
        if isinstance(e, flyte.TaskEnvironment):
            for task in e.tasks.values():
                translate_task_to_wire(task, sc)
        else:
            await translate_app_env_to_idl.aio(e, sc)
        d = Mock()
        d.get_name.return_value = e.name
        return d

    cfg = Mock(root_dir=pathlib.Path(__file__).resolve().parent, images={}, project="p", domain="d", org="o")
    bundle = Mock(computed_version="v", pkl=None, tgz="s3://b/x.tgz", destination=".", downloaded_path=None)
    cache = ImageCache(image_lookup={"plain_env": "img", "plain-app": "img"})
    with (
        patch("flyte._initialize.is_initialized", return_value=True),
        patch("flyte._deploy.get_init_config", return_value=cfg),
        patch("flyte._deploy._build_images_for_plans", new=AsyncMock(return_value=cache)),
        patch("flyte._code_bundle.build_code_bundle", new=AsyncMock(return_value=bundle)),
        patch("flyte._deployer.get_deployer", return_value=deployer),
    ):
        out = flyte.deploy(env, app, dry_run=True)
    assert seen == [False, False], seen
    assert all(d.lineage is None for d in out)
    print("\\n".join(sys.modules))
    """
)


def test_plain_deploy_imports_no_lineage_module(tmp_path):
    script = tmp_path / "plain_deploy.py"
    script.write_text(_PLAIN_DEPLOY)
    out = subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True, cwd=tmp_path)
    loaded = set(out.stdout.split())
    forbidden = [
        "flyte.artifacts._lineage",
        "flyte.artifacts._refresh",
        "flyte.artifacts._refs",
        "flyte.artifacts._handle",
    ]
    assert not [m for m in forbidden if m in loaded]


def test_remote_app_managed_labels_key_matches_lineage():
    from flyte.artifacts._lineage import MANAGED_LABELS_KEY
    from flyte.remote._app import _MANAGED_LABELS_KEY

    assert _MANAGED_LABELS_KEY == MANAGED_LABELS_KEY
