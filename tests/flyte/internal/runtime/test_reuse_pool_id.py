"""
flyteorg/flyte#8326: a reusable task serialized inside a running task must get the same pool id
(`custom["version"]`) it was registered with at deploy, or it never lands on the deployed pool.

No cluster is needed: the deploy side is `_deploy_task` in dryrun, as `flyte deploy` calls it with the
default `sync_local_sys_paths=True`; the in-task side runs the real `a0` entrypoint on the deployed
parent's container args (only the task launch is stubbed) and then serializes the child the way
`RemoteController._submit` does.
"""

import asyncio
import pathlib
from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner
from google.protobuf.json_format import MessageToDict
from kubernetes.client import V1Container, V1EnvVar, V1PodSpec

import flyte
from flyte._constants import FLYTE_SYS_PATH
from flyte._deploy import _deploy_task
from flyte._internal.runtime.task_serde import _primary_container_args, translate_task_to_wire
from flyte.models import CodeBundle, SerializationContext

REUSE = flyte.ReusePolicy(replicas=1, concurrency=4, idle_ttl=timedelta(minutes=10))
env = flyte.TaskEnvironment(name="pool_repro", image="ghcr.io/example/pool-repro:1", reusable=REUSE)
pod_env = flyte.TaskEnvironment(
    name="pool_repro_pod",
    image="ghcr.io/example/pool-repro:1",
    reusable=REUSE,
    pod_template=flyte.PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(containers=[V1Container(name="primary", env=[V1EnvVar(name="X", value="1")])]),
    ),
)


@env.task
async def child() -> str:
    return "ok"


@env.task
async def parent() -> str:
    return await child()


@pod_env.task
async def pod_child() -> str:
    return "ok"


@pod_env.task
async def pod_parent() -> str:
    return await pod_child()


ROOT = pathlib.Path.cwd()
# What build_code_bundle returns after upload: computed_version is the archive's md5, the tgz URI is
# where the upload landed (the file name carries the bundle's content digest).
DEPLOYED_BUNDLE = CodeBundle(
    tgz="s3://bucket/fast/3f2a9c/fast0c8d1e2f.tar.gz",
    destination=".",
    computed_version="5d41402abc4b2a76b9719d911017c592",
)
CHANGED_BUNDLE = CodeBundle(
    tgz="s3://bucket/fast/77aa01/fast9e9e9e9e.tar.gz",
    destination=".",
    computed_version="7d793037a0760186574b",
)


def _make_ctx(bundle: CodeBundle | None, version: str, root_dir: pathlib.Path = ROOT) -> SerializationContext:
    return SerializationContext(
        project="p", domain="d", org="o", code_bundle=bundle, version=version, root_dir=root_dir
    )


def _read_pool_id(spec) -> str:
    return MessageToDict(spec.task_template.custom)["version"]


def _deploy(version: str, bundle: CodeBundle | None = DEPLOYED_BUNDLE, tasks=(parent, child)):
    """Register the tasks the way `flyte deploy` does, dryrun so no client is needed."""
    ctx = _make_ctx(bundle, version)
    with (
        patch("flyte._deploy.ensure_client"),
        patch("flyte._deploy.get_init_config", return_value=Mock(sync_local_sys_paths=True)),
    ):
        return [asyncio.run(_deploy_task(t, ctx, dryrun=True)).deployed_task for t in tasks]


def _serialize_child_in_task(bundle: CodeBundle | None, version: str, task=child) -> str:
    # Mirrors RemoteController._submit: a tgz bundle (or none) is passed down unchanged, with tctx.version.
    root_dir = pathlib.Path(bundle.destination).absolute() if bundle else pathlib.Path.cwd()
    return _read_pool_id(translate_task_to_wire(task, _make_ctx(bundle, version, root_dir)))


def _run_a0(args, monkeypatch, tmp_path):
    """Run the real `a0` entrypoint and return the (code_bundle, version) it hands the task."""
    import flyte._initialize
    import flyte._internal.runtime.entrypoints as entrypoints
    from flyte._bin.runtime import RUN_OUTPUT_BASE_DIR, _pass_through

    seen = {}

    async def record_task_launch(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(entrypoints, "load_and_run_task", record_task_launch)
    monkeypatch.setattr(flyte._initialize, "init_in_cluster", lambda **kw: {})
    # CliRunner's captured stderr has no fileno for faulthandler.
    monkeypatch.setattr("faulthandler.register", lambda *a, **kw: None)
    # A clustered worker runs with no controller, which is all this test needs.
    monkeypatch.setenv("FLYTE_CLUSTERED_WORKER", "1")
    # a0 reads --run-base-dir from this env var (_U_RUN_BASE), which the platform sets on task pods.
    monkeypatch.setenv(RUN_OUTPUT_BASE_DIR, str(tmp_path))
    args = [a.replace("{{.runName}}", "r1").replace("{{.actionName}}", "a0") for a in args]
    result = CliRunner().invoke(_pass_through, [*args, "--org", "o", "--project", "p", "--domain", "d"])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return seen["code_bundle"], seen["version"]


def _pool_id_in_task(parent_spec, monkeypatch, tmp_path) -> str:
    """The pool id `child` gets when the deployed parent, started by `a0`, calls it."""
    bundle, version = _run_a0(list(parent_spec.task_template.container.args), monkeypatch, tmp_path)
    assert bundle.tgz == parent_spec.task_template.metadata.code_bundle_uri
    return _serialize_child_in_task(bundle, version)


@pytest.mark.parametrize("deploy_version", ["v1", "derived-md5-of-envs-and-bundle"])
def test_child_pool_id_in_running_task_matches_deploy(deploy_version, monkeypatch, tmp_path):
    parent_spec, child_spec = _deploy(deploy_version)
    registered = _read_pool_id(child_spec)
    # Same environment, so parent and child share one pool.
    assert _read_pool_id(parent_spec) == registered
    # Deploy bakes the local sys.path into the registered template; the in-task child never has it.
    assert FLYTE_SYS_PATH in {kv.key for kv in child_spec.task_template.container.env}

    assert _pool_id_in_task(parent_spec, monkeypatch, tmp_path) == registered


def test_child_pool_id_on_reusable_worker_matches_deploy(monkeypatch):
    import flyte._internal.runtime.rusty as rusty

    _, child_spec = _deploy("v1")

    async def skip_download(bundle):
        return bundle

    monkeypatch.setattr(rusty, "download_code_bundle", skip_download)
    # unionai-reuse calls download_tgz(dest, version, tgz) with the action's --dest/--version/--tgz.
    bundle = asyncio.run(rusty.download_tgz(".", "v1", DEPLOYED_BUNDLE.tgz))
    assert _serialize_child_in_task(bundle, "v1") == _read_pool_id(child_spec)


def test_copy_style_none_child_matches_deploy():
    _, child_spec = _deploy("v1", bundle=None)
    assert _serialize_child_in_task(None, "v1") == _read_pool_id(child_spec)


def test_pod_template_child_matches_deploy(monkeypatch, tmp_path):
    parent_spec, child_spec = _deploy("v1", tasks=(pod_parent, pod_child))
    template = parent_spec.task_template
    pod_spec = MessageToDict(child_spec.task_template.k8s_pod.pod_spec)
    assert FLYTE_SYS_PATH in {e["name"] for c in pod_spec["containers"] for e in c.get("env", [])}

    bundle, version = _run_a0(_primary_container_args(template.k8s_pod, template.config), monkeypatch, tmp_path)
    assert _serialize_child_in_task(bundle, version, task=pod_child) == _read_pool_id(child_spec)


def test_unchanged_code_keeps_pool_across_versions(monkeypatch, tmp_path):
    p1, c1 = _deploy("v1")
    p2, c2 = _deploy("v2")
    assert _read_pool_id(c1) == _read_pool_id(c2)
    assert _pool_id_in_task(p1, monkeypatch, tmp_path) == _pool_id_in_task(p2, monkeypatch, tmp_path)


def test_changed_code_gets_new_pool(monkeypatch, tmp_path):
    p1, c1 = _deploy("v1")
    p2, c2 = _deploy("v1", CHANGED_BUNDLE)
    assert _read_pool_id(c1) != _read_pool_id(c2)
    assert _pool_id_in_task(p1, monkeypatch, tmp_path) != _pool_id_in_task(p2, monkeypatch, tmp_path)


def test_pkl_child_pool_id_is_stable_across_runs():
    # In pkl mode the controller pickles each child into the run's own base dir; the pool id must
    # follow the pickle's content, not that run-specific location.
    def make_pkl_bundle(run: str) -> CodeBundle:
        return CodeBundle(pkl=f"s3://bucket/{run}/a0/9b1f0c.pkl.gz", destination=".", computed_version="9b1f0c")

    assert _serialize_child_in_task(make_pkl_bundle("run-1"), "v1") == _serialize_child_in_task(
        make_pkl_bundle("run-2"), "v1"
    )
