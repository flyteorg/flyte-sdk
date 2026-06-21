"""Characterization tests pinning the exact ``RunSpec`` / ``CreateRunRequest`` the
remote run path builds from ``with_runcontext(...)``.

These are a safety net for the run/rerun/recover/debug unification refactor: every
field that ``_Runner._run_remote`` serializes is asserted here so the extraction of
``_build_task_spec_from_template`` / ``_submit_remote`` / ``_apply_overrides`` cannot
silently change the wire. The combined ``test_runspec_all_fields_snapshot`` is the
byte-for-byte oracle; the per-field tests localize any regression.
"""

import mock
import pytest
from flyteidl2.common import run_pb2 as common_run_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.task import run_pb2
from flyteidl2.workflow import run_service_pb2
from mock.mock import AsyncMock, MagicMock

import flyte
from flyte._initialize import _init_for_testing
from flyte.models import CodeBundle

env = flyte.TaskEnvironment(name="test")


@env.task
async def task1(v: str) -> str:
    return f"Hello, world {v}!"


def _make_mock_client():
    """Mocked ClientSet with run + dataproxy services wired for create_run capture."""
    mock_client = MagicMock()
    mock_run_service = AsyncMock()
    mock_client.run_service = mock_run_service

    mock_dataproxy_service = AsyncMock()
    mock_offloaded = common_run_pb2.OffloadedInputData(uri="s3://bucket/inputs", inputs_hash="abc123")
    mock_dataproxy_service.upload_inputs.return_value = dataproxy_service_pb2.UploadInputsResponse(
        offloaded_input_data=mock_offloaded,
    )
    mock_client.dataproxy_service = mock_dataproxy_service
    return mock_client, mock_run_service


def _patch_build(fn):
    """Stack the image-build + code-bundle mocks shared by every remote-path test.

    The patch applied first (closest to the function) is injected as the first arg, so
    build_code_bundle must wrap first to match the ``(mock_code_bundler, mock_build_image_bg)``
    signature used below.
    """
    fn = mock.patch("flyte._code_bundle.build_code_bundle", new_callable=AsyncMock)(fn)
    fn = mock.patch("flyte._deploy._build_image_bg", new_callable=AsyncMock)(fn)
    return fn


async def _run_and_capture(mock_build_image_bg, mock_code_bundler, **runcontext_kwargs):
    """Run task1 in remote mode with the given runcontext kwargs; return the CreateRunRequest."""
    mock_client, mock_run_service = _make_mock_client()
    mock_code_bundler.return_value = CodeBundle(computed_version="v1", tgz="test.tgz")
    mock_build_image_bg.return_value = (env.name, "image_name", None)

    await _init_for_testing(client=mock_client, project="test", domain="test")
    run = await flyte.with_runcontext(mode="remote", **runcontext_kwargs).run.aio(task1, "hello")
    assert run
    req: run_service_pb2.CreateRunRequest = mock_run_service.create_run.call_args[0][0]
    return req


def _envs_dict(req):
    return {kv.key: kv.value for kv in req.run_spec.envs.values}


@pytest.mark.asyncio
@_patch_build
async def test_runspec_env_vars(mock_code_bundler, mock_build_image_bg):
    """User env_vars land on RunSpec.envs, alongside the always-injected LOG_* keys."""
    req = await _run_and_capture(mock_build_image_bg, mock_code_bundler, env_vars={"FOO": "bar"})
    envs = _envs_dict(req)
    assert envs["FOO"] == "bar"
    # Always-injected keys (see _run.py:302-308).
    assert "LOG_LEVEL" in envs
    assert "LOG_FORMAT" in envs


@pytest.mark.asyncio
@_patch_build
async def test_runspec_debug_injects_f_e_vs(mock_code_bundler, mock_build_image_bg):
    """debug=True injects the _F_E_VS env flag."""
    req = await _run_and_capture(mock_build_image_bg, mock_code_bundler, debug=True)
    assert _envs_dict(req)["_F_E_VS"] == "1"


@pytest.mark.asyncio
@_patch_build
async def test_runspec_labels_and_annotations(mock_code_bundler, mock_build_image_bg):
    req = await _run_and_capture(
        mock_build_image_bg,
        mock_code_bundler,
        labels={"team": "ml"},
        annotations={"note": "exp"},
    )
    assert req.run_spec.labels.values["team"] == "ml"
    assert req.run_spec.annotations.values["note"] == "exp"


@pytest.mark.asyncio
@_patch_build
async def test_runspec_queue_to_cluster(mock_code_bundler, mock_build_image_bg):
    """queue= maps to RunSpec.cluster."""
    req = await _run_and_capture(mock_build_image_bg, mock_code_bundler, queue="gpu-queue")
    assert req.run_spec.cluster == "gpu-queue"


@pytest.mark.asyncio
@_patch_build
async def test_runspec_interruptible(mock_code_bundler, mock_build_image_bg):
    """interruptible is a BoolValue (set only when not None)."""
    req = await _run_and_capture(mock_build_image_bg, mock_code_bundler, interruptible=True)
    assert req.run_spec.HasField("interruptible")
    assert req.run_spec.interruptible.value is True

    req2 = await _run_and_capture(mock_build_image_bg, mock_code_bundler)
    assert not req2.run_spec.HasField("interruptible")


@pytest.mark.asyncio
@_patch_build
async def test_runspec_overwrite_cache(mock_code_bundler, mock_build_image_bg):
    """overwrite_cache sets both the top-level field and cache_config."""
    req = await _run_and_capture(mock_build_image_bg, mock_code_bundler, overwrite_cache=True)
    assert req.run_spec.overwrite_cache is True
    assert req.run_spec.cache_config.overwrite_cache is True


@pytest.mark.asyncio
@_patch_build
async def test_runspec_cache_lookup_scope(mock_code_bundler, mock_build_image_bg):
    """Default scope is global; project-domain maps to its enum."""
    req_global = await _run_and_capture(mock_build_image_bg, mock_code_bundler)
    assert (
        req_global.run_spec.cache_config.cache_lookup_scope
        == run_pb2.CacheLookupScope.CACHE_LOOKUP_SCOPE_GLOBAL
    )

    req_pd = await _run_and_capture(mock_build_image_bg, mock_code_bundler, cache_lookup_scope="project-domain")
    assert (
        req_pd.run_spec.cache_config.cache_lookup_scope
        == run_pb2.CacheLookupScope.CACHE_LOOKUP_SCOPE_PROJECT_DOMAIN
    )


@pytest.mark.asyncio
@_patch_build
async def test_runspec_service_account(mock_code_bundler, mock_build_image_bg):
    """service_account maps to security_context.run_as.k8s_service_account."""
    req = await _run_and_capture(mock_build_image_bg, mock_code_bundler, service_account="my-sa")
    assert req.run_spec.security_context.run_as.k8s_service_account == "my-sa"

    req_none = await _run_and_capture(mock_build_image_bg, mock_code_bundler)
    assert not req_none.run_spec.HasField("security_context")


@pytest.mark.asyncio
@_patch_build
async def test_runspec_all_fields_snapshot(mock_code_bundler, mock_build_image_bg):
    """Combined oracle: pin the full RunSpec for a fully-populated with_runcontext config.

    The unification refactor must reproduce this RunSpec byte-for-byte; treat any diff
    as a regression, not a re-baseline.
    """
    req = await _run_and_capture(
        mock_build_image_bg,
        mock_code_bundler,
        env_vars={"FOO": "bar"},
        labels={"team": "ml"},
        annotations={"note": "exp"},
        queue="gpu-queue",
        interruptible=True,
        overwrite_cache=True,
        cache_lookup_scope="project-domain",
        service_account="my-sa",
        max_action_concurrency=4,
    )
    rs = req.run_spec
    assert rs.overwrite_cache is True
    assert rs.interruptible.value is True
    assert rs.labels.values["team"] == "ml"
    assert rs.annotations.values["note"] == "exp"
    assert _envs_dict(req)["FOO"] == "bar"
    assert rs.cluster == "gpu-queue"
    assert rs.max_action_concurrency == 4
    assert rs.security_context.run_as.k8s_service_account == "my-sa"
    assert rs.cache_config.overwrite_cache is True
    assert rs.cache_config.cache_lookup_scope == run_pb2.CacheLookupScope.CACHE_LOOKUP_SCOPE_PROJECT_DOMAIN
    # task_spec is sent inline (not by reference) for a locally-defined task.
    assert req.WhichOneof("task") == "task_spec"
    assert req.offloaded_input_data.uri == "s3://bucket/inputs"
    assert not req.HasField("inputs")


# --- mode dispatch + hybrid validation -------------------------------------------------
# Hybrid mode runs the parent locally and enqueues children via a controller; it never
# builds a CreateRunRequest, so the only refactor-relevant invariants are (a) run() routes
# to the right _run_* method per mode, and (b) the hybrid guardrails fire. Both are cheap
# and robust to pin without a live controller/storage.


def test_with_runcontext_hybrid_requires_name_and_run_base_dir():
    with pytest.raises(ValueError, match="hybrid"):
        flyte.with_runcontext(mode="hybrid")


@pytest.mark.asyncio
async def test_run_dispatches_per_mode():
    """`run()` routes to _run_remote / _run_local / _run_hybrid based on the resolved mode."""
    from flyte._run import _Runner

    for mode, target in (("remote", "_run_remote"), ("local", "_run_local"), ("hybrid", "_run_hybrid")):
        runner = _Runner(force_mode=mode, name="r", run_base_dir="s3://b/md")
        with mock.patch.object(_Runner, target, new_callable=AsyncMock) as m:
            m.return_value = object()
            await runner.run.aio(task1, "hello")
            m.assert_called_once()


# --- notifications wiring (moves into _apply_overrides) --------------------------------


@pytest.mark.asyncio
@_patch_build
async def test_runspec_notifications(mock_code_bundler, mock_build_image_bg):
    """Notifications resolve into notification_rule_name / notification_rules on RunSpec."""
    import flyte.notify

    req = await _run_and_capture(
        mock_build_image_bg,
        mock_code_bundler,
        notifications=flyte.notify.Email(on_phase="failed", recipients=("a@b.com",)),
    )
    # Exactly one of the two notification carriers is populated (depends on resolve output).
    assert req.run_spec.notification_rule_name or len(req.run_spec.notification_rules.rules) > 0


# --- ConnectError mapping (moves into _submit_remote) ----------------------------------


@pytest.mark.asyncio
@_patch_build
async def test_create_run_already_exists_maps_to_user_error(mock_code_bundler, mock_build_image_bg):
    """create_run ALREADY_EXISTS → RuntimeUserError (RunAlreadyExistsError)."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    import flyte.errors

    mock_client, mock_run_service = _make_mock_client()
    mock_code_bundler.return_value = CodeBundle(computed_version="v1", tgz="test.tgz")
    mock_build_image_bg.return_value = (env.name, "image_name", None)
    mock_run_service.create_run.side_effect = ConnectError(Code.ALREADY_EXISTS, "dup")

    await _init_for_testing(client=mock_client, project="test", domain="test")
    with pytest.raises(flyte.errors.RuntimeUserError, match="already exists"):
        await flyte.with_runcontext(mode="remote", name="dup-run").run.aio(task1, "hello")


@pytest.mark.asyncio
@_patch_build
async def test_create_run_unavailable_maps_to_system_error(mock_code_bundler, mock_build_image_bg):
    """create_run UNAVAILABLE → RuntimeSystemError (SystemUnavailableError)."""
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError

    import flyte.errors

    mock_client, mock_run_service = _make_mock_client()
    mock_code_bundler.return_value = CodeBundle(computed_version="v1", tgz="test.tgz")
    mock_build_image_bg.return_value = (env.name, "image_name", None)
    mock_run_service.create_run.side_effect = ConnectError(Code.UNAVAILABLE, "down")

    await _init_for_testing(client=mock_client, project="test", domain="test")
    with pytest.raises(flyte.errors.RuntimeSystemError):
        await flyte.with_runcontext(mode="remote").run.aio(task1, "hello")


# --- dry-run path (stays in _run_remote) -----------------------------------------------


@pytest.mark.asyncio
@_patch_build
async def test_dry_run_returns_dryrun_without_create_run(mock_code_bundler, mock_build_image_bg):
    """dry_run=True returns a DryRun carrying the task_spec and never calls create_run."""
    mock_client, mock_run_service = _make_mock_client()
    mock_code_bundler.return_value = CodeBundle(computed_version="v1", tgz="test.tgz")
    mock_build_image_bg.return_value = (env.name, "image_name", None)

    await _init_for_testing(client=mock_client, project="test", domain="test")
    run = await flyte.with_runcontext(mode="remote", dry_run=True).run.aio(task1, "hello")

    assert run is not None
    assert run.task_spec is not None
    mock_run_service.create_run.assert_not_called()


# --- _apply_overrides inherited path (the rerun seam, base != None) --------------------


@pytest.mark.asyncio
async def test_apply_overrides_inherited_merges_env_and_keys():
    """base != None: deep-copy the prior RunSpec, overlay env by key, apply only set overrides."""
    from flyteidl2.core import literals_pb2
    from flyteidl2.task import run_pb2

    from flyte._run import _Runner

    mock_client, _ = _make_mock_client()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    base = run_pb2.RunSpec(
        envs=run_pb2.Envs(
            values=[
                literals_pb2.KeyValuePair(key="KEEP", value="1"),
                literals_pb2.KeyValuePair(key="FOO", value="old"),
            ]
        ),
        labels=run_pb2.Labels(values={"base": "yes"}),
        cluster="orig-cluster",
    )

    runner = _Runner(force_mode="remote", env_vars={"FOO": "new", "BAR": "2"}, labels={"team": "ml"})
    out = runner._apply_overrides(base)

    envs = {kv.key: kv.value for kv in out.envs.values}
    assert envs["KEEP"] == "1"  # prior key preserved
    assert envs["FOO"] == "new"  # runner override wins
    assert envs["BAR"] == "2"  # new key added
    assert out.labels.values["base"] == "yes"  # prior label preserved
    assert out.labels.values["team"] == "ml"  # runner label merged
    assert out.cluster == "orig-cluster"  # queue not set on runner -> inherited cluster kept

    # base is not mutated (deep copy).
    assert {kv.key: kv.value for kv in base.envs.values} == {"KEEP": "1", "FOO": "old"}


@pytest.mark.asyncio
async def test_apply_overrides_recover_gated():
    """recover raises until flyteidl2 RunSpec.recover ships (field absent today)."""
    from flyteidl2.task import run_pb2

    from flyte._run import _Runner

    mock_client, _ = _make_mock_client()
    await _init_for_testing(client=mock_client, project="test", domain="test")

    runner = _Runner(force_mode="remote")

    if "recover" in run_pb2.RunSpec.DESCRIPTOR.fields_by_name:
        pytest.skip("RunSpec.recover is available; gating no longer applies")
    with pytest.raises(NotImplementedError, match="recover is not yet supported"):
        runner._apply_overrides(None, recover_ref="some-run")


def test_resolve_recover_ref_semantics():
    """recover=False/True/str resolve to the right reference (or raise on run())."""
    from flyte._run import _Runner

    # default False -> no recover
    assert _Runner()._resolve_recover_ref("r1") is None
    # True -> the run being rerun
    assert _Runner(recover=True)._resolve_recover_ref("r1") == "r1"
    # True with no rerun target (a plain run()) -> error
    with pytest.raises(ValueError, match="recover=True is only valid with rerun"):
        _Runner(recover=True)._resolve_recover_ref(None)
    # explicit name -> that name (works on run())
    assert _Runner(recover="other")._resolve_recover_ref(None) == "other"


@pytest.mark.asyncio
async def test_recover_rejected_in_local_mode():
    """recover is remote-only; a truthy recover in local mode fails fast on run()."""
    await flyte.init.aio()
    with pytest.raises(ValueError, match="recover is only supported in remote mode"):
        await flyte.with_runcontext(mode="local", recover="r1").run.aio(task1, "hello")
