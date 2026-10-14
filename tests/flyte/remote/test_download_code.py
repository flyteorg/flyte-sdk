"""Tests for Run.download_code / Action.download_code and the code bundle they resolve."""

from __future__ import annotations

import pathlib
import tarfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.core import tasks_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.workflow import run_definition_pb2
from google.protobuf.struct_pb2 import Struct

from flyte._internal.runtime.task_serde import extract_code_bundle
from flyte.remote._action import Action, ActionDetails
from flyte.remote._run import Run, RunDetails

TGZ_URI = "s3://bucket/org/proj/fast123abc.tar.gz"
PKL_URI = "s3://bucket/org/proj/code_bundle.pkl.gz"


def _make_run(run_name: str = "run-1", action_name: str = "a0") -> Run:
    run_id = identifier_pb2.RunIdentifier(name=run_name)
    action_id = identifier_pb2.ActionIdentifier(run=run_id, name=action_name)
    return Run(pb2=run_definition_pb2.Run(action=run_definition_pb2.Action(id=action_id)))


def _make_action(run_name: str = "run-1", action_name: str = "n0-child") -> Action:
    run_id = identifier_pb2.RunIdentifier(name=run_name)
    action_id = identifier_pb2.ActionIdentifier(run=run_id, name=action_name)
    return Action(pb2=run_definition_pb2.Action(id=action_id))


def _details_with_container_args(*args: str, attempts: int = 1) -> ActionDetails:
    pb2 = run_definition_pb2.ActionDetails()
    pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
    pb2.status.attempts = attempts
    pb2.task.task_template.container.args.extend(args)
    return ActionDetails(pb2=pb2)


def _tgz_details(attempts: int = 1) -> ActionDetails:
    return _details_with_container_args("--tgz", TGZ_URI, "--dest", ".", "--version", "123abc", attempts=attempts)


def _signed_url_response(url: str = "https://signed/fast123abc.tar.gz"):
    resp = dataproxy_service_pb2.CreateDownloadLinkResponse()
    resp.pre_signed_urls.signed_url.append(url)
    return resp


def _make_tarball(tmp_path: pathlib.Path, *names: str) -> bytes:
    """A real gzipped tarball, so extraction is exercised end to end."""
    src = tmp_path / "src"
    src.mkdir()
    for name in names:
        f = src / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"# {name}\n")
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in names:
            tar.add(src / name, arcname=name)
    return archive.read_bytes()


class _FakeStreamResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self):
        yield self._payload


def _patch_httpx(payload: bytes):
    """Patch httpx.AsyncClient so stream() yields the given bytes without network."""
    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=_FakeStreamResponse(payload))
    stream_ctx.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.stream = MagicMock(return_value=stream_ctx)
    client_ctx = MagicMock()
    client_ctx.__aenter__ = AsyncMock(return_value=client)
    client_ctx.__aexit__ = AsyncMock(return_value=False)
    return patch("flyte.remote._code.httpx.AsyncClient", return_value=client_ctx), client


# --------------------------------------------------------------------------------------
# bundle resolution
# --------------------------------------------------------------------------------------


def test_extract_code_bundle_from_container_args():
    details = _tgz_details()
    bundle = details.code_bundle
    assert bundle is not None
    assert bundle.tgz == TGZ_URI
    assert bundle.pkl is None
    assert bundle.computed_version == "123abc"
    assert bundle.destination == "."


def test_extract_code_bundle_from_k8s_pod_primary_container():
    """A pod-template task keeps its args in the pod spec, not in template.container."""
    pod_spec = Struct()
    pod_spec.update(
        {
            "containers": [
                {"name": "sidecar", "args": ["serve"]},
                {"name": "primary", "args": ["--tgz", TGZ_URI, "--dest", ".", "--version", "v9"]},
            ]
        }
    )
    spec = run_definition_pb2.ActionDetails().task
    spec.task_template.k8s_pod.CopyFrom(tasks_pb2.K8sPod(pod_spec=pod_spec, primary_container_name="primary"))

    bundle = extract_code_bundle(spec)
    assert bundle is not None
    assert bundle.tgz == TGZ_URI
    assert bundle.computed_version == "v9"


def test_extract_code_bundle_falls_back_to_metadata_uri():
    """An image-baked entrypoint with no --tgz arg still records the bundle in metadata."""
    spec = run_definition_pb2.ActionDetails().task
    spec.task_template.metadata.code_bundle_uri = TGZ_URI
    bundle = extract_code_bundle(spec)
    assert bundle is not None and bundle.tgz == TGZ_URI and bundle.pkl is None

    spec2 = run_definition_pb2.ActionDetails().task
    spec2.task_template.metadata.code_bundle_uri = PKL_URI
    bundle2 = extract_code_bundle(spec2)
    assert bundle2 is not None and bundle2.pkl == PKL_URI and bundle2.tgz is None


def test_code_bundle_is_none_without_a_bundle():
    pb2 = run_definition_pb2.ActionDetails()
    pb2.task.task_template.container.args.extend(["a0", "--run-name", "run-1"])
    assert ActionDetails(pb2=pb2).code_bundle is None


@pytest.mark.asyncio
async def test_run_code_bundle_reads_the_root_action():
    run = _make_run()
    # A terminal, already-cached RunDetails is served from the Run without hitting the API.
    run._details = RunDetails(pb2=run_definition_pb2.RunDetails(action=_tgz_details().pb2))

    bundle = await run.code_bundle.aio()

    assert bundle is not None and bundle.tgz == TGZ_URI


# --------------------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_code_extracts_tarball_from_signed_url(tmp_path):
    run = _make_run()
    payload = _make_tarball(tmp_path, "workflows/main.py", "README.md")
    client = MagicMock()
    create = AsyncMock(return_value=_signed_url_response())
    client.dataproxy_service.create_download_link = create
    httpx_patch, _ = _patch_httpx(payload)
    dest = tmp_path / "out"

    with (
        patch("flyte.remote._code.ensure_client"),
        patch("flyte.remote._code.get_client", return_value=client),
        patch.object(run.action.__class__, "details", new=AsyncMock(return_value=_tgz_details(attempts=2))),
        httpx_patch,
    ):
        path = await run.download_code.aio(dest=dest)

    assert path == dest
    assert (dest / "workflows" / "main.py").read_text() == "# workflows/main.py\n"
    assert (dest / "README.md").exists()
    # The archive is kept alongside the extracted source, as it is on the worker.
    assert (dest / "fast123abc.tar.gz").exists()

    sent = create.await_args[0][0]
    assert sent.artifact_type == dataproxy_service_pb2.ARTIFACT_TYPE_CODE_BUNDLE
    assert sent.action_attempt_id.action_id == run.action.action_id
    assert sent.action_attempt_id.attempt == 2


@pytest.mark.asyncio
async def test_download_code_without_extract_keeps_the_archive(tmp_path):
    run = _make_run()
    payload = _make_tarball(tmp_path, "main.py")
    client = MagicMock()
    client.dataproxy_service.create_download_link = AsyncMock(return_value=_signed_url_response())
    httpx_patch, _ = _patch_httpx(payload)
    dest = tmp_path / "out"

    with (
        patch("flyte.remote._code.ensure_client"),
        patch("flyte.remote._code.get_client", return_value=client),
        patch.object(run.action.__class__, "details", new=AsyncMock(return_value=_tgz_details())),
        httpx_patch,
    ):
        path = await run.download_code.aio(dest=dest, extract=False)

    assert path == dest / "fast123abc.tar.gz"
    assert path.read_bytes() == payload
    assert not (dest / "main.py").exists()


@pytest.mark.asyncio
async def test_download_code_falls_back_to_object_store(tmp_path):
    """Backends that don't sign code bundles (OSS Flyte) leave the blob store as the only route."""
    run = _make_run()
    payload = _make_tarball(tmp_path, "main.py")
    client = MagicMock()
    client.dataproxy_service.create_download_link = AsyncMock(side_effect=RuntimeError("unimplemented"))
    dest = tmp_path / "out"

    async def fake_get(uri: str, target: str):
        assert uri == TGZ_URI
        pathlib.Path(target).write_bytes(payload)

    with (
        patch("flyte.remote._code.ensure_client"),
        patch("flyte.remote._code.get_client", return_value=client),
        patch.object(run.action.__class__, "details", new=AsyncMock(return_value=_tgz_details())),
        patch("flyte.storage.get", new=AsyncMock(side_effect=fake_get)),
    ):
        path = await run.download_code.aio(dest=dest)

    assert path == dest
    assert (dest / "main.py").read_text() == "# main.py\n"


@pytest.mark.asyncio
async def test_download_code_reports_the_data_proxy_error_when_both_routes_fail(tmp_path):
    run = _make_run()
    client = MagicMock()
    client.dataproxy_service.create_download_link = AsyncMock(
        side_effect=RuntimeError("code bundle is a pickled archive")
    )

    with (
        patch("flyte.remote._code.ensure_client"),
        patch("flyte.remote._code.get_client", return_value=client),
        patch.object(run.action.__class__, "details", new=AsyncMock(return_value=_tgz_details())),
        patch("flyte.storage.get", new=AsyncMock(side_effect=OSError("no credentials"))),
    ):
        with pytest.raises(RuntimeError, match="pickled archive") as exc:
            await run.download_code.aio(dest=tmp_path / "out")

    assert isinstance(exc.value.__cause__, OSError)
    # A failed download leaves no half-written archive behind.
    assert not (tmp_path / "out" / "fast123abc.tar.gz").exists()


@pytest.mark.asyncio
async def test_download_code_raises_when_the_task_has_no_bundle(tmp_path):
    run = _make_run()
    pb2 = run_definition_pb2.ActionDetails()
    pb2.status.attempts = 1
    details = ActionDetails(pb2=pb2)

    with patch.object(run.action.__class__, "details", new=AsyncMock(return_value=details)):
        with pytest.raises(RuntimeError, match="No code bundle is associated with action 'a0' in run 'run-1'"):
            await run.download_code.aio(dest=tmp_path / "out")


@pytest.mark.asyncio
async def test_action_download_code_uses_its_own_action_id(tmp_path):
    """A nested action can be packaged separately from the run's root action."""
    action = _make_action(action_name="n0-child")
    payload = _make_tarball(tmp_path, "child.py")
    client = MagicMock()
    create = AsyncMock(return_value=_signed_url_response())
    client.dataproxy_service.create_download_link = create
    httpx_patch, _ = _patch_httpx(payload)

    with (
        patch("flyte.remote._code.ensure_client"),
        patch("flyte.remote._code.get_client", return_value=client),
        patch.object(action.__class__, "details", new=AsyncMock(return_value=_tgz_details(attempts=3))),
        httpx_patch,
    ):
        path = await action.download_code.aio(dest=tmp_path / "out", attempt=1)

    assert (path / "child.py").exists()
    sent = create.await_args[0][0]
    assert sent.action_attempt_id.action_id == action.action_id
    # An explicit attempt wins over the latest one.
    assert sent.action_attempt_id.attempt == 1


@pytest.mark.asyncio
async def test_download_code_defaults_destination_to_the_run_name(tmp_path, monkeypatch):
    run = _make_run(run_name="my-run")
    payload = _make_tarball(tmp_path, "main.py")
    monkeypatch.chdir(tmp_path)
    client = MagicMock()
    client.dataproxy_service.create_download_link = AsyncMock(return_value=_signed_url_response())
    httpx_patch, _ = _patch_httpx(payload)

    with (
        patch("flyte.remote._code.ensure_client"),
        patch("flyte.remote._code.get_client", return_value=client),
        patch.object(run.action.__class__, "details", new=AsyncMock(return_value=_tgz_details())),
        httpx_patch,
    ):
        path = await run.download_code.aio()

    assert path == tmp_path / "my-run"
    assert (path / "main.py").exists()
