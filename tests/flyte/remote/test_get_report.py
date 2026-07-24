"""Tests for Run.get_report."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from flyteidl2.common import identifier_pb2, phase_pb2
from flyteidl2.dataproxy import dataproxy_service_pb2
from flyteidl2.workflow import run_definition_pb2

from flyte.remote._action import ActionDetails
from flyte.remote._run import Run


def _make_run(run_name: str = "run-1", action_name: str = "a0") -> Run:
    run_id = identifier_pb2.RunIdentifier(name=run_name)
    action_id = identifier_pb2.ActionIdentifier(run=run_id, name=action_name)
    action_pb2 = run_definition_pb2.Action(id=action_id)
    run_pb2 = run_definition_pb2.Run(action=action_pb2)
    return Run(pb2=run_pb2)


def _make_done_details(attempts: int = 1) -> ActionDetails:
    pb2 = run_definition_pb2.ActionDetails()
    pb2.status.phase = phase_pb2.ACTION_PHASE_SUCCEEDED
    pb2.status.attempts = attempts
    return ActionDetails(pb2=pb2)


def _download_link_response(
    url: str = "https://signed/report.html",
) -> dataproxy_service_pb2.CreateDownloadLinkResponse:
    resp = dataproxy_service_pb2.CreateDownloadLinkResponse()
    resp.pre_signed_urls.signed_url.append(url)
    return resp


class _FakeHTTPResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        return None


def _patch_httpx(text: str):
    """Patch httpx.AsyncClient so get() returns the given text without network."""
    get_mock = AsyncMock(return_value=_FakeHTTPResponse(text))
    async_client = MagicMock()
    async_client.get = get_mock
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=async_client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return patch("flyte.remote._run.httpx.AsyncClient", return_value=ctx), get_mock


@pytest.mark.asyncio
async def test_get_report_downloads_html_from_signed_url():
    run = _make_run()
    client = MagicMock()
    client.dataproxy_service.create_download_link = AsyncMock(return_value=_download_link_response())

    httpx_patch, get_mock = _patch_httpx("<h1>report</h1>")

    with (
        patch("flyte.remote._run.ensure_client"),
        patch("flyte.remote._run.get_client", return_value=client),
        patch.object(run.action.__class__, "details", new=AsyncMock(return_value=_make_done_details(attempts=1))),
        httpx_patch,
    ):
        html = await run.get_report.aio()

    assert html == "<h1>report</h1>"
    get_mock.assert_awaited_once_with("https://signed/report.html")


@pytest.mark.asyncio
async def test_get_report_uses_latest_attempt_and_report_artifact_type():
    run = _make_run()
    client = MagicMock()
    create = AsyncMock(return_value=_download_link_response())
    client.dataproxy_service.create_download_link = create

    httpx_patch, _ = _patch_httpx("<html></html>")

    with (
        patch("flyte.remote._run.ensure_client"),
        patch("flyte.remote._run.get_client", return_value=client),
        patch.object(run.action.__class__, "details", new=AsyncMock(return_value=_make_done_details(attempts=3))),
        httpx_patch,
    ):
        await run.get_report.aio()

    create.assert_awaited_once()
    sent = create.await_args[0][0]
    assert sent.artifact_type == dataproxy_service_pb2.ARTIFACT_TYPE_REPORT
    assert sent.action_attempt_id.attempt == 3
    assert sent.action_attempt_id.action_id == run.action.action_id


@pytest.mark.asyncio
async def test_get_report_respects_explicit_attempt():
    run = _make_run()
    client = MagicMock()
    create = AsyncMock(return_value=_download_link_response())
    client.dataproxy_service.create_download_link = create
    details_mock = AsyncMock(return_value=_make_done_details(attempts=5))

    httpx_patch, _ = _patch_httpx("<html></html>")

    with (
        patch("flyte.remote._run.ensure_client"),
        patch("flyte.remote._run.get_client", return_value=client),
        patch.object(run.action.__class__, "details", new=details_mock),
        httpx_patch,
    ):
        await run.get_report.aio(attempt=2)

    # Explicit attempt short-circuits the details() lookup.
    details_mock.assert_not_awaited()
    sent = create.await_args[0][0]
    assert sent.action_attempt_id.attempt == 2


@pytest.mark.asyncio
async def test_get_report_raises_when_no_signed_url():
    run = _make_run()
    client = MagicMock()
    client.dataproxy_service.create_download_link = AsyncMock(
        return_value=dataproxy_service_pb2.CreateDownloadLinkResponse()
    )

    with (
        patch("flyte.remote._run.ensure_client"),
        patch("flyte.remote._run.get_client", return_value=client),
        patch.object(run.action.__class__, "details", new=AsyncMock(return_value=_make_done_details(attempts=1))),
    ):
        with pytest.raises(RuntimeError, match="No report is available"):
            await run.get_report.aio()
