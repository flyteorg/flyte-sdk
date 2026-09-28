"""CPU-only unit tests for torch_profile — no GPU or cluster needed."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from flyteplugins.pytorch import torch_profile
from flyteplugins.pytorch._profile import _render_html, _summary_rows


@pytest.fixture
def report_env(monkeypatch):
    """Fake an active task report: a real Tab to capture HTML, mocked flush and context."""
    import flyte._context
    import flyte.report
    from flyte.report._report import Tab

    tab = Tab("t")
    flush = MagicMock()
    flush.aio = AsyncMock()
    ctx = MagicMock()
    ctx.get_report.return_value = MagicMock()
    monkeypatch.setattr(flyte._context, "internal_ctx", lambda: ctx)
    monkeypatch.setattr(flyte.report, "get_tab", lambda name, **kw: tab)
    monkeypatch.setattr(flyte.report, "flush", flush)
    return tab, flush


def _workload():
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    return a @ b


def test_dual_protocol():
    cm = torch_profile()
    assert hasattr(cm, "__enter__") and hasattr(cm, "__exit__")
    assert hasattr(cm, "__aenter__") and hasattr(cm, "__aexit__")


def test_sync_renders_tab(report_env):
    tab, flush = report_env
    with torch_profile() as prof:
        _workload()
    assert prof is not None
    out = tab.get_html()
    assert "ui.perfetto.dev" in out
    assert "aten::" in out
    assert "PT_B64_" in out  # embedded base64 trace
    assert "ptDownload_" in out
    flush.assert_called_once()


def test_async_protocol(report_env):
    tab, flush = report_env

    async def go():
        async with torch_profile() as prof:
            _workload()
            assert prof is not None

    asyncio.run(go())
    assert "ui.perfetto.dev" in tab.get_html()
    flush.aio.assert_awaited_once()


def test_oversize_skips_embed(report_env, monkeypatch):
    tab, _ = report_env
    from flyte.io import File

    f = MagicMock()
    f.path = "s3://bucket/traces/trace.json"
    monkeypatch.setattr(File, "from_local_sync", MagicMock(return_value=f))
    with torch_profile(max_embed_mb=0):
        _workload()
    out = tab.get_html()
    assert "PT_B64_" not in out
    assert "pt-frame-" not in out
    assert "s3://bucket/traces/trace.json" in out


def test_body_exception_not_suppressed(report_env):
    tab, _ = report_env
    with pytest.raises(ValueError, match="boom"):
        with torch_profile():
            _workload()
            raise ValueError("boom")
    # profile still rendered
    assert "aten::" in tab.get_html()


def test_no_report_context_warns(monkeypatch, caplog):
    import flyte._context

    ctx = MagicMock()
    ctx.get_report.return_value = None
    monkeypatch.setattr(flyte._context, "internal_ctx", lambda: ctx)
    with torch_profile():
        _workload()
    assert any("report=True" in r.message for r in caplog.records)


def test_render_html_escapes():
    rows = [
        {
            "name": "<script>alert(1)</script>",
            "count": 1,
            "self_cpu_us": 10.0,
            "cpu_us": 10.0,
            "self_device_us": 5.0,
            "device_us": 5.0,
            "self_device_mem": 0.0,
            "cpu_mem": 0.0,
        }
    ]
    out = _render_html("t", rows, None, 0)
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_summary_rows_from_real_profile():
    import torch.profiler

    with torch.profiler.profile() as prof:
        _workload()
    rows = _summary_rows(prof.key_averages())
    assert rows
    assert any("aten::" in r["name"] for r in rows)
    assert all(r["count"] >= 1 for r in rows)
