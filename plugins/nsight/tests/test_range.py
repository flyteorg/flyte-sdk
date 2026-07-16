"""Tests for the nsys.range region context manager (both `with` and `async with`).

`@nsys_profile` accepts async and sync task bodies alike, so `nsys.range(...)` must drive collection
under both `async with` (async task) and plain `with` (sync task). These exercise both protocols and
the off-nsys no-op with the nsys CLI stubbed out — no GPU required.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from flyteplugins.nsight import _capture as capture
from flyteplugins.nsight import _control as ctl
from flyteplugins.nsight import nsys


class TestDualProtocol:
    def test_range_supports_both_protocols(self):
        r = nsys.range("region")
        # One object, usable as either a sync or an async context manager.
        assert hasattr(r, "__enter__") and hasattr(r, "__exit__")
        assert hasattr(r, "__aenter__") and hasattr(r, "__aexit__")

    def test_profile_is_range(self):
        assert nsys.profile is nsys.range


class TestSyncProtocol:
    def test_noop_when_not_under_nsys(self, monkeypatch):
        monkeypatch.setattr(ctl, "under_nsys", lambda: False)
        start = MagicMock()
        monkeypatch.setattr(ctl, "start_collection_sync", start)

        ran = False
        with nsys.range("region"):
            ran = True

        assert ran is True
        start.assert_not_called()  # no collection attempted off-nsys

    def test_collects_when_under_nsys(self, monkeypatch):
        monkeypatch.setattr(ctl, "under_nsys", lambda: True)
        start = MagicMock(return_value="/tmp/nsys/a0/region.nsys-rep")
        stop = MagicMock()
        finalize = MagicMock(return_value={"kernel_launches": 5})
        monkeypatch.setattr(ctl, "start_collection_sync", start)
        monkeypatch.setattr(ctl, "stop_collection_sync", stop)
        monkeypatch.setattr(capture, "finalize_sync", finalize)

        with nsys.range("region"):
            pass

        start.assert_called_once_with("region")
        stop.assert_called_once()
        finalize.assert_called_once()

    def test_finalizes_when_body_raises(self, monkeypatch):
        monkeypatch.setattr(ctl, "under_nsys", lambda: True)
        monkeypatch.setattr(ctl, "start_collection_sync", MagicMock(return_value="/tmp/x.nsys-rep"))
        stop = MagicMock()
        finalize = MagicMock(return_value={})
        monkeypatch.setattr(ctl, "stop_collection_sync", stop)
        monkeypatch.setattr(capture, "finalize_sync", finalize)

        try:
            with nsys.range("region"):
                raise ValueError("kaboom")
        except ValueError:
            pass

        # a profile of the failure is still finalized, and the exception is not swallowed
        stop.assert_called_once()
        finalize.assert_called_once()

    def test_start_failure_runs_body_unprofiled(self, monkeypatch):
        monkeypatch.setattr(ctl, "under_nsys", lambda: True)
        monkeypatch.setattr(ctl, "start_collection_sync", MagicMock(side_effect=ctl.NsysError("no session")))
        stop = MagicMock()
        finalize = MagicMock()
        monkeypatch.setattr(ctl, "stop_collection_sync", stop)
        monkeypatch.setattr(capture, "finalize_sync", finalize)

        ran = False
        with nsys.range("region"):
            ran = True

        assert ran is True
        stop.assert_not_called()  # nothing started, so nothing to finalize
        finalize.assert_not_called()


class TestAsyncProtocol:
    def test_noop_when_not_under_nsys(self, monkeypatch):
        monkeypatch.setattr(ctl, "under_nsys", lambda: False)
        start = AsyncMock()
        monkeypatch.setattr(ctl, "start_collection", start)

        async def go():
            async with nsys.range("region"):
                return True

        assert asyncio.run(go()) is True
        start.assert_not_awaited()

    def test_collects_when_under_nsys(self, monkeypatch):
        monkeypatch.setattr(ctl, "under_nsys", lambda: True)
        start = AsyncMock(return_value="/tmp/nsys/a0/region.nsys-rep")
        stop = AsyncMock()
        finalize = AsyncMock(return_value={"kernel_launches": 5})
        monkeypatch.setattr(ctl, "start_collection", start)
        monkeypatch.setattr(ctl, "stop_collection", stop)
        monkeypatch.setattr(capture, "finalize", finalize)

        async def go():
            async with nsys.range("region"):
                pass

        asyncio.run(go())
        start.assert_awaited_once_with("region")
        stop.assert_awaited_once()
        finalize.assert_awaited_once()
