"""Tests for the nvtx helpers — they must be safe no-ops when torch/CUDA is unavailable."""

from unittest.mock import MagicMock

from flyteplugins.nsight import nvtx


class TestNoOpWithoutTorch:
    def test_range_is_noop(self, monkeypatch):
        monkeypatch.setattr(nvtx, "_nvtx", lambda: None)
        with nvtx.range("forward"):
            pass  # must not raise

    def test_mark_is_noop(self, monkeypatch):
        monkeypatch.setattr(nvtx, "_nvtx", lambda: None)
        nvtx.mark("checkpoint")  # must not raise


class TestDelegatesToNvtx:
    def test_range_pushes_and_pops(self, monkeypatch):
        fake = MagicMock()
        monkeypatch.setattr(nvtx, "_nvtx", lambda: fake)
        with nvtx.range("forward"):
            fake.range_push.assert_called_once_with("forward")
            fake.range_pop.assert_not_called()
        fake.range_pop.assert_called_once()

    def test_range_pops_on_exception(self, monkeypatch):
        fake = MagicMock()
        monkeypatch.setattr(nvtx, "_nvtx", lambda: fake)
        try:
            with nvtx.range("forward"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        fake.range_pop.assert_called_once()

    def test_mark_delegates(self, monkeypatch):
        fake = MagicMock()
        monkeypatch.setattr(nvtx, "_nvtx", lambda: fake)
        nvtx.mark("checkpoint")
        fake.mark.assert_called_once_with("checkpoint")
