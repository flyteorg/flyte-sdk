"""Tests for the nvtx helpers: they must be safe no-ops when torch or its CUDA build is unavailable."""

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


class TestNoOpWithCpuOnlyTorch:
    """A CPU-only torch build imports torch.cuda.nvtx but raises RuntimeError on every call."""

    @staticmethod
    def _cpu_only():
        fake = MagicMock()
        err = RuntimeError("NVTX functions not installed. Are you sure you have a CUDA build?")
        fake.range_push.side_effect = err
        fake.mark.side_effect = err
        return fake

    def test_range_is_noop(self, monkeypatch):
        fake = self._cpu_only()
        monkeypatch.setattr(nvtx, "_nvtx", lambda: fake)
        with nvtx.range("forward"):
            pass  # must not raise
        fake.range_pop.assert_not_called()

    def test_range_does_not_swallow_body_errors(self, monkeypatch):
        monkeypatch.setattr(nvtx, "_nvtx", self._cpu_only)
        try:
            with nvtx.range("forward"):
                raise ValueError("from the body")
        except ValueError:
            pass
        else:
            raise AssertionError("the body's exception must propagate")

    def test_mark_is_noop(self, monkeypatch):
        monkeypatch.setattr(nvtx, "_nvtx", self._cpu_only)
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
