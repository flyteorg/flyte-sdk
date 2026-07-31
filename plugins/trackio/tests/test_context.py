from types import SimpleNamespace

import trackio

from flyteplugins.trackio._context import (
    _TRACKIO_RUN_KEY,
    _TrackioConfig,
    clear_trackio_run,
    get_trackio_context,
    get_trackio_run,
    set_trackio_run,
    trackio_config,
)


class TestTrackioConfig:
    """Tests for Trackio configuration."""

    def test_to_trackio_init(self):
        cfg = _TrackioConfig(
            project="vision",
            server_url="https://trackio.example.com",
            auto_log_gpu=None,
        )

        assert cfg.to_trackio_init() == {
            "project": "vision",
            "server_url": "https://trackio.example.com",
            "resume": "never",
            "gpu_log_interval": 10.0,
            "cpu_log_interval": 10.0,
        }

    def test_to_dict_from_dict_roundtrip(self):
        cfg = _TrackioConfig(
            project="vision",
            server_url="https://trackio.example.com",
            config={"lr": 1e-3},
            auto_log_gpu=True,
            auto_log_cpu=False,
        )

        restored = _TrackioConfig.from_dict(cfg.to_dict())

        assert restored == cfg

    def test_trackio_config_factory(self):
        cfg = trackio_config(
            project="vision",
            space_id="user/demo",
        )

        assert cfg.project == "vision"
        assert cfg.space_id == "user/demo"


class TestTrackioRun:
    """Tests for Trackio run helpers."""

    def test_set_trackio_run(self, monkeypatch):
        ctx = SimpleNamespace(data={})

        monkeypatch.setattr(
            "flyteplugins.trackio._context.flyte.ctx",
            lambda: ctx,
        )

        run = object()

        set_trackio_run(run)

        assert ctx.data[_TRACKIO_RUN_KEY] is run

    def test_clear_trackio_run(self, monkeypatch):
        run = object()

        ctx = SimpleNamespace(
            data={
                _TRACKIO_RUN_KEY: run,
            }
        )

        monkeypatch.setattr(
            "flyteplugins.trackio._context.flyte.ctx",
            lambda: ctx,
        )

        clear_trackio_run()

        assert _TRACKIO_RUN_KEY not in ctx.data

    def test_get_trackio_run_from_context(self, monkeypatch):
        run = object()

        ctx = SimpleNamespace(
            data={
                _TRACKIO_RUN_KEY: run,
            }
        )

        monkeypatch.setattr(
            "flyteplugins.trackio._context.flyte.ctx",
            lambda: ctx,
        )

        assert get_trackio_run() is run

    def test_get_trackio_run_falls_back_to_trackio(self, monkeypatch):
        run = object()

        monkeypatch.setattr(
            "flyteplugins.trackio._context.flyte.ctx",
            lambda: SimpleNamespace(data={}),
        )

        monkeypatch.setattr(trackio, "run", run)

        assert get_trackio_run() is run

    def test_get_trackio_run_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            "flyteplugins.trackio._context.flyte.ctx",
            lambda: None,
        )

        monkeypatch.setattr(trackio, "run", None)

        assert get_trackio_run() is None


class TestTrackioContext:
    """Tests for Trackio Flyte context."""

    def test_get_trackio_context_none(self, monkeypatch):
        monkeypatch.setattr(
            "flyteplugins.trackio._context.flyte.ctx",
            lambda: None,
        )

        assert get_trackio_context() is None

    def test_get_trackio_context_empty(self, monkeypatch):
        ctx = SimpleNamespace(custom_context={})

        monkeypatch.setattr(
            "flyteplugins.trackio._context.flyte.ctx",
            lambda: ctx,
        )

        assert get_trackio_context() is None

    def test_get_trackio_context(self, monkeypatch):
        ctx = SimpleNamespace(
            custom_context={
                "trackio_project": "vision",
                "trackio_server_url": "https://trackio.example.com",
            }
        )

        monkeypatch.setattr(
            "flyteplugins.trackio._context.flyte.ctx",
            lambda: ctx,
        )

        cfg = get_trackio_context()

        assert cfg.project == "vision"
        assert cfg.server_url == "https://trackio.example.com"
