"""Validation and mode-gating for publishing local runs.

The upload-suppression helpers gate whether offloaded types (File/Dir/DataFrame) stay on local
disk or get materialized to configured storage. Getting these wrong either breaks plain local
runs (by forcing uploads) or breaks published runs (by linking to paths that only exist on the
developer's machine), so they are pinned here.
"""

import pytest

import flyte
from flyte._run import _publish_var, _run_mode_var, local_uploads_suppressed, offloading_to_storage


@pytest.fixture(autouse=True)
def _reset_run_vars():
    mode = _run_mode_var.set(None)
    pub = _publish_var.set(False)
    yield
    _run_mode_var.reset(mode)
    _publish_var.reset(pub)


class TestUploadSuppression:
    def test_plain_local_run_keeps_data_local(self):
        _run_mode_var.set("local")
        assert local_uploads_suppressed() is True
        assert offloading_to_storage() is False

    def test_published_local_run_materializes_to_storage(self):
        _run_mode_var.set("local")
        _publish_var.set(True)
        assert local_uploads_suppressed() is False
        assert offloading_to_storage() is True

    def test_remote_run_materializes_to_storage(self):
        _run_mode_var.set("remote")
        assert local_uploads_suppressed() is False
        assert offloading_to_storage() is True

    def test_hybrid_is_unchanged_by_publish_helpers(self):
        # Hybrid previously took neither the "local" nor the "remote" branch; keep it that way.
        _run_mode_var.set("hybrid")
        assert local_uploads_suppressed() is False
        assert offloading_to_storage() is False

    def test_no_run_context(self):
        assert local_uploads_suppressed() is False
        assert offloading_to_storage() is False


class TestLocalPublishConfig:
    """`local.publish` turns publishing on for every local run.

    This is what lets a plain `python my_script.py` publish without threading a flag through
    each call site -- mirroring how `local.persistence` works.
    """

    def test_config_entry_defaults_off(self):
        from flyte.config._config import LocalConfig

        assert LocalConfig().publish is False

    def test_runner_picks_up_local_publish(self, monkeypatch):
        import flyte._run as run_mod

        class _Cfg:
            client = object()
            local_publish = True

        cfg = _Cfg()
        monkeypatch.setattr(run_mod, "_get_init_config", lambda: cfg)
        # No publish= anywhere, and the configured client must not flip it to remote.
        runner = flyte.with_runcontext()
        assert runner._publish is True
        assert runner._mode == "local"

    def test_explicit_publish_still_works_without_config(self, monkeypatch):
        import flyte._run as run_mod

        class _Cfg:
            client = None
            local_publish = False

        cfg = _Cfg()
        monkeypatch.setattr(run_mod, "_get_init_config", lambda: cfg)
        assert flyte.with_runcontext(publish=True)._publish is True
        assert flyte.with_runcontext()._publish is False


class TestWithRunContextValidation:
    def test_publish_needs_no_run_base_dir(self):
        """Inputs, outputs, report and code bundle all use signed URLs to backend-chosen paths."""
        assert flyte.with_runcontext(publish=True)._publish is True

    def test_publish_defaults_to_false(self):
        assert flyte.with_runcontext()._publish is False

    def test_publish_forces_local_mode_even_when_client_configured(self, monkeypatch):
        # A configured client normally flips the default mode to "remote". Publishing executes
        # locally, so it must not be hijacked.
        import flyte._run as run_mod

        class _Cfg:
            client = object()
            local_publish = False

        cfg = _Cfg()
        monkeypatch.setattr(run_mod, "_get_init_config", lambda: cfg)
        assert flyte.with_runcontext(publish=True)._mode == "local"
        # Without publish, the client still wins as before.
        assert flyte.with_runcontext()._mode == "remote"

    def test_publish_rejects_explicit_remote_mode(self):
        runner = flyte.with_runcontext(publish=True, mode="remote")
        with pytest.raises(ValueError, match="cannot be combined with mode='remote'"):
            runner.run(_noop_task)


env = flyte.TaskEnvironment("test_publish")


@env.task
async def _noop_task() -> int:
    return 1
