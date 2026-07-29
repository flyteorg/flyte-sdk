"""Tests for the low-level nsys control helpers (no GPU / no nsys binary required)."""

import flyteplugins.nsight._control as ctl


class TestBuildContainerEnv:
    def test_sets_wrapper_and_session(self):
        env = ctl.build_container_env(
            session_template="flyte-$RUN_NAME-$ACTION_NAME",
            output_template="/tmp/nsys/$ACTION_NAME/report",
            trace=["cuda", "nvtx", "osrt"],
            sample=None,
        )
        # Trace goes on the nsys launch command (nsys rejects --trace on `nsys start`).
        assert env[ctl.ENV_WRAPPER] == "nsys launch --session-new=flyte-$RUN_NAME-$ACTION_NAME --trace=cuda,nvtx,osrt"
        assert env[ctl.ENV_SESSION] == "flyte-$RUN_NAME-$ACTION_NAME"
        assert env[ctl.ENV_OUTPUT] == "/tmp/nsys/$ACTION_NAME/report"
        # sample omitted -> not added to the launch command
        assert "--sample" not in env[ctl.ENV_WRAPPER]

    def test_sample_included_when_given(self):
        env = ctl.build_container_env(session_template="s", output_template="o", trace=["cuda"], sample="cpu")
        assert env[ctl.ENV_WRAPPER] == "nsys launch --session-new=s --trace=cuda --sample=cpu"

    def test_clustered_opt_in(self):
        # clustered=True stamps the opt-in the runtime reads to wrap the primary torchrun worker.
        env = ctl.build_container_env(
            session_template="s", output_template="o", trace=["cuda"], sample=None, clustered=True
        )
        assert env[ctl.ENV_CLUSTERED] == "1"
        # off by default (single-GPU / non-clustered tasks).
        base = ctl.build_container_env(session_template="s", output_template="o", trace=["cuda"], sample=None)
        assert ctl.ENV_CLUSTERED not in base


class TestUnderNsys:
    def test_false_without_env(self, monkeypatch):
        monkeypatch.delenv(ctl.ENV_WRAPPED, raising=False)
        monkeypatch.delenv(ctl.ENV_SESSION, raising=False)
        assert ctl.under_nsys() is False

    def test_false_with_only_wrapped(self, monkeypatch):
        monkeypatch.setenv(ctl.ENV_WRAPPED, "1")
        monkeypatch.delenv(ctl.ENV_SESSION, raising=False)
        assert ctl.under_nsys() is False

    def test_true_with_both(self, monkeypatch):
        monkeypatch.setenv(ctl.ENV_WRAPPED, "1")
        monkeypatch.setenv(ctl.ENV_SESSION, "flyte-run-a0")
        assert ctl.under_nsys() is True


class TestSessionAndOutput:
    def test_session_name_expands_env(self, monkeypatch):
        monkeypatch.setenv("ACTION_NAME", "a0")
        monkeypatch.setenv(ctl.ENV_SESSION, "flyte-$ACTION_NAME")
        assert ctl.session_name() == "flyte-a0"

    def test_output_base_named_region_is_sanitized(self, monkeypatch):
        monkeypatch.setenv("ACTION_NAME", "a0")
        monkeypatch.setenv(ctl.ENV_OUTPUT, "/tmp/nsys/$ACTION_NAME/report")
        # A named region writes beside the base with a filesystem-safe name.
        assert ctl._output_base("hot loop!") == "/tmp/nsys/a0/hot_loop_"

    def test_output_base_default(self, monkeypatch):
        monkeypatch.delenv(ctl.ENV_OUTPUT, raising=False)
        assert ctl._output_base(None) == "/tmp/nsys/report"
